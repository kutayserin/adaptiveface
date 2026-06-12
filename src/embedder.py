"""ArcFace embedding extractor with full-face and upper-face modes.

InsightFace's `FaceAnalysis.get()` already returns aligned ArcFace
embeddings for detected faces -- we expose that as the "full" mode. For the
masked branch we want to feed the embedder an *upper-face* crop (because the
lower half is mask fabric carrying no useful identity signal). The cleanest
way to do that on top of the existing API is:

  1. Take the full image and the detected face from RetinaFace.
  2. Build a new image where the lower portion of the face bbox is replaced
     with the average skin colour sampled from the forehead. This preserves
     the alignment landmarks (the network expects eyes/nose/mouth at fixed
     locations after the 112x112 warp) while removing the mask-corrupted
     signal so the network does not key on the mask pattern.
  3. Re-run the recognition model on that masked-out image.

The motivation: ArcFace was trained on full faces; cropping aggressively
breaks alignment. Replacing the lower half with a smooth fill keeps the
warp valid and forces the network to use only the visible top-half
information -- the trick is the same one used in several masked-face
benchmark baselines (e.g., "Masked Face Recognition: Human vs Machine",
Anwar & Raychowdhury 2020) and it lets us reuse the off-the-shelf model
without fine-tuning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import EMBEDDER, PIPELINE
from .face_detector import DetectedFace, FaceDetector, read_image


@dataclass
class EmbeddingResult:
    embedding: np.ndarray  # L2-normalised 512-D vector.
    mode: str  # "full" or "upper".
    face: DetectedFace | None
    image_path: str | None = None


class Embedder:
    """Wrap an InsightFace recognition model in both modes.

    We share a single FaceDetector instance with the rest of the pipeline
    so the model is loaded once.
    """

    def __init__(self, detector: Optional[FaceDetector] = None) -> None:
        self.detector = detector or FaceDetector()
        # `rec_model` is the recognition sub-model inside FaceAnalysis.
        # It exposes .get(img, face_or_kps) which performs the 112x112
        # alignment + forward pass.
        rec = self.detector.app.models.get("recognition")
        if rec is None:
            # Newer insightface stores it under a different key.
            for k, v in self.detector.app.models.items():
                if "recognition" in k.lower() or hasattr(v, "get"):
                    rec = v
                    break
        if rec is None:
            raise RuntimeError("Could not locate recognition model in FaceAnalysis stack.")
        self._rec = rec

    # ---- Public API ----------------------------------------------------------

    def embed_full(self, image_bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
        """Return the L2-normalised ArcFace embedding for the full face."""
        if face.raw is not None and hasattr(face.raw, "normed_embedding"):
            # Reuse the embedding FaceAnalysis already computed during detect().
            return np.asarray(face.raw.normed_embedding, dtype=np.float32)
        face_arg = self._face_for_rec(face)
        vec = self._rec.get(image_bgr, face_arg)
        return _l2_normalise(np.asarray(vec, dtype=np.float32))

    def embed_upper(
        self,
        image_bgr: np.ndarray,
        face: DetectedFace,
        keep_ratio: float = PIPELINE.upper_face_crop_ratio,
    ) -> np.ndarray:
        """Embed the upper-face region with proper landmark alignment.

        Pipeline:
          1. `face_align.norm_crop` warps the face to the canonical
             112x112 ArcFace input using the 5-point landmarks. In this
             aligned space the eyes / nose / mouth are at fixed pixel
             locations regardless of pose, so we can mask the lower half
             by a fixed row threshold.
          2. Replace the lower ``1 - keep_ratio`` rows with the mean
             forehead colour sampled from the top of the crop.
          3. Feed the resulting 112x112 image to ArcFace's recognition
             backbone directly via `rec.get_feat`.

        Why this works (and previous attempts didn't):
          * The "fill bbox in original coordinates" approach didn't
            align well across pose -- ArcFace's internal warp resampled
            the fill region into different pixels depending on rotation,
            so identical fills produced inconsistent embeddings.
          * The "tight crop and resize" approach skipped alignment
            entirely; ArcFace then keyed on crop framing rather than
            identity, which made the wrong-person impostor problem worse.
          * Aligning *first*, then masking, keeps the geometry constant
            and gives the network a clean signal about which pixels are
            real face vs filled region.
        """
        from insightface.utils import face_align  # local import keeps the module light

        aligned = face_align.norm_crop(image_bgr, landmark=face.kps, image_size=112)
        h = aligned.shape[0]
        cut = max(1, int(h * keep_ratio))
        # Sample the top ~25% as a skin-tone fill so we don't introduce a
        # high-contrast horizontal edge ArcFace might latch onto.
        fill = aligned[: max(1, h // 4)].reshape(-1, 3).mean(axis=0).astype(np.uint8)
        aligned[cut:, :] = fill
        vec = self._rec.get_feat(aligned).flatten()
        return _l2_normalise(np.asarray(vec, dtype=np.float32))

    @staticmethod
    def _face_for_rec(face: DetectedFace):
        """Build the Face-like argument insightface's arcface_onnx expects.

        The recognition head reads ``face.kps`` (and only kps) to do its
        112x112 norm-crop. When ``face.raw`` is None (because our detector
        upscaled the image and the cached embedding wouldn't be valid), we
        synthesise the smallest object that quacks like a Face: a
        ``SimpleNamespace`` with the kps array attached.
        """
        if face.raw is not None:
            return face.raw
        from types import SimpleNamespace

        return SimpleNamespace(kps=face.kps, bbox=face.bbox)

    def embed_image(
        self,
        path_or_array,
        mode: str = "full",
    ) -> EmbeddingResult | None:
        """Convenience helper: read an image, run detection, return embedding.

        Returns None if no face was detected -- the caller decides whether
        that's a failure or just a sample to skip.
        """
        if isinstance(path_or_array, np.ndarray):
            image_bgr = path_or_array
            image_path = None
        else:
            image_bgr = read_image(path_or_array)
            image_path = str(path_or_array)
        if image_bgr is None:
            return None
        faces = self.detector.detect(image_bgr, max_faces=1)
        if not faces:
            return None
        face = faces[0]
        if mode == "upper":
            emb = self.embed_upper(image_bgr, face)
        elif mode == "full":
            emb = self.embed_full(image_bgr, face)
        else:
            raise ValueError(f"unknown mode: {mode}")
        return EmbeddingResult(embedding=emb, mode=mode, face=face, image_path=image_path)


# ---- Helpers -----------------------------------------------------------------


def _l2_normalise(v: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v if norm < eps else (v / norm).astype(np.float32)


def _mask_lower_face(image_bgr: np.ndarray, face: DetectedFace, keep_ratio: float) -> np.ndarray:
    """Replace the lower portion of the face bbox with a forehead-colour fill.

    Sampling the forehead instead of using a fixed grey keeps the
    image colour-balanced and avoids introducing a high-contrast horizontal
    edge that ArcFace might latch onto.
    """
    out = image_bgr.copy()
    h, w = out.shape[:2]
    x1, y1, x2, y2 = face.bbox
    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(w, x2)
    y2c = min(h, y2)
    face_h = y2c - y1c
    if face_h <= 0:
        return out

    cut_y = y1c + int(face_h * keep_ratio)
    cut_y = max(y1c + 1, min(y2c - 1, cut_y))

    # Sample the upper 25% of the face crop -- that's typically forehead +
    # brow region, mostly skin tones.
    sample_top = y1c
    sample_bottom = y1c + max(1, int(face_h * 0.25))
    sample = out[sample_top:sample_bottom, x1c:x2c]
    fill_colour = (
        sample.reshape(-1, 3).mean(axis=0)
        if sample.size > 0
        else np.array([128, 128, 128], dtype=np.float32)
    )
    fill_colour = fill_colour.astype(np.uint8)

    out[cut_y:y2c, x1c:x2c] = fill_colour
    return out


__all__ = ["Embedder", "EmbeddingResult"]
