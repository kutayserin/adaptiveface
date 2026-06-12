"""Face detection + alignment wrapper.

We use InsightFace's `FaceAnalysis` because it ships an end-to-end pipeline
(detector + landmark predictor + aligned ArcFace embedder) and is the same
stack the pre-trained ArcFace was evaluated with. The rest of the code talks
only to `FaceDetector.detect`, so swapping the backend later (dlib, MediaPipe,
or a hand-rolled MTCNN) means changing this file alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from .config import EMBEDDER


@dataclass
class DetectedFace:
    """One face from a frame.

    `bbox` is (x1, y1, x2, y2) in pixel coords. `kps` are the 5-point
    landmarks RetinaFace produces (left eye, right eye, nose, left mouth,
    right mouth) -- we use those to crop the upper-face region for the
    masked branch and to know the inter-ocular distance.

    `raw` carries the underlying insightface Face object so the embedder can
    reuse its `normed_embedding`, but callers should treat it as opaque.
    """

    bbox: tuple[int, int, int, int]
    kps: np.ndarray
    det_score: float
    raw: object | None = None

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)


class FaceDetector:
    """Thin wrapper around `insightface.app.FaceAnalysis`.

    The InsightFace import is deferred so that experiments that don't need
    detection (e.g., metric-only re-runs from cached embeddings) don't pay
    the model-loading cost.
    """

    def __init__(
        self,
        model_pack: str = EMBEDDER.model_pack,
        det_size: tuple[int, int] = EMBEDDER.det_size,
        ctx_id: int = EMBEDDER.ctx_id,
    ) -> None:
        from insightface.app import FaceAnalysis

        providers, effective_ctx = self._resolve_providers(ctx_id)
        self._app = FaceAnalysis(name=model_pack, providers=providers)
        self._app.prepare(ctx_id=effective_ctx, det_size=det_size)
        self.providers_used = providers

    @staticmethod
    def _resolve_providers(ctx_id: int) -> tuple[list[str], int]:
        """Return (providers, effective_ctx_id).

        We probe ``onnxruntime.get_available_providers`` so that asking
        for CUDA on a CPU-only install silently downgrades instead of
        spamming warnings on every model load.
        """
        try:
            import onnxruntime as ort  # type: ignore

            available = set(ort.get_available_providers())
        except Exception:
            available = {"CPUExecutionProvider"}
        if ctx_id < 0 or "CUDAExecutionProvider" not in available:
            return ["CPUExecutionProvider"], -1
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], ctx_id

    @property
    def app(self):
        """Expose the underlying FaceAnalysis so the embedder can reuse it."""
        return self._app

    # Minimum dimension below which we upscale before detection. RMFD ships
    # pre-cropped 130x150-ish head shots that RetinaFace's default 640-grid
    # downsamples into oblivion.
    _MIN_INPUT_DIM = 320

    def detect(self, image_bgr: np.ndarray, max_faces: int | None = 1) -> list[DetectedFace]:
        """Return detected faces, sorted by detector confidence (descending).

        `max_faces=1` is the common case -- for identification we work on the
        most prominent face in the frame. Pass `None` to return everything.

        Tiny inputs (pre-cropped face datasets like RMFD) get upscaled
        before detection; on a miss we retry at a larger scale. The
        returned bbox coordinates are mapped back to the *original*
        image's coordinate system so callers can crop without surprises.
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        h, w = image_bgr.shape[:2]
        scale = 1.0
        work = image_bgr
        if min(h, w) < self._MIN_INPUT_DIM:
            scale = self._MIN_INPUT_DIM / float(min(h, w))
            work = cv2.resize(image_bgr, (int(round(w * scale)), int(round(h * scale))))

        faces = self._app.get(work)
        if not faces and scale < 4.0:
            # Retry with an even larger canvas before giving up.
            scale *= 2.0
            work = cv2.resize(image_bgr, (int(round(w * scale)), int(round(h * scale))))
            faces = self._app.get(work)

        faces.sort(key=lambda f: getattr(f, "det_score", 0.0), reverse=True)
        if max_faces is not None:
            faces = faces[:max_faces]

        inv_scale = 1.0 / scale
        out: list[DetectedFace] = []
        for f in faces:
            bx1, by1, bx2, by2 = (int(round(v * inv_scale)) for v in f.bbox)
            kps_src = (
                np.asarray(f.kps, dtype=np.float32) * inv_scale
                if f.kps is not None else np.zeros((5, 2), dtype=np.float32)
            )
            # The `raw` Face object still references the upscaled image's
            # alignment; downstream embedding uses our re-detected bbox
            # via the (bbox, kps) tuple, so we keep `raw=None` whenever
            # we upscaled to avoid silently re-running on stale buffers.
            raw = f if scale == 1.0 else None
            out.append(
                DetectedFace(
                    bbox=(bx1, by1, bx2, by2),
                    kps=kps_src,
                    det_score=float(getattr(f, "det_score", 0.0)),
                    raw=raw,
                )
            )
        return out


def crop_bbox(image_bgr: np.ndarray, face: DetectedFace, padding: float = 0.0) -> np.ndarray:
    """Crop the bbox with optional symmetric padding (fraction of width/height)."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = face.bbox
    if padding:
        pad_w = int(face.width * padding)
        pad_h = int(face.height * padding)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
    return image_bgr[y1:y2, x1:x2].copy()


def crop_upper_face(image_bgr: np.ndarray, face: DetectedFace, keep_ratio: float = 0.55) -> np.ndarray:
    """Keep the top `keep_ratio` of the face bbox.

    Used for the masked branch -- a mask occludes nose/mouth, so we throw
    away the lower portion of the crop before feeding it to the embedder.
    `keep_ratio = 0.55` keeps eyes, brows, forehead, and the bridge of the
    nose, which is what an ArcFace network keys on for periocular cues.
    """
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = face.bbox
    face_h = y2 - y1
    y2_new = y1 + int(face_h * keep_ratio)
    y2_new = max(y1 + 1, min(h, y2_new))
    x1 = max(0, x1)
    x2 = min(w, x2)
    return image_bgr[y1:y2_new, x1:x2].copy()


def read_image(path) -> np.ndarray | None:
    """Read an image via OpenCV in BGR. Returns None on failure."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


__all__ = ["DetectedFace", "FaceDetector", "crop_bbox", "crop_upper_face", "read_image"]
