"""Mask / no-mask classifier.

We try three backends in order:
  1. AIZOO FaceMaskDetection ONNX (if the .onnx file is on disk).
  2. AIZOO FaceMaskDetection Caffe (loaded via cv2.dnn; primary backend
     because the Caffe weights are still hosted on GitHub raw and the ONNX
     mirror disappeared).
  3. A geometric heuristic based on the InsightFace 5-point landmarks --
     specifically the colour saturation in the nose/mouth region.

Either way the public interface is the same: `MaskClassifier.predict_proba`
returns the probability of a mask given an image + an optional DetectedFace.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import MASK_CLF
from .face_detector import DetectedFace


@dataclass
class MaskPrediction:
    p_mask: float
    label: str  # "masked" or "unmasked"
    backend: str

    @property
    def is_masked(self) -> bool:
        return self.label == "masked"


class _ONNXMaskBackend:
    """AIZOO face_mask_detection.onnx wrapper.

    The bundled model expects a 260x260 RGB image normalised to [0, 1].
    It outputs class probabilities for {face, face_mask}.
    """

    INPUT_SIZE = (260, 260)
    THRESHOLD = MASK_CLF.threshold

    def __init__(self, model_path) -> None:
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(model_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        # Some exports return both classification and bbox heads; we hunt
        # for the (N, 2) classification head at runtime.
        self._output_names = [o.name for o in self._session.get_outputs()]

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.INPUT_SIZE)
        arr = resized.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[None, ...]
        return arr

    def predict(self, crop_bgr: np.ndarray) -> MaskPrediction:
        inputs = {self._input_name: self._preprocess(crop_bgr)}
        outputs = self._session.run(self._output_names, inputs)
        # Find a (?, 2) tensor among outputs -- that's the per-anchor class
        # probabilities for the AIZOO export. Aggregate by taking the max
        # mask score across anchors (the most confident "mask" region).
        cls = None
        for o in outputs:
            if o.ndim >= 2 and o.shape[-1] == 2:
                cls = o.reshape(-1, 2)
                break
        if cls is None:
            # Fallback: assume single softmax output.
            probs = outputs[0].reshape(-1)
            p_mask = float(probs[-1]) if probs.size >= 2 else 0.5
        else:
            p_mask = float(cls[:, 1].max())
        label = "masked" if p_mask >= self.THRESHOLD else "unmasked"
        return MaskPrediction(p_mask=p_mask, label=label, backend="aizoo_onnx")


class _CaffeMaskBackend:
    """AIZOO face_mask_detection.caffemodel wrapper via cv2.dnn.

    The model is a small SSD-style detector with two classes and a
    260x260 RGB input normalised to [0, 1]. The Caffe export has two
    output blobs (no built-in NMS):

      * ``loc_branch_concat``  -> (1, N_anchors, 4) bbox regressions.
      * ``cls_branch_concat``  -> (1, N_anchors, 2) softmax class scores.

    Per AIZOO's reference Python pipeline the class indices are
    ``0 = face_mask`` and ``1 = face``. We aggregate across anchors by:
      1. Picking each anchor's argmax class and confidence.
      2. Keeping anchors with confidence above ``ANCHOR_KEEP_CONF`` -- this
         filters away the thousands of background anchors that always
         carry near-uniform scores.
      3. Returning the top-scoring kept anchor's class as the verdict.
    If no anchor clears the keep threshold we treat the image as low
    confidence and emit ``p_mask = 0.5``.
    """

    INPUT_SIZE = (260, 260)
    THRESHOLD = MASK_CLF.threshold
    ANCHOR_KEEP_CONF = 0.5
    MASK_CLASS_ID = 0
    UNMASK_CLASS_ID = 1
    OUTPUT_NAMES = ("loc_branch_concat", "cls_branch_concat")

    def __init__(self, proto_path, weights_path) -> None:
        self._net = cv2.dnn.readNetFromCaffe(str(proto_path), str(weights_path))
        # pip's opencv-python is built without CUDA DNN support, so
        # requesting the CUDA backend errors at forward() time rather
        # than at setPreferableBackend(). We probe with a dummy forward
        # pass to decide; on failure we revert to the CPU default.
        try:
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            probe = np.zeros((1, 3, self.INPUT_SIZE[1], self.INPUT_SIZE[0]), dtype=np.float32)
            self._net.setInput(probe)
            self._net.forward()
        except Exception:
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def predict(self, crop_bgr: np.ndarray) -> MaskPrediction:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.INPUT_SIZE)
        blob = cv2.dnn.blobFromImage(
            resized, scalefactor=1.0 / 255.0, size=self.INPUT_SIZE,
            mean=(0, 0, 0), swapRB=False, crop=False,
        )
        self._net.setInput(blob)
        try:
            _loc, cls = self._net.forward(self.OUTPUT_NAMES)
        except cv2.error:
            # Some OpenCV versions return outputs in a single concatenated
            # tensor when asked for multiple names; fall back to the
            # default forward which yields the last layer (cls_branch).
            cls = self._net.forward()
        cls = np.asarray(cls).reshape(-1, 2)  # (N_anchors, 2)
        best_class = cls.argmax(axis=1)
        best_score = cls.max(axis=1)
        keep = best_score > self.ANCHOR_KEEP_CONF
        if not keep.any():
            return MaskPrediction(p_mask=0.5, label="unmasked", backend="aizoo_caffe")
        kept_classes = best_class[keep]
        kept_scores = best_score[keep]
        top = int(np.argmax(kept_scores))
        top_is_mask = kept_classes[top] == self.MASK_CLASS_ID
        # Use the top-anchor confidence as P(mask) when the verdict is
        # "masked", and 1 - confidence when the verdict is "face". This
        # gives downstream code a meaningful probability for ROC plots.
        p_mask = float(kept_scores[top]) if top_is_mask else float(1.0 - kept_scores[top])
        label = "masked" if top_is_mask else "unmasked"
        return MaskPrediction(p_mask=p_mask, label=label, backend="aizoo_caffe")


class _LandmarkHeuristicBackend:
    """Fallback when no ONNX model is available.

    Idea: if a mask covers the lower face, the region under the eye line
    is dominated by a single colour (mask fabric) rather than skin tones.
    We crop the lower-face quadrant defined by the landmarks and look at
    the saturation distribution + colour-channel std. Low std + high
    saturation means "uniformly coloured region" -> likely a mask.

    This is deliberately a coarse decision; the ONNX backend is what we
    actually evaluate. The heuristic just keeps the demo runnable when
    the download mirrors are offline.
    """

    SAT_THRESHOLD = 35.0  # mean saturation above this -> coloured mask.
    STD_THRESHOLD = 25.0  # std below this -> uniform region.

    def predict(self, crop_bgr: np.ndarray) -> MaskPrediction:
        h, w = crop_bgr.shape[:2]
        if h < 10 or w < 10:
            return MaskPrediction(p_mask=0.5, label="unmasked", backend="heuristic")
        lower = crop_bgr[int(h * 0.45):, :]
        hsv = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)
        sat_mean = float(hsv[..., 1].mean())
        std = float(lower.std())
        # Map to a pseudo-probability for downstream code.
        score = 0.0
        if sat_mean > self.SAT_THRESHOLD:
            score += 0.4
        if std < self.STD_THRESHOLD:
            score += 0.4
        # Skin-tone hue band is ~5-25 in OpenCV HSV; mask hues fall outside.
        hue_mean = float(hsv[..., 0].mean())
        if hue_mean > 30 or hue_mean < 5:
            score += 0.2
        p_mask = min(1.0, score)
        label = "masked" if p_mask >= 0.5 else "unmasked"
        return MaskPrediction(p_mask=p_mask, label=label, backend="heuristic")


class MaskClassifier:
    """Public mask classifier facade."""

    def __init__(self, model_path: Optional[object] = None) -> None:
        """Pick the first backend whose weights are on disk.

        Pass ``model_path`` to force a specific ONNX file; otherwise we
        probe in order: ONNX -> Caffe -> heuristic.
        """
        self._backend: object
        self.backend_name = "heuristic"

        onnx_path = model_path or MASK_CLF.onnx_path
        try:
            if onnx_path and getattr(onnx_path, "exists", lambda: False)():
                self._backend = _ONNXMaskBackend(onnx_path)
                self.backend_name = "aizoo_onnx"
                return
        except Exception as exc:
            print(f"[mask] ONNX backend init failed: {exc}")

        try:
            proto = MASK_CLF.caffe_proto_path
            weights = MASK_CLF.caffe_weights_path
            if proto.exists() and weights.exists():
                self._backend = _CaffeMaskBackend(proto, weights)
                self.backend_name = "aizoo_caffe"
                return
        except Exception as exc:
            print(f"[mask] Caffe backend init failed: {exc}")

        self._backend = _LandmarkHeuristicBackend()
        self.backend_name = "heuristic"

    def predict(self, image_bgr: np.ndarray, face: Optional[DetectedFace] = None) -> MaskPrediction:
        crop = self._face_crop(image_bgr, face)
        return self._backend.predict(crop)

    def predict_proba(self, image_bgr: np.ndarray, face: Optional[DetectedFace] = None) -> float:
        return self.predict(image_bgr, face).p_mask

    @staticmethod
    def _face_crop(image_bgr: np.ndarray, face: Optional[DetectedFace]) -> np.ndarray:
        if face is None:
            return image_bgr
        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = face.bbox
        # 20% margin to keep the chin area when present.
        pad_w = int((x2 - x1) * 0.2)
        pad_h = int((y2 - y1) * 0.2)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        return image_bgr[y1:y2, x1:x2]


__all__ = ["MaskClassifier", "MaskPrediction"]
