"""End-to-end AdaptiveFace pipeline.

Pipeline.recognize takes one BGR image and returns a PipelineDecision:

    Image
      -> Face detection
      -> Mask classification (decides which branch)
      -> Embedding (full or upper)
      -> Matching against the dual-template gallery (with mode-specific threshold)
      -> Accept / Reject + optional identity

Everything in here is a thin orchestrator; the interesting decisions live in
the leaf modules. The class is deliberately stateless aside from the loaded
models so it's safe to share across threads / experiment runs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PIPELINE
from .embedder import Embedder
from .face_detector import DetectedFace, FaceDetector
from .gallery import Gallery
from .mask_classifier import MaskClassifier, MaskPrediction
from .matcher import MatchResult, Matcher, MultiTemplateMatcher


@dataclass
class PipelineDecision:
    accepted: bool
    identity: str | None
    similarity: float
    mode: str  # "full" or "upper"
    mask: MaskPrediction
    face: DetectedFace | None
    ranking: list[tuple[str, float]]
    threshold: float

    def to_row(self) -> dict:
        return {
            "accepted": self.accepted,
            "identity": self.identity,
            "similarity": self.similarity,
            "mode": self.mode,
            "p_mask": self.mask.p_mask,
            "mask_label": self.mask.label,
            "threshold": self.threshold,
            "top5": [f"{i}:{s:.3f}" for i, s in self.ranking[:5]],
        }


class AdaptiveFacePipeline:
    """Adaptive face authentication with mask-aware routing."""

    def __init__(
        self,
        gallery: Gallery,
        detector: FaceDetector | None = None,
        embedder: Embedder | None = None,
        mask_classifier: MaskClassifier | None = None,
        threshold_full: float = PIPELINE.cosine_threshold_unmasked,
        threshold_upper: float = PIPELINE.cosine_threshold_masked,
    ) -> None:
        self.detector = detector or FaceDetector()
        self.embedder = embedder or Embedder(detector=self.detector)
        self.mask_classifier = mask_classifier or MaskClassifier()
        self.gallery = gallery
        self.matcher_full = Matcher(gallery, threshold=threshold_full, kind="full")
        self.matcher_upper = Matcher(gallery, threshold=threshold_upper, kind="upper")

    # ---- Public API ----------------------------------------------------------

    def recognize(self, image_bgr: np.ndarray) -> PipelineDecision | None:
        faces = self.detector.detect(image_bgr, max_faces=1)
        if not faces:
            return None
        face = faces[0]
        mask_pred = self.mask_classifier.predict(image_bgr, face)
        if mask_pred.is_masked:
            embedding = self.embedder.embed_upper(image_bgr, face)
            match = self.matcher_upper.match(embedding)
            mode = "upper"
        else:
            embedding = self.embedder.embed_full(image_bgr, face)
            match = self.matcher_full.match(embedding)
            mode = "full"
        return PipelineDecision(
            accepted=match.accepted,
            identity=match.top_identity,
            similarity=match.top_similarity,
            mode=mode,
            mask=mask_pred,
            face=face,
            ranking=match.ranking,
            threshold=match.threshold,
        )

    def recognize_with_forced_mode(self, image_bgr: np.ndarray, mode: str) -> PipelineDecision | None:
        """Bypass mask classification; useful for ablations and Experiments 1/2."""
        if mode not in {"full", "upper"}:
            raise ValueError(mode)
        faces = self.detector.detect(image_bgr, max_faces=1)
        if not faces:
            return None
        face = faces[0]
        mask_pred = MaskPrediction(p_mask=float("nan"), label="forced", backend="forced")
        if mode == "upper":
            embedding = self.embedder.embed_upper(image_bgr, face)
            match = self.matcher_upper.match(embedding)
        else:
            embedding = self.embedder.embed_full(image_bgr, face)
            match = self.matcher_full.match(embedding)
        return PipelineDecision(
            accepted=match.accepted,
            identity=match.top_identity,
            similarity=match.top_similarity,
            mode=mode,
            mask=mask_pred,
            face=face,
            ranking=match.ranking,
            threshold=match.threshold,
        )


class MultiTemplatePipeline:
    """Pipeline-B (multi_max): no routing, max over two gallery rows per identity.

    This is the configuration that won Experiment 3. We still run the
    mask classifier so the on-screen overlay can show what it thought,
    but its output **does not affect the matching decision** -- the
    pipeline always scores the probe against both gallery rows and takes
    the per-identity max.
    """

    def __init__(
        self,
        gallery: Gallery,
        detector: FaceDetector | None = None,
        embedder: Embedder | None = None,
        mask_classifier: MaskClassifier | None = None,
        threshold: float = PIPELINE.cosine_threshold_unmasked,
    ) -> None:
        self.detector = detector or FaceDetector()
        self.embedder = embedder or Embedder(detector=self.detector)
        self.mask_classifier = mask_classifier or MaskClassifier()
        self.gallery = gallery
        self.matcher = MultiTemplateMatcher(gallery, threshold=threshold)

    def recognize(self, image_bgr: np.ndarray) -> PipelineDecision | None:
        faces = self.detector.detect(image_bgr, max_faces=1)
        if not faces:
            return None
        face = faces[0]
        # Mask classification is informational only -- the matcher always
        # scores against both gallery rows. Still useful for the on-screen
        # status banner and for any downstream logging.
        mask_pred = self.mask_classifier.predict(image_bgr, face)
        embedding = self.embedder.embed_full(image_bgr, face)
        match = self.matcher.match(embedding)
        return PipelineDecision(
            accepted=match.accepted,
            identity=match.top_identity,
            similarity=match.top_similarity,
            mode="multi_max",
            mask=mask_pred,
            face=face,
            ranking=match.ranking,
            threshold=match.threshold,
        )


__all__ = ["AdaptiveFacePipeline", "MultiTemplatePipeline", "PipelineDecision"]
