"""Cosine-similarity matching against a Gallery.

Open-set 1:N: for a probe embedding we compute similarities against every
enrolled template, take the top match, and accept it iff the similarity
clears a threshold. Otherwise the probe is rejected as "unknown".

Why cosine: ArcFace embeddings are L2-normalised by construction, so
cosine and dot-product are equivalent. Cosine also matches what the
training loss optimised, which is the right thing to score on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gallery import Gallery


@dataclass
class MatchResult:
    top_identity: str | None  # None means "rejected".
    top_similarity: float
    ranking: list[tuple[str, float]]
    threshold: float

    @property
    def accepted(self) -> bool:
        return self.top_identity is not None


class Matcher:
    def __init__(self, gallery: Gallery, threshold: float, kind: str = "full") -> None:
        if kind not in {"full", "upper"}:
            raise ValueError(kind)
        self.gallery = gallery
        self.threshold = threshold
        self.kind = kind
        ids, matrix = gallery.template_matrix(kind=kind)
        self._identities = ids
        self._matrix = matrix  # already L2-normalised (each row)

    def match(self, probe_embedding: np.ndarray) -> MatchResult:
        if self._matrix.size == 0:
            return MatchResult(top_identity=None, top_similarity=0.0, ranking=[], threshold=self.threshold)
        # probe is L2-normalised so dot product == cosine similarity.
        sims = self._matrix @ probe_embedding.astype(np.float32)
        order = np.argsort(-sims)
        ranking = [(self._identities[i], float(sims[i])) for i in order]
        top_id, top_sim = ranking[0]
        accepted = top_sim >= self.threshold
        return MatchResult(
            top_identity=top_id if accepted else None,
            top_similarity=top_sim,
            ranking=ranking,
            threshold=self.threshold,
        )

    def match_batch(self, probes: np.ndarray) -> np.ndarray:
        """Return raw similarity matrix (N_probes, N_gallery) without thresholding.

        Used by the evaluation code which needs the full distribution to
        sweep thresholds and compute FAR/FRR curves.
        """
        return probes.astype(np.float32) @ self._matrix.T


class MultiTemplateMatcher:
    """Pipeline-B matcher: each identity has TWO gallery rows.

    Row 1 -- ``full_template``         (the user's real unmasked photo).
    Row 2 -- ``synth_masked_template`` (same photo with a synthetic mask
              overlaid).

    For each probe we compute cosine similarity against both matrices,
    take the max per identity, and threshold on that. This is the
    ``multi_max`` configuration that won Experiment 3.

    Falls back gracefully to the single-row ``full`` matcher when the
    gallery has no synth-masked templates (e.g., legacy enrolment).
    """

    def __init__(self, gallery: Gallery, threshold: float) -> None:
        self.gallery = gallery
        self.threshold = threshold
        ids_full, mat_full = gallery.template_matrix(kind="full")
        ids_synth, mat_synth = gallery.template_matrix(kind="synth")
        self._identities = ids_full
        self._real = mat_full  # always present
        # Align the synth matrix to the same identity ordering as full.
        # Identities without a synth template get a zero row which will
        # never win the max (since cosines vs L2-normalised embeddings
        # are in [-1, 1] and the real row already covers them).
        synth_aligned = np.zeros_like(mat_full)
        synth_pos = {ident: idx for idx, ident in enumerate(ids_synth)}
        for k, ident in enumerate(ids_full):
            if ident in synth_pos:
                synth_aligned[k] = mat_synth[synth_pos[ident]]
        self._synth = synth_aligned
        self._has_synth = gallery.has_synth_templates()

    def match(self, probe_embedding: np.ndarray) -> MatchResult:
        if self._real.size == 0:
            return MatchResult(top_identity=None, top_similarity=0.0,
                               ranking=[], threshold=self.threshold)
        p = probe_embedding.astype(np.float32)
        sims_real = self._real @ p
        if self._has_synth:
            sims_synth = self._synth @ p
            sims = np.maximum(sims_real, sims_synth)
        else:
            sims = sims_real
        order = np.argsort(-sims)
        ranking = [(self._identities[i], float(sims[i])) for i in order]
        top_id, top_sim = ranking[0]
        accepted = top_sim >= self.threshold
        return MatchResult(
            top_identity=top_id if accepted else None,
            top_similarity=top_sim,
            ranking=ranking,
            threshold=self.threshold,
        )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


__all__ = ["Matcher", "MatchResult", "MultiTemplateMatcher", "cosine_similarity"]
