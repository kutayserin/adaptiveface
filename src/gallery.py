"""Enrolled-user gallery with multiple templates per identity.

We store three template variants for every enrolled user, all derived
from the same single unmasked enrollment photo:

  * ``full_template``         -- ArcFace embedding of the full face.
  * ``upper_template``        -- ArcFace embedding of the landmark-aligned
                                 upper face (lower half replaced by
                                 forehead-colour fill). Used by the
                                 Pipeline-A routing experiments.
  * ``synth_masked_template`` -- ArcFace embedding of the same face
                                 with a synthetic surgical-mask polygon
                                 overlaid. Used by Pipeline-B multi-template
                                 matching (the winning configuration).

The synth-masked template is optional on older galleries; ``load`` will
return ``None`` for that field if the npz file predates the addition.

Persistence is intentionally simple: a numpy ``.npz`` plus a JSON
sidecar with the identity ordering. We deliberately avoid pickling
embedder objects so gallery files stay portable across Python versions
and machines.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .embedder import Embedder
from .face_detector import read_image
from .synth_mask import overlay_synthetic_mask


@dataclass
class GalleryEntry:
    identity: str
    full_template: np.ndarray  # (512,) L2-normalised
    upper_template: np.ndarray  # (512,) L2-normalised
    source_image: str
    synth_masked_template: np.ndarray | None = None  # (512,) L2-normalised; optional for older galleries


class Gallery:
    """Maps identities to enrolled embeddings.

    Each identity carries one full-face template and one upper-face template
    derived from the same enrollment image. Multi-template enrollment
    (averaging several photos per identity, as Smart Peephole does) is a
    straightforward extension: call `enroll_identity` repeatedly and average
    the resulting templates -- left as an experimental knob.
    """

    def __init__(self, entries: dict[str, GalleryEntry] | None = None) -> None:
        self._entries: dict[str, GalleryEntry] = dict(entries or {})

    # ---- Mutation ------------------------------------------------------------

    def enroll(
        self,
        identity: str,
        image_path,
        embedder: Embedder,
        with_synth_mask: bool = True,
    ) -> GalleryEntry | None:
        """Enroll one identity from a single unmasked photo.

        Returns None if the photo had no detectable face (so the caller
        can log the skip rather than silently ending up with a
        half-built gallery).

        When ``with_synth_mask`` is True (the default for the
        multi-template pipeline) we additionally render a synthetic
        surgical-mask overlay on top of the same image and embed *that*
        as a second template -- this is the configuration that won
        Experiment 3. Set to False if you only want the legacy
        dual-template (full + upper) gallery.
        """
        image_bgr = read_image(image_path)
        if image_bgr is None:
            return None
        faces = embedder.detector.detect(image_bgr, max_faces=1)
        if not faces:
            return None
        face = faces[0]
        full = embedder.embed_full(image_bgr, face)
        upper = embedder.embed_upper(image_bgr, face)
        synth_template = None
        if with_synth_mask:
            try:
                synth_img = overlay_synthetic_mask(image_bgr, face, color="surgical_blue")
                # Re-detect on the rendered image so the embedder gets a
                # face object pointing at the masked pixels rather than
                # at the original unmasked landmarks.
                synth_faces = embedder.detector.detect(synth_img, max_faces=1)
                synth_face = synth_faces[0] if synth_faces else face
                synth_template = embedder.embed_full(synth_img, synth_face)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[gallery] synth-mask enrollment failed for {identity}: {exc}")
                synth_template = None
        entry = GalleryEntry(
            identity=identity,
            full_template=full,
            upper_template=upper,
            source_image=str(image_path),
            synth_masked_template=synth_template,
        )
        self._entries[identity] = entry
        return entry

    def enroll_many(
        self,
        records: Iterable[tuple[str, Path]],
        embedder: Embedder,
        verbose: bool = True,
    ) -> dict[str, int]:
        """Enroll a batch. Returns counters useful for progress logging."""
        stats = {"ok": 0, "no_face": 0}
        records = list(records)
        for i, (identity, path) in enumerate(records):
            entry = self.enroll(identity, path, embedder)
            if entry is None:
                stats["no_face"] += 1
            else:
                stats["ok"] += 1
            if verbose and (i + 1) % 50 == 0:
                print(f"  enrolled {i + 1}/{len(records)} (ok={stats['ok']}, no_face={stats['no_face']})")
        return stats

    # ---- Access --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, identity: str) -> bool:
        return identity in self._entries

    def identities(self) -> list[str]:
        return list(self._entries.keys())

    def get(self, identity: str) -> GalleryEntry | None:
        return self._entries.get(identity)

    def template_matrix(self, kind: str = "full") -> tuple[list[str], np.ndarray]:
        """Stack templates into a single (N, 512) matrix for batched similarity.

        Returns the parallel identity list so callers can map row -> identity.
        For ``kind="synth"`` only identities with a synth-masked template
        are returned; the caller can detect a smaller-than-expected
        matrix and degrade to dual-template matching.
        """
        if kind not in {"full", "upper", "synth"}:
            raise ValueError(kind)
        if kind == "synth":
            identities = [i for i, e in self._entries.items()
                          if e.synth_masked_template is not None]
            if not identities:
                return [], np.zeros((0, 512), dtype=np.float32)
            matrix = np.stack(
                [self._entries[i].synth_masked_template for i in identities], axis=0
            )
            return identities, matrix.astype(np.float32)
        identities = list(self._entries.keys())
        if not identities:
            return [], np.zeros((0, 512), dtype=np.float32)
        attr = "full_template" if kind == "full" else "upper_template"
        matrix = np.stack([getattr(self._entries[i], attr) for i in identities], axis=0)
        return identities, matrix.astype(np.float32)

    def has_synth_templates(self) -> bool:
        """True if at least one identity carries a synth-masked template."""
        return any(e.synth_masked_template is not None for e in self._entries.values())

    # ---- Persistence ---------------------------------------------------------

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        identities = self.identities()
        if identities:
            full = np.stack([self._entries[i].full_template for i in identities], axis=0)
            upper = np.stack([self._entries[i].upper_template for i in identities], axis=0)
        else:
            full = np.zeros((0, 512), dtype=np.float32)
            upper = np.zeros((0, 512), dtype=np.float32)
        sources = [self._entries[i].source_image for i in identities]

        # Synth-masked template is optional per identity. We persist a
        # parallel boolean mask alongside the stacked matrix so loaders
        # know which rows are real templates vs. zero-filled padding.
        synth_valid = np.array(
            [self._entries[i].synth_masked_template is not None for i in identities],
            dtype=bool,
        )
        synth = np.zeros((len(identities), 512), dtype=np.float32)
        for k, ident in enumerate(identities):
            tpl = self._entries[ident].synth_masked_template
            if tpl is not None:
                synth[k] = tpl

        np.savez(
            path.with_suffix(".npz"),
            full=full, upper=upper,
            synth=synth, synth_valid=synth_valid,
        )
        path.with_suffix(".json").write_text(
            json.dumps({"identities": identities, "sources": sources}, indent=2)
        )

    @classmethod
    def load(cls, path) -> "Gallery":
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        data = np.load(path.with_suffix(".npz"))
        # Older gallery files don't have synth arrays; we treat them as
        # all-None so the dual-template pipeline still works.
        has_synth = "synth" in data.files and "synth_valid" in data.files
        entries: dict[str, GalleryEntry] = {}
        for i, identity in enumerate(meta["identities"]):
            synth_tpl = None
            if has_synth and bool(data["synth_valid"][i]):
                synth_tpl = data["synth"][i].astype(np.float32)
            entries[identity] = GalleryEntry(
                identity=identity,
                full_template=data["full"][i].astype(np.float32),
                upper_template=data["upper"][i].astype(np.float32),
                source_image=meta["sources"][i],
                synth_masked_template=synth_tpl,
            )
        return cls(entries=entries)


__all__ = ["Gallery", "GalleryEntry"]
