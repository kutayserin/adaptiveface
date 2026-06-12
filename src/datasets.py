"""Dataset loaders for LFW, RMFD, and MFR2.

Everything is exposed as `IdentityDataset` -- a list of (identity, image_path)
tuples -- because every experiment ultimately needs the same primitive: "for
each identity, give me its images." Paired masked/unmasked iteration is built
on top of that primitive.

Why this shape: the past projects we're modelling (Smart Peephole, Biotouch)
all kept enrollment and probe handling explicit. Hiding the dataset behind a
PyTorch DataLoader would obscure which identity supplied which image, which
matters for the cross-mode experiment where pair construction is the point.
"""
from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from .config import LFW_DIR, MFR2_DIR, RMFD_DIR


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class IdentityDataset:
    """A flat collection of (identity, image_path) records.

    `name` is just a human-readable tag used when writing results files.
    `tag` flags whether the source contains masked images so downstream
    code can route correctly.
    """

    name: str
    records: list[tuple[str, Path]] = field(default_factory=list)
    tag: str = "unmasked"  # one of {"unmasked", "masked", "mixed"}

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[tuple[str, Path]]:
        return iter(self.records)

    def identities(self) -> list[str]:
        seen: dict[str, None] = {}
        for identity, _ in self.records:
            seen.setdefault(identity, None)
        return list(seen)

    def by_identity(self) -> dict[str, list[Path]]:
        grouped: dict[str, list[Path]] = defaultdict(list)
        for identity, path in self.records:
            grouped[identity].append(path)
        return grouped

    def filter_min_samples(self, k: int) -> "IdentityDataset":
        """Return a new dataset keeping only identities with >= k images."""
        grouped = self.by_identity()
        keep = {identity for identity, paths in grouped.items() if len(paths) >= k}
        records = [(i, p) for i, p in self.records if i in keep]
        return IdentityDataset(name=self.name, records=records, tag=self.tag)

    def sample_identities(self, n: int, seed: int = 0) -> "IdentityDataset":
        """Return a deterministic random subset of identities (not images)."""
        rng = random.Random(seed)
        ids = self.identities()
        if n >= len(ids):
            return self
        chosen = set(rng.sample(ids, n))
        records = [(i, p) for i, p in self.records if i in chosen]
        return IdentityDataset(name=f"{self.name}[{n}id]", records=records, tag=self.tag)


def _scan_folder_per_identity(root: Path) -> list[tuple[str, Path]]:
    """LFW/RMFD-style layout: one folder per identity, images inside."""
    records: list[tuple[str, Path]] = []
    if not root.exists():
        return records
    for identity_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for img in sorted(identity_dir.iterdir()):
            if img.suffix.lower() in IMG_EXTS:
                records.append((identity_dir.name, img))
    return records


# ---------- LFW ----------------------------------------------------------------


def load_lfw() -> IdentityDataset:
    records = _scan_folder_per_identity(LFW_DIR)
    if not records:
        raise FileNotFoundError(
            f"LFW not found at {LFW_DIR}. Run scripts/download_datasets.py --lfw."
        )
    return IdentityDataset(name="lfw", records=records, tag="unmasked")


# ---------- RMFD --------------------------------------------------------------


def load_rmfd_masked() -> IdentityDataset:
    """Wuhan University RMFD: AFDB_masked_face_dataset/<person>/*.jpg.

    We also accept AFDB_face_dataset (the unmasked counterpart of the same
    identities), which is what enables the paired same-identity experiment.
    """
    masked_root = RMFD_DIR / "AFDB_masked_face_dataset"
    if not masked_root.exists():
        masked_root = RMFD_DIR / "self-built-masked-face-recognition-dataset" / "AFDB_masked_face_dataset"
    records = _scan_folder_per_identity(masked_root)
    if not records:
        raise FileNotFoundError(
            f"RMFD masked split not found under {RMFD_DIR}. See README."
        )
    return IdentityDataset(name="rmfd_masked", records=records, tag="masked")


def load_rmfd_unmasked() -> IdentityDataset:
    unmasked_root = RMFD_DIR / "AFDB_face_dataset"
    if not unmasked_root.exists():
        unmasked_root = RMFD_DIR / "self-built-masked-face-recognition-dataset" / "AFDB_face_dataset"
    records = _scan_folder_per_identity(unmasked_root)
    if not records:
        raise FileNotFoundError(
            f"RMFD unmasked split not found under {RMFD_DIR}. See README."
        )
    return IdentityDataset(name="rmfd_unmasked", records=records, tag="unmasked")


# ---------- MFR2 --------------------------------------------------------------

# MFR2 stores everything in a flat folder; filenames look like
#   Aamir_Khan_0001.png
# where the trailing _NNNN distinguishes images of the same identity. We also
# accept a per-identity-folder layout for robustness.

_MFR2_PATTERN = re.compile(r"^(?P<identity>.+?)_(?P<idx>\d{2,4})$")
_MFR2_MASKED_HINT = re.compile(r"mask|m_\d|masked", re.IGNORECASE)


def _mfr2_is_masked(path: Path) -> bool:
    """MFR2 ships a `mfr2_labels.txt`; absent that we fall back to filename hints.

    The real label file is CSV: ``<identity>, <idx>, <mask_type>`` where
    ``mask_type`` is ``no-mask`` for unmasked images and anything else
    (``surgical_blue``, ``surgical_white``, ``cloth``, ...) means masked.
    The lookup key built from a filename like ``AdrianDunbar_0001.png`` is
    ``(AdrianDunbar, 1)``.
    """
    labels_file = MFR2_DIR / "mfr2_labels.txt"
    if labels_file.exists():
        cache = getattr(_mfr2_is_masked, "_cache", None)
        if cache is None:
            cache = {}
            for line in labels_file.read_text().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                identity, idx_str, mask_type = parts[0], parts[1], parts[2]
                try:
                    idx = int(idx_str)
                except ValueError:
                    continue
                key = (identity, idx)
                cache[key] = mask_type.lower() != "no-mask"
            _mfr2_is_masked._cache = cache  # type: ignore[attr-defined]
        m = _MFR2_PATTERN.match(path.stem)
        if m:
            key = (m.group("identity"), int(m.group("idx")))
            if key in cache:
                return cache[key]
        # Unknown stem -> fall through to the filename heuristic.
    return bool(_MFR2_MASKED_HINT.search(path.stem))


def load_mfr2(masked: bool | None = None) -> IdentityDataset:
    """Return MFR2 records. `masked=None` returns everything."""
    if not MFR2_DIR.exists():
        raise FileNotFoundError(
            f"MFR2 not found at {MFR2_DIR}. Run scripts/download_datasets.py --mfr2."
        )

    candidates = list(MFR2_DIR.rglob("*"))
    image_paths = [p for p in candidates if p.is_file() and p.suffix.lower() in IMG_EXTS]
    records: list[tuple[str, Path]] = []
    for path in image_paths:
        match = _MFR2_PATTERN.match(path.stem)
        if not match:
            continue
        identity = match.group("identity")
        if masked is not None and _mfr2_is_masked(path) != masked:
            continue
        records.append((identity, path))

    tag = {True: "masked", False: "unmasked", None: "mixed"}[masked]
    return IdentityDataset(name=f"mfr2_{tag}", records=records, tag=tag)


# ---------- Pair construction ------------------------------------------------


@dataclass
class PairedSamples:
    """Bookkeeping for the cross-mode same-identity experiment."""

    identity: str
    enroll: Path
    probe_masked: Path | None
    probe_unmasked: Path | None


def build_paired_samples(
    masked_ds: IdentityDataset,
    unmasked_ds: IdentityDataset,
    enroll_from: str = "unmasked",
    seed: int = 0,
) -> list[PairedSamples]:
    """Pair masked and unmasked images per identity.

    For each identity present in both datasets we pick one unmasked image as
    the enrollment template (the realistic registration scenario) and the
    remaining images become probes -- one masked, one unmasked.
    """
    rng = random.Random(seed)
    masked_by_id = masked_ds.by_identity()
    unmasked_by_id = unmasked_ds.by_identity()
    shared = sorted(set(masked_by_id) & set(unmasked_by_id))

    pairs: list[PairedSamples] = []
    for identity in shared:
        un_imgs = list(unmasked_by_id[identity])
        m_imgs = list(masked_by_id[identity])
        rng.shuffle(un_imgs)
        rng.shuffle(m_imgs)

        if enroll_from == "unmasked":
            if not un_imgs:
                continue
            enroll = un_imgs.pop()
        else:
            if not m_imgs:
                continue
            enroll = m_imgs.pop()

        probe_unmasked = un_imgs[0] if un_imgs else None
        probe_masked = m_imgs[0] if m_imgs else None
        if probe_masked is None and probe_unmasked is None:
            continue
        pairs.append(
            PairedSamples(
                identity=identity,
                enroll=enroll,
                probe_masked=probe_masked,
                probe_unmasked=probe_unmasked,
            )
        )
    return pairs


def split_gallery_probe(
    dataset: IdentityDataset,
    seed: int = 0,
) -> tuple[dict[str, Path], list[tuple[str, Path]]]:
    """Classic open-set split: one image per identity for the gallery,
    the rest become probes. Mirrors the LFW evaluation protocol used by
    most of the past course projects.
    """
    rng = random.Random(seed)
    grouped = dataset.by_identity()
    gallery: dict[str, Path] = {}
    probes: list[tuple[str, Path]] = []
    for identity, paths in grouped.items():
        if not paths:
            continue
        shuffled = list(paths)
        rng.shuffle(shuffled)
        gallery[identity] = shuffled[0]
        for img in shuffled[1:]:
            probes.append((identity, img))
    return gallery, probes


__all__ = [
    "IdentityDataset",
    "PairedSamples",
    "load_lfw",
    "load_rmfd_masked",
    "load_rmfd_unmasked",
    "load_mfr2",
    "build_paired_samples",
    "split_gallery_probe",
]
