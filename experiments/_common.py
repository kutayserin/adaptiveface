"""Shared utilities for experiment scripts.

The main job here is embedding caching. Embedding a single face on GPU takes
~10 ms, but for LFW that's still 13K images per mode and we want to iterate
on metrics without re-embedding every time. We hash on (dataset name, image
path, mode) and dump results to `results/cache/`.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

# Allow `python -m experiments.expN` from project root and `python experiments/expN.py` both.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CACHE_DIR, RESULTS_DIR  # noqa: E402
from src.embedder import Embedder  # noqa: E402
from src.face_detector import read_image  # noqa: E402


@dataclass
class EmbeddingCache:
    """Mapping from image path -> (mode -> embedding).

    Keys are POSIX strings to stay stable across OSes. Missing entries are
    treated as "needs computing"; failed detections are stored as None.
    """

    name: str
    embeddings: dict[str, dict[str, np.ndarray | None]]

    @classmethod
    def empty(cls, name: str) -> "EmbeddingCache":
        return cls(name=name, embeddings={})

    def needs(self, path: Path, mode: str) -> bool:
        key = path.as_posix()
        return key not in self.embeddings or mode not in self.embeddings[key]

    def get(self, path: Path, mode: str) -> np.ndarray | None:
        return self.embeddings.get(path.as_posix(), {}).get(mode)

    def set(self, path: Path, mode: str, value: np.ndarray | None) -> None:
        key = path.as_posix()
        self.embeddings.setdefault(key, {})[mode] = value

    @property
    def cache_path(self) -> Path:
        h = hashlib.md5(self.name.encode()).hexdigest()[:10]
        return CACHE_DIR / f"emb_{self.name}_{h}.npz"

    @property
    def index_path(self) -> Path:
        return self.cache_path.with_suffix(".json")

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self.embeddings.keys())
        modes = sorted({m for v in self.embeddings.values() for m in v})
        arrays = {}
        index = {"keys": keys, "modes": modes}
        for mode in modes:
            stack = np.zeros((len(keys), 512), dtype=np.float32)
            valid = np.zeros(len(keys), dtype=bool)
            for i, k in enumerate(keys):
                v = self.embeddings[k].get(mode)
                if v is not None:
                    stack[i] = v
                    valid[i] = True
            arrays[f"{mode}_emb"] = stack
            arrays[f"{mode}_valid"] = valid
        np.savez(self.cache_path, **arrays)
        self.index_path.write_text(json.dumps(index))

    @classmethod
    def load(cls, name: str) -> "EmbeddingCache":
        c = cls.empty(name)
        if not c.cache_path.exists():
            return c
        index = json.loads(c.index_path.read_text())
        data = np.load(c.cache_path)
        for i, key in enumerate(index["keys"]):
            entry: dict[str, np.ndarray | None] = {}
            for mode in index["modes"]:
                if data[f"{mode}_valid"][i]:
                    entry[mode] = data[f"{mode}_emb"][i].astype(np.float32)
                else:
                    entry[mode] = None
            c.embeddings[key] = entry
        return c


def compute_embeddings(
    records: Iterable[tuple[str, Path]],
    embedder: Embedder,
    modes: tuple[str, ...] = ("full", "upper"),
    cache_name: str | None = None,
    desc: str = "embedding",
) -> tuple[EmbeddingCache, list[tuple[str, Path]]]:
    """Compute embeddings for a list of (identity, path) records.

    Returns (cache, skipped) where `skipped` lists records with no detection.
    Uses the cache to avoid re-embedding identical (path, mode) pairs.
    """
    records = list(records)
    cache = EmbeddingCache.load(cache_name) if cache_name else EmbeddingCache.empty("inline")
    skipped: list[tuple[str, Path]] = []

    for identity, path in tqdm(records, desc=desc):
        if all(not cache.needs(path, m) for m in modes):
            # Already cached -- but check whether either mode failed before.
            if any(cache.get(path, m) is None for m in modes):
                skipped.append((identity, path))
            continue
        image_bgr = read_image(path)
        if image_bgr is None:
            for m in modes:
                cache.set(path, m, None)
            skipped.append((identity, path))
            continue
        faces = embedder.detector.detect(image_bgr, max_faces=1)
        if not faces:
            for m in modes:
                cache.set(path, m, None)
            skipped.append((identity, path))
            continue
        face = faces[0]
        for mode in modes:
            if mode == "full":
                cache.set(path, mode, embedder.embed_full(image_bgr, face))
            elif mode == "upper":
                cache.set(path, mode, embedder.embed_upper(image_bgr, face))
            else:
                raise ValueError(mode)

    if cache_name:
        cache.save()
    return cache, skipped


def build_similarity_matrix(
    probe_embeddings: list[np.ndarray],
    gallery_embeddings: np.ndarray,
) -> np.ndarray:
    """(N_probe, 512) @ (N_gallery, 512).T -> (N_probe, N_gallery)."""
    if not probe_embeddings:
        return np.zeros((0, gallery_embeddings.shape[0]), dtype=np.float32)
    probes = np.stack(probe_embeddings, axis=0).astype(np.float32)
    return probes @ gallery_embeddings.T


def write_summary(name: str, summary: dict) -> Path:
    out = RESULTS_DIR / f"{name}_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=float))
    return out


__all__ = [
    "EmbeddingCache",
    "compute_embeddings",
    "build_similarity_matrix",
    "write_summary",
]
