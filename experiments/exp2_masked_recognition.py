"""Experiment 2 -- Masked face recognition: full-face vs upper-face routing.

Probes are masked images of identities enrolled with their unmasked photos.
We compare two operating modes on the *same* probe set:

  * `full` -- feed the masked probe straight into ArcFace (baseline).
  * `upper` -- mask out the lower half before embedding (our masked branch).

If RMFD is available we use it (real masks, paired same-identity). Otherwise
we synthesise masks on top of LFW so the experiment still runs -- the
README documents the difference and Experiment 3 keeps the cross-mode
analysis on whichever dataset we end up using.

Run:
    python -m experiments.exp2_masked_recognition --source rmfd  # or 'synth_lfw'
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.config import CACHE_DIR, FIGURES_DIR, RESULTS_DIR
from src.datasets import (
    IdentityDataset,
    load_lfw,
    load_rmfd_masked,
    load_rmfd_unmasked,
    split_gallery_probe,
)
from src.embedder import Embedder
from src.evaluator import (
    evaluate,
    plot_cmc,
    plot_far_frr,
    plot_roc,
    plot_score_distributions,
)
from src.face_detector import read_image
from src.synth_mask import overlay_synthetic_mask

from ._common import build_similarity_matrix, compute_embeddings, write_summary


def _materialise_synth_masked(
    dataset: IdentityDataset,
    embedder: Embedder,
    out_dir: Path,
    max_per_identity: int = 3,
) -> IdentityDataset:
    """Render synthetic masks for `dataset` and write them under `out_dir`.

    We cap per-identity counts to keep the dataset small and balanced. The
    rendered images live on disk so future runs can reuse them.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[tuple[str, Path]] = []
    by_id = dataset.by_identity()
    for identity, paths in tqdm(by_id.items(), desc="synth-masking"):
        kept = 0
        for src in paths:
            if kept >= max_per_identity:
                break
            dst = out_dir / identity / f"{src.stem}_synthmask.jpg"
            if dst.exists():
                records.append((identity, dst))
                kept += 1
                continue
            img = read_image(src)
            if img is None:
                continue
            faces = embedder.detector.detect(img, max_faces=1)
            if not faces:
                continue
            rendered = overlay_synthetic_mask(img, faces[0], color="surgical_blue")
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), rendered)
            records.append((identity, dst))
            kept += 1
    return IdentityDataset(name=f"synth_masked_{dataset.name}", records=records, tag="masked")


def _build_rmfd_pairs(seed: int) -> tuple[IdentityDataset, IdentityDataset]:
    """Return (unmasked_enrollment_pool, masked_probe_pool) from RMFD."""
    masked = load_rmfd_masked()
    unmasked = load_rmfd_unmasked()
    shared = sorted(set(masked.identities()) & set(unmasked.identities()))
    if not shared:
        raise RuntimeError("No identities are shared between RMFD masked and unmasked splits.")
    masked_records = [(i, p) for i, p in masked.records if i in shared]
    unmasked_records = [(i, p) for i, p in unmasked.records if i in shared]
    return (
        IdentityDataset(name="rmfd_unmasked_shared", records=unmasked_records, tag="unmasked"),
        IdentityDataset(name="rmfd_masked_shared", records=masked_records, tag="masked"),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=("rmfd", "synth_lfw", "auto"), default="auto")
    ap.add_argument("--n-identities", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 70)
    print("Experiment 2 -- Masked recognition")
    print("=" * 70)

    embedder = Embedder()

    # ---- Choose data source -------------------------------------------------
    use_rmfd = args.source == "rmfd" or (args.source == "auto")
    unmasked_pool: IdentityDataset
    masked_pool: IdentityDataset
    if use_rmfd:
        try:
            unmasked_pool, masked_pool = _build_rmfd_pairs(args.seed)
            print(f"Using RMFD: {len(set(unmasked_pool.identities()))} shared identities")
        except (FileNotFoundError, RuntimeError) as exc:
            if args.source == "rmfd":
                raise
            print(f"[warn] RMFD unavailable ({exc}); falling back to synthetic LFW.")
            use_rmfd = False

    if not use_rmfd:
        lfw = load_lfw().filter_min_samples(2).sample_identities(args.n_identities, seed=args.seed)
        unmasked_pool = lfw
        masked_pool = _materialise_synth_masked(lfw, embedder, CACHE_DIR / "synth_masked_lfw")
        print(f"Using synth-masked LFW: {len(set(masked_pool.identities()))} identities")

    # Optional subsampling for speed.
    shared_ids = sorted(set(unmasked_pool.identities()) & set(masked_pool.identities()))
    rng = random.Random(args.seed)
    rng.shuffle(shared_ids)
    if len(shared_ids) > args.n_identities:
        shared_ids = shared_ids[: args.n_identities]
    print(f"Working with {len(shared_ids)} identities for evaluation.")

    unmasked_pool = IdentityDataset(
        name=unmasked_pool.name,
        records=[(i, p) for i, p in unmasked_pool.records if i in set(shared_ids)],
        tag="unmasked",
    )
    masked_pool = IdentityDataset(
        name=masked_pool.name,
        records=[(i, p) for i, p in masked_pool.records if i in set(shared_ids)],
        tag="masked",
    )

    # ---- Build gallery from unmasked enrollment ----------------------------
    gallery_map, _ = split_gallery_probe(unmasked_pool, seed=args.seed)
    print(f"Gallery: {len(gallery_map)} unmasked enrollment templates")

    # All masked images of enrolled identities become probes.
    masked_probes: list[tuple[str, Path]] = [
        (i, p) for i, p in masked_pool.records if i in gallery_map
    ]
    print(f"Masked probes: {len(masked_probes)}")

    # ---- Embeddings ---------------------------------------------------------
    gallery_records = list(gallery_map.items())
    g_cache, _ = compute_embeddings(gallery_records, embedder, modes=("full", "upper"),
                                    cache_name=f"exp2_gallery_{'rmfd' if use_rmfd else 'synth'}",
                                    desc="enroll")
    p_cache, _ = compute_embeddings(masked_probes, embedder, modes=("full", "upper"),
                                    cache_name=f"exp2_probes_{'rmfd' if use_rmfd else 'synth'}",
                                    desc="probes")

    summaries: dict[str, dict] = {}
    fig_root = FIGURES_DIR / "exp2"
    fig_root.mkdir(parents=True, exist_ok=True)

    for mode in ("full", "upper"):
        gallery_ids = [i for i, p in gallery_records if g_cache.get(p, mode) is not None]
        gallery_emb = np.stack([g_cache.get(dict(gallery_records)[i], mode) for i in gallery_ids], axis=0)
        probe_emb_list: list[np.ndarray] = []
        probe_labels: list[str] = []
        for identity, path in masked_probes:
            emb = p_cache.get(path, mode)
            if emb is None:
                continue
            probe_emb_list.append(emb)
            probe_labels.append(identity)

        sims = build_similarity_matrix(probe_emb_list, gallery_emb)
        report = evaluate(sims, probe_labels, gallery_ids, probe_is_genuine=[True] * len(probe_labels))
        print(f"[{mode}] {report.summary()}")
        summaries[mode] = report.summary()
        # Identifier for the plot files differentiates the two operating modes.
        title_suffix = "masked probes, full-face embed" if mode == "full" else "masked probes, upper-face embed"
        plot_far_frr(report, title_suffix, fig_root / f"far_frr_{mode}.png")
        plot_roc(report, title_suffix, fig_root / f"roc_{mode}.png")
        plot_cmc(report, title_suffix, fig_root / f"cmc_{mode}.png", max_rank=20)
        plot_score_distributions(report, title_suffix, fig_root / f"score_dist_{mode}.png")

    write_summary("exp2_masked", {
        "data_source": "rmfd" if use_rmfd else "synth_lfw",
        "n_gallery": len(gallery_map),
        "n_probes": len(masked_probes),
        "results": summaries,
        "args": vars(args),
    })
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
