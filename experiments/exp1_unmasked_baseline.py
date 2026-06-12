"""Experiment 1 -- Unmasked face recognition baseline on LFW.

This is the analogue of Smart Peephole's LFW experiment: classical open-set
identification with one gallery template per identity. The point is to
establish how well our pre-trained ArcFace stack performs in the easy case,
so the masked experiments have a number to be measured against.

Protocol:
  * Restrict LFW to identities with >= 2 images so each one has at least
    one enrollment + one probe.
  * Sample N identities (default 200) for the gallery (these become the
    "enrolled users"). The remaining identities supply impostor probes.
  * Each enrolled identity contributes one randomly-chosen image to the
    gallery; the rest of their images become genuine probes.
  * Each impostor identity contributes up to 2 probes.

We compute FAR/FRR, ROC, CMC, score histograms, and a JSON summary.

Run:
    python -m experiments.exp1_unmasked_baseline --n-identities 200 --probes-per-impostor 2
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from src.config import FIGURES_DIR, RESULTS_DIR
from src.datasets import IdentityDataset, load_lfw, split_gallery_probe
from src.embedder import Embedder
from src.evaluator import (
    evaluate,
    plot_cmc,
    plot_far_frr,
    plot_roc,
    plot_score_distributions,
)

from ._common import build_similarity_matrix, compute_embeddings, write_summary


def _split_identities(
    dataset: IdentityDataset,
    n_genuine: int,
    n_impostor: int,
    seed: int,
) -> tuple[IdentityDataset, IdentityDataset]:
    """Carve the dataset into "enrolled" and "impostor" identity pools."""
    rng = random.Random(seed)
    ids = dataset.identities()
    rng.shuffle(ids)
    enrolled_ids = set(ids[:n_genuine])
    impostor_ids = set(ids[n_genuine : n_genuine + n_impostor])
    enrolled = IdentityDataset(
        name=f"{dataset.name}_enrolled",
        records=[(i, p) for i, p in dataset.records if i in enrolled_ids],
        tag=dataset.tag,
    )
    impostors = IdentityDataset(
        name=f"{dataset.name}_impostors",
        records=[(i, p) for i, p in dataset.records if i in impostor_ids],
        tag=dataset.tag,
    )
    return enrolled, impostors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-identities", type=int, default=200)
    ap.add_argument("--n-impostor-identities", type=int, default=150)
    ap.add_argument("--probes-per-impostor", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 70)
    print("Experiment 1 -- Unmasked LFW baseline")
    print("=" * 70)

    lfw = load_lfw().filter_min_samples(2)
    print(f"LFW with >=2 images per identity: {len(lfw.identities())} identities, {len(lfw)} images")

    enrolled, impostors = _split_identities(
        lfw, args.n_identities, args.n_impostor_identities, args.seed
    )
    print(f"Enrolled: {len(enrolled.identities())} identities | Impostors: {len(impostors.identities())}")

    # Gallery + genuine probes from the enrolled pool.
    gallery_map, genuine_probes = split_gallery_probe(enrolled, seed=args.seed)
    print(f"Gallery: {len(gallery_map)} templates | Genuine probes: {len(genuine_probes)}")

    # Two impostor probes per impostor identity (cap).
    rng = random.Random(args.seed)
    impostor_probes: list[tuple[str, Path]] = []
    for identity, paths in impostors.by_identity().items():
        sampled = rng.sample(paths, k=min(args.probes_per_impostor, len(paths)))
        impostor_probes.extend((identity, p) for p in sampled)
    print(f"Impostor probes: {len(impostor_probes)}")

    # ---- Embed everything (cached) -----------------------------------------
    embedder = Embedder()
    gallery_records = [(i, p) for i, p in gallery_map.items()]
    g_cache, g_skipped = compute_embeddings(
        gallery_records, embedder, modes=("full",), cache_name="exp1_gallery",
        desc="gallery",
    )
    p_cache, p_skipped = compute_embeddings(
        genuine_probes + impostor_probes, embedder, modes=("full",),
        cache_name="exp1_probes", desc="probes",
    )
    print(f"Skipped (no face) -- gallery: {len(g_skipped)} | probes: {len(p_skipped)}")

    # ---- Stack into matrices ------------------------------------------------
    gallery_ids = [i for i, p in gallery_records if g_cache.get(p, "full") is not None]
    gallery_emb = np.stack([g_cache.get(dict(gallery_records)[i], "full") for i in gallery_ids], axis=0)

    probe_records = genuine_probes + impostor_probes
    probe_is_genuine_full: list[bool] = (
        [True] * len(genuine_probes) + [False] * len(impostor_probes)
    )
    probe_emb_list: list[np.ndarray] = []
    probe_labels: list[str] = []
    probe_is_genuine: list[bool] = []
    for (identity, path), is_gen in zip(probe_records, probe_is_genuine_full):
        emb = p_cache.get(path, "full")
        if emb is None:
            continue
        probe_emb_list.append(emb)
        probe_labels.append(identity)
        probe_is_genuine.append(is_gen)

    sims = build_similarity_matrix(probe_emb_list, gallery_emb)
    print(f"Similarity matrix: {sims.shape}")

    report = evaluate(sims, probe_labels, gallery_ids, probe_is_genuine)
    print(json.dumps(report.summary(), indent=2))

    # ---- Plots --------------------------------------------------------------
    fig_dir = FIGURES_DIR / "exp1"
    plot_far_frr(report, "Unmasked LFW", fig_dir / "far_frr.png")
    plot_roc(report, "Unmasked LFW", fig_dir / "roc.png")
    plot_cmc(report, "Unmasked LFW", fig_dir / "cmc.png", max_rank=20)
    plot_score_distributions(report, "Unmasked LFW", fig_dir / "score_dist.png")

    write_summary("exp1_unmasked", {
        "n_gallery": len(gallery_ids),
        "n_genuine_probes": int(sum(probe_is_genuine)),
        "n_impostor_probes": int(len(probe_is_genuine) - sum(probe_is_genuine)),
        **report.summary(),
        "args": vars(args),
    })
    print(f"Summary written to {RESULTS_DIR / 'exp1_unmasked_summary.json'}")
    print(f"Figures: {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
