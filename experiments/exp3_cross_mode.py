"""Experiment 3 -- Cross-mode same-identity (the headline experiment).

The story we want to tell on the slides:

  1. A user enrols once with an unmasked photo.
  2. At test time the user may show up masked or unmasked.
  3. A naive face-recognition system trained on unmasked faces breaks
     when the user wears a mask.
  4. Our adaptive system -- mask detection + upper-face routing + dual
     gallery templates -- closes most of the gap *without retraining*.

We measure that gap quantitatively with three configurations:

  * `naive` -- always use the full-face matcher and full-face gallery.
  * `oracle` -- route based on the ground-truth mask label.
  * `adaptive` -- route based on our mask classifier.

For each configuration we report Rank-1 identification rate, EER on
verification, and accept rate on impostor probes (an open-set splice
provides the impostors).

Run:
    python -m experiments.exp3_cross_mode --source auto
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

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
from src.evaluator import evaluate, plot_cmc, plot_far_frr, plot_roc, plot_score_distributions
from src.face_detector import read_image
from src.mask_classifier import MaskClassifier
from src.synth_mask import overlay_synthetic_mask

from ._common import build_similarity_matrix, compute_embeddings, write_summary
from .exp2_masked_recognition import _build_rmfd_pairs, _materialise_synth_masked


@dataclass
class Probe:
    identity: str
    path: Path
    is_masked_truth: bool
    is_genuine: bool  # False -> impostor (unknown identity, not in gallery)


def _build_probe_set(
    enrolled_unmasked: IdentityDataset,
    enrolled_masked: IdentityDataset,
    impostor_pool: IdentityDataset,
    gallery_ids: set[str],
    seed: int,
    max_probes_per_id: int = 2,
    max_impostor_probes: int = 200,
) -> list[Probe]:
    rng = random.Random(seed)
    probes: list[Probe] = []

    for identity, paths in enrolled_unmasked.by_identity().items():
        if identity not in gallery_ids:
            continue
        sampled = rng.sample(paths, k=min(max_probes_per_id, len(paths)))
        probes.extend(Probe(identity, p, False, True) for p in sampled)

    for identity, paths in enrolled_masked.by_identity().items():
        if identity not in gallery_ids:
            continue
        sampled = rng.sample(paths, k=min(max_probes_per_id, len(paths)))
        probes.extend(Probe(identity, p, True, True) for p in sampled)

    impostor_records = list(impostor_pool.records)
    rng.shuffle(impostor_records)
    impostor_records = impostor_records[:max_impostor_probes]
    for identity, path in impostor_records:
        probes.append(Probe(identity, path, False, False))

    rng.shuffle(probes)
    return probes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=("rmfd", "synth_lfw", "auto"), default="auto")
    ap.add_argument("--n-identities", type=int, default=150)
    ap.add_argument("--n-impostors", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 70)
    print("Experiment 3 -- Cross-mode same-identity")
    print("=" * 70)

    embedder = Embedder()

    # ---- Choose data --------------------------------------------------------
    use_rmfd = args.source in ("rmfd", "auto")
    if use_rmfd:
        try:
            unmasked_pool, masked_pool = _build_rmfd_pairs(args.seed)
            impostor_pool = load_lfw().filter_min_samples(1).sample_identities(args.n_impostors, seed=args.seed + 1)
            data_source = "rmfd"
        except (FileNotFoundError, RuntimeError) as exc:
            if args.source == "rmfd":
                raise
            print(f"[warn] RMFD unavailable ({exc}); falling back to synthetic LFW.")
            use_rmfd = False

    if not use_rmfd:
        lfw = load_lfw().filter_min_samples(2)
        enrolled = lfw.sample_identities(args.n_identities + args.n_impostors, seed=args.seed)
        all_ids = enrolled.identities()
        rng = random.Random(args.seed)
        rng.shuffle(all_ids)
        keep_for_enroll = set(all_ids[: args.n_identities])
        impostor_ids = set(all_ids[args.n_identities : args.n_identities + args.n_impostors])
        unmasked_pool = IdentityDataset(
            name="lfw_enrolled_unmasked",
            records=[(i, p) for i, p in enrolled.records if i in keep_for_enroll],
            tag="unmasked",
        )
        masked_pool = _materialise_synth_masked(
            unmasked_pool, embedder, RESULTS_DIR.parent / "data" / "synth_masked_lfw"
        )
        impostor_pool = IdentityDataset(
            name="lfw_impostors",
            records=[(i, p) for i, p in enrolled.records if i in impostor_ids],
            tag="unmasked",
        )
        data_source = "synth_lfw"

    # Subsample identities for tractable runtime.
    shared_ids = sorted(set(unmasked_pool.identities()) & set(masked_pool.identities()))
    rng = random.Random(args.seed)
    rng.shuffle(shared_ids)
    shared_ids = shared_ids[: args.n_identities]
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
    print(f"Using {data_source}: {len(shared_ids)} enrolled identities | impostors: {len(impostor_pool.identities())}")

    # ---- Gallery: one unmasked photo per identity --------------------------
    gallery_map, leftover_unmasked = split_gallery_probe(unmasked_pool, seed=args.seed)
    # `leftover_unmasked` are the unmasked probes (one was reserved as gallery).
    leftover_pool = IdentityDataset(
        name="unmasked_probes",
        records=leftover_unmasked,
        tag="unmasked",
    )

    gallery_records = list(gallery_map.items())
    g_cache, _ = compute_embeddings(
        gallery_records, embedder, modes=("full", "upper"),
        cache_name=f"exp3_gallery_{data_source}", desc="enroll",
    )

    # ---- Multi-template gallery: also enroll a synthetically-masked
    # version of each unmasked enrollment photo. This gives the matcher a
    # template that already "looks masked", so a real masked probe has
    # something distribution-matched to compare against without us having
    # to do any routing or fusion at the score level. We render the synth
    # mask once and embed it via the normal full-face path -- the model
    # sees a masked face exactly the way it would at recognition time.
    synth_dir = CACHE_DIR / f"exp3_synth_gallery_{data_source}"
    synth_dir.mkdir(parents=True, exist_ok=True)
    synth_records: list[tuple[str, Path]] = []
    print(f"Building synthetic-masked enrollment counterparts in {synth_dir}")
    for identity, path in tqdm(gallery_records, desc="synth-enroll"):
        dst = synth_dir / identity / f"{path.stem}_synth.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            img = read_image(path)
            if img is None:
                continue
            faces = embedder.detector.detect(img, max_faces=1)
            if not faces:
                continue
            rendered = overlay_synthetic_mask(img, faces[0], color="surgical_blue")
            import cv2 as _cv2
            _cv2.imwrite(str(dst), rendered)
        if dst.exists():
            synth_records.append((identity, dst))

    sg_cache, _ = compute_embeddings(
        synth_records, embedder, modes=("full",),
        cache_name=f"exp3_synth_gallery_{data_source}", desc="synth-embed",
    )

    # ---- Probes -------------------------------------------------------------
    probes = _build_probe_set(
        enrolled_unmasked=leftover_pool,
        enrolled_masked=masked_pool,
        impostor_pool=impostor_pool,
        gallery_ids=set(gallery_map.keys()),
        seed=args.seed,
    )
    print(f"Probes: {len(probes)} total "
          f"(genuine={sum(p.is_genuine for p in probes)}, "
          f"impostor={sum(not p.is_genuine for p in probes)}, "
          f"masked={sum(p.is_masked_truth for p in probes)}).")
    probe_records = [(p.identity, p.path) for p in probes]
    p_cache, p_skipped = compute_embeddings(
        probe_records, embedder, modes=("full", "upper"),
        cache_name=f"exp3_probes_{data_source}", desc="probes",
    )

    # ---- Mask classifier on probes -----------------------------------------
    mask_clf = MaskClassifier()
    print(f"Mask classifier backend: {mask_clf.backend_name}")
    mask_predictions: list[bool] = []
    for p in tqdm(probes, desc="mask-detect"):
        img = read_image(p.path)
        if img is None:
            mask_predictions.append(False)
            continue
        faces = embedder.detector.detect(img, max_faces=1)
        face = faces[0] if faces else None
        pred = mask_clf.predict(img, face)
        mask_predictions.append(pred.is_masked)

    # ---- Build matrices for each configuration -----------------------------
    gallery_dict = dict(gallery_records)
    synth_dict = dict(synth_records)
    # Keep identities that have a real full-face template AND a synthetic
    # masked counterpart -- otherwise the multi-template configurations
    # can't score a probe against both gallery rows.
    gallery_ids = [
        i for i in (g_id for g_id, _ in gallery_records)
        if g_cache.get(gallery_dict[i], "full") is not None
        and i in synth_dict
        and sg_cache.get(synth_dict[i], "full") is not None
    ]
    gallery_full = np.stack([g_cache.get(gallery_dict[i], "full") for i in gallery_ids], axis=0)
    gallery_upper = np.stack([g_cache.get(gallery_dict[i], "upper") for i in gallery_ids], axis=0)
    gallery_synth = np.stack([sg_cache.get(synth_dict[i], "full") for i in gallery_ids], axis=0)

    summaries: dict[str, dict] = {}
    figroot = FIGURES_DIR / "exp3"
    figroot.mkdir(parents=True, exist_ok=True)

    # Keep probes that have BOTH a full and an upper embedding (we need both
    # for the fusion configurations; routing configurations only need one,
    # but using a common probe set keeps comparisons fair).
    kept_probes: list[Probe] = []
    kept_indices: list[int] = []
    probe_full_list: list[np.ndarray] = []
    probe_upper_list: list[np.ndarray] = []
    for idx, probe in enumerate(probes):
        emb_full = p_cache.get(probe.path, "full")
        emb_upper = p_cache.get(probe.path, "upper")
        if emb_full is None or emb_upper is None:
            continue
        probe_full_list.append(emb_full)
        probe_upper_list.append(emb_upper)
        kept_probes.append(probe)
        kept_indices.append(idx)

    if not kept_probes:
        print("[err ] no usable probes; aborting.")
        return 1

    probes_full = np.stack(probe_full_list, axis=0)
    probes_upper = np.stack(probe_upper_list, axis=0)
    sims_full = probes_full @ gallery_full.T
    sims_upper = probes_upper @ gallery_upper.T
    # Multi-template: probe-full vs synthetic-masked gallery template.
    # We use the probe's full embedding (not upper) because the synthetic
    # gallery image already encodes the mask, so the probe should be
    # whole-face for a fair comparison.
    sims_synth = probes_full @ gallery_synth.T

    # Routing configs pick one similarity matrix per probe.
    # Fusion configs combine both matrices element-wise.
    def routed(modes: list[str]) -> np.ndarray:
        return np.where(np.array(modes).reshape(-1, 1) == "upper", sims_upper, sims_full)

    mask_pred_for_kept = np.array([mask_predictions[i] for i in kept_indices])
    truth_for_kept = np.array([p.is_masked_truth for p in kept_probes])

    naive_modes = ["full"] * len(kept_probes)
    oracle_modes = ["upper" if t else "full" for t in truth_for_kept]
    adaptive_modes = ["upper" if m else "full" for m in mask_pred_for_kept]

    # Fusion strategies. The fusion is computed across BOTH gallery branches:
    # each fusion combines (probe_full vs gallery_full) with (probe_upper vs
    # gallery_upper) element-wise. This is the standard score-level fusion
    # pattern (analogous to Biotouch's avg / max / weighted_avg ensembles).
    sims_max = np.maximum(sims_full, sims_upper)
    sims_avg = 0.5 * (sims_full + sims_upper)
    # Weighted fusion: when the mask detector flags masked, lean on upper;
    # when it flags unmasked, lean on full. We bias gently (0.7 / 0.3) so
    # one wrong mask call doesn't catastrophically reroute the score.
    w_upper = np.where(mask_pred_for_kept, 0.7, 0.3).reshape(-1, 1)
    w_full = 1.0 - w_upper
    sims_weighted = w_full * sims_full + w_upper * sims_upper

    # Multi-template strategies: each gallery identity has TWO rows --
    # one unmasked (gallery_full) and one synthetically masked
    # (gallery_synth). Match against whichever fits the probe best.
    sims_multi_max = np.maximum(sims_full, sims_synth)
    # Adaptive multi-template: if the mask classifier flags a probe as
    # masked, only score it against the synthetic-masked template (better
    # distribution match); otherwise score against the unmasked template.
    sims_multi_adaptive = np.where(
        mask_pred_for_kept.reshape(-1, 1),
        sims_synth,
        sims_full,
    )

    configurations: dict[str, tuple[np.ndarray, str]] = {
        "naive": (routed(naive_modes), "naive (always full)"),
        "oracle": (routed(oracle_modes), "oracle routing"),
        "adaptive": (routed(adaptive_modes), "adaptive routing"),
        "fusion_max": (sims_max, "fusion: max(full, upper)"),
        "fusion_avg": (sims_avg, "fusion: avg(full, upper)"),
        "fusion_weighted": (sims_weighted, "fusion: detector-weighted"),
        "multi_max": (sims_multi_max, "multi-template: max(real, synth-mask)"),
        "multi_adaptive": (sims_multi_adaptive, "multi-template: adaptive routing"),
    }

    # Track which mode each probe was routed to, for the CSV.
    modes_by_cfg = {
        "naive": naive_modes,
        "oracle": oracle_modes,
        "adaptive": adaptive_modes,
        "fusion_max": ["both"] * len(kept_probes),
        "fusion_avg": ["both"] * len(kept_probes),
        "fusion_weighted": ["both"] * len(kept_probes),
        "multi_max": ["multi"] * len(kept_probes),
        "multi_adaptive": [
            "synth_tpl" if m else "real_tpl" for m in mask_pred_for_kept
        ],
    }

    probe_labels = [p.identity for p in kept_probes]
    probe_is_genuine = [p.is_genuine for p in kept_probes]

    rows = []
    for cfg_name, (sims, title) in configurations.items():
        report = evaluate(sims, probe_labels, gallery_ids, probe_is_genuine)
        summaries[cfg_name] = report.summary()
        print(f"[{cfg_name:18s}] {report.summary()}")
        plot_far_frr(report, title, figroot / f"far_frr_{cfg_name}.png")
        plot_roc(report, title, figroot / f"roc_{cfg_name}.png")
        plot_cmc(report, title, figroot / f"cmc_{cfg_name}.png", max_rank=20)
        plot_score_distributions(report, title, figroot / f"score_dist_{cfg_name}.png")

        modes = modes_by_cfg[cfg_name]
        for i, probe in enumerate(kept_probes):
            rows.append({
                "config": cfg_name,
                "identity": probe.identity,
                "path": str(probe.path),
                "is_masked_truth": probe.is_masked_truth,
                "is_genuine": probe.is_genuine,
                "mask_detector_says_masked": bool(mask_pred_for_kept[i]),
                "routed_mode": modes[i],
                "top_similarity": float(sims[i].max()),
                "top_match": gallery_ids[int(sims[i].argmax())],
            })

    # ---- Mask classifier accuracy on this probe set ------------------------
    correct = sum(1 for p, pred in zip(probes, mask_predictions) if p.is_masked_truth == pred)
    mask_acc = correct / max(1, len(probes))
    print(f"Mask classifier accuracy on probe set: {mask_acc:.3f}")

    write_summary("exp3_cross_mode", {
        "data_source": data_source,
        "mask_classifier_backend": mask_clf.backend_name,
        "mask_classifier_accuracy": mask_acc,
        "n_gallery": len(gallery_ids),
        "n_probes": len(probes),
        "results": summaries,
        "args": vars(args),
    })

    # Per-probe CSV for failure analysis.
    import csv
    out_csv = RESULTS_DIR / "exp3_per_probe.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Per-probe rows written to {out_csv}")
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
