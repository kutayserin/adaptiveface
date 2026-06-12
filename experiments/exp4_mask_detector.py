"""Experiment 4 -- Mask classifier sanity check.

Routing the pipeline depends on getting the mask/no-mask decision right.
We evaluate the classifier (ONNX or heuristic fallback) on a mixed set of
masked + unmasked images and report:

  * confusion matrix
  * accuracy, precision/recall per class
  * ROC + AUC on `p_mask`

We sample from whichever sources are available so this experiment can run
without RMFD.

Run:
    python -m experiments.exp4_mask_detector
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# IMPORTANT: instantiate onnxruntime BEFORE importing sklearn. On Windows the
# BLAS/LAPACK DLLs shipped with newer scikit-learn versions clash with the C
# runtime onnxruntime expects -- whichever DLL set loads first wins. We
# preload onnxruntime + insightface explicitly here so a later sklearn import
# can't poison the DLL search path.
import onnxruntime as _ort  # noqa: F401  (side effect: load DLLs first)
import insightface  # noqa: F401  (ensures insightface's bundled C libs come up)
from insightface.app import FaceAnalysis as _FaceAnalysis  # noqa: F401

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

from src.config import FIGURES_DIR, RESULTS_DIR
from src.datasets import IdentityDataset, load_lfw, load_rmfd_masked, load_rmfd_unmasked
from src.embedder import Embedder
from src.face_detector import read_image
from src.mask_classifier import MaskClassifier
from src.synth_mask import overlay_synthetic_mask

from ._common import write_summary


def _sample_records(dataset: IdentityDataset, n: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    paths = [p for _, p in dataset.records]
    rng.shuffle(paths)
    return paths[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples-per-class", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-synth-fallback", action="store_true",
                    help="If RMFD unavailable, build masked samples synthetically from LFW.")
    args = ap.parse_args()

    print("=" * 70)
    print("Experiment 4 -- Mask classifier evaluation")
    print("=" * 70)

    embedder = Embedder()
    classifier = MaskClassifier()
    print(f"Mask classifier backend: {classifier.backend_name}")

    masked_paths: list[Path] = []
    unmasked_paths: list[Path] = []

    try:
        rmfd_m = load_rmfd_masked()
        rmfd_u = load_rmfd_unmasked()
        masked_paths = _sample_records(rmfd_m, args.samples_per_class, args.seed)
        unmasked_paths = _sample_records(rmfd_u, args.samples_per_class, args.seed)
        source = "rmfd"
    except FileNotFoundError as exc:
        if not args.use_synth_fallback:
            print(f"[warn] RMFD unavailable ({exc}); rerun with --use-synth-fallback to evaluate against synthetic masks.")
            return 1
        lfw = load_lfw().filter_min_samples(1)
        unmasked_paths = _sample_records(lfw, args.samples_per_class, args.seed)
        masked_paths = []
        for p in tqdm(unmasked_paths, desc="synth-masking"):
            img = read_image(p)
            if img is None:
                continue
            faces = embedder.detector.detect(img, max_faces=1)
            if not faces:
                continue
            rendered = overlay_synthetic_mask(img, faces[0])
            out_dir = RESULTS_DIR / "cache" / "exp4_synth_masked"
            out_dir.mkdir(parents=True, exist_ok=True)
            dst = out_dir / f"{p.stem}_mask.jpg"
            import cv2

            cv2.imwrite(str(dst), rendered)
            masked_paths.append(dst)
        source = "synth_lfw"

    print(f"Source: {source} -- {len(masked_paths)} masked, {len(unmasked_paths)} unmasked")

    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []
    for path, label in tqdm(
        [(p, 1) for p in masked_paths] + [(p, 0) for p in unmasked_paths],
        desc="classify",
    ):
        img = read_image(path)
        if img is None:
            continue
        faces = embedder.detector.detect(img, max_faces=1)
        face = faces[0] if faces else None
        pred = classifier.predict(img, face)
        y_true.append(label)
        y_pred.append(int(pred.is_masked))
        y_prob.append(pred.p_mask)

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_prob_arr = np.asarray(y_prob)
    cm = confusion_matrix(y_true_arr, y_pred_arr)
    accuracy = float((y_true_arr == y_pred_arr).mean())
    auc = float(roc_auc_score(y_true_arr, y_prob_arr)) if len(set(y_true_arr.tolist())) == 2 else float("nan")
    report_text = classification_report(y_true_arr, y_pred_arr, target_names=["unmasked", "masked"])
    print(report_text)
    print(f"Confusion matrix:\n{cm}\nAccuracy: {accuracy:.3f}  AUC: {auc:.3f}")

    # ROC plot
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    if not np.isnan(auc):
        fpr, tpr, _ = roc_curve(y_true_arr, y_prob_arr)
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Mask classifier ROC ({classifier.backend_name})")
    ax.legend()
    out_fig = FIGURES_DIR / "exp4" / "roc.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=140)
    plt.close(fig)

    write_summary("exp4_mask_detector", {
        "backend": classifier.backend_name,
        "source": source,
        "accuracy": accuracy,
        "auc": auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_text,
        "args": vars(args),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
