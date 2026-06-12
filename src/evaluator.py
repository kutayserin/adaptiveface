"""Biometric evaluation: FAR, FRR, ROC, DET, CMC, EER, sigmoid fits.

The metrics here mirror the protocols used by the past course projects we're
modelling -- Smart Peephole's HD-MCS / LFW evaluation produced exactly these
plots, and Biotouch added ROC + CMC. We compute everything from a raw
similarity matrix so the same code serves verification (1:1) and
identification (1:N).

Key inputs:
  - `similarities`: (N_probe, N_gallery) cosine similarity matrix.
  - `probe_labels`: length-N_probe list of true identities.
  - `gallery_labels`: length-N_gallery list of enrolled identities (same
     order as similarity matrix columns).

For identification we also accept an `is_genuine` parameter -- whether each
probe is actually an enrolled user (True) or an impostor whose identity is
not in the gallery (False). Open-set evaluation hinges on this distinction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Score collection
# ---------------------------------------------------------------------------


@dataclass
class ScoreSet:
    """Collected similarity scores split into genuine vs impostor.

    `genuine` are scores of "same person" comparisons; `impostor` are scores
    of "different person" comparisons. Every metric below derives from
    these two arrays.
    """

    genuine: np.ndarray
    impostor: np.ndarray
    rank_correct: np.ndarray = field(default_factory=lambda: np.array([]))
    """For each genuine probe, the 1-indexed rank at which the true identity
    appeared in the gallery sorted by similarity. Used for CMC."""

    n_gallery: int = 0


def collect_scores(
    similarities: np.ndarray,
    probe_labels: Sequence[str],
    gallery_labels: Sequence[str],
    probe_is_genuine: Sequence[bool] | None = None,
) -> ScoreSet:
    """Slice the (N_probe, N_gallery) similarity matrix into genuine/impostor.

    For each probe:
      * the column corresponding to its true identity contributes a
        "genuine" score (only if `probe_is_genuine` is True at that index,
        which it is by default).
      * the remaining columns contribute "impostor" scores.

    Impostor probes (open-set "unknown" intruders) contribute only impostor
    scores -- every column is by definition a different identity.
    """
    if probe_is_genuine is None:
        probe_is_genuine = [True] * len(probe_labels)

    gallery_index = {label: i for i, label in enumerate(gallery_labels)}
    genuine_scores: list[float] = []
    impostor_scores: list[float] = []
    ranks: list[int] = []

    n_probes, n_gallery = similarities.shape
    if n_probes != len(probe_labels):
        raise ValueError("probe_labels length mismatches similarity matrix rows")
    if n_gallery != len(gallery_labels):
        raise ValueError("gallery_labels length mismatches similarity matrix cols")

    for i, (label, genuine) in enumerate(zip(probe_labels, probe_is_genuine)):
        row = similarities[i]
        sorted_idx = np.argsort(-row)
        if genuine and label in gallery_index:
            target_col = gallery_index[label]
            genuine_scores.append(float(row[target_col]))
            rank = int(np.where(sorted_idx == target_col)[0][0]) + 1
            ranks.append(rank)
            # All other columns are impostors for this probe.
            for j in range(n_gallery):
                if j != target_col:
                    impostor_scores.append(float(row[j]))
        else:
            # Impostor probe (truly unknown identity); every column is an
            # impostor comparison and there is no rank to record.
            for j in range(n_gallery):
                impostor_scores.append(float(row[j]))

    return ScoreSet(
        genuine=np.asarray(genuine_scores, dtype=np.float64),
        impostor=np.asarray(impostor_scores, dtype=np.float64),
        rank_correct=np.asarray(ranks, dtype=np.int64),
        n_gallery=n_gallery,
    )


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------


def far_frr_curve(scores: ScoreSet, thresholds: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Sweep thresholds and report FAR / FRR.

    FAR = fraction of impostor scores >= threshold (false accept).
    FRR = fraction of genuine scores < threshold (false reject).
    """
    if thresholds is None:
        # Choose a fine grid spanning both distributions.
        lo = min(scores.genuine.min(), scores.impostor.min()) if scores.genuine.size and scores.impostor.size else 0.0
        hi = max(scores.genuine.max(), scores.impostor.max()) if scores.genuine.size and scores.impostor.size else 1.0
        thresholds = np.linspace(lo, hi, 1001)

    far = np.array([(scores.impostor >= t).mean() if scores.impostor.size else 0.0 for t in thresholds])
    frr = np.array([(scores.genuine < t).mean() if scores.genuine.size else 0.0 for t in thresholds])
    return {"thresholds": thresholds, "far": far, "frr": frr}


def equal_error_rate(curve: dict[str, np.ndarray]) -> tuple[float, float]:
    """Return (EER, threshold-at-EER)."""
    far = curve["far"]
    frr = curve["frr"]
    diffs = np.abs(far - frr)
    idx = int(np.argmin(diffs))
    eer = float((far[idx] + frr[idx]) / 2.0)
    return eer, float(curve["thresholds"][idx])


def roc_curve(scores: ScoreSet, n_points: int = 1001) -> dict[str, np.ndarray]:
    """Standard ROC: True Accept Rate vs False Accept Rate."""
    if scores.genuine.size == 0 or scores.impostor.size == 0:
        return {"far": np.array([0.0, 1.0]), "tar": np.array([0.0, 1.0]), "auc": 0.5}
    lo = min(scores.genuine.min(), scores.impostor.min())
    hi = max(scores.genuine.max(), scores.impostor.max())
    thresholds = np.linspace(hi, lo, n_points)
    tar = np.array([(scores.genuine >= t).mean() for t in thresholds])
    far = np.array([(scores.impostor >= t).mean() for t in thresholds])
    # `np.trapezoid` is the post-NumPy-2.0 spelling; fall back to `np.trapz`
    # so we work on the 1.x line our requirements pin.
    integ = getattr(np, "trapezoid", None) or np.trapz
    # AUC is sorted-x-monotone: FAR axis is decreasing in our threshold sweep.
    order = np.argsort(far)
    auc = float(integ(tar[order], far[order]))
    return {"far": far, "tar": tar, "thresholds": thresholds, "auc": auc}


def cmc_curve(scores: ScoreSet, max_rank: int | None = None) -> dict[str, np.ndarray]:
    """Cumulative Match Characteristic for closed-set identification."""
    if scores.rank_correct.size == 0:
        return {"rank": np.array([1]), "rate": np.array([0.0])}
    max_rank = max_rank or int(scores.n_gallery)
    ranks = np.arange(1, max_rank + 1)
    rate = np.array([(scores.rank_correct <= r).mean() for r in ranks])
    return {"rank": ranks, "rate": rate}


# ---------------------------------------------------------------------------
# Sigmoid fit (mirrors Smart Peephole's plots)
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    return a / (1.0 + np.exp(-c * (x - b))) + d


def fit_sigmoid(thresholds: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a generic sigmoid to a (threshold, value) curve.

    Returns (predicted_values_on_input_grid, fitted_params). Falls back to
    the original values if curve_fit fails to converge.
    """
    if values.size == 0:
        return values, np.array([])
    try:
        # Initialise from the data so curve_fit doesn't wander off.
        p0 = (1.0, float(np.median(thresholds)), 10.0, 0.0)
        popt, _ = curve_fit(_sigmoid, thresholds, values, p0=p0, maxfev=10000)
        return _sigmoid(thresholds, *popt), popt
    except Exception:
        return values, np.array([])


# ---------------------------------------------------------------------------
# Convenience: full report from a single similarity matrix
# ---------------------------------------------------------------------------


@dataclass
class EvaluationReport:
    scores: ScoreSet
    far_frr: dict[str, np.ndarray]
    roc: dict[str, np.ndarray]
    cmc: dict[str, np.ndarray]
    eer: float
    eer_threshold: float
    rank1: float

    def summary(self) -> dict[str, float]:
        return {
            "n_genuine": int(self.scores.genuine.size),
            "n_impostor": int(self.scores.impostor.size),
            "eer": self.eer,
            "eer_threshold": self.eer_threshold,
            "auc": float(self.roc["auc"]),
            "rank1": self.rank1,
        }


def evaluate(
    similarities: np.ndarray,
    probe_labels: Sequence[str],
    gallery_labels: Sequence[str],
    probe_is_genuine: Sequence[bool] | None = None,
) -> EvaluationReport:
    scores = collect_scores(similarities, probe_labels, gallery_labels, probe_is_genuine)
    far_frr = far_frr_curve(scores)
    eer, eer_t = equal_error_rate(far_frr)
    roc = roc_curve(scores)
    cmc = cmc_curve(scores)
    rank1 = float(cmc["rate"][0]) if cmc["rate"].size else 0.0
    return EvaluationReport(
        scores=scores,
        far_frr=far_frr,
        roc=roc,
        cmc=cmc,
        eer=eer,
        eer_threshold=eer_t,
        rank1=rank1,
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_far_frr(report: EvaluationReport, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    t = report.far_frr["thresholds"]
    ax.plot(t, report.far_frr["far"], label="FAR", color="tab:blue")
    ax.plot(t, report.far_frr["frr"], label="FRR", color="tab:red")
    # Overlay sigmoid fits so the plot looks like the one Smart Peephole used.
    far_fit, _ = fit_sigmoid(t, report.far_frr["far"])
    frr_fit, _ = fit_sigmoid(t, report.far_frr["frr"])
    ax.plot(t, far_fit, ":", color="tab:blue", alpha=0.6, label="FAR sigmoid fit")
    ax.plot(t, frr_fit, ":", color="tab:red", alpha=0.6, label="FRR sigmoid fit")
    ax.axvline(report.eer_threshold, color="grey", linestyle="--", alpha=0.5,
               label=f"EER = {report.eer:.3f} @ t={report.eer_threshold:.3f}")
    ax.set_xlabel("threshold")
    ax.set_ylabel("error rate")
    ax.set_title(f"FAR vs FRR — {title}")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_roc(report: EvaluationReport, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(report.roc["far"], report.roc["tar"], label=f"AUC = {report.roc['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.5)
    ax.set_xlabel("False Accept Rate")
    ax.set_ylabel("True Accept Rate")
    ax.set_title(f"ROC — {title}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_cmc(report: EvaluationReport, title: str, out_path: Path, max_rank: int = 20) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    rank = report.cmc["rank"]
    rate = report.cmc["rate"]
    cutoff = min(max_rank, rank.size)
    ax.plot(rank[:cutoff], rate[:cutoff], marker="o")
    ax.set_xlabel("rank")
    ax.set_ylabel("identification rate")
    ax.set_ylim(0, 1.01)
    ax.set_title(f"CMC — {title}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_score_distributions(report: EvaluationReport, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(
        min(report.scores.genuine.min(initial=0.0), report.scores.impostor.min(initial=0.0)),
        max(report.scores.genuine.max(initial=1.0), report.scores.impostor.max(initial=1.0)),
        60,
    )
    ax.hist(report.scores.impostor, bins=bins, alpha=0.55, color="tab:red", label="impostor", density=True)
    ax.hist(report.scores.genuine, bins=bins, alpha=0.55, color="tab:green", label="genuine", density=True)
    ax.axvline(report.eer_threshold, color="black", linestyle="--", alpha=0.7,
               label=f"EER threshold = {report.eer_threshold:.3f}")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.set_title(f"Score distributions — {title}")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


__all__ = [
    "ScoreSet",
    "EvaluationReport",
    "collect_scores",
    "far_frr_curve",
    "equal_error_rate",
    "roc_curve",
    "cmc_curve",
    "fit_sigmoid",
    "evaluate",
    "plot_far_frr",
    "plot_roc",
    "plot_cmc",
    "plot_score_distributions",
]
