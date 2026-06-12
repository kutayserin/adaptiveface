"""Produce the comparison plots used on the headline slides.

We generate three single-figure summaries:
  * ``exp3_rank1_bar.png`` -- overall rank-1 across all 8 configurations.
  * ``exp3_masked_bar.png`` -- masked-only rank-1 (the hard subset).
  * ``exp3_eer_bar.png``   -- EER across configurations.

Inputs:
  * results/exp3_cross_mode_summary.json  -- aggregate metrics.
  * results/exp3_per_probe.csv            -- masked-only breakdown.
Both are written by experiments/exp3_cross_mode.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FIGURES_DIR, RESULTS_DIR  # noqa: E402


# Order to display the configurations on every bar chart. We start with
# the baseline + routing configurations, then fusion, then the winning
# multi-template ones at the end so the eye lands on them last.
ORDER = [
    "naive",
    "oracle",
    "adaptive",
    "fusion_max",
    "fusion_avg",
    "fusion_weighted",
    "multi_max",
    "multi_adaptive",
]


def _bar(ax, labels, values, title, ylabel, highlight=("multi_max", "multi_adaptive")):
    colors = ["#1f77b4" if l not in highlight else "#2ca02c" for l in labels]
    bars = ax.bar(labels, values, color=colors)
    # Naive baseline line so the eye can compare every bar against it.
    if "naive" in labels:
        naive_val = values[labels.index("naive")]
        ax.axhline(naive_val, color="#444", linestyle="--", linewidth=1, alpha=0.6,
                   label=f"naive = {naive_val:.3f}")
        ax.legend(loc="upper left")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)


def main() -> int:
    summary = json.loads((RESULTS_DIR / "exp3_cross_mode_summary.json").read_text())
    per_probe = pd.read_csv(RESULTS_DIR / "exp3_per_probe.csv")

    labels = [c for c in ORDER if c in summary["results"]]
    rank1 = [summary["results"][c]["rank1"] for c in labels]
    eer = [summary["results"][c]["eer"] for c in labels]

    masked_rank1 = []
    for c in labels:
        sub = per_probe[(per_probe["config"] == c)
                         & (per_probe["is_genuine"])
                         & (per_probe["is_masked_truth"])]
        if len(sub) == 0:
            masked_rank1.append(0.0)
            continue
        correct = (sub["identity"] == sub["top_match"]).sum()
        masked_rank1.append(correct / len(sub))

    out_dir = FIGURES_DIR / "exp3"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Rank-1 bar chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _bar(ax, labels, rank1, "Overall rank-1 identification rate (RMFD + LFW impostors)", "rank-1")
    fig.tight_layout()
    fig.savefig(out_dir / "exp3_rank1_bar.png", dpi=160)
    plt.close(fig)

    # Masked-only rank-1
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _bar(ax, labels, masked_rank1, "Rank-1 on masked probes only (the hard case)", "rank-1 (masked-only)")
    fig.tight_layout()
    fig.savefig(out_dir / "exp3_masked_bar.png", dpi=160)
    plt.close(fig)

    # EER (lower is better, so flip colour highlighting logic)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _bar(ax, labels, eer, "Equal Error Rate by configuration (lower is better)", "EER")
    fig.tight_layout()
    fig.savefig(out_dir / "exp3_eer_bar.png", dpi=160)
    plt.close(fig)

    print(f"wrote {out_dir / 'exp3_rank1_bar.png'}")
    print(f"wrote {out_dir / 'exp3_masked_bar.png'}")
    print(f"wrote {out_dir / 'exp3_eer_bar.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
