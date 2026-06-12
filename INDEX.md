# AdaptiveFace — Project Bundle

**Course:** Biometric Systems, MSc Computer Science
**Authors:** Taha, Kutay
**Date:** May 2026

---

## TL;DR for the grader

> Mask-aware face authentication. Pre-trained ArcFace + RetinaFace +
> AIZOO mask classifier, evaluated with two parallel mitigation
> strategies (routing-based dual-template **vs.** multi-template
> enrollment with synthetic masks). Headline finding:
> routing-based mitigation **underperforms** the naive baseline on
> real masked probes; multi-template enrollment **beats** the
> baseline on every metric.

Start with **`report/REPORT.md`** for the full write-up, then
**`report/PRESENTATION.md`** for the slide outline and Q&A prep.

---

## How to navigate this bundle

| If you want to... | Open this |
|---|---|
| Read the full report | `report/REPORT.md` |
| See the presentation plan + Q&A | `report/PRESENTATION.md` |
| See the headline numbers per experiment | `results/exp*_summary.json` |
| Inspect failure cases probe-by-probe | `results/exp3_per_probe.csv` |
| Look at the FAR/FRR/ROC/CMC plots | `report/figures/exp{1,2,3,4}/` |
| Understand the pipeline architecture | `README.md` + `src/` |
| Run the experiments yourself | `README.md` § "Running the experiments" |
| Run the live webcam demo | `README.md` § "Demo" |

---

## What's inside

```
adaptiveface/
├── INDEX.md                  <- you are here
├── README.md                 <- setup + run instructions
├── requirements.txt
├── .gitignore
│
├── report/                   <- the deliverables
│   ├── REPORT.md             <- written report (Sections 1-8 + refs)
│   ├── PRESENTATION.md       <- slide outline + 10 Q&A answers
│   └── figures/              <- generated plots
│       ├── exp1/             <- LFW unmasked baseline plots
│       ├── exp2/             <- masked recognition plots
│       ├── exp3/             <- cross-mode (the headline) plots
│       └── exp4/             <- mask classifier plots
│
├── results/                  <- raw experiment outputs
│   ├── exp1_unmasked_summary.json
│   ├── exp2_masked_summary.json
│   ├── exp3_cross_mode_summary.json     <- the headline numbers
│   ├── exp3_per_probe.csv               <- per-probe failure analysis
│   ├── exp3_v1_*                        <- pre-multi-template backup
│   ├── exp4_mask_detector_summary.json
│   └── cache/                <- cached embeddings (re-run in seconds)
│
├── src/                      <- core pipeline modules
│   ├── config.py
│   ├── datasets.py           <- LFW / RMFD / MFR2 loaders
│   ├── face_detector.py      <- RetinaFace wrapper
│   ├── mask_classifier.py    <- AIZOO Caffe + heuristic fallback
│   ├── embedder.py           <- ArcFace, full + landmark-aligned upper
│   ├── gallery.py            <- dual-template enrollment + persistence
│   ├── matcher.py            <- cosine 1:N matcher
│   ├── pipeline.py           <- end-to-end orchestrator
│   ├── evaluator.py          <- FAR/FRR/ROC/CMC/EER + sigmoid fits
│   └── synth_mask.py         <- 5pt-landmark synthetic mask overlay
│
├── experiments/              <- the four numbered experiments
│   ├── _common.py            <- embedding caching utilities
│   ├── exp1_unmasked_baseline.py
│   ├── exp2_masked_recognition.py
│   ├── exp3_cross_mode.py    <- 8 configurations including multi-template
│   └── exp4_mask_detector.py
│
├── demo/                     <- live presentation demo
│   ├── enroll.py             <- enrol identities (photo or webcam)
│   └── webcam_demo.py        <- live recognition with overlay
│
└── scripts/                  <- one-time setup utilities
    ├── download_datasets.py
    └── download_models.py
```

## What's NOT in this bundle (and why)

* **`data/`** -- LFW (~180 MB), RMFD (~450 MB), MFR2 (~10 MB). Public
  datasets. See `README.md` for download instructions.
* **`models/`** -- AIZOO mask classifier weights (~4 MB). Auto-fetched
  by `python -m scripts.download_models`.
* **`~/.insightface/models/buffalo_l/`** -- the InsightFace ArcFace +
  RetinaFace bundle (~250 MB). Auto-fetched on first instantiation of
  `FaceDetector`.

These exclusions keep the zip under 50 MB. To reproduce end-to-end
from a fresh machine:

```powershell
# 1. unzip and cd
cd adaptiveface

# 2. install
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 3. datasets
python -m scripts.download_datasets --lfw --mfr2
# RMFD: see README for the Kaggle mirror

# 4. mask classifier weights
python -m scripts.download_models

# 5. run experiments (embeddings re-use the cached versions if present)
python -m experiments.exp1_unmasked_baseline
python -m experiments.exp2_masked_recognition --source rmfd
python -m experiments.exp3_cross_mode --source rmfd
python -m experiments.exp4_mask_detector

# 6. live demo
python -m demo.enroll --identity Taha --webcam
python -m demo.enroll --identity Kutay --webcam
python -m demo.webcam_demo --gallery demo_gallery
```

## Headline results (read this if you only read one thing)

### Experiment 1 -- Unmasked baseline (LFW)
| Metric | Value |
|---|---|
| Rank-1 | 92.7 % |
| EER | 5.3 % |
| AUC | 0.985 |

### Experiment 3 -- Cross-mode end-to-end (RMFD + LFW impostors)

| Configuration | Rank-1 | EER | Masked-only Rank-1 |
|---|---|---|---|
| naive (always full-face) | 76.8 % | 9.8 % | 51.9 % |
| oracle routing | 68.5 % | 9.8 % | 35.3 % |
| adaptive routing | 69.7 % | 9.6 % | 37.8 % |
| score-level fusion (best of 3) | 70.5 % | 10.0 % | 45.2 % |
| **multi_max (multi-template)** | **77.4 %** | **9.1 %** | **52.7 %** |
| **multi_adaptive** | 76.1 % | **8.7 %** | 50.6 % |

Mask classifier accuracy on the probe set: **95.9 %** (so the routing
failures aren't a classifier issue).

### Experiment 4 -- Mask classifier
Accuracy 94 %, AUC 0.991 on 300 masked + 300 unmasked RMFD samples.

---

## Final story for the defence

We tried two routes:

1. **Routing-based mitigation** -- swap the matching key (full vs.
   upper-face embedding) based on a mask classifier's call. **Failed.**
   Net loss of 48 probes vs naive on the masked-only subset: ArcFace
   already extracts useful identity signal from the visible upper
   face, and our synthetic forehead fill erased that signal.
2. **Multi-template enrollment** -- give each user two gallery rows
   (their real unmasked photo + a synthetic-masked rendering of the
   same photo), score against both, take the max. **Won.** Small
   (+0.6 pts rank-1, -0.7 pts EER) but consistent across every
   metric and on the masked-only subset.

Negative-result-plus-fix-that-works is the scientific output here.
