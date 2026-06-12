# AdaptiveFace

Mask-aware face authentication. A user enrols once with an unmasked photo;
at recognition time the system detects whether the face is masked and
routes to one of two embedding branches against the same dual-template
gallery, so the same person is accepted whether or not they are wearing a
mask.

This is the implementation for our MSc Biometric Systems project. The
pipeline reuses pre-trained models for face detection (RetinaFace), face
recognition (ArcFace via InsightFace), and mask detection (AIZOO ONNX with
a landmark-based fallback). Our contribution is the dual-template
enrollment, mask-aware routing, and the evaluation across three operating
modes on two datasets.

## Pipeline

```
                webcam frame / image
                       v
              RetinaFace face detection
                       v
              AIZOO mask classifier
                /          \
        unmasked            masked
            v                  v
   ArcFace (full face)   ArcFace (upper-face only)
            v                  v
    cosine vs FULL TEMPLATE     cosine vs UPPER TEMPLATE
            \                   /
                top-1 + threshold
                       v
                Accept / Reject
```

The "upper-face only" branch reuses the same ArcFace network but feeds it
an image where the lower portion of the bounding box is replaced with the
forehead's mean colour. This keeps RetinaFace alignment valid (the network
expects eyes/nose/mouth at fixed locations after the 112x112 warp) while
removing the mask-corrupted signal so the network keys on the visible
upper-face information only -- no fine-tuning required.

## Setup

```bash
# 1. Python environment (Python >= 3.10)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Datasets
python -m scripts.download_datasets --lfw --mfr2

# 3. Pre-trained weights for the mask classifier
python -m scripts.download_models
```

InsightFace downloads its own RetinaFace + ArcFace bundle the first time
you instantiate `FaceDetector`. That download is ~250 MB and goes into
`~/.insightface/`.

### RMFD (manual)

RMFD does not have a stable open mirror. To use it:

1. Visit <https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset> and
   follow their instructions (one of the most reliable mirrors at the time
   of writing is on Baidu Pan; access varies).
2. Extract `AFDB_masked_face_dataset/` and `AFDB_face_dataset/` into
   `adaptiveface/data/rmfd/` so the tree looks like:
   ```
   data/rmfd/
     AFDB_masked_face_dataset/
       <identity_1>/*.jpg
       ...
     AFDB_face_dataset/
       <identity_1>/*.jpg
       ...
   ```
3. Verify with `python -m scripts.download_datasets --rmfd`.

If RMFD is unavailable, every experiment that needs masked images can fall
back to **synthetic masks** overlaid on LFW. The fallback is automatic when
`--source auto` is used. Synthetic masks are good enough for a working
demo but the masked numbers are slightly optimistic versus real RMFD masks,
which the report discusses.

## Running the experiments

All experiments cache embeddings to `results/cache/` so re-runs (e.g.
when sweeping thresholds) are fast.

```bash
# 1. Unmasked baseline -- establishes that the ArcFace stack works.
python -m experiments.exp1_unmasked_baseline

# 2. Masked recognition with naive vs upper-face routing.
python -m experiments.exp2_masked_recognition --source auto

# 3. The headline experiment: cross-mode same-identity, comparing
#    naive / oracle / adaptive routing strategies.
python -m experiments.exp3_cross_mode --source auto

# 4. Mask classifier sanity check on the routing decision.
python -m experiments.exp4_mask_detector --use-synth-fallback
```

Each experiment writes:

* `results/<expN>_summary.json` -- numbers for the report.
* `report/figures/<expN>/*.png` -- FAR/FRR, ROC, CMC, score distributions.
* `results/cache/*` -- cached embeddings (safe to delete to force a re-run).

## Demo

The presentation demo enrols any number of people (each `enroll` call
adds one identity to the gallery) and recognises them with / without
masks in real time. Two identities below only because we are a
two-person team -- the matcher stacks all templates into one matrix,
and the same code path handled 150-identity galleries in Experiment 3.

```bash
# Enrol identities. Press SPACE to capture the photo when you look good.
# Repeat for as many people as you want in the gallery.
python -m demo.enroll --identity Taha --webcam
python -m demo.enroll --identity Kutay --webcam

# Run the live demo (multi_max pipeline by default).
# ESC or q to quit, `s` for a screenshot, `r` to record an MP4.
python -m demo.webcam_demo --gallery demo_gallery
```

Each enrolment stores two templates derived from the same photo: the
real unmasked embedding and a synthetic-masked one. The demo scores
every frame against both and takes the max (the `multi_max`
configuration that won Experiment 3), so a masked probe matches the
synth-masked row with no mask-classifier routing involved. The banner
shows the accept/reject decision, top-1 identity, cosine similarity,
and the mask classifier's call (informational only). Pass
`--pipeline routing` to see the legacy Pipeline-A behaviour instead.

See `demo/DEMO_SCRIPT.md` for the 30-second recording choreography
used for the submitted demo video.

## Project layout

```
adaptiveface/
  src/
    config.py           central paths + thresholds
    datasets.py         LFW / RMFD / MFR2 loaders and split helpers
    face_detector.py    RetinaFace wrapper (via insightface)
    mask_classifier.py  AIZOO ONNX + landmark heuristic fallback
    embedder.py         ArcFace embeddings, full + upper modes
    gallery.py          dual-template enrollment + persistence
    matcher.py          cosine 1:N matcher
    pipeline.py         end-to-end orchestrator
    evaluator.py        FAR/FRR/ROC/CMC/EER, sigmoid fits, plots
    synth_mask.py       synthetic mask overlay using 5pt landmarks
  experiments/
    _common.py          embedding caching utilities
    exp1_unmasked_baseline.py
    exp2_masked_recognition.py
    exp3_cross_mode.py
    exp4_mask_detector.py
  demo/
    enroll.py           build a gallery from photos / webcam
    webcam_demo.py      live recognition with on-screen overlay
  scripts/
    download_datasets.py
    download_models.py
  data/                 (gitignored) datasets land here
  models/               (gitignored) ONNX weights
  results/              experiment outputs + caches
  report/figures/       plots written by the experiments
```

## Tuning knobs

Edit `src/config.py` to change defaults:

* `PipelineConfig.cosine_threshold_unmasked / _masked` -- thresholds. Pick
  these from the EER point reported by Experiments 1 and 2.
* `PipelineConfig.upper_face_crop_ratio` -- how much of the face bbox to
  keep before feeding ArcFace in upper-face mode (0.55 by default).
* `EmbedderConfig.ctx_id` -- set to `-1` to force CPU; `0` uses the first
  CUDA device.

## Troubleshooting

* **`onnxruntime` complains about CUDA**: install `onnxruntime` (CPU) and
  set `ctx_id=-1`, or install `onnxruntime-gpu` matching your CUDA version.
* **`insightface` model download stalls**: it caches under `~/.insightface/`.
  Delete that directory and rerun to retry.
* **Mask classifier falls back to "heuristic"**: the ONNX weights download
  failed. The heuristic still works but the masked routing accuracy in
  Experiment 4 will be lower; rerun `scripts/download_models.py`.
* **`MemoryError` on LFW**: drop `--n-identities` to 100 in Experiment 1.

## Credits / models used

* RetinaFace + ArcFace: <https://github.com/deepinsight/insightface>
* AIZOO FaceMaskDetection: <https://github.com/AIZOOTech/FaceMaskDetection>
* LFW: Huang et al., Univ. of Massachusetts Amherst.
* RMFD: <https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset>
* MFR2: <https://github.com/aqeelanwar/MaskTheFace>
