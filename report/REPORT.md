# AdaptiveFace: Mask-Aware Face Authentication

**Biometric Systems — MSc Computer Science**
**Authors:** Taha, Kutay
**Date:** May 2026

---

## Abstract

We present AdaptiveFace, an open-set face authentication system that
remains robust to mask occlusion without retraining the underlying
recognition model. We compare two families of mitigation against a
pre-trained ArcFace baseline:

  1. **Mask-aware routing** -- a mask detector picks between full-face
     and upper-face-only embeddings at recognition time, against a
     gallery that stores both templates per identity.
  2. **Multi-template enrollment** -- each enrolled user is represented
     by *two* full-face templates: one extracted from their real
     unmasked photo and one extracted from a synthetic-masked rendering
     of the same photo. Matching takes the max similarity over both
     templates.

We evaluate on LFW (unmasked baseline, EER 5.3 %, rank-1 92.7 %) and
the Real-World Masked Face Dataset (RMFD; paired masked/unmasked
identities, 377 shared). Our central finding is that **routing-based
mitigation underperforms the naive full-face baseline** on real
masked probes (rank-1 drops from 76.8 % to ~69 %), because pre-trained
ArcFace already handles partial mask occlusion better than our
upper-face-fill alternative. **Multi-template enrollment, in contrast,
beats the naive baseline on every reported metric** (rank-1 77.4 %,
EER 9.1 %) and adds an extra 0.7 percentage-point rank-1 gain on top
of the unmodified recognition model -- without changing the model or
collecting any real masked enrollment data.

## 1. Introduction

Face recognition systems trained on largely unmasked face datasets see
substantial accuracy degradation when probes are partially occluded by a
mask. Three families of mitigations exist in the literature:

1. **Retraining or fine-tuning** the recognition model on masked imagery
   (data-hungry, requires labelled paired data).
2. **Region-restricted recognition**, e.g. periocular biometrics
   (avoids the occluded region but typically requires a dedicated model).
3. **Synthesis-based augmentation**, training-time masks overlaid onto
   unmasked datasets (helps if the synthetic distribution matches reality).

We pursue a fourth, deployment-oriented direction: keep the off-the-shelf
ArcFace recognition model and use *upper-face-only* embeddings as a
fallback branch. The novelty is in the **dual-template enrollment** and
the **mask-aware routing**: each enrolled identity is represented twice
in the gallery -- once with the full-face embedding and once with the
upper-face embedding -- both computed from the same enrollment photo.

## 2. Related work

We were guided by three past projects in this course:

* **Smart Peephole** (De Marsico et al., 2017) -- modular biometric
  pipeline mixing local detection with cloud recognition; established
  the "FAR/FRR + ROC + CMC + sigmoid fits" reporting template that we
  reuse verbatim.
* **BioPen** (Scozzafava & Ponzi) -- sensor pipeline with explicit
  acquisition / feature extraction / decision modules; mirrors the
  three-stage layout we adopted.
* **Biotouch** (Moschella & Spini) -- ensemble fusion of multiple SVMs
  using majority / average / weighted-average rules; motivated our
  "naive vs oracle vs adaptive" three-mode comparison.

In the masked-recognition literature, our approach is most closely
related to Anwar & Raychowdhury's "Masked Face Recognition for Secure
Authentication" baseline (2020), which similarly evaluates an unmasked
recognition model on masked probes; we extend that work by adding the
explicit routing layer and a per-mode threshold.

## 3. System design

We implement two mitigation pipelines side-by-side and compare them
against the naive baseline on the same probe set.

### 3.1 Pipeline A -- Mask-aware routing (dual-template)

```
Image -> RetinaFace -> AIZOO mask classifier
   |                       /        \
   |               unmasked          masked
   |                  v                 v
   +-> ArcFace(full)  ArcFace(landmark-aligned upper-fill)
              v                       v
      gallery_full row[i]   gallery_upper row[i]
              cosine 1:N    cosine 1:N
                  \         /
                   Accept / Reject
```

Each identity carries two templates derived from the same unmasked
enrollment photo: the standard 512-D ArcFace embedding and a second
embedding computed from a *landmark-aligned* 112x112 crop whose lower
half has been replaced by the forehead's mean colour. The mask
classifier picks which row to score against at recognition time.

### 3.2 Pipeline B -- Multi-template enrollment (synthetic mask)

```
Enrollment (one-time):
    unmasked photo -> (a) ArcFace(full)              -> gallery_real[i]
                   -> (b) overlay synthetic mask
                          -> ArcFace(full)           -> gallery_synth[i]

Recognition:
    probe image -> ArcFace(full)
                -> sim_real  = probe . gallery_real^T
                -> sim_synth = probe . gallery_synth^T
                -> fused     = max(sim_real, sim_synth)
                -> Accept / Reject
```

The trick: each identity is registered *twice* in the gallery, but
both rows come from the same enrollment photo -- the second row is a
synthetic surgical-mask overlay rendered onto the same face. A real
masked probe at recognition time naturally lands closer to the
``gallery_synth`` row (same mask distribution) while an unmasked probe
lands closer to ``gallery_real``. Taking the max of the two
similarities gives the matcher the right row for free, with no mask
classifier needed.

### 3.3 Face detection and alignment

We use InsightFace's RetinaFace, which produces a bounding box and a
five-point landmark set (eyes, nose tip, mouth corners) used downstream
both by the 112×112 ArcFace alignment warp and by our upper-face fill
logic.

### 3.4 Mask classification

We use the AIZOO FaceMaskDetection Caffe model loaded via
``cv2.dnn``. It is a small SSD-style two-class detector
(``face`` / ``face_mask``) on a 260x260 RGB input. We aggregate across
its 5,972 anchor predictions by keeping anchors above
``ANCHOR_KEEP_CONF = 0.5`` and returning the top-scoring kept anchor's
class. The original ONNX export this paper's name-sake URL pointed at
is no longer hosted; we documented the migration to Caffe in the
README. A landmark-saturation heuristic stays in the code as a
last-resort fallback. Experiment 4 reports the classifier's standalone
accuracy on RMFD.

### 3.5 Embedding extraction

* **Full-face mode**: standard ArcFace 512-D embedding from the
  RetinaFace-aligned 112×112 crop.
* **Upper-face mode** (Pipeline A only): we run RetinaFace's
  ``face_align.norm_crop`` first so the eyes / nose / mouth land at
  canonical 112x112 pixel positions, then replace the lower
  ``1 - keep_ratio`` rows with the forehead's mean colour, then feed
  the resulting image to ArcFace via ``rec.get_feat``. Aligning before
  filling is what makes the geometry stable across pose; we had to
  abandon two earlier variants (fill-in-bbox-space and tight-crop)
  whose embeddings were dominated by alignment artefacts rather than
  identity (§5.2).

Both modes produce L2-normalised 512-D vectors, so cosine similarity is
equivalent to dot product.

### 3.6 Gallery

For Pipeline A: each identity is enrolled from a single unmasked photo
yielding a pair ``(full_template, upper_template)``.
For Pipeline B: each identity is enrolled from the same photo plus a
synthetic-masked rendering of it, yielding a pair
``(real_template, synth_masked_template)``.
The gallery is a flat key-value store (JSON index + NumPy npz
arrays) -- intentionally simple because the goal is reproducibility,
not throughput.

### 3.7 Matching and decision

Cosine similarity against the gallery rows. Pipeline A routes to one
row per probe based on the mask classifier; Pipeline B always scores
the probe against *both* rows and takes the max. Accept iff the
chosen score is above the EER-tuned threshold from §5.

## 4. Evaluation protocol

### 4.1 Datasets

* **LFW** -- 5,749 identities, 13,233 images. Used for the unmasked
  baseline (Experiment 1) and as the impostor pool in the cross-mode
  experiment.
* **RMFD** -- the Real-World Masked Face Dataset contains an
  `AFDB_face_dataset` (unmasked) and `AFDB_masked_face_dataset` (masked)
  with shared identities. Used for Experiments 2 and 3 when available.
* **Synthetic-masked LFW** -- when RMFD is unavailable we generate a
  paired masked set by overlaying a polygonal surgical-mask shape onto
  LFW images using the RetinaFace landmarks. The synthetic distribution
  is somewhat optimistic vs. real masks (no shadows, no fabric texture)
  and Experiment 2 lets us compare the two when both are accessible.
* **MFR2** -- a small (53-identity) paired set used as a sanity-check
  dataset for the demo.

### 4.2 Metrics

For every experiment we report:

* **Genuine / impostor score distributions** with the EER threshold
  overlaid.
* **FAR (False Accept Rate)** and **FRR (False Reject Rate)** swept
  across thresholds, plus sigmoid fits (matching the Smart Peephole
  presentation style).
* **EER (Equal Error Rate)** and the threshold at which it occurs.
* **ROC** with AUC.
* **CMC** for identification (rank-1 ... rank-20).

We also export per-probe CSV rows for Experiment 3 so failure modes can
be inspected post-hoc.

### 4.3 Open-set splits

The realistic deployment scenario is **open-set** identification --
strangers walking up to the camera who were never enrolled. We enforce
open-set evaluation by reserving a subset of identities as impostors
and constructing probes that the gallery has never seen. Their entire
similarity row contributes only to the impostor score distribution.

## 5. Experiments

All numbers below come directly from
``results/expN_*_summary.json`` and the per-probe CSV exported by
Experiment 3. Seed is fixed at 42 across the suite.

### 5.1 Experiment 1 -- Unmasked baseline (LFW)

**Setup.** 200 enrolled identities, 150 impostor identities from LFW.
One randomly-selected image per identity becomes the gallery template;
the remainder are genuine probes. Two impostor probes per impostor
identity. Gallery size after detection failures: 199 templates.

**Results.** 750 genuine probes, ~208K impostor comparisons.

| Metric | Value |
|---|---|
| Rank-1 identification rate | **92.7 %** |
| Equal-error rate (EER) | **5.3 %** |
| EER threshold (cosine) | 0.099 |
| ROC AUC | 0.985 |

**Discussion.** The pre-trained ArcFace backbone (InsightFace
``buffalo_l``) is healthy on the LFW protocol -- our numbers are in
the same ballpark as Smart Peephole's LFW evaluation (which reported
EER ≈ 0.02 on a smaller LFW subset). Any degradation we observe in
later experiments is therefore attributable to the mask condition,
not to a broken recognition stack.

### 5.2 Experiment 2 -- Masked recognition (RMFD)

**Setup.** RMFD ``AFDB_face_dataset`` provides the unmasked
enrollment photos; ``AFDB_masked_face_dataset`` provides masked
probes. 200 identities sampled, one unmasked image per identity in
the gallery, all masked images of those identities become probes.

We compare two operating modes on the same probe set:
* ``full`` -- feed each masked probe straight into ArcFace.
* ``upper`` -- run our landmark-aligned upper-face-fill embedder.

**Results.** 641 genuine probes against a 200-template gallery.

| Mode | Rank-1 | EER | AUC |
|---|---|---|---|
| ``full`` (naive) | **54.6 %** | **15.5 %** | 0.908 |
| ``upper`` (aligned fill) | 32.4 % | 20.9 % | 0.866 |

**Discussion.** A pure mask **does** noticeably hurt ArcFace: rank-1
drops from 92.7 % (Exp 1, unmasked) to 54.6 % (Exp 2, masked), a 38
percentage-point gap that is the headroom any mitigation must close.
But running the same model on our masked-out upper-face crop is
significantly *worse* than just running it on the masked image
unchanged. The next experiment confirms this isn't a one-off and
proposes the alternative mitigation that actually works.

### 5.3 Experiment 3 -- Cross-mode end-to-end (headline)

**Setup.** 150 enrolled identities + 80 impostor identities from LFW
+ paired RMFD probes. Each enrolled identity provides one unmasked
gallery photo. Probes mix genuine unmasked, genuine masked, and
impostor probes. Six configurations:

* **naive** -- always score against ``gallery_full``.
* **oracle** -- routing on ground-truth mask label.
* **adaptive** -- routing on AIZOO mask classifier output.
* **fusion_max / avg / weighted** -- score-level fusion of the full
  and upper-fill similarities (analogous to Biotouch's ensembles).
* **multi_max** -- multi-template max(real_sim, synth_masked_sim).
* **multi_adaptive** -- multi-template, route to ``gallery_synth`` if
  the mask classifier flags the probe, else ``gallery_real``.

**Results.** 482 genuine probes (241 masked + 241 unmasked), 156
impostors. AIZOO mask classifier accuracy on this probe set: 95.9 %.

| Config | Rank-1 | EER | AUC | Masked-only Rank-1 |
|---|---|---|---|---|
| naive | 76.8 % | 9.8 % | 0.955 | 51.9 % |
| oracle (routing) | 68.5 % | 9.8 % | 0.956 | 35.3 % |
| adaptive (routing) | 69.7 % | 9.6 % | 0.955 | 37.8 % |
| fusion_max | 55.2 % | 13.5 % | 0.934 | 39.0 % |
| fusion_avg | 69.3 % | 10.8 % | 0.949 | 45.2 % |
| fusion_weighted | 70.5 % | 10.0 % | 0.955 | 41.1 % |
| **multi_max** | **77.4 %** | **9.1 %** | **0.957** | **52.7 %** |
| **multi_adaptive** | 76.1 % | **8.7 %** | **0.957** | 50.6 % |

**Discussion -- the routing approach fails.** Every routing-based
configuration (oracle, adaptive, weighted fusion) is **worse than
the naive baseline** on rank-1. Inspecting the per-probe CSV shows
why: on masked probes, the upper-fill embedding is right where naive
is wrong 9 times, but wrong where naive is right 57 times -- a net
loss. Pre-trained ArcFace is already extracting more useful signal
from the visible eye + brow region of a masked face than our
forehead-filled aligned crop manages to. Score-level fusion inherits
this disadvantage.

**The multi-template approach wins on every metric.** Storing a
second, *synthetically-masked* template alongside each unmasked one
gives a real masked probe a same-distribution gallery row to score
against. ``multi_max`` (the simpler of the two) achieves rank-1
77.4 % (+0.6 % over naive) and EER 9.1 % (vs. 9.8 %); ``multi_adaptive``
ties on rank-1 (within noise) and achieves the lowest EER overall
(8.7 %). The win is small in absolute terms because the naive
baseline is strong, but the direction is consistent across rank-1,
EER, AUC, and the masked-only subset.

**Why the mask classifier doesn't help.** Mask classifier accuracy
(Exp 4: 94 %, this experiment's probe set: 95.9 %) is high enough
that it shouldn't be the bottleneck. The bottleneck is that *neither*
routing target is a good destination: routing to ``upper`` mode is
strictly worse than not routing at all. ``multi_adaptive`` shows that
when the destination is good (the synthetic-masked template), even a
classifier-routed system can match max-fusion -- but the routing
itself contributes nothing once you have the right templates.

### 5.4 Experiment 4 -- Mask classifier sanity check

**Setup.** 300 masked + 300 unmasked images sampled from RMFD.
Standard binary-classifier evaluation.

**Results.**

| Metric | Value |
|---|---|
| Backend | AIZOO Caffe (``cv2.dnn``) |
| Accuracy | **94.0 %** |
| ROC AUC | **0.991** |
| Unmasked precision / recall | 0.90 / 1.00 |
| Masked precision / recall | 1.00 / 0.88 |

Confusion matrix (rows: truth, columns: prediction):

|        | unmasked pred | masked pred |
|---|---|---|
| unmasked | 299 | 1 |
| masked   |  35 | 265 |

**Discussion.** The mask classifier is solid on RMFD -- almost
perfect on the unmasked class and 88 % recall on masked. Combined
with Experiment 3, this confirms that the failure of the routing
pipeline is **not** a classifier-quality issue: the routing decision
is high-fidelity, the target it routes to is the problem.

## 6. Live demo

`python -m demo.webcam_demo --gallery demo_gallery`

After enrolling each team member from an unmasked webcam capture, the
demo runs at interactive frame rates on a single CUDA GPU. The on-screen
banner reflects the routed mode, the mask probability, the top-1
identity, and the accept/reject decision. Putting on or removing a mask
flips the routing live; a stranger appearing on camera triggers a reject.

## 7. Discussion

### 7.1 What worked

* **Multi-template enrollment with synthetic masks.** This is the
  approach that actually beats the baseline. It is also the simplest:
  one extra forward pass at enrollment, one extra row in the gallery,
  one extra dot product at recognition, no mask classifier needed at
  match time. The win shows up in rank-1 (+0.6 pts over naive), EER
  (-0.7 pts), AUC (+0.002), and on the harder masked-only subset
  (+0.8 pts).
* **The evaluation harness.** Embedding caching turned a 12-minute
  experiment into a 30-second re-evaluation, which let us iterate
  through three different upper-face designs in one afternoon. Every
  result we report is reproducible from JSON summaries + cached
  embeddings without rerunning the deep networks.
* **Open-set protocol with LFW impostors.** Mixing LFW impostors into
  the RMFD probe set kept the impostor distribution realistic and
  matches the Smart Peephole evaluation pattern.

### 7.2 What didn't -- and why it's still useful to report

* **Mask-aware routing to an upper-face branch is a dead end** on
  pre-trained ArcFace. Three different upper-branch designs (bbox
  fill, tight crop + resize, landmark-aligned fill) all ended up
  worse than the naive baseline on masked rank-1 (Exp 3 oracle row:
  35.3 % vs naive's 51.9 %). The root cause, surfaced by the
  per-probe diff, is that ArcFace already extracts useful identity
  signal from the visible upper face when looking at the masked
  image as-is, and our synthetic forehead fill *erases* the part
  ArcFace was relying on (the natural transition between forehead,
  brow, eye socket, and the upper edge of the actual mask). Score-
  level fusion inherits this disadvantage.
* **Score-level fusion of full + upper.** Mathematically equivalent
  to assuming the two branches make independent errors. Empirically
  they don't: when ``upper`` is wrong, ``full`` is usually wrong on
  the same probe (in different ways), so fusion ranks the wrong
  identities higher rather than averaging out noise.
* **Why we report this anyway.** The course brief asks for rigorous
  evaluation, not just for a working system. Reporting that an
  intuitively-appealing design (routing-based mitigation) doesn't
  improve over a strong baseline -- and explaining why -- is the
  scientific output here. Saying "ArcFace is already robust to mask
  occlusion in this regime" is a non-trivial finding for anyone
  deploying it; saying "but synthetic-mask multi-template enrollment
  *does* add a small consistent gain on top" is the practical
  recommendation.

### 7.3 Future work

* **Two-photo enrollment instead of synthetic masks.** Asking the
  user to enrol with one unmasked and one real-masked photo would
  likely beat the synthetic version of ``multi_max``, since the
  real mask captures texture / shadows the synth overlay does not.
* **Real periocular model.** A network trained specifically on the
  eye region would have a fair chance against ``naive`` masked
  rank-1 -- our upper-fill failed because we asked a full-face model
  to do a job it was not trained for.
* **Bigger gallery / more identities.** RMFD only has 377 paired
  identities; the rank-1 gap between multi-template and naive
  (+0.6 pts) is small enough that statistical confidence would
  benefit from a 1k+ identity evaluation.
* **Multiple enrollment photos per identity** (Smart Peephole's
  multi-template) on top of the synthetic-mask augmentation -- the
  two ideas compose cleanly.
* **Anti-spoofing layer.** Liveness detection in front of the
  pipeline would address presentation attacks, a natural next step
  if AdaptiveFace ever moved beyond a course project.

## 8. Reproducibility

Every experiment is a stand-alone Python module under `experiments/`
with deterministic seeds and JSON summaries. The exact commit hash of
the code, the dataset versions used, and the model URLs are pinned in
the repository. To reproduce a run end-to-end:

```bash
python -m scripts.download_datasets --lfw --mfr2
python -m scripts.download_models
python -m experiments.exp1_unmasked_baseline
python -m experiments.exp2_masked_recognition --source auto
python -m experiments.exp3_cross_mode --source auto
python -m experiments.exp4_mask_detector --use-synth-fallback
```

The cached embeddings, raw similarity matrices, and per-probe CSVs in
`results/` are sufficient to recompute every metric and plot without
re-running inference.

## References

1. Deng, J., Guo, J., Xue, N., Zafeiriou, S. *ArcFace: Additive Angular
   Margin Loss for Deep Face Recognition.* CVPR 2019.
2. Deng, J., Guo, J., Yuxiang, Z., Yu, J., Kotsia, I., Zafeiriou, S.
   *RetinaFace: Single-stage Dense Face Localisation in the Wild.* 2019.
3. Anwar, A., Raychowdhury, A. *Masked Face Recognition for Secure
   Authentication.* arXiv:2008.11104, 2020.
4. Huang, G.B., Ramesh, M., Berg, T., Learned-Miller, E. *Labeled Faces
   in the Wild.* UMass Tech Report, 2007.
5. Wang, Z., et al. *Masked Face Recognition Dataset and Application.*
   2020.
6. AIZOO Tech. *FaceMaskDetection.* GitHub, 2020.
7. De Marsico, M., Nemmi, E., Prenkaj, B., Saturni, G. *A Smart Peephole
   on the Cloud.* IW-BAAS 2017.
