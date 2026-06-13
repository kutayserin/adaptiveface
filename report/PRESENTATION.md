# AdaptiveFace — Presentation Plan

Target length: ~12 slides, ~12 minutes presenting + Q&A.

## Slide 1 — Title

> **AdaptiveFace: Mask-Aware Face Authentication**
> Taha & Kutay — Biometric Systems, MSc Computer Science — May 2026.

Talking point: "Same core problem Smart Peephole solved — verify that
the person at the door is who they claim to be — with one twist that
matters in the post-2020 world: what if the user is wearing a mask?"

## Slide 2 — The problem

Show two photos of the same person: masked and unmasked. State the
user story:

> The user enrols **once** with an unmasked photo. Later they show up
> at the camera **masked or unmasked**. The system must recognise them
> **in both states**, without re-enrolling.

The catch: face recognition models trained on unmasked faces (like
ArcFace) collapse when the lower half of the face disappears. Naïve
solutions either retrain the model (data-hungry) or refuse to operate
when a mask is detected (poor user experience). Can we keep the
off-the-shelf model and still solve this?

## Slide 3 — Related work

Three projects from the course shaped our design choices:

* **Smart Peephole** — modular detection / recognition / fallback
  pipeline, plus the FAR/FRR/ROC/CMC/sigmoid reporting template.
* **BioPen** — three-stage acquisition / feature / decision layout.
* **Biotouch** — ensemble methods (majority / average / weighted
  average) for combining multiple classifiers.

> "We borrow the modular pipeline from Smart Peephole, the three-stage
> structure from BioPen, and the multiple-configuration evaluation
> pattern from Biotouch."

## Slide 4 — Two mitigation strategies we built

We implemented two pipelines in parallel and compared them on the
same probe set:

* **Pipeline A — Mask-aware routing**: dual-template gallery (full +
  upper-face), AIZOO mask classifier decides which template to score
  the probe against at recognition time.
* **Pipeline B — Multi-template enrollment**: each identity gets two
  full-face templates (one from the real unmasked photo, one from a
  synthetic-masked rendering of the same photo). Matching takes the
  max of the two cosine similarities, no mask classifier needed.

Show both ASCII diagrams from `REPORT.md` §3.1 and §3.2 side by side.

## Slide 5 — Datasets and metrics

* **LFW** — unmasked baseline (Experiment 1).
* **RMFD** — real masked + unmasked images of the same identities
  (Experiments 2 & 3).
* **MFR2** — small paired set used for demo sanity checks.

Metrics suite: FAR, FRR, ROC, CMC, EER + sigmoid fits — exactly the
Smart Peephole reporting style.

**Open-set protocol**: we mix 80 LFW impostor identities into the
RMFD probe set, so the gallery sees probes it has never enrolled.

## Slide 6 — Experiment 1: Unmasked baseline (LFW)

| Metric | Value |
|---|---|
| Rank-1 | **92.7 %** |
| EER | **5.3 %** |
| AUC | 0.985 |

> "Our ArcFace stack is healthy. On the standard LFW protocol Smart
> Peephole reported EER around 0.02 on a smaller subset — we're at
> 0.05 with a wider impostor pool, in the same ballpark."

This is the "this is the ceiling, everything else degrades from here"
slide.

## Slide 7 — Experiment 2: Masked-only probes (RMFD)

| Mode | Rank-1 | EER | AUC |
|---|---|---|---|
| `full` (naive) | **54.6 %** | 15.5 % | 0.908 |
| `upper` (aligned fill) | 32.4 % | 20.9 % | 0.866 |

> "A mask hurts: rank-1 falls from 92.7 % to 54.6 %, a 38-point gap
> that any mitigation has to close. But running our upper-face-fill
> embedder by itself is *worse* than just feeding the masked image
> straight into ArcFace — 22 points worse. First warning sign that
> routing may not be the right idea."

## Slide 8 — Experiment 3: Cross-mode (the headline)

Show all 8 configurations:

| Config | Rank-1 | EER | Masked-only Rank-1 |
|---|---|---|---|
| naive | 76.8 % | 9.8 % | 51.9 % |
| oracle (routing) | 68.5 % | 9.8 % | 35.3 % |
| adaptive (routing) | 69.7 % | 9.6 % | 37.8 % |
| fusion_max | 55.2 % | 13.5 % | 39.0 % |
| fusion_avg | 69.3 % | 10.8 % | 45.2 % |
| fusion_weighted | 70.5 % | 10.0 % | 41.1 % |
| **multi_max** ⭐ | **77.4 %** | **9.1 %** | **52.7 %** |
| **multi_adaptive** | 76.1 % | **8.7 %** | 50.6 % |

**Three observations, each its own bullet:**

1. **Routing-based mitigation doesn't work** — oracle and adaptive
   are *worse* than naive. Surprising? Not when you realise the mask
   classifier hits 95.9 % accuracy on this cross-mode probe set (and
   94 % on the balanced set in Experiment 4) — the routing *decision*
   is correct; the destination it routes to is the problem.

2. **Score-level fusion doesn't help either** — full + upper aren't
   making independent errors, so fusion isn't averaging out noise.

3. **Multi-template enrollment wins.** Each identity gets a
   `gallery_real` row and a `gallery_synth` row (same photo, synthetic
   mask overlaid). `multi_max` improves rank-1 by +0.6 pts and EER
   by -0.7 pts. The absolute numbers are small, but the direction is
   **consistent** across all four metrics.

## Slide 9 — Why does routing fail?

The single most informative number from our per-probe CSV:

> Among masked probes, **naive wrong but upper right = 9 cases**.
> **Naive right but upper wrong = 57 cases**. Net loss: 48 probes.

**Interpretation**: ArcFace already extracts useful identity signal
from the visible upper face (brows, eyes, forehead) even when the
lower half is masked. Our synthetic forehead-fill **erases that
signal** without adding new information.

This is a finding we couldn't have predicted before running the
experiments — **a scientifically defensible negative result.**

## Slide 10 — Why does multi-template work?

Visual: three crops of the same identity:

* `gallery_real` — original unmasked photo.
* `gallery_synth` — same photo with synthetic blue mask overlaid.
* `probe_masked` — a real RMFD masked image of that person.

`probe_masked` and `gallery_synth` live in the same mask distribution:
ArcFace gives them higher similarity. When `probe_masked` is matched
against `gallery_real`, the mask creates a half-face mismatch that
drags the similarity down.

`max(sim_real, sim_synth)` automatically picks the right gallery row
for each probe. **It's not routing — it's gallery augmentation.**

## Slide 11 — Experiment 4: Mask classifier sanity

| Metric | Value |
|---|---|
| Backend | AIZOO Caffe via `cv2.dnn` |
| Accuracy | 94 % |
| AUC | 0.991 |
| Confusion | 299/300 unmasked OK, 265/300 masked OK |

> "The classifier itself is solid. This rules out the 'maybe the mask
> detector is too noisy' explanation for why routing failed in
> Experiment 3."

## Slide 12 — Live demo + conclusion

Live demo (webcam):
1. Walk up unmasked → **accept**, identity displayed.
2. Put on a mask → **still accept** (the multi-template matcher picks
   the synth-masked row); the on-screen banner shows mode = `multi_max`.
3. Hold up a photo of an unenrolled person → **reject**.

**Take-away**:
> "We tried routing-based mitigation — it didn't work. We tried
> multi-template enrollment — it gave a small but consistent
> improvement across every metric. Achieved without changing the
> pre-trained ArcFace model and without collecting any real masked
> enrollment data — just one extra synthetic photo per user at
> enrollment time."

---

## Likely questions, prepared answers

**Q: Multi-template's gain is small (rank-1 +0.6 pts). Is it
meaningful?**

> "The absolute number is small because the naive baseline is already
> strong — ArcFace handles partial occlusion better than we expected.
> But the direction is consistent on all four metrics (rank-1, EER,
> AUC, masked-only rank-1), and importantly **multi_adaptive achieves
> the lowest EER overall** (8.7 % vs 9.8 % for naive), which is the
> most relevant metric for the verification use case. We'd expect the
> gap to widen on a larger gallery (1k+ identities)."

**Q: Why did routing fail? The mask classifier is 94 % accurate —
isn't that good enough?**

> "Classifier isn't the issue. The per-probe diff shows the routing
> destination is the problem: on masked probes, the upper-face branch
> is right where naive is wrong 9 times, but wrong where naive is
> right 57 times. Even a perfect mask classifier wouldn't help,
> because routing to the upper branch is strictly worse than not
> routing at all. Pre-trained ArcFace already mines useful signal
> from the visible upper face; our synthetic forehead-fill destroys
> that signal."

**Q: How did you pick the thresholds?**

> "Equal-error point of each configuration's own FAR/FRR curve. The
> JSON summaries report these per-mode thresholds. We deliberately
> didn't use a single global threshold because the score distributions
> differ across configurations."

**Q: The synthetic mask distribution differs from real RMFD masks —
doesn't that invalidate your gain?**

> "Partially — synthetic and real masks don't share the same
> distribution, but both share the key property of occluding the
> lower face. In ArcFace's embedding space that partial overlap is
> enough. We list this as future work: enrolling with a *real* masked
> photo as the second template should give a larger gain than the
> synthetic version we used."

**Q: Did you use open-set evaluation?**

> "Yes — 150 enrolled identities plus 80 LFW impostor identities.
> Impostor probes are people the gallery has never seen; their entire
> similarity row contributes only to the impostor distribution. The
> RMFD genuine probes and LFW impostor probes share the same matching
> protocol."

**Q: Is the multi-template idea yours?**

> "Smart Peephole proposed multi-template enrollment for a different
> use case (multiple poses per identity). Our contribution is applying
> it to a *synthetic augmentation* of the same enrollment photo, and
> evaluating it specifically in the masked-recognition setting. The
> literature mentions tools like MaskTheFace for augmentation at
> training time, but integrating it as an enrollment-time dual
> template — so it costs nothing at recognition time — is the new
> angle here."

**Q: Tight crop vs. fill — why did you try both?**

> "Scientific honesty. Tight crop was the first attempt (1–8 % rank-1,
> terrible). Fill-in-bbox-space (no alignment) was also bad.
> Landmark-aligned fill preserved the geometry but still trailed the
> naive baseline. Reporting all three shows what we tried, why it
> didn't work, and why we eventually abandoned the routing direction."

**Q: Would a different recognition model change the result?**

> "Alternative full-face embedding models (FaceNet, VGGFace, AdaFace)
> would likely behave similarly in `naive` mode — they were all
> trained on largely unmasked data. The real change would come from
> a dedicated periocular network (e.g. PocketNet-Periocular). Setting
> that up would require training an additional model, which was out
> of scope for this two-week project."

**Q: Does the demo run multi-template?**

> "The webcam demo currently uses the `adaptive` routing path in
> `pipeline.py`. We switched the demo matcher to `multi_max` before
> the presentation — the gallery format already supports two
> templates per identity, only the field name had to change from
> full+upper to real+synth_masked."

**Q: Is 150 enrolled identities enough?**

> "Standard RMFD evaluations use between 80 and 500 identities; 150
> is a reasonable middle ground. Running larger on CPU would have
> taken hours. We fixed the random seed at 42 for reproducibility,
> and the embedding cache lets us re-evaluate all configurations on
> the *same* probe set — so any variance we report is inter-config,
> not intra-config noise."
