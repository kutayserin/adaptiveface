# AdaptiveFace — Demo Recording Script

A 30–60 second screen recording that hits the four moments the
professor will look for: (1) unmasked recognition, (2) masked
recognition of the same enrolled user, (3) impostor rejection,
(4) the on-screen overlay reflecting all of this in real time.

You record this **once** before the presentation. We'll embed the
MP4 in the slide deck (or play it from disk on slide 12).

---

## Before you press record

1. Plug in good lighting (a desk lamp pointed at the wall behind the
   webcam works) — the mask classifier and ArcFace both behave better
   when the face isn't underexposed.
2. Close every other window. The presentation slot is small; you
   don't want Slack notifications cropping into the recording.
3. Have **one real mask** ready (surgical / cloth, blue or black).
4. Have a printed photo or a phone showing somebody **not enrolled**
   (e.g., a random celebrity from LFW you didn't enrol). This is
   the impostor probe.

## Step 1 — Enrol two identities (do this once, before recording)

```powershell
cd C:\Users\tahak\ogproject\adaptiveface
.venv\Scripts\activate    # if you use a venv

# Enrol identity 1 (you, unmasked)
python -m demo.enroll --identity Taha --webcam
# When the webcam window appears: face the camera, press SPACE.

# Enrol identity 2 (your teammate, also unmasked)
python -m demo.enroll --identity Kutay --webcam
```

You should see a confirmation like `[ok ] enrolled Taha. Gallery now
has 2 identities.` and the gallery files appear under
`results/demo_gallery.{json,npz}`. The new enrolment also stores a
**synthetic-masked template** for each identity (that's what makes
the multi_max pipeline work without a routing classifier).

## Step 2 — Start the recording

Use Windows' built-in screen recorder — no extra install needed:

1. Open the demo window first (next step), then
2. Press **`Win` + `G`** to open the Game Bar.
3. In the "Capture" widget, click the round **record** button
   (or **`Win` + `Alt` + `R`**) to start recording.
4. The recording file lands in
   `C:\Users\tahak\Videos\Captures\` as an MP4.

**If Game Bar doesn't catch the OpenCV window** (some Windows
configurations only record the foreground app), use OBS Studio
instead — it's free and records any window. In OBS:

* Sources → ➕ → Window Capture → pick
  "AdaptiveFace -- live demo".
* Start Recording.

There's also an in-app recorder: press `r` inside the demo window
to start/stop an MP4 that goes to `results/demo_screenshots/`. Use
this if both Game Bar and OBS misbehave — it just records the
processed frames so the overlay always matches what we narrate.

## Step 3 — Launch the demo

```powershell
python -m demo.webcam_demo --gallery demo_gallery --pipeline multi
```

What you should see on the OpenCV window:
* A bounding box around the most prominent face.
* A status banner: `ACCEPT  Taha  sim=0.4xx  [multi_max]`.
* A line with the mask classifier's call (info only — multi_max
  ignores it).
* The cosine threshold (≈ 0.35 by default).

## Step 4 — The 30-second on-camera routine

| Time   | Action                                                   | What should appear on screen                                         |
|--------|----------------------------------------------------------|----------------------------------------------------------------------|
| 0:00   | Sit centred, face the camera, no mask.                   | `ACCEPT  Taha  sim≈0.5  [multi_max]`. Bounding box GREEN.            |
| 0:05   | Slowly turn your head ~30° left, then right.             | Banner stays GREEN; identity unchanged. Demonstrates pose tolerance. |
| 0:10   | Put the mask on (cover nose + mouth fully).              | Banner stays GREEN. Mask line flips to `mask: masked (0.95+)`.       |
| 0:15   | Hold steady for 2 seconds so the camera gets a clean shot. | Similarity may dip 0.05–0.10; system still ACCEPTs.                  |
| 0:18   | Remove the mask.                                         | Mask line flips back to `unmasked`. Similarity rises.                |
| 0:22   | Hand the seat to your teammate (Kutay), no mask.         | Banner: `ACCEPT  Kutay  sim≈0.5  [multi_max]`.                       |
| 0:27   | Show the impostor photo (printed / phone screen) toward the camera. | Banner flips to **RED** `REJECT  UNKNOWN  sim≈0.15`.        |
| 0:35   | Press `q` to quit cleanly.                               | Window closes; you can stop the screen recorder.                     |

## Step 5 — Stop the recording

* Game Bar: `Win` + `Alt` + `R` again, or click the capture
  widget's stop button.
* OBS: click "Stop Recording".
* In-app: press `r` again, the MP4 path is printed in the terminal.

## Step 6 — Trim the video (optional)

Windows Photos can trim:
1. Right-click the MP4 → Open with → Photos.
2. Click "Edit & Create" → "Trim".
3. Save trimmed copy. Aim for 30–45 seconds total.

## Step 7 — Add it to the bundle

Drop the MP4 into `report/demo_video/AdaptiveFace_demo.mp4` and
re-run the zip script if you want it inside the bundle. (The
slide deck stays the same — we just point the audience at the
video file during slide 12.)

---

## If something looks wrong on camera

| Symptom | Likely cause | Fix |
|---|---|---|
| Banner is RED while you're enrolled and unmasked | Lighting too dark, threshold too tight | More light, or pass `--threshold 0.20` (default is 0.25) |
| Similarity hovers below 0.2 | The enrolment photo was a bad angle | Re-enrol with a clean front-facing capture |
| Mask line says `unmasked` while you're masked | Mask classifier missed a partial occlusion | Make sure the mask covers nose AND mouth; lower threshold via `MASK_CLF.threshold` in src/config.py |
| No face detected | Webcam covered / wrong device | `--device 1` (try the other camera index) |
| Demo crashes on import | onnxruntime DLL conflict | Re-run `pip install --force-reinstall --no-deps onnxruntime` |

## What to say while recording (voiceover, optional)

> "This is AdaptiveFace running the multi_max configuration. I'm
> enrolled with one unmasked photo; the system has built two gallery
> templates from it — one real, one with a synthetic mask overlaid.
> Right now I'm being recognised as Taha. Watch what happens when I
> put on a real mask … same identity, same green banner, no
> retraining required. Now I hand over to Kutay … recognised
> independently. And here's a stranger — rejected, similarity
> stays below the threshold."

(Don't read this word-for-word — keep it conversational.)
