"""Live AdaptiveFace demo for the presentation.

Run after enrolling at least one identity:

    python -m demo.enroll --identity Taha --webcam
    python -m demo.enroll --identity Kutay --webcam
    python -m demo.webcam_demo --gallery demo_gallery

By default we use **Pipeline B (multi_max)** -- the configuration that
won Experiment 3 (rank-1 77.4 %, EER 9.1 %). Pass
``--pipeline routing`` to see the older Pipeline A (adaptive routing)
behaviour we report in the negative-result section.

What you see on screen:
  * The webcam feed with a bounding box around the most prominent face.
  * A status banner: accept/reject, top-1 identity, similarity, mode.
  * The mask classifier's call (informational; multi_max ignores it).
  * The cosine threshold used.

Hotkeys:
  * ``q`` / ESC : quit.
  * ``t``       : toggle the top-3 ranking overlay.
  * ``s``       : screenshot to ``results/demo_screenshots/``.
  * ``r``       : start/stop recording an MP4 (saved next to screenshots).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PIPELINE, RESULTS_DIR  # noqa: E402
from src.embedder import Embedder  # noqa: E402
from src.face_detector import FaceDetector  # noqa: E402
from src.gallery import Gallery  # noqa: E402
from src.mask_classifier import MaskClassifier  # noqa: E402
from src.pipeline import (  # noqa: E402
    AdaptiveFacePipeline,
    MultiTemplatePipeline,
    PipelineDecision,
)


COLOR_OK = (60, 200, 80)
COLOR_REJECT = (40, 40, 200)
COLOR_INFO = (220, 220, 220)


def _draw_overlay(frame, decision: PipelineDecision | None, fps: float, verbose: bool) -> None:
    if decision is None:
        cv2.putText(frame, "no face", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_INFO, 2)
        cv2.putText(frame, f"{fps:5.1f} fps", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_INFO, 1)
        return

    face = decision.face
    bbox_color = COLOR_OK if decision.accepted else COLOR_REJECT
    if face is not None:
        x1, y1, x2, y2 = face.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)

    label = decision.identity or "UNKNOWN"
    status = "ACCEPT" if decision.accepted else "REJECT"
    mode_str = f"[{decision.mode}]"
    mask_str = f"mask: {decision.mask.label} ({decision.mask.p_mask:.2f})"
    banner = f"{status}  {label}  sim={decision.similarity:.3f}  {mode_str}"

    cv2.putText(frame, banner, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bbox_color, 2)
    cv2.putText(frame, mask_str, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_INFO, 1)
    cv2.putText(frame, f"threshold={decision.threshold:.3f}", (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_INFO, 1)
    cv2.putText(frame, f"{fps:5.1f} fps", (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_INFO, 1)

    if verbose and decision.ranking:
        y = 130
        for identity, sim in decision.ranking[:3]:
            cv2.putText(frame, f"  {identity:<15} {sim:.3f}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_INFO, 1)
            y += 22


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gallery", default="demo_gallery")
    ap.add_argument("--device", type=int, default=0, help="OpenCV camera index")
    ap.add_argument("--pipeline", choices=("multi", "routing"), default="multi",
                    help="multi = MultiTemplatePipeline (multi_max, the winner). "
                         "routing = AdaptiveFacePipeline (legacy Pipeline A).")
    # 0.25 sits between the masked-genuine band we measured on MFR2
    # (0.23-0.53) and the impostor band from Experiment 3 (max 0.24 over
    # ~100K comparisons against 145 identities; with a 2-person demo
    # gallery the impostor tail is far below that). The config default
    # (0.35) is tuned for unmasked-only matching and rejects legitimate
    # masked probes.
    ap.add_argument("--threshold", type=float, default=0.25,
                    help="cosine threshold for multi-template pipeline")
    ap.add_argument("--threshold-full", type=float, default=PIPELINE.cosine_threshold_unmasked,
                    help="cosine threshold for the full-face branch (routing pipeline)")
    ap.add_argument("--threshold-upper", type=float, default=PIPELINE.cosine_threshold_masked,
                    help="cosine threshold for the upper-face branch (routing pipeline)")
    args = ap.parse_args()

    gallery_path = RESULTS_DIR / args.gallery
    if not gallery_path.with_suffix(".json").exists():
        print(f"[err ] no gallery at {gallery_path}. Run `python -m demo.enroll` first.")
        return 1
    gallery = Gallery.load(gallery_path)
    print(f"[demo] gallery loaded with {len(gallery)} identities")
    if args.pipeline == "multi" and not gallery.has_synth_templates():
        print("[warn] selected multi pipeline but the gallery has no synth-masked "
              "templates. Re-enrol with the updated demo/enroll.py, or fall back "
              "to --pipeline routing.")

    detector = FaceDetector()
    embedder = Embedder(detector=detector)
    mask_clf = MaskClassifier()
    if args.pipeline == "multi":
        pipeline = MultiTemplatePipeline(
            gallery=gallery, detector=detector, embedder=embedder,
            mask_classifier=mask_clf, threshold=args.threshold,
        )
        print(f"[demo] pipeline = multi_max | threshold={args.threshold}")
    else:
        pipeline = AdaptiveFacePipeline(
            gallery=gallery, detector=detector, embedder=embedder,
            mask_classifier=mask_clf,
            threshold_full=args.threshold_full,
            threshold_upper=args.threshold_upper,
        )
        print(f"[demo] pipeline = routing | full={args.threshold_full} "
              f"upper={args.threshold_upper}")
    print(f"[demo] mask classifier backend: {mask_clf.backend_name}")

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"[err ] could not open webcam device {args.device}")
        return 2

    verbose = True
    last_time = time.time()
    fps = 0.0
    screenshots_dir = RESULTS_DIR / "demo_screenshots"
    video_writer: cv2.VideoWriter | None = None
    video_path: str | None = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decision = pipeline.recognize(frame)
            now = time.time()
            dt = now - last_time
            last_time = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else fps)

            _draw_overlay(frame, decision, fps, verbose)
            cv2.imshow("AdaptiveFace -- live demo", frame)

            if video_writer is not None:
                video_writer.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("t"):
                verbose = not verbose
            if key == ord("s"):
                screenshots_dir.mkdir(parents=True, exist_ok=True)
                dst = screenshots_dir / f"shot_{int(now)}.jpg"
                cv2.imwrite(str(dst), frame)
                print(f"[demo] screenshot -> {dst}")
            if key == ord("r"):
                if video_writer is None:
                    screenshots_dir.mkdir(parents=True, exist_ok=True)
                    video_path = str(screenshots_dir / f"recording_{int(now)}.mp4")
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (w, h))
                    if not video_writer.isOpened():
                        print("[err ] failed to open VideoWriter for recording")
                        video_writer = None
                    else:
                        print(f"[demo] recording -> {video_path}")
                else:
                    video_writer.release()
                    print(f"[demo] recording stopped: {video_path}")
                    video_writer = None
                    video_path = None
    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"[demo] recording closed: {video_path}")
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
