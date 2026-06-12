"""Enroll users into a Gallery from photos on disk or a live webcam capture.

Two enrollment modes:

  * Photo: `python -m demo.enroll --gallery demo_gallery --identity Taha \
        --image path/to/photo.jpg`
  * Webcam: `python -m demo.enroll --gallery demo_gallery --identity Taha \
        --webcam` (press SPACE to capture, ESC to quit).

The resulting gallery is written to `results/<gallery_name>.npz` + `.json`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RESULTS_DIR  # noqa: E402
from src.embedder import Embedder  # noqa: E402
from src.gallery import Gallery  # noqa: E402


def _capture_from_webcam(window_title: str) -> str | None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[err ] could not open webcam (device 0).")
        return None
    print("[demo] webcam open. SPACE = capture, ESC = quit.")
    saved: str | None = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow(window_title, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == 32:  # SPACE
            tmp = Path(RESULTS_DIR) / "demo_captures"
            tmp.mkdir(parents=True, exist_ok=True)
            dst = tmp / f"capture_{cv2.getTickCount()}.jpg"
            cv2.imwrite(str(dst), frame)
            saved = str(dst)
            print(f"[demo] saved capture to {dst}")
            break
    cap.release()
    cv2.destroyAllWindows()
    return saved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gallery", default="demo_gallery", help="Gallery name (file stem).")
    ap.add_argument("--identity", required=True)
    src_group = ap.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--image", type=Path)
    src_group.add_argument("--webcam", action="store_true")
    args = ap.parse_args()

    gallery_path = RESULTS_DIR / args.gallery
    if gallery_path.with_suffix(".json").exists():
        gallery = Gallery.load(gallery_path)
        print(f"[demo] loaded existing gallery with {len(gallery)} identities")
    else:
        gallery = Gallery()
        print("[demo] starting a fresh gallery")

    image_path = args.image
    if args.webcam:
        captured = _capture_from_webcam(f"Enroll: {args.identity}")
        if not captured:
            print("[demo] no capture; aborting.")
            return 1
        image_path = Path(captured)

    embedder = Embedder()
    entry = gallery.enroll(args.identity, image_path, embedder)
    if entry is None:
        print(f"[err ] no face detected in {image_path}; not enrolled.")
        return 2

    gallery.save(gallery_path)
    print(f"[ok  ] enrolled {args.identity}. Gallery now has {len(gallery)} identities.")
    print(f"       saved to {gallery_path}.npz / .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
