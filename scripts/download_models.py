"""Fetch pre-trained model weights used by the pipeline.

InsightFace downloads its own ArcFace/RetinaFace bundle on first use, so we
only worry about the mask classifier here. We prefer the AIZOO Caffe
model (still hosted on GitHub raw at the time of writing) over the dead
ONNX export. The pipeline silently falls back to a landmark heuristic
when no weights are present, so failing here is not fatal.
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import urlretrieve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import MASK_CLF, MODELS_DIR  # noqa: E402


# Caffe assets (primary path). prototxt + caffemodel are loaded by
# cv2.dnn.readNetFromCaffe; no extra Python dependency needed.
CAFFE_PROTO_URLS = [
    "https://raw.githubusercontent.com/AIZOOTech/FaceMaskDetection/master/models/face_mask_detection.prototxt",
]
CAFFE_WEIGHTS_URLS = [
    "https://raw.githubusercontent.com/AIZOOTech/FaceMaskDetection/master/models/face_mask_detection.caffemodel",
]
# ONNX assets (kept for completeness; both currently 404/401).
ONNX_URLS = [
    "https://github.com/AIZOOTech/FaceMaskDetection/releases/download/v1.0/face_mask_detection.onnx",
]


def _download(url: str, dest: Path) -> Path | None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already present ({dest.stat().st_size / 1024:.1f} KB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[get ] {url}")
    try:
        urlretrieve(url, dest)
        size = dest.stat().st_size
        print(f"[ok  ] saved {dest.name} ({size / 1024:.1f} KB)")
        return dest
    except Exception as exc:
        print(f"[warn] failed: {exc}")
        # Don't leave a zero-byte file lying around.
        if dest.exists() and dest.stat().st_size == 0:
            dest.unlink()
        return None


def _try_mirrors(urls: list[str], dest: Path) -> Path | None:
    for url in urls:
        result = _download(url, dest)
        if result is not None:
            return result
    return None


def fetch_caffe_mask_classifier() -> bool:
    proto = _try_mirrors(CAFFE_PROTO_URLS, MASK_CLF.caffe_proto_path)
    weights = _try_mirrors(CAFFE_WEIGHTS_URLS, MASK_CLF.caffe_weights_path)
    if proto is None or weights is None:
        print("[warn] Caffe mask classifier incomplete; pipeline will fall back.")
        return False
    print(f"[ok  ] Caffe mask classifier ready at {MODELS_DIR}")
    return True


def fetch_onnx_mask_classifier() -> bool:
    return _try_mirrors(ONNX_URLS, MASK_CLF.onnx_path) is not None


def main() -> int:
    print(f"Models dir: {MODELS_DIR}")
    caffe_ok = fetch_caffe_mask_classifier()
    if not caffe_ok:
        # Try ONNX as a long-shot.
        if fetch_onnx_mask_classifier():
            print("[ok  ] ONNX mask classifier present.")
    print("InsightFace weights download lazily on first .prepare() call.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
