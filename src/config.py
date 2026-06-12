"""Central configuration for AdaptiveFace.

All paths are resolved relative to the project root (the directory that
contains the `src/` package). Keeping every magic constant here means
experiments stay reproducible and we can sweep parameters from one place.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = PROJECT_ROOT / "report" / "figures"
CACHE_DIR: Path = RESULTS_DIR / "cache"

for _d in (DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


LFW_DIR: Path = DATA_DIR / "lfw"
RMFD_DIR: Path = DATA_DIR / "rmfd"
MFR2_DIR: Path = DATA_DIR / "mfr2"


@dataclass(frozen=True)
class EmbedderConfig:
    """ArcFace embedder settings.

    `model_pack` refers to the InsightFace model zoo entry. `buffalo_l`
    bundles RetinaFace detector + ArcFace (R100) recognizer and is the
    default the library downloads on first use.
    """

    model_pack: str = "buffalo_l"
    # 640x640 is RetinaFace's default but it downsamples small pre-cropped
    # faces (RMFD's 130x150 head shots) into oblivion. 320x320 is big enough
    # for full LFW frames (250x250) and small enough to still see a face
    # that fills most of a tiny RMFD crop.
    det_size: tuple[int, int] = (320, 320)
    embedding_dim: int = 512
    ctx_id: int = 0  # GPU id; set to -1 to force CPU.


@dataclass(frozen=True)
class MaskClassifierConfig:
    """Mask classifier settings.

    Backends tried in order: ONNX (if present), AIZOO Caffe model
    (opencv.dnn, no extra dependency), landmark heuristic. The first one
    whose weights are on disk wins.
    """

    onnx_path: Path = MODELS_DIR / "mask_classifier.onnx"
    caffe_proto_path: Path = MODELS_DIR / "face_mask_detection.prototxt"
    caffe_weights_path: Path = MODELS_DIR / "face_mask_detection.caffemodel"
    # Legacy alias retained so older calls keep working.
    model_path: Path = MODELS_DIR / "mask_classifier.onnx"
    input_size: tuple[int, int] = (260, 260)
    threshold: float = 0.5  # P(mask) above this -> "masked".


@dataclass(frozen=True)
class PipelineConfig:
    """Defaults for the end-to-end pipeline.

    Cosine similarity thresholds are chosen at the equal-error point on the
    unmasked dev split and refined per mode. Values here are starting
    points; experiments overwrite them.
    """

    cosine_threshold_unmasked: float = 0.35
    cosine_threshold_masked: float = 0.30
    upper_face_crop_ratio: float = 0.55  # keep top 55% of face bbox.
    rank_k: int = 5  # default for CMC plots.


EMBEDDER = EmbedderConfig()
MASK_CLF = MaskClassifierConfig()
PIPELINE = PipelineConfig()


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "MODELS_DIR",
    "RESULTS_DIR",
    "FIGURES_DIR",
    "CACHE_DIR",
    "LFW_DIR",
    "RMFD_DIR",
    "MFR2_DIR",
    "EmbedderConfig",
    "MaskClassifierConfig",
    "PipelineConfig",
    "EMBEDDER",
    "MASK_CLF",
    "PIPELINE",
]
