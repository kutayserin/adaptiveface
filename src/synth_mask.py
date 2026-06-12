"""Synthetic mask overlay using InsightFace's 5-point landmarks.

We draw a coloured polygon spanning nose-bridge -> ears -> chin -> back to
nose-bridge to approximate a surgical mask. The polygon is anchored by the
nose tip and mouth corners returned by RetinaFace, so it sits where a real
mask would sit on each face.

This is a pragmatic baseline; a more sophisticated implementation would use
68-point landmarks and a textured PNG overlay (e.g., MaskTheFace). The
simple polygon is good enough to (a) populate a "masked LFW" eval set when
the real RMFD download is unavailable and (b) give the demo a fallback
synthetic mask source so the live webcam path always has something to show.
"""
from __future__ import annotations

import cv2
import numpy as np

from .face_detector import DetectedFace


MASK_COLORS = {
    "surgical_blue": (180, 145, 92),  # BGR
    "white": (240, 240, 240),
    "black": (40, 40, 40),
    "cloth_grey": (140, 140, 140),
}


def overlay_synthetic_mask(
    image_bgr: np.ndarray,
    face: DetectedFace,
    color: str = "surgical_blue",
    opacity: float = 1.0,
) -> np.ndarray:
    """Draw a synthetic mask polygon on a copy of `image_bgr`.

    Landmarks (`face.kps`) are in (x, y) pixel coordinates:
        0: left eye, 1: right eye, 2: nose tip,
        3: left mouth corner, 4: right mouth corner.

    The polygon vertices we use:
        - just below nose tip (mask top centre)
        - left ear-side anchor (bbox left, slightly above mouth)
        - chin anchor (centre-x, just below bbox bottom)
        - right ear-side anchor (bbox right, slightly above mouth)
    """
    if color not in MASK_COLORS:
        raise ValueError(f"unknown mask color: {color}")
    out = image_bgr.copy()
    kps = face.kps
    if kps.shape[0] < 5:
        return out  # safety: shouldn't happen for RetinaFace.

    nose = kps[2]
    mouth_l = kps[3]
    mouth_r = kps[4]
    bbox = face.bbox
    h_img, w_img = out.shape[:2]

    # Mask top: half-way between nose and mouth average y.
    mouth_mid_y = (mouth_l[1] + mouth_r[1]) / 2.0
    top_y = nose[1] + 0.35 * (mouth_mid_y - nose[1])
    top_left = (int(nose[0] - (mouth_r[0] - mouth_l[0]) * 0.7), int(top_y))
    top_right = (int(nose[0] + (mouth_r[0] - mouth_l[0]) * 0.7), int(top_y))

    # Side anchors near the bbox edges, slightly above mouth.
    side_y = int(mouth_mid_y)
    left_anchor = (max(0, int(bbox[0])), side_y)
    right_anchor = (min(w_img - 1, int(bbox[2])), side_y)

    # Chin: extend a bit past bbox bottom for realism.
    chin_x = int((mouth_l[0] + mouth_r[0]) / 2.0)
    chin_y = min(h_img - 1, int(bbox[3] + 0.1 * (bbox[3] - bbox[1])))
    chin = (chin_x, chin_y)

    polygon = np.array([
        top_left,
        left_anchor,
        chin,
        right_anchor,
        top_right,
    ], dtype=np.int32)

    overlay = out.copy()
    cv2.fillPoly(overlay, [polygon], MASK_COLORS[color])
    # Subtle dark border for realism.
    cv2.polylines(overlay, [polygon], isClosed=True, color=(0, 0, 0), thickness=1)

    if opacity >= 1.0:
        return overlay
    return cv2.addWeighted(overlay, opacity, out, 1.0 - opacity, 0)


__all__ = ["overlay_synthetic_mask", "MASK_COLORS"]
