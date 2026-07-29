"""
GrabCut segmentation — port of udder_pipeline/src/segment.py.

Same MAX_DIM, same margin fraction, same iteration count, same fallback rule,
so what you see here is what segment.py wrote to data/processed/segmented/.
The only addition is that the mask is returned as well, because the console
needs to draw it.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MAX_DIM = 320
MARGIN_FRAC = 0.04
ITERS = 3
MIN_FOREGROUND_FRAC = 0.05


@dataclass
class SegmentResult:
    image: np.ndarray          # BGR, background zeroed (or the original on fallback)
    mask: np.ndarray | None    # uint8 {0,1} at full resolution, None on fallback
    ok: bool                   # False means GrabCut was rejected and the original is returned
    coverage: float            # fraction of pixels kept as foreground
    reason: str


def grabcut_segment(img: np.ndarray) -> SegmentResult:
    h, w = img.shape[:2]
    scale = min(1.0, MAX_DIM / max(h, w))
    small = (
        cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
        if scale < 1.0
        else img
    )
    sh, sw = small.shape[:2]

    mx, my = max(1, int(sw * MARGIN_FRAC)), max(1, int(sh * MARGIN_FRAC))
    rect = (mx, my, sw - 2 * mx, sh - 2 * my)

    mask = np.zeros((sh, sw), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(small, mask, rect, bgd_model, fgd_model, ITERS, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return SegmentResult(img, None, False, 1.0, "GrabCut failed — original image kept.")

    mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype("uint8")
    coverage = float(mask2.sum()) / float(sh * sw)

    if coverage < MIN_FOREGROUND_FRAC:
        return SegmentResult(
            img, None, False, coverage,
            f"Foreground was only {coverage:.1%} of the frame — below the 5% floor, "
            "so the original image was kept.",
        )

    full_mask = cv2.resize(mask2, (w, h), interpolation=cv2.INTER_NEAREST)
    result = img * full_mask[:, :, np.newaxis]
    return SegmentResult(
        result, full_mask, True, coverage,
        f"GrabCut kept {coverage:.1%} of the frame as foreground.",
    )


def mask_outline(img_bgr: np.ndarray, mask: np.ndarray,
                 colour=(95, 110, 11), thickness: int = 2) -> np.ndarray:
    """Draw the foreground boundary onto a copy of the image (BGR in, BGR out)."""
    out = img_bgr.copy()
    contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, colour, thickness)
    return out
