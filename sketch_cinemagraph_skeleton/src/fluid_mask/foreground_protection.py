"""Detect and protect static foreground objects (e.g. boat) inside the fluid mask."""

import cv2
import numpy as np


def build_foreground_protection_mask(
    fluid_mask,
    min_hole_area: int = 50,
    max_hole_area_ratio: float = 0.3,
) -> np.ndarray:
    """
    Find black holes INSIDE the white water region by connected-component analysis.

    The final fluid mask has water=255 and static regions=0.  Objects like a
    boat sit as black holes enclosed by the white water area.  This function
    returns a mask of those enclosed holes so they can be protected from motion.

    Algorithm:
      1. Invert the mask → zero-region = anything that is NOT water
      2. Label connected components of the zero-region
      3. Labels touching the image border = outer static (sky/land/shore) → excluded
      4. Remaining labels = holes enclosed by water = foreground objects (boat, dock …)
      5. Filter by area to discard noise and over-large regions

    Returns:
        hole_mask: uint8 H×W, 255 at detected foreground holes, 0 elsewhere
    """
    m = fluid_mask[:, :, 0] if fluid_mask.ndim == 3 else fluid_mask
    mask_2d = m > 0
    H, W = mask_2d.shape

    inv = (~mask_2d).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(inv)

    border_labels: set = set()
    for arr in (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]):
        border_labels.update(arr.tolist())

    hole_mask = np.zeros((H, W), dtype=np.uint8)
    max_area = max_hole_area_ratio * H * W
    for label_id in range(1, num_labels):
        if label_id in border_labels:
            continue
        comp = labels == label_id
        area = int(comp.sum())
        if min_hole_area <= area <= max_area:
            hole_mask[comp] = 255

    n_holes = int((hole_mask > 0).sum())
    if n_holes > 0:
        print(f"[ForegroundProtection] Detected {n_holes} foreground hole pixels "
              f"({100.0 * n_holes / (H * W):.2f}% of image)")
    else:
        print("[ForegroundProtection] No foreground holes detected inside water mask")

    return hole_mask


def dilate_and_feather_protection_mask(
    hole_mask: np.ndarray,
    dilation_px: int = 10,
    feather_px: int = 15,
):
    """
    Expand the protection zone and produce smooth transition alphas.

    Args:
        hole_mask:   uint8 H×W, 255 at raw holes (from build_foreground_protection_mask)
        dilation_px: pixels to expand the protection zone beyond the raw hole
        feather_px:  pixels over which flow transitions from 0→full beyond the
                     protection zone boundary

    Returns:
        dilated_mask    (uint8)   — binary protection zone (hole + dilation ring), 255/0
        composite_alpha (float32) — 1.0 at hole centre → 0.0 at dilation boundary;
                                    use as: frame = warped*(1-α) + original*α
        motion_alpha    (float32) — 0.0 inside protection zone → 1.0 at feather_px
                                    beyond; use as: flow = flow * motion_α
    """
    if hole_mask.max() == 0:
        H, W = hole_mask.shape[:2]
        ones = np.ones((H, W), dtype=np.float32)
        return hole_mask.copy(), np.zeros((H, W), dtype=np.float32), ones

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilation_px + 1, 2 * dilation_px + 1)
    )
    dilated = cv2.dilate(hole_mask, kernel)

    # composite_alpha: 1.0 at hole centre, fades to 0.0 at dilation boundary
    # distanceTransform gives distance to nearest 0 pixel inside `dilated`
    dist_inside = cv2.distanceTransform(dilated, cv2.DIST_L2, 5)
    composite_alpha = np.minimum(
        dist_inside / max(float(dilation_px), 1.0), 1.0
    ).astype(np.float32)

    # motion_alpha: 0.0 inside protection zone, ramps to 1.0 at feather_px beyond
    protected_bin = (dilated > 0).astype(np.uint8)
    dist_from_protection = cv2.distanceTransform(1 - protected_bin, cv2.DIST_L2, 5)
    motion_alpha = np.minimum(
        dist_from_protection / max(float(feather_px), 1.0), 1.0
    ).astype(np.float32)

    return dilated, composite_alpha, motion_alpha
