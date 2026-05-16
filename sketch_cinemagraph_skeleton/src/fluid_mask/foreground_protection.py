"""Detect and protect static foreground objects (e.g. boat) inside the fluid mask."""

import cv2
import numpy as np


def build_foreground_protection_mask(
    fluid_mask,
    min_hole_area: int = 50,
    max_hole_area_ratio: float = 0.3,
) -> np.ndarray:
    """Find black holes enclosed by the white water region (boats, docks) and return their mask."""
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
    """Dilate the hole mask and produce composite/motion alpha fields.

    Returns (dilated_mask, composite_alpha, motion_alpha):
      composite_alpha — 1 at hole centre fading to 0 at dilation boundary
      motion_alpha    — 0 inside protected zone, ramping to 1 over feather_px beyond.
    """
    if hole_mask.max() == 0:
        H, W = hole_mask.shape[:2]
        ones = np.ones((H, W), dtype=np.float32)
        return hole_mask.copy(), np.zeros((H, W), dtype=np.float32), ones

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilation_px + 1, 2 * dilation_px + 1)
    )
    dilated = cv2.dilate(hole_mask, kernel)

    dist_inside = cv2.distanceTransform(dilated, cv2.DIST_L2, 5)
    composite_alpha = np.minimum(
        dist_inside / max(float(dilation_px), 1.0), 1.0
    ).astype(np.float32)

    protected_bin = (dilated > 0).astype(np.uint8)
    dist_from_protection = cv2.distanceTransform(1 - protected_bin, cv2.DIST_L2, 5)
    motion_alpha = np.minimum(
        dist_from_protection / max(float(feather_px), 1.0), 1.0
    ).astype(np.float32)

    return dilated, composite_alpha, motion_alpha
