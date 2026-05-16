"""Visualization helpers for debugging intermediate results in the pipeline.

NOTE: Member-B-side stub written so demo runs can produce qualitative
debug images. Member C is expected to extend these for the final report
(side-by-side comparison panels, paper-style figure layouts, etc.).
"""

from __future__ import annotations

import cv2
import numpy as np


def visualize_mask(mask: np.ndarray) -> np.ndarray:
    """Render a binary mask as a 3-channel uint8 image for inspection."""
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    binary = (array > 0).astype(np.uint8) * 255
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


def visualize_motion_field(flow: np.ndarray) -> np.ndarray:
    """
    Convert a dense motion field into the standard HSV color visualization.

    Hue encodes flow direction (angle), value encodes magnitude.
    """
    flow_array = np.asarray(flow, dtype=np.float32)
    if flow_array.ndim != 3 or flow_array.shape[2] != 2:
        raise ValueError(f"Expected H x W x 2 flow array, got shape {flow_array.shape}")

    fx = flow_array[..., 0]
    fy = flow_array[..., 1]
    magnitude, angle = cv2.cartToPolar(fx, fy)

    hsv = np.zeros((flow_array.shape[0], flow_array.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (angle * 180.0 / np.pi / 2.0).astype(np.uint8)
    hsv[..., 1] = 255
    if magnitude.max() > 0:
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        hsv[..., 2] = 0

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def visualize_pipeline_summary(
    scene_out: dict,
    mask_out: dict,
    motion_out: dict,
    target_size: int = 256,
) -> np.ndarray:
    """
    Build a 1-row summary panel: stylized image | final fluid mask | motion field.

    Each tile is resized to ``target_size`` x ``target_size`` and the panels
    are concatenated horizontally with a thin white separator.
    """
    stylized = _resize(scene_out["stylized_image"], target_size)
    mask_vis = _resize(visualize_mask(mask_out["final_fluid_mask"]), target_size)
    flow_vis = _resize(visualize_motion_field(motion_out["dense_motion_field"]), target_size)

    sep = np.full((target_size, 4, 3), 255, dtype=np.uint8)
    return np.concatenate([stylized, sep, mask_vis, sep, flow_vis], axis=1)


def _resize(image: np.ndarray, size: int) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    return cv2.resize(array, (size, size), interpolation=cv2.INTER_AREA)
