"""Post-processing for dense motion fields."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


_GAUSSIAN_SIGMA = 1.5
_DEFAULT_MAX_MAGNITUDE = 1.5


def smooth_motion_field(flow, mask, max_magnitude: float | None = None):
    """
    Smooth the dense motion field while keeping motion confined to the mask.

    Args:
        flow:          H x W x 2 float32 dense flow
        mask:          H x W (or H x W x 1) uint8 fluid mask
        max_magnitude: clip per-pixel speed to this value (pixels/frame);
                       if None uses the module default (1.5)
    """
    flow = np.asarray(flow, dtype=np.float32)
    smoothed = np.empty_like(flow)
    smoothed[..., 0] = gaussian_filter(flow[..., 0], sigma=_GAUSSIAN_SIGMA)
    smoothed[..., 1] = gaussian_filter(flow[..., 1], sigma=_GAUSSIAN_SIGMA)

    smoothed = enforce_mask_boundary(smoothed, mask)
    effective_max = max_magnitude if max_magnitude is not None else _DEFAULT_MAX_MAGNITUDE
    smoothed = normalize_motion_magnitude(smoothed, max_magnitude=effective_max)
    return smoothed


def apply_motion_protection_to_flow(
    flow: np.ndarray, motion_alpha: np.ndarray
) -> np.ndarray:
    """
    Attenuate flow near static foreground objects.

    Args:
        flow:         H x W x 2 float32 dense flow
        motion_alpha: H x W float32, 0.0 = fully protected (no flow),
                                     1.0 = open water (full flow)

    Returns:
        Protected flow: zero near protected objects, full elsewhere.
    """
    return (np.asarray(flow, dtype=np.float32) * motion_alpha[..., None]).astype(np.float32)


def enforce_mask_boundary(flow, mask):
    """Zero-out motion outside the valid fluid mask region."""
    flow = np.asarray(flow, dtype=np.float32)
    mask_2d = mask if mask.ndim == 2 else mask[:, :, 0]
    mask_factor = (mask_2d > 0).astype(np.float32)[..., None]
    return flow * mask_factor


def normalize_motion_magnitude(flow, max_magnitude=None):
    """
    Clip per-pixel motion vectors so their magnitude does not exceed
    ``max_magnitude`` while preserving direction.
    """
    flow = np.asarray(flow, dtype=np.float32)
    if max_magnitude is None or max_magnitude <= 0:
        return flow

    magnitude = np.linalg.norm(flow, axis=-1)
    safe_magnitude = np.where(magnitude > _eps_like(magnitude), magnitude, 1.0)
    scale = np.minimum(1.0, max_magnitude / safe_magnitude)
    scale = np.where(magnitude > 0, scale, 1.0)
    return flow * scale[..., None]


def _eps_like(array):
    return np.finfo(array.dtype).eps if np.issubdtype(array.dtype, np.floating) else 1e-6
