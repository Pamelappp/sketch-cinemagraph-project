"""Post-processing for dense motion fields."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


_GAUSSIAN_SIGMA = 5.0
_DEFAULT_MAX_MAGNITUDE = 1.5


def smooth_motion_field(flow, mask):
    """
    Smooth the dense motion field while keeping motion confined to the mask.
    """
    flow = np.asarray(flow, dtype=np.float32)
    smoothed = np.empty_like(flow)
    smoothed[..., 0] = gaussian_filter(flow[..., 0], sigma=_GAUSSIAN_SIGMA)
    smoothed[..., 1] = gaussian_filter(flow[..., 1], sigma=_GAUSSIAN_SIGMA)

    smoothed = enforce_mask_boundary(smoothed, mask)
    smoothed = normalize_motion_magnitude(smoothed, max_magnitude=_DEFAULT_MAX_MAGNITUDE)
    return smoothed


def enforce_mask_boundary(flow, mask):
    """
    Zero-out motion outside the valid fluid mask region.
    """
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
