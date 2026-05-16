"""Evaluation metrics for motion quality and cinemagraph output quality."""

from __future__ import annotations

from typing import List

import numpy as np


def compute_motion_smoothness(flow: np.ndarray, mask: np.ndarray) -> float:
    """Mean ‖∇flow‖ inside the fluid mask (lower = smoother)."""
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"Expected H x W x 2 flow, got shape {flow.shape}")
    mask_2d = np.asarray(mask)
    if mask_2d.ndim == 3:
        mask_2d = mask_2d[..., 0]
    fluid = mask_2d > 0
    if not fluid.any():
        return 0.0

    grad_y = np.diff(flow, axis=0)
    grad_x = np.diff(flow, axis=1)
    fluid_y = fluid[1:, :] & fluid[:-1, :]
    fluid_x = fluid[:, 1:] & fluid[:, :-1]

    energy_y = np.linalg.norm(grad_y[fluid_y], axis=-1) if fluid_y.any() else np.zeros(0)
    energy_x = np.linalg.norm(grad_x[fluid_x], axis=-1) if fluid_x.any() else np.zeros(0)
    if energy_y.size + energy_x.size == 0:
        return 0.0
    return float((energy_x.sum() + energy_y.sum()) / (energy_x.size + energy_y.size))


def compute_loop_consistency(frames: List[np.ndarray]) -> float:
    """MSE between the first and last frame (0 = perfect loop)."""
    if not frames or len(frames) < 2:
        return 0.0
    first = np.asarray(frames[0], dtype=np.float32)
    last = np.asarray(frames[-1], dtype=np.float32)
    if first.shape != last.shape:
        return float("inf")
    return float(np.mean((first - last) ** 2))


def compute_mask_leakage(frames: List[np.ndarray], mask: np.ndarray) -> float:
    """Mean temporal std of background pixels (0 = static background)."""
    if not frames:
        return 0.0
    stack = np.stack([np.asarray(f, dtype=np.float32) for f in frames], axis=0)
    mask_2d = np.asarray(mask)
    if mask_2d.ndim == 3:
        mask_2d = mask_2d[..., 0]
    background = mask_2d == 0
    if not background.any():
        return 0.0

    std_t = stack.std(axis=0)
    if std_t.ndim == 3:
        std_t = std_t.mean(axis=-1)
    return float(std_t[background].mean())
