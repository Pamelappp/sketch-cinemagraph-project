"""Evaluation metrics for motion quality and cinemagraph output quality.

Original metrics (motion_smoothness, loop_consistency, mask_leakage) are kept
unchanged.  New standard metrics added to match the paper's evaluation protocol:

  Frame quality  : compute_psnr, compute_ms_ssim
  Video quality  : compute_temporal_consistency  (avg inter-frame PSNR)
  Flow quality   : compute_flow_aepe, compute_flow_mse  (need GT flow)
"""

from __future__ import annotations

import math

import numpy as np


def sanitize_for_json(metrics: dict) -> dict:
    """Replace float inf/nan with None so json.dump produces valid JSON."""
    out = {}
    for k, v in metrics.items():
        if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
            out[k] = None
        else:
            out[k] = v
    return out


def compute_mask_valid(mask) -> bool:
    """
    Return True when the mask's foreground ratio is plausible (0.1 % – 80 %).

    Outside this range the mask is likely empty (intersection failed) or inverted
    (almost everything marked fluid).
    """
    total = mask.size
    if total == 0:
        return False
    ratio = float((mask > 0).sum()) / total
    return 0.001 <= ratio <= 0.8


# ── Original metrics ──────────────────────────────────────────────────────────

def compute_motion_smoothness(flow, mask):
    """
    Measure whether the dense motion field changes smoothly inside valid fluid regions.

    Returns:
        smoothness_score: scalar, lower means smoother flow
    """
    if mask.ndim == 3:
        mask_2d = mask[:, :, 0]
    else:
        mask_2d = mask

    valid = mask_2d > 0
    if np.sum(valid) == 0:
        return 0.0

    fx = flow[:, :, 0]
    fy = flow[:, :, 1]

    diff_x_fx = fx[:, 1:] - fx[:, :-1]
    diff_y_fx = fx[1:, :] - fx[:-1, :]
    diff_x_fy = fy[:, 1:] - fy[:, :-1]
    diff_y_fy = fy[1:, :] - fy[:-1, :]

    valid_x = valid[:, 1:] & valid[:, :-1]
    valid_y = valid[1:, :] & valid[:-1, :]

    values = []
    if np.any(valid_x):
        values.append(diff_x_fx[valid_x] ** 2)
        values.append(diff_x_fy[valid_x] ** 2)
    if np.any(valid_y):
        values.append(diff_y_fx[valid_y] ** 2)
        values.append(diff_y_fy[valid_y] ** 2)

    if not values:
        return 0.0
    return float(np.mean(np.concatenate(values)))


def compute_loop_consistency(frames):
    """
    MSE between the first and last frame.

    Returns:
        loop_score: scalar, lower means better loop consistency (0 = perfect)
    """
    if len(frames) < 2:
        return 0.0
    first = frames[0].astype(np.float32)
    last = frames[-1].astype(np.float32)
    return float(np.mean((first - last) ** 2))


def compute_mask_leakage(frames, mask):
    """
    Average pixel change in the static background (outside the fluid mask).

    Returns:
        leakage_score: scalar, lower means less leakage
    """
    if len(frames) < 2:
        return 0.0

    mask_2d = mask[:, :, 0] if mask.ndim == 3 else mask
    background = mask_2d == 0
    if np.sum(background) == 0:
        return 0.0

    first = frames[0].astype(np.float32)
    diffs = []
    for frame in frames[1:]:
        diff = np.mean(np.abs(frame.astype(np.float32) - first), axis=2)
        diffs.append(diff[background])

    if not diffs:
        return 0.0
    return float(np.mean(np.concatenate(diffs)))


# ── Standard image / video quality metrics ────────────────────────────────────

def compute_psnr(img1, img2):
    """
    Peak Signal-to-Noise Ratio between two uint8 images (higher is better).

    Args:
        img1, img2: H x W x C uint8 numpy arrays

    Returns:
        psnr: float in dB; returns inf when the images are identical
    """
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0.0:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def compute_ms_ssim_loop(frames):
    """
    Structural Similarity (SSIM) between the first and last frame.

    Uses skimage.metrics.structural_similarity with multichannel support.
    Higher is better (1.0 = perfect loop).

    Returns:
        ssim_score: float in [-1, 1]
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        return float("nan")

    if len(frames) < 2:
        return 1.0

    f0 = frames[0].astype(np.float32)
    fn = frames[-1].astype(np.float32)
    score = ssim(f0, fn, channel_axis=2, data_range=255.0)
    return float(score)


def compute_temporal_consistency(frames):
    """
    Average PSNR between consecutive frame pairs (higher is better).

    Measures how smoothly the video changes over time; a low value hints
    at flickering or abrupt motion.

    Returns:
        avg_psnr: float in dB
    """
    if len(frames) < 2:
        return float("inf")

    psnrs = [
        compute_psnr(frames[i], frames[i + 1])
        for i in range(len(frames) - 1)
        if not (np.isinf(compute_psnr(frames[i], frames[i + 1])))
    ]
    if not psnrs:
        return float("inf")
    return float(np.mean(psnrs))


# ── Flow quality metrics (require ground-truth flow) ──────────────────────────

def compute_flow_aepe(pred_flow, gt_flow, mask=None):
    """
    Average End-Point Error between predicted and ground-truth flow (lower is better).

    Matches the AEPE metric reported in Table 1 of the paper.

    Args:
        pred_flow: H x W x 2 predicted flow (float32)
        gt_flow:   H x W x 2 ground-truth flow (float32)
        mask:      optional H x W or H x W x 1 mask; if given, only masked
                   pixels are included in the average

    Returns:
        aepe: float
    """
    diff = pred_flow.astype(np.float32) - gt_flow.astype(np.float32)
    epe = np.linalg.norm(diff, axis=-1)  # H x W

    if mask is not None:
        m = mask[:, :, 0] > 0 if mask.ndim == 3 else mask > 0
        if m.sum() > 0:
            return float(epe[m].mean())
        return 0.0
    return float(epe.mean())


def compute_flow_mse(pred_flow, gt_flow, mask=None):
    """
    Mean Squared Error between predicted and ground-truth flow (lower is better).

    Matches the MSE metric reported in Table 1 of the paper.

    Args:
        pred_flow: H x W x 2 predicted flow (float32)
        gt_flow:   H x W x 2 ground-truth flow (float32)
        mask:      optional mask (same convention as compute_flow_aepe)

    Returns:
        mse: float
    """
    diff = pred_flow.astype(np.float32) - gt_flow.astype(np.float32)
    mse_map = (diff ** 2).sum(axis=-1)  # sum dx² + dy², H x W

    if mask is not None:
        m = mask[:, :, 0] > 0 if mask.ndim == 3 else mask > 0
        if m.sum() > 0:
            return float(mse_map[m].mean())
        return 0.0
    return float(mse_map.mean())
