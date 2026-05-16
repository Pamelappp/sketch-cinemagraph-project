"""Evaluation metrics for motion quality and cinemagraph output quality."""

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
    """True when the foreground ratio is in [0.001, 0.8] — otherwise empty or inverted."""
    total = mask.size
    if total == 0:
        return False
    ratio = float((mask > 0).sum()) / total
    return 0.001 <= ratio <= 0.8


def compute_motion_smoothness(flow, mask):
    """Mean squared spatial gradient of the flow inside the mask (lower = smoother)."""
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
    """MSE between the first and last frame (0 = perfect loop)."""
    if len(frames) < 2:
        return 0.0
    first = frames[0].astype(np.float32)
    last = frames[-1].astype(np.float32)
    return float(np.mean((first - last) ** 2))


def compute_mask_leakage(frames, mask):
    """Average pixel change in the static background (outside the fluid mask)."""
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


def compute_psnr(img1, img2):
    """PSNR in dB between two uint8 images (inf when identical)."""
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0.0:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def compute_ms_ssim_loop(frames):
    """SSIM between the first and last frame (1.0 = perfect loop)."""
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
    """Average PSNR between consecutive frame pairs (higher = less flicker)."""
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


def compute_flow_aepe(pred_flow, gt_flow, mask=None):
    """Average end-point error between predicted and ground-truth flow."""
    diff = pred_flow.astype(np.float32) - gt_flow.astype(np.float32)
    epe = np.linalg.norm(diff, axis=-1)

    if mask is not None:
        m = mask[:, :, 0] > 0 if mask.ndim == 3 else mask > 0
        if m.sum() > 0:
            return float(epe[m].mean())
        return 0.0
    return float(epe.mean())


def compute_flow_mse(pred_flow, gt_flow, mask=None):
    """MSE between predicted and ground-truth flow."""
    diff = pred_flow.astype(np.float32) - gt_flow.astype(np.float32)
    mse_map = (diff ** 2).sum(axis=-1)

    if mask is not None:
        m = mask[:, :, 0] > 0 if mask.ndim == 3 else mask > 0
        if m.sum() > 0:
            return float(mse_map[m].mean())
        return 0.0
    return float(mse_map.mean())
