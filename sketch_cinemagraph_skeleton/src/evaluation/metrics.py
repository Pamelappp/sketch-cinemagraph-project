"""Evaluation metrics for motion quality and cinemagraph output quality."""

import numpy as np


def compute_motion_smoothness(flow, mask):
    """
    Measure whether the dense motion field changes smoothly inside valid fluid regions.

    Args:
        flow: H x W x 2 dense motion field
        mask: H x W binary / grayscale fluid mask

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

    # Neighbor differences in x and y directions
    diff_x_fx = fx[:, 1:] - fx[:, :-1]
    diff_y_fx = fx[1:, :] - fx[:-1, :]

    diff_x_fy = fy[:, 1:] - fy[:, :-1]
    diff_y_fy = fy[1:, :] - fy[:-1, :]

    # Valid neighbor masks
    valid_x = valid[:, 1:] & valid[:, :-1]
    valid_y = valid[1:, :] & valid[:-1, :]

    values = []

    if np.any(valid_x):
        values.append(diff_x_fx[valid_x] ** 2)
        values.append(diff_x_fy[valid_x] ** 2)

    if np.any(valid_y):
        values.append(diff_y_fx[valid_y] ** 2)
        values.append(diff_y_fy[valid_y] ** 2)

    if len(values) == 0:
        return 0.0

    smoothness_score = float(np.mean(np.concatenate(values)))
    return smoothness_score


def compute_loop_consistency(frames):
    """
    Measure how similar the final frame is to the first frame for looping quality.

    Args:
        frames: list of H x W x C numpy arrays

    Returns:
        loop_score: scalar, lower means better loop consistency
    """
    if len(frames) < 2:
        return 0.0

    first_frame = frames[0].astype(np.float32)
    last_frame = frames[-1].astype(np.float32)

    mse = np.mean((first_frame - last_frame) ** 2)
    loop_score = float(mse)

    return loop_score


def compute_mask_leakage(frames, mask):
    """
    Estimate whether motion artifacts spill into background regions outside the mask.

    Args:
        frames: list of H x W x C numpy arrays
        mask: H x W binary / grayscale fluid mask

    Returns:
        leakage_score: scalar, lower means less leakage outside the valid motion region
    """
    if len(frames) < 2:
        return 0.0

    if mask.ndim == 3:
        mask_2d = mask[:, :, 0]
    else:
        mask_2d = mask

    background = mask_2d == 0

    if np.sum(background) == 0:
        return 0.0

    first_frame = frames[0].astype(np.float32)

    background_changes = []

    for frame in frames[1:]:
        frame = frame.astype(np.float32)

        if frame.ndim == 3:
            diff = np.mean(np.abs(frame - first_frame), axis=2)
        else:
            diff = np.abs(frame - first_frame)

        background_changes.append(diff[background])

    if len(background_changes) == 0:
        return 0.0

    leakage_score = float(np.mean(np.concatenate(background_changes)))
    return leakage_score
