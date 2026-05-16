"""Functions for loop construction and temporal smoothing."""

import numpy as np


def enforce_loop(frames):
    """Copy frames[0] over frames[-1] so the loop boundary is exact."""
    if len(frames) == 0:
        return []

    loopable_frames = [frame.copy() for frame in frames]
    if len(loopable_frames) > 1:
        loopable_frames[-1] = loopable_frames[0].copy()
    return loopable_frames


def blend_loop_boundary(frames):
    """Cross-fade matching frames near the start and end to hide the loop seam."""
    if len(frames) <= 2:
        return frames

    blended_frames = [frame.copy().astype(np.float32) for frame in frames]

    blend_window = min(5, len(frames) // 2)
    for i in range(blend_window):
        alpha = (i + 1) / (blend_window + 1)
        start_idx = i
        end_idx = len(frames) - blend_window + i

        start_frame = blended_frames[start_idx]
        end_frame = blended_frames[end_idx]

        blended = (1 - alpha) * start_frame + alpha * end_frame
        blended_frames[start_idx] = blended
        blended_frames[end_idx] = blended

    return [np.clip(frame, 0, 255).astype(np.uint8) for frame in blended_frames]


def temporal_smooth_frames(frames):
    """Apply [0.25, 0.5, 0.25] temporal averaging; first/last frames are left alone."""
    if len(frames) <= 2:
        return frames

    smoothed_frames = []
    num_frames = len(frames)

    for i in range(num_frames):
        current = frames[i].astype(np.float32)

        if i == 0 or i == num_frames - 1:
            smoothed_frames.append(frames[i].copy())
            continue

        prev_frame = frames[i - 1].astype(np.float32)
        next_frame = frames[i + 1].astype(np.float32)

        smoothed = 0.25 * prev_frame + 0.5 * current + 0.25 * next_frame
        smoothed = np.clip(smoothed, 0, 255).astype(np.uint8)

        smoothed_frames.append(smoothed)

    return smoothed_frames
