"""Functions for loop construction and temporal smoothing."""

import numpy as np


def enforce_loop(frames):
    """
    Adjust synthesized frames so the output video forms a seamless loop.

    Args:
        frames: list of H x W x C numpy arrays

    Returns:
        loopable_frames: list of frames with first/last boundary aligned
    """
    if len(frames) == 0:
        return []

    # Make a copy so the input list is not modified in-place
    loopable_frames = [frame.copy() for frame in frames]

    # Force the last frame to match the first frame exactly
    if len(loopable_frames) > 1:
        loopable_frames[-1] = loopable_frames[0].copy()

    return loopable_frames


def blend_loop_boundary(frames):
    """
    Blend the beginning and end of the sequence to reduce visible looping artifacts.

    Args:
        frames: list of H x W x C numpy arrays

    Returns:
        blended_frames: list of frames with smoother start/end transition
    """
    if len(frames) <= 2:
        return frames

    blended_frames = [frame.copy().astype(np.float32) for frame in frames]

    # Use a small blending window near the sequence boundary
    blend_window = min(5, len(frames) // 2)

    for i in range(blend_window):
        alpha = (i + 1) / (blend_window + 1)

        # Blend early frames with corresponding late frames
        start_idx = i
        end_idx = len(frames) - blend_window + i

        start_frame = blended_frames[start_idx]
        end_frame = blended_frames[end_idx]

        blended = (1 - alpha) * start_frame + alpha * end_frame
        blended_frames[start_idx] = blended
        blended_frames[end_idx] = blended

    # Convert back to uint8 for image/video export
    blended_frames = [np.clip(frame, 0, 255).astype(np.uint8) for frame in blended_frames]

    return blended_frames


def temporal_smooth_frames(frames):
    """
    Apply temporal smoothing to reduce flicker and inconsistent motion across frames.

    Args:
        frames: list of H x W x C numpy arrays

    Returns:
        smoothed_frames: temporally stabilized frame list
    """
    if len(frames) <= 2:
        return frames

    smoothed_frames = []
    num_frames = len(frames)

    for i in range(num_frames):
        current = frames[i].astype(np.float32)

        # Keep the first and last frame unchanged
        if i == 0 or i == num_frames - 1:
            smoothed_frames.append(frames[i].copy())
            continue

        prev_frame = frames[i - 1].astype(np.float32)
        next_frame = frames[i + 1].astype(np.float32)

        # Simple temporal averaging with center frame weighted more
        smoothed = 0.25 * prev_frame + 0.5 * current + 0.25 * next_frame
        smoothed = np.clip(smoothed, 0, 255).astype(np.uint8)

        smoothed_frames.append(smoothed)

    return smoothed_frames
