"""Loop construction utilities."""

from __future__ import annotations

from typing import List

import numpy as np


def enforce_loop(frames: List[np.ndarray]) -> List[np.ndarray]:
    """Append a copy of the first frame so the GIF/MP4 wraps exactly."""
    if not frames:
        return frames
    last = frames[-1]
    first = frames[0]
    if last.shape != first.shape:
        return list(frames) + [first]
    if np.array_equal(last, first):
        return list(frames)
    return list(frames) + [first]


def make_pingpong_loop(frames: List[np.ndarray]) -> List[np.ndarray]:
    """Forward sequence followed by the reverse (boundary duplicates dropped)."""
    if not frames:
        return frames
    if len(frames) == 1:
        return list(frames)
    forward = list(frames)
    backward = list(frames[-2:0:-1])  # exclude both endpoints
    return forward + backward


def blend_loop_boundary(frames: List[np.ndarray], window: int = 4) -> List[np.ndarray]:
    """Cross-fade the last `window` frames into the first `window` to soften the wrap."""
    if not frames or window <= 0 or len(frames) <= 2 * window:
        return list(frames)

    blended = list(frames)
    for i in range(window):
        alpha = (i + 1) / (window + 1)
        end_idx = -window + i
        a = blended[end_idx].astype(np.float32)
        b = blended[i].astype(np.float32)
        mixed = (1.0 - alpha) * a + alpha * b
        blended[end_idx] = np.clip(mixed, 0, 255).astype(np.uint8)
    return blended


def temporal_smooth_frames(frames: List[np.ndarray], radius: int = 1) -> List[np.ndarray]:
    """Average each frame with ±radius neighbours to reduce high-freq flicker."""
    if not frames or radius <= 0:
        return list(frames)

    n = len(frames)
    smoothed: List[np.ndarray] = []
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        stack = np.stack([frames[j].astype(np.float32) for j in range(lo, hi)], axis=0)
        avg = stack.mean(axis=0)
        smoothed.append(np.clip(avg, 0, 255).astype(np.uint8))
    return smoothed
