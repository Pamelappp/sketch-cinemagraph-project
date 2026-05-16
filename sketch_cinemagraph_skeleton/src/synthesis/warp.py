"""Frame warping via Euler integration plus ping-pong loop for cinemagraph synthesis."""

from pathlib import Path

import numpy as np
import cv2


def warp_frames(
    image,
    flow,
    mask,
    num_frames: int = 120,
    safe_moving_mask=None,
    composite_alpha=None,
    debug_dir=None,
):
    """Build a ping-pong looping frame sequence by Euler-integrating *flow* over *image*.

    safe_moving_mask restricts warping to open water; composite_alpha (1=protect, 0=warp)
    restores static foreground after every step. Use an even num_frames.
    """
    H, W = image.shape[:2]

    active_mask = safe_moving_mask if safe_moving_mask is not None else mask
    active_2d = active_mask[:, :, 0] if active_mask.ndim == 3 else active_mask
    fluid = active_2d > 0

    grid_y, grid_x = np.mgrid[0:H, 0:W].astype(np.float32)
    half = num_frames // 2
    _debug_dir = Path(debug_dir) if debug_dir is not None else None

    forward_frames = []
    current_disp = np.zeros((H, W, 2), dtype=np.float32)

    for t in range(half + 1):
        frame = _apply_displacement(
            image, current_disp, fluid, grid_x, grid_y,
            composite_alpha=composite_alpha,
        )
        forward_frames.append(frame)

        if _debug_dir is not None and t == half:
            frame_pre = _apply_displacement(
                image, current_disp, fluid, grid_x, grid_y, composite_alpha=None
            )
            cv2.imwrite(
                str(_debug_dir / "debug_frame_030_before_composite.png"),
                cv2.cvtColor(frame_pre, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                str(_debug_dir / "debug_frame_030_after_composite.png"),
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            )

        if t < half:
            current_disp = _euler_step(flow, current_disp, grid_x, grid_y, W, H)

    # Ping-pong: forward[0..half] || forward[half-1..1] → 2*half frames.
    frames = forward_frames + forward_frames[-2:0:-1]
    return frames


def _euler_step(flow, current_disp, grid_x, grid_y, W, H):
    # Sample flow at the *current* displaced position so steps compound correctly.
    sample_x = np.clip(grid_x + current_disp[:, :, 0], 0, W - 1).astype(np.float32)
    sample_y = np.clip(grid_y + current_disp[:, :, 1], 0, H - 1).astype(np.float32)

    fx = cv2.remap(
        flow[:, :, 0], sample_x, sample_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    fy = cv2.remap(
        flow[:, :, 1], sample_x, sample_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    return current_disp + np.stack([fx, fy], axis=-1)


def _apply_displacement(image, disp, fluid_mask, grid_x, grid_y, composite_alpha=None):
    # Backward warp (negative disp): content travels WITH the flow direction.
    map_x = (grid_x - disp[:, :, 0]).astype(np.float32)
    map_y = (grid_y - disp[:, :, 1]).astype(np.float32)

    warped = cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    frame = image.copy()
    frame[fluid_mask] = warped[fluid_mask]

    if composite_alpha is not None:
        alpha3 = composite_alpha[:, :, None]
        frame = (
            frame.astype(np.float32) * (1.0 - alpha3)
            + image.astype(np.float32) * alpha3
        )
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    return frame
