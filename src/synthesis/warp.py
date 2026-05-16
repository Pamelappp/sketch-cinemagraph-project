"""Symmetric forward-splatting warp for seamless cinemagraph loops (paper §3.4)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


_BORDER_MODES = {
    "reflect": cv2.BORDER_REFLECT,
    "replicate": cv2.BORDER_REPLICATE,
    "wrap": cv2.BORDER_WRAP,
    "constant": cv2.BORDER_CONSTANT,
}


def warp_frames(
    image: np.ndarray,
    flow: np.ndarray,
    mask: np.ndarray,
    num_frames: int = 60,
    motion_scale: float = 1.0,
    border_mode: str = "reflect",
    blend_window: float = 0.30,       # ignored in symmetric mode (kept for compat)
    method: str = "symmetric_splat",   # "symmetric_splat" or "backward"
) -> List[np.ndarray]:
    """Animate fluid pixels of *image* along *flow* for num_frames; bg is fixed."""
    if num_frames < 2:
        return [image.copy()]

    if method == "backward":
        return _backward_warp_frames(
            image, flow, mask, num_frames, motion_scale, border_mode, blend_window,
        )

    return _symmetric_splat_frames(
        image, flow, mask, num_frames, motion_scale,
    )


def warp_single_frame(
    image: np.ndarray,
    flow: np.ndarray,
    mask: np.ndarray,
    t: int,
    num_frames: int,
    motion_scale: float = 1.0,
    border_mode: str = "reflect",
) -> np.ndarray:
    """Render a single frame at index *t*."""
    frames = warp_frames(
        image, flow, mask, num_frames=num_frames,
        motion_scale=motion_scale, border_mode=border_mode,
    )
    return frames[max(0, min(t, len(frames) - 1))]


def _symmetric_splat_frames(
    image: np.ndarray,
    flow: np.ndarray,
    mask: np.ndarray,
    num_frames: int,
    motion_scale: float,
) -> List[np.ndarray]:
    h, w = image.shape[:2]
    flow_scaled = np.asarray(flow, dtype=np.float32) * float(motion_scale)

    mask_2d = mask if mask.ndim == 2 else mask[..., 0]
    fluid = mask_2d > 0

    # Importance for softmax z-buffer: faster-moving pixels occlude slower ones.
    importance = np.linalg.norm(flow_scaled, axis=-1).astype(np.float32)
    imp_max = importance.max()
    if imp_max > 1e-3:
        importance = importance / imp_max * 3.0

    src_float = image.astype(np.float32)
    fluid_src = src_float.copy()
    fluid_src[~fluid] = 0.0
    fluid_imp = importance.copy()
    fluid_imp[~fluid] = -1e6  # very low importance → won't win softmax

    frames: List[np.ndarray] = []

    for t in range(num_frames):
        phase = t / num_frames

        # Forward stream: splat by +phase * F; backward stream: splat by -(1-phase) * F.
        # At phase=0 forward is the untouched source; at phase≈1 backward is. The
        # alpha=phase blend keeps motion unidirectional across the loop boundary.
        fwd_flow = flow_scaled * phase
        fwd_rgb, fwd_weight = _forward_splat(fluid_src, fwd_flow, fluid_imp, h, w)

        bwd_flow = flow_scaled * -(1.0 - phase)
        bwd_rgb, bwd_weight = _forward_splat(fluid_src, bwd_flow, fluid_imp, h, w)

        alpha = phase

        blended = np.zeros((h, w, 3), dtype=np.float32)
        denom = np.zeros((h, w, 1), dtype=np.float32)

        fwd_w = (1.0 - alpha) * fwd_weight[..., None]
        bwd_w = alpha * bwd_weight[..., None]
        total_w = fwd_w + bwd_w

        valid = total_w[..., 0] > 1e-8
        blended[valid] = (fwd_w[valid] * fwd_rgb[valid] + bwd_w[valid] * bwd_rgb[valid]) / total_w[valid]

        # Fill holes where neither stream contributed — use source pixels.
        blended[~valid] = src_float[~valid]

        # Inpaint small remaining holes inside the fluid region.
        splatted_fluid = valid & fluid
        missing_fluid = (~valid) & fluid
        if missing_fluid.any():
            blended = _inpaint_holes(blended, splatted_fluid, missing_fluid)

        # Compose: background unchanged, fluid region from splatting.
        out = image.copy()
        out[fluid] = np.clip(blended[fluid], 0, 255).astype(np.uint8)
        frames.append(out)

    return frames


def _forward_splat(
    src: np.ndarray,
    flow: np.ndarray,
    importance: np.ndarray,
    h: int,
    w: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward-splat *src* along *flow* with a softmax z-buffer; return (rgb, weight)."""
    c = src.shape[2] if src.ndim == 3 else 1

    ys, xs = np.meshgrid(np.arange(h, dtype=np.float32),
                         np.arange(w, dtype=np.float32), indexing="ij")

    dst_x = xs + flow[..., 0]
    dst_y = ys + flow[..., 1]

    # Clamp exponent to avoid overflow; relative ranking is what matters.
    safe_z = np.clip(importance, -20.0, 20.0)
    exp_z = np.exp(safe_z)

    x0 = np.floor(dst_x).astype(np.int32)
    y0 = np.floor(dst_y).astype(np.int32)
    fx = dst_x - x0.astype(np.float32)
    fy = dst_y - y0.astype(np.float32)

    corners = [
        (x0,     y0,     (1 - fx) * (1 - fy)),
        (x0 + 1, y0,     fx * (1 - fy)),
        (x0,     y0 + 1, (1 - fx) * fy),
        (x0 + 1, y0 + 1, fx * fy),
    ]

    # Buffers are 1-D and contiguous so np.add.at writes in place (a non-contiguous
    # .ravel() would copy and the accumulation would silently no-op).
    n_pixels = h * w
    rgb_flat = [np.zeros(n_pixels, dtype=np.float64) for _ in range(c)]
    w_flat = np.zeros(n_pixels, dtype=np.float64)

    src_64 = src.astype(np.float64)
    src_ch_flat = [src_64[..., ch].ravel().copy() for ch in range(c)]

    exp_z_flat = exp_z.ravel()

    for cx, cy, bilinear_w in corners:
        valid = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
        flat_src = valid.ravel()  # indices into flattened source
        flat_dst = (cy[valid] * w + cx[valid]).astype(np.intp)
        sw = (bilinear_w[valid] * exp_z_flat[valid.ravel()]).astype(np.float64)

        for ch in range(c):
            np.add.at(rgb_flat[ch], flat_dst, sw * src_ch_flat[ch][valid.ravel()])
        np.add.at(w_flat, flat_dst, sw)

    # Normalise.
    nonzero = w_flat > 1e-10
    for ch in range(c):
        rgb_flat[ch][nonzero] /= w_flat[nonzero]
        rgb_flat[ch][~nonzero] = 0.0

    rgb_acc = np.stack([ch_arr.reshape(h, w) for ch_arr in rgb_flat], axis=-1)
    return rgb_acc.astype(np.float32), w_flat.reshape(h, w).astype(np.float32)


def _inpaint_holes(
    image: np.ndarray,
    valid: np.ndarray,
    holes: np.ndarray,
    iterations: int = 3,
) -> np.ndarray:
    """Average 4-connected neighbours to fill the 1–2 px gaps from forward splatting."""
    result = image.copy()
    remaining = holes.copy()

    for _ in range(iterations):
        if not remaining.any():
            break
        padded = np.pad(result, ((1, 1), (1, 1), (0, 0)), mode="edge")
        padded_valid = np.pad(valid | (~remaining), ((1, 1), (1, 1)), mode="constant",
                              constant_values=False)

        neighbour_sum = (
            padded[:-2, 1:-1] +
            padded[2:, 1:-1] +
            padded[1:-1, :-2] +
            padded[1:-1, 2:]
        )
        neighbour_count = (
            padded_valid[:-2, 1:-1].astype(np.float32) +
            padded_valid[2:, 1:-1].astype(np.float32) +
            padded_valid[1:-1, :-2].astype(np.float32) +
            padded_valid[1:-1, 2:].astype(np.float32)
        )

        fillable = remaining & (neighbour_count > 0)
        if not fillable.any():
            break

        nc = neighbour_count[fillable][..., None]
        result[fillable] = neighbour_sum[fillable] / nc
        remaining[fillable] = False
        valid[fillable] = True

    return result


def _backward_warp_frames(
    image: np.ndarray,
    flow: np.ndarray,
    mask: np.ndarray,
    num_frames: int,
    motion_scale: float,
    border_mode: str,
    blend_window: float,
) -> List[np.ndarray]:
    """Legacy backward-warp + cubic-dissolve fallback (kept for compatibility)."""
    cv_border = _BORDER_MODES.get(str(border_mode).lower(), cv2.BORDER_REFLECT)
    h, w = image.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    grid_x = grid_x.astype(np.float32)
    grid_y = grid_y.astype(np.float32)

    flow_arr = np.asarray(flow, dtype=np.float32) * float(motion_scale)
    Dx = flow_arr[..., 0]
    Dy = flow_arr[..., 1]

    mask_2d = mask if mask.ndim == 2 else mask[..., 0]
    fluid_region = mask_2d > 0

    forward_frames: List[np.ndarray] = []
    for t in range(num_frames):
        phase = t / (num_frames - 1)
        map_x = (grid_x - phase * Dx).astype(np.float32)
        map_y = (grid_y - phase * Dy).astype(np.float32)
        warped = cv2.remap(
            image, map_x, map_y,
            interpolation=cv2.INTER_LINEAR, borderMode=cv_border,
        )
        out = image.copy()
        out[fluid_region] = warped[fluid_region]
        forward_frames.append(out)

    blend_window = float(np.clip(blend_window, 0.0, 0.5))
    W = int(round(num_frames * blend_window))
    if blend_window <= 0.0 or W <= 1:
        return forward_frames

    final = list(forward_frames)
    source = forward_frames[0].astype(np.float32)

    for i in range(W):
        late_idx = num_frames - W + i
        ratio = i / (W - 1)
        alpha = ratio ** 3
        late = forward_frames[late_idx].astype(np.float32)
        blended = (1.0 - alpha) * late + alpha * source
        final[late_idx] = np.clip(blended, 0, 255).astype(np.uint8)

    final[-1] = forward_frames[0].copy()
    return final
