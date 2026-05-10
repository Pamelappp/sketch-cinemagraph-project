"""Visualization helpers for debugging intermediate results in the pipeline."""

import cv2
import numpy as np


def visualize_mask(mask):
    """
    Render a mask for qualitative inspection.

    Args:
        mask: H x W binary / grayscale mask

    Returns:
        vis_mask: H x W x 3 uint8 visualization image
    """
    if mask.ndim == 3:
        mask_2d = mask[:, :, 0]
    else:
        mask_2d = mask

    # Normalize to visible range
    mask_uint8 = np.clip(mask_2d, 0, 255).astype(np.uint8)

    # If mask is binary 0/1, scale it up to 0/255
    if mask_uint8.max() <= 1:
        mask_uint8 = mask_uint8 * 255

    # Convert grayscale mask to 3-channel image for saving/display
    vis_mask = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)

    return vis_mask


def visualize_motion_field(flow):
    """
    Convert a dense motion field into a color visualization.

    Args:
        flow: H x W x 2 dense motion field

    Returns:
        flow_vis: H x W x 3 uint8 BGR color map
    """
    fx = flow[:, :, 0]
    fy = flow[:, :, 1]

    magnitude, angle = cv2.cartToPolar(fx, fy, angleInDegrees=False)

    # HSV image:
    # H -> direction
    # S -> fixed high saturation
    # V -> magnitude
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)

    # OpenCV hue range is [0, 180]
    hsv[:, :, 0] = (angle * 180 / np.pi / 2).astype(np.uint8)
    hsv[:, :, 1] = 255

    mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    hsv[:, :, 2] = mag_norm.astype(np.uint8)

    flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return flow_vis


def visualize_pipeline_summary(scene_out, mask_out, motion_out):
    """
    Create a summary panel showing scene image, mask, and dense motion field.

    Args:
        scene_out: dict containing at least "stylized_image"
        mask_out: dict containing at least "final_fluid_mask"
        motion_out: dict containing at least "dense_motion_field"

    Returns:
        summary: a single concatenated visualization image
    """
    stylized_image = scene_out["stylized_image"]
    final_mask = mask_out["final_fluid_mask"]
    dense_motion_field = motion_out["dense_motion_field"]

    mask_vis = visualize_mask(final_mask)
    flow_vis = visualize_motion_field(dense_motion_field)

    # Ensure all panels have the same size
    h, w = stylized_image.shape[:2]
    mask_vis = cv2.resize(mask_vis, (w, h))
    flow_vis = cv2.resize(flow_vis, (w, h))

    # Add simple labels
    scene_panel = stylized_image.copy()
    mask_panel = mask_vis.copy()
    flow_panel = flow_vis.copy()

    cv2.putText(scene_panel, "Stylized Image", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(mask_panel, "Final Fluid Mask", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(flow_panel, "Dense Motion Field", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Concatenate side by side
    summary = np.concatenate([scene_panel, mask_panel, flow_panel], axis=1)

    return summary
