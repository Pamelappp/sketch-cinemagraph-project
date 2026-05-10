"""Export synthesized frames into GIF or MP4 cinemagraph outputs."""

import os
import cv2
import imageio.v2 as imageio
import numpy as np


def export_cinemagraph(frames, output_path: str, fps: int = 20):
    """
    Save the looping frame sequence as a cinemagraph file on disk.

    Args:
        frames: list of H x W x C numpy arrays
        output_path: output file path, e.g. "result.gif" or "result.mp4"
        fps: frames per second

    Returns:
        output_path: final saved path
    """
    ext = os.path.splitext(output_path)[1].lower()

    if ext == ".gif":
        return export_gif(frames, output_path, fps)
    elif ext == ".mp4":
        return export_mp4(frames, output_path, fps)
    else:
        raise ValueError(f"Unsupported output format: {ext}. Use .gif or .mp4")


def export_gif(frames, output_path: str, fps: int = 20):
    """
    Export the cinemagraph as a GIF file.

    Args:
        frames: list of H x W x C numpy arrays
        output_path: output gif path
        fps: frames per second

    Returns:
        output_path: saved gif path
    """
    if len(frames) == 0:
        raise ValueError("frames is empty")

    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None

    processed_frames = []
    for frame in frames:
        frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
        processed_frames.append(frame_uint8)

    duration = 1.0 / fps
    imageio.mimsave(output_path, processed_frames, duration=duration, loop=0)

    return output_path


def export_mp4(frames, output_path: str, fps: int = 20):
    """
    Export the cinemagraph as an MP4 file.

    Args:
        frames: list of H x W x C numpy arrays
        output_path: output mp4 path
        fps: frames per second

    Returns:
        output_path: saved mp4 path
    """
    if len(frames) == 0:
        raise ValueError("frames is empty")

    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None

    first_frame = np.clip(frames[0], 0, 255).astype(np.uint8)
    h, w = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in frames:
        frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)

        # OpenCV expects BGR
        if frame_uint8.ndim == 3 and frame_uint8.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame_uint8

        writer.write(frame_bgr)

    writer.release()
    return output_path
