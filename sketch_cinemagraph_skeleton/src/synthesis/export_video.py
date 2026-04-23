"""Export synthesized frames into GIF or MP4 cinemagraph outputs."""


def export_cinemagraph(frames, output_path: str, fps: int = 20):
    """
    Save the looping frame sequence as a cinemagraph file on disk.

    Detailed TODO:
    1. Inspect file extension from output_path.
    2. Dispatch to export_gif() or export_mp4().
    3. Return the final path after export.
    """
    raise NotImplementedError


def export_gif(frames, output_path: str, fps: int = 20):
    """
    Export the cinemagraph as a GIF file.

    Detailed TODO:
    1. Convert frames to RGB uint8 if needed.
    2. Save with looping enabled.
    3. Keep fps or duration consistent with project settings.
    """
    raise NotImplementedError


def export_mp4(frames, output_path: str, fps: int = 20):
    """
    Export the cinemagraph as an MP4 file.

    Detailed TODO:
    1. Initialize a video writer.
    2. Write frames in order.
    3. Close the writer safely.
    """
    raise NotImplementedError
