"""Interactive drawing tool for motion sketches.

Opens an OpenCV window where you can draw motion strokes with the mouse.
Each stroke is automatically rendered with a white-to-black gradient along
its length (white end = motion start, black end = motion destination),
following the convention of the Sketch2Cinemagraph paper.

Controls:
    Left-drag   : draw a stroke
    n           : start a new stroke without saving (clear current preview)
    u           : undo the last completed stroke
    c           : clear the canvas entirely
    s           : save the canvas as a PNG and exit
    q / Esc     : quit without saving

Usage::

    python scripts/draw_motion_sketch.py --output motion_sketch.png
    python scripts/draw_motion_sketch.py --size 384 --background structural_sketch.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/inputs/motion_sketch/my_sketch.png"))
    parser.add_argument("--size", type=int, default=384, help="Canvas size (square).")
    parser.add_argument("--background", type=Path, default=None,
                        help="Optional structural-sketch PNG to render under the strokes for reference.")
    parser.add_argument("--thickness", type=int, default=3)
    return parser.parse_args()


class SketchCanvas:
    def __init__(self, size: int, thickness: int, background: np.ndarray | None) -> None:
        self.size = size
        self.thickness = thickness
        self.background = background  # display-only, never saved
        self.strokes: List[List[Tuple[int, int]]] = []
        self.current: List[Tuple[int, int]] = []
        self.drawing = False

    def on_mouse(self, event, x, y, flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current = [(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            if self.current and (x, y) != self.current[-1]:
                self.current.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if len(self.current) >= 2:
                self.strokes.append(self.current)
            self.current = []

    def render(self, include_current: bool = True) -> np.ndarray:
        """Return the canvas as a 3-channel uint8 RGB image with strokes baked in."""
        canvas = np.full((self.size, self.size, 3), 255, dtype=np.uint8)
        for stroke in self.strokes:
            self._draw_stroke(canvas, stroke)
        if include_current and len(self.current) >= 2:
            self._draw_stroke(canvas, self.current)
        return canvas

    def render_with_background(self) -> np.ndarray:
        """Render with the optional structural-sketch background overlay (display only)."""
        canvas = self.render(include_current=True)
        if self.background is None:
            return canvas
        # Blend background at low opacity so strokes stay clearly visible.
        bg = self.background.astype(np.float32) * 0.4 + canvas.astype(np.float32) * 0.6
        return np.clip(bg, 0, 255).astype(np.uint8)

    def _draw_stroke(self, canvas: np.ndarray, stroke: List[Tuple[int, int]]) -> None:
        n = len(stroke)
        if n < 2:
            return
        for i in range(n - 1):
            ratio = (i + 1) / n
            grey = int(round(255 * (1.0 - ratio)))
            cv2.line(canvas, stroke[i], stroke[i + 1],
                     (grey, grey, grey), thickness=self.thickness, lineType=cv2.LINE_AA)


def load_background(path: Path | None, size: int) -> np.ndarray | None:
    if path is None:
        return None
    if not path.exists():
        print(f"Background not found: {path}")
        return None
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def main() -> None:
    args = parse_args()
    background = load_background(args.background, args.size)

    canvas = SketchCanvas(args.size, args.thickness, background)

    window = "motion sketch (left-drag draws, s=save, u=undo, c=clear, q=quit)"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, canvas.on_mouse)

    print("Drawing window opened.")
    print("Each stroke gets a white-to-black gradient automatically.")
    print("White end = motion start, Black end = motion destination.")

    while True:
        display = canvas.render_with_background()
        cv2.imshow(window, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("q"), 27):  # q or Esc
            print("Quit without saving.")
            break
        if key == ord("c"):
            canvas.strokes.clear()
            canvas.current.clear()
            print("Canvas cleared.")
        elif key == ord("u"):
            if canvas.strokes:
                canvas.strokes.pop()
                print("Undid last stroke.")
        elif key == ord("n"):
            canvas.current.clear()
            print("Cleared current stroke preview.")
        elif key == ord("s"):
            args.output.parent.mkdir(parents=True, exist_ok=True)
            final = canvas.render(include_current=False)
            cv2.imwrite(str(args.output), cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
            print(f"Saved {len(canvas.strokes)} stroke(s) to {args.output}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
