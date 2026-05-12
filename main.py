"""Command-line entry point for the sketch-guided cinemagraph project.

Reads a YAML config, loads a structural sketch / motion sketch / prompt,
runs the full ``CinemagraphPipeline`` and writes the looping cinemagraph
plus all intermediate visualisations to disk.

Usage::

    python main.py
    python main.py --config configs/default.yaml
    python main.py --config configs/my_run.yaml --debug

The default config file is ``configs/default.yaml``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from src.io_utils import load_config, load_user_input, ensure_dir, save_image, save_mask, save_motion_field
from src.pipeline import CinemagraphPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sketch-guided cinemagraph pipeline.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"),
                        help="YAML config file.")
    parser.add_argument("--debug", action="store_true",
                        help="Save every intermediate stage (mask, flow, stylized, etc.).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    user_input = load_user_input(cfg.get("input", {}))
    pipeline = CinemagraphPipeline(cfg)
    result = pipeline.run(
        structural_sketch=user_input.structural_sketch,
        motion_sketch=user_input.motion_sketch,
        text_prompt=user_input.text_prompt,
    )

    output_cfg = cfg.get("output", {})
    debug_dir = Path(output_cfg.get("debug_dir", "data/outputs/debug"))
    if args.debug or output_cfg.get("save_debug", False):
        ensure_dir(debug_dir)
        save_image(result["stylized_image"], debug_dir / "stylized.png")
        if result.get("realistic_reference") is not None:
            save_image(result["realistic_reference"], debug_dir / "realistic.png")
        save_mask(result["semantic_mask"], debug_dir / "semantic_mask.png")
        save_mask(result["refined_mask"], debug_dir / "refined_mask.png")
        save_mask(result["final_fluid_mask"], debug_dir / "final_fluid_mask.png")
        save_motion_field(result["dense_motion_field"], debug_dir / "motion_field.npy")
        print(f"[debug] intermediate artefacts saved to {debug_dir}")

    stylized_path = output_cfg.get("stylized_image_path")
    if stylized_path:
        save_image(result["stylized_image"], stylized_path)
        print(f"[output] stylized image -> {stylized_path}")

    print(f"[output] cinemagraph mp4   -> {result['video_path']}")
    print(f"[output] cinemagraph gif   -> {result['gif_path']}")
    print(f"[output] frame count       -> {len(result['frames'])}")


if __name__ == "__main__":
    main()
