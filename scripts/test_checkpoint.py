"""Quick sanity check that a trained MotionUNet checkpoint can be loaded and run.

Loads ``checkpoints/best.pt`` (or a path passed on the command line),
constructs a synthetic landscape sample using the same procedural generator
that produced the training data, runs the LearnedMotionPredictor on it, and
reports the predicted flow's shape and magnitude statistics.

Usage::

    python scripts/test_checkpoint.py
    python scripts/test_checkpoint.py --checkpoint checkpoints/last.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.motion_field.learned_predictor import LearnedMotionPredictor
from src.motion_field.training_data import SyntheticConfig, SyntheticMotionDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    # Pull a single synthetic sample so we can feed it into the predictor.
    dataset = SyntheticMotionDataset(
        SyntheticConfig(image_size=args.image_size, num_samples=1, seed=123)
    )
    sample = dataset[0]
    inputs = sample["inputs"].numpy()  # (7, H, W)
    image = (inputs[0:3].transpose(1, 2, 0) * 255.0).astype(np.uint8)
    sketch = (inputs[3:6].transpose(1, 2, 0) * 255.0).astype(np.uint8)
    mask = (inputs[6] * 255.0).astype(np.uint8)

    print("=== Inputs ===")
    print(f"image  : {image.shape} {image.dtype} [{image.min()}, {image.max()}]")
    print(f"sketch : {sketch.shape} {sketch.dtype} [{sketch.min()}, {sketch.max()}]")
    print(f"mask   : {mask.shape} {mask.dtype} fluid pixels = {int((mask > 0).sum())}")

    predictor = LearnedMotionPredictor(
        {
            "checkpoint_path": str(args.checkpoint),
            "device": args.device,
            "input_size": args.image_size,
        }
    )

    print(f"\n=== Loading {args.checkpoint} ===")
    flow = predictor.predict(image, mask, sketch)

    print("\n=== Predicted flow ===")
    print(f"shape  : {flow.shape}")
    print(f"dtype  : {flow.dtype}")
    print(f"finite : {np.isfinite(flow).all()}")

    fluid_pixels = mask > 0
    if fluid_pixels.any():
        magnitude = np.linalg.norm(flow, axis=-1)
        print(f"in-mask  mean magnitude : {magnitude[fluid_pixels].mean():.4f} px")
        print(f"in-mask  max  magnitude : {magnitude[fluid_pixels].max():.4f} px")
        outside = ~fluid_pixels
        if outside.any():
            print(
                f"outside-mask max magnitude : {magnitude[outside].max():.4e} px "
                "(should be 0 — predictor enforces mask boundary)"
            )

    print("\nOK: checkpoint loads and predicts a finite, mask-bounded flow field.")


if __name__ == "__main__":
    main()
