"""Side-by-side comparison of heuristic vs learned motion backends.

Loads the latest checkpoint, generates the same cinemagraph with each
backend, and saves a 4-panel comparison grid (image | sketch+mask
overlay | heuristic flow | learned flow) for every example. Also prints
quantitative stats so you can see at a glance whether the trained model
escaped the zero-prediction plateau.

Usage::

    python scripts/compare_backends.py
    python scripts/compare_backends.py --checkpoint checkpoints/best.pt
    python scripts/compare_backends.py --image-size 384 --motion-scale 1.5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--motion-scale", type=float, default=2.0)
    parser.add_argument("--num-frames", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=Path("data/outputs/comparison"))
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stub_grounded_sam():
    from src.fluid_mask import refine_mask as refine_module

    def _noop(image, text_prompt: str):
        h, w = np.asarray(image).shape[:2]
        return np.zeros((h, w), dtype=np.uint8)

    refine_module.refine_fluid_mask = _noop


def patch_scene_with_textured(stylized: np.ndarray) -> None:
    from src.scene_generation import generator as gen_module
    from src.types import SceneOutput

    def _patched(self, structural_sketch, text_prompt: str) -> SceneOutput:
        return SceneOutput(stylized_image=stylized, realistic_reference=stylized.copy())

    gen_module.SceneGenerator.generate = _patched


def make_test_pair(size: int, seed: int) -> tuple:
    """Produce a (structural_sketch, motion_sketch, textured_landscape, prompt) tuple
    using the same procedural generator as the demo runner."""
    from scripts.run_demo import build_sketches, build_textured_landscape

    rng_seed = seed
    structural, motion, horizon_y, peaks = build_sketches(size)
    textured = build_textured_landscape(size, horizon_y, peaks, rng_seed)
    return structural, motion, textured


def run_backend(
    backend: str,
    structural: np.ndarray,
    motion: np.ndarray,
    image_size: int,
    motion_scale: float,
    num_frames: int,
    checkpoint: Path,
    out_dir: Path,
):
    from src.pipeline import CinemagraphPipeline
    from src.evaluation.visualize import visualize_motion_field, visualize_mask
    from src.evaluation.metrics import (
        compute_motion_smoothness,
        compute_loop_consistency,
        compute_mask_leakage,
    )

    cfg = {
        "scene": {"image_size": [image_size, image_size], "save_reference": True},
        "motion": {"backend": backend},
        "synthesis": {
            "num_frames": num_frames, "fps": 18, "loop_mode": "linear",
            "motion_scale": motion_scale, "border_mode": "reflect",
        },
        "output": {"dir": str(out_dir), "video_name": f"cinemagraph_{backend}.mp4",
                   "gif_name": f"cinemagraph_{backend}.gif"},
    }
    if backend == "learned":
        cfg["motion"]["learned"] = {
            "checkpoint_path": str(checkpoint),
            "device": "mps", "input_size": 256,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pipeline = CinemagraphPipeline(cfg)
    result = pipeline.run(structural, motion, "ocean waves under blue sky")
    elapsed = time.time() - t0

    flow = result["dense_motion_field"]
    fluid = result["final_fluid_mask"] > 0
    if fluid.any():
        magnitude = np.linalg.norm(flow, axis=-1)[fluid]
        mean_mag, max_mag = float(magnitude.mean()), float(magnitude.max())
    else:
        mean_mag = max_mag = 0.0

    smoothness = compute_motion_smoothness(flow, result["final_fluid_mask"])
    loop_mse = compute_loop_consistency(result["frames"])
    leakage = compute_mask_leakage(result["frames"], result["final_fluid_mask"])

    return {
        "backend": backend,
        "stylized": result["stylized_image"],
        "mask_vis": visualize_mask(result["final_fluid_mask"]),
        "flow_vis": visualize_motion_field(flow),
        "frame_first": result["frames"][0],
        "frame_mid": result["frames"][len(result["frames"]) // 2],
        "elapsed_s": elapsed,
        "mean_flow_px": mean_mag,
        "max_flow_px": max_mag,
        "smoothness": smoothness,
        "loop_mse": loop_mse,
        "leakage": leakage,
        "video_path": result["video_path"],
        "gif_path": result["gif_path"],
        "num_frames": len(result["frames"]),
    }


def build_panel(items: list, target_size: int) -> np.ndarray:
    """Concatenate a list of (label, image) tuples horizontally with separators."""
    sep = np.full((target_size, 6, 3), 255, dtype=np.uint8)
    pieces = []
    for i, (_label, img) in enumerate(items):
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.shape[:2] != (target_size, target_size):
            img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
        pieces.append(img)
        if i < len(items) - 1:
            pieces.append(sep)
    panel = np.concatenate(pieces, axis=1)

    # Add labels by stacking a row of text below each tile.
    label_h = 32
    label_strip = np.full((label_h, panel.shape[1], 3), 240, dtype=np.uint8)
    cursor = 0
    for i, (label, _) in enumerate(items):
        x = cursor + target_size // 2 - 6 * len(label) // 2
        cv2.putText(label_strip, label, (max(cursor + 4, x), 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
        cursor += target_size + 6
    return np.concatenate([panel, label_strip], axis=0)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    print(f"=== Comparing heuristic vs learned ({args.checkpoint}) ===\n")
    stub_grounded_sam()

    summary_rows = []
    for example_idx in range(args.num_examples):
        seed = args.seed + example_idx
        print(f"--- Example {example_idx + 1} (seed={seed}) ---")
        structural, motion, textured = make_test_pair(args.image_size, seed)
        patch_scene_with_textured(textured)

        per_dir = args.output_dir / f"example_{example_idx + 1:02d}"
        per_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(per_dir / "input_structural.png"),
                    cv2.cvtColor(structural, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(per_dir / "input_motion.png"),
                    cv2.cvtColor(motion, cv2.COLOR_RGB2BGR))

        heuristic = run_backend("heuristic", structural, motion, args.image_size,
                                args.motion_scale, args.num_frames, args.checkpoint,
                                per_dir / "heuristic")
        learned = run_backend("learned", structural, motion, args.image_size,
                              args.motion_scale, args.num_frames, args.checkpoint,
                              per_dir / "learned")

        for name, r in [("heuristic", heuristic), ("learned", learned)]:
            print(f"  [{name}] mean_flow={r['mean_flow_px']:.3f}px  max={r['max_flow_px']:.3f}px"
                  f"  smoothness={r['smoothness']:.3f}  loop_mse={r['loop_mse']:.2f}"
                  f"  leakage={r['leakage']:.3f}  ({r['elapsed_s']:.1f}s)")

        # 1x4 panel: stylized | mask | heuristic flow | learned flow
        panel = build_panel([
            ("Stylized image", heuristic["stylized"]),
            ("Fluid mask", heuristic["mask_vis"]),
            ("Heuristic flow", heuristic["flow_vis"]),
            ("Learned flow", learned["flow_vis"]),
        ], target_size=args.image_size)
        cv2.imwrite(str(per_dir / "comparison_flow.png"),
                    cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

        # 1x4 panel showing the same frame in each backend for animation comparison
        anim_panel = build_panel([
            ("Source", heuristic["stylized"]),
            ("Heuristic mid-frame", heuristic["frame_mid"]),
            ("Learned mid-frame", learned["frame_mid"]),
        ], target_size=args.image_size)
        cv2.imwrite(str(per_dir / "comparison_frames.png"),
                    cv2.cvtColor(anim_panel, cv2.COLOR_RGB2BGR))

        summary_rows.append((example_idx + 1, heuristic, learned))
        print()

    # Final stats table
    print("\n=== Summary ===")
    print(f"{'Example':<10}{'Backend':<12}{'mean_flow':<12}{'max_flow':<12}"
          f"{'smoothness':<12}{'loop_mse':<12}{'leakage':<12}")
    for idx, h, l in summary_rows:
        print(f"{idx:<10}{'heuristic':<12}{h['mean_flow_px']:<12.3f}{h['max_flow_px']:<12.3f}"
              f"{h['smoothness']:<12.3f}{h['loop_mse']:<12.2f}{h['leakage']:<12.3f}")
        print(f"{'':<10}{'learned':<12}{l['mean_flow_px']:<12.3f}{l['max_flow_px']:<12.3f}"
              f"{l['smoothness']:<12.3f}{l['loop_mse']:<12.2f}{l['leakage']:<12.3f}")

    # Verdict
    learned_means = [l["mean_flow_px"] for _, _, l in summary_rows]
    if max(learned_means) < 0.1:
        print("\n❌ Learned backend output is essentially zero across all examples.")
        print("   The model did NOT escape the predict-zero plateau during training.")
        print("   Use heuristic backend for the final report; the learned U-Net needs")
        print("   either more training, a different loss formulation, or more diverse data.")
    elif max(learned_means) < 1.0:
        print("\n⚠  Learned backend predicts non-trivial but weak motion.")
        print("   Heuristic still likely produces stronger animations; both are usable.")
    else:
        print("\n✅ Learned backend predicts meaningful motion.")
        print("   Both backends are usable; pick whichever looks better in the demo.")

    print(f"\nAll comparison artefacts under {args.output_dir}/")


if __name__ == "__main__":
    main()
