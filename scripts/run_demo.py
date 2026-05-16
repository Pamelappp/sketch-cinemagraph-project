"""End-to-end pipeline smoke demo with a properly textured scene.

The default placeholder scene generator (Member A) produces a flat colored
image, which makes warping invisible. To actually showcase the cinemagraph
effect, this demo:

1. Builds a textured procedural landscape (gradient sky, mountains with
   noise, multi-octave water surface) so warping has high-frequency content
   to move.
2. Synthesises matching structural + motion sketches.
3. Bypasses the scene generator by injecting the textured image directly.
4. Runs the rest of the pipeline (fluid mask + motion field + warping)
   exactly as it would on a real Stable-Diffusion-generated image.
5. Saves intermediate visuals AND a 6-frame strip so the motion is
   immediately visible without having to open the GIF.

Usage::

    python scripts/run_demo.py
    python scripts/run_demo.py --skip-learned
    python scripts/run_demo.py --motion-scale 8 --num-frames 60
    python scripts/run_demo.py --input-image path/to/your/photo.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--num-frames", type=int, default=60)
    parser.add_argument("--motion-scale", type=float, default=1.5,
                        help="Multiplier on the motion field; max per-cycle displacement.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--skip-learned", action="store_true")
    parser.add_argument("--input-image", type=Path, default=None,
                        help="Optional real landscape photo to use as stylized image.")
    parser.add_argument("--structural-sketch", type=Path, default=None,
                        help="Optional user-drawn structural sketch PNG (overrides synthetic one).")
    parser.add_argument("--motion-sketch", type=Path, default=None,
                        help="Optional user-drawn motion sketch PNG (overrides synthetic one).")
    parser.add_argument("--skip-grounded-sam", action="store_true", default=True)
    parser.add_argument("--use-grounded-sam", dest="skip_grounded_sam", action="store_false")
    parser.add_argument("--output-root", type=Path, default=Path("data/outputs/demo"))
    parser.add_argument("--prompt", type=str, default="ocean waves under blue sky")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Procedural landscape with real texture
# ---------------------------------------------------------------------------


def build_textured_landscape(
    size: int, horizon_y: int, peaks: list, seed: int = 7
) -> np.ndarray:
    """
    Render a landscape image with proper water/sky texture so frame warping
    produces a clearly visible animation.
    """
    rng = np.random.default_rng(seed)
    image = np.zeros((size, size, 3), dtype=np.float32)

    # ---- Sky: vertical blue gradient with subtle cloud noise ----
    sky_top = np.array([110, 160, 220], dtype=np.float32)
    sky_bottom = np.array([200, 220, 240], dtype=np.float32)
    for y in range(horizon_y):
        ratio = y / max(horizon_y - 1, 1)
        image[y, :, :] = sky_top * (1 - ratio) + sky_bottom * ratio
    cloud_noise = _multi_octave_noise(size, horizon_y, rng, octaves=3, base=20)
    cloud_alpha = np.clip(cloud_noise * 0.6 - 6, 0, 30)[..., None]
    image[:horizon_y] = np.clip(image[:horizon_y] + cloud_alpha, 0, 255)

    # ---- Mountains: triangular silhouettes with noise shading ----
    mountain_layer = np.zeros((size, size), dtype=np.uint8)
    for cx, cy in peaks:
        pts = np.array([
            [cx - size // 8, horizon_y],
            [cx, cy],
            [cx + size // 8, horizon_y],
        ], dtype=np.int32)
        cv2.fillPoly(mountain_layer, [pts], 255)
    mountain_mask = mountain_layer > 0
    mountain_color = np.array([95, 80, 70], dtype=np.float32)
    mountain_noise = _multi_octave_noise(size, size, rng, octaves=2, base=15)[..., None]
    image[mountain_mask] = np.clip(
        mountain_color + mountain_noise[mountain_mask] * 0.5, 0, 255
    )

    # ---- Water: high-frequency multi-octave noise on a blue base ----
    water_h = size - horizon_y
    water_base = np.zeros((water_h, size, 3), dtype=np.float32)
    water_base[..., 0] = 25     # R
    water_base[..., 1] = 80     # G
    water_base[..., 2] = 150    # B (deep blue)

    # Ripple texture: anisotropic noise stretched horizontally to look like waves.
    waves = _multi_octave_noise(size, water_h, rng, octaves=5, base=40,
                                aspect=(1.0, 4.0))
    waves = (waves - waves.mean()) * 1.2
    water_base[..., 0] += waves * 0.3
    water_base[..., 1] += waves * 0.6
    water_base[..., 2] += waves * 1.0

    # Specular foam highlights on wave crests.
    foam_threshold = waves.mean() + 1.2 * waves.std()
    foam_mask = waves > foam_threshold
    water_base[foam_mask] += np.array([100, 100, 110], dtype=np.float32)

    # Fade the water deeper as it gets closer to the horizon (perspective).
    perspective = np.linspace(0.55, 1.0, water_h)[..., None, None]
    water_base = water_base * perspective + np.array([20, 50, 90]) * (1 - perspective)

    image[horizon_y:] = np.clip(water_base, 0, 255)

    # Slight overall blur to soften pixel edges.
    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    image_uint8 = cv2.GaussianBlur(image_uint8, (3, 3), 0)
    return image_uint8


def _multi_octave_noise(
    width: int,
    height: int,
    rng: np.random.Generator,
    octaves: int = 4,
    base: int = 20,
    aspect: Tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    """Sum-of-octaves value noise. ``aspect=(y, x)`` controls anisotropy."""
    out = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    total_amp = 0.0
    for octave in range(octaves):
        scale = 2 ** octave
        h = max(2, int(height // (base // scale * aspect[0])))
        w = max(2, int(width // (base // scale * aspect[1])))
        coarse = rng.standard_normal((h, w)).astype(np.float32)
        layer = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
        out += layer * amplitude
        total_amp += amplitude
        amplitude *= 0.5
    out /= max(total_amp, 1e-6)
    out = (out - out.min()) / (out.max() - out.min() + 1e-6)
    return out * 255.0


# ---------------------------------------------------------------------------
# Synthetic sketches (paired with the procedural landscape above)
# ---------------------------------------------------------------------------


def build_sketches(size: int) -> Tuple[np.ndarray, np.ndarray, int, list]:
    horizon_y = int(size * 0.55)
    peaks = [
        (int(size * 0.25), int(size * 0.30)),
        (int(size * 0.65), int(size * 0.36)),
    ]

    structural = np.full((size, size, 3), 255, dtype=np.uint8)
    pts = [(0, horizon_y)]
    for cx, cy in peaks:
        pts.extend([(cx - size // 12, horizon_y), (cx, cy), (cx + size // 12, horizon_y)])
    pts.append((size - 1, horizon_y))
    poly = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(structural, [poly], isClosed=False, color=(0, 0, 0), thickness=2)
    cv2.line(structural, (0, horizon_y), (0, size - 1), (0, 0, 0), 2)
    cv2.line(structural, (size - 1, horizon_y), (size - 1, size - 1), (0, 0, 0), 2)
    cv2.line(structural, (0, size - 1), (size - 1, size - 1), (0, 0, 0), 2)

    # Curved motion strokes — demonstrates the paper's polyline+gradient
    # convention for direction control. Three flowing curves at varying
    # depths in the water region.
    motion = np.full((size, size, 3), 255, dtype=np.uint8)
    curve_specs = [
        # (start_x, end_x, base_y, amplitude, frequency)
        (int(size * 0.15), int(size * 0.85), int(size * 0.70), int(size * 0.020), 1.4),
        (int(size * 0.20), int(size * 0.80), int(size * 0.80), int(size * 0.025), 1.0),
        (int(size * 0.18), int(size * 0.82), int(size * 0.90), int(size * 0.018), 1.6),
    ]
    for x_start, x_end, base_y, amp, freq in curve_specs:
        num_segments = 80
        for i in range(num_segments):
            ratio_a = i / num_segments
            ratio_b = (i + 1) / num_segments
            xa = int(round(x_start + (x_end - x_start) * ratio_a))
            xb = int(round(x_start + (x_end - x_start) * ratio_b))
            ya = int(round(base_y + amp * np.sin(2 * np.pi * freq * ratio_a)))
            yb = int(round(base_y + amp * np.sin(2 * np.pi * freq * ratio_b)))
            grey = int(round(255 * (1.0 - ratio_b)))
            cv2.line(motion, (xa, ya), (xb, yb), (grey, grey, grey),
                     thickness=3, lineType=cv2.LINE_AA)

    return structural, motion, horizon_y, peaks


def load_user_sketch(path: Path, size: int) -> np.ndarray:
    """Load a user-drawn sketch PNG and resize it to the demo canvas."""
    if not path.exists():
        raise SystemExit(f"Sketch not found: {path}")
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise SystemExit(f"Could not read sketch: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Pipeline glue
# ---------------------------------------------------------------------------


def stub_grounded_sam():
    from src.fluid_mask import refine_mask as refine_module

    def _noop_refine(image, text_prompt: str):
        h, w = np.asarray(image).shape[:2]
        return np.zeros((h, w), dtype=np.uint8)

    refine_module.refine_fluid_mask = _noop_refine
    print("(refine_mask) Grounded-SAM bypassed for this demo run.")


def patch_scene_generator(stylized_image: np.ndarray, realistic_image: np.ndarray) -> None:
    """Replace SceneGenerator.generate so the pipeline uses our textured image."""
    from src.scene_generation import generator as gen_module
    from src.types import SceneOutput

    def _patched_generate(self, structural_sketch, text_prompt: str) -> SceneOutput:
        return SceneOutput(
            stylized_image=stylized_image,
            realistic_reference=realistic_image,
        )

    gen_module.SceneGenerator.generate = _patched_generate
    print("(scene_generation) Patched generator to use textured demo landscape.")


def save_frame_strip(frames: list, path: Path, num_tiles: int = 6) -> None:
    """Save N evenly-spaced frames as a single horizontal strip."""
    if len(frames) <= num_tiles:
        chosen = frames
    else:
        idx = np.linspace(0, len(frames) - 1, num_tiles, dtype=int)
        chosen = [frames[i] for i in idx]

    h, w = chosen[0].shape[:2]
    sep = np.full((h, 4, 3), 255, dtype=np.uint8)
    tiles = []
    for i, f in enumerate(chosen):
        tiles.append(f)
        if i < len(chosen) - 1:
            tiles.append(sep)
    strip = np.concatenate(tiles, axis=1)
    cv2.imwrite(str(path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))


def run_one_backend(
    backend: str,
    structural: np.ndarray,
    motion: np.ndarray,
    prompt: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict:
    from src.pipeline import CinemagraphPipeline
    from src.evaluation.visualize import (
        visualize_mask, visualize_motion_field, visualize_pipeline_summary,
    )

    cfg = {
        "scene": {"image_size": [args.image_size, args.image_size], "save_reference": True},
        "motion": {"backend": backend},
        "synthesis": {
            "num_frames": args.num_frames,
            "fps": 18,
            "loop_mode": "linear",
            "motion_scale": args.motion_scale,
            "border_mode": "reflect",
        },
        "output": {
            "dir": str(out_dir),
            "video_name": f"cinemagraph_{backend}.mp4",
            "gif_name": f"cinemagraph_{backend}.gif",
        },
    }
    if backend == "learned":
        cfg["motion"]["learned"] = {
            "checkpoint_path": str(args.checkpoint),
            "device": "mps",
            "input_size": 256,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Running pipeline (backend={backend}) ===")
    pipeline = CinemagraphPipeline(cfg)
    result = pipeline.run(structural, motion, prompt)

    cv2.imwrite(str(out_dir / "stylized.png"),
                cv2.cvtColor(result["stylized_image"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "final_mask.png"),
                visualize_mask(result["final_fluid_mask"]))
    cv2.imwrite(str(out_dir / "flow.png"),
                cv2.cvtColor(visualize_motion_field(result["dense_motion_field"]), cv2.COLOR_RGB2BGR))
    summary = visualize_pipeline_summary(
        scene_out={"stylized_image": result["stylized_image"]},
        mask_out={"final_fluid_mask": result["final_fluid_mask"]},
        motion_out={"dense_motion_field": result["dense_motion_field"]},
    )
    cv2.imwrite(str(out_dir / "summary.png"), cv2.cvtColor(summary, cv2.COLOR_RGB2BGR))
    save_frame_strip(result["frames"], out_dir / "frame_strip.png", num_tiles=6)

    flow = result["dense_motion_field"]
    mask = result["final_fluid_mask"] > 0
    if mask.any():
        magnitude = np.linalg.norm(flow, axis=-1)[mask]
        print(f"  in-mask flow: mean={magnitude.mean():.3f}px max={magnitude.max():.3f}px")
    print(f"  frames    : {len(result['frames'])}")
    print(f"  video     : {result['video_path']}")
    print(f"  gif       : {result['gif_path']}")
    print(f"  strip     : {out_dir / 'frame_strip.png'}")
    return result


def main() -> None:
    args = parse_args()

    if args.skip_grounded_sam:
        stub_grounded_sam()

    structural, motion, horizon_y, peaks = build_sketches(args.image_size)

    if args.structural_sketch is not None:
        structural = load_user_sketch(args.structural_sketch, args.image_size)
        print(f"Loaded user structural sketch from {args.structural_sketch}")
    if args.motion_sketch is not None:
        motion = load_user_sketch(args.motion_sketch, args.image_size)
        print(f"Loaded user motion sketch from {args.motion_sketch}")

    if args.input_image is not None:
        if not args.input_image.exists():
            raise SystemExit(f"--input-image not found: {args.input_image}")
        bgr = cv2.imread(str(args.input_image))
        textured = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        textured = cv2.resize(textured, (args.image_size, args.image_size),
                              interpolation=cv2.INTER_AREA)
        print(f"Using {args.input_image} as stylized image.")
    else:
        textured = build_textured_landscape(args.image_size, horizon_y, peaks, args.seed)

    # Use the textured image as both stylized and realistic reference.
    patch_scene_generator(textured, textured.copy())

    inputs_dir = args.output_root / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(inputs_dir / "structural_sketch.png"),
                cv2.cvtColor(structural, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(inputs_dir / "motion_sketch.png"),
                cv2.cvtColor(motion, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(inputs_dir / "stylized_input.png"),
                cv2.cvtColor(textured, cv2.COLOR_RGB2BGR))
    print(f"Wrote synthetic inputs to {inputs_dir}")

    run_one_backend("heuristic", structural, motion, args.prompt, args,
                    args.output_root / "heuristic")

    if not args.skip_learned and args.checkpoint.exists():
        run_one_backend("learned", structural, motion, args.prompt, args,
                        args.output_root / "learned")
    else:
        if args.skip_learned:
            print("\n(skipping learned backend by request)")
        else:
            print(f"\n(skipping learned backend: checkpoint not found at {args.checkpoint})")

    print(f"\nDemo done. All artefacts under {args.output_root}/")
    print("Inspect data/outputs/demo/heuristic/frame_strip.png to see motion at a glance.")


if __name__ == "__main__":
    main()
