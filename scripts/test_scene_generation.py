"""
Standalone test script for scene generation (stylized + realistic reference).
Runs ONLY the ControlNet diffusion step — no mask, motion, or synthesis.

Usage:
    python test_scene_generation.py
    python test_scene_generation.py --scale 1.4
    python test_scene_generation.py --scale 1.5 --seed 123
    python test_scene_generation.py --sketch path/to/sketch.png --prompt "a river in a forest"
    python test_scene_generation.py --scale 0.9 --out data/outputs/scene/test_scale09
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.io_utils import load_config, load_user_input
from src.scene_generation.generator import SceneGenerator
from src.scene_generation.prompt_utils import (
    build_scene_prompt,
    build_reference_prompt,
    build_negative_prompt,
)


def parse_args():
    p = argparse.ArgumentParser(description="Test scene generation in isolation.")
    p.add_argument("--config", default="configs/default.yaml",
                   help="Path to YAML config file")
    p.add_argument("--scale", type=float, default=None,
                   help="Override controlnet_conditioning_scale")
    p.add_argument("--seed", type=int, default=None,
                   help="Override random seed (use None in config for random)")
    p.add_argument("--sketch", default=None,
                   help="Override structural sketch image path")
    p.add_argument("--prompt", default=None,
                   help="Override text prompt")
    p.add_argument("--out", default="data/outputs/scene/test_scene",
                   help="Output path prefix; suffixes _stylized.png / _reference.png / _comparison.png are appended")
    return p.parse_args()


def _to_bgr(arr):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    elif arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def _add_label(img, text):
    out = img.copy()
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
    return out


def main():
    args = parse_args()
    cfg = load_config(Path(args.config))

    scene_cfg = cfg.setdefault("scene", {})
    if args.scale is not None:
        scene_cfg["controlnet_conditioning_scale"] = args.scale
    if args.seed is not None:
        scene_cfg["seed"] = args.seed

    print("=== Scene Generation Config ===")
    for key in ("model_id", "controlnet_id", "controlnet_conditioning_scale",
                "guidance_scale", "num_inference_steps", "seed", "style_prompt"):
        print(f"  {key:<34}: {scene_cfg.get(key)}")

    input_cfg = cfg.get("input", {})
    if args.sketch:
        input_cfg = dict(input_cfg)
        input_cfg["structural_sketch_path"] = args.sketch

    user_input = load_user_input(input_cfg)
    sketch = user_input.structural_sketch
    text_prompt = args.prompt if args.prompt else user_input.text_prompt

    print(f"\n=== Inputs ===")
    print(f"  sketch shape : {np.asarray(sketch).shape}")
    print(f"  text_prompt  : {text_prompt!r}")

    style_prompt = scene_cfg.get("style_prompt") or None
    scene_prior  = scene_cfg.get("scene_prior", "")
    extra_neg    = scene_cfg.get("extra_negative_prompt", "")
    stylized_prompt  = build_scene_prompt(text_prompt, style_prompt=style_prompt, scene_prior=scene_prior)
    reference_prompt = build_reference_prompt(text_prompt, scene_prior=scene_prior)
    negative_prompt  = build_negative_prompt(text_prompt, scene_prior=scene_prior, extra_negative_prompt=extra_neg)
    print(f"\n=== Prompts sent to diffusion model ===")
    print(f"  stylized  : {stylized_prompt!r}")
    print(f"  reference : {reference_prompt!r}")
    print(f"  negative  : {negative_prompt!r}")

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    scene_cfg["debug_dir"] = str(out_prefix.parent)

    print("\n=== Running scene generation (may take a while on CPU) ===")
    generator = SceneGenerator(scene_cfg)
    scene_out = generator.generate(sketch, text_prompt)

    stylized_path = Path(str(out_prefix) + "_stylized.png")
    cv2.imwrite(str(stylized_path), _to_bgr(scene_out.stylized_image))
    print(f"\nSaved stylized image   → {stylized_path}")

    ref_path = None
    if scene_out.realistic_reference is not None:
        ref_path = Path(str(out_prefix) + "_reference.png")
        cv2.imwrite(str(ref_path), _to_bgr(scene_out.realistic_reference))
        print(f"Saved reference image  → {ref_path}")
    else:
        print("Reference image not generated (save_reference=false in config).")

    stylized_bgr = _to_bgr(scene_out.stylized_image)
    h, w = stylized_bgr.shape[:2]
    sketch_bgr = cv2.resize(_to_bgr(np.asarray(sketch)), (w, h))

    scale_label = f"scale={scene_cfg.get('controlnet_conditioning_scale')}"
    panels = [
        _add_label(sketch_bgr, "Input Sketch"),
        _add_label(stylized_bgr, f"Stylized ({scale_label})"),
    ]
    if scene_out.realistic_reference is not None:
        ref_bgr = cv2.resize(_to_bgr(scene_out.realistic_reference), (w, h))
        panels.append(_add_label(ref_bgr, "Realistic Reference"))

    comparison = np.concatenate(panels, axis=1)
    cmp_path = Path(str(out_prefix) + "_comparison.png")
    cv2.imwrite(str(cmp_path), comparison)
    print(f"Saved comparison panel → {cmp_path}")

    print("\n=== Done ===")
    print("Tip: if output is not faithful to the sketch, try --scale 1.4 or --scale 1.5")


if __name__ == "__main__":
    main()
