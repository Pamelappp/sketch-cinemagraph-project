"""Entry point for the sketch-guided cinemagraph project."""

import json
from pathlib import Path

import cv2

from src.io_utils import load_config, load_user_input, save_image, ensure_dir
from src.pipeline import CinemagraphPipeline
from src.evaluation.visualize import (
    visualize_mask,
    visualize_motion_field,
    visualize_pipeline_summary,
)


def main() -> None:
    config_path = Path("configs/default.yaml")
    cfg = load_config(config_path)

    user_input = load_user_input(cfg.get("input", {}))
    pipeline = CinemagraphPipeline(cfg)

    results = pipeline.run(user_input)

    scene_out = results["scene_out"]
    mask_out = results["mask_out"]
    motion_out = results["motion_out"]
    synth_out = results["synth_out"]
    metrics = results["metrics"]

    output_cfg = cfg.get("output", {})

    # 1. save scene images
    stylized_image_path = output_cfg.get("stylized_image_path")
    if stylized_image_path:
        save_image(scene_out["stylized_image"], stylized_image_path)

    realistic_reference_path = output_cfg.get("realistic_reference_path")
    if realistic_reference_path and scene_out["realistic_reference"] is not None:
        save_image(scene_out["realistic_reference"], realistic_reference_path)

    # 2. save debug visualizations
    debug_dir = ensure_dir(output_cfg.get("debug_dir", "data/outputs/debug"))

    mask_vis = visualize_mask(mask_out["final_fluid_mask"])
    flow_vis = visualize_motion_field(motion_out["dense_motion_field"])
    summary_vis = visualize_pipeline_summary(scene_out, mask_out, motion_out)

    cv2.imwrite(str(debug_dir / "final_mask.png"), mask_vis)
    cv2.imwrite(str(debug_dir / "dense_motion_field.png"), flow_vis)
    cv2.imwrite(str(debug_dir / "pipeline_summary.png"), summary_vis)

    # 3. save metrics
    metrics_path = debug_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # 4. print summary
    print("=== Sketch Cinemagraph Pipeline Finished ===")
    print(f"Stylized image:      {stylized_image_path}")
    print(f"Reference image:     {realistic_reference_path}")
    print(f"Final cinemagraph:   {synth_out['final_video_path']}")
    print(f"Debug folder:        {debug_dir}")
    print(f"Metrics saved to:    {metrics_path}")
    print()
    print("=== Simple Metrics ===")
    print(f"Motion smoothness: {metrics['motion_smoothness']:.6f}  (lower is better)")
    print(f"Loop consistency:  {metrics['loop_consistency']:.6f}  (lower is better)")
    print(f"Mask leakage:      {metrics['mask_leakage']:.6f}  (lower is better)")


if __name__ == "__main__":
    main()