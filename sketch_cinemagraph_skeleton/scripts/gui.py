"""Minimal Gradio GUI for the sketch-cinemagraph pipeline.

Usage:
    python scripts/gui.py               # local browser on http://127.0.0.1:7860
    python scripts/gui.py --share       # public Gradio link
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import traceback
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import load_config
from src.pipeline import CinemagraphPipeline
from src.types import UserInput
from src.evaluation.metrics import sanitize_for_json
from src.evaluation.visualize import visualize_mask, visualize_motion_field


EXAMPLES_DIR = ROOT / "data" / "inputs"
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"

EXAMPLE_NAMES = ["village", "lake", "river", "waterfall", "smoke"]


def _example_paths(name: str) -> tuple[str | None, str | None, str]:
    struct_p = EXAMPLES_DIR / "structural_sketch" / f"{name}.png"
    motion_p = EXAMPLES_DIR / "motion_sketch" / f"{name}.png"
    prompt_p = EXAMPLES_DIR / "prompts" / f"{name}.txt"

    struct = str(struct_p) if struct_p.exists() else None
    motion = str(motion_p) if motion_p.exists() else None
    prompt = prompt_p.read_text(encoding="utf-8").strip() if prompt_p.exists() else ""
    return struct, motion, prompt


def _to_rgb_uint8(value) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    if array.ndim == 3 and array.shape[2] == 4:
        array = cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
    return array


def _bgr_png_to_rgb(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def run_pipeline(
    structural_sketch,
    motion_sketch,
    text_prompt: str,
    backend: str,
    num_frames: int,
    fps: int,
    max_displacement: float,
):
    if structural_sketch is None or motion_sketch is None:
        raise gr.Error("Please provide both a structural sketch and a motion sketch.")
    if not text_prompt or not text_prompt.strip():
        raise gr.Error("Please enter a text prompt.")

    cfg = load_config(DEFAULT_CONFIG)
    cfg.setdefault("scene", {})["backend"] = backend
    cfg.setdefault("synthesis", {})["num_frames"] = int(num_frames)
    cfg["synthesis"]["fps"] = int(fps)
    cfg.setdefault("motion", {})["max_displacement"] = float(max_displacement)

    work_dir = Path(tempfile.mkdtemp(prefix="cinemagraph_gui_"))
    cfg["output"] = {
        "debug_dir": str(work_dir / "debug"),
        "final_video_path": str(work_dir / "output.gif"),
    }
    cfg["scene"]["debug_dir"] = cfg["output"]["debug_dir"]

    user_input = UserInput(
        structural_sketch=_to_rgb_uint8(structural_sketch),
        motion_sketch=_to_rgb_uint8(motion_sketch),
        text_prompt=text_prompt.strip(),
    )

    t0 = time.time()
    pipeline = CinemagraphPipeline(cfg)
    results = pipeline.run(user_input)
    elapsed = time.time() - t0

    scene_out  = results["scene_out"]
    mask_out   = results["mask_out"]
    motion_out = results["motion_out"]
    synth_out  = results["synth_out"]
    metrics    = sanitize_for_json(results["metrics"])

    stylized   = scene_out["stylized_image"]
    reference  = scene_out.get("realistic_reference")
    mask_vis   = cv2.cvtColor(visualize_mask(mask_out["final_fluid_mask"]), cv2.COLOR_BGR2RGB)
    flow_vis   = cv2.cvtColor(visualize_motion_field(motion_out["dense_motion_field"]), cv2.COLOR_BGR2RGB)
    safe_mask  = _bgr_png_to_rgb(Path(cfg["output"]["debug_dir"]) / "debug_safe_moving_mask_after_border.png")
    motion_alpha = _bgr_png_to_rgb(Path(cfg["output"]["debug_dir"]) / "debug_motion_alpha.png")
    gif_path   = synth_out["final_video_path"]

    summary = f"Pipeline finished in {elapsed:.1f}s — {len(synth_out['frames'])} frames\n"
    summary += "\n".join(
        f"  {k:30s}: {v}" for k, v in metrics.items()
    )

    return stylized, reference, mask_vis, safe_mask, flow_vis, motion_alpha, gif_path, summary


def on_example(name: str):
    return _example_paths(name)


def build_ui():
    struct_default, motion_default, prompt_default = _example_paths("village")

    with gr.Blocks(title="Sketch Cinemagraph") as demo:
        gr.Markdown("# Sketch Cinemagraph Pipeline")
        gr.Markdown(
            "Sketch + text prompt → looping cinemagraph. "
            "`placeholder` backend skips diffusion (fast, ~30s); "
            "`diffusion` runs SD1.5 + ControlNet (slow on CPU)."
        )

        with gr.Row():
            with gr.Column(scale=1):
                example_dropdown = gr.Dropdown(
                    choices=EXAMPLE_NAMES, value="village",
                    label="Example preset (loads sketches + prompt)",
                )
                struct_in = gr.Image(
                    label="Structural sketch", type="numpy", height=300,
                    value=struct_default,
                )
                motion_in = gr.Image(
                    label="Motion sketch", type="numpy", height=300,
                    value=motion_default,
                )
                prompt_in = gr.Textbox(
                    label="Text prompt", lines=4, value=prompt_default,
                )

                with gr.Row():
                    backend = gr.Radio(
                        choices=["placeholder", "diffusion"],
                        value="placeholder",
                        label="Scene backend",
                    )
                with gr.Row():
                    num_frames = gr.Slider(20, 120, value=60, step=2, label="Num frames")
                    fps = gr.Slider(10, 60, value=30, step=1, label="FPS")
                max_disp = gr.Slider(
                    0.2, 5.0, value=2.0, step=0.1,
                    label="Max displacement (px/step) — higher = more visible motion",
                )

                run_btn = gr.Button("Generate cinemagraph", variant="primary")

            with gr.Column(scale=1):
                gif_out      = gr.Image(label="Cinemagraph (GIF)", height=400)
                summary_out  = gr.Textbox(label="Summary / metrics", lines=10, interactive=False)

        gr.Markdown("### Intermediate stages")
        with gr.Row():
            stylized_out  = gr.Image(label="Stylized image", height=260)
            reference_out = gr.Image(label="Realistic reference", height=260)
            mask_out_img  = gr.Image(label="Final fluid mask", height=260)
        with gr.Row():
            safe_mask_out    = gr.Image(label="Safe moving mask", height=260)
            flow_out         = gr.Image(label="Dense motion field", height=260)
            motion_alpha_out = gr.Image(label="Motion alpha (protection)", height=260)

        example_dropdown.change(
            on_example, inputs=[example_dropdown],
            outputs=[struct_in, motion_in, prompt_in],
        )
        run_btn.click(
            run_pipeline,
            inputs=[struct_in, motion_in, prompt_in, backend, num_frames, fps, max_disp],
            outputs=[
                stylized_out, reference_out, mask_out_img,
                safe_mask_out, flow_out, motion_alpha_out,
                gif_out, summary_out,
            ],
        )
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
