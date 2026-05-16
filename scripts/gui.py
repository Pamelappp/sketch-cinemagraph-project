"""Gradio GUI for the Sketch2Cinemagraph pipeline.

Usage:
    python scripts/gui.py
    python scripts/gui.py --share          # public Gradio link
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Expose via a public Gradio URL.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    return parser.parse_args()


def _editor_to_numpy(value: Any) -> np.ndarray | None:
    """Normalise a Gradio ImageEditor / Image value into a HxWx3 uint8 array."""
    if value is None:
        return None
    if isinstance(value, dict):
        # gr.ImageEditor returns {"background", "layers", "composite"}.
        composite = value.get("composite")
        if composite is not None:
            value = composite
        else:
            background = value.get("background")
            layers = value.get("layers", [])
            if background is None and not layers:
                return None
            base = background if background is not None else layers[0]
            value = base
    array = np.asarray(value)
    if array is None:
        return None
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.size > 0 and array.max() <= 1.0:
            array = (array * 255.0).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    if array.shape[2] == 4:
        # Composite RGBA onto white so downstream sees white BG + black strokes.
        rgb = array[..., :3].astype(np.float32)
        alpha = array[..., 3:4].astype(np.float32) / 255.0
        white = np.full_like(rgb, 255.0)
        composed = rgb * alpha + white * (1.0 - alpha)
        array = np.clip(composed, 0, 255).astype(np.uint8)
    return array


def _ensure_white_background(sketch: np.ndarray) -> np.ndarray:
    """Invert if the canvas came through dark — pipeline expects white BG."""
    if sketch is None:
        return sketch
    gray = cv2.cvtColor(sketch, cv2.COLOR_RGB2GRAY)
    if gray.mean() < 80:
        return 255 - sketch
    return sketch


def apply_motion_gradient(motion_sketch: np.ndarray) -> np.ndarray:
    """Add a leftmost-white → rightmost-black gradient to each connected stroke."""
    if motion_sketch is None:
        return None
    rgb = motion_sketch
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ink = (gray < 200).astype(np.uint8)
    if ink.sum() == 0:
        return rgb

    num_labels, labels = cv2.connectedComponents(ink, connectivity=8)
    out = np.full_like(rgb, 255)

    for label_id in range(1, num_labels):
        component = labels == label_id
        if int(component.sum()) < 5:
            continue
        ys, xs = np.where(component)
        x_min, x_max = xs.min(), xs.max()
        if x_max == x_min:
            # Vertical / single-column stroke: gradient top→bottom instead.
            y_min, y_max = ys.min(), ys.max()
            if y_max == y_min:
                continue
            ratios = (ys - y_min) / max(y_max - y_min, 1)
        else:
            ratios = (xs - x_min) / max(x_max - x_min, 1)
        greys = np.clip(255.0 * (1.0 - ratios), 0, 255).astype(np.uint8)
        out[ys, xs, 0] = greys
        out[ys, xs, 1] = greys
        out[ys, xs, 2] = greys

    return out


def stub_grounded_sam_if_needed(skip_sam: bool) -> None:
    if not skip_sam:
        return
    from src.fluid_mask import refine_mask as refine_module

    def _noop(image, text_prompt: str):
        h, w = np.asarray(image).shape[:2]
        return np.zeros((h, w), dtype=np.uint8)

    refine_module.refine_fluid_mask = _noop


def run_pipeline(
    structural_input,
    motion_input,
    prompt: str,
    scene_backend: str,
    motion_backend: str,
    controlnet_model: str,
    image_size: int,
    num_inference_steps: int,
    controlnet_scale: float,
    motion_scale: float,
    num_frames: int,
    skip_grounded_sam: bool,
    auto_gradient: bool,
    photo_input,
):
    """Invoked by the GUI 'Generate' button."""
    log: list[str] = []
    t0 = time.time()

    structural_np = _ensure_white_background(_editor_to_numpy(structural_input))
    motion_np = _ensure_white_background(_editor_to_numpy(motion_input))
    photo_np = _editor_to_numpy(photo_input) if photo_input is not None else None

    # Paper §5.6 photo mode: uploaded photo skips scene gen; blank structural
    # canvas forces the semantic mask to fall back to dilated motion strokes.
    photo_mode = photo_np is not None

    if motion_np is None:
        return (None, None, None, None, None, None,
                "❌ Need a motion sketch (drawing of flow strokes).")
    if not photo_mode and structural_np is None:
        return (None, None, None, None, None, None,
                "❌ Need either a structural sketch OR an uploaded photo.")
    if photo_mode:
        structural_np = np.full(
            (motion_np.shape[0], motion_np.shape[1], 3), 255, dtype=np.uint8
        )
        # Force Grounded-SAM ON so the photo still gets a clean fluid mask.
        if skip_grounded_sam:
            log.append("Photo mode: enabling Grounded-SAM for fluid-region detection.")
            skip_grounded_sam = False

    if auto_gradient:
        motion_np = apply_motion_gradient(motion_np)
        log.append("Auto-applied white→black gradient to motion strokes.")

    structural_np = cv2.resize(structural_np, (image_size, image_size),
                               interpolation=cv2.INTER_AREA)
    motion_np = cv2.resize(motion_np, (image_size, image_size),
                           interpolation=cv2.INTER_AREA)
    if photo_mode:
        photo_np = cv2.resize(photo_np, (image_size, image_size),
                              interpolation=cv2.INTER_AREA)

    stub_grounded_sam_if_needed(skip_grounded_sam)

    cfg = {
        "scene": {
            "backend": scene_backend,
            "image_size": [image_size, image_size],
            "save_reference": False,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": 7.5,
            "controlnet_conditioning_scale": controlnet_scale,
            "controlnet_id": {
                "lineart_v1.1": "lllyasviel/control_v11p_sd15_lineart",
                "scribble_v1.0": "lllyasviel/sd-controlnet-scribble",
            }.get(controlnet_model, "lllyasviel/control_v11p_sd15_lineart"),
        },
        "motion": {"backend": motion_backend},
        "synthesis": {
            "num_frames": num_frames,
            "fps": 18,
            "loop_mode": "linear",
            "motion_scale": motion_scale,
            "border_mode": "reflect",
        },
        "output": {
            "dir": str(Path(tempfile.mkdtemp(prefix="cinemagraph_"))),
            "video_name": "cinemagraph.mp4",
            "gif_name": "cinemagraph.gif",
        },
    }
    if motion_backend == "learned":
        cfg["motion"]["learned"] = {
            "checkpoint_path": "checkpoints/best.pt",
            "device": "mps",
            "input_size": 256,
        }

    try:
        from src.pipeline import CinemagraphPipeline
        from src.evaluation.visualize import visualize_mask, visualize_motion_field

        if photo_mode:
            log.append("Mode: photo input (skipping scene generation)")
        else:
            log.append(f"Scene backend: {scene_backend}")
            if scene_backend == "diffusion":
                log.append("(diffusion) loading SD + ControlNet — first run downloads ~5 GB")
        log.append(f"Motion backend: {motion_backend}")

        pipeline = CinemagraphPipeline(cfg)
        result = pipeline.run(
            structural_np, motion_np, prompt,
            pre_generated_image=photo_np if photo_mode else None,
        )

        stylized = result["stylized_image"]
        mask_vis = visualize_mask(result["final_fluid_mask"])
        flow_vis = visualize_motion_field(result["dense_motion_field"])

        flow = result["dense_motion_field"]
        m = result["final_fluid_mask"] > 0
        if m.any():
            mag = np.linalg.norm(flow, axis=-1)[m]
            log.append(f"In-mask flow: mean={mag.mean():.2f}px max={mag.max():.2f}px")
        log.append(f"Generated {len(result['frames'])} frames")
        log.append(f"GIF: {result['gif_path']}")
        log.append(f"MP4: {result['video_path']}")
        log.append(f"Total time: {time.time() - t0:.1f}s")

        return (
            structural_np,
            motion_np,
            stylized,
            mask_vis,
            flow_vis,
            result["gif_path"],
            "\n".join(log),
        )
    except Exception as error:
        tb = traceback.format_exc()
        log.append(f"❌ {error}")
        log.append(tb)
        return (None, None, None, None, None, None, "\n".join(log))


def build_app():
    import gradio as gr

    with gr.Blocks(title="Sketch2Cinemagraph (course MVP)") as app:
        gr.Markdown(
            "# Sketch-Guided Stylized Landscape Cinemagraph\n"
            "Type a prompt, draw a **structural sketch** (scene layout) and a "
            "**motion sketch** (white→black gradient = flow direction), then click *Generate*."
        )

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    value="ocean waves under blue sky",
                    placeholder="e.g. waterfall in the mountains, oil painting",
                )

                gr.Markdown(
                    "**Choose ONE input mode below.** "
                    "Either draw a structural sketch (will be turned into a "
                    "stylized scene by SD/ControlNet), OR upload a real "
                    "landscape photo to animate directly."
                )

                with gr.Tab("A. Structural sketch (sketch-to-cinemagraph)"):
                    gr.Markdown("Outline mountains, horizon, riverbanks. Used by Stable Diffusion to generate the stylized scene.")
                    structural = gr.ImageEditor(
                        label="Structural sketch",
                        value=None,
                        sources=("upload",),
                        height=320,
                        type="numpy",
                        canvas_size=(384, 384),
                        image_mode="RGBA",
                        brush=gr.Brush(default_color="#000000",
                                       default_size=3,
                                       colors=["#000000"]),
                    )

                with gr.Tab("B. Real photo (image-to-cinemagraph)"):
                    gr.Markdown(
                        "Upload a landscape photograph. The pipeline skips "
                        "Stable Diffusion and animates the photo directly "
                        "(paper §5.6). The text prompt is still used by "
                        "Grounded-SAM to localise the fluid region "
                        "(e.g. 'water', 'cloud', 'smoke')."
                    )
                    photo = gr.Image(
                        label="Landscape photo (overrides structural sketch when set)",
                        type="numpy",
                        sources=("upload", "clipboard"),
                        height=320,
                    )

                gr.Markdown(
                    "**Motion sketch** — draw flow strokes inside fluid regions. "
                    "Auto-gradient turns them white→black left-to-right."
                )
                motion = gr.ImageEditor(
                    label="Motion sketch",
                    value=None,
                    sources=("upload",),
                    height=320,
                    type="numpy",
                    canvas_size=(384, 384),
                    image_mode="RGBA",
                    brush=gr.Brush(default_color="#000000",
                                   default_size=4,
                                   colors=["#000000"]),
                )

                with gr.Row():
                    scene_backend = gr.Radio(
                        ["placeholder", "diffusion"],
                        value="placeholder",
                        label="Scene backend",
                        info="placeholder = instant; diffusion = SD+ControlNet (slow first run)",
                    )
                    motion_backend = gr.Radio(
                        ["heuristic", "learned"],
                        value="heuristic",
                        label="Motion backend",
                    )
                controlnet_model = gr.Radio(
                    ["lineart_v1.1", "scribble_v1.0"],
                    value="lineart_v1.1",
                    label="ControlNet model (diffusion only)",
                    info="lineart = best for line drawings; scribble = original HED-based",
                )

                with gr.Accordion("Advanced", open=False):
                    image_size = gr.Slider(256, 768, value=384, step=64, label="Image size")
                    num_inference_steps = gr.Slider(10, 50, value=25, step=1,
                                                    label="SD steps (diffusion only)")
                    controlnet_scale = gr.Slider(0.5, 2.0, value=1.0, step=0.1,
                                                  label="ControlNet scale (0.7-1.0 = natural, 1.2+ = strict structure)")
                    motion_scale = gr.Slider(0.2, 6.0, value=1.5, step=0.1,
                                             label="Motion scale (max per-cycle displacement multiplier)")
                    num_frames = gr.Slider(20, 120, value=60, step=2, label="Frame count")
                    skip_grounded_sam = gr.Checkbox(value=True,
                                                    label="Skip Grounded-SAM mask refine (saves a 1 GB download)")
                    auto_gradient = gr.Checkbox(value=True,
                                                label="Auto-apply white→black gradient to motion strokes")

                run_btn = gr.Button("Generate cinemagraph", variant="primary")

            with gr.Column(scale=1):
                with gr.Tab("Cinemagraph"):
                    out_gif = gr.Image(label="Looping cinemagraph", height=420)
                with gr.Tab("Stylized image"):
                    out_stylized = gr.Image(label="Stylized landscape", height=420)
                with gr.Tab("Fluid mask"):
                    out_mask = gr.Image(label="Final fluid mask", height=420)
                with gr.Tab("Motion field"):
                    out_flow = gr.Image(label="Dense motion field (HSV)", height=420)
                with gr.Tab("Inputs (after preprocessing)"):
                    with gr.Row():
                        out_struct = gr.Image(label="Structural", height=300)
                        out_motion = gr.Image(label="Motion (gradient applied)", height=300)
                status = gr.Textbox(label="Log", lines=10, interactive=False)

        run_btn.click(
            run_pipeline,
            inputs=[
                structural, motion, prompt,
                scene_backend, motion_backend, controlnet_model,
                image_size, num_inference_steps, controlnet_scale,
                motion_scale, num_frames,
                skip_grounded_sam, auto_gradient,
                photo,
            ],
            outputs=[
                out_struct, out_motion,
                out_stylized, out_mask, out_flow, out_gif,
                status,
            ],
        )

        gr.Markdown(
            "---\n"
            "**Tips**\n"
            "- *placeholder* scene backend gives a fast preview but flat colours; switch to "
            "*diffusion* for an actual stylized landscape (first run downloads ~5 GB).\n"
            "- *heuristic* motion backend is the proposal §3.5 main version; *learned* uses "
            "the U-Net trained by `scripts/train_learned_predictor.py`.\n"
            "- For directionally-controlled motion strokes, use "
            "`python scripts/draw_motion_sketch.py` and upload the resulting PNG."
        )

    return app


def main() -> None:
    args = parse_args()
    app = build_app()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
