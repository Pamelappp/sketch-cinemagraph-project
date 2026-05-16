"""High-quality SD 1.5 + ControlNet scene generator.

Quality choices (vs the previous draft):
- Base model defaults to ``Lykon/dreamshaper-8`` — a SD 1.5 fine-tune that
  produces noticeably better landscape painterly output than vanilla SD 1.5
  while still being lightweight enough to run on Apple Silicon (MPS).
- External VAE (``stabilityai/sd-vae-ft-mse``) overrides the baked-in VAE
  for sharper colours and fewer high-saturation artifacts.
- DPM++ 2M with Karras sigmas — consensus best scheduler for SD 1.5 quality
  at modest step counts (25-30 here vs 40+ on UniPC).
- Free-U (b1=1.1, b2=1.2, s1=0.9, s2=0.2) — community-tuned values that
  improve coherence/detail with zero extra cost.
- Two-pass refinement: ControlNet generates the structural pass, then an
  img2img pass at low strength (~0.35) sharpens texture without moving
  the composition. Disabled per call by ``refine=False``.
- MPS memory savers (attention slicing + VAE slicing/tiling) keep peak
  memory in check on Apple Silicon's unified RAM.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np


_UNSET = object()  # distinguishes "caller passed nothing" from "caller passed None"


_DEFAULT_MODEL_ID     = os.environ.get("SD_MODEL_ID", "Lykon/dreamshaper-8")
_DEFAULT_CONTROLNET   = os.environ.get("CONTROLNET_ID", "lllyasviel/control_v11p_sd15_lineart")
_DEFAULT_VAE          = os.environ.get("SD_VAE_ID", "stabilityai/sd-vae-ft-mse")


class DiffusionSceneBackend:
    """Stable-Diffusion-1.5-finetune + ControlNet + img2img refinement."""

    def __init__(self, cfg: dict) -> None:
        self.model_id      = cfg.get("model_id",      _DEFAULT_MODEL_ID)
        self.controlnet_id = cfg.get("controlnet_id", _DEFAULT_CONTROLNET)
        self.vae_id        = cfg.get("vae_id",        _DEFAULT_VAE)

        self.num_inference_steps           = int(cfg.get("num_inference_steps", 30))
        self.guidance_scale                = float(cfg.get("guidance_scale", 6.5))
        self.controlnet_conditioning_scale = float(cfg.get("controlnet_conditioning_scale", 0.9))
        self.control_guidance_start        = float(cfg.get("control_guidance_start", 0.0))
        self.control_guidance_end          = float(cfg.get("control_guidance_end", 0.75))

        # Two-pass refinement
        self.use_refine     = bool(cfg.get("use_refine", True))
        self.refine_strength = float(cfg.get("refine_strength", 0.35))
        self.refine_steps   = int(cfg.get("refine_steps", 20))

        # Quality knobs
        self.use_freeu      = bool(cfg.get("use_freeu", True))
        self.scheduler_name = str(cfg.get("scheduler", "dpmpp_2m_karras"))
        self.clip_skip      = int(cfg.get("clip_skip", 1))

        self._cfg_negative_prompt = cfg.get("negative_prompt", None)

        raw_seed   = cfg.get("seed", 42)
        self.seed  = None if raw_seed is None else int(raw_seed)
        self.image_size         = cfg.get("image_size", None)
        self.enable_cpu_offload = bool(cfg.get("enable_cpu_offload", False))

        self._cn_pipe = None        # ControlNet pipeline (structural pass)
        self._i2i_pipe = None       # img2img pipeline (refinement pass)
        self._torch = None
        self._device = None
        self._dtype = None
        self._lock = threading.Lock()

    def generate(
        self,
        prompt: str,
        control_image,
        seed=_UNSET,
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        controlnet_conditioning_scale: Optional[float] = None,
        control_guidance_start: Optional[float] = None,
        control_guidance_end: Optional[float] = None,
        refine: Optional[bool] = None,
    ) -> np.ndarray:
        """Run the structural pass and (optionally) an img2img refinement pass."""
        self._ensure_loaded()
        torch = self._torch

        actual_seed   = self.seed if seed is _UNSET else seed
        actual_steps  = int(num_inference_steps if num_inference_steps is not None else self.num_inference_steps)
        actual_cfg    = float(guidance_scale if guidance_scale is not None else self.guidance_scale)
        actual_ccs    = float(controlnet_conditioning_scale if controlnet_conditioning_scale is not None else self.controlnet_conditioning_scale)
        actual_cgs    = float(control_guidance_start if control_guidance_start is not None else self.control_guidance_start)
        actual_cge    = float(control_guidance_end if control_guidance_end is not None else self.control_guidance_end)
        neg           = negative_prompt if negative_prompt is not None else self._cfg_negative_prompt
        do_refine     = self.use_refine if refine is None else bool(refine)

        generator = (
            torch.Generator(device="cpu").manual_seed(actual_seed)
            if actual_seed is not None else None
        )
        width, height = control_image.size

        # ── Structural pass: SD + ControlNet ────────────────────────────────
        base_image = self._cn_pipe(
            prompt=prompt,
            image=control_image,
            negative_prompt=neg,
            num_inference_steps=actual_steps,
            guidance_scale=actual_cfg,
            controlnet_conditioning_scale=actual_ccs,
            control_guidance_start=actual_cgs,
            control_guidance_end=actual_cge,
            generator=generator,
            height=height,
            width=width,
            clip_skip=self.clip_skip if self.clip_skip > 1 else None,
        ).images[0]

        if not do_refine:
            return np.asarray(base_image.convert("RGB"))

        # ── Refinement pass: img2img at low strength ────────────────────────
        # Re-use the structural-pass seed so refinement is deterministic
        # relative to the same prompt + sketch.
        refine_generator = (
            torch.Generator(device="cpu").manual_seed(actual_seed + 7919)
            if actual_seed is not None else None
        )
        refined = self._i2i_pipe(
            prompt=prompt,
            image=base_image,
            negative_prompt=neg,
            strength=self.refine_strength,
            num_inference_steps=self.refine_steps,
            guidance_scale=actual_cfg,
            generator=refine_generator,
            clip_skip=self.clip_skip if self.clip_skip > 1 else None,
        ).images[0]

        return np.asarray(refined.convert("RGB"))

    def _ensure_loaded(self) -> None:
        if self._cn_pipe is not None and self._i2i_pipe is not None:
            return
        with self._lock:
            if self._cn_pipe is not None and self._i2i_pipe is not None:
                return
            self._load_pipelines()

    def _load_pipelines(self) -> None:
        import torch
        from diffusers import (
            AutoencoderKL,
            ControlNetModel,
            StableDiffusionControlNetPipeline,
            StableDiffusionImg2ImgPipeline,
            DPMSolverMultistepScheduler,
            UniPCMultistepScheduler,
        )

        self._torch = torch

        if torch.cuda.is_available():
            device, dtype = "cuda", torch.float16
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            # MPS fp16 attention is buggy on some torch builds; fp32 is the safe default.
            device, dtype = "mps", torch.float32
        else:
            device, dtype = "cpu", torch.float32
        self._device, self._dtype = device, dtype

        # ── ControlNet pass ─────────────────────────────────────────────────
        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_id, torch_dtype=dtype
        )

        # Use an external VAE when configured. The shipped SD 1.5 VAE bakes in
        # a slight contrast crush; sd-vae-ft-mse is the community standard
        # replacement and gives noticeably crisper colour.
        vae = None
        if self.vae_id:
            try:
                vae = AutoencoderKL.from_pretrained(self.vae_id, torch_dtype=dtype)
            except Exception as err:
                print(f"[diffusion] VAE override failed ({err}); falling back to baked-in VAE.")
                vae = None

        kwargs = dict(
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        if vae is not None:
            kwargs["vae"] = vae

        cn_pipe = StableDiffusionControlNetPipeline.from_pretrained(self.model_id, **kwargs)
        cn_pipe.scheduler = self._build_scheduler(cn_pipe.scheduler.config,
                                                   DPMSolverMultistepScheduler,
                                                   UniPCMultistepScheduler)

        if self.use_freeu:
            # Community-tuned values for SD 1.5 painterly output.
            cn_pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.1, b2=1.2)

        # ── img2img pass ────────────────────────────────────────────────────
        # Share the same UNet/VAE/text-encoder weights — the i2i pipeline is
        # just a wrapper, not a second model in memory.
        i2i_pipe = StableDiffusionImg2ImgPipeline(
            vae=cn_pipe.vae,
            text_encoder=cn_pipe.text_encoder,
            tokenizer=cn_pipe.tokenizer,
            unet=cn_pipe.unet,
            scheduler=cn_pipe.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        if self.use_freeu:
            i2i_pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.1, b2=1.2)

        self._place_pipelines(cn_pipe, i2i_pipe, device)
        self._cn_pipe = cn_pipe
        self._i2i_pipe = i2i_pipe

    def _build_scheduler(self, base_config, DPM, UniPC):
        """Pick scheduler by name; default DPM++ 2M Karras."""
        name = self.scheduler_name.lower()
        if name in ("dpmpp_2m_karras", "dpm++_2m_karras", "dpm++2m_karras"):
            return DPM.from_config(
                base_config,
                algorithm_type="dpmsolver++",
                solver_order=2,
                use_karras_sigmas=True,
            )
        if name in ("dpmpp_2m", "dpm++_2m"):
            return DPM.from_config(
                base_config,
                algorithm_type="dpmsolver++",
                solver_order=2,
            )
        # Fallback to UniPC (the old default).
        return UniPC.from_config(base_config)

    def _place_pipelines(self, cn_pipe, i2i_pipe, device: str) -> None:
        """Move pipelines to device and enable memory savers."""
        if device == "cuda" and self.enable_cpu_offload:
            cn_pipe.enable_model_cpu_offload()
            i2i_pipe.enable_model_cpu_offload()
        else:
            cn_pipe.to(device)
            i2i_pipe.to(device)

        # MPS unified memory benefits a lot from these; CUDA also fine.
        for pipe in (cn_pipe, i2i_pipe):
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass
            try:
                pipe.enable_vae_tiling()
            except Exception:
                pass
