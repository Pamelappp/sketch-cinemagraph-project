import numpy as np

_UNSET = object()   # distinguishes "caller passed nothing" from "caller passed None"


class DiffusionSceneBackend:
    def __init__(self, cfg: dict) -> None:
        self.model_id      = cfg.get("model_id",      "runwayml/stable-diffusion-v1-5")
        self.controlnet_id = cfg.get("controlnet_id", "lllyasviel/control_v11p_sd15_lineart")

        self.num_inference_steps           = int(cfg.get("num_inference_steps", 40))
        self.guidance_scale                = float(cfg.get("guidance_scale", 7.5))
        self.controlnet_conditioning_scale = float(cfg.get("controlnet_conditioning_scale", 1.0))
        self.control_guidance_start        = float(cfg.get("control_guidance_start", 0.0))
        self.control_guidance_end          = float(cfg.get("control_guidance_end", 0.85))

        self._cfg_negative_prompt = cfg.get("negative_prompt", None)

        raw_seed    = cfg.get("seed", 42)
        self.seed   = None if raw_seed is None else int(raw_seed)
        self.image_size        = cfg.get("image_size", None)
        self.enable_cpu_offload = bool(cfg.get("enable_cpu_offload", True))

        self._pipe  = None
        self._torch = None

    def generate(self,
                 prompt: str,
                 control_image,
                 seed=_UNSET,
                 negative_prompt: str | None = None,
                 num_inference_steps: int | None = None,
                 guidance_scale: float | None = None,
                 controlnet_conditioning_scale: float | None = None,
                 control_guidance_start: float | None = None,
                 control_guidance_end: float | None = None) -> np.ndarray:
        """Generate one image; per-call params override instance defaults when not None.

        seed: int overrides self.seed; None forces random; _UNSET reuses self.seed.
        """
        pipe  = self._get_pipeline()
        torch = self._torch

        actual_seed  = self.seed if seed is _UNSET else seed
        actual_steps = num_inference_steps if num_inference_steps is not None else self.num_inference_steps
        actual_cfg   = guidance_scale if guidance_scale is not None else self.guidance_scale
        actual_ccs   = controlnet_conditioning_scale if controlnet_conditioning_scale is not None else self.controlnet_conditioning_scale
        actual_cgs   = control_guidance_start if control_guidance_start is not None else self.control_guidance_start
        actual_cge   = control_guidance_end if control_guidance_end is not None else self.control_guidance_end
        neg          = negative_prompt if negative_prompt is not None else self._cfg_negative_prompt

        generator = (
            torch.Generator(device="cpu").manual_seed(actual_seed)
            if actual_seed is not None else None
        )

        width, height = control_image.size

        kwargs = dict(
            prompt=prompt,
            image=control_image,
            negative_prompt=neg,
            num_inference_steps=actual_steps,
            guidance_scale=actual_cfg,
            controlnet_conditioning_scale=actual_ccs,
            generator=generator,
            height=height,
            width=width,
        )

        # control_guidance_start/end require diffusers ≥ 0.19; fall back gracefully.
        try:
            kwargs["control_guidance_start"] = actual_cgs
            kwargs["control_guidance_end"]   = actual_cge
            result = pipe(**kwargs)
        except TypeError:
            del kwargs["control_guidance_start"], kwargs["control_guidance_end"]
            result = pipe(**kwargs)

        return np.asarray(result.images[0].convert("RGB"))

    def _get_pipeline(self):
        if self._pipe is not None:
            return self._pipe

        import torch
        from diffusers import (
            ControlNetModel,
            StableDiffusionControlNetPipeline,
            UniPCMultistepScheduler,
        )

        self._torch = torch

        if torch.cuda.is_available():
            device      = "cuda"
            torch_dtype = torch.float16
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            # Apple Silicon GPU. fp16 attention is unstable on MPS; keep fp32.
            device      = "mps"
            torch_dtype = torch.float32
        else:
            device      = "cpu"
            torch_dtype = torch.float32

        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_id,
            torch_dtype=torch_dtype,
        )

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.model_id,
            controlnet=controlnet,
            torch_dtype=torch_dtype,
        )

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        if device == "cuda":
            if self.enable_cpu_offload:
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(device)
        else:
            pipe.to(device)

        self._pipe = pipe
        return self._pipe
