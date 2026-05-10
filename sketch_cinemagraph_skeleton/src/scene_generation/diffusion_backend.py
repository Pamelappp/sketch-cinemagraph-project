# src/scene_generation/diffusion_backend.py

import numpy as np


class DiffusionSceneBackend:
    def __init__(self, cfg: dict) -> None:
        self.model_id = cfg.get("model_id", "runwayml/stable-diffusion-v1-5")
        self.controlnet_id = cfg.get("controlnet_id", "lllyasviel/sd-controlnet-canny")
        self.num_inference_steps = int(cfg.get("num_inference_steps", 30))
        self.guidance_scale = float(cfg.get("guidance_scale", 7.5))
        self.controlnet_conditioning_scale = float(cfg.get("controlnet_conditioning_scale", 1.0))
        self.negative_prompt = cfg.get(
            "negative_prompt",
            "low quality, blurry, distorted, extra objects, messy composition",
        )
        self.seed = int(cfg.get("seed", 42))
        self.image_size = tuple(cfg.get("image_size", [512, 512]))
        self.enable_cpu_offload = bool(cfg.get("enable_cpu_offload", True))

        self._pipe = None
        self._torch = None

    def generate(self, prompt: str, control_image) -> np.ndarray:
        pipe = self._get_pipeline()
        torch = self._torch

        generator = torch.Generator(device="cpu").manual_seed(self.seed)

        result = pipe(
            prompt=prompt,
            image=control_image,
            negative_prompt=self.negative_prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            controlnet_conditioning_scale=self.controlnet_conditioning_scale,
            generator=generator,
            height=self.image_size[1],
            width=self.image_size[0],
        )

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
            device = "cuda"
            torch_dtype = torch.float16
        else:
            device = "cpu"
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