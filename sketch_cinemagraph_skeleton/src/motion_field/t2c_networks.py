"""T2C motion-direction U-Net with SPADE conditioning.

Architecture reverse-engineered from
  checkpoints/motion-direction-pretrained/latest_net_G.pth

Encoder : 8 × SN-Conv(4×4, stride-2)  — 6 → 32 → 64 → 128 → 256×5
Decoder : 8 × SN-Conv(3×3) preceded by 2× bilinear upsample + U-Net skip-cat
SPADE   : post-conv spatial feature modulation conditioned on the 6-ch input

At 256×256 input the bottleneck is 1×1; the decoder restores to 128×128,
so the output flow is at half input resolution and must be upsampled with
matching scale correction before use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class SPADE(nn.Module):
    """Spatially Adaptive Normalisation conditioned on a 6-channel map."""

    def __init__(self, norm_nc: int, cond_nc: int = 6) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(norm_nc, affine=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(cond_nc, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mlp_gamma = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta  = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, segmap: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        seg   = F.interpolate(segmap, size=(H, W), mode="nearest")
        x_n   = self.norm(x)
        actv  = self.mlp_shared(seg)
        gamma = self.mlp_gamma(actv)
        beta  = self.mlp_beta(actv)
        return x_n * (1.0 + gamma) + beta


class T2CDirectionFlowNet(nn.Module):
    """Spectral-norm U-Net that predicts (dx, dy) flow from image + sketch.

    Input  : (B, 6, H, W)  — RGB image || RGB sketch, normalised to [-1, 1]
    Output : (B, 2, H/2, W/2) — raw (dx, dy) flow at half input resolution
    """

    def __init__(self) -> None:
        super().__init__()

        # ── Encoder (stride-2 SN-Conv, 4×4 kernel, padding=1) ────────────────
        self.conv1 = spectral_norm(nn.Conv2d(6,   32,  4, stride=2, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(32,  64,  4, stride=2, padding=1))
        self.conv3 = spectral_norm(nn.Conv2d(64,  128, 4, stride=2, padding=1))
        self.conv4 = spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1))
        self.conv5 = spectral_norm(nn.Conv2d(256, 256, 4, stride=2, padding=1))
        self.conv6 = spectral_norm(nn.Conv2d(256, 256, 4, stride=2, padding=1))
        self.conv7 = spectral_norm(nn.Conv2d(256, 256, 4, stride=2, padding=1))
        self.conv8 = spectral_norm(nn.Conv2d(256, 256, 4, stride=2, padding=1))

        # ── Decoder (SN-Conv, 3×3 kernel, padding=1; no stride) ──────────────
        # Channel arithmetic: upsample(prev) || skip → in_ch → out_ch
        self.dconv1 = spectral_norm(nn.Conv2d(256, 256, 3, padding=1))  # no skip
        self.dconv2 = spectral_norm(nn.Conv2d(512, 256, 3, padding=1))  # +e7(256)
        self.dconv3 = spectral_norm(nn.Conv2d(512, 256, 3, padding=1))  # +e6(256)
        self.dconv4 = spectral_norm(nn.Conv2d(512, 256, 3, padding=1))  # +e5(256)
        self.dconv5 = spectral_norm(nn.Conv2d(512, 128, 3, padding=1))  # +e4(256)
        self.dconv6 = spectral_norm(nn.Conv2d(256, 64,  3, padding=1))  # +e3(128)
        self.dconv7 = spectral_norm(nn.Conv2d(128, 32,  3, padding=1))  # +e2(64)
        self.dconv8 = spectral_norm(nn.Conv2d(64,  2,   3, padding=1))  # +e1(32)

        # ── SPADE modules (post-conv, conditioned on the 6-ch input) ─────────
        # 256-ch blocks (dconv1-4): 2 SPADEs each → 8 total
        self.spade_layer8_0 = SPADE(256)
        self.spade_layer8_1 = SPADE(256)
        self.spade_layer8_2 = SPADE(256)
        self.spade_layer8_3 = SPADE(256)
        self.spade_layer8_4 = SPADE(256)
        self.spade_layer8_5 = SPADE(256)
        self.spade_layer8_6 = SPADE(256)
        self.spade_layer8_7 = SPADE(256)

        # 128-ch block (dconv5): 2 SPADEs
        self.spade_layer4_0 = SPADE(128)
        self.spade_layer4_1 = SPADE(128)

        # 64-ch block (dconv6): 2 SPADEs
        self.spade_layer2_0 = SPADE(64)
        self.spade_layer2_1 = SPADE(64)

        # 32-ch block (dconv7): 1 SPADE
        self.spade_layer = SPADE(32)

        # dconv8 (2-ch output): no SPADE — raw flow output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 6, H, W) normalised to [-1, 1]

        Returns:
            flow: (B, 2, H/2, W/2) raw (dx, dy) values
        """
        cond = x  # re-use as SPADE conditioning map

        # ── Encoder ───────────────────────────────────────────────────────────
        e1 = F.leaky_relu(self.conv1(x),  0.2)
        e2 = F.leaky_relu(self.conv2(e1), 0.2)
        e3 = F.leaky_relu(self.conv3(e2), 0.2)
        e4 = F.leaky_relu(self.conv4(e3), 0.2)
        e5 = F.leaky_relu(self.conv5(e4), 0.2)
        e6 = F.leaky_relu(self.conv6(e5), 0.2)
        e7 = F.leaky_relu(self.conv7(e6), 0.2)
        e8 = F.leaky_relu(self.conv8(e7), 0.2)  # bottleneck: 1×1 for 256px input

        # ── Decoder: conv → SPADE → LReLU  (×2 for 256/128/64-ch, ×1 for 32-ch)
        d1 = self.dconv1(e8)
        d1 = F.leaky_relu(self.spade_layer8_0(d1, cond), 0.2)
        d1 = F.leaky_relu(self.spade_layer8_1(d1, cond), 0.2)

        d2 = self.dconv2(torch.cat([_up(d1), e7], dim=1))
        d2 = F.leaky_relu(self.spade_layer8_2(d2, cond), 0.2)
        d2 = F.leaky_relu(self.spade_layer8_3(d2, cond), 0.2)

        d3 = self.dconv3(torch.cat([_up(d2), e6], dim=1))
        d3 = F.leaky_relu(self.spade_layer8_4(d3, cond), 0.2)
        d3 = F.leaky_relu(self.spade_layer8_5(d3, cond), 0.2)

        d4 = self.dconv4(torch.cat([_up(d3), e5], dim=1))
        d4 = F.leaky_relu(self.spade_layer8_6(d4, cond), 0.2)
        d4 = F.leaky_relu(self.spade_layer8_7(d4, cond), 0.2)

        d5 = self.dconv5(torch.cat([_up(d4), e4], dim=1))
        d5 = F.leaky_relu(self.spade_layer4_0(d5, cond), 0.2)
        d5 = F.leaky_relu(self.spade_layer4_1(d5, cond), 0.2)

        d6 = self.dconv6(torch.cat([_up(d5), e3], dim=1))
        d6 = F.leaky_relu(self.spade_layer2_0(d6, cond), 0.2)
        d6 = F.leaky_relu(self.spade_layer2_1(d6, cond), 0.2)

        d7 = self.dconv7(torch.cat([_up(d6), e2], dim=1))
        d7 = F.leaky_relu(self.spade_layer(d7, cond), 0.2)

        d8 = self.dconv8(torch.cat([_up(d7), e1], dim=1))
        return d8  # (B, 2, H/2, W/2) raw flow


def _up(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


def load_t2c_direction_net(
    checkpoint_path: str,
    device: str = "cpu",
) -> T2CDirectionFlowNet:
    """Instantiate and load weights; returns model in eval mode."""
    model = T2CDirectionFlowNet()
    state = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        tag = missing[:4]
        print(f"[T2C] {len(missing)} missing keys (first 4): {tag}")
    if unexpected:
        tag = unexpected[:4]
        print(f"[T2C] {len(unexpected)} unexpected keys (first 4): {tag}")
    model.to(device)
    model.eval()
    return model
