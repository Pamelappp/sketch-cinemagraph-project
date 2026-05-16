"""Neural network definitions for the learned motion predictor."""

from __future__ import annotations

import torch
from torch import nn


class _DoubleConv(nn.Module):
    """Two 3x3 convolutions with GroupNorm + ReLU."""

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8) -> None:
        super().__init__()
        groups = min(num_groups, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Down(nn.Module):
    """MaxPool + double conv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = _DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class _Up(nn.Module):
    """Bilinear upsample, 1x1 reduce, concat with skip, double conv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = _DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.reduce(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MotionUNet(nn.Module):
    """
    Lightweight U-Net that predicts a dense motion field from three
    conditioning inputs stacked into a single tensor.

    Input layout (B, 7, H, W):
        - channels 0..2 : stylized landscape image (RGB, normalised to [0, 1])
        - channels 3..5 : motion sketch (RGB, normalised to [0, 1])
        - channel  6    : fluid mask (binary {0, 1})

    Output (B, 2, H, W): flow with channels (dx, dy) in pixel units.
    """

    def __init__(self, in_channels: int = 7, base_channels: int = 64) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        )

        self.in_conv = _DoubleConv(in_channels, c1)
        self.down1 = _Down(c1, c2)
        self.down2 = _Down(c2, c3)
        self.down3 = _Down(c3, c4)
        self.down4 = _Down(c4, c5)

        self.up1 = _Up(c5, c4, c4)
        self.up2 = _Up(c4, c3, c3)
        self.up3 = _Up(c3, c2, c2)
        self.up4 = _Up(c2, c1, c1)

        self.out_conv = nn.Conv2d(c1, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.in_conv(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        bottleneck = self.down4(s4)

        x = self.up1(bottleneck, s4)
        x = self.up2(x, s3)
        x = self.up3(x, s2)
        x = self.up4(x, s1)
        return self.out_conv(x)


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_device(prefer: str | None = None) -> torch.device:
    """
    Pick the best available torch device.

    Order of preference: explicit ``prefer`` argument > CUDA > Apple MPS > CPU.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
