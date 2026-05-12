"""Train ``MotionUNet`` on (image, sketch, mask) -> motion field tuples.

Default configuration runs on the synthetic data generator so the pipeline
can be smoke-tested without any external dataset. Pass ``--data-root`` to
train on real samples produced by ``scripts/prepare_dataset.py``.

Typical commands::

    # Quick sanity run (synthetic, ~10 minutes on M4 Max)
    python scripts/train_learned_predictor.py --epochs 5 --num-samples 256

    # Longer synthetic training
    python scripts/train_learned_predictor.py --epochs 50

    # Real data training
    python scripts/train_learned_predictor.py \
        --data-root data/training/landscape \
        --val-data-root data/training/landscape_val \
        --epochs 80 --batch-size 8

Checkpoints are saved under ``checkpoints/`` by default; the best
validation loss is mirrored to ``checkpoints/best.pt`` and the latest
state to ``checkpoints/last.pt``.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.motion_field.networks import MotionUNet, count_parameters, resolve_device
from src.motion_field.training_data import (
    LandscapeMotionDataset,
    SyntheticConfig,
    SyntheticMotionDataset,
)


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    image_size: int
    base_channels: int
    num_workers: int
    val_split: float
    log_every: int
    save_every: int
    output_dir: Path
    device: Optional[str]
    data_root: Optional[Path]
    val_data_root: Optional[Path]
    num_samples: int
    seed: int
    augment: bool
    flow_scale: float
    tv_weight: float


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train MotionUNet for sketch-conditioned motion prediction.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-split", type=float, default=0.1, help="Used only for synthetic data.")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1, help="Save 'last.pt' every N epochs.")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--device", type=str, default=None, help='Force device: "cuda", "mps" or "cpu".')
    parser.add_argument("--data-root", type=Path, default=None, help="Directory of real training samples.")
    parser.add_argument("--val-data-root", type=Path, default=None, help="Directory of real validation samples.")
    parser.add_argument("--num-samples", type=int, default=2000, help="Synthetic dataset size when --data-root is not set.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true", help="Disable horizontal-flip augmentation for real data.")
    parser.add_argument("--flow-scale", type=float, default=1.0, help="Multiplier applied to ground-truth flows.")
    parser.add_argument("--tv-weight", type=float, default=0.05,
                        help="Weight on the total-variation smoothness regulariser. Set 0 to disable.")
    args = parser.parse_args()

    return TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        image_size=args.image_size,
        base_channels=args.base_channels,
        num_workers=args.num_workers,
        val_split=args.val_split,
        log_every=args.log_every,
        save_every=args.save_every,
        output_dir=args.output_dir,
        device=args.device,
        data_root=args.data_root,
        val_data_root=args.val_data_root,
        num_samples=args.num_samples,
        seed=args.seed,
        augment=not args.no_augment,
        flow_scale=args.flow_scale,
        tv_weight=args.tv_weight,
    )


def build_datasets(cfg: TrainConfig):
    if cfg.data_root is not None:
        train_set = LandscapeMotionDataset(
            cfg.data_root,
            image_size=cfg.image_size,
            flow_scale=cfg.flow_scale,
            augment=cfg.augment,
        )
        if cfg.val_data_root is not None:
            val_set = LandscapeMotionDataset(
                cfg.val_data_root,
                image_size=cfg.image_size,
                flow_scale=cfg.flow_scale,
                augment=False,
            )
        else:
            val_size = max(1, int(len(train_set) * cfg.val_split))
            train_size = len(train_set) - val_size
            generator = torch.Generator().manual_seed(cfg.seed)
            train_set, val_set = random_split(train_set, [train_size, val_size], generator=generator)
        return train_set, val_set

    synthetic_cfg = SyntheticConfig(
        image_size=cfg.image_size,
        num_samples=cfg.num_samples,
        seed=cfg.seed,
    )
    full_set = SyntheticMotionDataset(synthetic_cfg)
    val_size = max(1, int(len(full_set) * cfg.val_split))
    train_size = len(full_set) - val_size
    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = random_split(full_set, [train_size, val_size], generator=generator)
    return train_set, val_set


def masked_endpoint_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-pixel L1 between predicted and target flow, masked to the fluid region."""
    weight = mask.expand_as(prediction)
    diff = (prediction - target).abs() * weight
    denom = weight.sum().clamp_min(1.0)
    return diff.sum() / denom


def total_variation(flow: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Encourage spatial smoothness within the fluid mask."""
    diff_x = (flow[..., :, 1:] - flow[..., :, :-1]).abs()
    diff_y = (flow[..., 1:, :] - flow[..., :-1, :]).abs()
    weight_x = mask[..., :, 1:].expand_as(diff_x)
    weight_y = mask[..., 1:, :].expand_as(diff_y)
    denom = weight_x.sum().clamp_min(1.0) + weight_y.sum().clamp_min(1.0)
    return ((diff_x * weight_x).sum() + (diff_y * weight_y).sum()) / denom


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device, non_blocking=True)
            target = batch["flow"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            prediction = model(inputs)
            loss = masked_endpoint_loss(prediction, target, mask)
            batch_size = inputs.shape[0]
            total_loss += loss.item() * batch_size
            total_count += batch_size
    return total_loss / max(total_count, 1)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    cfg: TrainConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "input_size": cfg.image_size,
            "base_channels": cfg.base_channels,
        },
        path,
    )


def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"Using device: {device}")

    train_set, val_set = build_datasets(cfg)
    print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    model = MotionUNet(in_channels=7, base_channels=cfg.base_channels).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        running_count = 0

        for step, batch in enumerate(train_loader, start=1):
            inputs = batch["inputs"].to(device, non_blocking=True)
            target = batch["flow"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            prediction = model(inputs)
            data_loss = masked_endpoint_loss(prediction, target, mask)
            if cfg.tv_weight > 0:
                tv_loss = total_variation(prediction, mask)
                loss = data_loss + cfg.tv_weight * tv_loss
            else:
                tv_loss = torch.tensor(0.0, device=device)
                loss = data_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = inputs.shape[0]
            running_loss += loss.item() * batch_size
            running_count += batch_size

            if step % cfg.log_every == 0:
                avg = running_loss / max(running_count, 1)
                print(
                    f"epoch {epoch:03d} | step {step:04d}/{len(train_loader)} | "
                    f"loss {avg:.4f} (data {data_loss.item():.4f}, tv {tv_loss.item():.4f})"
                )

        scheduler.step()
        train_loss = running_loss / max(running_count, 1)
        val_loss = evaluate(model, val_loader, device)
        epoch_time = time.time() - epoch_start
        print(
            f"epoch {epoch:03d} done | train {train_loss:.4f} | val {val_loss:.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.2e} | {epoch_time:.1f}s"
        )

        if epoch % cfg.save_every == 0:
            save_checkpoint(cfg.output_dir / "last.pt", model, optimizer, epoch, val_loss, cfg)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(cfg.output_dir / "best.pt", model, optimizer, epoch, val_loss, cfg)
            print(f"  ↳ new best val loss: {best_val:.4f}")

    print(f"Training finished. Best val loss: {best_val:.4f}")
    print(f"Best checkpoint: {cfg.output_dir / 'best.pt'}")


if __name__ == "__main__":
    train(parse_args())
