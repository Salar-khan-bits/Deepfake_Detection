#!/usr/bin/env python3
"""Train binary Real/Fake classifier on ImageFolder splits (GPU + AMP)."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from torch import amp
from tqdm import tqdm

from config import CONFIG, Config
from dataset import build_dataloaders, build_transforms
from model import build_model, config_to_dict, resolve_normalize_from_timm
from utils import (
    TrainLogger,
    configure_training_backends,
    dataloader_kwargs,
    evaluate_model,
    get_device,
    save_checkpoint,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train deepfake detector (ViT / SigLIP timm)")
    p.add_argument("--data-root", type=Path, default=None, help="Root with train/val/test")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--model", type=str, default=None, help="timm model name")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--no-pretrained", action="store_true", help="Random init (faster smoke tests)")
    p.add_argument("--experiment-name", type=str, default=None)
    p.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="If val/ is missing, fraction of train/ used for validation (default from config)",
    )
    return p.parse_args()


def apply_cli(cfg: Config, args: argparse.Namespace) -> None:
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.model is not None:
        cfg.model_name = args.model
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.no_amp:
        cfg.use_amp = False
    if args.no_tensorboard:
        cfg.use_tensorboard = False
    if args.experiment_name is not None:
        cfg.experiment_name = args.experiment_name
    if args.no_pretrained:
        cfg.pretrained = False
    if args.val_fraction is not None:
        cfg.val_split_fraction = args.val_fraction


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    cuda_available: bool,
    use_amp: bool,
    scaler: amp.GradScaler,
    grad_accum: int,
    epoch: int,
) -> float:
    model.train()
    running = 0.0
    n = 0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"Train epoch {epoch}", leave=False)
    step = -1
    for step, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with amp.autocast("cuda", enabled=use_amp and cuda_available):
            logits = model(images)
            loss = criterion(logits, labels) / grad_accum

        if use_amp and cuda_available:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum == 0:
            if use_amp and cuda_available:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        bs = labels.size(0)
        running += loss.item() * bs * grad_accum
        n += bs
        pbar.set_postfix(loss=f"{running / max(n, 1):.4f}")

    if step < 0:
        return 0.0
    pending = (step + 1) % grad_accum
    if pending != 0:
        if use_amp and cuda_available:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return running / max(n, 1)


def main() -> None:
    cfg = deepcopy(CONFIG)
    apply_cli(cfg, parse_args())
    set_seed(cfg.seed)
    configure_training_backends()

    device, cuda_available = get_device()
    print(f"CUDA available: {cuda_available}", flush=True)
    if cuda_available:
        print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)

    if cfg.normalize_mean is not None and cfg.normalize_std is not None:
        mean, std = cfg.normalize_mean, cfg.normalize_std
    else:
        mean, std = resolve_normalize_from_timm(cfg.model_name)

    train_tf = build_transforms(
        cfg.img_size, mean, std, is_train=True, use_augmentation=cfg.use_train_augmentation
    )
    val_tf = build_transforms(cfg.img_size, mean, std, is_train=False, use_augmentation=False)
    test_tf = val_tf

    loader_kw = dataloader_kwargs(
        cfg.num_workers,
        pin_memory=cfg.pin_memory and cuda_available,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        prefetch_factor=cfg.prefetch_factor,
    )

    val_path = cfg.data_root / "val"
    if not val_path.is_dir():
        print(
            f"No val/ at {val_path}; using {cfg.val_split_fraction:.0%} stratified hold-out from train/.",
            flush=True,
        )

    train_loader, val_loader, _, class_names = build_dataloaders(
        cfg.data_root,
        train_tf,
        val_tf,
        test_tf,
        cfg.batch_size,
        loader_kw,
        val_split_fraction=cfg.val_split_fraction,
        split_seed=cfg.seed,
    )
    print(f"Classes (order matters): {class_names}", flush=True)

    model = build_model(cfg)
    if cuda_available:
        model.to("cuda")

    if cfg.compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("torch.compile enabled", flush=True)
        except Exception as e:
            print(f"torch.compile skipped: {e}", flush=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    use_amp = cfg.use_amp
    scaler = amp.GradScaler("cuda", enabled=use_amp and cuda_available)

    exp = cfg.experiment_name
    ckpt_dir = cfg.checkpoint_dir / exp
    log_dir = cfg.log_dir / exp
    logger = TrainLogger(
        log_dir if cfg.use_tensorboard else None,
        cfg.use_tensorboard,
        name="train",
    )

    best_val_acc = 0.0
    patience_left = cfg.early_stopping_patience
    best_path = ckpt_dir / "best.pt"

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            cuda_available,
            use_amp,
            scaler,
            cfg.gradient_accumulation_steps,
            epoch,
        )
        val_loss, y_true, y_pred = evaluate_model(
            model, val_loader, device, use_amp, cuda_available
        )
        val_acc = float((y_true == y_pred).mean())

        logger.log_scalars(
            "epoch",
            {"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc},
            step=epoch,
        )
        print(
            f"Epoch {epoch}/{cfg.num_epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}",
            flush=True,
        )

        improved = val_acc > best_val_acc + cfg.early_stopping_min_delta
        if improved:
            best_val_acc = val_acc
            patience_left = cfg.early_stopping_patience
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                best_val_acc,
                config_to_dict(cfg),
            )
            print(f"  Saved new best checkpoint (val_acc={best_val_acc:.4f}) -> {best_path}", flush=True)
        else:
            if cfg.early_stopping:
                patience_left -= 1
                print(f"  No val improvement. Early-stop patience: {patience_left}", flush=True)
                if patience_left <= 0:
                    print("Early stopping.", flush=True)
                    break

    logger.close()
    print(f"Done. Best val acc: {best_val_acc:.4f}. Checkpoint: {best_path}", flush=True)


if __name__ == "__main__":
    main()
