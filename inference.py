#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the test split."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch

from config import CONFIG, Config
from Train.dataset import build_imagefolder, build_transforms
from Train.model import build_model, resolve_normalize_from_timm
from torch.utils.data import DataLoader
from Train.utils import (
    classification_report_arrays,
    configure_training_backends,
    dataloader_kwargs,
    evaluate_model,
    get_device,
    load_checkpoint,
    print_metrics,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate deepfake detector on test set")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to best.pt (default: checkpoints/<experiment_name>/best.pt)",
    )
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--experiment-name", type=str, default=None)
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def apply_cli(cfg: Config, args: argparse.Namespace) -> None:
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.experiment_name is not None:
        cfg.experiment_name = args.experiment_name
    if args.no_amp:
        cfg.use_amp = False


def main() -> None:
    cfg = deepcopy(CONFIG)
    args = parse_args()
    apply_cli(cfg, args)

    set_seed(cfg.seed)
    configure_training_backends()
    device, cuda_available = get_device()
    print(f"CUDA available: {cuda_available}", flush=True)

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = cfg.checkpoint_dir / cfg.experiment_name / "best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = load_checkpoint(ckpt_path, map_location=device)
    saved_cfg = payload.get("config")
    if saved_cfg:
        for k, v in saved_cfg.items():
            if hasattr(cfg, k) and k in (
                "model_name",
                "pretrained",
                "freeze_backbone",
                "img_size",
                "metrics_positive_label",
            ):
                setattr(cfg, k, v)
        if isinstance(saved_cfg.get("data_root"), str):
            cfg.data_root = Path(saved_cfg["data_root"])

    apply_cli(cfg, args)

    if cfg.normalize_mean is not None and cfg.normalize_std is not None:
        mean, std = cfg.normalize_mean, cfg.normalize_std
    else:
        mean, std = resolve_normalize_from_timm(cfg.model_name)

    test_tf = build_transforms(
        cfg.img_size, mean, std, is_train=False, use_augmentation=False
    )
    test_ds = build_imagefolder(cfg.data_root, "test", test_tf)
    class_names = test_ds.classes
    print(f"Test classes: {class_names}", flush=True)

    loader_kw = dataloader_kwargs(
        cfg.num_workers,
        pin_memory=cfg.pin_memory and cuda_available,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        prefetch_factor=cfg.prefetch_factor,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        **loader_kw,
    )

    model = build_model(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if cuda_available:
        model.to("cuda")

    use_amp = cfg.use_amp
    test_loss, y_true, y_pred = evaluate_model(
        model, test_loader, device, use_amp, cuda_available
    )
    metrics = classification_report_arrays(
        y_true, y_pred, class_names, pos_label=cfg.metrics_positive_label
    )
    metrics["class_names"] = class_names

    print(f"\nTest loss: {test_loss:.4f}", flush=True)
    print_metrics("Test metrics", metrics)


if __name__ == "__main__":
    main()
