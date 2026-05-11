#!/usr/bin/env python3
"""Evaluate a checkpoint on an ImageFolder split and write a Markdown report."""

from __future__ import annotations

import argparse
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CONFIG, Config
from dataset import build_imagefolder, build_transforms
from model import build_model, resolve_normalize_from_timm
from utils import (
    classification_report_arrays,
    configure_training_backends,
    dataloader_kwargs,
    evaluate_model,
    get_device,
    load_checkpoint,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoint and generate Markdown report."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained checkpoint (best.pt)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Dataset root containing split directories",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split name under data root (default: test)",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Folder name under reports/ to save markdown + plots (example: --save run_01)",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Output markdown path (default: <checkpoint_dir>/test_report_<dataset>_<split>.md)",
    )
    return parser.parse_args()


def apply_saved_model_config(cfg: Config, payload: Dict[str, Any]) -> None:
    saved_cfg = payload.get("config")
    if not isinstance(saved_cfg, dict):
        return
    for key in (
        "model_name",
        "pretrained",
        "freeze_backbone",
        "img_size",
        "metrics_positive_label",
        "normalize_mean",
        "normalize_std",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
    ):
        if key in saved_cfg and hasattr(cfg, key):
            setattr(cfg, key, saved_cfg[key])


def count_split_samples(ds) -> Dict[str, int]:
    counts: Dict[str, int] = {name: 0 for name in ds.classes}
    for _, label in ds.samples:
        counts[ds.classes[label]] += 1
    return counts


def format_md_report(
    *,
    ckpt_path: Path,
    data_root: Path,
    split: str,
    device: torch.device,
    use_amp: bool,
    runtime_s: float,
    dataset_size: int,
    class_counts: Dict[str, int],
    metrics: Dict[str, Any],
    test_loss: float,
    save_dir: Path,
) -> str:
    class_names: List[str] = metrics["class_names"]
    cm = metrics["confusion_matrix"]

    lines: List[str] = []
    lines.append("# Model Test Report")
    lines.append("")
    lines.append("## Run Information")
    lines.append("")
    lines.append(f"- **Generated (UTC):** `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append(f"- **Checkpoint:** `{ckpt_path}`")
    lines.append(f"- **Dataset root:** `{data_root}`")
    lines.append(f"- **Evaluated split:** `{split}`")
    lines.append(f"- **Device:** `{device}`")
    lines.append(f"- **AMP enabled:** `{use_amp}`")
    lines.append(f"- **Evaluation runtime:** `{runtime_s:.2f}s`")
    lines.append(f"- **Saved outputs:** `{save_dir}`")
    lines.append("")
    lines.append("## Dataset Summary")
    lines.append("")
    lines.append(f"- **Total images:** `{dataset_size}`")
    for class_name in class_names:
        n = class_counts.get(class_name, 0)
        share = (100.0 * n / dataset_size) if dataset_size else 0.0
        lines.append(f"- **{class_name}:** `{n}` ({share:.2f}%)")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append(f"- **Cross-entropy loss:** `{test_loss:.6f}`")
    lines.append(f"- **Accuracy:** `{metrics['accuracy']:.6f}`")
    lines.append(f"- **Precision:** `{metrics['precision']:.6f}`")
    lines.append(f"- **Recall:** `{metrics['recall']:.6f}`")
    lines.append(f"- **F1-score:** `{metrics['f1']:.6f}`")
    lines.append("")
    lines.append("## Confusion Matrix")
    lines.append("")
    lines.append("Rows = true labels, columns = predicted labels.")
    lines.append("")
    lines.append("| True \\ Pred | " + " | ".join(class_names) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(class_names)) + "|")
    for i, row_name in enumerate(class_names):
        row_vals = " | ".join(str(int(cm[i, j])) for j in range(len(class_names)))
        lines.append(f"| {row_name} | {row_vals} |")
    lines.append("")
    lines.append("## Saved Plots")
    lines.append("")
    lines.append("- `class_distribution.png`")
    lines.append("- `metrics_bar.png`")
    lines.append("- `confusion_matrix.png`")
    lines.append("")
    return "\n".join(lines)


def save_plots(save_dir: Path, class_counts: Dict[str, int], metrics: Dict[str, Any]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    class_names: List[str] = metrics["class_names"]
    cm = np.asarray(metrics["confusion_matrix"])
    metric_names = ["accuracy", "precision", "recall", "f1"]
    metric_values = [float(metrics[m]) for m in metric_names]

    # 1) Class distribution bar chart
    fig = plt.figure(figsize=(6, 4))
    plt.bar(class_names, [class_counts.get(c, 0) for c in class_names], color=["#4e79a7", "#e15759"])
    plt.title("Class Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    fig.savefig(save_dir / "class_distribution.png", dpi=180)
    plt.close(fig)

    # 2) Metrics bar chart
    fig = plt.figure(figsize=(7, 4))
    plt.bar(metric_names, metric_values, color="#59a14f")
    plt.ylim(0.0, 1.0)
    plt.title("Evaluation Metrics")
    plt.ylabel("Score")
    plt.tight_layout()
    fig.savefig(save_dir / "metrics_bar.png", dpi=180)
    plt.close(fig)

    # 3) Confusion matrix heatmap
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=20, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
    plt.tight_layout()
    fig.savefig(save_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = deepcopy(CONFIG)

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")

    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.no_amp:
        cfg.use_amp = False

    set_seed(cfg.seed)
    configure_training_backends()
    device, cuda_available = get_device()
    print(f"Using device: {device}", flush=True)
    print(f"CUDA available: {cuda_available}", flush=True)

    payload = load_checkpoint(args.checkpoint, map_location=device)
    apply_saved_model_config(cfg, payload)

    if cfg.normalize_mean is not None and cfg.normalize_std is not None:
        mean, std = cfg.normalize_mean, cfg.normalize_std
    else:
        mean, std = resolve_normalize_from_timm(cfg.model_name)

    test_tf = build_transforms(
        cfg.img_size, mean, std, is_train=False, use_augmentation=False
    )
    test_ds = build_imagefolder(args.data_root, args.split, test_tf)
    class_names = test_ds.classes

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
        drop_last=False,
        **loader_kw,
    )

    model = build_model(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if cuda_available:
        model.to("cuda")

    start = time.perf_counter()
    test_loss, y_true, y_pred = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        use_amp=cfg.use_amp,
        cuda_available=cuda_available,
    )
    runtime_s = time.perf_counter() - start
    metrics = classification_report_arrays(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        pos_label=cfg.metrics_positive_label,
    )
    metrics["class_names"] = class_names

    dataset_name = args.data_root.name
    default_save_name = f"{dataset_name}_{args.split}"
    save_name = args.save or default_save_name
    save_dir = Path("reports") / save_name
    default_md = save_dir / "test_report.md"
    output_md = args.output_md or default_md
    output_md.parent.mkdir(parents=True, exist_ok=True)
    class_counts = count_split_samples(test_ds)

    save_plots(save_dir, class_counts, metrics)

    report = format_md_report(
        ckpt_path=args.checkpoint,
        data_root=args.data_root,
        split=args.split,
        device=device,
        use_amp=cfg.use_amp and cuda_available,
        runtime_s=runtime_s,
        dataset_size=len(test_ds),
        class_counts=class_counts,
        metrics=metrics,
        test_loss=test_loss,
        save_dir=save_dir,
    )
    output_md.write_text(report, encoding="utf-8")

    print(f"Test images: {len(test_ds)}", flush=True)
    print(f"Accuracy: {metrics['accuracy']:.6f}", flush=True)
    print(f"F1-score: {metrics['f1']:.6f}", flush=True)
    print(f"Plots saved to: {save_dir}", flush=True)
    print(f"Report written to: {output_md}", flush=True)


if __name__ == "__main__":
    main()
