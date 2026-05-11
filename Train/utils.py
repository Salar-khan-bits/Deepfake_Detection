"""Device setup, logging, metrics, and dataloader helpers."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import amp
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # optional dependency
    SummaryWriter = None  # type: ignore[misc, assignment]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> Tuple[torch.device, bool]:
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    return device, cuda_available


def configure_training_backends() -> None:
    """Faster convolutions on GPU; safe defaults for throughput."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def dataloader_kwargs(
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> Dict:
    kw: Dict = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        kw["worker_init_fn"] = worker_init_fn
        kw["prefetch_factor"] = prefetch_factor
        if persistent_workers:
            kw["persistent_workers"] = True
    return kw


class TrainLogger:
    def __init__(self, log_dir: Optional[Path], use_tensorboard: bool, name: str):
        self.log_dir = log_dir
        self._tb: Optional[Any] = None
        if use_tensorboard and log_dir is not None:
            if SummaryWriter is None:
                print(
                    "TensorBoard requested but tensorboard is not installed; "
                    "logging to console only. pip install tensorboard",
                    flush=True,
                )
            else:
                log_dir.mkdir(parents=True, exist_ok=True)
                self._tb = SummaryWriter(log_dir=str(log_dir / name))

    def log_scalars(self, tag_prefix: str, scalars: Dict[str, float], step: int) -> None:
        parts = [f"{tag_prefix}/{k}={v:.6f}" for k, v in scalars.items()]
        msg = "  ".join(parts)
        print(msg, flush=True)
        if self._tb is not None:
            for k, v in scalars.items():
                self._tb.add_scalar(f"{tag_prefix}/{k}", v, step)

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
    cuda_available: bool,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    total_loss = 0.0
    n = 0
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with amp.autocast("cuda", enabled=use_amp and cuda_available):
            logits = model(images)
            loss = criterion(logits, labels)
        bs = labels.size(0)
        total_loss += loss.item() * bs
        n += bs
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(n, 1)
    return avg_loss, np.array(all_labels), np.array(all_preds)


def classification_report_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    pos_label: int = 0,
) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": cm,
        "class_names": class_names,
    }


def print_metrics(title: str, metrics: Dict[str, Any]) -> None:
    print(f"\n{title}", flush=True)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}", flush=True)
    print(f"  Precision: {metrics['precision']:.4f}", flush=True)
    print(f"  Recall:    {metrics['recall']:.4f}", flush=True)
    print(f"  F1-score:  {metrics['f1']:.4f}", flush=True)
    cm = metrics["confusion_matrix"]
    names = metrics["class_names"]
    print("  Confusion matrix (rows=true, cols=pred):", flush=True)
    header = "        " + "  ".join(f"{n:>8}" for n in names)
    print(header, flush=True)
    for i, row_name in enumerate(names):
        row = "  ".join(f"{cm[i, j]:8d}" for j in range(cm.shape[1]))
        print(f"    {row_name:>4}  {row}", flush=True)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_acc: float,
    config_dict: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "config": config_dict,
        },
        path,
    )


def load_checkpoint(path: Path, map_location: torch.device) -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def suggest_num_workers() -> int:
    cpu = os.cpu_count() or 2
    return max(1, min(8, cpu - 1))
