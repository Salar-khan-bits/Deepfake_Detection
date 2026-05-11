"""Hyperparameters and paths for deepfake detection training and evaluation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class Config:
    # Data layout: data_root/train/{fake,real or Fake,Real}, val/ (optional), test/...
    # If val/ is missing, a stratified fraction of train/ is held out for validation.
    data_root: Path = Path("dataset")
    val_split_fraction: float = 0.1
    img_size: int = 224

    # timm ViT / SigLIP-style backbone (224x224 SigLIP ViT)
    model_name: str = "ViT-B-16-SigLIP-384"
    pretrained: bool = True
    freeze_backbone: bool = False

    # Training
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.0
    num_workers: int = 8
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    # Mixed precision (CUDA only; no-op on CPU)
    use_amp: bool = True

    # Early stopping (validation accuracy)
    early_stopping: bool = True
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-4

    # Checkpoints and logs
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")
    experiment_name: str = "deepfake_siglip"
    use_tensorboard: bool = True

    # Reproducibility
    seed: int = 42

    # Optional speed (PyTorch 2+); can fail on some setups — disable if needed
    compile_model: bool = False

    # Override normalization; if None, resolved from timm pretrained cfg for model_name
    normalize_mean: Optional[Tuple[float, float, float]] = None
    normalize_std: Optional[Tuple[float, float, float]] = None

    # Light train-time augmentation
    use_train_augmentation: bool = True

    class_names: List[str] = field(default_factory=lambda: ["Fake", "Real"])
    # sklearn binary metrics: label index treated as positive class (Fake=0, Real=1)
    metrics_positive_label: int = 0

    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


CONFIG = Config()
