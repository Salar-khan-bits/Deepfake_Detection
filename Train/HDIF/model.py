"""Vision Transformer backbone (SigLIP / timm) with binary classification head."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import timm
import torch.nn as nn

from config import Config


def resolve_normalize_from_timm(model_name: str):
    cfg = timm.models.get_pretrained_cfg(model_name)
    if cfg is None:
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    return tuple(cfg.mean), tuple(cfg.std)


def build_model(cfg: Config) -> nn.Module:
    model = timm.create_model(
        cfg.model_name,
        pretrained=cfg.pretrained,
        num_classes=2,
    )
    if cfg.freeze_backbone:
        for name, p in model.named_parameters():
            if "head" not in name and "classifier" not in name:
                p.requires_grad = False
    return model


def config_to_dict(cfg: Config) -> Dict[str, Any]:
    d = {}
    for k, v in cfg.__dict__.items():
        if isinstance(v, Path):
            d[k] = str(v)
        else:
            d[k] = v
    return d
