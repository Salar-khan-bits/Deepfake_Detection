"""ImageFolder-based datasets for train / val / test splits."""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def build_transforms(
    img_size: int,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    is_train: bool,
    use_augmentation: bool,
) -> transforms.Compose:
    if is_train and use_augmentation:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def get_class_names(data_root: Path, split: str) -> list:
    split_dir = data_root / split
    ds = datasets.ImageFolder(str(split_dir))
    return ds.classes


def build_imagefolder(
    data_root: Path,
    split: str,
    transform: transforms.Compose,
) -> datasets.ImageFolder:
    split_dir = data_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")
    return datasets.ImageFolder(str(split_dir), transform=transform)


class _ImageFolderSubset(Dataset):
    """Subset of ImageFolder samples with a chosen transform (train vs val)."""

    def __init__(
        self,
        base: datasets.ImageFolder,
        indices: List[int],
        transform: transforms.Compose,
    ):
        self.base = base
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        path, label = self.base.samples[self.indices[i]]
        img = self.base.loader(path)
        if self.base.target_transform is not None:
            label = self.base.target_transform(label)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def stratified_train_val_indices(
    image_folder: datasets.ImageFolder,
    val_fraction: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    by_class: dict[int, List[int]] = {}
    for idx, (_, y) in enumerate(image_folder.samples):
        by_class.setdefault(y, []).append(idx)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for idxs in by_class.values():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        n = len(idxs)
        if n == 0:
            continue
        n_val = int(round(n * val_fraction))
        if n == 1:
            n_val = 0
        else:
            n_val = min(max(n_val, 0), n - 1)
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def build_dataloaders(
    data_root: Path,
    train_tf: transforms.Compose,
    val_tf: transforms.Compose,
    test_tf: transforms.Compose,
    batch_size: int,
    loader_common_kwargs: dict,
    val_split_fraction: float = 0.1,
    split_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, list]:
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Missing train directory: {train_dir}")

    if val_dir.is_dir():
        train_ds = build_imagefolder(data_root, "train", train_tf)
        val_ds = build_imagefolder(data_root, "val", val_tf)
    else:
        base = datasets.ImageFolder(str(train_dir), transform=None)
        t_idx, v_idx = stratified_train_val_indices(
            base, val_fraction=val_split_fraction, seed=split_seed
        )
        train_ds = _ImageFolderSubset(base, t_idx, train_tf)
        val_ds = _ImageFolderSubset(base, v_idx, val_tf)

    test_ds = build_imagefolder(data_root, "test", test_tf)
    if val_dir.is_dir():
        class_names = train_ds.classes
    else:
        class_names = base.classes
    if [c.lower() for c in test_ds.classes] != [c.lower() for c in class_names]:
        raise ValueError(
            f"Train vs test class names differ: train {class_names!r} vs test {test_ds.classes!r}"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        **loader_common_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_common_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_common_kwargs,
    )
    return train_loader, val_loader, test_loader, class_names
