#!/usr/bin/env python3
"""Download pretrained checkpoints and arrange them in this repo layout.

Usage:
    python checkpoints/download_models.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = ROOT / "checkpoints"

MODEL_SPECS = [
    {
        "name": "CiFake — Real vs AI-Generated",
        "repo": "salarkhan12345/deepfake-siglip2-models",
        "filename": "CiFake_model.pth",
        "target": CHECKPOINTS_DIR / "cifake" / "cifake_trained.pt",
    },
    {
        "name": "SID — Binary (Real vs Fake)",
        "repo": "salarkhan12345/deepfake-siglip2-models",
        "filename": "SID_binary.pt",
        "target": CHECKPOINTS_DIR / "sid" / "sid_trained.pt",
    },
    {
        "name": "SID — 3-Class (Real/Synth/Tamp)",
        "repo": "salarkhan12345/deepfake-siglip2-models",
        "filename": "SID_3class.pt",
        "target": CHECKPOINTS_DIR / "sid" / "sid_3class.pt",
    },
    {
        "name": "HiDF — Binary (Ayaan)",
        "repo": "ayaani12/deepfake_siglip_vitb16_Finetuned_HIDF",
        "filename": "deepfake_siglip_vitb16_Finetuned_HIDF.pt",
        "target": CHECKPOINTS_DIR / "hidf" / "hidf_trained.pt",
    },
    {
        "name": "Ensembled SIGLIP2",
        "repo": "ayaani12/Ensembled_SIGLIP2_FineTuned",
        "filename": "weighted_avg_sid_hidf_cifake.pt",
        "target": CHECKPOINTS_DIR / "ensemble" / "weighted_avg_sid_hidf_cifake.pt",
    },
]


def hf_resolve_url(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}?download=true"


def download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(target)


def main() -> None:
    print(f"Repo root: {ROOT}")
    print("Downloading models to current checkpoint layout...\n")

    failures = []
    for spec in MODEL_SPECS:
        name = spec["name"]
        repo = spec["repo"]
        filename = spec["filename"]
        target = spec["target"]
        url = hf_resolve_url(repo, filename)

        print(f"[START] {name}")
        print(f"  Source: {repo}/{filename}")
        print(f"  Target: {target.relative_to(ROOT)}")
        try:
            download_file(url, target)
            print("  Status: OK\n")
        except Exception as exc:
            failures.append((name, str(exc)))
            print(f"  Status: FAILED ({exc})\n")

    if failures:
        print("Completed with errors:")
        for name, err in failures:
            print(f"- {name}: {err}")
        raise SystemExit(1)

    print("All checkpoints downloaded and arranged successfully.")


if __name__ == "__main__":
    main()
