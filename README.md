

# 🔍 DeepGuard — Deepfake Detection via SigLIP2

**A research framework for training and evaluating binary deepfake detectors**  
using ViT / SigLIP-based backbones across CiFake, SID, and HiDF datasets.

[PyTorch](https://pytorch.org)
[timm](https://github.com/huggingface/pytorch-image-models)
[HuggingFace](https://huggingface.co/salarkhan12345/deepfake-siglip2-models)
[Demo](https://huggingface.co/spaces/SalarKhan12345/SigLip_Space)

---

### 👥 Research Team


|     | Researcher                   |
| --- | ---------------------------- |
| 🔵  | **Ayaan Faisal**             |
| 🟣  | **Muhammad Abdullah Nadeem** |
| 🟢  | **Muhammad Salar Khan**      |




---

## 📌 Overview

DeepGuard supports end-to-end deepfake detection research:

- 🧠 **Train** detectors on standard `train/val/test` image splits
- 📊 **Evaluate** checkpoints on held-out test data
- 🔁 **Fine-tune** from large datasets such as `140k_reduced`
- 💾 **Organize** model checkpoints and training logs automatically

The default backbone is **ViT-B-16-SigLIP-384** with pretrained `timm` weights, fine-tuned for binary `real` / `fake` classification.

---

## 🗂️ Repository Structure

```
deepguard/
├── config.py          # Default hyperparameters, paths, experiment settings
├── train.py           # Main training script (binary real/fake classification)
├── inference.py       # Evaluation script for saved checkpoints
├── requirements.txt   # Python dependencies
│
├── Train/             # Dataset, model, and training utility modules
├── checkpoints/       # Saved model weights per experiment
├── logs/              # Training logs
├── data/              # Sampled subset datasets
├── 140k_reduced/      # Large source dataset for HiDF fine-tuning
└── notebooks/         # Example notebooks and experiment workflows
```

---

## 📦 Dataset Layout

Datasets must follow a standard **ImageFolder** layout under `data_root`:

```
data_root/
├── train/
│   ├── fake/          # synthetic / manipulated images
│   └── real/          # authentic images
├── val/               # optional — auto-split from train/ if missing
│   ├── fake/
│   └── real/
└── test/
    ├── fake/
    └── real/
```

> **💡 Tip:** If `val/` is missing, `train.py` automatically holds out a stratified fraction of `train/` using `--val-fraction`.

> **🗒️ SID Dataset:** The loader supports `real`, `synthetic`, and `tampered` labels. During training, `real` is the positive class and `synthetic` / `tampered` are both mapped to `fake`.

---

## ⚙️ Installation

```bash
python -m pip install -r requirements.txt
```

**Recommended package versions:**


| Package        | Version  |
| -------------- | -------- |
| `torch`        | ≥ 2.0.0  |
| `torchvision`  | ≥ 0.15.0 |
| `timm`         | ≥ 0.9.0  |
| `scikit-learn` | ≥ 1.2.0  |
| `tqdm`         | ≥ 4.64.0 |
| `matplotlib`   | ≥ 3.6.0  |
| `pillow`       | ≥ 9.0.0  |
| `numpy`        | ≥ 1.24.0 |


---

## 🚀 Training

```bash
python train.py \
  --data-root       data/hidf_subset \
  --experiment-name hidf \
  --epochs          10 \
  --batch-size      4 \
  --val-fraction    0.333 \
  --safe-cuda
```

### Training Arguments


| Flag                | Description                                                                 |
| ------------------- | --------------------------------------------------------------------------- |
| `--data-root`       | Root dataset folder containing `train/`, optional `val/`, and `test/`       |
| `--experiment-name` | Folder name under `checkpoints/` and `logs/`                                |
| `--epochs`          | Number of training epochs                                                   |
| `--batch-size`      | Batch size for training and validation                                      |
| `--lr`              | Learning rate override (see `config.py` for default)                        |
| `--model`           | Custom `timm` model name                                                    |
| `--val-fraction`    | Fraction of train data reserved for validation when `val/` is absent        |
| `--train-fraction`  | Fraction of train data to use overall                                       |
| `--safe-cuda`       | Conservative CUDA mode: `num_workers=0`, no pin/persistent workers, AMP off |
| `--no-amp`          | Disable automatic mixed precision                                           |
| `--no-pretrained`   | Random weight initialization (smoke-test mode)                              |


### Example Commands

**Train on HiDF subset:**

```bash
python train.py --data-root data/hidf_subset --experiment-name hidf --epochs 10 --batch-size 4 --val-fraction 0.333 --safe-cuda
```

**Train on CiFake subset:**

```bash
python train.py --data-root data/cifake_subset --experiment-name cifake --epochs 10 --batch-size 4 --val-fraction 0.333 --safe-cuda
```

---

## 🔬 Inference & Evaluation

```bash
python inference.py \
  --checkpoint checkpoints/hidf/finetuned_hidf.pt \
  --data-root  data/hidf_subset \
  --batch-size 4
```

> If `--checkpoint` is omitted, the script automatically looks for `checkpoints/<experiment_name>/best.pt`.

---

## 🔁 Fine-tuning on `140k_reduced`

A notebook workflow samples a subset from `140k_reduced` and fine-tunes the HIDF model. The fine-tuned checkpoint is saved separately to avoid overwriting the original.

```python
# 1. Sample subset from 140k_reduced
prepare_dataset_subset("140k_reduced", HIDF_140K_SOURCE_PATH, HIDF_140K_DATASET_PATH)

# 2. Train
cmd = (
    f"python train.py --data-root {HIDF_140K_DATASET_PATH} "
    "--experiment-name hidf --epochs 10 --batch-size 4 "
    "--val-fraction 0.333 --safe-cuda"
)
run_command(cmd)

# 3. Preserve finetuned checkpoint separately
shutil.copy2("checkpoints/hidf/best.pt", "checkpoints/hidf/finetuned_hidf.pt")
```

---

## 🤖 Pretrained Models

Download all checkpoints automatically:

```bash
python checkpoints/download_models.py
```


| Model                                | HuggingFace Repo                                 | File                                       | Local Path                                             |
| ------------------------------------ | ------------------------------------------------ | ------------------------------------------ | ------------------------------------------------------ |
| **CiFake** — Real vs AI-Generated    | `salarkhan12345/deepfake-siglip2-models`         | `CiFake_model.pth`                         | `checkpoints/cifake/cifake_trained.pt`                 |
| **SID** — Binary (Real vs Fake)      | `salarkhan12345/deepfake-siglip2-models`         | `SID_binary.pt`                            | `checkpoints/sid/sid_trained.pt`                       |
| **SID** — 3-Class (Real/Synth/Tamp)  | `salarkhan12345/deepfake-siglip2-models`         | `SID_3class.pt`                            | `checkpoints/sid/sid_3class.pt`                        |
| **HiDF** — Binary (Ayaan)            | `ayaani12/deepfake_siglip_vitb16_Finetuned_HIDF` | `deepfake_siglip_vitb16_Finetuned_HIDF.pt` | `checkpoints/hidf/hidf_trained.pt`                     |
| ⭐ **Ensemble** — SID + HiDF + CiFake | `ayaani12/Ensembled_SIGLIP2_FineTuned`           | `weighted_avg_sid_hidf_cifake.pt`          | `checkpoints/ensemble/weighted_avg_sid_hidf_cifake.pt` |


---

## 💡 Notes

- **CUDA issues?** Use `--safe-cuda` or set `CUDA_LAUNCH_BLOCKING=1` for debugging.
- **Hyperparameters** are defined in `config.py` — image size, learning rate, augmentation pipeline, etc.
- **Separate experiments** by setting a unique `--experiment-name` per run to keep checkpoints and logs clean.
- **Smoke testing** is possible with `--no-pretrained` to skip downloading pretrained weights.

---

## 🌐 Live Demo

Try the deployed ensemble model on Hugging Face Spaces:

**[🚀 SigLIP Space → huggingface.co/spaces/SalarKhan12345/SigLip_Space](https://huggingface.co/spaces/SalarKhan12345/SigLip_Space)**

---



*For issues or custom experiments, inspect `config.py` for defaults*  
*and set `--experiment-name` to keep your checkpoints organized.*

**DeepGuard · 2025 · Ayaan Faisal · Muhammad Abdullah Nadeem · Muhammad Salar Khan**

