# Model Test Report

## Run Information

- **Generated (UTC):** `2026-05-05 19:51:54`
- **Checkpoint:** `checkpoints/deepfake_siglip_vitB16_384/deepfake_siglip_vitb16_Finetuned_HIDF.pt`
- **Dataset root:** `dataset`
- **Evaluated split:** `test`
- **Device:** `cuda`
- **AMP enabled:** `True`
- **Evaluation runtime:** `21.24s`
- **Saved outputs:** `reports/HIDF_eval`

## Dataset Summary

- **Total images:** `2000`
- **fake:** `1000` (50.00%)
- **real:** `1000` (50.00%)

## Metrics

- **Cross-entropy loss:** `0.067734`
- **Accuracy:** `0.984000`
- **Precision:** `0.984000`
- **Recall:** `0.984000`
- **F1-score:** `0.984000`

## Confusion Matrix

Rows = true labels, columns = predicted labels.

| True \ Pred | fake | real |
|---|---:|---:|
| fake | 984 | 16 |
| real | 16 | 984 |

## Saved Plots

- `class_distribution.png`
- `metrics_bar.png`
- `confusion_matrix.png`
