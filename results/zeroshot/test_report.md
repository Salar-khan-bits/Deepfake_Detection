# Model Test Report

## Run Information

- **Generated (UTC):** `2026-05-05 21:47:01`
- **Checkpoint:** `checkpoints/deepfake_siglip_vitB16_384/deepfake_siglip_vitb16_Finetuned_HIDF.pt`
- **Dataset root:** `140k_reduced`
- **Evaluated split:** `test`
- **Device:** `cuda`
- **AMP enabled:** `True`
- **Evaluation runtime:** `17.81s`
- **Saved outputs:** `reports/zeroshot`

## Dataset Summary

- **Total images:** `2000`
- **fake:** `1000` (50.00%)
- **real:** `1000` (50.00%)

## Metrics

- **Cross-entropy loss:** `2.331874`
- **Accuracy:** `0.470000`
- **Precision:** `0.430556`
- **Recall:** `0.186000`
- **F1-score:** `0.259777`

## Confusion Matrix

Rows = true labels, columns = predicted labels.

| True \ Pred | fake | real |
|---|---:|---:|
| fake | 186 | 814 |
| real | 246 | 754 |

## Saved Plots

- `class_distribution.png`
- `metrics_bar.png`
- `confusion_matrix.png`
