# DeepBioParts independent test datasets

## Summary

This directory contains four independent held-out test datasets used to
evaluate DeepBioParts activity predictors. Each dataset contains 100 regulatory
part sequences and the corresponding experimental measurements. These
sequences were not used for model training, validation, hyperparameter tuning,
or model selection. Model predictions are deliberately excluded and must be
generated from the released weights.

## Files

| File | Task | Reported metric |
| --- | --- | --- |
| `promoter_test_100.csv` | Promoter activity regression | Pearson *r* |
| `rbs_test_100.csv` | RBS activity regression | Pearson *r* |
| `terminator_continuing_test_100.csv` | Terminator activity regression | Pearson *r* |
| `terminator_strong_test_100.csv` | Strong-terminator classification | AUROC |

## Variables

The regression files contain:

| Column | Definition |
| --- | --- |
| `part_id` | Unique regulatory-part identifier |
| `sequence` | DNA sequence |
| `true_activity` | Experimentally measured activity |

The strong-terminator classification file contains:

| Column | Definition |
| --- | --- |
| `part_id` | Unique regulatory-part identifier |
| `sequence` | DNA sequence |
| `activity` | Experimentally measured termination efficiency (%) |

For classification, a sequence is labelled as a strong terminator when
`activity > 95`.

## Reproduce the reported results by model inference

First complete the installation, weight download, and weight-placement steps in
the repository root `README.md`. Promoter evaluation additionally requires the
local Evo 1.5 backbone and the merged promoter checkpoint.

From the repository root, run on a CUDA-capable Linux system:

```bash
python scripts/reproduce_test_results.py \
  --device cuda:0 \
  --batch-size 8
```

The script loads the released checkpoints, predicts every sequence, and
recomputes the evaluation metrics from the new predictions and experimental
measurements. It uses the same ensemble-inference implementation as
`scripts/predict.py` and writes:

```text
results/test_reproduction/
├── promoter/predictions.csv
├── promoter/metrics.csv
├── rbs/predictions.csv
├── rbs/metrics.csv
├── terminator_continuing/predictions.csv
├── terminator_continuing/metrics.csv
├── terminator_strong/predictions.csv
├── terminator_strong/metrics.csv
└── summary.csv
```

Reference values are shown below; minor floating-point differences may occur
between CUDA and PyTorch versions.

```text
promoter                 Pearson_r  0.851862
rbs                      Pearson_r  0.881904
terminator_continuing    Pearson_r  0.924198
terminator_strong        AUC        0.862319
```

## Licence

These files are distributed under the repository's [MIT licence](../../LICENSE).
