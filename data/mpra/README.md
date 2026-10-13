# DeepBioParts MPRA datasets

These processed MPRA datasets contain DNA sequences and experimentally measured
activity. They are provided for data reuse, sequence analysis, and evaluation
with the released pretrained models.

| File | Regulatory part | Sequences | Length (bp) | Activity |
| --- | --- | ---: | ---: | --- |
| `promoter_mpra.csv` | Promoter | 22,853 | 40 | Relative promoter activity |
| `rbs_mpra.csv` | Ribosome binding site | 11,613 | 15 | Relative RBS activity |
| `terminator_mpra.csv` | Terminator | 11,536 | 50 | Termination efficiency (%) |

## Columns

Each CSV contains the same two columns:

- `sequence`: DNA sequence in uppercase A/C/G/T.
- `activity`: Experimental activity in the part-specific scale shown above.

The CSV contents, row order, and measured values are unchanged from the source
tables; only the filenames have been standardized. No model predictions or
classification labels have been added. The three activity scales should be
interpreted separately.

These are the MPRA reference datasets used in model development, rather than
independent held-out test sets. Predictions on these data do not establish
performance on unseen sequences.

## Predict using released weights

After following the installation and weight-placement instructions in the root
[README](../../README.md), run from the repository root:

```bash
python scripts/predict.py \
  --data_path data/mpra/rbs_mpra.csv \
  --biopart rbs \
  --output_dir results/mpra_rbs_prediction
```

Use `promoter_mpra.csv` with `--biopart promoter`, or `terminator_mpra.csv` with
`--biopart terminator`, for the other datasets. Promoter prediction additionally
requires the Evo backbone and released promoter checkpoint. The prediction
script reads `sequence`; the experimental `activity` values remain in the input
CSV for downstream comparison. See the root README for the generated files.

## License and citation

These datasets are distributed under the repository's [MIT license](../../LICENSE).
Please cite DeepBioParts using the citation in the root README.
