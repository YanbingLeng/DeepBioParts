# DeepBioParts

**A generative framework for the *de novo* design of non-repetitive regulatory part libraries.**

DeepBioParts treats a regulatory-part library—not an individual sequence—as
the design object. It combines DDPM sequence generation, activity prediction
(Evo 1.5 + LoRA or CNN–Attention–BiLSTM), and selection under a pairwise
homology constraint ($L_{\max} \le 10$ bp) in an iterative
generate–select–predict–update workflow.

## 1. System requirements and installation

The supported reference setup is Linux x86-64, Python 3.10, and an NVIDIA GPU
with driver ≥ 525 / CUDA 12.x. The compact RBS and terminator predictors require
far less memory than the Evo promoter predictor; allow approximately 24 GB GPU
memory and 20 GB free disk space if using the Evo 1.5 backbone.

```bash
git clone https://github.com/YanbingLeng/DeepBioParts.git
cd DeepBioParts

# Create and activate the pinned environment.
conda env create -f environment.yml
conda activate deepbioparts

# Install flash-attn from the prebuilt CUDA 12 / PyTorch 2.7 wheel.
# Do not use `pip install flash-attn`: that attempts a source build and needs
# the complete CUDA toolkit.
curl -fL -o /tmp/flash_attn.whl \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
cp /tmp/flash_attn.whl \
  /tmp/flash_attn-2.7.4.post1-cp310-cp310-linux_x86_64.whl
pip install /tmp/flash_attn-2.7.4.post1-cp310-cp310-linux_x86_64.whl

# Install DeepBioParts in editable mode.
pip install -e .
```

Environment creation and package downloads commonly take 10–30 minutes,
excluding the optional 13 GB Evo backbone download. Network speed and the
Conda solver can change this substantially.

## 2. Download and place pretrained weights

The 215.8 MB Zenodo archive contains the three DDPM generators, the compact
five-fold supervised predictors, the Evo promoter LoRA adapter, and the adapter
merge utility:

- record and DOI: [Zenodo 22097086](https://zenodo.org/records/22097086)
- direct file: [DeepBioParts_weights_v1.0.zip](https://zenodo.org/records/22097086/files/DeepBioParts_weights_v1.0.zip?download=1)

Run the following from the repository root. The loop converts the flat
`fold_1_checkpoint.pth` files in the archive into the directory layout expected
by the inference code.

```bash
curl -fL -o DeepBioParts_weights_v1.0.zip \
  "https://zenodo.org/records/22097086/files/DeepBioParts_weights_v1.0.zip?download=1"
unzip DeepBioParts_weights_v1.0.zip

WEIGHTS_DIR=DeepBioParts_weights_v1.0
cp -R "$WEIGHTS_DIR/diffusion_checkpoints" .

for MODEL_DIR in "$WEIGHTS_DIR"/supervised_model/*; do
  MODEL_NAME=$(basename "$MODEL_DIR")
  for FOLD in 1 2 3 4 5; do
    TARGET_DIR="predictor_checkpoints/supervised_model/$MODEL_NAME/fold_$FOLD"
    mkdir -p "$TARGET_DIR"
    cp "$MODEL_DIR/fold_${FOLD}_checkpoint.pth" \
      "$TARGET_DIR/checkpoint.pth"
  done
done
```

After this step, the compact models should have this layout:

```text
diffusion_checkpoints/
├── promoter_direct/checkpoints/best.pth
├── rbs_direct/checkpoints/best.pth
└── terminator_direct/checkpoints/best.pth

predictor_checkpoints/supervised_model/
├── rbs_CNN_Attn_BiLSTM/fold_1/checkpoint.pth ... fold_5/checkpoint.pth
├── terminator_CNN_Attn_BiLSTM/fold_1/checkpoint.pth ... fold_5/checkpoint.pth
└── terminator_CNN_Attn_BiLSTM_cla/fold_1/checkpoint.pth ... fold_5/checkpoint.pth
```

### Optional: enable promoter prediction

The compact archive does not redistribute the frozen Evo 1.5 backbone. For
promoter prediction or promoter library design:

1. download
   [`evo-design/evo-1.5-8k-base`](https://huggingface.co/evo-design/evo-1.5-8k-base)
   into `src/evo/models/evo-1.5-8k-base/`;
2. follow `DeepBioParts_weights_v1.0/README.md` and use
   `DeepBioParts_weights_v1.0/tools/merge_lora_adapter.py` to merge the supplied
   adapter with that backbone;
3. place the merged checkpoint and `label_scaler.pkl` under
   `predictor_checkpoints/language_model/promoter_LoRA_finetune/`.

The final promoter checkpoint path used by the default configuration is:

```text
predictor_checkpoints/language_model/promoter_LoRA_finetune/best_model.pth
```

Inspect the merge utility's exact arguments with:

```bash
python DeepBioParts_weights_v1.0/tools/merge_lora_adapter.py --help
```

## 3. Verify installation and configuration

```bash
# Expected: Python 3.10, CUDA available: True, and a visible GPU name.
python -c "import sys, torch; print(sys.version); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA device')"

# Confirm that the public command-line interfaces load successfully.
python scripts/design_iterative_library.py --help
python scripts/predict.py --help

# Confirm the files needed by the compact RBS demonstration.
test -f diffusion_checkpoints/rbs_direct/checkpoints/best.pth
test -f predictor_checkpoints/supervised_model/rbs_CNN_Attn_BiLSTM/fold_5/checkpoint.pth
```

Default model locations and activity transformations are defined in
`configs/bioparts/promoter.yaml`, `rbs.yaml`, and `terminator.yaml`. Pass
`--generator-dir`, `--predictor-dir`, or `--model_dir` to override those paths
without editing the YAML files.

## 4. Quick start

The examples below use explicit output directories so every expected artifact
is easy to find. A reproducible RBS demo and small reference outputs are
provided in `examples/`.

### 4.1 *De novo* design of a non-repetitive library

The RBS example uses only the compact weights and is the simplest full design
demonstration.

```bash
/usr/bin/time -v python scripts/design_iterative_library.py \
  --biopart rbs \
  --generator-dir diffusion_checkpoints/rbs_direct \
  --predictor-dir predictor_checkpoints/supervised_model/rbs_CNN_Attn_BiLSTM \
  --output-dir results/demo_rbs_library \
  --target-library-size 100 \
  --lmax 10
```

Expected terminal output contains one line per round, for example
`round=1 size=... accepted=... swapped=... coverage_loss=...`, followed by:

```text
complete=True output=results/demo_rbs_library
```

The command returns exit code 0 when the target library is complete and exit
code 2 if it stops before reaching the target. In either case it writes:

```text
results/demo_rbs_library/
├── designed_library.csv  # sequence, predicted_activity, activity_quantile, accepted_round
├── round_history.json    # per-round size, acceptance, swaps, and coverage loss
└── run_summary.json      # completion state, final size, coverage, paths, device, and seed
```

Runtime depends on the number of design rounds, DDPM sampling, and the
acceptance rate under the homology constraint. If `complete` is false, inspect
`run_summary.json` and increase `--max-rounds` or `--generation-batch`.

### 4.2 Predict activity for a sequence

```bash
# Single 15 bp RBS.
/usr/bin/time -v python scripts/predict.py \
  --sequence AAGGAGGTAAAAAAT \
  --biopart rbs \
  --output_dir results/demo_rbs_prediction

# Long input: sliding-window scanning is enabled by default.
python scripts/predict.py \
  --sequence ATGCGTGAGGAGGCTATCGATCGATCG \
  --biopart rbs \
  --output_dir results/demo_rbs_scan

# Batch prediction; the CSV must contain a `sequence` column.
python scripts/predict.py \
  --data_path ./my_seqs.csv \
  --biopart rbs \
  --output_dir results/demo_rbs_batch
```

For the single-sequence command, terminal output reports the model path, raw
fitness, calibrated activity, ensemble uncertainty, result count, and output
directory. It writes all of the following:

```text
results/demo_rbs_prediction/
├── data/predictions.csv  # sequence, length, fitness, activity, and uncertainty
├── scan_plot.svg         # editable vector plot
└── scan_plot.png         # raster preview
```

Batch and sliding-window runtimes scale with the number of sequences/windows.
Promoter prediction additionally requires loading the Evo backbone.

If `--output_dir` is omitted, prediction outputs are written to
`results/quick_predict/predict_<YYYYMMDD_HHMMSS>/` with the same internal
layout.

## 5. Reproduce the reported test-set metrics

Four independent held-out test datasets are provided in [`data/test/`](data/test/).
They contain 100 promoter, 100 RBS, 100 continuous-activity terminator, and 100
strong-terminator sequences with experimental measurements. Model predictions
are not included. After installing the released weights and the Evo promoter
backbone as described above, rerun inference for all four test sets with:

```bash
python scripts/reproduce_test_results.py \
  --device cuda:0 \
  --batch-size 8
```

The script loads the four predictor configurations, generates new predictions,
recomputes the three Pearson correlations and strong-terminator AUROC, and
writes all predictions and metrics under `results/test_reproduction/`. See
[`data/test/README.md`](data/test/README.md) for column definitions, model
requirements, the classification threshold, output files, and reference values.

## Citation

If you use DeepBioParts, please cite:

```bibtex
@article{deepbioparts2026,
  title   = {DeepBioParts: A generative framework for the de novo design of
             non-repetitive regulatory part library},
  author  = {Leng, Yanbing},
  year    = {2026},
  journal = {},
  doi     = {},
}
```

## License

[MIT](LICENSE) — see `pyproject.toml`.
