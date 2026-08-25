# DeepBioParts

**A generative framework for the *de novo* design of non-repetitive regulatory part libraries.**

DeepBioParts treats the regulatory part library — not the individual
sequence — as the design object. It couples *de novo* sequence generation
(DDPM), activity prediction (Evo 1.5 + LoRA and CNN–Attention–BiLSTM) and
selection under pairwise homology constraints ($L_{\max} \le 10$ bp) in an
iterative generate–select–predict–update loop, producing promoter, RBS and
terminator libraries with broad, graded activity coverage and low sequence
homology for quantitative expression tuning and genetically stable circuits.

---

## Installation

Requires **Python 3.10** and a CUDA-capable GPU (driver ≥ 525 / CUDA 12.x).

```bash
git clone https://github.com/YanbingLeng/DeepBioParts.git
cd DeepBioParts

# 1. Create the env (Python 3.10, all deps via pip)
conda env create -f environment.yml
conda activate deepbioparts

# 2. flash-attn — install from the prebuilt cu12 wheel. Do NOT run `pip install flash-attn`,
#    which builds from source and requires the full CUDA toolkit.
curl -fL -o /tmp/flash_attn.whl \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
cp /tmp/flash_attn.whl /tmp/flash_attn-2.7.4.post1-cp310-cp310-linux_x86_64.whl   # pip rejects '+' in the filename
pip install /tmp/flash_attn-2.7.4.post1-cp310-cp310-linux_x86_64.whl

# Optional: install as an editable package
pip install -e .
```

---

## Quick start

### 1. Predict activity for a sequence

```bash
# Single sequence (uses the default calibrated model for each part)
python scripts/predict.py --sequence AAGGAGGTAAAAAAT --biopart rbs

# Long sequence → sliding-window scan (enabled by default)
python scripts/predict.py --sequence ATGCGTGAGGAGGCTATCGATCGATCG --biopart rbs

# Batch predict from CSV
python scripts/predict.py --data_path ./my_seqs.csv --biopart rbs
```

### 2. Train a predictor

```bash
# CNN–Attention–BiLSTM with 5-fold CV (promoter / rbs / terminator)
python scripts/train_predictor.py \
    --model_type attnbilstm --biopart promoter --encoding_type onehot \
    --boxcox --n_folds 5 --data_path <training-CSV>

# Evo 1.5 + LoRA (same deepbioparts env — flash-attn included)
python scripts/train_predictor.py \
    --model_type evo --biopart promoter --n_folds 5 --lora_rank 16 \
    --data_path <training-CSV>
```

### 3. Train the DDPM sequence generator

```bash
# Direct one-hot DDPM (manuscript Methods; T=1000, linear beta 1e-4–2e-2)
python scripts/train_ddpm.py --biopart promoter --data_path <training-CSV>
# writes diffusion_checkpoints/promoter_direct/checkpoints/best.pth
```

### 4. *De novo* design non-repetitive library

```bash
# Iterative generate–select–predict–update cycle with coverage-loss
# optimization under the Lmax constraint (NRP Calculator backend)
python scripts/design_iterative_library.py \
    --biopart promoter \
    --generator-dir diffusion_checkpoints/promoter_direct \
    --target-library-size 100 --lmax 10
```

---

## Model weights

Trained checkpoints (DDPM generators, supervised predictors, and the
Evo 1.5 + LoRA promoter adapter) are distributed separately:

> **Download**: [10.5281/zenodo.22097086](https://zenodo.org/records/22097086)

After unpacking, place the directories at the repository root so the
defaults above resolve:

```text
diffusion_checkpoints/<biopart>_direct/     # DDPM generators
supervised_model/<model>/fold_<k>/checkpoint.pth   # supervised predictors (5-fold)
```

The **Evo 1.5 + LoRA promoter predictor** ships as a 60 MB adapter
(the 13 GB evo-1.5-8k-base backbone is not redistributed). Download the
backbone from the official Evo release and rebuild the full checkpoint
with the `merge_lora_adapter.py` tool bundled in the weights archive —
the rebuilt checkpoint is bit-identical to the original training
checkpoint. Place the result, together with `label_scaler.pkl`, under
`predictor_checkpoints/language_model/promoter_LoRA_finetune/`.

---

## Citation

If you use DeepBioParts, please cite:

```bibtex
@article{deepbioparts2026,
  title  = {DeepBioParts: A generative framework for the de novo design of
            non-repetitive regulatory part library},
  author = {Leng, Yanbing},
  year   = {2026},
  journal= {},
  doi    = {},
}
```

## License

[MIT](LICENSE) — see `pyproject.toml`.
