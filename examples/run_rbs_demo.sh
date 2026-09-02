#!/usr/bin/env bash
# Reproducible RBS demo for DeepBioParts.
#
# Runs three demos with fixed parameters and seed 42:
#   A. Single-sequence RBS activity prediction
#   B. CSV batch prediction (examples/demo_rbs.csv)
#   C. De novo design of a 100-sequence non-repetitive RBS library
#
# Each demo writes its full log, /usr/bin/time -v resource report, and real
# exit code under examples/demo_outputs/. A failing demo does not stop the
# remaining demos.
#
# Usage (from any working directory, with the deepbioparts environment active):
#   conda activate deepbioparts
#   ./examples/run_rbs_demo.sh

set -uo pipefail

# Locate the repository root from this script's location, so the demo works
# from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

OUT_DIR="$REPO_ROOT/examples/demo_outputs"
mkdir -p "$OUT_DIR"
EXIT_CODES="$OUT_DIR/exit_codes.txt"
: > "$EXIT_CODES"

# --- Check the Python environment instead of activating Conda here ---------
echo "[demo] repository root: $REPO_ROOT"
echo "[demo] python: $(command -v python || echo 'python not found on PATH')"
python --version || exit 1
if ! python -c "import torch" 2>/dev/null; then
    echo "[demo] ERROR: PyTorch is not importable. Activate the deepbioparts environment first:"
    echo "[demo]        conda activate deepbioparts"
    exit 1
fi
if ! python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "[demo] WARNING: CUDA is not available; prediction and design need a GPU."
fi

# --- Record the reference environment --------------------------------------
{
    date -Is
    echo "git_commit: $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    python --version 2>&1
    python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('CUDA runtime:', torch.version.cuda)"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
} > "$OUT_DIR/environment.txt" 2>&1

run_demo() {
    # run_demo <name> <command...>
    local name="$1"; shift
    local log="$OUT_DIR/$name.log"
    local time_file="$OUT_DIR/$name.time.txt"
    echo "[demo] starting $name: $*"
    /usr/bin/time -v -o "$time_file" "$@" > "$log" 2>&1
    local code=$?
    echo "$name $code" >> "$EXIT_CODES"
    echo "[demo] $name finished with exit code $code (log: $log)"
    return 0
}

cd "$REPO_ROOT"

# --- Demo A: single-sequence RBS activity prediction ------------------------
run_demo rbs_single_prediction \
    python scripts/predict.py \
      --sequence AAGGAGGTAAAAAAT \
      --biopart rbs \
      --output_dir examples/demo_outputs/rbs_single_prediction

# --- Demo B: batch prediction from CSV ---------------------------------------
run_demo rbs_batch_prediction \
    python scripts/predict.py \
      --data_path examples/demo_rbs.csv \
      --biopart rbs \
      --output_dir examples/demo_outputs/rbs_batch_prediction

# --- Demo C: de novo design of a 100-sequence non-repetitive RBS library ----
# Exit code 0 = target reached; exit code 2 = stopped before the target
# (see run_summary.json in the output directory).
run_demo rbs_library_design \
    python scripts/design_iterative_library.py \
      --biopart rbs \
      --generator-dir diffusion_checkpoints/rbs_direct \
      --predictor-dir predictor_checkpoints/supervised_model/rbs_CNN_Attn_BiLSTM \
      --output-dir examples/demo_outputs/rbs_library_design \
      --target-library-size 100 \
      --lmax 10 \
      --seed 42

echo "[demo] all demos attempted. Exit codes:"
cat "$EXIT_CODES"
