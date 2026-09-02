# RBS demo examples

This directory contains a reproducible RBS (ribosome binding site) demo for
DeepBioParts. The root [README](../README.md#4-quick-start) describes the main
design and prediction workflows.

## Files

- `demo_rbs.csv` — demo input: three 15 bp RBS sequences taken from
  `results/biopart_database/NR_rbs.csv` (sequence column only).
- `run_rbs_demo.sh` — runs three demos with fixed parameters and seed 42:
  A. single-sequence activity prediction,
  B. batch prediction of `demo_rbs.csv`,
  C. *de novo* design of a 100-sequence non-repetitive RBS library.
- `expected_outputs/` — small reference outputs of one run (prediction CSVs,
  designed library, round history, and run summary) for comparison. The run
  uses `--seed 42`; small differences may still occur across hardware or
  software stacks. Machine-specific paths are normalized in the reference
  JSON.

## Run

The script locates the repository root from its own location, so it can be
called from any working directory. Activate the environment first; the script
does not activate Conda itself.

```bash
conda activate deepbioparts
./examples/run_rbs_demo.sh
```

Each demo writes its log, `/usr/bin/time -v` resource report, and real exit
code to the Git-ignored `examples/demo_outputs/`; a failing demo does not stop
the others.
The design demo returns exit code 0 when the 100-sequence target is complete
and exit code 2 when it stops early — check
`examples/demo_outputs/rbs_library_design/run_summary.json` for the completion
state.
