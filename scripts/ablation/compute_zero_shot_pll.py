#!/usr/bin/env python3
"""Compute per-sequence zero-shot pseudo-log-likelihood (PLL) scores for Evo.

Used for the Supplementary Fig 3a scatter plot: for every sequence in the fixed test
sets (promoter/rbs/terminator), compute the mean log-likelihood with the pretrained
Evo 1.5 (no LoRA / no head) as the zero-shot score, then correlate it with activity.
Reproduces the r values of zero_shot_pll_evo.csv
(promoter~-0.16, rbs~0.037, terminator~0.031).

Writes results/ablations/zero_shot_pll_perseq.csv:  biopart, sequence, activity, pll_score
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJ = Path(__file__).resolve().parents[2]
for p in [str(PROJ), str(PROJ / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.evo.lora_finetune_evo import load_evo_from_local  # noqa: E402
from src.evo.evo.scoring import score_sequences  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

TEST_FILES = {
    "promoter": "results/test_set_results/test_promoter.csv",
    "rbs": "results/test_set_results/test_rbs.csv",
    "terminator": "results/test_set_results/test_terminator_reg.csv",
}
EVO_LOCAL = "src/evo/models/evo-1.5-8k-base"
DEVICE = "cuda:0"
BATCH = 16


def main():
    print("[zero-shot] Loading pretrained Evo 1.5 (no LoRA / no head)...")
    evo = load_evo_from_local(EVO_LOCAL, model_name="evo-1.5-8k-base", device=DEVICE)
    evo.model.eval()

    out_rows = []
    for bp, path in TEST_FILES.items():
        df = pd.read_csv(PROJ / path)
        seqs = df["sequence"].astype(str).tolist()
        acts = df["activity"].astype(float).values
        print(f"[zero-shot] {bp}: n={len(seqs)}, computing PLL (batch={BATCH}) ...")

        scores = []
        with torch.inference_mode():
            for i in range(0, len(seqs), BATCH):
                chunk = seqs[i:i + BATCH]
                s = score_sequences(chunk, evo.model, evo.tokenizer,
                                    reduce_method="mean", device=DEVICE)
                scores.extend(s)
                if (i // BATCH) % 10 == 0:
                    print(f"    {bp} {i+len(chunk)}/{len(seqs)}")
        scores = np.array(scores, dtype=float)
        r = pearsonr(acts, scores)[0]
        print(f"[zero-shot] {bp}: r(activity, PLL) = {r:.4f}")
        for s, a, sc in zip(seqs, acts, scores):
            out_rows.append({"biopart": bp, "sequence": s, "activity": float(a),
                             "pll_score": float(sc)})

    out = pd.DataFrame(out_rows)
    out_path = PROJ / "results" / "ablations" / "zero_shot_pll_perseq.csv"
    out.to_csv(out_path, index=False)
    print(f"[zero-shot] Wrote {out_path}  ({len(out)} rows)")
    print("[zero-shot] Per-biopart r summary:")
    print(out.groupby("biopart").apply(
        lambda g: pearsonr(g["activity"], g["pll_score"])[0]).to_string())


if __name__ == "__main__":
    main()
