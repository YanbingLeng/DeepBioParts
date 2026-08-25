#!/usr/bin/env python3
"""Aggregation script for predictor ablation experiments.

Reads, from every run directory (<runs-root>/<task>/<family>-<variant>/seed_<S>/):
- run_config.json / parameter_counts.json / fold_metrics.csv / fold_details.jsonl / test_results.csv

Generates (written to --out-dir, default results/ablations/):
1. run_manifest.csv        — completion status of each run
2. metrics_long.csv        — (task,biopart,model_family,variant,seed,fold,split) x full metric set
3. predictions_long.csv    — per-fold, per-sample predictions (including similarity passthrough columns)
4. ablation_summary.csv    — paired statistics, Full vs variant (bootstrap CI + sign-flip permutation + Holm)
5. similarity_bin_summary.csv — ΔPearson stratified by max_train_similarity (regression tasks only)

Statistics:
- Blocking: seed x fold (per-fold primary metric on the fixed test set: regression -> Pearson r, classification -> AUROC)
- 10000 bootstrap iterations for the 95% CI of Δ (percentile)
- Two-sided sign-flip permutation test (10000 iterations)
- Holm correction within each task -> q_value
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.data.ablation_split import FIXED_TEST_FILES  # (biopart, task_type) -> filename


# ---------------------------------------------------------------------------
# Variant / family inference
# ---------------------------------------------------------------------------

EVO_VARIANT_MAP = {
    ("lora", "attention"): "full",
    ("head_only", "attention"): "head_only",
    ("lora", "mean"): "mean_pool",
}

METRIC_COLS = ["pearson_r", "spearman_r", "rmse", "mae", "auroc", "auprc",
               "trainable_params", "total_params", "best_epoch", "runtime_sec"]


def derive_family_variant(cfg):
    """Infer (model_family, variant) from run_config.

    The family is simply model_type ('evo' / 'attnbilstm' / '1dcnn'), matching the
    scheduler registry so that the DL figure (attnbilstm + 1dcnn) and the Evo
    figure (evo) can be grouped separately.
    """
    model_type = cfg.get("model_type")
    if model_type == "evo":
        family = "evo"
        key = (cfg.get("evo_adaptation"), cfg.get("pooling_mode"))
        variant = EVO_VARIANT_MAP.get(key, cfg.get("variant_tag", "unknown"))
    else:
        # attnbilstm / 1dcnn: family = model_type, variant = ablation_variant
        family = model_type or "unknown"
        variant = cfg.get("ablation_variant") or "full"
    return family, variant


def task_id_of(cfg):
    return f"{cfg.get('biopart')}_{cfg.get('task_type')}"


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _safe_pearson(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return np.nan
    return float(pearsonr(y_true, y_pred)[0])


def _safe_spearman(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return np.nan
    return float(spearmanr(y_true, y_pred)[0])


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(np.mean((y_true - y_pred) ** 2)) if len(y_true) else np.nan
    return {
        "pearson_r": _safe_pearson(y_true, y_pred),
        "spearman_r": _safe_spearman(y_true, y_pred),
        "rmse": float(np.sqrt(mse)) if np.isfinite(mse) else np.nan,
        "mae": float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else np.nan,
        "auroc": np.nan,
        "auprc": np.nan,
    }


def classification_metrics(y_true, y_prob):
    """y_prob is the class-1 probability."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    out = {"pearson_r": np.nan, "spearman_r": np.nan, "rmse": np.nan, "mae": np.nan}
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        if len(np.unique(y_true)) == 2:
            out["auroc"] = float(roc_auc_score(y_true, y_prob))
            out["auprc"] = float(average_precision_score(y_true, y_prob))
        else:
            out["auroc"] = np.nan
            out["auprc"] = np.nan
    except Exception:
        out["auroc"] = np.nan
        out["auprc"] = np.nan
    return out


def primary_metric_for(task_type):
    """Column name of the primary metric used for statistics / plotting."""
    return "auroc" if task_type == "classification" else "pearson_r"


# ---------------------------------------------------------------------------
# Statistics: bootstrap CI + sign-flip permutation + Holm
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(diffs, n_boot, rng):
    """diffs: array of paired differences. Returns (mean, ci_low, ci_high) as a 95% percentile CI."""
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return np.nan, np.nan, np.nan
    obs = float(np.mean(diffs))
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    boot_means = diffs[idx].mean(axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return obs, float(ci_low), float(ci_high)


def sign_flip_permutation_p(diffs, n_perm, rng):
    """Two-sided sign-flip permutation test."""
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    n = len(diffs)
    if n == 0:
        return np.nan
    obs = float(np.mean(diffs))
    signs = rng.integers(0, 2, size=(n_perm, n)).astype(np.int8) * 2 - 1  # ±1
    perm_means = (signs * diffs).mean(axis=1)
    p = (np.sum(np.abs(perm_means) >= np.abs(obs)) + 1) / (n_perm + 1)
    return float(p)


def holm_correct(pvals):
    """Holm-Bonferroni correction; returns q values in the same order as the input."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    q = np.empty(n, dtype=float)
    q[:] = np.nan
    finite = np.isfinite(pvals)
    if finite.sum() == 0:
        return q
    idx = np.where(finite)[0]
    order = idx[np.argsort(pvals[idx])]
    prev = 0.0
    for rank, i in enumerate(order, start=1):
        adj = (n - rank + 1) * pvals[i]
        adj = max(prev, adj)  # enforce monotone non-decreasing q values
        adj = min(adj, 1.0)
        q[i] = adj
        prev = adj
    return q


# ---------------------------------------------------------------------------
# Run discovery and loading
# ---------------------------------------------------------------------------

def discover_runs(runs_root):
    """Return [run_dir, ...], each containing run_config.json."""
    runs = []
    root = Path(runs_root)
    if not root.exists():
        return runs
    for cfg_path in sorted(root.rglob("run_config.json")):
        runs.append(cfg_path.parent)
    return runs


def load_run(run_dir):
    """Load all artifacts of a single run; returns a dict (or None if a key file is missing)."""
    run_dir = Path(run_dir)
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        cfg = json.load(f)

    out = {"run_dir": str(run_dir), "cfg": cfg}
    # parameter_counts
    pc_path = run_dir / "parameter_counts.json"
    out["params"] = {}
    if pc_path.exists():
        with open(pc_path) as f:
            out["params"] = json.load(f)
    # fold_details
    fd_path = run_dir / "fold_details.jsonl"
    out["fold_details"] = []
    if fd_path.exists():
        with open(fd_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out["fold_details"].append(json.loads(line))
    # test_results
    tr_path = run_dir / "test_results.csv"
    out["test_df"] = pd.read_csv(tr_path) if tr_path.exists() else None
    # Count fold directories (only fold_<int> dirs; excludes fold_details.jsonl / fold_metrics.csv)
    import re
    out["n_fold_dirs"] = sum(
        1 for d in run_dir.iterdir()
        if d.is_dir() and re.fullmatch(r"fold_\d+", d.name)
    )
    out["has_test_results"] = tr_path.exists()
    out["mtime"] = os.path.getmtime(cfg_path)
    return out


def fold_pred_columns(test_df):
    """Return the Fold_k_Prediction column names in test_df, sorted by k."""
    cols = [c for c in test_df.columns if c.startswith("Fold_") and c.endswith("_Prediction")]
    def _k(c):
        try:
            return int(c.split("_")[1])
        except Exception:
            return 0
    return sorted(cols, key=_k)


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def aggregate(args):
    runs = discover_runs(args.runs_root)
    if not runs:
        print(f"[aggregate] No runs found: {args.runs_root}")
        return
    print(f"[aggregate] Found {len(runs)} runs")

    loaded = []
    for rd in runs:
        rec = load_run(rd)
        if rec is None:
            continue
        family, variant = derive_family_variant(rec["cfg"])
        rec["family"] = family
        rec["variant"] = variant
        rec["task_id"] = task_id_of(rec["cfg"])
        loaded.append(rec)

    rng = np.random.default_rng(args.seed)

    # ===== 1. run_manifest =====
    manifest_rows = []
    for r in loaded:
        cfg = r["cfg"]
        n_folds_cfg = cfg.get("n_folds", r["n_fold_dirs"])
        complete = r["has_test_results"] and r["n_fold_dirs"] >= n_folds_cfg
        status = "complete" if complete else ("partial" if r["has_test_results"] else "missing")
        fixed_file = FIXED_TEST_FILES.get((cfg.get("biopart"), cfg.get("task_type")), "")
        manifest_rows.append({
            "task": r["task_id"], "biopart": cfg.get("biopart"), "task_type": cfg.get("task_type"),
            "model_family": r["family"], "variant": r["variant"], "seed": cfg.get("seed"),
            "run_dir": r["run_dir"], "status": status, "n_folds": n_folds_cfg,
            "n_fold_checkpoints": r["n_fold_dirs"], "has_test_results": r["has_test_results"],
            "fixed_test_file": fixed_file, "timestamp": int(r["mtime"]),
        })
    manifest_df = pd.DataFrame(manifest_rows)

    # ===== 2 & 3. metrics_long + predictions_long =====
    metric_rows = []
    pred_rows = []
    for r in loaded:
        cfg = r["cfg"]
        task_type = cfg.get("task_type")
        biopart = cfg.get("biopart")
        seed = cfg.get("seed")
        params = r["params"]
        trainable = params.get("trainable", np.nan)
        total = params.get("total", np.nan)

        common = dict(
            task=r["task_id"], biopart=biopart, task_type=task_type,
            model_family=r["family"], variant=r["variant"], seed=seed,
            trainable_params=trainable, total_params=total,
        )

        # ---- val rows (fold_details.jsonl) ----
        for fd in r["fold_details"]:
            row = dict(common)
            row["fold"] = fd.get("fold")
            row["split"] = "val"
            for k in METRIC_COLS[:-4]:  # pearson_r..auprc
                row[k] = fd.get(k, np.nan)
            row["best_epoch"] = fd.get("best_epoch", np.nan)
            row["runtime_sec"] = fd.get("runtime_sec", np.nan)
            metric_rows.append(row)

        # ---- test rows (test_results.csv, per-fold + ensemble) ----
        if r["test_df"] is None:
            continue
        tdf = r["test_df"]
        y_true = tdf["y_true"].values
        fold_cols = fold_pred_columns(tdf)
        ens = tdf["ensemble_prediction"].values

        # Per fold
        for k, col in enumerate(fold_cols, start=1):
            y_pred = tdf[col].values
            if task_type == "classification":
                m = classification_metrics(y_true, y_pred)
            else:
                m = regression_metrics(y_true, y_pred)
            row = dict(common); row["fold"] = k; row["split"] = "test"
            row.update(m); row["best_epoch"] = np.nan; row["runtime_sec"] = np.nan
            metric_rows.append(row)
            # predictions_long
            sim = tdf.get("max_3mer_cosine_sim")
            sbin = tdf.get("sim_bin")
            for i in range(len(tdf)):
                pred_rows.append({
                    **{kk: common[kk] for kk in ("task", "biopart", "model_family", "variant", "seed")},
                    "fold": k, "sample_id": int(tdf["sample_id"].iloc[i]),
                    "y_true": y_true[i], "y_pred": y_pred[i],
                    "max_train_similarity": float(sim.iloc[i]) if sim is not None else np.nan,
                    "sim_bin": int(sbin.iloc[i]) if (sbin is not None and pd.notna(sbin.iloc[i])) else np.nan,
                })
        # Ensemble row
        if task_type == "classification":
            m = classification_metrics(y_true, ens)
        else:
            m = regression_metrics(y_true, ens)
        row = dict(common); row["fold"] = "ensemble"; row["split"] = "test"
        row.update(m); row["best_epoch"] = np.nan; row["runtime_sec"] = np.nan
        metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)
    preds_df = pd.DataFrame(pred_rows)

    # Ensure all columns are present and ordered
    front = ["task", "biopart", "model_family", "variant", "seed", "fold", "split"]
    metrics_df = metrics_df.reindex(columns=front + METRIC_COLS)

    # ===== 4. ablation_summary (paired statistics) =====
    summary_rows = []
    for (task, family), grp in metrics_df.groupby(["task", "model_family"]):
        task_type = (manifest_df[(manifest_df.task == task) & (manifest_df.model_family == family)]
                     ["task_type"].dropna().iloc[0]
                     if ((manifest_df.task == task) & (manifest_df.model_family == family)).any() else "regression")
        metric_col = primary_metric_for(task_type)
        # Per-fold test metrics: (seed, fold) -> value
        test_grp = grp[(grp.split == "test") & (grp.fold != "ensemble")]
        full_rows = test_grp[test_grp.variant == "full"]
        if full_rows.empty:
            continue
        full_map = {}
        for _, rr in full_rows.iterrows():
            full_map[(rr.seed, rr.fold)] = rr[metric_col]
        # Each non-full variant
        comparisons = []  # (variant, p_value, row_partial)
        for variant, vgrp in test_grp.groupby("variant"):
            if variant == "full":
                continue
            diffs = []
            full_vals, var_vals = [], []
            for _, rr in vgrp.iterrows():
                fv = full_map.get((rr.seed, rr.fold))
                if fv is None or not np.isfinite(fv) or not np.isfinite(rr[metric_col]):
                    continue
                diffs.append(rr[metric_col] - fv)
                full_vals.append(fv); var_vals.append(rr[metric_col])
            if not diffs:
                continue
            diffs = np.array(diffs)
            delta, ci_low, ci_high = paired_bootstrap_ci(diffs, args.n_bootstrap, rng)
            p = sign_flip_permutation_p(diffs, args.n_permutation, rng)
            comparisons.append({
                "task": task, "biopart": grp.biopart.iloc[0], "model_family": family,
                "variant": variant, "metric": metric_col, "n_pairs": len(diffs),
                "delta": delta, "ci_low": ci_low, "ci_high": ci_high, "p_value": p,
                "mean_full": float(np.mean(full_vals)), "mean_variant": float(np.mean(var_vals)),
            })
        # Holm correction (within task)
        qs = holm_correct([c["p_value"] for c in comparisons])
        for c, q in zip(comparisons, qs):
            c["q_value"] = q
            summary_rows.append(c)

    SUMMARY_COLS = ["task", "biopart", "model_family", "variant", "metric",
                    "n_pairs", "delta", "ci_low", "ci_high", "p_value", "q_value",
                    "mean_full", "mean_variant"]
    # Always include column names (the header is written even with no comparisons,
    # so downstream readers never see a 0-byte file)
    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLS)

    # ===== 5. similarity_bin_summary (regression only) =====
    # Four bins by max_train_similarity quantiles (the test set is identical for all runs):
    #   top10 (>=p90) / 10-30 [p70,p90) / 30-60 [p40,p70) / bottom40 (<p40)
    sim_rows = []
    SIM_BINS = [
        ("top10", lambda s, t: s >= t[90]),
        ("10-30", lambda s, t: (s >= t[70]) & (s < t[90])),
        ("30-60", lambda s, t: (s >= t[40]) & (s < t[70])),
        ("bottom40", lambda s, t: s < t[40]),
    ]
    for task, tgrp in preds_df.groupby("task"):
        tt = (manifest_df[manifest_df.task == task]["task_type"].dropna().iloc[0]
              if (manifest_df.task == task).any() else "regression")
        if tt == "classification":
            continue  # classification has no similarity fields; excluded
        # Define quantile thresholds from max_train_similarity in predictions_long
        # (the test set is identical for all runs)
        sims = tgrp["max_train_similarity"].dropna().unique()
        if len(sims) < 8:
            continue
        thresh = {p: np.percentile(sims, p) for p in (40, 70, 90)}
        # Collect test_df for each (family, variant, seed)
        def _seed_dfs(family, variant):
            out = {}
            for r in loaded:
                if r["family"] == family and r["variant"] == variant and r["task_id"] == task and r["test_df"] is not None:
                    out[r["cfg"].get("seed")] = r["test_df"]
            return out
        all_fv = {(fam, var): _seed_dfs(fam, var)
                  for (fam, var), _ in tgrp.groupby(["model_family", "variant"])}
        for (family, variant), seed_to_df in all_fv.items():
            full_seed_to_df = all_fv.get((family, "full"), {})
            for (bname, masker) in SIM_BINS:
                d_var, d_full = [], []
                n_samples = 0
                for seed, df in seed_to_df.items():
                    fdf = full_seed_to_df.get(seed)
                    if fdf is None or "max_3mer_cosine_sim" not in df.columns:
                        continue
                    mask = masker(df["max_3mer_cosine_sim"].values, thresh)
                    if mask.sum() < 3:
                        continue
                    pv = _safe_pearson(df.loc[mask, "y_true"], df.loc[mask, "ensemble_prediction"])
                    fv = _safe_pearson(fdf.loc[mask, "y_true"], fdf.loc[mask, "ensemble_prediction"])
                    if np.isfinite(pv) and np.isfinite(fv):
                        d_var.append(pv); d_full.append(fv)
                        n_samples = int(mask.sum())
                if d_var:
                    sim_rows.append({
                        "task": task, "biopart": tgrp.biopart.iloc[0], "model_family": family,
                        "variant": variant, "bin": bname, "n": n_samples,
                        "pearson_full": float(np.mean(d_full)),
                        "pearson_variant": float(np.mean(d_var)),
                        "delta_pearson": float(np.mean(d_var) - np.mean(d_full)),
                    })
    sim_df = pd.DataFrame(sim_rows)

    # ===== Write outputs =====
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(out_dir / "run_manifest.csv", index=False)
    metrics_df.to_csv(out_dir / "metrics_long.csv", index=False)
    preds_df.to_csv(out_dir / "predictions_long.csv", index=False)
    summary_df.to_csv(out_dir / "ablation_summary.csv", index=False)
    sim_df.to_csv(out_dir / "similarity_bin_summary.csv", index=False)

    print(f"[aggregate] Wrote 5 tables to {out_dir}")
    print(f"  run_manifest: {len(manifest_df)} rows | metrics_long: {len(metrics_df)} rows | "
          f"predictions_long: {len(preds_df)} rows")
    print(f"  ablation_summary: {len(summary_df)} rows | similarity_bin_summary: {len(sim_df)} rows")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--runs-root", default="predictor_checkpoints/ablation")
    p.add_argument("--out-dir", default="results/ablations")
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--n-permutation", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    aggregate(parse_args())
