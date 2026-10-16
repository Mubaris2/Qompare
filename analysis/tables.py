"""Generate summary tables, skill scores, drift recovery, and Wilcoxon-Holm tests.

Usage:
  python -m analysis.tables --dirs results/classical results/quantum --out results/tables
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import metrics as M
from common.stream_runner import COLUMNS

PRIMARY_METRIC = {"T1": "mse", "T2": "auc_pooled", "T3": "accuracy"}
LOWER_IS_BETTER = {"mse", "mae"}


def load_all_results(dirs: list[str]) -> pd.DataFrame:
    frames = []
    for d in dirs:
        p = Path(d)
        if not p.exists():
            continue
        for csv_file in sorted(p.glob("*.csv")):
            df = pd.read_csv(csv_file)
            if set(COLUMNS).issubset(df.columns):
                df["variant"] = df["variant"].fillna("default").replace("", "default")
                frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No valid 19-column result CSVs found in {dirs}")
    return pd.concat(frames, ignore_index=True)


def compute_pooled_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean across steps per seed, then mean +/- std across seeds."""
    per_seed = (
        df.groupby(["task", "model", "variant", "W", "H", "drift", "seed", "metric_name"], as_index=False)
        .agg(
            metric_mean=("metric_value", "mean"),
            t_encode_mean=("t_encode", "mean"),
            t_fit_mean=("t_fit", "mean"),
            t_predict_mean=("t_predict", "mean"),
            n_circuit_evals_mean=("n_circuit_evals", "mean"),
            peak_mem_mb_max=("peak_mem_mb", "max"),
        )
    )
    pooled = (
        per_seed.groupby(["task", "model", "variant", "W", "H", "drift", "metric_name"], as_index=False)
        .agg(
            n_seeds=("seed", "count"),
            mean=("metric_mean", "mean"),
            std=("metric_mean", "std"),
            t_encode=("t_encode_mean", "mean"),
            t_fit=("t_fit_mean", "mean"),
            t_predict=("t_predict_mean", "mean"),
            circuit_evals=("n_circuit_evals_mean", "mean"),
            peak_mem_mb=("peak_mem_mb_max", "mean"),
        )
    )
    return pooled


def compute_skill_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute T1 skill vs naive_mean and T3 skill vs naive_last + kappa_plus."""
    records = []

    # T1: Skill vs naive_mean = 1 - (MSE_model / MSE_naive_mean)
    t1 = df[(df["task"] == "T1") & (df["metric_name"] == "mse")]
    if not t1.empty and "naive_mean" in t1["model"].unique():
        base = (
            t1[t1["model"] == "naive_mean"]
            .groupby(["W", "drift", "seed", "step"])["metric_value"]
            .mean()
            .rename("base_mse")
        )
        merged = t1.merge(base, on=["W", "drift", "seed", "step"], how="inner")
        merged["skill_vs_naive_mean"] = 1.0 - (merged["metric_value"] / np.maximum(merged["base_mse"], 1e-12))
        grp = (
            merged.groupby(["task", "model", "variant", "W", "drift"], as_index=False)
            .agg(
                mse_mean=("metric_value", "mean"),
                skill_mean=("skill_vs_naive_mean", "mean"),
                skill_std=("skill_vs_naive_mean", "std"),
            )
        )
        grp["skill_type"] = "T1_skill_vs_naive_mean"
        records.append(grp)

    # T3: Skill vs naive_last (or acc_nochange) = (acc - acc_ref) / (1 - acc_ref)
    t3 = df[df["task"] == "T3"]
    if not t3.empty:
        acc_df = t3[t3["metric_name"] == "accuracy"]
        kp_df = t3[t3["metric_name"] == "kappa_plus"]

        if "naive_last" in acc_df["model"].unique():
            ref = (
                acc_df[acc_df["model"] == "naive_last"]
                .groupby(["W", "drift", "seed", "step"])["metric_value"]
                .mean()
                .rename("ref_acc")
            )
            m3 = acc_df.merge(ref, on=["W", "drift", "seed", "step"], how="inner")
            denom = np.where(1.0 - m3["ref_acc"] > 1e-9, 1.0 - m3["ref_acc"], np.nan)
            m3["skill_val"] = (m3["metric_value"] - m3["ref_acc"]) / denom
        else:
            m3 = kp_df.copy()
            m3["skill_val"] = m3["metric_value"]

        grp3 = (
            m3.groupby(["task", "model", "variant", "W", "drift"], as_index=False)
            .agg(
                acc_or_kp_mean=("metric_value", "mean"),
                skill_mean=("skill_val", "mean"),
                skill_std=("skill_val", "std"),
            )
        )
        grp3["skill_type"] = "T3_skill_vs_naive_last"
        records.append(grp3)

    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def _call_recovery_time(series: np.ndarray, drift_idx: int, lower_is_better: bool) -> float:
    """Use common.metrics.recovery_time if available, else fallback to 90% pre-drift recovery."""
    if hasattr(M, "recovery_time"):
        try:
            return float(M.recovery_time(series, drift_idx))
        except TypeError:
            try:
                return float(M.recovery_time(series[:drift_idx], series[drift_idx:]))
            except Exception:
                pass

    if drift_idx <= 0 or drift_idx >= len(series):
        return float("nan")
    pre = float(np.nanmean(series[max(0, drift_idx - 5) : drift_idx]))
    post = series[drift_idx:]
    threshold = pre * 1.10 if lower_is_better else pre * 0.90
    for i, val in enumerate(post):
        if np.isnan(val):
            continue
        if (lower_is_better and val <= threshold) or (not lower_is_better and val >= threshold):
            return float(i)
    return float(len(post))


def compute_drift_recovery(df: pd.DataFrame) -> pd.DataFrame:
    """Compute steps to recover after drift_flag transitions from 0 to 1."""
    drifted = df[df["drift"].isin(["abrupt", "gradual"])]
    rows = []
    for (task, model, variant, W, drift, seed), grp in drifted.groupby(
        ["task", "model", "variant", "W", "drift", "seed"]
    ):
        pm = PRIMARY_METRIC.get(task)
        sub = grp[grp["metric_name"] == pm].sort_values("step")
        if sub.empty:
            continue
        flags = sub["drift_flag"].to_numpy()
        onset = np.where(flags == 1)[0]
        if len(onset) == 0:
            continue
        drift_idx = int(onset[0])
        vals = sub["metric_value"].to_numpy(dtype=float)
        rec_steps = _call_recovery_time(vals, drift_idx, pm in LOWER_IS_BETTER)
        rows.append(
            dict(
                task=task,
                model=model,
                variant=variant,
                W=W,
                drift=drift,
                seed=seed,
                recovery_steps=rec_steps,
            )
        )
    if not rows:
        return pd.DataFrame()
    rec_df = pd.DataFrame(rows)
    return (
        rec_df.groupby(["task", "model", "variant", "W", "drift"], as_index=False)
        .agg(
            recovery_steps_mean=("recovery_steps", "mean"),
            recovery_steps_std=("recovery_steps", "std"),
        )
    )


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm step-down p-value adjustment."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adjusted = np.zeros(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = p_values[idx] * (m - rank)
        running_max = max(running_max, adj)
        adjusted[idx] = min(1.0, running_max)
    return adjusted.tolist()


def compute_wilcoxon_holm(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank tests between quantum and classical models with Holm correction."""
    df = df.copy()
    df["is_quantum"] = (df["n_qubits"].fillna(0) > 0) | (
        df["model"].str.lower().isin({"vqr", "qsvr", "qsvc", "qnn"})
    )

    results = []
    for task, tdf in df.groupby("task"):
        pm = PRIMARY_METRIC.get(task)
        sub = tdf[tdf["metric_name"] == pm]
        q_models = sub[sub["is_quantum"]][["model", "variant"]].drop_duplicates().to_dict("records")
        c_models = sub[~sub["is_quantum"]][["model", "variant"]].drop_duplicates().to_dict("records")

        task_rows = []
        for qm in q_models:
            q_slice = (
                sub[(sub["model"] == qm["model"]) & (sub["variant"] == qm["variant"])]
                .groupby(["W", "drift", "seed"])["metric_value"]
                .mean()
                .rename("q_val")
            )
            for cm in c_models:
                c_slice = (
                    sub[(sub["model"] == cm["model"]) & (sub["variant"] == cm["variant"])]
                    .groupby(["W", "drift", "seed"])["metric_value"]
                    .mean()
                    .rename("c_val")
                )
                paired = pd.concat([q_slice, c_slice], axis=1).dropna()
                if len(paired) < 5:
                    continue
                diff = paired["q_val"] - paired["c_val"]
                if np.allclose(diff, 0.0):
                    stat, p_raw = 0.0, 1.0
                else:
                    res = wilcoxon(paired["q_val"], paired["c_val"], zero_method="zsplit")
                    stat, p_raw = float(res.statistic), float(res.pvalue)

                q_mean = float(paired["q_val"].mean())
                c_mean = float(paired["c_val"].mean())
                q_better = (q_mean < c_mean) if (pm in LOWER_IS_BETTER) else (q_mean > c_mean)
                task_rows.append(
                    dict(
                        task=task,
                        metric=pm,
                        quantum_model=f"{qm['model']} ({qm['variant']})",
                        classical_model=f"{cm['model']} ({cm['variant']})",
                        n_pairs=len(paired),
                        q_mean=q_mean,
                        c_mean=c_mean,
                        quantum_wins_mean=q_better,
                        wilcoxon_stat=stat,
                        p_unadjusted=p_raw,
                    )
                )

        if task_rows:
            p_adj = holm_bonferroni([r["p_unadjusted"] for r in task_rows])
            for r, pa in zip(task_rows, p_adj):
                r["p_holm"] = pa
                r["significant_005"] = pa < 0.05
            results.extend(task_rows)

    return pd.DataFrame(results)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+", default=["results/classical", "results/quantum"])
    p.add_argument("--out", default="results/tables")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_results(args.dirs)

    pooled = compute_pooled_summary(df)
    pooled.to_csv(out_dir / "pooled_summary.csv", index=False)
    print(f"Saved {out_dir / 'pooled_summary.csv'} ({len(pooled)} rows)")

    skills = compute_skill_scores(df)
    if not skills.empty:
        skills.to_csv(out_dir / "skill_scores.csv", index=False)
        print(f"Saved {out_dir / 'skill_scores.csv'} ({len(skills)} rows)")

    recovery = compute_drift_recovery(df)
    if not recovery.empty:
        recovery.to_csv(out_dir / "drift_recovery.csv", index=False)
        print(f"Saved {out_dir / 'drift_recovery.csv'} ({len(recovery)} rows)")

    wilc = compute_wilcoxon_holm(df)
    if not wilc.empty:
        wilc.to_csv(out_dir / "wilcoxon_holm.csv", index=False)
        print(f"Saved {out_dir / 'wilcoxon_holm.csv'} ({len(wilc)} rows)")


if __name__ == "__main__":
    main()