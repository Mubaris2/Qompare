"""Generate the 5 report plots from Classical and Quantum result CSVs.

Usage:
  python -m analysis.plots --dirs results/classical results/quantum --out results/figures
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.tables import load_all_results, PRIMARY_METRIC


def _label(row_or_tuple) -> str:
    model, variant = row_or_tuple
    return f"{model}" if variant in ("", "default", None) else f"{model} [{variant}]"


def plot_metric_vs_seen(df: pd.DataFrame, out_dir: Path, W_target: int = 200, drift_target: str = "none"):
    """Plot 1: Primary metric vs n_seen for each task."""
    for task, pm in PRIMARY_METRIC.items():
        sub = df[(df["task"] == task) & (df["metric_name"] == pm) & (df["W"] == W_target) & (df["drift"] == drift_target)]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for (model, variant), grp in sub.groupby(["model", "variant"]):
            agg = grp.groupby("n_seen")["metric_value"].agg(["mean", "std"]).reset_index()
            lbl = _label((model, variant))
            ax.plot(agg["n_seen"], agg["mean"], label=lbl, linewidth=1.8)
            if agg["std"].notna().any():
                ax.fill_between(agg["n_seen"], agg["mean"] - agg["std"], agg["mean"] + agg["std"], alpha=0.18)
        ax.set_title(f"{task}: {pm} vs Samples Seen (W={W_target}, drift={drift_target})")
        ax.set_xlabel("Samples Seen (n_seen)")
        ax.set_ylabel(pm)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"fig1_metric_vs_seen_{task}.png", dpi=200)
        plt.close(fig)


def plot_latency_vs_W(df: pd.DataFrame, out_dir: Path):
    """Plot 2: Total step wall time (encode + fit + predict) vs window size W."""
    df = df.copy()
    df["step_time_s"] = df["t_encode"].fillna(0) + df["t_fit"].fillna(0) + df["t_predict"].fillna(0)
    for task in PRIMARY_METRIC:
        sub = df[(df["task"] == task) & (df["drift"] == "none")]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for (model, variant), grp in sub.groupby(["model", "variant"]):
            agg = grp.groupby("W")["step_time_s"].agg(["mean", "std"]).reset_index()
            ax.errorbar(agg["W"], agg["mean"], yerr=agg["std"], marker="o", capsize=3, label=_label((model, variant)))
        ax.set_yscale("log")
        ax.set_title(f"{task}: Step Latency vs Window Size W (log scale)")
        ax.set_xlabel("Window Size W")
        ax.set_ylabel("Mean Step Latency (s)")
        ax.set_xticks(sorted(sub["W"].unique()))
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"fig2_latency_vs_W_{task}.png", dpi=200)
        plt.close(fig)


def plot_around_drift(df: pd.DataFrame, out_dir: Path, W_target: int = 200, window_steps: int = 15):
    """Plot 3: Metric trajectory zoomed around abrupt drift onset."""
    for task, pm in PRIMARY_METRIC.items():
        sub = df[(df["task"] == task) & (df["metric_name"] == pm) & (df["W"] == W_target) & (df["drift"] == "abrupt")]
        if sub.empty:
            continue
        flagged = sub[sub["drift_flag"] == 1]
        if flagged.empty:
            continue
        drift_step = int(flagged["step"].min())
        zoomed = sub[(sub["step"] >= drift_step - window_steps) & (sub["step"] <= drift_step + window_steps)]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for (model, variant), grp in zoomed.groupby(["model", "variant"]):
            agg = grp.groupby("step")["metric_value"].mean().reset_index()
            ax.plot(agg["step"] - drift_step, agg["metric_value"], marker=".", label=_label((model, variant)))
        ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="Drift Onset")
        ax.set_title(f"{task}: Recovery Around Abrupt Drift (W={W_target})")
        ax.set_xlabel("Steps Relative to Drift Onset")
        ax.set_ylabel(pm)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"fig3_around_drift_{task}.png", dpi=200)
        plt.close(fig)


def plot_encode_share(df: pd.DataFrame, out_dir: Path, W_target: int = 200):
    """Plot 4: Stacked bar chart of t_encode, t_fit, and t_predict share (E3 breakdown)."""
    sub = df[(df["W"] == W_target) & (df["drift"] == "none")]
    if sub.empty:
        return
    agg = (
        sub.groupby(["task", "model", "variant"], as_index=False)[["t_encode", "t_fit", "t_predict"]]
        .mean()
    )
    totals = agg["t_encode"] + agg["t_fit"] + agg["t_predict"]
    agg = agg[totals > 0].copy()
    if agg.empty:
        return

    labels = [f"{r.task}:{_label((r.model, r.variant))}" for _, r in agg.iterrows()]
    x = np.arange(len(labels))
    t_enc = agg["t_encode"].to_numpy()
    t_fit = agg["t_fit"].to_numpy()
    t_prd = agg["t_predict"].to_numpy()
    tot = np.maximum(t_enc + t_fit + t_prd, 1e-12)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x, 100.0 * t_enc / tot, label="Encode (t_encode)")
    ax.bar(x, 100.0 * t_fit / tot, bottom=100.0 * t_enc / tot, label="Fit (t_fit)")
    ax.bar(x, 100.0 * t_prd / tot, bottom=100.0 * (t_enc + t_fit) / tot, label="Predict (t_predict)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Share of Step Time (%)")
    ax.set_title(f"Encode vs Fit vs Predict Time Share (W={W_target})")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_encode_share.png", dpi=200)
    plt.close(fig)


def plot_circuit_evals(df: pd.DataFrame, out_dir: Path):
    """Plot 5: Quantum circuit evaluations across W (with vs without E1 kernel cache)."""
    qdf = df[(df["n_circuit_evals"].fillna(0) > 0) & (df["drift"] == "none")]
    if qdf.empty:
        return
    agg = qdf.groupby(["task", "model", "variant", "W"], as_index=False)["n_circuit_evals"].mean()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (task, model, variant), grp in agg.groupby(["task", "model", "variant"]):
        ax.plot(grp["W"], grp["n_circuit_evals"], marker="s", label=f"{task}:{_label((model, variant))}")
    ax.set_title("Mean Circuit Executions per Step vs Window Size W")
    ax.set_xlabel("Window Size W")
    ax.set_ylabel("Circuit Executions (n_circuit_evals)")
    ax.set_xticks(sorted(qdf["W"].unique()))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_circuit_evals_cache.png", dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+", default=["results/classical", "results/quantum"])
    p.add_argument("--out", default="results/figures")
    p.add_argument("--W", type=int, default=200)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_results(args.dirs)
    plot_metric_vs_seen(df, out_dir, W_target=args.W)
    plot_latency_vs_W(df, out_dir)
    plot_around_drift(df, out_dir, W_target=args.W)
    plot_encode_share(df, out_dir, W_target=args.W)
    plot_circuit_evals(df, out_dir)
    print(f"All available plots saved to {out_dir}")


if __name__ == "__main__":
    main()