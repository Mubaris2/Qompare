"""Usage: python -m common.validate_results results/classical/*.csv results/quantum/*.csv
Exit code 1 on any violation. Run on every PR."""
import sys, glob
import numpy as np
import pandas as pd
from .harness import COLUMNS, TASKS, METRICS_BY_TASK

ALLOWED_METRICS = {m for v in METRICS_BY_TASK.values() for m in v} | {"prequential_acc"}
ALLOWED_DRIFT = {"none", "abrupt", "gradual", "natural"}


def validate(path):
    errs = []
    df = pd.read_csv(path)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        return [f"{path}: missing columns {missing}"]
    if set(df["task"]) - TASKS: errs.append(f"{path}: bad task values {set(df['task']) - TASKS}")
    if set(df["metric_name"]) - ALLOWED_METRICS:
        errs.append(f"{path}: bad metric_name {set(df['metric_name']) - ALLOWED_METRICS}")
    if set(df["drift"]) - ALLOWED_DRIFT: errs.append(f"{path}: bad drift {set(df['drift']) - ALLOWED_DRIFT}")
    for c in ["task", "model", "seed", "W", "H", "step", "n_seen", "metric_name"]:
        if df[c].isna().any(): errs.append(f"{path}: nulls in {c}")
    for c in ["t_encode", "t_fit", "t_predict"]:
        if (pd.to_numeric(df[c], errors="coerce") < 0).any() or df[c].isna().any():
            errs.append(f"{path}: invalid {c}")
    if df.duplicated(["task", "model", "variant", "seed", "W", "H", "drift", "step", "metric_name"]).any():
        errs.append(f"{path}: duplicate rows (same run key and step)")
    if not np.isfinite(pd.to_numeric(df["n_seen"], errors="coerce")).all():
        errs.append(f"{path}: non-numeric n_seen")
    return errs


if __name__ == "__main__":
    files = [f for a in sys.argv[1:] for f in glob.glob(a)]
    if not files:
        print("no files"); sys.exit(0)
    bad = [e for f in files for e in validate(f)]
    print("\n".join(bad) if bad else f"OK: {len(files)} file(s) valid")
    sys.exit(1 if bad else 0)
