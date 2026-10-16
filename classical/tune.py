"""Equal-budget random search for classical models (Member A).

Rules (CONTRACT.md): search ONLY on the prefix rows [0, TUNE_ROWS); W = TUNE_W; N_TRIALS random trials per model;
best = best of the random trials (defaults are logged as a reference row, trial = -1, and never selected).
Objective: T1 mean MSE (min), T2 pooled AUC over the whole prefix evaluation (max), T3 mean accuracy (max).

  python -m classical.tune --tasks T1 T2 T3
  python -m classical.tune --tasks T2 --models ae --n-trials 5
"""
import argparse, json, time, zlib
from pathlib import Path
import numpy as np, pandas as pd
from common import config as C
from common.harness import run_stream
from . import registry as R
from .data_io import load_task, ROOT, PROC

DIRECTION = {"T1": "min", "T2": "max", "T3": "max"}


def objective(task, df):
    if df.empty:
        return float("nan")
    if task == "T1":
        return float(df[df.metric_name == "mse"].metric_value.mean())
    if task == "T2":
        return float(df[df.metric_name == "auc_pooled"].metric_value.iloc[-1])   # pooled over entire prefix (pool=huge)
    return float(df[df.metric_name == "accuracy"].metric_value.mean())


def evaluate(task, name, params, X, y, seed=0):
    Xp, yp = X[:C.TUNE_ROWS], y[:C.TUNE_ROWS]
    t0 = time.perf_counter()
    df = run_stream(R.build(task, name, params), Xp, yp, task, W=C.TUNE_W, H=C.H[task], seed=seed, pool=10 ** 9)
    return objective(task, df), time.perf_counter() - t0


def tune(task, models=None, n_trials=C.N_TRIALS, seed=C.TUNE_SEED, data_dir=PROC, out_json=None, trials_csv=None):
    out_json = Path(out_json or ROOT / "configs" / "best_params_classical.json")
    trials_csv = Path(trials_csv or ROOT / "tuning" / f"classical_trials_{task}.csv")
    X, y, _ = load_task(task, data_dir)
    names = [n for n in R.tunable(task) if models is None or n in models]
    best_all = json.loads(out_json.read_text()) if out_json.exists() else {}
    rows = []
    for name in names:
        rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
        better = (lambda a, b: a < b) if DIRECTION[task] == "min" else (lambda a, b: a > b)
        best = None
        d_obj, d_sec = evaluate(task, name, {}, X, y)                       # reference only
        rows.append(dict(task=task, model=name, trial=-1, params="{}", objective=d_obj, direction=DIRECTION[task], seconds=d_sec))
        for i in range(n_trials):
            p = R.sample(task, name, rng)
            obj, sec = evaluate(task, name, p, X, y)
            rows.append(dict(task=task, model=name, trial=i, params=json.dumps(p), objective=obj, direction=DIRECTION[task], seconds=sec))
            if not np.isnan(obj) and (best is None or better(obj, best[0])):
                best = (obj, p)
        if best is None:
            print(f"[{task}/{name}] no valid trial (all NaN). Check that the prefix contains anomalies / both classes."); continue
        best_all.setdefault(task, {})[name] = dict(params=best[1], objective=best[0], default_objective=d_obj, direction=DIRECTION[task],
                                                  n_trials=n_trials, tune_rows=C.TUNE_ROWS, tune_W=C.TUNE_W, seed=seed)
        print(f"[{task}/{name}] best {best[0]:.4f} (default {d_obj:.4f})  {best[1]}")
    trials_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(trials_csv, index=False)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(best_all, indent=2, default=str))
    return best_all


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["T1", "T2", "T3"]); ap.add_argument("--models", nargs="+")
    ap.add_argument("--n-trials", type=int, default=C.N_TRIALS); ap.add_argument("--seed", type=int, default=C.TUNE_SEED)
    a = ap.parse_args()
    for t in a.tasks:
        tune(t, a.models, a.n_trials, a.seed)
