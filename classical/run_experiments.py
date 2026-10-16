"""Run the full classical grid through the shared harness (Member A).

  python -m classical.run_experiments --tasks T1 T2 T3                     # full grid (tuned params)
  python -m classical.run_experiments --tasks T1 --models svr_rbf --W 200 --seeds 0 --drifts none --max-steps 20   # smoke
Grid: task x model x W x drift x seed. Output: results/classical/<task>_<model>.csv (+ .meta.json sidecar).
Streaming evaluation starts at config.STREAM_START; rows before it are the tuning prefix."""
import argparse, json, subprocess, time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from common import config as C
from common.drift import inject_drift
from common.harness import run_stream
from . import registry as R
from .data_io import load_task, ROOT, PROC


def _commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def job(a):
    task, name, params, W, drift, seed, max_steps, data_dir = a
    X, y, meta = load_task(task, data_dir)
    Xd, ds = inject_drift(X, drift, C.DRIFT_START_FRAC, C.DRIFT_WIDTH_FRAC, C.DRIFT_MAGNITUDE, seed=seed) if drift != "none" else (X, None)
    label = drift
    if drift == "none" and (meta.get("natural_drift") or meta.get("natural_drift_at") is not None):
        label = "natural"; ds = meta.get("natural_drift_at")
    return run_stream(R.build(task, name, params), Xd, y, task, W=W, H=C.H[task], seed=seed, drift=label,
                      drift_start=ds, start=C.STREAM_START, max_steps=max_steps)


def run(tasks, models=None, Ws=C.W_GRID, seeds=C.SEEDS, drifts=C.DRIFTS, max_steps=None, workers=1, use_defaults=False,
        data_dir=PROC, out_dir=None, best_json=None, resume=False):
    out_dir = Path(out_dir or ROOT / "results" / "classical"); out_dir.mkdir(parents=True, exist_ok=True)
    bj = Path(best_json or ROOT / "configs" / "best_params_classical.json")
    best = json.loads(bj.read_text()) if (bj.exists() and not use_defaults) else {}
    if not use_defaults and not best:
        print("WARNING: no best_params_classical.json, running DEFAULT params. Run `python -m classical.tune` first.")
    written = []
    for task in tasks:
        for name in R.REGISTRY[task]:
            if models and name not in models:
                continue
            f = out_dir / f"{task}_{name}.csv"
            if resume and f.exists():
                print(f"[skip] {f.name}"); continue
            params = best.get(task, {}).get(name, {}).get("params", {})
            jobs = [(task, name, params, W, d, s, max_steps, str(data_dir)) for W in Ws for d in drifts for s in seeds]
            t0 = time.perf_counter()
            if workers > 1:
                with ProcessPoolExecutor(workers) as ex:
                    dfs = list(ex.map(job, jobs))
            else:
                dfs = [job(j) for j in jobs]
            df = pd.concat(dfs, ignore_index=True)
            df.to_csv(f, index=False)
            f.with_suffix(".meta.json").write_text(json.dumps(dict(
                task=task, model=name, params=params, tuned=bool(params), commit=_commit(), n_jobs=len(jobs), Ws=list(Ws),
                seeds=list(seeds), drifts=list(drifts), start=C.STREAM_START, max_steps=max_steps,
                finished=datetime.now(timezone.utc).isoformat(timespec="seconds")), indent=2))
            print(f"[done] {f.name}: {len(df)} rows, {len(jobs)} runs, {time.perf_counter() - t0:.0f}s")
            written.append(f)
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["T1", "T2", "T3"]); ap.add_argument("--models", nargs="+")
    ap.add_argument("--W", nargs="+", type=int, default=C.W_GRID); ap.add_argument("--seeds", nargs="+", type=int, default=C.SEEDS)
    ap.add_argument("--drifts", nargs="+", default=C.DRIFTS, choices=["none", "abrupt", "gradual"])
    ap.add_argument("--max-steps", type=int); ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--defaults", action="store_true", help="ignore tuned params"); ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    run(a.tasks, a.models, a.W, a.seeds, a.drifts, a.max_steps, a.workers, a.defaults, resume=a.resume)
