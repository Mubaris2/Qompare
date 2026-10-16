"""Shared sliding-window, test-then-train harness.
Protocol (CONTRACT.md): train on last W samples, test on next H, slide by H."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from . import metrics as M
from .timing import mem_tracker

COLUMNS = ["task", "model", "variant", "seed", "W", "H", "drift", "step", "n_seen",
           "metric_name", "metric_value", "t_encode", "t_fit", "t_predict",
           "n_params", "n_qubits", "n_circuit_evals", "peak_mem_mb", "drift_flag"]
TASKS = {"T1", "T2", "T3"}
METRICS_BY_TASK = {"T1": ["mse", "mae"], "T2": ["auc_pooled", "ap_pooled"],
                   "T3": ["accuracy", "f1_macro", "acc_nochange", "kappa_plus"]}


def run_stream(model, X, y, task, W, H, seed=0, drift="none", drift_start=None,
               pool=500, track_mem=False, max_steps=None, start=None):
    """Return long-format DataFrame (one row per step x metric).
    start: index of the first test block (default W). Use config.STREAM_START so rows before it stay a tuning-only prefix."""
    assert task in TASKS, task
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    N = len(X)
    assert N >= W + H, "stream shorter than W+H"
    np.random.seed(seed)
    if hasattr(model, "set_seed"):
        model.set_seed(seed)

    rows, pool_lab, pool_sc = [], [], []
    t, step = (W if start is None else start), 0
    assert t >= W, "start must be >= W (need a full training window)"
    while t + H <= N:
        if max_steps is not None and step >= max_steps:
            break
        Xtr, ytr = X[t - W:t], y[t - W:t]
        Xte, yte = X[t:t + H], y[t:t + H]
        mem = {}
        with mem_tracker(track_mem, mem):
            t0 = time.perf_counter()
            if task == "T2":                       # semisupervised: normal samples only
                model.fit(Xtr[ytr == 0], None)
            else:
                model.fit(Xtr, ytr)
            wall_fit = time.perf_counter() - t0
            t0 = time.perf_counter()
            out = model.score_anomaly(Xte) if task == "T2" else model.predict(Xte)
            wall_pred = time.perf_counter() - t0

        if task == "T1":
            vals = {"mse": M.mse(yte, out), "mae": M.mae(yte, out)}
        elif task == "T3":
            acc = M.accuracy(yte, out)
            acc_nc = M.nochange_accuracy(y[t - 1:t + H - 1], yte)      # persistence on the same block
            vals = {"accuracy": acc, "f1_macro": M.f1_macro(yte, out),
                    "acc_nochange": acc_nc, "kappa_plus": M.kappa_plus(acc, acc_nc)}
        else:
            pool_lab.extend(yte.tolist()); pool_sc.extend(np.asarray(out).tolist())
            pool_lab, pool_sc = pool_lab[-pool:], pool_sc[-pool:]
            vals = {"auc_pooled": M.pooled_auc(pool_lab, pool_sc),
                    "ap_pooled": M.pooled_ap(pool_lab, pool_sc)}

        tm = getattr(model, "timings", {}) or {}
        t_fit = tm.get("t_fit") if tm.get("t_fit") is not None else wall_fit
        t_pred = tm.get("t_predict") if tm.get("t_predict") is not None else wall_pred
        meta = getattr(model, "meta", {}) or {}
        flag = int(drift_start is not None and t >= drift_start)
        for name, v in vals.items():
            rows.append(dict(task=task, model=model.name, variant=getattr(model, "variant", ""),
                             seed=seed, W=W, H=H, drift=drift, step=step, n_seen=t + H,
                             metric_name=name, metric_value=v,
                             t_encode=tm.get("t_encode", 0.0) or 0.0, t_fit=t_fit, t_predict=t_pred,
                             n_params=meta.get("n_params", 0), n_qubits=meta.get("n_qubits", 0),
                             n_circuit_evals=meta.get("n_circuit_evals", 0),
                             peak_mem_mb=mem.get("peak_mem_mb", np.nan), drift_flag=flag))
        t += H; step += 1
    return pd.DataFrame(rows, columns=COLUMNS)
