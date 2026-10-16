"""Shared sliding-window, test-then-train harness.
Protocol (CONTRACT.md): train on last W samples, test on next H, slide by H."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .stream_runner import StreamRunner, COLUMNS, TASKS, METRICS_BY_TASK

__all__ = ["COLUMNS", "TASKS", "METRICS_BY_TASK", "run_stream"]


def run_stream(model, X, y, task, W, H, seed=0, drift="none", drift_start=None,
               pool=500, track_mem=False, max_steps=None, start=None):
    """Return long-format DataFrame (one row per step x metric)."""
    assert task in TASKS, task
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    N = len(X)
    t = W if start is None else int(start)
    assert t >= W, f"start ({t}) must be >= W ({W})"
    assert N >= t + H, "stream shorter than start+H"

    runner = StreamRunner(
        model=model, task=task, W=W, H=H, seed=seed,
        drift=drift, drift_start=drift_start, pool=pool, track_mem=track_mem
    )

    rows = []
    step = 0
    while t + H <= N:
        if max_steps is not None and step >= max_steps:
            break
        Xtr, ytr = X[t - W:t], y[t - W:t]
        Xte, yte = X[t:t + H], y[t:t + H]
        rows.extend(runner.step_block(Xtr, ytr, Xte, yte, y[t - 1], t))
        t += H
        step += 1

    return pd.DataFrame(rows, columns=COLUMNS)