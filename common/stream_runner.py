"""Stateful sliding-window runner shared by offline harness and live Spark/Kafka pipeline."""
from __future__ import annotations
import time
import numpy as np
from . import metrics as M
from .timing import mem_tracker

COLUMNS = ["task", "model", "variant", "seed", "W", "H", "drift", "step", "n_seen",
           "metric_name", "metric_value", "t_encode", "t_fit", "t_predict",
           "n_params", "n_qubits", "n_circuit_evals", "peak_mem_mb", "drift_flag"]
TASKS = {"T1", "T2", "T3"}
METRICS_BY_TASK = {"T1": ["mse", "mae"], "T2": ["auc_pooled", "ap_pooled"],
                   "T3": ["accuracy", "f1_macro", "acc_nochange", "kappa_plus"]}


class StreamRunner:
    """Maintains sliding window buffer and evaluation pools across micro-batches."""

    def __init__(self, model, task: str, W: int, H: int, seed: int = 0,
                 drift: str = "none", drift_start: int | None = None,
                 pool: int = 500, track_mem: bool = False):
        assert task in TASKS, f"Unknown task: {task}"
        self.model = model
        self.task = task
        self.W = W
        self.H = H
        self.seed = seed
        self.drift = drift
        self.drift_start = drift_start
        self.pool = pool
        self.track_mem = track_mem

        np.random.seed(seed)
        if hasattr(self.model, "set_seed"):
            self.model.set_seed(seed)

        self.step = 0
        self.pool_lab: list = []
        self.pool_sc: list = []
        self._buf_X: np.ndarray | None = None
        self._buf_y: np.ndarray | None = None
        self.n_seen = 0

    def seed_window(self, X_init: np.ndarray, y_init: np.ndarray, n_seen_init: int | None = None):
        """Pre-load the initial training window (e.g. last W rows before stream start)."""
        X_init = np.asarray(X_init, dtype=float)
        y_init = np.asarray(y_init)
        assert len(X_init) >= self.W, f"Need at least W={self.W} rows to seed, got {len(X_init)}"
        self._buf_X = X_init[-self.W:].copy()
        self._buf_y = y_init[-self.W:].copy()
        self.n_seen = n_seen_init if n_seen_init is not None else len(X_init)

    def step_block(self, Xtr: np.ndarray, ytr: np.ndarray,
                   Xte: np.ndarray, yte: np.ndarray,
                   y_prev_last, t_pos: int) -> list[dict]:
        """Run one fit-then-predict step on explicit train/test arrays and return long-format row dicts."""
        mem = {}
        with mem_tracker(self.track_mem, mem):
            t0 = time.perf_counter()
            if self.task == "T2":
                self.model.fit(Xtr[ytr == 0], None)
            else:
                self.model.fit(Xtr, ytr)
            wall_fit = time.perf_counter() - t0

            t0 = time.perf_counter()
            out = self.model.score_anomaly(Xte) if self.task == "T2" else self.model.predict(Xte)
            wall_pred = time.perf_counter() - t0

        if self.task == "T1":
            vals = {"mse": M.mse(yte, out), "mae": M.mae(yte, out)}
        elif self.task == "T3":
            acc = M.accuracy(yte, out)
            y_shifted = np.concatenate([[y_prev_last], yte[:-1]])
            acc_nc = M.nochange_accuracy(y_shifted, yte)
            vals = {"accuracy": acc, "f1_macro": M.f1_macro(yte, out),
                    "acc_nochange": acc_nc, "kappa_plus": M.kappa_plus(acc, acc_nc)}
        else:
            self.pool_lab.extend(yte.tolist())
            self.pool_sc.extend(np.asarray(out).tolist())
            self.pool_lab = self.pool_lab[-self.pool:]
            self.pool_sc = self.pool_sc[-self.pool:]
            vals = {"auc_pooled": M.pooled_auc(self.pool_lab, self.pool_sc),
                    "ap_pooled": M.pooled_ap(self.pool_lab, self.pool_sc)}

        tm = getattr(self.model, "timings", {}) or {}
        t_fit = tm.get("t_fit") if tm.get("t_fit") is not None else wall_fit
        t_pred = tm.get("t_predict") if tm.get("t_predict") is not None else wall_pred
        meta = getattr(self.model, "meta", {}) or {}
        flag = int(self.drift_start is not None and t_pos >= self.drift_start)

        step_rows = []
        for name, v in vals.items():
            step_rows.append(dict(
                task=self.task, model=self.model.name, variant=getattr(self.model, "variant", ""),
                seed=self.seed, W=self.W, H=self.H, drift=self.drift, step=self.step,
                n_seen=t_pos + len(Xte), metric_name=name, metric_value=v,
                t_encode=tm.get("t_encode", 0.0) or 0.0, t_fit=t_fit, t_predict=t_pred,
                n_params=meta.get("n_params", 0), n_qubits=meta.get("n_qubits", 0),
                n_circuit_evals=meta.get("n_circuit_evals", 0),
                peak_mem_mb=mem.get("peak_mem_mb", np.nan), drift_flag=flag
            ))
        self.step += 1
        return step_rows

    def ingest_batch(self, X_batch: np.ndarray, y_batch: np.ndarray) -> list[dict]:
        """Append incoming batch (size H) to buffer; if buffer had W rows, run step_block and slide."""
        X_batch = np.asarray(X_batch, dtype=float)
        y_batch = np.asarray(y_batch)
        if self._buf_X is None:
            self._buf_X = X_batch.copy()
            self._buf_y = y_batch.copy()
            self.n_seen += len(X_batch)
            return []

        if len(self._buf_X) < self.W:
            self._buf_X = np.vstack([self._buf_X, X_batch])[-self.W:]
            self._buf_y = np.concatenate([self._buf_y, y_batch])[-self.W:]
            self.n_seen += len(X_batch)
            return []

        Xtr, ytr = self._buf_X[-self.W:], self._buf_y[-self.W:]
        t_pos = self.n_seen
        rows = self.step_block(Xtr, ytr, X_batch, y_batch, ytr[-1], t_pos)

        self._buf_X = np.vstack([self._buf_X, X_batch])[-self.W:]
        self._buf_y = np.concatenate([self._buf_y, y_batch])[-self.W:]
        self.n_seen += len(X_batch)
        return rows