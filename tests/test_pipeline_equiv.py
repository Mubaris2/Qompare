"""Gate tests for Section 3:
Gate 1: FrozenTransform (transform.json) matches data/prep.py fit_scale within 1e-6.
Gate 2: StreamRunner micro-batching matches offline harness.run_stream within 1e-6.
"""
from __future__ import annotations
import numpy as np
import pytest
from common.harness import run_stream
from common.stream_runner import StreamRunner
from common.transform import FrozenTransform
from data.prep import fit_scale


class DummyRegressor:
    name = "dummy_reg"
    variant = "default"

    def set_seed(self, s):
        pass

    def fit(self, X, y):
        self.coef_ = np.linalg.pinv(X) @ y

    def predict(self, X):
        return X @ self.coef_


class DummyAnomaly:
    name = "dummy_anom"
    variant = "default"

    def set_seed(self, s):
        pass

    def fit(self, X, y=None):
        self.center_ = np.mean(X, axis=0)

    def score_anomaly(self, X):
        return np.sum((X - self.center_) ** 2, axis=1)


class DummyClassifier:
    name = "dummy_clf"
    variant = "default"

    def set_seed(self, s):
        pass

    def fit(self, X, y):
        self.mean0_ = np.mean(X[y == 0], axis=0) if np.any(y == 0) else np.zeros(X.shape[1])
        self.mean1_ = np.mean(X[y == 1], axis=0) if np.any(y == 1) else np.ones(X.shape[1])

    def predict(self, X):
        d0 = np.sum((X - self.mean0_) ** 2, axis=1)
        d1 = np.sum((X - self.mean1_) ** 2, axis=1)
        return (d1 < d0).astype(int)


@pytest.mark.parametrize("n_components,margin,use_mask", [(7, 0.35, False), (5, 0.0, True), (4, 0.0, False)])
def test_gate1_frozen_transform_parity(n_components, margin, use_mask):
    """Gate 1: FrozenTransform rebuilt features equal data/prep.py features within 1e-6."""
    rng = np.random.default_rng(123)
    X_raw = rng.normal(loc=5.0, scale=2.5, size=(1000, 10))
    mask = (rng.uniform(size=1000) > 0.1) if use_mask else None

    F_prep, _, tf_spec = fit_scale(X_raw, fit_rows=800, n_components=n_components, fit_mask=mask, margin=margin)
    ft = FrozenTransform(tf_spec)
    F_spark = ft.transform_features(X_raw)

    max_diff = float(np.max(np.abs(F_prep - F_spark)))
    assert max_diff < 1e-6, f"Gate 1 failed: max diff {max_diff} >= 1e-6"


@pytest.mark.parametrize(
    "task,model_cls,W,H",
    [
        ("T1", DummyRegressor, 100, 20),
        ("T2", DummyAnomaly, 100, 50),
        ("T3", DummyClassifier, 100, 20),
    ],
)
def test_gate2_stream_runner_equivalence(task, model_cls, W, H):
    """Gate 2: Live micro-batch StreamRunner metrics equal offline run_stream metrics within 1e-6."""
    rng = np.random.default_rng(42)
    N = 1200
    start = 800
    X = rng.normal(size=(N, 5))
    if task == "T1":
        y = X @ np.array([0.5, -0.2, 0.1, 0.0, 0.3]) + rng.normal(0, 0.05, size=N)
    elif task == "T2":
        y = (rng.uniform(size=N) < 0.08).astype(int)
    else:
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

    df_offline = run_stream(
        model=model_cls(), X=X, y=y, task=task, W=W, H=H, seed=7, start=start
    )

    runner = StreamRunner(model=model_cls(), task=task, W=W, H=H, seed=7)
    runner.seed_window(X[start - W : start], y[start - W : start], n_seen_init=start)

    live_rows = []
    t = start
    while t + H <= N:
        live_rows.extend(runner.ingest_batch(X[t : t + H], y[t : t + H]))
        t += H

    assert len(df_offline) == len(live_rows)
    for r_off, r_live in zip(df_offline.to_dict("records"), live_rows):
        assert r_off["step"] == r_live["step"]
        assert r_off["n_seen"] == r_live["n_seen"]
        assert r_off["metric_name"] == r_live["metric_name"]
        assert np.isclose(r_off["metric_value"], r_live["metric_value"], atol=1e-6, equal_nan=True)