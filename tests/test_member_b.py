import numpy as np
import pandas as pd
from quantum.models.vqr_model import QuantumVQRModel
from quantum.models.qsvr_model import QuantumSVRAnomalyModel

def test_vqr_interface_on_mock():
    df = pd.read_parquet("data/processed/T1_mock.parquet")
    X = df[[f"f{i}" for i in range(7)]].values[:30]
    y = df["target"].values[:30]

    model = QuantumVQRModel(n_qubits=7, feature_map_type="Z", ansatz_type="RealAmplitudes", maxiter=3)
    model.fit(X[:20], y[:20])
    preds = model.predict(X[20:])

    assert preds.shape == (10,)
    assert np.all((preds >= 0.0) & (preds <= 1.0))
    assert model.meta["n_qubits"] == 7
    assert model.timings["t_encode"] > 0
    assert model.timings["t_fit"] > 0

def test_qsvr_interface_on_mock():
    df = pd.read_parquet("data/processed/T2_mock.parquet")
    X = df[[f"f" + str(i) for i in range(5)]].values[:40]
    y = df["label"].values[:40]

    model = QuantumSVRAnomalyModel(n_qubits=5, reps=1)
    model.fit(X[:25], y[:25])
    scores = model.score_anomaly(X[25:])

    assert scores.shape == (15,)
    assert np.all(scores >= 0.0)
    assert model.meta["n_circuit_evals"] > 0

def test_e1_incremental_kernel_cache_exact_equality():
    """
    Week 3 Gate 3 Exit Criterion:
    Verify E1 cached kernel matrix and anomaly scores equal full recompute to numerical tolerance,
    and verify hardware circuit evaluations decrease on sliding steps.
    """
    df = pd.read_parquet("data/processed/T2_mock.parquet")
    X = df[[f"f{i}" for i in range(5)]].values
    y = df["label"].values

    W, H = 60, 15
    model_full = QuantumSVRAnomalyModel(n_qubits=5, reps=1, use_cache=False)
    model_cached = QuantumSVRAnomalyModel(n_qubits=5, reps=1, use_cache=True)

    for step in range(4):
        s = step * H
        X_win, y_win = X[s : s + W], y[s : s + W]
        X_test = X[s + W : s + W + H]

        model_full.fit(X_win, y_win)
        model_cached.fit(X_win, y_win)

        K_full = model_full._fidelity_kernel(model_full._train_states, model_full._train_states)
        assert np.allclose(model_cached._cached_K, K_full, atol=1e-10), f"Kernel mismatch at step {step}"

        scores_full = model_full.score_anomaly(X_test)
        scores_cached = model_cached.score_anomaly(X_test)
        assert np.allclose(scores_cached, scores_full, atol=1e-10), f"Prediction mismatch at step {step}"

        if step > 0:
            assert model_cached.meta["n_circuit_evals"] < model_full.meta["n_circuit_evals"]

def test_e2_vqr_warm_start_preserves_weights():
    df = pd.read_parquet("data/processed/T1_mock.parquet")
    X = df[[f"f{i}" for i in range(7)]].values
    y = df["target"].values

    model = QuantumVQRModel(
        n_qubits=7, feature_map_type="Z", ansatz_type="RealAmplitudes",
        maxiter=5, warm_start=True, seed=42
    )
    model.fit(X[:30], y[:30])
    w1 = model._warm_weights.copy()
    assert w1 is not None

    model.fit(X[10:40], y[10:40])
    w2 = model._warm_weights.copy()
    assert w2 is not None
    assert len(w1) == len(w2)
