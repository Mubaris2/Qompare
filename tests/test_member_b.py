import numpy as np
import pandas as pd
from quantum.models.vqr_model import QuantumVQRModel
from quantum.models.qsvr_model import QuantumSVRAnomalyModel
from common.validate_results import validate_csv

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
    assert model.timings["t_fit"] > 0

def test_qsvr_interface_on_mock():
    df = pd.read_parquet("data/processed/T2_mock.parquet")
    X = df[[f"f{i}" for i in range(5)]].values[:40]
    y = df["label"].values[:40]

    model = QuantumSVRAnomalyModel(n_qubits=5, reps=1)
    model.fit(X[:25], y[:25])
    scores = model.score_anomaly(X[25:])

    assert scores.shape == (15,)
    assert np.all(scores >= 0.0)
    assert model.meta["n_circuit_evals"] > 0