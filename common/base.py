"""Model interface both members implement. See CONTRACT.md."""
from __future__ import annotations


class Model:
    """Base class. fit() must retrain from scratch on the given window
    unless the model sets warm_start = True (quantum enhancement E2)."""

    name: str = "model"
    variant: str = ""
    warm_start: bool = False

    def __init__(self):
        # seconds. Models may fill these precisely; harness falls back to wall time.
        self.timings = {"t_encode": 0.0, "t_fit": None, "t_predict": None}
        # hardware-relevant cost proxies (zeros for classical)
        self.meta = {"n_params": 0, "n_qubits": 0, "circuit_depth": 0, "n_circuit_evals": 0}

    def set_seed(self, seed: int) -> None:
        self.seed = seed

    def fit(self, X, y=None):
        raise NotImplementedError

    def predict(self, X):
        """T1 regression values or T3 class labels."""
        raise NotImplementedError

    def score_anomaly(self, X):
        """T2. Higher = more anomalous."""
        raise NotImplementedError
