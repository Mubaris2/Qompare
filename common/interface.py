from abc import ABC, abstractmethod
import numpy as np

class StreamingModel(ABC):
    def __init__(self, name: str):
        self.name = name
        self.timings = {"t_encode": 0.0, "t_fit": 0.0, "t_predict": 0.0}
        self.meta = {
            "n_params": 0,
            "n_qubits": 0,
            "circuit_depth": 0,
            "n_circuit_evals": 0,
        }

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{self.name} does not implement predict().")

    def score_anomaly(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{self.name} does not implement score_anomaly().")

    