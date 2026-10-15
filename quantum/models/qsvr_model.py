import time
import numpy as np
import pennylane as qml
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from common.interface import StreamingModel

class QuantumSVRAnomalyModel(StreamingModel):
    def __init__(
        self,
        n_qubits: int = 5,
        reps: int = 2,
        C: float = 1.0,
        epsilon: float = 0.05,
        variant: str = "ZZ_Fidelity",
    ):
        super().__init__(name="QSVR")
        self.variant = variant
        self.n_qubits = n_qubits
        self.reps = reps
        self.C = C
        self.epsilon = epsilon

        try:
            self.dev = qml.device("lightning.qubit", wires=n_qubits)
        except Exception:
            self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev)
        def state_circuit(x):
            for _ in range(self.reps):
                qml.AngleEmbedding(x, wires=range(self.n_qubits), rotation="Z")
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.RZ(2.0 * (np.pi - x[i]) * (np.pi - x[i + 1]), wires=i + 1)
                    qml.CNOT(wires=[i, i + 1])
            return qml.state()

        self._state_circuit = state_circuit
        self.svr = MultiOutputRegressor(SVR(kernel="precomputed", C=self.C, epsilon=self.epsilon))
        self._train_states = None
        self.meta = {
            "n_params": 0,
            "n_qubits": self.n_qubits,
            "circuit_depth": self.reps * (1 + 3 * (self.n_qubits - 1)),
            "n_circuit_evals": 0,
        }

    def _encode_states(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._state_circuit(x) for x in X])

    @staticmethod
    def _fidelity_kernel(states_a: np.ndarray, states_b: np.ndarray) -> np.ndarray:
        overlaps = states_a @ np.conj(states_b).T
        return np.abs(overlaps) ** 2

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        # Train ONLY on normal samples (semi-supervised protocol from Paper 3)
        if y is not None:
            X_norm = X[y == 0]
            if len(X_norm) < 5:
                X_norm = X
        else:
            X_norm = X

        t0 = time.perf_counter()
        self._train_states = self._encode_states(X_norm)
        K_train = self._fidelity_kernel(self._train_states, self._train_states)
        t_enc = time.perf_counter() - t0

        t1 = time.perf_counter()
        # Reconstruct input features X_norm -> X_norm
        self.svr.fit(K_train, X_norm)
        t_fit = time.perf_counter() - t1

        n = len(X_norm)
        self.timings["t_encode"] = t_enc
        self.timings["t_fit"] = t_fit
        # Hardware equivalent circuit evaluations for upper-triangular kernel matrix
        self.meta["n_circuit_evals"] = (n * (n - 1)) // 2
        return self

    def score_anomaly(self, X: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        test_states = self._encode_states(X)
        K_test = self._fidelity_kernel(test_states, self._train_states)
        t_enc_test = time.perf_counter() - t0

        t1 = time.perf_counter()
        X_reconstructed = self.svr.predict(K_test)
        # Reconstruction loss (MSE per instance) as anomaly score
        scores = np.mean((X - X_reconstructed) ** 2, axis=1)
        t_pred = time.perf_counter() - t1

        self.timings["t_encode"] += t_enc_test
        self.timings["t_predict"] = t_pred
        self.meta["n_circuit_evals"] += len(X) * len(self._train_states)
        return scores