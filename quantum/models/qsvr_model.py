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
        reps: int = 1,
        C: float = 1.0,
        epsilon: float = 0.05,
        use_cache: bool = False,
        shots: int | None = None,
        seed: int = 42,
        variant: str | None = None,
    ):
        if variant is None:
            cache_tag = "cached" if use_cache else "full"
            shot_tag = f"shots{shots}" if shots is not None else "exact"
            variant = f"ZZ_r{reps}_{cache_tag}_{shot_tag}"
        super().__init__(name="QSVR")
        self.variant = variant
        self.n_qubits = n_qubits
        self.reps = reps
        self.C = C
        self.epsilon = epsilon
        self.use_cache = use_cache
        self.shots = shots
        self.seed = seed
        self._rng = np.random.default_rng(seed)

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

        # E1 Cache state
        self._cached_X = None
        self._cached_states = None
        self._cached_K = None

        self.meta = {
            "n_params": 0,
            "n_qubits": self.n_qubits,
            "circuit_depth": self.reps * (1 + 3 * (self.n_qubits - 1)),
            "n_circuit_evals": 0,
        }

    def reset_cache(self):
        self._cached_X = None
        self._cached_states = None
        self._cached_K = None

    def _encode_states(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.empty((0, 2 ** self.n_qubits), dtype=complex)
        return np.array([self._state_circuit(x) for x in X])

    def _fidelity_kernel(self, states_a: np.ndarray, states_b: np.ndarray, symmetric: bool = False) -> np.ndarray:
        overlaps = states_a @ np.conj(states_b).T
        K = np.clip(np.abs(overlaps) ** 2, 0.0, 1.0)
        if self.shots is not None:
            if symmetric:
                # Sample upper triangle with binomial shot noise and mirror
                upper_idx = np.triu_indices_from(K, k=1)
                sampled = self._rng.binomial(self.shots, K[upper_idx]) / float(self.shots)
                K_noisy = np.eye(len(K), dtype=float)
                K_noisy[upper_idx] = sampled
                K_noisy[(upper_idx[1], upper_idx[0])] = sampled
                return K_noisy
            else:
                return self._rng.binomial(self.shots, K) / float(self.shots)
        return K

    @staticmethod
    def _find_overlap_count(prev_X: np.ndarray, curr_X: np.ndarray) -> int:
        """Finds k such that the last k rows of prev_X equal the first k rows of curr_X."""
        max_k = min(len(prev_X), len(curr_X))
        for k in range(max_k, 0, -1):
            if np.allclose(prev_X[-k:], curr_X[:k], atol=1e-12):
                return k
        return 0

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        # Semi-supervised AD: train only on normal samples (y == 0)
        if y is not None:
            X_norm = X[y == 0]
            if len(X_norm) < 5:
                X_norm = X
        else:
            X_norm = X

        t0 = time.perf_counter()
        n = len(X_norm)

        if self.use_cache and self._cached_X is not None:
            k_keep = self._find_overlap_count(self._cached_X, X_norm)
        else:
            k_keep = 0

        if k_keep > 0:
            # E1 Incremental Kernel Cache Update: reuse (k_keep x k_keep) submatrix
            n_new = n - k_keep
            X_new = X_norm[k_keep:]
            kept_states = self._cached_states[-k_keep:]
            new_states = self._encode_states(X_new)
            all_states = np.vstack([kept_states, new_states]) if n_new > 0 else kept_states

            K_train = np.empty((n, n), dtype=float)
            K_train[:k_keep, :k_keep] = self._cached_K[-k_keep:, -k_keep:]

            if n_new > 0:
                K_cross = self._fidelity_kernel(kept_states, new_states, symmetric=False)
                K_new_diag = self._fidelity_kernel(new_states, new_states, symmetric=True)
                K_train[:k_keep, k_keep:] = K_cross
                K_train[k_keep:, :k_keep] = K_cross.T
                K_train[k_keep:, k_keep:] = K_new_diag

            # Hardware circuit evaluations required for only the new entries:
            # k_keep * n_new cross pairs + n_new * (n_new - 1) // 2 new self pairs
            hw_evals = (k_keep * n_new) + (n_new * (n_new - 1)) // 2
        else:
            # Full kernel computation from scratch
            all_states = self._encode_states(X_norm)
            K_train = self._fidelity_kernel(all_states, all_states, symmetric=True)
            hw_evals = (n * (n - 1)) // 2

        if self.use_cache:
            self._cached_X = X_norm.copy()
            self._cached_states = all_states
            self._cached_K = K_train

        self._train_states = all_states
        t_enc = time.perf_counter() - t0

        t1 = time.perf_counter()
        self.svr.fit(K_train, X_norm)
        t_fit = time.perf_counter() - t1

        self.timings["t_encode"] = t_enc
        self.timings["t_fit"] = t_fit
        self.meta["n_circuit_evals"] = hw_evals
        return self

    def score_anomaly(self, X: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        test_states = self._encode_states(X)
        K_test = self._fidelity_kernel(test_states, self._train_states, symmetric=False)
        t_enc_test = time.perf_counter() - t0

        t1 = time.perf_counter()
        X_reconstructed = self.svr.predict(K_test)
        scores = np.mean((X - X_reconstructed) ** 2, axis=1)
        t_pred = time.perf_counter() - t1

        # E3: Keep fit encoding separate and track inference encoding + prediction
        self.timings["t_predict"] = t_enc_test + t_pred
        self.meta["n_circuit_evals"] += len(X) * len(self._train_states)
        return scores
