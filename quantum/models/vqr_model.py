import time
import numpy as np
from scipy.optimize import minimize
from qiskit.circuit.library import (
    ZFeatureMap, ZZFeatureMap, EfficientSU2, RealAmplitudes, TwoLocal, PauliTwoDesign
)
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from common.interface import StreamingModel

class QuantumVQRModel(StreamingModel):
    def __init__(
        self,
        n_qubits: int = 7,
        feature_map_type: str = "Z",
        entanglement: str = "linear",
        ansatz_type: str = "RealAmplitudes",
        reps: int = 1,
        optimizer_type: str = "COBYLA",
        maxiter: int = 30,
        max_train_samples: int = 100,
        warm_start: bool = False,
        target_mse: float | None = None,
        seed: int = 42,
        variant: str | None = None,
    ):
        if variant is None:
            ws_tag = "warm" if warm_start else "cold"
            variant = f"{feature_map_type}_{ansatz_type}_r{reps}_{optimizer_type}_{ws_tag}"
        super().__init__(name="VQR")
        self.variant = variant
        self.n_qubits = n_qubits
        self.feature_map_type = feature_map_type
        self.entanglement = entanglement
        self.ansatz_type = ansatz_type
        self.reps = reps
        self.optimizer_type = optimizer_type
        self.maxiter = maxiter
        self.max_train_samples = max_train_samples
        self.warm_start = warm_start
        self.target_mse = target_mse
        self.seed = seed
        self._warm_weights = None
        self._build_components()

    def _build_components(self):
        if self.feature_map_type == "Z":
            fm = ZFeatureMap(feature_dimension=self.n_qubits, reps=1)
        elif self.feature_map_type == "ZZ":
            fm = ZZFeatureMap(
                feature_dimension=self.n_qubits, reps=1, entanglement=self.entanglement
            )
        else:
            raise ValueError(f"Unknown feature map: {self.feature_map_type}")
        self.feature_map = fm.decompose()

        if self.ansatz_type == "EfficientSU2":
            ans = EfficientSU2(num_qubits=self.n_qubits, reps=self.reps, entanglement=self.entanglement)
        elif self.ansatz_type == "RealAmplitudes":
            ans = RealAmplitudes(num_qubits=self.n_qubits, reps=self.reps, entanglement=self.entanglement)
        elif self.ansatz_type == "TwoLocal":
            ans = TwoLocal(
                num_qubits=self.n_qubits, rotation_blocks=["ry", "rz"],
                entanglement_blocks="cz", entanglement=self.entanglement, reps=self.reps
            )
        elif self.ansatz_type == "PauliTwoDesign":
            ans = PauliTwoDesign(num_qubits=self.n_qubits, reps=self.reps)
        else:
            raise ValueError(f"Unknown ansatz: {self.ansatz_type}")
        self.ansatz = ans.decompose()

        self.full_circuit = self.feature_map.compose(self.ansatz)
        self.estimator = AerEstimator(
            run_options={"shots": None, "seed": self.seed},
            transpile_options={"optimization_level": 0},
            approximation=True,
        )
        obs = SparsePauliOp.from_list([("Z" * self.n_qubits, 1.0)])
        self.qnn = EstimatorQNN(
            circuit=self.full_circuit,
            observables=obs,
            input_params=list(self.feature_map.parameters),
            weight_params=list(self.ansatz.parameters),
            estimator=self.estimator,
        )

        self._fm_depth_ratio = max(1, self.feature_map.depth()) / max(1, self.full_circuit.depth())
        self.meta = {
            "n_params": self.ansatz.num_parameters,
            "n_qubits": self.n_qubits,
            "circuit_depth": self.full_circuit.depth(),
            "n_circuit_evals": 0,
        }

    def reset_weights(self):
        self._warm_weights = None

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        rng = np.random.default_rng(self.seed)
        if self.max_train_samples and len(X) > self.max_train_samples:
            idx = rng.choice(len(X), size=self.max_train_samples, replace=False)
            X_fit, y_fit = X[idx], y[idx]
        else:
            X_fit, y_fit = X, y

        if self.warm_start and self._warm_weights is not None:
            init_weights = self._warm_weights.copy()
        else:
            init_weights = rng.uniform(-np.pi, np.pi, size=self.meta["n_params"])

        best_weights = init_weights.copy()
        best_loss = float("inf")
        n_evals = 0
        qnn_time_accum = 0.0
        target_reached = False

        def objective(w):
            nonlocal best_weights, best_loss, n_evals, qnn_time_accum, target_reached
            # Short-circuit without running QNN once target_mse is reached (avoids Fortran callback error)
            if target_reached:
                return best_loss

            t_q0 = time.perf_counter()
            preds_raw = self.qnn.forward(X_fit, w).ravel()
            qnn_time_accum += time.perf_counter() - t_q0
            n_evals += 1

            preds_01 = (preds_raw + 1.0) / 2.0
            mse = float(np.mean((y_fit - preds_01) ** 2))
            if mse < best_loss:
                best_loss = mse
                best_weights = w.copy()
            if self.target_mse is not None and mse <= self.target_mse:
                target_reached = True
            return mse

        t0 = time.perf_counter()
        if self.optimizer_type == "SPSA":
            w = init_weights.copy()
            a, c, A = 0.15, 0.1, 0.1 * self.maxiter
            for k in range(1, self.maxiter + 1):
                if target_reached:
                    break
                ak = a / ((k + A) ** 0.602)
                ck = c / (k ** 0.101)
                delta = rng.choice([-1.0, 1.0], size=len(w))
                loss_plus = objective(w + ck * delta)
                loss_minus = objective(w - ck * delta)
                grad = (loss_plus - loss_minus) / (2.0 * ck * delta)
                w = w - ak * grad
        else:
            method = "COBYLA" if self.optimizer_type == "COBYLA" else "Nelder-Mead"
            minimize(objective, init_weights, method=method, options={"maxiter": self.maxiter})

        t_total = time.perf_counter() - t0
        self._warm_weights = best_weights.copy()

        # E3: Exact separation of feature encoding share vs variational ansatz + classical optimizer
        t_enc = qnn_time_accum * self._fm_depth_ratio
        t_fit = max(1e-6, t_total - t_enc)

        self.timings["t_encode"] = t_enc
        self.timings["t_fit"] = t_fit
        self.meta["n_circuit_evals"] = n_evals * len(X_fit)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        y_pred_raw = self.qnn.forward(X, self._warm_weights).ravel()
        y_pred = np.clip((y_pred_raw + 1.0) / 2.0, 0.0, 1.0)
        self.timings["t_predict"] = time.perf_counter() - t0
        self.meta["n_circuit_evals"] += len(X)
        return y_pred
