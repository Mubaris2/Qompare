import time
import numpy as np
from qiskit.circuit.library import (
    ZFeatureMap, ZZFeatureMap, EfficientSU2, RealAmplitudes, TwoLocal, PauliTwoDesign
)
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_algorithms.optimizers import SPSA, COBYLA, NELDER_MEAD
from qiskit_machine_learning.algorithms import VQR
from common.interface import StreamingModel

class QuantumVQRModel(StreamingModel):
    def __init__(
        self,
        n_qubits: int = 7,
        feature_map_type: str = "Z",
        entanglement: str = "linear",
        ansatz_type: str = "EfficientSU2",
        reps: int = 1,
        optimizer_type: str = "COBYLA",
        maxiter: int = 30,
        max_train_samples: int = 100,
        seed: int = 42,
    ):
        variant = f"{feature_map_type}_{ansatz_type}_r{reps}_{optimizer_type}"
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
        self.seed = seed
        self._warm_weights = None
        self._build_components()

    def _build_components(self):
        t0 = time.perf_counter()
        # Pre-decompose blueprint circuits once so Qiskit doesn't re-decompose 400x per step
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

        if self.optimizer_type == "SPSA":
            self.optimizer = SPSA(maxiter=self.maxiter)
        elif self.optimizer_type == "COBYLA":
            self.optimizer = COBYLA(maxiter=self.maxiter)
        elif self.optimizer_type == "Nelder-Mead":
            self.optimizer = NELDER_MEAD(maxiter=self.maxiter)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

        # Use C++ Qiskit Aer exact statevector estimator (orders of magnitude faster than Python V2 reference)
        self.estimator = AerEstimator(
            run_options={"shots": None, "seed": self.seed},
            transpile_options={"optimization_level": 0},
            approximation=True,
        )

        self.vqr = VQR(
            feature_map=self.feature_map,
            ansatz=self.ansatz,
            optimizer=self.optimizer,
            estimator=self.estimator,
            initial_point=self._warm_weights,
        )
        t_init = time.perf_counter() - t0
        full_circuit = self.feature_map.compose(self.ansatz)
        self.meta = {
            "n_params": self.ansatz.num_parameters,
            "n_qubits": self.n_qubits,
            "circuit_depth": full_circuit.depth(),
            "n_circuit_evals": 0,
        }
        self.timings["t_encode"] = t_init

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        # Subsample window if larger than max_train_samples (per Section 9 Risk Mitigation)
        if self.max_train_samples and len(X) > self.max_train_samples:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(X), size=self.max_train_samples, replace=False)
            X_fit, y_fit = X[idx], y[idx]
        else:
            X_fit, y_fit = X, y

        t0 = time.perf_counter()
        y_scaled = 2.0 * y_fit - 1.0
        self.vqr.fit(X_fit, y_scaled)
        t_total = time.perf_counter() - t0

        fm_depth = max(1, self.feature_map.depth())
        tot_depth = max(1, self.meta["circuit_depth"])
        encode_fraction = fm_depth / tot_depth

        self.timings["t_encode"] = t_total * encode_fraction
        self.timings["t_fit"] = t_total * (1.0 - encode_fraction)

        evals_per_iter = 2 if self.optimizer_type == "SPSA" else 1
        self.meta["n_circuit_evals"] = len(X_fit) * self.maxiter * evals_per_iter
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        y_pred_raw = self.vqr.predict(X).ravel()
        y_pred = np.clip((y_pred_raw + 1.0) / 2.0, 0.0, 1.0)
        self.timings["t_predict"] = time.perf_counter() - t0
        self.meta["n_circuit_evals"] += len(X)
        return y_pred
