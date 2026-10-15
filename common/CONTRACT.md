# Technical Contract (Frozen Day 1)

## 1. Model Interface
Every model in `classical/` and `quantum/` must inherit from `common.interface.StreamingModel`:
- `name`: `str`
- `fit(X: np.ndarray, y: np.ndarray = None) -> self`
- `predict(X: np.ndarray) -> np.ndarray` (Tasks T1, T3)
- `score_anomaly(X: np.ndarray) -> np.ndarray` (Task T2, higher = more anomalous)
- `timings`: `dict` with keys `t_encode`, `t_fit`, `t_predict` (in seconds)
- `meta`: `dict` with keys `n_params`, `n_qubits`, `circuit_depth`, `n_circuit_evals`

## 2. Result CSV Schema
Exact column order required by `common/validate_results.py`:
`task,model,variant,seed,W,H,drift,step,n_seen,metric_name,metric_value,t_encode,t_fit,t_predict,n_params,n_qubits,n_circuit_evals,peak_mem_mb`

## 3. Protocol Constants
- Seeds: `[42, 100, 2024, 7, 99]` (Quantum fallback: `[42, 100, 2024]`)
- Window Grid (W): `[100, 200, 400]`
- Step Size (H): `20` (T1, T3), `50` (T2)
- Feature Scaling: PCA fit on first window only, scaled to `[0, pi]`, T1 target scaled to `[0, 1]`.