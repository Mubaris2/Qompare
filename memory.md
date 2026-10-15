# Project Memory & State Tracker (`memory.md`)

**Project Title:** Quantum vs Classical Approaches for Streaming Data Analytics: An Experimental Study  
**Active Role:** Member B (Quantum Models, Paper 2 & Paper 3 Reproductions, Streaming Quantum Enhancements E1–E3)  
**Active Branch:** `b/checkpoint-1`  
**Current Milestone:** Checkpoint 1 (Day 0 + Week 1 + Week 2 Completed)

---

## 1. Environment & Dependency Decisions
- **Conda Environment:** `qml_stream` (Python 3.11.16 on Linux)
- **Quantum Stack:**
  - `qiskit==1.2.4`, `qiskit-aer==0.15.1`, `qiskit-algorithms==0.3.1`, `qiskit-machine-learning==0.8.2`
  - `pennylane==0.38.0`, `pennylane-lightning==0.38.0`, `autoray==0.6.12` (pinned `autoray==0.6.12` to resolve `NumpyMimic` removal in `autoray>=0.8`)
- **Performance Optimization Applied:**
  - Qiskit's blueprint circuits (`ZFeatureMap`, `ZZFeatureMap`, `EfficientSU2`, `RealAmplitudes`, `TwoLocal`, `PauliTwoDesign`) are pre-decomposed once during initialization (`decompose()`) and evaluated via C++ `AerEstimator(approximation=True, shots=None)`. This reduced per-config VQR training runtime from >10 minutes (Python V2 reference estimator) to **0.67s – 13.27s**.
  - PennyLane QSVR uses `lightning.qubit` statevector caching (`_encode_states`) and computes the exact quantum fidelity kernel matrix via inner products $K(x_i, x_j) = \vert{}\langle \psi(x_i) \vert{} \psi(x_j) \rangle\vert{}^2$, while logging hardware-equivalent circuit execution counts $W(W-1)/2$.

---

## 2. Directory & File Inventory
- `common/CONTRACT.md`: Frozen Day 1 technical contract specifying `StreamingModel` interface, 18-column CSV schema, window sizes $W \in \{100, 200, 400\}$, step sizes $H \in \{20, 50\}$, and seeds.
- `common/interface.py`: Abstract base class `StreamingModel` enforcing `fit()`, `predict()`, `score_anomaly()`, `timings`, and `meta`.
- `common/validate_results.py`: Schema and NaN validator for result CSVs.
- `data/prepare_quantum_data.py`: Leakage-free stream preprocessor (fits `StandardScaler`, `PCA`, and $[0, \pi]$ `MinMaxScaler` strictly on the first window).
  - `data/processed/T1_mock.parquet` (650 rows, 7 features)
  - `data/processed/T2_mock.parquet` (600 rows, 5 features)
  - `data/processed/T1_stream.parquet` (Seoul Bike Sharing Demand UCI ID 560, 8,760 rows, 7 PCA features in $[0, \pi]$, target in $[0, 1]$)
  - `data/processed/T2_stream.parquet` (Breast Cancer Wisconsin AD benchmark, 569 rows, 5 PCA features in $[0, \pi]$, 150 normal prefix for semi-supervised fit)
- `quantum/models/vqr_model.py`: `QuantumVQRModel` wrapping Qiskit `VQR` with decomposition of `t_encode`, `t_fit`, `t_predict`, and circuit execution counters.
- `quantum/models/qsvr_model.py`: `QuantumSVRAnomalyModel` wrapping PennyLane ZZ-style angle/entangling feature map + `MultiOutputRegressor(SVR(kernel='precomputed'))` scoring anomalies by feature reconstruction MSE.
- `quantum/reproductions/reproduce_p2_vqr.py`: Reproduces Herbst et al. (2024) on Seoul Bike data and outputs `results/quantum/T1_VQR.csv`.
- `quantum/reproductions/reproduce_p3_qsvr.py`: Reproduces Tscharke et al. (2023) on T2 anomaly detection data and outputs `results/quantum/T2_QSVR.csv`.
- `tests/test_member_b.py`: Pytest unit tests validating `QuantumVQRModel` and `QuantumSVRAnomalyModel` against mock Parquet data.

---

## 3. Key Empirical Results Achieved (Checkpoint 1)
1. **Task T1 (Paper 2 — Herbst et al., 2024 Reproduction):**
   - Best model: `Z_RealAmplitudes_r2_SPSA` (`MSE = 0.0693`, `MAE = 0.2154`, `t_fit = 9.64s`).
   - Best fast model: `Z_EfficientSU2_r2_COBYLA` (`MSE = 0.0758`, `MAE = 0.2486`, `t_fit = 4.17s`) and `Z_RealAmplitudes_r1_COBYLA` (`MSE = 0.0761`, `MAE = 0.2457`, `t_fit = 2.31s`).
   - Confirmed Paper 2's primary claim: `ZFeatureMap` consistently outperforms `ZZFeatureMap` (`MSE 0.0693–0.1075` vs `0.0930–0.1143`), `SPSA` achieves the lowest error, and `COBYLA` is 3x faster with near-optimal accuracy.
2. **Task T2 (Paper 3 — Tscharke et al., 2023 Reproduction):**
   - `QSVR` achieved `ROC-AUC = 0.9642` and `Average Precision = 0.9661` across `reps={1,2}` and `C={1.0, 10.0}`.
   - Outperformed the classical `SVR-RBF` reconstruction baseline (`ROC-AUC = 0.9539`) by `+0.0103` AUC.
   - Revealed the streaming bottleneck: quantum encoding took `1.33s – 2.12s` vs `0.0057s` classical fit time.

---

## 4. Next Steps for Week 3 (Post-Checkpoint 1)
- [ ] **E1 Incremental Kernel Cache (`quantum/models/qsvr_streaming.py`)**: Cache the $(W-H) \times (W-H)$ submatrix when sliding by $H$, computing only $W \times H$ new kernel entries. Write unit test verifying `np.allclose(K_cached, K_full, atol=1e-10)`.
- [ ] **E2 Warm-Started VQR (`quantum/models/vqr_streaming.py`)**: Pass trained ansatz weights from window $t-1$ as `initial_point` to window $t$ with reduced optimizer iterations (`maxiter=15` vs cold-start `maxiter=40`).
- [ ] **E3 Encoding Overhead Decomposition**: Log exact state-preparation vs optimizer/SVR solve time across $W \in \{100, 200, 400\}$.

---

## 5. Week 3 Completed Deliverables (Streaming Enhancements E1, E2, E3 & Shot-Noise Ablation)
- **Active Branch:** `Sumit-Quantum`
- **Unit Tests (`tests/test_member_b.py`):** 4/4 passed in 2.55s (`test_vqr_interface_on_mock`, `test_qsvr_interface_on_mock`, `test_e1_incremental_kernel_cache_exact_equality`, `test_e2_vqr_warm_start_preserves_weights`).
- **Validated Result Files:**
  - `results/quantum/T2_QSVR_streaming_week3.csv` (`[PASS]`, 78 rows)
  - `results/quantum/T1_VQR_streaming_week3.csv` (`[PASS]`, 120 rows)
- **Empirical Highlights:**
  - **E1 (QSVR Kernel Cache):** Identical Final Pooled ROC-AUC (`0.9626`), **3.52x speedup** in `t_encode` (`0.1958s` -> `0.0557s`), and **34.5% reduction** in total circuit evaluations per step (`6,895` -> `4,518`).
  - **Shot-Noise Ablation (`shots=1024`):** Preserved `0.9626` Final Pooled ROC-AUC under 1024-shot binomial sampling noise.
  - **E2 (VQR Warm Start):** Under equal budget (`maxiter=15`), reduced Mean MSE from `0.0967` to `0.0928`. Under accuracy target (`target_mse=0.068`), achieved `0.0886` Mean MSE vs `0.0905` (`cold_iter35`) while cutting circuit evaluations by **62.0%** (`1,071` vs `2,820`) and step runtime by **2.66x**.
