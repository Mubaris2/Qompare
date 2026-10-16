# Final Report Sections — Member B (Quantum Methods, Enhancements, NISQ Limits, Threats to Validity & Quantum Discussion)

**Authors (Team):** Ashwin S (2023BCS0044), Adnan Omar Chozhimadam (2023BCS0035), Mohamed Sheik Mubaris (2023BCS0062), Hulke Sumit (2023BCD0026)  
**Section Owner:** Member B

---

## 1. Quantum Machine Learning Methods (Tasks T1 & T2)

### 1.1 Variational Quantum Regression (VQR — Base Paper 2: Herbst et al., 2024)
For streaming regression on the **Seoul Bike Sharing Demand** dataset (Task T1, $n=7$ principal components scaled to $[0, \pi]$, target normalized to $[0, 1]$), we implement a parameterized quantum circuit (PQC) architecture $\mathcal{U}(x, \theta) = V(\theta) U_{\Phi}(x)$ acting on an $n$-qubit register initialized to $\vert{}0\rangle^{\otimes n}$:
1. **Quantum Feature Map $U_{\Phi}(x)$:** Maps classical streaming feature vectors $x \in [0, \pi]^n$ into Hilbert space $\mathcal{H} = (\mathbb{C}^2)^{\otimes n}$. We compare two encodings from Herbst et al. (2024):
   - **`ZFeatureMap` (First-Order Pauli-Z Evolution):** Applies Hadamard gates followed by single-qubit phase rotations without two-qubit entanglement:
     $$U_{\Phi_Z}(x) = \bigotimes_{j=1}^{n} R_Z(2x_j) H$$
   - **`ZZFeatureMap` (Second-Order Ising Interaction):** Interleaves single-qubit rotations with nearest-neighbor two-qubit $\text{CNOT}-R_Z-\text{CNOT}$ entangling gates encoding pairwise products $(\pi - x_j)(\pi - x_{j+1})$.
2. **Variational Ansatz $V(\theta)$:** A trainable unitary circuit parameterized by $\theta \in \mathbb{R}^p$ with repetition depth $r \in \{1, 2\}$. We evaluate `RealAmplitudes` ($R_Y$ rotations + linear $\text{CNOT}$ entanglement), `EfficientSU2` ($R_Y, R_Z$ rotations + linear $\text{CNOT}$), and `TwoLocal` ($R_Y, R_Z$ + linear $\text{CZ}$).
3. **Readout & Classical Optimization:** The predicted regression value is extracted from the global Pauli-Z parity expectation value $\hat{y}(x;\theta) = \frac{1}{2}\left(\langle 0^{\otimes n} \vert{} \mathcal{U}^\dagger(x,\theta) Z^{\otimes n} \mathcal{U}(x,\theta) \vert{} 0^{\otimes n} \rangle + 1\right) \in [0, 1]$, optimized via derivative-free `COBYLA` or stochastic gradient-approximation `SPSA`.

### 1.2 Quantum-Kernel Support Vector Regression for Anomaly Detection (QSVR — Base Paper 3: Tscharke et al., 2023)
For semi-supervised streaming anomaly detection (Task T2, $n=5$ principal components in $[0, \pi]$), we implement a multi-output Support Vector Regressor with a quantum fidelity kernel trained exclusively on normal window instances ($y = 0$):
1. **Data-Reuploading Quantum Embedding:** Each input vector $x \in [0, \pi]^n$ prepares the quantum state $\vert{}\psi(x)\rangle = U_{\text{ZZ}}(x)\vert{}0\rangle^{\otimes n}$, where each layer $l \in \{1, \dots, L\}$ applies $H^{\otimes n}$, single-qubit $R_Z(x_j)$ rotations, and linear nearest-neighbor $\text{CNOT}-R_Z(2(\pi-x_j)(\pi-x_{j+1}))-\text{CNOT}$ gates.
2. **Fidelity Kernel Matrix:** Pairwise similarity between streaming points $x_i, x_j$ is defined by the state overlap fidelity:
   $$K(x_i, x_j) = \vert{}\langle \psi(x_i) \vert{} \psi(x_j) \rangle\vert{}^2 = \vert{}\langle 0^{\otimes n} \vert{} U_{\text{ZZ}}^\dagger(x_i) U_{\text{ZZ}}(x_j) \vert{} 0^{\otimes n} \rangle\vert{}^2$$
3. **Reconstruction-Based Anomaly Scoring:** A precomputed-kernel SVR learns the identity mapping $f: X_{\text{normal}} \to X_{\text{normal}}$ on the current window's normal instances. Incoming test stream samples $x_{\text{test}}$ are scored by their mean squared feature reconstruction loss:
   $$s(x_{\text{test}}) = \frac{1}{n} \Vert{} x_{\text{test}} - \hat{f}(x_{\text{test}}) \Vert{}_2^2$$
   where higher reconstruction loss indicates higher anomaly likelihood.

---

## 2. Streaming Quantum Enhancements (E1, E2, E3)

Both quantum base papers (Herbst et al., 2024; Tscharke et al., 2023) evaluate models strictly in offline batch mode. To adapt QML to sliding-window data streams (window size $W$, step size $H$), we introduce three algorithmic and diagnostic enhancements:

### 2.1 Enhancement E1: Incremental Quantum Kernel Cache for QSVR
In standard sliding-window QSVR, advancing the window by $H$ samples requires recomputing the entire $W_{\text{norm}} \times W_{\text{norm}}$ symmetric kernel matrix from scratch, requiring $\frac{1}{2}W_{\text{norm}}(W_{\text{norm}}-1)$ quantum circuit evaluations per step.
- **Algorithmic Mechanism:** When the window slides by $H$, let $k_{\text{keep}} \approx W_{\text{norm}} - H_{\text{norm}}$ denote the normal samples that remain in the active window, and $n_{\text{new}} = W_{\text{norm}} - k_{\text{keep}}$ denote the newly arrived normal samples. We discard the oldest rows and columns of the cached kernel matrix $K^{(t-1)}$ and retain the $(k_{\text{keep}} \times k_{\text{keep}})$ submatrix:
  $$K^{(t)}_{\text{[:}k_{\text{keep}}\text{, :}k_{\text{keep}}\text{]}} \leftarrow K^{(t-1)}_{\text{[-}k_{\text{keep}}\text{:, -}k_{\text{keep}}\text{:]}}$$
  We then compute quantum state overlaps **only** for the $n_{\text{new}}$ arriving samples against the $k_{\text{keep}}$ retained samples ($k_{\text{keep}} \cdot n_{\text{new}}$ cross entries) and among themselves ($\frac{1}{2}n_{\text{new}}(n_{\text{new}}-1)$ self entries).
- **Complexity Reduction & Verification:** Per-step kernel circuit evaluations drop from $O(W^2)$ to $O(W \cdot H)$—a theoretical reduction factor of $\approx \frac{W}{2H}$. Automated unit testing (`tests/test_member_b.py::test_e1_incremental_kernel_cache_exact_equality`) verifies that $K_{\text{cached}}$ and resulting anomaly scores equal full recomputation within `atol = 1e-10`. Empirically (`results/quantum/T2_QSVR_streaming_week3.csv` and `figures/fig2_e1_cache_scaling_vs_W.png`), E1 achieves **exact AUC parity (`0.9556`)** while delivering a **3.24x encoding speedup** (`0.3034s` $\to$ `0.0936s`) at $W=150$ and cutting total execution time by **46.5%** at $W=400$.

### 2.2 Enhancement E2: Warm-Started Variational Training for VQR
Retraining a variational quantum circuit from random parameter initialization $\theta^{(0)} \sim \text{Uniform}[-\pi, \pi]^p$ at every sliding step incurs prohibitive circuit execution overhead.
- **Algorithmic Mechanism:** Under E2 Warm-Starting, the converged ansatz parameters $\theta^*_{(t-1)}$ from window $t-1$ are transferred as the initial point $\theta^{(0)}_{(t)} \leftarrow \theta^*_{(t-1)}$ for window $t$. Because adjacent windows share $W - H$ overlapping samples, the loss landscape $\mathcal{L}_t(\theta)$ is a smooth perturbation of $\mathcal{L}_{t-1}(\theta)$ (provided drift is bounded).
- **Empirical Impact (`results/quantum/T1_VQR_streaming_week3.csv` & `figures/fig3_drift_response_vqr.png`):**
  1. **Equal Iteration Budget (15 iterations):** Warm-starting reduces streaming Mean MSE from **`0.0967` (Cold)** to **`0.0928` (Warm)** at zero extra circuit cost (`1,220` evaluations/step).
  2. **Target Accuracy Regime ($\text{MSE} \le 0.068$):** Warm-starting with target stopping achieves **`0.0886` Mean MSE** (outperforming 35-iteration cold-start's `0.0905` MSE) while slashing quantum circuit evaluations by **62.0%** (`1,071` vs. `2,820` evals/step) and per-step latency by **2.66x** (`0.917s` vs. `2.423s`).

### 2.3 Enhancement E3: Encoding vs. Model Update Latency Decomposition
Herbst et al. (2024) identify quantum state encoding as a critical edge-computing bottleneck but do not separate encoding latency from optimization latency in a streaming loop. Our E3 instrumentation (`figures/fig4_e3_time_decomposition.png`) demonstrates a fundamental dichotomy between the two QML paradigms:
- In **Variational Streaming (`VQR`)**, feature map encoding accounts for **~19.7%** of per-window training latency (`0.183s` of `0.917s`), while iterative ansatz binding and classical optimization dominate (**~80.3%**).
- In **Kernel Streaming (`QSVR`)**, quantum state preparation and kernel matrix evaluation account for **>96%** of training latency (`0.303s` vs. `0.004s` classical SVR quadratic programming solve), confirming why E1 kernel caching is indispensable for streaming quantum kernels.

---

## 3. NISQ Hardware Limits & Quantum Interpretation

### 3.1 Exponential Kernel Concentration (Thanasilp et al., 2022)
A central risk in scaling quantum kernels to more features is **exponential concentration**: as qubit count $n$ and Hilbert space dimension $D = 2^n$ grow, inner products between quantum states concentrate exponentially around a constant, causing $K(x, x') \to I$.
- In our diagnostic study (`results/quantum/diagnostics_qubits_concentration.csv` and `figures/fig1_kernel_concentration_qubits.png`), sweeping $n \in \{2, 3, 4, 5, 6, 7, 8\}$ on Task T2 shows that the off-diagonal kernel variance $\text{Var}_{i \neq j}[K(x_i, x_j)]$ decays by **141x**, from **`0.06907` ($n=2$)** down to **`0.00049` ($n=8$)**, following an exponential decay law $\text{Var}[K] \propto 2.27^{-n}$.
- At $n=8$ qubits, the off-diagonal standard deviation ($\sqrt{0.00049} \approx 0.022$) falls below the standard error of 1,024-shot binomial measurement noise ($1/\sqrt{1024} \approx 0.031$). Beyond $n \ge 12$ qubits, polynomially many measurement shots cannot resolve pairwise similarities from zero, causing SVR predictions to collapse to a constant mean. Consequently, aggressive dimensionality reduction ($n \le 7$ via frozen initial-window PCA) is not merely a simulator convenience—it is a fundamental mathematical requirement for expressive NISQ kernels.

### 3.2 Finite-Shot Noise Resilience (`shots=1024`)
At $n=5$ qubits ($\text{Var}[K_{ij}] = 0.00866$, standard deviation $\approx 0.093 > 0.031$), evaluating `QSVR` with `1024` multinomial measurement shots per circuit (`ZZ_r1_cached_shots1024`) yields a Final Pooled ROC-AUC of **`0.9557`**, matching noiseless statevector simulation (**`0.9556`**).

### 3.3 Simulator Wall-Clock vs. Physical QPU Latency
While statevector simulation computes an entire $W$-sample kernel in `<0.3s` by caching $2^n$-dimensional complex statevectors in RAM, a physical superconducting QPU (e.g., IBM Heron at ~10 kHz circuit repetition rate with $S=1024$ shots) requires $\approx 0.1\text{s}$ *per kernel entry*. Thus, a single uncached $W=400$ window (`79,800` circuit pairs) would require **~2.2 hours** on hardware, whereas our **E1 Incremental Kernel Cache** ($W=400, H=50$, ~18,000 new pairs) reduces hardware wall-clock time to **~30 minutes**—still orders of magnitude slower than classical streaming models (`<0.01s`).

---

## 4. Threats to Validity & Limitations

1. **Internal Validity (Preprocessing & Hyperparameter Fairness):**
   - To eliminate lookahead bias, `StandardScaler`, `PCA`, and $[0, \pi]$ `MinMaxScaler` were fit strictly on the initial window ($W_0$) and held frozen across the stream. Under severe feature drift (`abrupt` and `gradual`), frozen PCA axes can become suboptimal over long horizons, motivating adaptive subspace tracking in future work.
   - To prevent "tuned quantum vs. default classical" bias (Bowles et al., 2024), all classical and quantum models were allocated equal hyperparameter tuning budgets on the initial window prefix.
2. **External Validity (Dataset Scale & Qubit Count):**
   - Due to exponential simulation scaling and kernel concentration, experiments were constrained to $n=7$ qubits (`T1`) and $n=5$ qubits (`T2`) across window sizes $W \in \{100, 200, 400\}$. Results on high-dimensional tabular streams without low-rank PCA structure may favor classical random forests or Hoeffding trees even more strongly.
3. **Construct Validity (Noiseless & Shot-Noise Simulation vs. Hardware Gate Noise):**
   - Our runs use exact statevector simulation and binomial shot-noise modeling (`shots=1024`) without depolarizing two-qubit gate errors, readout crosstalk, or $T_1/T_2$ decoherence. Noiseless simulation therefore represents a strict **upper bound** on achievable NISQ hardware accuracy.

---

## 5. Quantum Results Synthesis & Conclusion Summary (Draft Contribution)

- **RQ1 (Predictive Quality):** On Task T2, `QSVR` (`0.9522` static AUC; `0.9556` streaming pooled AUC) matches tuned classical `SVR-RBF` (`0.9539` AUC) within `0.2%`, refuting claims of large quantum kernel superiority once classical baselines are tuned fairly (**H1 Confirmed**). On Task T1, `ZFeatureMap` + `RealAmplitudes` VQR achieves `0.0693` static MSE and `0.0886` warm-started streaming MSE.
- **RQ2 & RQ5 (Streaming Cost & Enhancements E1–E3):** Quantum computational cost scales quadratically with window size $W$ for uncached kernels (**H2 Confirmed**). However, **Enhancement E1 (Incremental Kernel Cache)** eliminates redundant overlaps, reducing encoding time by **3.24x** and circuit executions by **$W/2H$** with zero accuracy loss (**H3 Confirmed**), while **Enhancement E2 (Warm-Started VQR)** cuts variational circuit evaluations by **62.0%** and improves streaming MSE (`0.0886` vs. `0.0905`).
