# Appendix: Classical Simulability and Quantum Complexity Analysis of Streaming Circuits

**Owner:** Member B (Week 4 Deliverable)

---

## 1. Overview of Circuits Used in Tasks T1 and T2

In our experimental comparison, we evaluate two distinct classes of parameterized quantum circuits (PQCs):
1. **Task T1 (Variational Quantum Regression — `VQR`):**
   - **Feature Maps:** `ZFeatureMap` ($U_{\Phi}(x) = \bigotimes_{j=1}^n H R_Z(2x_j)$) and `ZZFeatureMap` with linear nearest-neighbor entanglement ($e^{i (\pi - x_j)(\pi - x_{j+1}) Z_j Z_{j+1}}$).
   - **Variational Ansätze:** `RealAmplitudes` ($R_Y$ rotations + linear $\text{CNOT}$ ladder), `EfficientSU2` ($R_Y, R_Z$ + linear $\text{CNOT}$ ladder), and `TwoLocal` ($R_Y, R_Z$ + linear $\text{CZ}$).
   - **Observable:** Global Pauli-Z parity operator $O = Z^{\otimes n}$.
2. **Task T2 (Quantum Kernel Anomaly Detection — `QSVR`):**
   - **Feature Map:** 5-qubit angle embedding + nearest-neighbor $\text{CNOT}-R_Z(2(\pi-x_j)(\pi-x_{j+1}))-\text{CNOT}$ data-reuploading circuit ($L \in \{1, 2\}$ repetitions).
   - **Kernel Measurement:** Fidelity overlap $K(x, x') = \vert{}\langle 0^{\otimes n} \vert{} U^\dagger(x') U(x) \vert{} 0^{\otimes n} \rangle\vert{}^2$.

---

## 2. Classical Simulability of `ZFeatureMap` and Product-State Encodings

When `ZFeatureMap` is used without entangling gates in the feature map layer, the encoded quantum state is a **separable tensor product state**:
$$\vert{}\psi(x)\rangle = \bigotimes_{j=1}^n \left( \cos(x_j)\vert{}0\rangle - i\sin(x_j)\vert{}1\rangle \right)$$

- **Pure Kernel Implication:** Any quantum kernel constructed solely from `ZFeatureMap` factorizes into a product of single-qubit trigonometric inner products:
  $$K_Z(x, x') = \prod_{j=1}^n \vert{}\langle \psi_j(x_j) \vert{} \psi_j(x'_j) \rangle\vert{}^2 = \prod_{j=1}^n \cos^2(x_j - x'_j)$$
  This kernel is **analytically computable in $O(n)$ classical time** without a quantum computer.
- **Variational (`VQR`) Implication:** In `VQR`, `ZFeatureMap` is followed by an entangling ansatz (`RealAmplitudes` or `EfficientSU2` with linear $\text{CNOT}$ gates). While the ansatz creates entanglement across the $n=7$ qubits, a shallow 1D nearest-neighbor circuit of depth $d = O(1)$ has bounded **treewidth** and **bond dimension** $\chi \le 2^d$. By **Matrix Product State (MPS)** tensor network contraction, expectation values of local or factorizable observables like $Z^{\otimes n}$ on 1D circuits of logarithmic depth $d = O(\log n)$ can be simulated classically in polynomial time $O(n \cdot 2^{2d})$.

---

## 3. Simulability and Complexity of `ZZFeatureMap` Quantum Kernels

The `ZZFeatureMap` (Havlíček et al., 2019) alternates Hadamard layers with diagonal two-qubit unitaries $U_{ZZ}(x) = \exp\left(i \sum_{(j,k)\in E} \phi_{jk}(x) Z_j Z_k\right)$:
- **Single-Layer ($L=1$) IQP Connection:** For $L=1$, $H^{\otimes n} U_{ZZ}(x) H^{\otimes n}$ is an **Instantaneous Quantum Polynomial (IQP)** circuit. However, when the interaction graph $E$ is a **1D chain (linear entanglement)** as used in our `T2` stream, the circuit remains a 1D tensor network with bond dimension $\chi \le 4$ and can still be contracted classically in $O(n)$ time.
- **Multi-Layer ($L \ge 2$) with All-to-All Entanglement:** Classical hardness (under #P-hardness of IQP circuit output probabilities) only emerges when $L \ge 2$, the entanglement graph has non-planar/all-to-all connectivity of treewidth $\Omega(n)$, and $n \gtrsim 50$ qubits.

---

## 4. Exponential Kernel Concentration (Thanasilp et al., 2022)

A critical theoretical barrier emerges if one attempts to prevent classical tensor-network simulation by increasing qubit count $n$ and circuit depth $L$:
- Per **Thanasilp et al. (2022)**, as Hilbert space dimension $D = 2^n$ and entanglement grow, the variance of off-diagonal quantum kernel values decays exponentially:
  $$\text{Var}_{x, x'}[K(x, x')] \in O(b^{-n}), \quad b > 1$$
- Our Week 4 diagnostic study (`results/quantum/diagnostics_qubits_concentration.csv` and `figures/fig1_kernel_concentration_qubits.png`) empirically confirms this phenomenon even between $n=2$ and $n=8$ qubits: off-diagonal kernel variance drops by **nearly two orders of magnitude** ($0.095 \to 0.0029$), causing $K(x, x')$ to approach the identity matrix $I$ under finite shot noise (`shots=1024`).

---

## 5. Practical Takeaway for Streaming Analytics

At NISQ-era scales ($n \le 8$ qubits, 1D connectivity), all circuits studied in Base Papers 2 and 3 lie strictly inside the **classically simulable regime** (via exact statevector caching or MPS contraction). Moreover, statevector caching allows classical CPUs to compute an entire $W \times W$ noiseless fidelity kernel in $O(W \cdot 2^n + W^2 \cdot 2^n)$ operations instead of executing $O(W^2 \cdot S)$ noisy quantum circuits on hardware. Consequently, our **E1 Incremental Kernel Cache** is doubly essential: it reduces simulation encoding latency by **3.5x–6.8x** today and cuts physical quantum hardware circuit executions from $O(W^2)$ to $O(W \cdot H)$ for future fault-tolerant deployments.
