import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from quantum.models.qsvr_model import QuantumSVRAnomalyModel

def run_qubit_concentration_study():
    """
    Tests Thanasilp et al. (2022) Exponential Concentration in Quantum Kernel Methods:
    Sweeps qubit count n in {2, 3, 4, 5, 6, 7, 8} on the T2 dataset and records:
    - Off-diagonal kernel variance Var_{i != j}[K(x_i, x_j)]
    - Off-diagonal kernel mean E_{i != j}[K(x_i, x_j)]
    - Test ROC-AUC (exact vs 1024 shots)
    """
    data = load_breast_cancer()
    X_raw = data.data
    y_anom = (data.target == 0).astype(int)

    normal_idx = np.where(y_anom == 0)[0]
    anom_idx = np.where(y_anom == 1)[0]
    rng = np.random.default_rng(42)
    rest_idx = np.concatenate([normal_idx[150:], anom_idx])
    rng.shuffle(rest_idx)
    order = np.concatenate([normal_idx[:150], rest_idx])

    X_ord, y_ord = X_raw[order], y_anom[order]

    rows = []
    print("=== Running Qubits vs. Accuracy & Kernel Concentration Study (Thanasilp et al.) ===")
    for n_qubits in [2, 3, 4, 5, 6, 7, 8]:
        std = StandardScaler()
        pca = PCA(n_components=n_qubits, random_state=42)
        ang = MinMaxScaler(feature_range=(0, np.pi))

        X_pref = pca.fit_transform(std.fit_transform(X_ord[:150]))
        ang.fit(X_pref)
        X_all = np.clip(ang.transform(pca.transform(std.transform(X_ord))), 0.0, np.pi)

        X_tr, y_tr = X_all[:150], y_ord[:150]
        X_te, y_te = X_all[150:350], y_ord[150:350]

        model_exact = QuantumSVRAnomalyModel(n_qubits=n_qubits, reps=2, shots=None, seed=42)
        model_exact.fit(X_tr, y_tr)
        scores_exact = model_exact.score_anomaly(X_te)
        auc_exact = roc_auc_score(y_te, scores_exact)

        model_shots = QuantumSVRAnomalyModel(n_qubits=n_qubits, reps=2, shots=1024, seed=42)
        model_shots.fit(X_tr, y_tr)
        scores_shots = model_shots.score_anomaly(X_te)
        auc_shots = roc_auc_score(y_te, scores_shots)

        K_mat = model_exact._fidelity_kernel(model_exact._train_states, model_exact._train_states)
        upper_vals = K_mat[np.triu_indices_from(K_mat, k=1)]
        k_mean = float(np.mean(upper_vals))
        k_var = float(np.var(upper_vals))

        rows.append({
            "n_qubits": n_qubits,
            "hilbert_dim": 2 ** n_qubits,
            "off_diag_kernel_mean": round(k_mean, 6),
            "off_diag_kernel_var": round(k_var, 6),
            "auc_exact": round(float(auc_exact), 4),
            "auc_shots1024": round(float(auc_shots), 4),
            "t_encode_s": round(model_exact.timings["t_encode"], 4),
        })
        print(
            f"  n_qubits={n_qubits} (dim={2**n_qubits:3d}) | "
            f"Mean(K_ij)={k_mean:.4f} | Var(K_ij)={k_var:.5f} | "
            f"AUC(exact)={auc_exact:.4f} | AUC(1024 shots)={auc_shots:.4f}"
        )

    out_path = "results/quantum/diagnostics_qubits_concentration.csv"
    df_diag = pd.DataFrame(rows)
    df_diag.to_csv(out_path, index=False)
    print(f"[OK] Saved kernel concentration study to {out_path}")
    return df_diag

def generate_checkpoint2_figures():
    os.makedirs("figures", exist_ok=True)
    df_t2 = pd.read_csv("results/quantum/T2_QSVR_full_grid.csv")
    df_t1 = pd.read_csv("results/quantum/T1_VQR_full_grid.csv")
    df_conc = pd.read_csv("results/quantum/diagnostics_qubits_concentration.csv")

    # Figure 1: Qubits vs Kernel Concentration & ROC-AUC (Thanasilp et al. diagnostic)
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(df_conc["n_qubits"], df_conc["off_diag_kernel_var"], "o-", color="#d95f02", linewidth=2, label="Off-Diagonal Kernel Variance Var(K_ij)")
    ax1.set_xlabel("Number of Qubits (n)")
    ax1.set_ylabel("Kernel Variance Var(K_ij)", color="#d95f02")
    ax1.set_yscale("log")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(df_conc["n_qubits"], df_conc["auc_exact"], "s--", color="#1b9e77", linewidth=2, label="ROC-AUC (Exact Statevector)")
    ax2.plot(df_conc["n_qubits"], df_conc["auc_shots1024"], "^:", color="#7570b3", linewidth=2, label="ROC-AUC (1024 Shots)")
    ax2.set_ylabel("Anomaly Detection ROC-AUC", color="#1b9e77")
    ax2.set_ylim(0.85, 1.0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8.5)
    plt.title("Quantum Kernel Concentration & Accuracy vs. Qubit Count")
    plt.tight_layout()
    plt.savefig("figures/fig1_kernel_concentration_qubits.png", dpi=200)
    plt.close()

    # Figure 2: E1 Circuit Executions & Encoding Latency vs Window Size W
    df_t2_auc = df_t2[(df_t2["metric_name"] == "AUC") & (df_t2["drift"] == "none") & (df_t2["step"] > 0)]
    summary_w = df_t2_auc.groupby(["variant", "W"])[["n_circuit_evals", "t_encode"]].mean().reset_index()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for var, label, marker in [
        ("ZZ_r1_full_exact", "Full Kernel Recompute", "o-"),
        ("ZZ_r1_cached_exact", "E1 Incremental Kernel Cache", "s--"),
    ]:
        sub = summary_w[summary_w["variant"] == var]
        ax_a.plot(sub["W"], sub["n_circuit_evals"], marker, linewidth=2, label=label)
        ax_b.plot(sub["W"], sub["t_encode"], marker, linewidth=2, label=label)

    ax_a.set_xlabel("Sliding Window Size (W)")
    ax_a.set_ylabel("Hardware Circuit Evaluations / Step")
    ax_a.set_title("Circuit Executions vs. Window Size W")
    ax_a.grid(True, linestyle="--", alpha=0.4)
    ax_a.legend()

    ax_b.set_xlabel("Sliding Window Size (W)")
    ax_b.set_ylabel("State Encoding Latency t_encode (s)")
    ax_b.set_title("Encoding Time vs. Window Size W")
    ax_b.grid(True, linestyle="--", alpha=0.4)
    ax_b.legend()
    plt.tight_layout()
    plt.savefig("figures/fig2_e1_cache_scaling_vs_W.png", dpi=200)
    plt.close()

    # Figure 3: Behavior Around Concept Drift (T1 VQR Warm vs Cold & T2 QSVR)
    df_t1_mse = df_t1[(df_t1["metric_name"] == "MSE") & (df_t1["W"] == 200) & (df_t1["seed"] == 42)]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for drift, style in [("none", "-"), ("abrupt", "--"), ("gradual", ":")]:
        sub = df_t1_mse[(df_t1_mse["variant"] == "Z_RealAmplitudes_r1_COBYLA_warm") & (df_t1_mse["drift"] == drift)]
        ax.plot(sub["n_seen"], sub["metric_value"], linestyle=style, marker="o", label=f"VQR Warm-Start (drift={drift})")
    ax.axvline(x=300, color="red", linestyle="--", alpha=0.5, label="Drift Injection Point")
    ax.set_xlabel("Stream Samples Seen (n_seen)")
    ax.set_ylabel("Sliding Window Test MSE")
    ax.set_title("VQR Streaming MSE Across Concept Drift Scenarios (W=200)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8.5)
    plt.tight_layout()
    plt.savefig("figures/fig3_drift_response_vqr.png", dpi=200)
    plt.close()

    # Figure 4: E3 Runtime Share Decomposition (Encode vs Fit vs Predict)
    t1_means = df_t1[df_t1["metric_name"] == "MSE"].groupby("W")[["t_encode", "t_fit", "t_predict"]].mean()
    t2_means = df_t2[(df_t2["metric_name"] == "AUC") & (df_t2["variant"] == "ZZ_r1_cached_exact")].groupby("W")[["t_encode", "t_fit", "t_predict"]].mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))
    t1_means.plot(kind="bar", stacked=True, ax=ax1, color=["#66c2a5", "#fc8d62", "#8da0cb"])
    ax1.set_title("E3 Time Breakdown: T1 VQR (Variational)")
    ax1.set_xlabel("Window Size (W)")
    ax1.set_ylabel("Mean Step Latency (s)")

    t2_means.plot(kind="bar", stacked=True, ax=ax2, color=["#66c2a5", "#fc8d62", "#8da0cb"])
    ax2.set_title("E3 Time Breakdown: T2 QSVR (Cached Kernel)")
    ax2.set_xlabel("Window Size (W)")
    ax2.set_ylabel("Mean Step Latency (s)")
    plt.tight_layout()
    plt.savefig("figures/fig4_e3_time_decomposition.png", dpi=200)
    plt.close()

    print("[OK] Saved 4 diagnostic plots to figures/")

if __name__ == "__main__":
    run_qubit_concentration_study()
    generate_checkpoint2_figures()
