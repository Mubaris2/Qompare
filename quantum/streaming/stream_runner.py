import os
import tracemalloc
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, average_precision_score
from quantum.models.vqr_model import QuantumVQRModel
from quantum.models.qsvr_model import QuantumSVRAnomalyModel

def run_t1_vqr_streaming(
    parquet_path: str = "data/processed/T1_stream.parquet",
    W: int = 200,
    H: int = 20,
    n_steps: int = 15,
    seed: int = 42,
):
    df = pd.read_parquet(parquet_path)
    feature_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feature_cols].values
    y = df["target"].values

    # Compare 4 regimes required by Week 3 E2:
    # 1. Cold-Start Baseline (maxiter=15)
    # 2. Warm-Start Equal Iteration Budget (maxiter=15)
    # 3. Cold-Start High Budget (maxiter=35)
    # 4. Warm-Start Equal Accuracy Target (target_mse=0.068, maxiter=35)
    configs = [
        {"variant": "Z_RealAmplitudes_r1_COBYLA_cold_iter15", "warm_start": False, "maxiter": 15, "target_mse": None},
        {"variant": "Z_RealAmplitudes_r1_COBYLA_warm_iter15", "warm_start": True,  "maxiter": 15, "target_mse": None},
        {"variant": "Z_RealAmplitudes_r1_COBYLA_cold_iter35", "warm_start": False, "maxiter": 35, "target_mse": None},
        {"variant": "Z_RealAmplitudes_r1_COBYLA_warm_target", "warm_start": True,  "maxiter": 35, "target_mse": 0.068},
    ]

    rows = []
    os.makedirs("results/quantum", exist_ok=True)

    for cfg in configs:
        model = QuantumVQRModel(
            n_qubits=len(feature_cols),
            feature_map_type="Z",
            ansatz_type="RealAmplitudes",
            reps=1,
            optimizer_type="COBYLA",
            maxiter=cfg["maxiter"],
            max_train_samples=80,
            warm_start=cfg["warm_start"],
            target_mse=cfg["target_mse"],
            seed=seed,
            variant=cfg["variant"],
        )
        print(f"\n[T1 Stream E2/E3] Running {model.variant} (W={W}, H={H}, steps={n_steps})...")

        for step in range(n_steps):
            start_idx = step * H
            train_end = start_idx + W
            test_end = train_end + H
            if test_end > len(X):
                break

            X_win, y_win = X[start_idx:train_end], y[start_idx:train_end]
            X_test, y_test = X[train_end:test_end], y[train_end:test_end]

            tracemalloc.start()
            model.fit(X_win, y_win)
            y_pred = model.predict(X_test)
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mb = round(peak_bytes / (1024 * 1024), 3)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            for m_name, m_val in [("MSE", mse), ("MAE", mae)]:
                rows.append({
                    "task": "T1",
                    "model": "VQR",
                    "variant": model.variant,
                    "seed": seed,
                    "W": W,
                    "H": H,
                    "drift": "none",
                    "step": step,
                    "n_seen": train_end,
                    "metric_name": m_name,
                    "metric_value": round(float(m_val), 6),
                    "t_encode": round(model.timings["t_encode"], 4),
                    "t_fit": round(model.timings["t_fit"], 4),
                    "t_predict": round(model.timings["t_predict"], 4),
                    "n_params": model.meta["n_params"],
                    "n_qubits": model.meta["n_qubits"],
                    "n_circuit_evals": model.meta["n_circuit_evals"],
                    "peak_mem_mb": peak_mb,
                })

        df_cfg = pd.DataFrame([r for r in rows if r["variant"] == model.variant and r["metric_name"] == "MSE"])
        print(
            f"  -> Mean MSE: {df_cfg['metric_value'].mean():.4f} | "
            f"Mean t_encode: {df_cfg['t_encode'].mean():.3f}s | "
            f"Mean t_fit: {df_cfg['t_fit'].mean():.3f}s | "
            f"Mean Evals/step: {df_cfg['n_circuit_evals'].mean():.0f}"
        )

    out_csv = "results/quantum/T1_VQR_streaming_week3.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] Saved T1 VQR streaming results to {out_csv}")


def run_t2_qsvr_streaming(
    parquet_path: str = "data/processed/T2_stream.parquet",
    W: int = 150,
    H: int = 30,
    seed: int = 42,
):
    df = pd.read_parquet(parquet_path)
    feature_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feature_cols].values
    y = df["label"].values

    # Compare 3 regimes required by Week 3 E1 + Shot Noise Ablation:
    # 1. Full Recompute (Noiseless)
    # 2. E1 Incremental Kernel Cache (Noiseless)
    # 3. Finite-Shot Noise Ablation (shots=1024 + Cache)
    configs = [
        {"variant": "ZZ_r1_full_exact",       "use_cache": False, "shots": None},
        {"variant": "ZZ_r1_cached_exact",     "use_cache": True,  "shots": None},
        {"variant": "ZZ_r1_cached_shots1024", "use_cache": True,  "shots": 1024},
    ]

    rows = []
    os.makedirs("results/quantum", exist_ok=True)

    for cfg in configs:
        model = QuantumSVRAnomalyModel(
            n_qubits=len(feature_cols),
            reps=1,
            C=1.0,
            epsilon=0.05,
            use_cache=cfg["use_cache"],
            shots=cfg["shots"],
            seed=seed,
            variant=cfg["variant"],
        )
        print(f"\n[T2 Stream E1/Ablation] Running {model.variant} (W={W}, H={H})...")

        pooled_y = []
        pooled_scores = []
        step = 0

        while True:
            start_idx = step * H
            train_end = start_idx + W
            test_end = train_end + H
            if test_end > len(X):
                break

            X_win, y_win = X[start_idx:train_end], y[start_idx:train_end]
            X_test, y_test = X[train_end:test_end], y[train_end:test_end]

            tracemalloc.start()
            model.fit(X_win, y_win)
            scores = model.score_anomaly(X_test)
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mb = round(peak_bytes / (1024 * 1024), 3)

            pooled_y.extend(y_test.tolist())
            pooled_scores.extend(scores.tolist())

            # Compute rolling pooled AUC & AP (per Section 2.2 protocol to handle blocks with few anomalies)
            if len(set(pooled_y)) > 1:
                auc = roc_auc_score(pooled_y, pooled_scores)
                ap = average_precision_score(pooled_y, pooled_scores)
            else:
                auc, ap = 0.5, 0.5

            for m_name, m_val in [("AUC", auc), ("AP", ap)]:
                rows.append({
                    "task": "T2",
                    "model": "QSVR",
                    "variant": model.variant,
                    "seed": seed,
                    "W": W,
                    "H": H,
                    "drift": "none",
                    "step": step,
                    "n_seen": train_end,
                    "metric_name": m_name,
                    "metric_value": round(float(m_val), 6),
                    "t_encode": round(model.timings["t_encode"], 4),
                    "t_fit": round(model.timings["t_fit"], 4),
                    "t_predict": round(model.timings["t_predict"], 4),
                    "n_params": model.meta["n_params"],
                    "n_qubits": model.meta["n_qubits"],
                    "n_circuit_evals": model.meta["n_circuit_evals"],
                    "peak_mem_mb": peak_mb,
                })
            step += 1

        df_cfg = pd.DataFrame([r for r in rows if r["variant"] == model.variant and r["metric_name"] == "AUC"])
        print(
            f"  -> Final Pooled AUC: {df_cfg['metric_value'].iloc[-1]:.4f} | "
            f"Mean t_encode: {df_cfg['t_encode'].mean():.4f}s | "
            f"Mean Circuit Evals/step: {df_cfg['n_circuit_evals'].mean():.0f}"
        )

    out_csv = "results/quantum/T2_QSVR_streaming_week3.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] Saved T2 QSVR streaming results to {out_csv}")


if __name__ == "__main__":
    run_t2_qsvr_streaming()
    run_t1_vqr_streaming()
