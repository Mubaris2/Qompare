import os
import time
import subprocess
import tracemalloc
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, average_precision_score
from quantum.models.vqr_model import QuantumVQRModel
from quantum.models.qsvr_model import QuantumSVRAnomalyModel

def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"

def inject_drift(X: np.ndarray, y: np.ndarray, drift_type: str, shift_magnitude: float = 0.8):
    """
    Implements Section 2.2 Drift Scenarios:
    - none: unmodified stream
    - abrupt: feature mean shift at 50% of stream
    - gradual: linear feature shift over 20% of stream (from 40% to 60%)
    """
    X_d = X.copy()
    y_d = y.copy()
    n = len(X_d)
    if drift_type == "none":
        return X_d, y_d
    elif drift_type == "abrupt":
        mid = n // 2
        X_d[mid:] = np.mod(X_d[mid:] + shift_magnitude, np.pi)
        return X_d, y_d
    elif drift_type == "gradual":
        s_idx = int(0.4 * n)
        e_idx = int(0.6 * n)
        span = max(1, e_idx - s_idx)
        for i in range(s_idx, n):
            alpha = min(1.0, (i - s_idx) / span)
            X_d[i] = np.mod(X_d[i] + alpha * shift_magnitude, np.pi)
        return X_d, y_d
    else:
        raise ValueError(f"Unknown drift_type: {drift_type}")

def run_full_grid():
    os.makedirs("results/quantum", exist_ok=True)
    commit_hash = get_git_commit()
    run_log_rows = []

    # =========================================================================
    # 1. TASK T2 FULL GRID: QSVR (W in {100, 200, 400} x Drift x Cache x Seeds)
    # =========================================================================
    df_t2 = pd.read_parquet("data/processed/T2_stream.parquet")
    f_cols_t2 = [c for c in df_t2.columns if c.startswith("f")]
    X_t2_raw = df_t2[f_cols_t2].values
    y_t2_raw = df_t2["label"].values

    W_grid = [100, 200, 400]
    H_t2 = 50
    drifts = ["none", "abrupt", "gradual"]
    seeds = [42, 100, 2024]
    qsvr_variants = [
        {"name": "ZZ_r1_cached_exact", "use_cache": True,  "shots": None},
        {"name": "ZZ_r1_full_exact",   "use_cache": False, "shots": None},
    ]

    t2_rows = []
    print(f"=== Starting Week 4 Full Grid for T2 QSVR (commit: {commit_hash}) ===")

    for W in W_grid:
        for drift in drifts:
            X_stream, y_stream = inject_drift(X_t2_raw, y_t2_raw, drift_type=drift)
            for var in qsvr_variants:
                # Run full recompute on seed 42 across all W/drifts for E1 comparison, and cached across all 3 seeds
                active_seeds = seeds if var["use_cache"] else [42]
                for seed in active_seeds:
                    t_job0 = time.perf_counter()
                    rng = np.random.default_rng(seed)
                    # Perturb stream order slightly after initial normal window per seed
                    idx = np.arange(len(X_stream))
                    tail = idx[150:].copy()
                    rng.shuffle(tail)
                    idx[150:] = tail
                    X_s, y_s = X_stream[idx], y_stream[idx]

                    model = QuantumSVRAnomalyModel(
                        n_qubits=len(f_cols_t2),
                        reps=1,
                        C=1.0,
                        epsilon=0.05,
                        use_cache=var["use_cache"],
                        shots=var["shots"],
                        seed=seed,
                        variant=var["name"],
                    )

                    pooled_y, pooled_scores = [], []
                    step = 0
                    while True:
                        s_idx = step * H_t2
                        tr_end = s_idx + W
                        te_end = tr_end + H_t2
                        if te_end > len(X_s):
                            break

                        X_win, y_win = X_s[s_idx:tr_end], y_s[s_idx:tr_end]
                        X_test, y_test = X_s[tr_end:te_end], y_s[tr_end:te_end]

                        tracemalloc.start()
                        model.fit(X_win, y_win)
                        scores = model.score_anomaly(X_test)
                        _, peak_bytes = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        peak_mb = round(peak_bytes / (1024 * 1024), 3)

                        pooled_y.extend(y_test.tolist())
                        pooled_scores.extend(scores.tolist())

                        if len(set(pooled_y)) > 1:
                            auc = roc_auc_score(pooled_y, pooled_scores)
                            ap = average_precision_score(pooled_y, pooled_scores)
                        else:
                            auc, ap = 0.5, 0.5

                        for m_name, m_val in [("AUC", auc), ("AP", ap)]:
                            t2_rows.append({
                                "task": "T2",
                                "model": "QSVR",
                                "variant": var["name"],
                                "seed": seed,
                                "W": W,
                                "H": H_t2,
                                "drift": drift,
                                "step": step,
                                "n_seen": tr_end,
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

                    job_time = round(time.perf_counter() - t_job0, 2)
                    run_log_rows.append({
                        "task": "T2", "model": "QSVR", "variant": var["name"],
                        "W": W, "H": H_t2, "drift": drift, "seed": seed,
                        "steps_completed": step, "runtime_s": job_time, "commit": commit_hash
                    })
                    print(f"  [T2 Done] {var['name']} | W={W} | drift={drift} | seed={seed} | steps={step} ({job_time}s)")

    out_t2 = "results/quantum/T2_QSVR_full_grid.csv"
    pd.DataFrame(t2_rows).to_csv(out_t2, index=False)
    print(f"[OK] Saved {len(t2_rows)} rows to {out_t2}")

    # =========================================================================
    # 2. TASK T1 FULL GRID: VQR (W in {100, 200, 400} x Drift x Warm/Cold x Seeds)
    # =========================================================================
    df_t1 = pd.read_parquet("data/processed/T1_stream.parquet")
    f_cols_t1 = [c for c in df_t1.columns if c.startswith("f")]
    # Use a 640-sample stream horizon per configuration so W=100,200,400 all execute cleanly
    X_t1_raw = df_t1[f_cols_t1].values[:640]
    y_t1_raw = df_t1["target"].values[:640]

    H_t1 = 20
    n_steps_t1 = 10
    vqr_variants = [
        {"name": "Z_RealAmplitudes_r1_COBYLA_warm", "warm_start": True,  "maxiter": 12, "target_mse": 0.068},
        {"name": "Z_RealAmplitudes_r1_COBYLA_cold", "warm_start": False, "maxiter": 12, "target_mse": None},
    ]

    t1_rows = []
    print(f"\n=== Starting Week 4 Full Grid for T1 VQR (commit: {commit_hash}) ===")

    for W in W_grid:
        for drift in drifts:
            stream_len = W + n_steps_t1 * H_t1
            X_sub, y_sub = X_t1_raw[:stream_len], y_t1_raw[:stream_len]
            X_stream, y_stream = inject_drift(X_sub, y_sub, drift_type=drift)

            for var in vqr_variants:
                active_seeds = seeds if var["warm_start"] else [42]
                for seed in active_seeds:
                    t_job0 = time.perf_counter()
                    model = QuantumVQRModel(
                        n_qubits=len(f_cols_t1),
                        feature_map_type="Z",
                        ansatz_type="RealAmplitudes",
                        reps=1,
                        optimizer_type="COBYLA",
                        maxiter=var["maxiter"],
                        max_train_samples=60,
                        warm_start=var["warm_start"],
                        target_mse=var["target_mse"],
                        seed=seed,
                        variant=var["name"],
                    )

                    for step in range(n_steps_t1):
                        s_idx = step * H_t1
                        tr_end = s_idx + W
                        te_end = tr_end + H_t1

                        X_win, y_win = X_stream[s_idx:tr_end], y_stream[s_idx:tr_end]
                        X_test, y_test = X_stream[tr_end:te_end], y_stream[tr_end:te_end]

                        tracemalloc.start()
                        model.fit(X_win, y_win)
                        y_pred = model.predict(X_test)
                        _, peak_bytes = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        peak_mb = round(peak_bytes / (1024 * 1024), 3)

                        mse = mean_squared_error(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)

                        for m_name, m_val in [("MSE", mse), ("MAE", mae)]:
                            t1_rows.append({
                                "task": "T1",
                                "model": "VQR",
                                "variant": var["name"],
                                "seed": seed,
                                "W": W,
                                "H": H_t1,
                                "drift": drift,
                                "step": step,
                                "n_seen": tr_end,
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

                    job_time = round(time.perf_counter() - t_job0, 2)
                    run_log_rows.append({
                        "task": "T1", "model": "VQR", "variant": var["name"],
                        "W": W, "H": H_t1, "drift": drift, "seed": seed,
                        "steps_completed": n_steps_t1, "runtime_s": job_time, "commit": commit_hash
                    })
                    print(f"  [T1 Done] {var['name']} | W={W} | drift={drift} | seed={seed} | steps={n_steps_t1} ({job_time}s)")

    out_t1 = "results/quantum/T1_VQR_full_grid.csv"
    pd.DataFrame(t1_rows).to_csv(out_t1, index=False)
    print(f"[OK] Saved {len(t1_rows)} rows to {out_t1}")

    log_path = "results/quantum/run_log.csv"
    pd.DataFrame(run_log_rows).to_csv(log_path, index=False)
    print(f"[OK] Saved experiment run log ({len(run_log_rows)} jobs) to {log_path}")

if __name__ == "__main__":
    run_full_grid()
