import os
import tracemalloc
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from quantum.models.vqr_model import QuantumVQRModel

def run_week1_static_p2(parquet_path: str = "data/processed/T1_stream.parquet"):
    df = pd.read_parquet(parquet_path)
    feature_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feature_cols].values
    y = df["target"].values

    # Exact Paper 2 static split: 400 train, 250 test on 7 PCA features
    X_train, y_train = X[:400], y[:400]
    X_test, y_test = X[400:650], y[400:650]

    os.makedirs("results/quantum", exist_ok=True)
    rows = []

    print("=== Week 1: Paper 2 Static Reproduction (SeoulBikeData.csv, 7 PCA features, 400 train / 250 test) ===")
    for opt in ["COBYLA", "SPSA"]:
        tracemalloc.start()
        model = QuantumVQRModel(
            n_qubits=7,
            feature_map_type="Z",
            ansatz_type="RealAmplitudes",
            reps=1,
            optimizer_type=opt,
            maxiter=40,
            max_train_samples=100,
            warm_start=False,
            seed=42,
            variant=f"Z_RealAmplitudes_r1_{opt}",
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = round(peak_bytes / (1024 * 1024), 3)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        total_time = model.timings["t_encode"] + model.timings["t_fit"] + model.timings["t_predict"]

        print(
            f"  [{model.variant}] MSE: {mse:.4f} | MAE: {mae:.4f} | "
            f"Encode: {model.timings['t_encode']:.2f}s | Fit: {model.timings['t_fit']:.2f}s | "
            f"Predict: {model.timings['t_predict']:.2f}s | Total: {total_time:.2f}s"
        )

        for m_name, m_val in [("MSE", mse), ("MAE", mae)]:
            rows.append({
                "task": "T1",
                "model": "VQR",
                "variant": model.variant,
                "seed": 42,
                "W": 400,
                "H": 250,
                "drift": "none",
                "step": 0,
                "n_seen": 400,
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

    out_csv = "results/quantum/week1_T1_VQR_static.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] Saved Week 1 static reproduction to {out_csv}")

if __name__ == "__main__":
    run_week1_static_p2()
