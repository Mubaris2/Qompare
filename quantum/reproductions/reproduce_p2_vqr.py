import os
import tracemalloc
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from quantum.models.vqr_model import QuantumVQRModel

def run_p2_sweep(parquet_path: str = "data/processed/T1_stream.parquet", quick_mode: bool = False):
    df = pd.read_parquet(parquet_path)
    feature_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feature_cols].values
    y = df["target"].values

    # Paper 2 exact split sizes: 400 train, 250 test
    X_train, y_train = X[:400], y[:400]
    X_test, y_test = X[400:650], y[400:650]

    feature_maps = ["Z", "ZZ"]
    ansatze = ["EfficientSU2", "RealAmplitudes", "TwoLocal"]
    reps_list = [1] if quick_mode else [1, 2]
    optimizers = ["COBYLA", "SPSA"]
    maxiter = 15 if quick_mode else 40

    rows = []
    os.makedirs("results/quantum", exist_ok=True)

    for fm in feature_maps:
        for ans in ansatze:
            for r in reps_list:
                for opt in optimizers:
                    tracemalloc.start()
                    model = QuantumVQRModel(
                        n_qubits=len(feature_cols),
                        feature_map_type=fm,
                        ansatz_type=ans,
                        reps=r,
                        optimizer_type=opt,
                        maxiter=maxiter,
                        seed=42,
                    )
                    print(f"[P2 Sweep] Running {model.variant}...")
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    _, peak_bytes = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    peak_mb = round(peak_bytes / (1024 * 1024), 3)

                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    print(f"  -> MSE: {mse:.4f} | MAE: {mae:.4f} | Fit: {model.timings['t_fit']:.2f}s")

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

    out_csv = "results/quantum/T1_VQR.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] Saved P2 reproduction & sweep to {out_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run fast smoke sweep")
    args = parser.parse_args()
    run_p2_sweep(quick_mode=args.quick)

    