import os
import time
import tracemalloc
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from quantum.models.qsvr_model import QuantumSVRAnomalyModel

def run_p3_reproduction(parquet_path: str = "data/processed/T2_stream.parquet"):
    df = pd.read_parquet(parquet_path)
    feature_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feature_cols].values
    label_col = "label" if "label" in df.columns else "y"
    y = df[label_col].values

    # Train on first 150 normal samples, evaluate on remaining stream
    X_train, y_train = X[:150], y[:150]
    X_test, y_test = X[150:], y[150:]

    rows = []
    os.makedirs("results/quantum", exist_ok=True)

    for reps in [1, 2]:
        for C in [1.0, 10.0]:
            tracemalloc.start()
            variant = f"ZZ_reps{reps}_C{C}"
            model = QuantumSVRAnomalyModel(
                n_qubits=len(feature_cols), reps=reps, C=C, epsilon=0.05, variant=variant
            )
            model.fit(X_train, y_train)
            scores = model.score_anomaly(X_test)
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mb = round(peak_bytes / (1024 * 1024), 3)

            auc = roc_auc_score(y_test, scores)
            ap = average_precision_score(y_test, scores)
            print(f"[P3 QSVR {variant}] AUC: {auc:.4f} | AP: {ap:.4f} | Encode: {model.timings['t_encode']:.2f}s")

            for m_name, m_val in [("AUC", auc), ("AP", ap)]:
                rows.append({
                    "task": "T2",
                    "model": "QSVR",
                    "variant": variant,
                    "seed": 42,
                    "W": 150,
                    "H": len(X_test),
                    "drift": "none",
                    "step": 0,
                    "n_seen": 150,
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

    # Classical SVR-RBF parity check for Checkpoint 1 comparison
    t0 = time.perf_counter()
    clf_svr = MultiOutputRegressor(SVR(kernel="rbf", C=1.0, epsilon=0.05))
    clf_svr.fit(X_train[y_train == 0], X_train[y_train == 0])
    t_fit_c = time.perf_counter() - t0
    t1 = time.perf_counter()
    scores_c = np.mean((X_test - clf_svr.predict(X_test)) ** 2, axis=1)
    t_pred_c = time.perf_counter() - t1
    auc_c = roc_auc_score(y_test, scores_c)
    print(f"[Classical SVR-RBF Reference] AUC: {auc_c:.4f} | Fit: {t_fit_c:.4f}s | Predict: {t_pred_c:.4f}s")

    out_csv = "results/quantum/T2_QSVR.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] Saved P3 reproduction to {out_csv}")

if __name__ == "__main__":
    run_p3_reproduction()