"""Tiny fake parquet files with the REAL schema so Member B is never blocked.
Columns: f0..f{d-1} in [0, pi], y. T1 y in [0,1]; T2 y in {0 normal, 1 anomaly}; T3 y in {0,1}."""
import json, numpy as np, pandas as pd
from pathlib import Path

OUT = Path(__file__).parent / "processed"; OUT.mkdir(exist_ok=True, parents=True)
rng = np.random.default_rng(0)


def save(name, X, y, meta):
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]); df["y"] = y
    df.to_parquet(OUT / f"{name}_stream.parquet", index=False)
    (OUT / f"{name}_stream.meta.json").write_text(json.dumps(meta, indent=2))


# T1: 2000 rows, 7 features, smooth nonlinear target + seasonal drift
n = 2000; X = rng.uniform(0, np.pi, (n, 7))
y = 0.5 + 0.25 * np.sin(X[:, 0]) * np.cos(X[:, 1]) + 0.1 * np.sin(2 * np.pi * np.arange(n) / 500) + rng.normal(0, .02, n)
save("T1", X, np.clip(y, 0, 1), {"mock": True, "task": "regression", "n_features": 7})

# T2: 3000 rows, 4 features, 5% anomalies (shifted cluster), anomaly-free first 600 rows
n = 3000; X = np.clip(rng.normal(np.pi / 2, 0.4, (n, 4)), 0, np.pi); y = np.zeros(n, int)
idx = rng.choice(np.arange(600, n), int(0.05 * n), replace=False)
X[idx] = np.clip(rng.normal(np.pi / 2 + 1.0, 0.5, (len(idx), 4)), 0, np.pi); y[idx] = 1
save("T2", X, y, {"mock": True, "task": "anomaly", "label": "1=anomaly", "n_features": 4})

# T3: 4000 rows, 4 features, binary label with abrupt concept drift at 50%
n = 4000; X = rng.uniform(0, np.pi, (n, 4)); y = (X[:, 0] + X[:, 1] > np.pi).astype(int)
y[n // 2:] = (X[n // 2:, 0] - X[n // 2:, 1] > 0).astype(int)
save("T3", X, y, {"mock": True, "task": "classification", "n_features": 4, "natural_drift_at": n // 2})
print("mock written to", OUT)
