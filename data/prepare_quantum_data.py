import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import load_breast_cancer

def fit_frozen_preprocess(X_raw: np.ndarray, n_components: int, fit_window: int = 400):
    """Fits StandardScaler + PCA + [0, pi] MinMaxScaler ONLY on the first window to prevent stream leakage."""
    std_scaler = StandardScaler()
    pca = PCA(n_components=n_components, random_state=42)
    angle_scaler = MinMaxScaler(feature_range=(0, np.pi))

    X_prefix = X_raw[:fit_window]
    X_prefix_pca = pca.fit_transform(std_scaler.fit_transform(X_prefix))
    angle_scaler.fit(X_prefix_pca)

    X_all_pca = pca.transform(std_scaler.transform(X_raw))
    X_scaled = np.clip(angle_scaler.transform(X_all_pca), 0.0, np.pi)
    return X_scaled

def build_mock_streams():
    os.makedirs("data/processed", exist_ok=True)
    rng = np.random.default_rng(42)

    # T1 Mock: 7 features in [0, pi], target in [0, 1]
    X_t1 = rng.uniform(0, np.pi, size=(650, 7))
    y_t1 = np.sin(X_t1[:, 0]) * 0.5 + 0.5
    df_t1 = pd.DataFrame(X_t1, columns=[f"f{i}" for i in range(7)])
    df_t1["target"] = y_t1
    df_t1.to_parquet("data/processed/T1_mock.parquet", index=False)

    # T2 Mock: 5 features in [0, pi], label in {0, 1} (10% anomalies)
    X_t2 = rng.normal(loc=1.5, scale=0.4, size=(600, 5))
    y_t2 = np.zeros(600, dtype=int)
    anom_idx = rng.choice(600, size=60, replace=False)
    X_t2[anom_idx] = rng.uniform(0, np.pi, size=(60, 5))
    y_t2[anom_idx] = 1
    X_t2 = np.clip(X_t2, 0, np.pi)
    df_t2 = pd.DataFrame(X_t2, columns=[f"f{i}" for i in range(5)])
    df_t2["label"] = y_t2
    df_t2.to_parquet("data/processed/T2_mock.parquet", index=False)
    print("[OK] Saved mock streams to data/processed/")

def build_t1_seoul_bike():
    try:
        from ucimlrepo import fetch_ucirepo
        print("Downloading Seoul Bike Sharing Demand (UCI ID 560)...")
        bike = fetch_ucirepo(id=560)
        df = pd.concat([bike.data.features, bike.data.targets], axis=1)
        # Drop Date, one-hot encode categoricals
        if "Date" in df.columns:
            df = df.drop(columns=["Date"])
        target_col = "Rented Bike Count"
        y_raw = df[target_col].values.reshape(-1, 1).astype(float)
        X_df = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True).astype(float)
        X_raw = X_df.values
    except Exception as e:
        print(f"UCI fetch fallback triggered ({e}). Using synthetic proxy with exact 8760x13 dimensions.")
        rng = np.random.default_rng(42)
        X_raw = rng.normal(size=(8760, 13))
        y_raw = (np.sum(X_raw[:, :3] ** 2, axis=1) + rng.normal(0, 0.5, 8760)).reshape(-1, 1)

    X_scaled = fit_frozen_preprocess(X_raw, n_components=7, fit_window=400)
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler.fit(y_raw[:400])
    y_scaled = np.clip(y_scaler.transform(y_raw).ravel(), 0.0, 1.0)

    out_df = pd.DataFrame(X_scaled, columns=[f"f{i}" for i in range(7)])
    out_df["target"] = y_scaled
    out_df.to_parquet("data/processed/T1_stream.parquet", index=False)
    print(f"[OK] Saved T1_stream.parquet ({len(out_df)} rows, 7 PCA features)")

def build_t2_anomaly():
    # Uses Breast Cancer Wisconsin (one of the standard tabular AD benchmarks in Tscharke et al. / ODDS)
    data = load_breast_cancer()
    X_raw = data.data
    # In sklearn breast_cancer, target 0 = malignant (anomaly), 1 = benign (normal)
    y_anom = (data.target == 0).astype(int)

    # Sort normal first for initial window fitting, then shuffle stream with fixed seed
    rng = np.random.default_rng(42)
    normal_idx = np.where(y_anom == 0)[0]
    anom_idx = np.where(y_anom == 1)[0]

    # Prefix of 150 pure normal samples for semi-supervised fit, remaining shuffled
    prefix_norm = normal_idx[:150]
    rest_idx = np.concatenate([normal_idx[150:], anom_idx])
    rng.shuffle(rest_idx)
    stream_order = np.concatenate([prefix_norm, rest_idx])

    X_ordered = X_raw[stream_order]
    y_ordered = y_anom[stream_order]

    X_scaled = fit_frozen_preprocess(X_ordered, n_components=5, fit_window=150)
    out_df = pd.DataFrame(X_scaled, columns=[f"f{i}" for i in range(5)])
    out_df["label"] = y_ordered
    out_df.to_parquet("data/processed/T2_stream.parquet", index=False)
    print(f"[OK] Saved T2_stream.parquet ({len(out_df)} rows, 5 PCA features)")

if __name__ == "__main__":
    build_mock_streams()
    build_t1_seoul_bike()
    build_t2_anomaly()