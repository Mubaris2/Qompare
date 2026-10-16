"""Build data/processed/T{1,2,3}_stream.parquet from raw files.
Contract: features f0..f{d-1} scaled to [0, pi] with scaler/PCA FIT ON FIRST `fit_rows` ONLY
(no leakage, honest streaming). Output order is final and shared with Member B.

Usage:
  python data/download.py                      # once: fills data/raw/
  python data/prep.py t1                       # Seoul Bike  (default data/raw/seoul_bike/)
  python data/prep.py t2 [--file data/raw/ad/6_cardio.npz] [--n-components 5] [--n-rows 10000]
  python data/prep.py t3 [--n-rows 10000] [--drop date]   # ELEC2 from data/raw/elec2/ (also --source synth|elec2|csv)
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import config as C
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"; PROC.mkdir(exist_ok=True, parents=True)
DEF_T1 = [RAW / "seoul_bike" / "SeoulBikeData.csv", RAW / "SeoulBikeData.csv"]   # 2nd = manual download
DEF_T2 = RAW / "ad" / "32_shuttle.npz"      # streaming T2: 61 anomalies in the tune prefix (mammography has 16). Mammography is used for P3 parity only
DEF_T3 = RAW / "elec2" / "electricity.csv"


def first_existing(paths):
    for q in paths:
        if Path(q).exists():
            return Path(q)
    raise SystemExit("raw file not found. Run: python data/download.py   (looked in: "
                     + ", ".join(str(Path(q).relative_to(ROOT)) for q in paths) + ")")


def fit_scale(X, fit_rows, n_components=None, fit_mask=None, margin=0.0):
    """Standardise -> optional PCA -> MinMax to [0, pi]; ALL statistics fitted on rows [0, fit_rows) only
    (optionally only the rows where fit_mask is True, e.g. normal samples for anomaly detection).
    margin widens the min/max range on both sides by margin x range (fewer clipped values later in the stream).
    Keep fit_rows <= TUNE_ROWS so no statistic is computed from rows that are used as test data."""
    X = np.asarray(X, float)
    F = X[:fit_rows] if fit_mask is None else X[:fit_rows][np.asarray(fit_mask)[:fit_rows]]
    mu, sd = F.mean(0), F.std(0) + 1e-12
    Z, F = (X - mu) / sd, (F - mu) / sd                       # F = standardised fit window
    info = {"fit_rows": fit_rows, "n_fit_samples": int(len(F)), "margin": margin}
    pca_mean, pca_comp = None, None
    if n_components:
        pca = PCA(n_components=n_components, random_state=0).fit(F)
        Z, F = pca.transform(Z), pca.transform(F)
        info["pca_explained_variance_ratio"] = pca.explained_variance_ratio_.round(4).tolist()
        pca_mean = pca.mean_.tolist()
        pca_comp = pca.components_.tolist()
    lo, hi = F.min(0), F.max(0); r = np.where(hi - lo == 0, 1, hi - lo); lo, hi = lo - margin * r, hi + margin * r
    S = np.pi * (Z - lo) / (hi - lo)
    outside = (S < 0) | (S > np.pi)
    info["frac_outside_range_after_first_window"] = float(outside.any(1)[fit_rows:].mean())
    info["clipped_share_per_feature"] = outside[fit_rows:].mean(0).round(4).tolist()   # measured before clipping
    info["fit_rows_exceed_tune_rows"] = bool(fit_rows > C.TUNE_ROWS)
    transform_spec = {
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "n_components": n_components,
        "pca_mean": pca_mean,
        "pca_components": pca_comp,
        "lo": lo.tolist(),
        "hi": hi.tolist(),
        "margin": float(margin),
        "clip_min": 0.0,
        "clip_max": float(np.pi),
    }
    return np.clip(S, 0, np.pi), info, transform_spec


def save(name, F, y, meta, transform_spec=None):
    df = pd.DataFrame(F, columns=[f"f{i}" for i in range(F.shape[1])]); df["y"] = np.asarray(y)
    df.to_parquet(PROC / f"{name}_stream.parquet", index=False)
    meta["n_rows"] = len(df); meta["n_features"] = F.shape[1]
    (PROC / f"{name}_stream.meta.json").write_text(json.dumps(meta, indent=2))
    if transform_spec is not None:
        transform_spec["task"] = name
        transform_spec["n_features_out"] = int(F.shape[1])
        (PROC / f"{name}_transform.json").write_text(json.dumps(transform_spec, indent=2))
    print(f"{name}: {df.shape} -> {PROC / (name + '_stream.parquet')}")


def t1(a):
    df = pd.read_csv(a.csv or first_existing(DEF_T1), encoding="latin1")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values(["Date", "Hour"]).reset_index(drop=True)
    y = df["Rented Bike Count"].astype(float).to_numpy()
    X = df.drop(columns=["Date", "Rented Bike Count"])
    X = pd.get_dummies(X, columns=X.select_dtypes(exclude="number").columns.tolist(), dtype=float)   # pandas 3: strings are not dtype object
    F, info, tf = fit_scale(X.to_numpy(), a.fit_rows, a.n_components, margin=a.margin)
    # DEVIATION (flag to B): target scaled GLOBALLY to [0,1] like P2, because a variational
    # regressor cannot output values outside its bounded range. Features stay leak-free.
    y_min, y_max = float(y.min()), float(y.max())
    yy = (y - y_min) / (y_max - y_min)
    tf["feature_columns"] = list(X.columns)
    tf["target_scaling"] = {"mode": "global_minmax", "y_min": y_min, "y_max": y_max}
    save("T1", F, yy, {"task": "regression", "source": "Seoul Bike (UCI)", "target_scaling": "global_minmax", **info}, tf)


def t2(a):
    p = Path(a.file) if a.file else first_existing([DEF_T2])
    if p.suffix == ".npz":
        m = np.load(p, allow_pickle=True); X, y = m["X"].astype(float), m["y"].ravel().astype(int)
    elif p.suffix == ".mat":
        from scipy.io import loadmat
        m = loadmat(p); X, y = m["X"].astype(float), m["y"].ravel().astype(int)
    else:
        d = pd.read_csv(p); y = d.iloc[:, -1].to_numpy().astype(int); X = d.iloc[:, :-1].to_numpy(float)
    rng = np.random.default_rng(a.order_seed); order = rng.permutation(len(X)); X, y = X[order], y[order]
    if a.n_rows: X, y = X[: a.n_rows], y[: a.n_rows]
    # No reordering: anomalies stay spread through the stream (incl. the tuning prefix).
    # Semisupervised: only NORMAL samples inside the first fit_rows rows are used to fit scaler/PCA.
    F, info, tf = fit_scale(X, a.fit_rows, a.n_components, fit_mask=(y == 0), margin=a.margin)
    tf["order_seed"] = int(a.order_seed)
    save("T2", F, y, {"task": "anomaly", "label": "1=anomaly", "source": str(p.name),
                      "anomaly_rate": float(y.mean()), "anomalies_in_tune_prefix": int(y[:C.TUNE_ROWS].sum()),
                      "order_seed": a.order_seed, **info}, tf)


def t3(a):
    if a.source == "local":
        a.csv = a.csv or str(first_existing([DEF_T3])); a.source = "csv"
    if a.source == "elec2":
        from river import datasets
        rows = [(x, int(yy)) for x, yy in datasets.Elec2()]
    elif a.source == "synth":
        from river.datasets import synth
        rows = [(x, int(yy)) for _, (x, yy) in zip(range(a.n_rows), synth.ConceptDriftStream(
            synth.SEA(seed=1, variant=0), synth.SEA(seed=2, variant=3), position=a.n_rows // 2, width=500, seed=3))]
    else:
        d = pd.read_csv(a.csv); lab = d.columns[-1]
        y_ = d[lab].map(lambda v: 1 if str(v).upper() in ("UP", "1", "TRUE") else 0)
        rows = [(r, int(v)) for r, v in zip(d.drop(columns=[lab]).to_dict("records"), y_)]
    rows = rows[: a.n_rows] if a.n_rows else rows
    D = pd.DataFrame([r for r, _ in rows]).select_dtypes("number")
    dropped = [c for c in a.drop if c in D.columns]      # 'date' is a time index: a trend, not a signal
    D = D.drop(columns=dropped)
    X = D.to_numpy(float); y = np.array([v for _, v in rows])
    F, info, tf = fit_scale(X, a.fit_rows, a.n_components, margin=a.margin)
    tf["feature_columns"] = list(D.columns)
    tf["dropped_columns"] = dropped
    save("T3", F, y, {"task": "classification", "source": a.source, "natural_drift": True,
                      "dropped_columns": dropped, "columns_used": list(D.columns), **info}, tf)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    for n, fn in [("t1", t1), ("t2", t2), ("t3", t3)]:
        s = sp.add_parser(n); s.set_defaults(fn=fn)
        s.add_argument("--csv"); s.add_argument("--file"); s.add_argument("--source", default="local")
        s.add_argument("--fit-rows", type=int, default=C.STREAM_FIT_ROWS); s.add_argument("--margin", type=float, default=0.0); s.add_argument("--n-components", type=int,
                       default={"t1": 7, "t2": 5, "t3": 4}[n]); s.add_argument("--n-rows", type=int, default=10000)
        s.add_argument("--order-seed", type=int, default=0); s.add_argument("--drop", nargs="*", default=["date"], help="t3: columns to drop (default: date, a monotone time index)")
    a = ap.parse_args(); a.fn(a)
