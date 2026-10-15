"""Build data/processed/T{1,2,3}_stream.parquet from raw files.
Contract: features f0..f{d-1} scaled to [0, pi] with scaler/PCA FIT ON FIRST `fit_rows` ONLY
(no leakage, honest streaming). Output order is final and shared with Member B.

Usage:
  python data/download.py                      # once: fills data/raw/
  python data/prep.py t1                       # Seoul Bike  (default data/raw/seoul_bike/)
  python data/prep.py t2 [--file data/raw/ad/6_cardio.npz] [--n-components 5] [--n-rows 10000]
  python data/prep.py t3 [--n-rows 10000]      # ELEC2 from data/raw/elec2/ (also --source synth|elec2|csv)
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"; PROC.mkdir(exist_ok=True, parents=True)
DEF_T1 = [RAW / "seoul_bike" / "SeoulBikeData.csv", RAW / "SeoulBikeData.csv"]   # 2nd = manual download
DEF_T2 = RAW / "ad" / "32_shuttle.npz"
DEF_T3 = RAW / "elec2" / "electricity.csv"


def first_existing(paths):
    for q in paths:
        if Path(q).exists():
            return Path(q)
    raise SystemExit("raw file not found. Run: python data/download.py   (looked in: "
                     + ", ".join(str(Path(q).relative_to(ROOT)) for q in paths) + ")")


def fit_scale(X, fit_rows, n_components=None):
    """PCA (optional) + MinMax to [0,pi], both fit on first fit_rows only."""
    X = np.asarray(X, float)
    mu, sd = X[:fit_rows].mean(0), X[:fit_rows].std(0) + 1e-12
    Z = (X - mu) / sd
    info = {"fit_rows": fit_rows}
    if n_components:
        p = PCA(n_components=n_components, random_state=0).fit(Z[:fit_rows])
        Z = p.transform(Z); info["pca_explained_variance_ratio"] = p.explained_variance_ratio_.round(4).tolist()
    lo, hi = Z[:fit_rows].min(0), Z[:fit_rows].max(0)
    S = np.pi * (Z - lo) / np.where(hi - lo == 0, 1, hi - lo)
    info["frac_outside_range_after_first_window"] = float(((S < 0) | (S > np.pi)).any(1)[fit_rows:].mean())
    return np.clip(S, 0, np.pi), info


def save(name, F, y, meta):
    df = pd.DataFrame(F, columns=[f"f{i}" for i in range(F.shape[1])]); df["y"] = np.asarray(y)
    df.to_parquet(PROC / f"{name}_stream.parquet", index=False)
    meta["n_rows"] = len(df); meta["n_features"] = F.shape[1]
    (PROC / f"{name}_stream.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{name}: {df.shape} -> {PROC / (name + '_stream.parquet')}")


def t1(a):
    df = pd.read_csv(a.csv or first_existing(DEF_T1), encoding="latin1")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values(["Date", "Hour"]).reset_index(drop=True)
    y = df["Rented Bike Count"].astype(float).to_numpy()
    X = df.drop(columns=["Date", "Rented Bike Count"])
    X = pd.get_dummies(X, columns=X.select_dtypes(exclude="number").columns.tolist(), dtype=float)   # pandas 3: strings are not dtype object
    F, info = fit_scale(X.to_numpy(), a.fit_rows, a.n_components)
    # DEVIATION (flag to B): target scaled GLOBALLY to [0,1] like P2, because a variational
    # regressor cannot output values outside its bounded range. Features stay leak-free.
    yy = (y - y.min()) / (y.max() - y.min())
    save("T1", F, yy, {"task": "regression", "source": "Seoul Bike (UCI)", "target_scaling": "global_minmax", **info})


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
    # first window must be anomaly-free so the scaler sees normal data only
    first_norm = np.where(y == 0)[0][: a.fit_rows]; rest = np.setdiff1d(np.arange(len(X)), first_norm)
    order = np.concatenate([first_norm, rest]); X, y = X[order], y[order]
    F, info = fit_scale(X, a.fit_rows, a.n_components)
    save("T2", F, y, {"task": "anomaly", "label": "1=anomaly", "source": str(p.name),
                      "anomaly_rate": float(y.mean()), "order_seed": a.order_seed, **info})


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
    X = pd.DataFrame([r for r, _ in rows]).select_dtypes("number").to_numpy(float); y = np.array([v for _, v in rows])
    F, info = fit_scale(X, a.fit_rows, a.n_components)
    save("T3", F, y, {"task": "classification", "source": a.source, "natural_drift": True, **info})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    for n, fn in [("t1", t1), ("t2", t2), ("t3", t3)]:
        s = sp.add_parser(n); s.set_defaults(fn=fn)
        s.add_argument("--csv"); s.add_argument("--file"); s.add_argument("--source", default="local")
        s.add_argument("--fit-rows", type=int, default=400); s.add_argument("--n-components", type=int,
                       default={"t1": 7, "t2": 5, "t3": 4}[n]); s.add_argument("--n-rows", type=int, default=10000)
        s.add_argument("--order-seed", type=int, default=0)
    a = ap.parse_args(); a.fn(a)
