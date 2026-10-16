"""Sanity report for data/processed/*. Run after every prep:  python data/check_streams.py
Flags: preprocessing statistics fitted on test rows, empty tuning prefix, anomaly-free stretches, heavy clipping."""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from common import config as C
from classical.data_io import fingerprint

ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", default=str(ROOT / "data" / "processed")); a = ap.parse_args()
warn = 0


def W(msg):
    global warn; warn += 1; print("   WARNING:", msg)


for t in ("T1", "T2", "T3"):
    f = Path(a.data_dir) / f"{t}_stream.parquet"
    if not f.exists():
        continue
    d = pd.read_parquet(f); X = d.filter(regex=r"^f\d+$").to_numpy(); y = d["y"].to_numpy()
    mp = f.with_suffix("").with_suffix(".meta.json"); mp = Path(str(f).replace(".parquet", ".meta.json"))
    m = json.loads(mp.read_text()) if mp.exists() else {}
    n, P, S = len(d), C.TUNE_ROWS, C.STREAM_START
    print(f"\n{t}: {n} rows x {X.shape[1]} features | source {m.get('source', '?')} | fit_rows {m.get('fit_rows', '?')} margin {m.get('margin', '?')}"
          f" | frac_outside {m.get('frac_outside_range_after_first_window', float('nan')):.3f} | mock={m.get('mock', False)}")
    if isinstance(m.get("fit_rows"), int) and m["fit_rows"] > P:
        W(f"scaler/PCA fitted on {m['fit_rows']} rows but test data starts at {S}: statistics from the test stream leak into preprocessing. Disclose or use --fit-rows {P}.")
    test = X[S:]; at_edge = ((test <= 1e-9) | (test >= np.pi - 1e-9)).mean(0)
    cs = np.array(m["clipped_share_per_feature"]) if "clipped_share_per_feature" in m else None
    print(f"   data fingerprint {fingerprint(X, y)} (A and B must see the same value)")
    if cs is not None:
        print("   clipped share per feature (values pushed outside [0, pi] by the stream):", np.round(cs, 3).tolist())
        ties = np.clip(at_edge - cs, 0, None)
        if ties.max() > 0.2: print("   note: >20% of test values sit exactly on an edge for a feature without being clipped (ties at the fit-window min/max, a heavily repeated value). Not an error.")
        if cs.max() > 0.2: W("a feature is clipped for >20% of the test stream (non-stationarity or too narrow fit window; try --margin).")
    else:
        print("   share of test values at the edge per feature (clipped OR tied; re-run prep for exact clipped share):", np.round(at_edge, 3).tolist())
    if t == "T1":
        q = np.array_split(y, 4)
        print("   target mean/std prefix:", round(y[:P].mean(), 3), round(y[:P].std(), 3), "| rest:", round(y[P:].mean(), 3), round(y[P:].std(), 3),
              "| by quarter mean:", [round(float(c.mean()), 3) for c in q])
        print("   naive-mean MSE ~ target variance: prefix", round(float(y[:P].var()), 5), "rest", round(float(y[P:].var()), 5))
    if t == "T2":
        ch = [int(c.sum()) for c in np.array_split(y, 10)]
        print(f"   anomalies: total {int(y.sum())} ({y.mean():.3%}) | tune prefix {int(y[:P].sum())} | tests region {int(y[S:].sum())} | per 10 chunks {ch}")
        if y[:P].sum() < 10: W("fewer than 10 anomalies in the tuning prefix: tuning AUC is very noisy (or NaN if zero).")
        if min(ch) == 0: W("a tenth of the stream has no anomalies: stream is not spread evenly.")
    if t == "T3":
        print("   class balance prefix:", round(float(y[:P].mean()), 3), "rest:", round(float(y[P:].mean()), 3),
              "| persistence accuracy (y[t]==y[t-1]):", round(float((y[1:] == y[:-1]).mean()), 3), "| majority acc:", round(float(max(y.mean(), 1 - y.mean())), 3))
print("\n" + ("no warnings" if not warn else f"{warn} warning(s)"))
sys.exit(1 if warn else 0)
