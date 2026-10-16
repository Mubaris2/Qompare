"""Paper-parity runs in BATCH mode (Member A). These give the quantum side (Member B) classical reference numbers
on identical splits and draws, and let us check our classical baselines against what the papers report.

P2 parity (Herbst et al.): T1, first 400 rows train / next 250 rows test (contiguous). MSE, MAE.
P3 parity (Tscharke et al.): ADBench dataset, P3 protocol via common/p3_protocol.py (30 normal train, 25+25 test,
  PCA-5 scaled to [-1,1], threshold = 3 x mean train loss), repeated over R seeded draws. AUC, precision, recall, F1, accuracy.

  python -m classical.batch_parity p2
  python -m classical.batch_parity p3 --dataset 23_mammography --draws 50
Output: parity/classical_parity.csv (appended by key) and parity/p3_splits_<dataset>.npz (draws for Member B)."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from common import metrics as M
from common.p3_protocol import draw_indices, prepare
from . import registry as R
from .data_io import load_task, ROOT, PROC

PARITY = ROOT / "parity"
# values the P3 paper reports for its own data (Table II): used only as a sanity reference
P3_REPORTED_AUC = {"23_mammography": {"svr_recon_rbf": 0.94, "ae": 0.74, "qsvr(paper)": 0.96, "qae(paper)": 0.22}}


def _best(task, name, best_json):
    bj = Path(best_json or ROOT / "configs" / "best_params_classical.json")
    return json.loads(bj.read_text()).get(task, {}).get(name, {}).get("params") if bj.exists() else None


def p2(data_dir=PROC, n_train=400, n_test=250, best_json=None):
    X, y, _ = load_task("T1", data_dir)
    Xtr, ytr, Xte, yte = X[:n_train], y[:n_train], X[n_train:n_train + n_test], y[n_train:n_train + n_test]
    rows = []
    for name in R.REGISTRY["T1"]:
        for setting, params in (("default", {}), ("tuned", _best("T1", name, best_json))):
            if params is None or (setting == "tuned" and R.REGISTRY["T1"][name][1] is None):
                continue
            mod = R.build("T1", name, params); mod.set_seed(0); mod.fit(Xtr, ytr); p = mod.predict(Xte)
            for met, fn in (("mse", M.mse), ("mae", M.mae)):
                rows.append(dict(paper="P2", dataset="T1 (Seoul Bike)", model=name, setting=setting, metric=met,
                                 mean=fn(yte, p), std=np.nan, n=1, note=f"train rows 0..{n_train - 1}, test next {n_test}"))
    return pd.DataFrame(rows)


def _load_npz(dataset):
    d = np.load(ROOT / "data" / "raw" / "ad" / f"{dataset}.npz", allow_pickle=True)
    return d["X"].astype(float), d["y"].ravel().astype(int)


def p3(dataset="23_mammography", draws=50, X=None, y=None, save_splits=True):
    if X is None:
        X, y = _load_npz(dataset)
    models = {"svr_recon_rbf": R.build("T2", "svr_recon_rbf", {}), "ae": R.build("T2", "ae", {}), "ocsvm_rbf": R.build("T2", "ocsvm_rbf", {})}
    acc = {k: {m: [] for m in ("auc", "precision", "recall", "f1", "accuracy")} for k in models}
    splits = {"train": [], "test_normal": [], "test_anomaly": []}
    for r in range(draws):
        tr, tn, ta = draw_indices(y, r)
        Xtr, Xte, yte = prepare(X, tr, tn, ta)
        for k, mod in models.items():
            mod.set_seed(r); mod.fit(Xtr)
            s_te, s_tr = mod.score_anomaly(Xte), mod.score_anomaly(Xtr)
            tau = 3.0 * float(np.mean(s_tr)); pred = (s_te > tau).astype(int)
            acc[k]["auc"].append(roc_auc_score(yte, s_te))
            acc[k]["precision"].append(precision_score(yte, pred, zero_division=np.nan))
            acc[k]["recall"].append(recall_score(yte, pred, zero_division=0))
            acc[k]["f1"].append(f1_score(yte, pred, zero_division=np.nan))
            acc[k]["accuracy"].append(accuracy_score(yte, pred))
        for key, v in zip(splits, (tr, tn, ta)):
            splits[key].append(v)
    if save_splits:
        PARITY.mkdir(exist_ok=True)
        np.savez(PARITY / f"p3_splits_{dataset}.npz", **{k: np.array(v) for k, v in splits.items()})
    rows = []
    for k, d in acc.items():
        for met, vals in d.items():
            v = np.array(vals, float)
            rows.append(dict(paper="P3", dataset=dataset, model=k, setting=f"default, {draws} draws", metric=met,
                             mean=float(np.nanmean(v)) if np.isfinite(v).any() else np.nan,
                             std=float(np.nanstd(v)) if np.isfinite(v).any() else np.nan, n=int(np.isfinite(v).sum()),
                             note=("paper reports AUC %.2f" % P3_REPORTED_AUC[dataset][k]) if (met == "auc" and k in P3_REPORTED_AUC.get(dataset, {})) else ""))
    return pd.DataFrame(rows)


def save(df):
    PARITY.mkdir(exist_ok=True); f = PARITY / "classical_parity.csv"
    old = pd.read_csv(f) if f.exists() else pd.DataFrame()
    keys = ["paper", "dataset", "model", "setting", "metric"]
    if len(old):
        old = old.merge(df[keys].drop_duplicates(), on=keys, how="left", indicator=True)
        old = old[old["_merge"] == "left_only"].drop(columns="_merge")
    pd.concat([old, df], ignore_index=True).to_csv(f, index=False); return f


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("p2"); s = sp.add_parser("p3"); s.add_argument("--dataset", default="23_mammography"); s.add_argument("--draws", type=int, default=50)
    a = ap.parse_args()
    df = p2() if a.cmd == "p2" else p3(a.dataset, a.draws)
    f = save(df); print(df.round(4).to_string(index=False)); print("->", f)
