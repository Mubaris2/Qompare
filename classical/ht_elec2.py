"""Reproduce P1 (Domingos & Hulten 2000): Hoeffding Tree vs batch tree on a drifting stream.
Prequential (test-then-train), results in 500-sample blocks, shared CSV schema.

  python data/download.py                                # once: puts ELEC2 in data/raw/elec2/
  python -m classical.ht_elec2                           # default: --source local (data/raw/elec2/electricity.csv)
  python -m classical.ht_elec2 --source synth            # offline smoke test
  python -m classical.ht_elec2 --source elec2            # River's own copy (~/river_data), for cross-checking only
"""
import argparse, json, time, tracemalloc
from pathlib import Path
import numpy as np, pandas as pd
from river import tree, naive_bayes, dummy
from sklearn.tree import DecisionTreeClassifier
from common.harness import COLUMNS
from common import metrics as M

ROOT = Path(__file__).resolve().parents[1]
LOCAL_CSV = ROOT / "data" / "raw" / "elec2" / "electricity.csv"
OUT = ROOT / "results" / "classical"; OUT.mkdir(parents=True, exist_ok=True)


def load(a):
    if a.source == "elec2":
        from river import datasets
        data = [(x, int(y)) for x, y in datasets.Elec2()]
    elif a.source == "synth":
        from river.datasets import synth
        g = synth.ConceptDriftStream(synth.SEA(seed=1, variant=0), synth.SEA(seed=2, variant=3),
                                     position=a.n_rows // 2, width=500, seed=3)
        data = [(x, int(y)) for _, (x, y) in zip(range(a.n_rows), g)]
    else:
        if a.source == "local":
            if not LOCAL_CSV.exists():
                raise SystemExit("ELEC2 not found. Run: python data/download.py --only elec2")
            a.csv = str(LOCAL_CSV)
        d = pd.read_csv(a.csv); lab = d.columns[-1]
        yy = d[lab].map(lambda v: 1 if str(v).upper() in ("UP", "1", "TRUE") else 0)
        data = [(r, int(v)) for r, v in zip(d.drop(columns=[lab]).to_dict("records"), yy)]
    return data[: a.n_rows] if a.n_rows else data


def run_river(name, model, data, B):
    rows, hit, t_pred, t_fit, summ = [], 0, 0.0, 0.0, {}
    tracemalloc.start(); t_all = time.perf_counter()
    for i, (x, y) in enumerate(data):
        t0 = time.perf_counter(); p = model.predict_one(x); t_pred += time.perf_counter() - t0
        hit += int(p == y)
        t0 = time.perf_counter(); model.learn_one(x, y); t_fit += time.perf_counter() - t0
        if (i + 1) % B == 0:
            rows.append((name, (i + 1) // B - 1, i + 1, hit / B, t_fit, t_pred,
                         getattr(model, "n_nodes", 0) or 0)); hit, t_pred, t_fit = 0, 0.0, 0.0
    total = time.perf_counter() - t_all; peak = tracemalloc.get_traced_memory()[1] / 1e6; tracemalloc.stop()
    acc = float(np.mean([r[3] for r in rows])); nn = [r[6] for r in rows if r[6]]
    return rows, dict(model=name, mean_block_accuracy=acc, throughput_per_s=len(data) / total,
                      seconds=total, peak_mem_mb=peak, final_nodes=getattr(model, "n_nodes", None),
                      mean_nodes=float(np.mean(nn)) if nn else None)


def run_batch(data, B, retrain_every):
    """Batch reference: retrain sklearn tree on ALL data seen so far, test on next block."""
    keys = sorted(data[0][0]); X = np.array([[x[k] for k in keys] for x, _ in data]); y = np.array([v for _, v in data])
    rows, t_all = [], time.perf_counter(); fit_t = 0.0
    for s in range(retrain_every, len(X) - B + 1, B):
        t0 = time.perf_counter(); m = DecisionTreeClassifier(random_state=0).fit(X[:s], y[:s]); ft = time.perf_counter() - t0
        t0 = time.perf_counter(); p = m.predict(X[s:s + B]); pt = time.perf_counter() - t0
        rows.append(("batch_tree_retrain_all", s // B - 1, s + B, float((p == y[s:s + B]).mean()), ft, pt, int(m.tree_.node_count)))
    tot = time.perf_counter() - t_all
    return rows, dict(model="batch_tree_retrain_all", mean_block_accuracy=float(np.mean([r[3] for r in rows])),
                      throughput_per_s=len(data) / tot, seconds=tot, final_nodes=rows[-1][6],
                      mean_nodes=float(np.mean([r[6] for r in rows])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="local", choices=["local", "elec2", "synth", "csv"]); ap.add_argument("--csv"); ap.add_argument("--n-rows", type=int, default=0)
    ap.add_argument("--block", type=int, default=500); ap.add_argument("--tag", default=None)
    a = ap.parse_args(); data = load(a); B = a.block; tag = a.tag or a.source
    models = {"river_ht": tree.HoeffdingTreeClassifier(), "river_hat": tree.HoeffdingAdaptiveTreeClassifier(seed=0),
              "river_gaussian_nb": naive_bayes.GaussianNB(),
              "no_change(persistence)": dummy.NoChangeClassifier(), "majority_prior": dummy.PriorClassifier()}
    all_rows, summaries = [], []
    for n, m in models.items():
        r, s = run_river(n, m, data, B); all_rows += r; summaries.append(s)
    r, s = run_batch(data, B, retrain_every=B); all_rows += r; summaries.append(s)
    recs = [dict(task="T3", model=n, variant="", seed=0, W=0, H=B, drift="natural", step=st, n_seen=ns,
                 metric_name="prequential_acc", metric_value=acc, t_encode=0.0, t_fit=tf, t_predict=tp,
                 n_params=nn, n_qubits=0, n_circuit_evals=0, peak_mem_mb=np.nan, drift_flag=0)
            for n, st, ns, acc, tf, tp, nn in all_rows]
    # Kappa+ per block, reference = no-change (persistence) model from THIS run, same blocks
    nc_name = "no_change(persistence)"
    nc = {st: acc for n, st, ns, acc, tf, tp, nn in all_rows if n == nc_name}
    kp = [dict(r, metric_name="kappa_plus", metric_value=M.kappa_plus(r["metric_value"], nc[r["step"]]))
          for r in recs if r["model"] != nc_name and r["step"] in nc]
    recs += kp
    pooled_nc = next(s["mean_block_accuracy"] for s in summaries if s["model"] == nc_name)
    pooled_maj = next(s["mean_block_accuracy"] for s in summaries if s["model"] == "majority_prior")
    for s in summaries:                      # pooled over the whole stream (use these in the report)
        s["kappa_plus"] = M.kappa_plus(s["mean_block_accuracy"], pooled_nc)
        s["kappa_m"] = M.kappa_m(s["mean_block_accuracy"], pooled_maj)
    pd.DataFrame(recs, columns=COLUMNS).to_csv(OUT / f"T3_ht_prequential_{tag}.csv", index=False)
    (OUT / f"T3_ht_summary_{tag}.json").write_text(json.dumps(summaries, indent=2, default=float))
    cols = ["model", "mean_block_accuracy", "kappa_plus", "kappa_m", "throughput_per_s", "mean_nodes", "final_nodes", "peak_mem_mb"]
    print(pd.DataFrame(summaries)[cols].round(4).to_string(index=False))
