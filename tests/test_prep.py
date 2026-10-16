import argparse, importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd

spec = importlib.util.spec_from_file_location("prep", Path(__file__).parents[1] / "data" / "prep.py")
prep = importlib.util.module_from_spec(spec); spec.loader.exec_module(prep)


def test_fit_scale_uses_only_fit_rows_and_mask():
    r = np.random.default_rng(0); X = np.vstack([r.normal(0, 1, (400, 3)), r.normal(50, 1, (400, 3))])   # later rows very different
    S, info = prep.fit_scale(X, 400)
    assert S[:400].min() >= 0 and S[:400].max() <= np.pi and info["frac_outside_range_after_first_window"] > 0.99
    assert np.allclose(S[:400].min(0), 0) and np.allclose(S[:400].max(0), np.pi)          # fitted on rows 0..399
    y = np.r_[np.ones(10), np.zeros(790)]; _, i2 = prep.fit_scale(X, 400, fit_mask=(y == 0))
    assert i2["n_fit_samples"] == 390


def test_margin_reduces_clipping():
    r = np.random.default_rng(1); X = r.normal(0, 1, (3000, 4))
    f0 = prep.fit_scale(X, 400)[1]["frac_outside_range_after_first_window"]
    f1 = prep.fit_scale(X, 400, margin=0.5)[1]["frac_outside_range_after_first_window"]
    assert f1 < f0


def test_t2_keeps_anomalies_spread(tmp_path, monkeypatch):
    r = np.random.default_rng(2); X = np.vstack([r.normal(0, 1, (2800, 6)), r.normal(4, 1, (200, 6))]); y = np.r_[np.zeros(2800, int), np.ones(200, int)]
    np.savez(tmp_path / "x.npz", X=X, y=y); monkeypatch.setattr(prep, "PROC", tmp_path)
    prep.t2(argparse.Namespace(file=str(tmp_path / "x.npz"), fit_rows=800, margin=0.0, n_components=5, n_rows=3000, order_seed=0))
    d = pd.read_parquet(tmp_path / "T2_stream.parquet"); m = json.loads((tmp_path / "T2_stream.meta.json").read_text())
    assert d.y[:800].sum() > 20 and min(c.sum() for c in np.array_split(d.y.to_numpy(), 10)) > 5   # not pushed to the end
    assert m["n_fit_samples"] < 800 and m["anomalies_in_tune_prefix"] == int(d.y[:800].sum())


def test_t3_drops_trending_date(tmp_path, monkeypatch):
    r = np.random.default_rng(3); n = 4000
    d = pd.DataFrame(dict(date=np.linspace(0, 1, n), day=np.arange(n) % 7 + 1, period=(np.arange(n) % 48) / 47,
                          nswprice=r.uniform(0, 1, n), nswdemand=r.uniform(0, 1, n), vicprice=r.uniform(0, 1, n)))
    d["class"] = np.where(d.nswdemand > 0.5, "UP", "DOWN"); d.to_csv(tmp_path / "e.csv", index=False)
    monkeypatch.setattr(prep, "PROC", tmp_path)
    mk = lambda drop: argparse.Namespace(source="csv", csv=str(tmp_path / "e.csv"), fit_rows=800, margin=0.0, n_components=4, n_rows=4000, drop=drop)
    prep.t3(mk(["date"])); m1 = json.loads((tmp_path / "T3_stream.meta.json").read_text())
    prep.t3(mk([])); m2 = json.loads((tmp_path / "T3_stream.meta.json").read_text())
    assert m1["dropped_columns"] == ["date"] and "date" not in m1["columns_used"]
    assert m1["frac_outside_range_after_first_window"] < 0.2 < 0.5 < m2["frac_outside_range_after_first_window"]


def test_clipped_share_per_feature_reported():
    r = np.random.default_rng(5); X = np.c_[r.normal(0, 1, 3000), np.r_[r.normal(0, 1, 800), r.normal(6, 1, 2200)]]
    _, info = prep.fit_scale(X, 800)
    cs = info["clipped_share_per_feature"]; assert cs[0] < 0.05 and cs[1] > 0.9
