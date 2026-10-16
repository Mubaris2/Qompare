import json
import numpy as np, pandas as pd, pytest
from sklearn.metrics import roc_auc_score
from common import config as C
from common.base import Model
from common.harness import run_stream
from common.p3_protocol import draw_indices, prepare
from common.validate_results import validate
from classical import registry as R
from classical.models import SVRReconstructionAnomaly, AutoencoderAnomaly


@pytest.fixture
def mock_dir(tmp_path):
    r = np.random.default_rng(0); d = tmp_path / "proc"; d.mkdir()

    def save(name, X, y, meta):
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]); df["y"] = y
        df.to_parquet(d / f"{name}_stream.parquet", index=False); (d / f"{name}_stream.meta.json").write_text(json.dumps(meta))
    n = 1600; X = r.uniform(0, np.pi, (n, 5)); save("T1", X, np.clip(.5 + .2 * np.sin(X[:, 0]) + r.normal(0, .02, n), 0, 1), {})
    X = np.clip(r.normal(np.pi / 2, .4, (n, 4)), 0, np.pi); y = np.zeros(n, int)
    idx = r.choice(np.arange(500, n), 100, replace=False); X[idx] = np.clip(r.normal(np.pi / 2 + 1.2, .4, (100, 4)), 0, np.pi); y[idx] = 1
    save("T2", X, y, {})
    X = r.uniform(0, np.pi, (n, 4)); save("T3", X, (X[:, 0] + X[:, 1] > np.pi).astype(int), {"natural_drift": True})
    return d


def test_harness_start_offset():
    class Spy(Model):
        name = "spy"
        def __init__(self): super().__init__(); self.tr, self.te = [], []
        def fit(self, X, y=None): self.tr.append(X[:, 0].copy())
        def predict(self, X): self.te.append(X[:, 0].copy()); return np.zeros(len(X))
    X = np.zeros((400, 2)); X[:, 0] = np.arange(400); m = Spy()
    df = run_stream(m, X, np.zeros(400), "T1", W=50, H=10, start=100)
    assert m.te[0][0] == 100 and m.tr[0][0] == 50 and m.tr[0][-1] == 99
    assert df.n_seen.min() == 110
    with pytest.raises(AssertionError):
        run_stream(Spy(), X, np.zeros(400), "T1", W=50, H=10, start=20)


def test_registry_models_fit_and_predict():
    rng = np.random.default_rng(0); X = rng.uniform(0, 3, (100, 5)); yb = (X[:, 0] > 1.5).astype(int)
    for t in R.REGISTRY:
        for n in R.REGISTRY[t]:
            m = R.build(t, n, R.sample(t, n, rng)); m.fit(X, None if t == "T2" else (X[:, 0] / 3 if t == "T1" else yb))
            out = m.score_anomaly(X[:4]) if t == "T2" else m.predict(X[:4]); assert len(out) == 4 and np.all(np.isfinite(out))


def test_reconstruction_models_separate_anomalies():
    r = np.random.default_rng(1); Xn = r.normal(0, .3, (150, 4)); Xa = r.normal(1.5, .3, (30, 4))
    Xt = np.vstack([r.normal(0, .3, (50, 4)), Xa[:20]]); y = np.r_[np.zeros(50), np.ones(20)]
    for mod in (SVRReconstructionAnomaly(), AutoencoderAnomaly(max_iter=300)):
        mod.fit(Xn); assert roc_auc_score(y, mod.score_anomaly(Xt)) > 0.9 and mod.meta["n_params"] > 0


def test_tune_best_of_budget_and_default_not_selected(mock_dir, tmp_path):
    from classical.tune import tune
    out = tmp_path / "best.json"; csv = tmp_path / "trials.csv"
    best = tune("T1", models=["svr_rbf"], n_trials=3, data_dir=mock_dir, out_json=out, trials_csv=csv)
    t = pd.read_csv(csv)
    assert (t.trial == -1).sum() == 1 and (t.trial >= 0).sum() == 3         # default is reference only
    rnd = t[t.trial >= 0]
    assert np.isclose(best["T1"]["svr_rbf"]["objective"], rnd.objective.min())   # T1 minimises MSE
    assert json.loads(out.read_text())["T1"]["svr_rbf"]["n_trials"] == 3


def test_run_experiments_smoke_and_validate(mock_dir, tmp_path):
    from classical.run_experiments import run
    files = run(["T1", "T2", "T3"], models=["svr_rbf", "ae", "river_ht", "naive_last"], Ws=[100], seeds=[0, 1],
                drifts=["none", "abrupt"], max_steps=4, data_dir=mock_dir, out_dir=tmp_path / "res",
                best_json=tmp_path / "missing.json")
    assert len(files) == 5 - 0 or len(files) >= 4
    for f in files:
        assert validate(f) == []
    d = pd.read_csv(tmp_path / "res" / "T3_river_ht.csv")
    assert set(d.drift) == {"natural", "abrupt"} and d.n_seen.min() >= C.STREAM_START
    assert len(json.loads((tmp_path / "res" / "T3_river_ht.meta.json").read_text())["data_fingerprint"]) == 12


def test_p3_protocol_deterministic_disjoint():
    y = np.r_[np.zeros(500, int), np.ones(60, int)]
    a, b = draw_indices(y, 3), draw_indices(y, 3)
    assert all(np.array_equal(u, v) for u, v in zip(a, b))
    tr, tn, ta = a; assert len(set(tr) & set(tn)) == 0 and y[tr].sum() == 0 and y[tn].sum() == 0 and y[ta].all()
    assert not np.array_equal(draw_indices(y, 4)[0], tr)
    X = np.random.default_rng(0).normal(size=(560, 9)); Xtr, Xte, yte = prepare(X, tr, tn, ta)
    assert Xtr.shape == (30, 5) and Xte.shape == (50, 5) and Xtr.min() >= -1 and Xte.max() <= 1 and yte.sum() == 25


def test_p3_parity_on_synthetic(tmp_path, monkeypatch):
    from classical import batch_parity as bp
    monkeypatch.setattr(bp, "PARITY", tmp_path / "parity")
    r = np.random.default_rng(0); X = np.vstack([r.normal(0, 1, (600, 8)), r.normal(3, 1, (80, 8))]); y = np.r_[np.zeros(600, int), np.ones(80, int)]
    df = bp.p3("synthetic", draws=3, X=X, y=y)
    auc = df[(df.metric == "auc")].set_index("model")["mean"]
    assert auc["svr_recon_rbf"] > 0.8 and (tmp_path / "parity" / "p3_splits_synthetic.npz").exists()
    f = bp.save(df); assert f.exists()


def test_p2_parity_columns(mock_dir):
    from classical.batch_parity import p2
    df = p2(data_dir=mock_dir, best_json=mock_dir / "none.json")
    assert {"svr_rbf", "random_forest", "naive_mean"} <= set(df.model) and set(df.metric) == {"mse", "mae"}
