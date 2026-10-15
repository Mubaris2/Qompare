import numpy as np, pandas as pd, pytest
from common.base import Model
from common.harness import run_stream, COLUMNS
from common.drift import inject_drift
from common import metrics as M
from common.validate_results import validate


class Spy(Model):
    name = "spy"
    def __init__(self): super().__init__(); self.trains, self.tests = [], []
    def fit(self, X, y=None): self.trains.append(np.asarray(X).copy())
    def predict(self, X): self.tests.append(np.asarray(X).copy()); return np.zeros(len(X))
    def score_anomaly(self, X): self.tests.append(np.asarray(X).copy()); return np.zeros(len(X))


def stream(n=300, d=3, seed=0):
    r = np.random.default_rng(seed); X = r.uniform(0, 3, (n, d)); return X, r.uniform(0, 1, n)


def test_windows_do_not_overlap_and_step_count():
    X, y = stream(); X[:, 0] = np.arange(len(X))          # row id in col 0
    m = Spy(); df = run_stream(m, X, y, "T1", W=50, H=10)
    assert len(m.trains) == (300 - 50) // 10 == len(m.tests)
    for tr, te in zip(m.trains, m.tests):
        assert len(tr) == 50 and len(te) == 10
        assert tr[:, 0].max() < te[:, 0].min()            # train strictly before test
    assert m.trains[1][0, 0] == 10                        # slide by H
    assert list(df.columns) == COLUMNS


def test_t2_trains_on_normal_only():
    X, _ = stream(); y = (np.arange(300) % 10 == 0).astype(int); m = Spy()
    run_stream(m, X, y, "T2", W=100, H=20)
    assert all(len(t) == 90 for t in m.trains)            # 10% anomalies removed from train


def test_drift_abrupt_and_gradual():
    X = np.random.default_rng(1).uniform(0.5, 2.5, (1000, 4)); X0 = X.copy()
    Xa, s = inject_drift(X, "abrupt", 0.5, magnitude=1.0)
    assert s == 500 and np.allclose(Xa[:500], X0[:500]) and not np.allclose(Xa[500:], X0[500:])
    assert np.array_equal(X, X0)                          # input untouched
    Xg, s = inject_drift(X, "gradual", 0.4, 0.2, magnitude=1.0, clip=None)
    d = np.abs(Xg - X0).mean(1)
    assert d[:400].max() == 0 and d[500] > d[420] and np.isclose(d[700], d[900])


def test_metrics():
    assert np.isnan(M.pooled_auc([0, 0, 0], [.1, .2, .3]))
    assert M.pooled_auc([0, 0, 1, 1], [.1, .2, .8, .9]) == 1.0
    n = np.arange(100, 1100, 100); v = np.array([.9]*5 + [.5, .6, .8, .89, .9])
    pre, worst, rec = M.recovery_time(n, v, drift_at=500)
    assert np.isclose(pre, .9) and np.isclose(worst, .5) and rec == 400


def test_validator(tmp_path):
    X, y = stream(); df = run_stream(Spy(), X, y, "T1", W=50, H=10)
    p = tmp_path / "ok.csv"; df.to_csv(p, index=False); assert validate(p) == []
    pd.concat([df, df]).to_csv(p, index=False); assert any("duplicate" in e for e in validate(p))
    df.drop(columns=["seed"]).to_csv(p, index=False); assert any("missing" in e for e in validate(p))


def test_kappa():
    assert np.isclose(M.kappa_plus(0.85, 0.85), 0.0)
    assert np.isclose(M.kappa_plus(1.0, 0.8), 1.0)
    assert np.isclose(M.kappa_plus(0.7, 0.8), -0.5)
    assert np.isnan(M.kappa_plus(0.9, 1.0))
    assert np.isclose(M.nochange_accuracy([0, 0, 1, 1], [0, 1, 1, 0]), 0.5)


def test_t3_harness_has_kappa_rows():
    from common.harness import METRICS_BY_TASK
    X, _ = stream(200); y = (np.arange(200) // 10 % 2).astype(int)       # label flips every 10 samples
    m = Spy(); m.predict = lambda Xt: np.zeros(len(Xt), dtype=int)
    df = run_stream(m, X, y, "T3", W=50, H=10)
    assert set(df.metric_name) == set(METRICS_BY_TASK["T3"])
    nc = df[df.metric_name == "acc_nochange"].metric_value
    assert np.allclose(nc, 0.9)                                          # only 1 of 10 labels differs from its predecessor
