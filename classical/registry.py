"""Model registry + random-search spaces (Member A). Every tunable model gets the SAME budget (config.N_TRIALS)."""
from functools import partial
import numpy as np
from . import models as m


def _logu(r, lo, hi): return float(np.exp(r.uniform(np.log(lo), np.log(hi))))
def _ch(r, opts): return opts[int(r.integers(len(opts)))]
def _int(r, lo, hi): return int(r.integers(lo, hi + 1))


SPACES = {
    "svr": lambda r: dict(C=_logu(r, .1, 100), epsilon=_logu(r, .005, .2), gamma=_logu(r, .01, 10)),
    "rf": lambda r: dict(n_estimators=_ch(r, [50, 100, 200]), max_depth=_ch(r, [4, 8, 12, None]),
                         min_samples_leaf=_ch(r, [1, 2, 5])),
    "ht_reg": lambda r: dict(grace_period=_int(r, 20, 200), delta=_logu(r, 1e-7, 1e-2),
                             leaf_prediction=_ch(r, ["mean", "model", "adaptive"])),
    "ocsvm": lambda r: dict(nu=_logu(r, .01, .3), gamma=_logu(r, .01, 10)),
    "ae": lambda r: dict(alpha=_logu(r, 1e-5, 1e-1), learning_rate_init=_logu(r, 1e-3, 1e-1), max_iter=_ch(r, [200, 500])),
    "ht_clf": lambda r: dict(grace_period=_int(r, 20, 200), delta=_logu(r, 1e-7, 1e-2),
                             split_criterion=_ch(r, ["info_gain", "gini", "hellinger"]), leaf_prediction=_ch(r, ["mc", "nb", "nba"])),
    "tree": lambda r: dict(max_depth=_ch(r, [3, 5, 8, 12, None]), min_samples_leaf=_ch(r, [1, 2, 5, 10])),
}

# task -> model name -> (factory, space or None).  Names equal Model.name (used in result file names).
REGISTRY = {
    "T1": {"svr_rbf": (m.SVRRegressor, "svr"), "random_forest": (m.RFRegressor, "rf"),
           "river_ht_reg": (partial(m.RiverHT, task="reg"), "ht_reg"),
           "naive_mean": (partial(m.NaiveRegressor, mode="mean"), None), "naive_last": (partial(m.NaiveRegressor, mode="last"), None)},
    "T2": {"svr_recon_rbf": (m.SVRReconstructionAnomaly, "svr"), "ae": (m.AutoencoderAnomaly, "ae"),
           "ocsvm_rbf": (m.OCSVMAnomaly, "ocsvm")},
    "T3": {"river_ht": (m.RiverHT, "ht_clf"), "river_hat": (partial(m.RiverHT, adaptive=True), "ht_clf"),
           "batch_tree": (m.BatchTreeClassifier, "tree"), "river_gaussian_nb": (m.RiverNB, None),
           "naive_majority": (partial(m.NaiveClassifier, mode="majority"), None),
           "naive_last": (partial(m.NaiveClassifier, mode="last"), None)},
}


def build(task, name, params=None):
    return REGISTRY[task][name][0](**(params or {}))


def sample(task, name, rng):
    sp = REGISTRY[task][name][1]
    return SPACES[sp](rng) if sp else {}


def tunable(task):
    return [n for n, (_, sp) in REGISTRY[task].items() if sp]
