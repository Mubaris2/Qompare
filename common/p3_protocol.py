"""P3 (Tscharke et al. 2023) evaluation protocol, shared by A (classical) and B (quantum) so both
use IDENTICAL random draws and preprocessing. Do not fork this file.

Per draw r: 30 normal samples for training, 25 normal + 25 anomalous for testing.
Preprocessing: standardise, PCA to 5 components, rescale to [-1, 1]. Fit on the 30 training normals only
(deviation from P3, which preprocessed the whole dataset; ours is leak-free). Test values are clipped to [-1, 1]."""
import numpy as np
from sklearn.decomposition import PCA

N_TRAIN, N_TEST_EACH, N_COMPONENTS = 30, 25, 5


def draw_indices(y, r, n_train=N_TRAIN, n_test=N_TEST_EACH):
    """Return (train_idx, test_normal_idx, test_anomaly_idx) into the ORIGINAL array order. Deterministic in r."""
    y = np.asarray(y).ravel()
    rng = np.random.default_rng(10_000 + r)
    norm, anom = np.where(y == 0)[0], np.where(y == 1)[0]
    pn = rng.permutation(norm)
    if len(pn) < n_train + n_test or len(anom) < n_test:
        raise ValueError("dataset too small for the P3 protocol")
    return pn[:n_train], pn[n_train:n_train + n_test], rng.permutation(anom)[:n_test]


def prepare(X, tr, te_n, te_a, n_components=N_COMPONENTS):
    """Return Xtr, Xte, yte (0 normal, 1 anomaly), all in [-1, 1]."""
    X = np.asarray(X, float)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
    k = min(n_components, len(tr) - 1, X.shape[1])
    pca = PCA(n_components=k, random_state=0).fit((X[tr] - mu) / sd)
    Z = lambda idx: pca.transform((X[idx] - mu) / sd)
    Ztr = Z(tr)
    lo, hi = Ztr.min(0), Ztr.max(0)
    sc = lambda A: np.clip(2 * (A - lo) / np.where(hi - lo == 0, 1, hi - lo) - 1, -1, 1)
    te = np.concatenate([te_n, te_a])
    return sc(Ztr), sc(Z(te)), np.r_[np.zeros(len(te_n), int), np.ones(len(te_a), int)]
