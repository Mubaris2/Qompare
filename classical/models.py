"""Classical models wrapped in the shared Model interface. Owned by Member A."""
import warnings
import numpy as np
from sklearn.svm import SVR, OneClassSVM
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from river import tree, naive_bayes
from common.base import Model


def _d(row):
    return {i: float(v) for i, v in enumerate(row)}


# ---------------- T1 regression ----------------
class SVRRegressor(Model):
    name = "svr_rbf"

    def __init__(self, C=1.0, epsilon=0.05, gamma="scale"):
        super().__init__(); self.kw = dict(C=C, epsilon=epsilon, gamma=gamma)
        self.variant = f"C={C:.4g},eps={epsilon:.4g},gamma={gamma if isinstance(gamma, str) else format(gamma, '.4g')}"

    def fit(self, X, y=None):
        self.m = SVR(kernel="rbf", **self.kw).fit(X, y)
        self.meta["n_params"] = int(len(self.m.support_))   # support vectors

    def predict(self, X): return self.m.predict(X)


class RFRegressor(Model):
    name = "random_forest"

    def __init__(self, n_estimators=100, max_depth=None, min_samples_leaf=1):
        super().__init__(); self.kw = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        self.variant = f"n={n_estimators},depth={max_depth},leaf={min_samples_leaf}"

    def fit(self, X, y=None):
        self.m = RandomForestRegressor(random_state=getattr(self, "seed", 0), n_jobs=1, **self.kw).fit(X, y)
        self.meta["n_params"] = int(sum(t.tree_.node_count for t in self.m.estimators_))

    def predict(self, X): return self.m.predict(X)


class NaiveRegressor(Model):
    """Reference baselines: window mean, or last label seen in the window."""
    def __init__(self, mode="mean"):
        super().__init__(); self.mode = mode; self.name = f"naive_{mode}"

    def fit(self, X, y=None): self.v = float(np.mean(y) if self.mode == "mean" else y[-1])
    def predict(self, X): return np.full(len(X), self.v)


# ---------------- T2 anomaly detection ----------------
class OCSVMAnomaly(Model):
    name = "ocsvm_rbf"

    def __init__(self, nu=0.05, gamma="scale"):
        super().__init__(); self.kw = dict(nu=nu, gamma=gamma)
        self.variant = f"nu={nu:.4g},gamma={gamma if isinstance(gamma, str) else format(gamma, '.4g')}"

    def fit(self, X, y=None):
        self.m = OneClassSVM(kernel="rbf", **self.kw).fit(X)
        self.meta["n_params"] = int(len(self.m.support_))

    def score_anomaly(self, X): return -self.m.decision_function(X)


class SVRReconstructionAnomaly(Model):
    """P3 classical baseline (CSVR). One SVR per feature, each predicting that feature from the full
    input; anomaly score = summed squared reconstruction error. Same structure as P3's QSVR, RBF kernel."""
    name = "svr_recon_rbf"

    def __init__(self, C=1.0, epsilon=0.1, gamma="scale"):
        super().__init__(); self.kw = dict(C=C, epsilon=epsilon, gamma=gamma)
        self.variant = f"C={C:.4g},eps={epsilon:.4g},gamma={gamma if isinstance(gamma, str) else format(gamma, '.4g')}"

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.ms = [SVR(kernel="rbf", **self.kw).fit(X, X[:, j]) for j in range(X.shape[1])]
        self.meta["n_params"] = int(sum(2 * len(m.support_) for m in self.ms))   # alpha, alpha* per SV (as in P3)

    def score_anomaly(self, X):
        X = np.asarray(X, float)
        P = np.column_stack([m.predict(X) for m in self.ms])
        return ((X - P) ** 2).sum(1)


class AutoencoderAnomaly(Model):
    """P3 classical autoencoder (CAE) analogue: d -> h1 -> h2 -> h1 -> d, ReLU, trained to reconstruct.
    Deviation from P3: sklearn MLP has a linear output layer (P3 used tanh)."""
    name = "ae"

    def __init__(self, alpha=1e-4, learning_rate_init=1e-2, max_iter=500):
        super().__init__(); self.kw = dict(alpha=alpha, learning_rate_init=learning_rate_init, max_iter=max_iter)
        self.variant = f"alpha={alpha:.4g},lr={learning_rate_init:.4g},iters={max_iter}"

    def fit(self, X, y=None):
        X = np.asarray(X, float); d = X.shape[1]
        h1, h2 = max(2, round(0.8 * d)), max(1, round(0.6 * d))        # d=5 -> 4, 3 as in P3
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.m = MLPRegressor(hidden_layer_sizes=(h1, h2, h1), activation="relu", solver="adam",
                                  random_state=getattr(self, "seed", 0), **self.kw).fit(X, X)
        self.meta["n_params"] = int(sum(c.size for c in self.m.coefs_) + sum(i.size for i in self.m.intercepts_))

    def score_anomaly(self, X):
        X = np.asarray(X, float)
        return ((X - self.m.predict(X)) ** 2).sum(1)


# ---------------- T3 classification ----------------
class BatchTreeClassifier(Model):
    """Batch reference: sklearn tree retrained on the window."""
    name = "batch_tree"

    def __init__(self, max_depth=None, min_samples_leaf=1):
        super().__init__(); self.kw = dict(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        self.variant = f"depth={max_depth},leaf={min_samples_leaf}"

    def fit(self, X, y=None):
        self.m = DecisionTreeClassifier(random_state=getattr(self, "seed", 0), **self.kw).fit(X, y)
        self.meta["n_params"] = int(self.m.tree_.node_count)

    def predict(self, X): return self.m.predict(X)


class RiverHT(Model):
    """River Hoeffding Tree retrained from scratch on each window (windowed protocol).
    task='clf' or 'reg'; adaptive=True gives Hoeffding Adaptive Tree (classification)."""

    def __init__(self, task="clf", adaptive=False, grace_period=50, delta=1e-5, split_criterion=None, leaf_prediction=None):
        super().__init__(); self.task, self.adaptive = task, adaptive
        self.kw = dict(grace_period=int(grace_period), delta=delta)
        if split_criterion and task == "clf": self.kw["split_criterion"] = split_criterion
        if leaf_prediction: self.kw["leaf_prediction"] = leaf_prediction
        self.name = ("river_hat" if adaptive else "river_ht") + ("_reg" if task == "reg" else "")
        self.variant = ",".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in self.kw.items())

    def _new(self):
        if self.task == "reg":
            return tree.HoeffdingTreeRegressor(**self.kw)
        if self.adaptive:
            return tree.HoeffdingAdaptiveTreeClassifier(seed=getattr(self, "seed", 0), **self.kw)
        return tree.HoeffdingTreeClassifier(**self.kw)

    def fit(self, X, y=None):
        self.m = self._new()
        for xi, yi in zip(X, y):
            self.m.learn_one(_d(xi), float(yi) if self.task == "reg" else int(yi))
        try:
            self.meta["n_params"] = int(self.m.n_nodes)
        except Exception:
            pass

    def predict(self, X):
        out = [self.m.predict_one(_d(xi)) for xi in X]
        if self.task == "reg":
            return np.array([0.0 if o is None else o for o in out], dtype=float)
        return np.array([0 if o is None else int(o) for o in out])


class RiverNB(Model):
    name = "river_gaussian_nb"

    def fit(self, X, y=None):
        self.m = naive_bayes.GaussianNB()
        for xi, yi in zip(X, y): self.m.learn_one(_d(xi), int(yi))

    def predict(self, X):
        return np.array([0 if (o := self.m.predict_one(_d(xi))) is None else int(o) for xi in X])


class NaiveClassifier(Model):
    """Reference baselines: 'majority' class of the window, or 'last' label seen in the window."""
    def __init__(self, mode="majority"):
        super().__init__(); self.mode = mode; self.name = f"naive_{mode}"

    def fit(self, X, y=None):
        y = np.asarray(y).astype(int)
        self.v = int(y[-1]) if self.mode == "last" else int(np.bincount(y).argmax())

    def predict(self, X): return np.full(len(X), self.v, dtype=int)
