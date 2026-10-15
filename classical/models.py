"""Classical models wrapped in the shared Model interface. Owned by Member A."""
import numpy as np
from sklearn.svm import SVR, OneClassSVM
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from river import tree, naive_bayes
from common.base import Model


def _d(row):
    return {i: float(v) for i, v in enumerate(row)}


class SVRRegressor(Model):
    name = "svr_rbf"

    def __init__(self, C=1.0, epsilon=0.05, gamma="scale"):
        super().__init__(); self.kw = dict(C=C, epsilon=epsilon, gamma=gamma)
        self.variant = f"C={C},eps={epsilon},gamma={gamma}"

    def fit(self, X, y=None):
        self.m = SVR(kernel="rbf", **self.kw).fit(X, y)
        self.meta["n_params"] = int(len(self.m.support_))   # support vectors

    def predict(self, X): return self.m.predict(X)


class RFRegressor(Model):
    name = "random_forest"

    def __init__(self, n_estimators=100, max_depth=None):
        super().__init__(); self.kw = dict(n_estimators=n_estimators, max_depth=max_depth)
        self.variant = f"n={n_estimators},depth={max_depth}"

    def fit(self, X, y=None):
        self.m = RandomForestRegressor(random_state=getattr(self, "seed", 0), n_jobs=1, **self.kw).fit(X, y)

    def predict(self, X): return self.m.predict(X)


class OCSVMAnomaly(Model):
    name = "ocsvm_rbf"

    def __init__(self, nu=0.05, gamma="scale"):
        super().__init__(); self.kw = dict(nu=nu, gamma=gamma); self.variant = f"nu={nu},gamma={gamma}"

    def fit(self, X, y=None):
        self.m = OneClassSVM(kernel="rbf", **self.kw).fit(X)
        self.meta["n_params"] = int(len(self.m.support_))

    def score_anomaly(self, X): return -self.m.decision_function(X)


class BatchTreeClassifier(Model):
    """Batch reference: sklearn tree retrained on the window."""
    name = "batch_tree"

    def __init__(self, max_depth=None):
        super().__init__(); self.max_depth = max_depth; self.variant = f"depth={max_depth}"

    def fit(self, X, y=None):
        self.m = DecisionTreeClassifier(max_depth=self.max_depth, random_state=getattr(self, "seed", 0)).fit(X, y)
        self.meta["n_params"] = int(self.m.tree_.node_count)

    def predict(self, X): return self.m.predict(X)


class RiverHT(Model):
    """River Hoeffding Tree retrained from scratch on each window (windowed protocol).
    task='clf' or 'reg'; adaptive=True gives Hoeffding Adaptive Tree (classification)."""

    def __init__(self, task="clf", adaptive=False, grace_period=50, delta=1e-5):
        super().__init__(); self.task, self.adaptive = task, adaptive
        self.kw = dict(grace_period=grace_period, delta=delta)
        self.name = ("river_hat" if adaptive else "river_ht") + ("_reg" if task == "reg" else "")
        self.variant = f"grace={grace_period},delta={delta}"

    def _new(self):
        if self.task == "reg":
            return tree.HoeffdingTreeRegressor(**self.kw)
        cls = tree.HoeffdingAdaptiveTreeClassifier if self.adaptive else tree.HoeffdingTreeClassifier
        return cls(**self.kw)

    def fit(self, X, y=None):
        self.m = self._new()
        for xi, yi in zip(X, y):
            self.m.learn_one(_d(xi), yi if self.task == "reg" else int(yi))
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
