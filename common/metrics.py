import numpy as np
from sklearn.metrics import (mean_squared_error, mean_absolute_error, accuracy_score,
                             f1_score, roc_auc_score, average_precision_score)


def mse(y, p): return float(mean_squared_error(y, p))
def mae(y, p): return float(mean_absolute_error(y, p))
def accuracy(y, p): return float(accuracy_score(y, p))
def f1_macro(y, p): return float(f1_score(y, p, average="macro", zero_division=0))


def pooled_auc(labels, scores):
    """AUC over pooled scored samples. NaN if only one class present."""
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def pooled_ap(labels, scores):
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def recovery_time(n_seen, values, drift_at, higher_is_better=True, tol=0.05, pre_k=5):
    """Return (pre_level, worst_post, recovery_samples).
    recovery_samples = samples after drift_at until the metric is back within tol
    (relative) of the pre-drift mean, searched after the worst post-drift point."""
    n = np.asarray(n_seen); v = np.asarray(values, dtype=float)
    pre = v[n <= drift_at][-pre_k:]
    post = n > drift_at
    if len(pre) == 0 or post.sum() == 0 or np.all(np.isnan(v[post])):
        return float("nan"), float("nan"), float("nan")
    pre_level = float(np.nanmean(pre))
    pn, pv = n[post], v[post]
    w = int(np.nanargmin(pv) if higher_is_better else np.nanargmax(pv))
    worst = float(pv[w])
    ok = (pv >= pre_level * (1 - tol)) if higher_is_better else (pv <= pre_level * (1 + tol))
    back = np.where(ok[w:])[0]
    rec = float(pn[w + back[0]] - drift_at) if len(back) else float("nan")
    return pre_level, worst, rec


def kappa_vs(acc, acc_ref):
    """Skill relative to a reference classifier: (acc - ref) / (1 - ref).
    1 = perfect, 0 = no better than the reference, < 0 = worse. NaN if reference is perfect."""
    if acc_ref is None or acc_ref >= 1.0 or np.isnan(acc_ref):
        return float("nan")
    return float((acc - acc_ref) / (1.0 - acc_ref))


def kappa_plus(acc, acc_nochange):
    """Kappa+ (a.k.a. Kappa Temporal): reference = no-change / persistence classifier
    (predict the previous label). The right yardstick on streams with temporal dependence (e.g. ELEC2)."""
    return kappa_vs(acc, acc_nochange)


def kappa_m(acc, acc_majority):
    """Kappa M: reference = majority-class classifier."""
    return kappa_vs(acc, acc_majority)


def nochange_accuracy(y_prev, y_true):
    """Accuracy of predicting 'same label as the sample just before'. y_prev[i] is the label preceding y_true[i]."""
    return float(np.mean(np.asarray(y_prev) == np.asarray(y_true)))
