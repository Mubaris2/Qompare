import numpy as np


def inject_drift(X, kind="abrupt", start_frac=0.5, width_frac=0.2, magnitude=1.0,
                 seed=0, clip=(0.0, np.pi)):
    """Shift feature means by `magnitude` x feature-std (std of first 10% of rows).
    abrupt : full shift from start index onward.
    gradual: linear ramp over width_frac of the stream, then full shift.
    Returns (X_drifted, drift_start_index). Original X is not modified."""
    X = np.array(X, dtype=float, copy=True)
    if kind == "none":
        return X, None
    n, d = X.shape
    rng = np.random.default_rng(seed)
    sign = rng.choice([-1.0, 1.0], size=d)
    shift = sign * X[: max(10, n // 10)].std(axis=0) * magnitude
    s = int(n * start_frac)
    if kind == "abrupt":
        X[s:] += shift
    elif kind == "gradual":
        w = max(1, int(n * width_frac))
        ramp = np.clip((np.arange(n) - s) / w, 0.0, 1.0)[:, None]
        X += ramp * shift
    else:
        raise ValueError(kind)
    if clip is not None:
        X = np.clip(X, clip[0], clip[1])
    return X, s
