"""Frozen preprocessing transform loader and applier for Spark micro-batches and Gate 1 tests."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


class FrozenTransform:
    """Applies the frozen standardisation -> PCA -> [0, pi] MinMax scaling saved by data/prep.py."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.mu = np.asarray(spec["mu"], dtype=float)
        self.sd = np.asarray(spec["sd"], dtype=float)
        self.n_components = spec.get("n_components")
        self.pca_mean = (
            np.asarray(spec["pca_mean"], dtype=float)
            if spec.get("pca_mean") is not None
            else None
        )
        self.pca_components = (
            np.asarray(spec["pca_components"], dtype=float)
            if spec.get("pca_components") is not None
            else None
        )
        self.lo = np.asarray(spec["lo"], dtype=float)
        self.hi = np.asarray(spec["hi"], dtype=float)
        self.clip_min = float(spec.get("clip_min", 0.0))
        self.clip_max = float(spec.get("clip_max", np.pi))

    @classmethod
    def from_json(cls, path: str | Path) -> "FrozenTransform":
        return cls(json.loads(Path(path).read_text()))

    def transform_features(self, X_raw: np.ndarray) -> np.ndarray:
        """Transform raw numeric matrix X_raw to [0, pi] feature matrix matching data/processed/."""
        X_raw = np.asarray(X_raw, dtype=float)
        Z = (X_raw - self.mu) / self.sd
        if self.n_components and self.pca_components is not None:
            Z = (Z - self.pca_mean) @ self.pca_components.T
        S = np.pi * (Z - self.lo) / (self.hi - self.lo)
        return np.clip(S, self.clip_min, self.clip_max)

    def transform_target(self, y_raw: np.ndarray) -> np.ndarray:
        """Apply global minmax scaling to y if configured (e.g. T1), else return as-is."""
        y_raw = np.asarray(y_raw)
        ts = self.spec.get("target_scaling")
        if ts and ts.get("mode") == "global_minmax":
            y_min, y_max = float(ts["y_min"]), float(ts["y_max"])
            return (y_raw.astype(float) - y_min) / (y_max - y_min)
        return y_raw
