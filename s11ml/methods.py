"""Model wrappers shared by training and the website (must be importable to load the saved model)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from . import core

SEED = 42

METHODS = {"rf_baseline": "A. Random Forest (notebook baseline)",
           "rf_rotaug": "B. Random Forest + rotation augmentation",
           "physics_peak": "C. Physics rule: strongest antenna (no ML)"}


class Method:
    """Common interface: fit(X, y) / predict(X) / predict_proba(X) on 504-feature rows."""

    def __init__(self, kind: str):
        if kind not in METHODS:
            raise ValueError(kind)
        self.kind = kind

    def fit(self, X, y):
        if self.kind == "physics_peak":
            self.offset_ = core.fit_physics_offset(X, y)
            return self
        if self.kind == "rf_rotaug":
            X, y = core.augment_rotations(X, y)          # training data only
        self.model_ = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        X = np.atleast_2d(X)
        if self.kind == "physics_peak":
            return core.physics_predict(X, self.offset_)
        return self.model_.predict(X)

    def predict_proba(self, X):
        X = np.atleast_2d(X)
        P = np.zeros((len(X), 4))
        if self.kind == "physics_peak":
            P[np.arange(len(X)), self.predict(X)] = 1.0
            return P
        P[:, self.model_.classes_] = self.model_.predict_proba(X)   # classes never seen -> 0
        return P
