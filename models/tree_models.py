"""
models/tree_models.py
=====================
Factory functions for the three tree-based learners and a stacking ensemble.

Defaults follow the user-specified configuration:
    RandomForest(n_estimators=300, max_depth=None, min_samples_leaf=2,
                 class_weight='balanced', random_state=42)
    XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8,
                  scale_pos_weight=auto, random_state=42, eval_metric='aucpr')
    LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8,
                   class_weight='balanced', random_state=42)
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def make_random_forest(seed: int = 42, **overrides) -> RandomForestClassifier:
    params = dict(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=seed, n_jobs=-1,
    )
    params.update(overrides)
    return RandomForestClassifier(**params)


def make_xgboost(y_train: Optional[np.ndarray] = None, seed: int = 42, **overrides
                 ) -> XGBClassifier:
    if y_train is not None and len(y_train):
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        scale_pos_weight = max(1.0, n_neg / max(n_pos, 1))
    else:
        scale_pos_weight = 1.0
    params = dict(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr", random_state=seed, n_jobs=-1,
        tree_method="hist", verbosity=0,
    )
    params.update(overrides)
    return XGBClassifier(**params)


def make_lightgbm(seed: int = 42, **overrides) -> LGBMClassifier:
    params = dict(
        n_estimators=300, num_leaves=63, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=seed, n_jobs=-1,
        verbose=-1,
    )
    params.update(overrides)
    return LGBMClassifier(**params)


def make_stacker(
    cv,
    y_train: Optional[np.ndarray] = None,
    seed: int = 42,
) -> StackingClassifier:
    """
    Stacking ensemble: RF + XGB + LGB → LogisticRegression meta.

    Parameters
    ----------
    cv : a splitter object — pass the same group-aware CV used for headline
         metrics so blending does not leak.
    """
    estimators = [
        ("rf",  make_random_forest(seed=seed)),
        ("xgb", make_xgboost(y_train=y_train, seed=seed)),
        ("lgb", make_lightgbm(seed=seed)),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced",
                                           random_state=seed),
        cv=cv,
        n_jobs=-1,
        passthrough=False,
    )


def model_factory(name: str, y_train: Optional[np.ndarray] = None, seed: int = 42, **kw):
    """Single entry-point. ``name`` ∈ {'rf', 'xgb', 'lgb'}."""
    name = name.lower()
    if name in ("rf", "random_forest"):
        return make_random_forest(seed=seed, **kw)
    if name in ("xgb", "xgboost"):
        return make_xgboost(y_train=y_train, seed=seed, **kw)
    if name in ("lgb", "lgbm", "lightgbm"):
        return make_lightgbm(seed=seed, **kw)
    raise ValueError(f"Unknown model name: {name!r}")
