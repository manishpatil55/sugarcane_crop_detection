"""
validation/metrics.py
=====================
Honest evaluation utilities.

Functions
---------
- ``compute_metrics``        — full metric battery for one fold
- ``youden_threshold``       — optimal threshold from ROC curve (Youden's J)
- ``permutation_importance_top_k`` — top-k feature importances on OOF data
- ``summarize_cv``           — aggregate per-fold metrics into mean ± std

Metrics computed per fold
-------------------------
- Overall Accuracy (OA)
- F1 sugarcane / F1 non-sugarcane / F1 macro
- Cohen's Kappa
- ROC-AUC, PR-AUC
- Brier score (probability calibration)
- Confusion matrix (counts and row-normalised %)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, roc_curve,
)
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute the full metric battery.

    Returns a dict with scalar metrics and confusion matrices.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100.0

    # Some metrics need both classes present
    has_both = len(np.unique(y_true)) == 2

    metrics = {
        "OA":              float(accuracy_score(y_true, y_pred)),
        "F1_sugarcane":    float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "F1_non_sugarcane": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "F1_macro":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Cohen_Kappa":     float(cohen_kappa_score(y_true, y_pred)),
        "ROC_AUC":         float(roc_auc_score(y_true, y_proba)) if has_both else float("nan"),
        "PR_AUC":          float(average_precision_score(y_true, y_proba)) if has_both else float("nan"),
        "Brier":           float(brier_score_loss(y_true, y_proba)),
        "threshold":       float(threshold),
        "n_test":          int(len(y_true)),
        "n_test_pos":      int((y_true == 1).sum()),
        "n_test_neg":      int((y_true == 0).sum()),
        "CM":              cm.tolist(),
        "CM_pct":          cm_pct.round(2).tolist(),
    }
    return metrics


def youden_threshold(y_true: np.ndarray, y_proba: np.ndarray, max_threshold: float = 0.60) -> float:
    """Optimal threshold from Youden's J = TPR - FPR on the ROC curve, capped for better recall."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    j = tpr - fpr
    best = int(np.argmax(j))
    best_thr = float(np.clip(thr[best], 0.0, 1.0))
    if best_thr > max_threshold:
        logger.info(f"Youden threshold {best_thr:.4f} is too strict. Capping at {max_threshold:.4f} to detect young sugarcane.")
        best_thr = max_threshold
    return best_thr


def permutation_importance_top_k(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_repeats: int = 10,
    k: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Top-k features by permutation importance on the provided dataset."""
    fn = feature_names or list(X.columns)
    pi = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=seed, n_jobs=-1,
    )
    order = np.argsort(pi.importances_mean)[::-1][:k]
    out = pd.DataFrame({
        "feature":          [fn[i] for i in order],
        "importance_mean":  pi.importances_mean[order],
        "importance_std":   pi.importances_std[order],
    })
    return out


def summarize_cv(per_fold: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-fold metrics into mean ± std."""
    scalar_keys = [
        "OA", "F1_sugarcane", "F1_non_sugarcane", "F1_macro",
        "Cohen_Kappa", "ROC_AUC", "PR_AUC", "Brier",
    ]
    out: Dict[str, Any] = {}
    for k in scalar_keys:
        vals = np.array([m.get(k, np.nan) for m in per_fold], dtype=float)
        out[f"{k}_mean"] = float(np.nanmean(vals))
        out[f"{k}_std"]  = float(np.nanstd(vals))
    # Sum confusion matrices across folds
    cm_total = np.zeros((2, 2), dtype=int)
    for m in per_fold:
        cm_total += np.array(m["CM"], dtype=int)
    out["CM_total"] = cm_total.tolist()
    out["CM_total_pct"] = (
        cm_total / np.maximum(cm_total.sum(axis=1, keepdims=True), 1) * 100
    ).round(2).tolist()
    out["n_folds"] = len(per_fold)
    return out
