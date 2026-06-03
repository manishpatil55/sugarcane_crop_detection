"""
train.py
========
Polygon-level training driver for the sugarcane detection pipeline.

What this script does
---------------------
1. Loads a final feature CSV produced by ``features.feature_table_builder``.
2. Builds X, y, groups via ``FeatureTableBuilder.split_X_y_groups``.
3. Runs **StratifiedGroupKFold(5)** keyed on plot_id (the canonical CV).
4. For each fold, trains:
       - RandomForest, XGBoost, LightGBM (default hyperparameters)
       - StackingClassifier(RF+XGB+LGB → LogisticRegression)
5. Aggregates per-fold metrics + computes the buffered-holdout (500 m)
   variant for each fold.
6. Optionally runs **Optuna (50 trials)** on the best base learner.
7. Persists the final retrained-on-all model along with:
       - the fitted estimator
       - the calibrated threshold (Youden's J on OOF probabilities)
       - the feature names list
       - per-fold metrics JSON

Usage
-----
    python train.py --features data/processed/sugarcane_features.csv \\
                     --centroids data/processed/centroids.csv \\
                     --model_out models/saved/best.pkl \\
                     --tune                # enable Optuna 50-trial tuning
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Make project imports resolve when run from anywhere
sys.path.insert(0, str(Path(__file__).parent))

from features.feature_table_builder import FeatureTableBuilder
from models.tree_models import (
    make_random_forest, make_xgboost, make_lightgbm, make_stacker, model_factory,
)
from validation.splits import (
    polygon_group_kfold, buffered_holdout_indices, spatial_block_kfold,
)
from validation.metrics import (
    compute_metrics, youden_threshold, summarize_cv,
    permutation_importance_top_k,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("training.log", mode="w", encoding="utf-8")],
)
logger = logging.getLogger(__name__)


# ────────────────── CV runner ──────────────────

def cv_evaluate(
    name: str,
    factory_fn,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    centroids: Optional[pd.DataFrame],
    n_splits: int = 5,
    seed: int = 42,
    buffer_m: float = 500.0,
) -> Dict:
    """
    Run group-aware CV, return a dict with per-fold metrics and aggregate
    summaries (with and without 500 m buffer exclusion).

    factory_fn(y_train) → fresh model instance per fold.
    """
    fold_reports = []
    fold_buffered_reports = []
    oof_proba = np.zeros(len(y), dtype=float)
    oof_mask = np.zeros(len(y), dtype=bool)

    for fold_idx, (tr_idx, va_idx) in enumerate(polygon_group_kfold(y, groups, n_splits, seed)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = factory_fn(y_tr)
        model.fit(X_tr.values, y_tr)
        proba = model.predict_proba(X_va.values)[:, 1]

        oof_proba[va_idx] = proba
        oof_mask[va_idx] = True

        m = compute_metrics(y_va, proba, threshold=0.5)
        m["fold"] = fold_idx
        fold_reports.append(m)
        logger.info(
            f"  [{name}] fold {fold_idx} | "
            f"F1_macro={m['F1_macro']:.3f}  Kappa={m['Cohen_Kappa']:.3f}  "
            f"AUC={m['ROC_AUC']:.3f}  n_val={m['n_test']}"
        )

        # Buffered hold-out variant (drop val polygons within 500 m of any train polygon)
        if centroids is not None and len(va_idx) > 0:
            keep_va = buffered_holdout_indices(centroids, tr_idx, va_idx, buffer_m=buffer_m)
            if len(keep_va) >= 2 and len(np.unique(y[keep_va])) == 2:
                proba_kept = model.predict_proba(X.iloc[keep_va].values)[:, 1]
                m_buf = compute_metrics(y[keep_va], proba_kept, threshold=0.5)
                m_buf["fold"] = fold_idx
                m_buf["n_kept"] = len(keep_va)
                fold_buffered_reports.append(m_buf)

    # Calibrate threshold on OOF predictions (Youden's J)
    opt_thr = youden_threshold(y, oof_proba) if oof_mask.all() else 0.5

    summary = summarize_cv(fold_reports)
    summary_buffered = summarize_cv(fold_buffered_reports) if fold_buffered_reports else None

    return {
        "name":              name,
        "per_fold":          fold_reports,
        "summary":           summary,
        "buffered_per_fold": fold_buffered_reports,
        "buffered_summary":  summary_buffered,
        "oof_proba":         oof_proba.tolist(),
        "oof_threshold_youden": opt_thr,
    }


# ────────────────── Optuna tuning ──────────────────

def optuna_tune(
    name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_trials: int = 50,
    n_splits: int = 5,
    seed: int = 42,
):
    """Run Optuna 50-trial study optimising macro-F1, return best params."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _objective(trial: "optuna.Trial") -> float:
        if name == "xgb":
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 200, 600, step=50),
                max_depth=trial.suggest_int("max_depth", 3, 9),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                min_child_weight=trial.suggest_float("min_child_weight", 0.5, 5.0),
                reg_lambda=trial.suggest_float("reg_lambda", 0.0, 5.0),
            )
            factory = lambda y_tr: make_xgboost(y_train=y_tr, seed=seed, **params)
        elif name == "lgb":
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 200, 600, step=50),
                num_leaves=trial.suggest_int("num_leaves", 15, 127),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                min_child_samples=trial.suggest_int("min_child_samples", 2, 30),
                reg_lambda=trial.suggest_float("reg_lambda", 0.0, 5.0),
            )
            factory = lambda y_tr: make_lightgbm(seed=seed, **params)
        elif name == "rf":
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
                max_depth=trial.suggest_int("max_depth", 4, 30),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3]),
            )
            factory = lambda y_tr: make_random_forest(seed=seed, **params)
        else:
            raise ValueError(f"Unsupported model for tuning: {name}")

        f1s = []
        for tr, va in polygon_group_kfold(y, groups, n_splits, seed):
            m = factory(y[tr])
            m.fit(X.iloc[tr].values, y[tr])
            p = m.predict_proba(X.iloc[va].values)[:, 1]
            yh = (p >= 0.5).astype(int)
            from sklearn.metrics import f1_score
            f1s.append(f1_score(y[va], yh, average="macro", zero_division=0))
        return float(np.mean(f1s))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"  Optuna best macro-F1 = {study.best_value:.4f}")
    logger.info(f"  Optuna best params  = {study.best_params}")
    return study.best_params, study.best_value


# ────────────────── Main ──────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features",  required=True, help="path to final features CSV")
    ap.add_argument("--centroids", default=None,
                    help="optional CSV with columns [plot_id, lon, lat] for buffered CV")
    ap.add_argument("--model_out", default="models/saved/best.pkl")
    ap.add_argument("--metrics_out", default="models/saved/metrics.json")
    ap.add_argument("--n_splits",  type=int, default=5)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--tune",      action="store_true",
                    help="run Optuna 50-trial tuning on the best base learner")
    ap.add_argument("--tune_trials", type=int, default=50)
    ap.add_argument("--quick",     action="store_true",
                    help="skip the stacker ensemble (RF/XGB/LGB only)")
    ap.add_argument("--season_invariant", action="store_true",
                    help="use ONLY season-invariant features (drops all timestep-indexed "
                         "columns). Produces a model that works in any season.")
    args = ap.parse_args()

    np.random.seed(args.seed)

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PRODUCTION SUGARCANE TRAINING — PolygonGroupKFold")
    logger.info("=" * 60)

    # 1. Load features
    df = pd.read_csv(args.features)
    logger.info(f"Loaded features: {df.shape}")
    X, y, groups, feature_names = FeatureTableBuilder.split_X_y_groups(
        df, drop_quality_cols=True,
        season_invariant_only=args.season_invariant,
    )
    logger.info(f"X={X.shape}  y(pos)={int((y==1).sum())}  y(neg)={int((y==0).sum())}  "
                f"unique groups={len(np.unique(groups))}")

    # \u2014\u2014\u2014 Geographic-leakage audit \u2014\u2014\u2014
    # Token-boundary match so 'lon' does not appear inside 'correlation'.
    SUSPECT_TOKENS = {"lat", "lon", "longitude", "latitude", "state", "district",
                      "plot_id", "source_file", "anchor_date", "date_start",
                      "date_end", "group_key", "centroid"}

    def _has_geo_token(col: str) -> bool:
        toks = col.lower().replace("-", "_").split("_")
        return any(t in SUSPECT_TOKENS for t in toks)

    leaks = [c for c in feature_names if _has_geo_token(c)]
    if leaks:
        logger.warning(f"!! POSSIBLE LEAKAGE: {leaks}")
        # Hard remove these to be safe
        X = X.drop(columns=leaks)
        feature_names = X.columns.tolist()
        logger.warning(f"   Dropped \u2192 X={X.shape}")
    else:
        logger.info("\u2713  Geography audit clean: no lat/lon/state/district/plot_id/source_file in features.")
    pos_class_ratio = (y == 1).mean()
    if pos_class_ratio < 0.4 or pos_class_ratio > 0.6:
        logger.warning(
            f"!! Class imbalance: pos={int((y==1).sum())} neg={int((y==0).sum())} "
            f"({pos_class_ratio:.1%} positive). Models will use class_weight='balanced'."
        )
    else:
        logger.info(f"\u2713  Class balance: pos_ratio={pos_class_ratio:.1%}  (target ~50%)")

    # Centroids for buffered hold-out
    centroids = None
    if args.centroids and Path(args.centroids).exists():
        cdf = pd.read_csv(args.centroids)
        # Align centroid order to df rows
        cdf = cdf.set_index("plot_id").loc[df["plot_id"].values, :].reset_index(drop=True)
        centroids = cdf[["lon", "lat"]]

    # 2. CV evaluation of all models
    logger.info("\n── Baseline CV evaluation ─────────────────────────")
    results = {}
    for name, factory in [
        ("rf",  lambda yt: make_random_forest(seed=args.seed)),
        ("xgb", lambda yt: make_xgboost(y_train=yt, seed=args.seed)),
        ("lgb", lambda yt: make_lightgbm(seed=args.seed)),
    ]:
        logger.info(f"\n[{name.upper()}]")
        results[name] = cv_evaluate(
            name, factory, X, y, groups, centroids,
            n_splits=args.n_splits, seed=args.seed,
        )
        s = results[name]["summary"]
        logger.info(
            f"  AGG: F1_macro={s['F1_macro_mean']:.3f}±{s['F1_macro_std']:.3f}  "
            f"Kappa={s['Cohen_Kappa_mean']:.3f}  AUC={s['ROC_AUC_mean']:.3f}"
        )

    # Stacking (skipped in --quick mode for fast iteration)
    from sklearn.model_selection import StratifiedKFold
    stacker_inner_cv = StratifiedKFold(n_splits=3, shuffle=True,
                                        random_state=args.seed)

    if not args.quick:
        logger.info("\n[STACK]")

        def _stack_factory(yt):
            return make_stacker(cv=stacker_inner_cv, y_train=yt, seed=args.seed)

        try:
            results["stack"] = cv_evaluate(
                "stack", _stack_factory, X, y, groups, centroids,
                n_splits=args.n_splits, seed=args.seed,
            )
            s = results["stack"]["summary"]
            logger.info(
                f"  AGG: F1_macro={s['F1_macro_mean']:.3f}\u00b1{s['F1_macro_std']:.3f}  "
                f"Kappa={s['Cohen_Kappa_mean']:.3f}  AUC={s['ROC_AUC_mean']:.3f}"
            )
        except Exception as exc:
            logger.warning(f"  Stacker failed, skipping: {exc}")
    else:
        logger.info("\n[STACK]  skipped (--quick)")

    # 3. Pick best by F1_macro_mean
    best_name = max(results, key=lambda k: results[k]["summary"]["F1_macro_mean"])
    logger.info(f"\nBest base model by F1_macro: {best_name.upper()}  "
                f"F1_macro={results[best_name]['summary']['F1_macro_mean']:.3f}")

    # 3b. Spatial-block CV variant on the best model (5 km blocks)
    spatial_block_results = None
    if centroids is not None and best_name in ("rf", "xgb", "lgb"):
        logger.info(
            f"\n── Spatial-block CV (best={best_name.upper()}, 0.05° \u2248 5 km blocks) ──"
        )
        try:
            block_folds = []
            for tr_idx, va_idx in spatial_block_kfold(
                centroids, y, n_splits=args.n_splits, block_deg=0.05, seed=args.seed,
            ):
                if best_name == "rf":
                    m = make_random_forest(seed=args.seed)
                elif best_name == "xgb":
                    m = make_xgboost(y_train=y[tr_idx], seed=args.seed)
                else:
                    m = make_lightgbm(seed=args.seed)
                m.fit(X.iloc[tr_idx].values, y[tr_idx])
                p = m.predict_proba(X.iloc[va_idx].values)[:, 1]
                fm = compute_metrics(y[va_idx], p, threshold=0.5)
                block_folds.append(fm)
                logger.info(
                    f"  spatial-block fold | F1_macro={fm['F1_macro']:.3f}  "
                    f"Kappa={fm['Cohen_Kappa']:.3f}  AUC={fm['ROC_AUC']:.3f}  "
                    f"n_val={fm['n_test']}"
                )
            spatial_block_results = {"per_fold": block_folds,
                                     "summary": summarize_cv(block_folds)}
            s = spatial_block_results["summary"]
            logger.info(
                f"  AGG: F1_macro={s['F1_macro_mean']:.3f}±{s['F1_macro_std']:.3f}  "
                f"Kappa={s['Cohen_Kappa_mean']:.3f}  AUC={s['ROC_AUC_mean']:.3f}"
            )
        except Exception as exc:
            logger.warning(f"Spatial-block CV failed: {exc}")

    # 4. Optional Optuna tuning on the best (only for rf/xgb/lgb)
    tuned_params = None
    if args.tune and best_name in ("rf", "xgb", "lgb"):
        logger.info(f"\n── Optuna tuning ({args.tune_trials} trials) on {best_name.upper()} ──")
        tuned_params, _ = optuna_tune(
            best_name, X, y, groups, n_trials=args.tune_trials,
            n_splits=args.n_splits, seed=args.seed,
        )

    # 5. Final fit on ALL data using best (possibly tuned) hyperparameters
    if tuned_params is not None:
        if best_name == "xgb":
            final_model = make_xgboost(y_train=y, seed=args.seed, **tuned_params)
        elif best_name == "lgb":
            final_model = make_lightgbm(seed=args.seed, **tuned_params)
        elif best_name == "rf":
            final_model = make_random_forest(seed=args.seed, **tuned_params)
    else:
        if best_name == "xgb":
            final_model = make_xgboost(y_train=y, seed=args.seed)
        elif best_name == "lgb":
            final_model = make_lightgbm(seed=args.seed)
        elif best_name == "rf":
            final_model = make_random_forest(seed=args.seed)
        elif best_name == "stack":
            final_model = make_stacker(cv=stacker_inner_cv, y_train=y, seed=args.seed)
    final_model.fit(X.values, y)

    # 6. Calibrated threshold on OOF probabilities of the *winning* model
    oof_proba = np.array(results[best_name]["oof_proba"])
    optimal_threshold = youden_threshold(y, oof_proba)
    logger.info(f"\nOptimal threshold (Youden's J on OOF): {optimal_threshold:.4f}")

    # 7. Top-20 feature importance.
    # Strategy: use the model's native feature_importances_ to PRESELECT the
    # 100 most likely candidates (instant), then run permutation importance
    # only on those 100 (3 repeats). This keeps the entire step <30 s even
    # with 1500+ features.
    # 7. Top-20 features.
    # Tree models: use the model's native ``feature_importances_`` (instant).
    # Non-tree: fall back to permutation importance (slower).
    try:
        if hasattr(final_model, "feature_importances_"):
            imp = np.asarray(final_model.feature_importances_)
            order = np.argsort(imp)[::-1][:20]
            top20 = pd.DataFrame({
                "feature":         [feature_names[i] for i in order],
                "importance_mean": imp[order],
                "importance_std":  np.zeros_like(imp[order]),
            })
        else:
            top20 = permutation_importance_top_k(
                final_model, X, y,
                feature_names=feature_names, n_repeats=3, k=20, seed=args.seed,
            )
        logger.info("\nTop-20 features by importance:")
        for _, r in top20.iterrows():
            logger.info(
                f"  {r['feature']:50s}  {r['importance_mean']:+.4f}"
            )
    except Exception as exc:
        logger.warning(f"Could not compute feature importance: {exc}")
        top20 = pd.DataFrame()

    # 8. Persist artefacts
    artefact = {
        "model":             final_model,
        "feature_names":     feature_names,
        "optimal_threshold": optimal_threshold,
        "best_model_name":   best_name,
        "tuned_params":      tuned_params,
        "n_features":        len(feature_names),
    }
    joblib.dump(artefact, args.model_out)
    logger.info(f"Saved final model → {args.model_out}")

    # JSON-safe metrics (drop oof_proba arrays for size)
    metrics_json = {
        "best_model_name":   best_name,
        "optimal_threshold": optimal_threshold,
        "tuned_params":      tuned_params,
        "n_features":        len(feature_names),
        "n_samples":         int(len(y)),
        "n_positive":        int((y == 1).sum()),
        "n_negative":        int((y == 0).sum()),
        "class_ratio_pos":   float((y == 1).mean()),
        "results":           {
            k: {
                "summary": v["summary"],
                "buffered_summary": v["buffered_summary"],
                "per_fold": v["per_fold"],
                "buffered_per_fold": v["buffered_per_fold"],
            }
            for k, v in results.items()
        },
        "spatial_block_results": spatial_block_results,
        "top20": top20.to_dict(orient="records") if len(top20) else [],
    }
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2, default=str)
    logger.info(f"Saved metrics    → {args.metrics_out}")

    # 9. Update config.yaml.inference.active_model and persisted threshold
    cfg_path = Path("config.yaml")
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        cfg.setdefault("inference", {})
        cfg["inference"]["active_model"] = best_name
        cfg["inference"]["probability_threshold"] = float(optimal_threshold)
        cfg["inference"]["model_path"] = str(Path(args.model_out).as_posix())
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        logger.info(f"Updated config.yaml → active_model={best_name}, "
                    f"threshold={optimal_threshold:.4f}")

    logger.info("\nTRAINING DONE.")


if __name__ == "__main__":
    main()
