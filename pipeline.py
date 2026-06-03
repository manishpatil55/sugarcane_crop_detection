"""
pipeline.py
===========
Single-command, deterministic, end-to-end driver for the rebuilt sugarcane
pipeline.

Stages (each runs only if its output does not already exist, OR if --force
is passed)
----------------------------------------------------------------------------
  1. parse     : KML  → GeoDataFrame  (data/processed/plots.gpkg, centroids.csv)
  2. extract   : GeoDataFrame  → polygon-level zonal stats CSV
                                  (data/processed/extraction_wide.csv)
  3. features  : extraction CSV  → final feature CSV
                                    (data/processed/sugarcane_features.csv)
  4. train     : feature CSV  → trained model + metrics JSON
                                  (models/saved/best.pkl,
                                   models/saved/metrics.json)
  5. predict   : optional, run inference on a target KML

Determinism
-----------
- Numpy + Python random seeds are set from config.yaml.sampling.random_seed.
- The polygon extractor caches per-plot results to data/cache/, so re-runs
  are bit-identical and skip the GEE I/O.
- StratifiedGroupKFold uses a fixed seed.

Examples
--------
Full end-to-end (extraction + training):
    python pipeline.py --kml_dir data/kml --train

Build features only (no model training):
    python pipeline.py --kml_dir data/kml

Train with Optuna tuning enabled:
    python pipeline.py --kml_dir data/kml --train --tune

Inference only:
    python pipeline.py --predict path/to/farm.kml --crop_date 2025-09-01
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("training.log", mode="a", encoding="utf-8")],
)
logger = logging.getLogger("pipeline")


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    # IMPORTANT (Windows stability):
    # Importing torch can hard-crash some Python builds (access violation)
    # before any exception can be caught. This pipeline's default path
    # (RF/XGB/LGB, including --quick) does not require torch, so we avoid
    # importing it here.
    #
    # If/when BiLSTM training is explicitly re-enabled, seed torch in that
    # dedicated code path instead of global pipeline startup.


# ────────────────── Stage 1: parse ──────────────────

def balance_dataset(gdf, strategy: str = "one_per_file",
                    seed: int = 42, n_per_class: int = None):
    """
    Balance positives vs negatives so the trained model is not class-biased.

    Strategies
    ----------
    'one_per_file' (default)
        Keep one polygon per source KML file — the LARGEST by area
        (deterministic). For your data this yields exactly:
            200 positive files \u2192 200 positive polygons
            200 negative files \u2192 200 negative polygons
        \u2192 a perfectly balanced 200 : 200 training set.

        Rationale: multi-polygon KMLs (some negatives have 30+ polygons drawn
        in the same village) are spatially correlated; keeping all of them
        gives those villages a much louder vote than the single-polygon
        positive plots, AND it lets a model memorise village-level
        signatures instead of crop signatures. One-per-file kills both
        problems in one move.

    'subsample_n'
        Randomly pick n_per_class polygons per class (file-grouped).

    'none'
        Keep all 855 polygons (200 pos + 655 neg) — falls back on
        class_weight='balanced' loss reweighting.
    """
    import numpy as np
    import geopandas as gpd
    rng = np.random.RandomState(seed)

    if strategy == "none":
        logger.info(f"[balance] strategy=none \u2192 keeping all {len(gdf)} polygons")
        return gdf.reset_index(drop=True)

    if strategy == "one_per_file":
        keep_rows = []
        for (lbl, gk), sub in gdf.groupby(["label", "group_key"], sort=False):
            # Largest polygon by area (most reliable signal-to-noise)
            keep = sub.loc[sub["area_ha"].astype(float).idxmax()]
            keep_rows.append(keep)
        out = gpd.GeoDataFrame(keep_rows, crs=gdf.crs).reset_index(drop=True)
        n_pos = int((out["label"] == 1).sum())
        n_neg = int((out["label"] == 0).sum())
        logger.info(
            f"[balance] strategy=one_per_file \u2192 {len(out)} polygons "
            f"(pos={n_pos}, neg={n_neg}, ratio={n_pos / max(n_neg,1):.2f})"
        )
        return out

    if strategy == "subsample_n":
        if n_per_class is None:
            n_per_class = int((gdf["label"] == 1).sum())  # default = positive count
        # First take 1 per file
        one_per = balance_dataset(gdf, "one_per_file", seed=seed)
        out = []
        for lbl in (1, 0):
            sub = one_per[one_per["label"] == lbl]
            if len(sub) > n_per_class:
                sub = sub.sample(n_per_class, random_state=seed)
            out.append(sub)
        out = pd.concat(out, ignore_index=True)
        out = gpd.GeoDataFrame(out, crs=gdf.crs).reset_index(drop=True)
        logger.info(f"[balance] strategy=subsample_n({n_per_class}) \u2192 {len(out)}")
        return out

    raise ValueError(f"Unknown balance strategy: {strategy!r}")


def stage_parse(kml_dir: Path, processed_dir: Path,
                cfg_anchor: str = "2025-09-01",
                balance_strategy: str = "one_per_file",
                seed: int = 42,
                force: bool = False) -> pd.DataFrame:
    plots_path = processed_dir / "plots.gpkg"
    centroids_path = processed_dir / "centroids.csv"
    if plots_path.exists() and centroids_path.exists() and not force:
        import geopandas as gpd
        gdf = gpd.read_file(plots_path)
        logger.info(f"[parse] cached: {plots_path} ({len(gdf)} polygons)")
        return gdf

    from datetime import datetime, timedelta
    from dateutil.relativedelta import relativedelta
    import yaml as _yaml
    with open("config.yaml") as _f:
        _cfg = _yaml.safe_load(_f)
    months_before = int(_cfg.get("compositing", {}).get("months_before", 5))
    months_after = int(_cfg.get("compositing", {}).get("months_after", 5))

    from data.kml_parser import KMLParser

    parser = KMLParser(months_before=months_before, months_after=months_after)
    pos_dir = kml_dir / "sugarcane"
    neg_dir = kml_dir / "non_sugarcane"
    if not pos_dir.exists() or not neg_dir.exists():
        raise FileNotFoundError(
            f"Expected directories {pos_dir} and {neg_dir} to exist. "
            "Place positive KMLs in data/kml/sugarcane and negatives in data/kml/non_sugarcane."
        )

    # Parse KMLs — each file gets its own anchor date from the filename
    # (e.g. "7_sept2024.kml" → 2024-09-01, "109_aug2023.kml" → 2023-08-01).
    # The KMLParser falls back to config.yaml.default_anchor_date for files
    # without a parseable date in their name.
    fallback_anchor = cfg_anchor or "2024-09-01"

    pos_gdf = parser.parse_directory(pos_dir, label=1)
    neg_gdf = parser.parse_directory(neg_dir, label=0)
    import geopandas as gpd
    gdf = gpd.GeoDataFrame(pd.concat([pos_gdf, neg_gdf], ignore_index=True),
                           crs="EPSG:4326")

    # --- PER-PLOT anchor dates (season-invariant approach) ---
    # Each plot keeps its own anchor_date from the KML parser.
    # For plots without a date (most non-sugarcane), apply the fallback.

    fallback_dt = datetime.strptime(fallback_anchor, "%Y-%m-%d").date()
    fb_ds = (fallback_dt - relativedelta(months=months_before)).strftime("%Y-%m-%d")
    fb_de = (fallback_dt + relativedelta(months=months_after)).strftime("%Y-%m-%d")

    # Fill missing anchor dates with fallback, compute per-plot windows
    for idx in gdf.index:
        ad = gdf.at[idx, "anchor_date"]
        if ad is None or (isinstance(ad, str) and ad in ("", "None", "nan")):
            gdf.at[idx, "anchor_date"] = str(fallback_dt)
            gdf.at[idx, "date_start"] = fb_ds
            gdf.at[idx, "date_end"] = fb_de
        elif gdf.at[idx, "date_start"] is None or (isinstance(gdf.at[idx, "date_start"], str) and gdf.at[idx, "date_start"] in ("", "None", "nan")):
            try:
                if isinstance(ad, str):
                    ad_dt = datetime.strptime(ad, "%Y-%m-%d").date()
                else:
                    ad_dt = ad
                gdf.at[idx, "date_start"] = (ad_dt - relativedelta(months=months_before)).strftime("%Y-%m-%d")
                gdf.at[idx, "date_end"] = (ad_dt + relativedelta(months=months_after)).strftime("%Y-%m-%d")
            except Exception:
                gdf.at[idx, "date_start"] = fb_ds
                gdf.at[idx, "date_end"] = fb_de

    unique_anchors = gdf["anchor_date"].astype(str).unique()
    logger.info(
        f"Per-plot anchor dates: {len(unique_anchors)} unique values: {sorted(unique_anchors)}"
        f"  Window config: -{months_before}m / +{months_after}m"
    )

    # Stable plot_id (one per polygon, sequential).
    # group_key: <class>_<source_file> — used by StratifiedGroupKFold so
    # multi-polygon negative KMLs do not split across folds, and so identical
    # filenames in pos/ vs neg/ never collide into the same group.
    gdf = gdf.reset_index(drop=True)
    gdf["plot_id"] = [
        f"{'pos' if r.label == 1 else 'neg'}_{i:04d}"
        for i, r in gdf.iterrows()
    ]
    gdf["group_key"] = [
        f"{'pos' if r.label == 1 else 'neg'}__{r.source_file}"
        for _, r in gdf.iterrows()
    ]

    # Cast date-typed columns to ISO strings for GPKG compatibility.
    for c in ("anchor_date", "date_start", "date_end"):
        if c in gdf.columns:
            gdf[c] = gdf[c].astype(str)

    # \u2014\u2014 Balance the dataset before extraction (CRITICAL: user requested 1:1 ratio)
    gdf = balance_dataset(gdf, strategy=balance_strategy, seed=seed)

    processed_dir.mkdir(parents=True, exist_ok=True)
    gdf.to_file(plots_path, driver="GPKG")

    centroids = gdf.geometry.centroid
    cdf = pd.DataFrame({
        "plot_id":   gdf["plot_id"].values,
        "group_key": gdf["group_key"].values,
        "lon":       centroids.x.values,
        "lat":       centroids.y.values,
    })
    cdf.to_csv(centroids_path, index=False)

    logger.info(f"[parse] saved {len(gdf)} polygons "
                f"(pos={int((gdf.label==1).sum())}, neg={int((gdf.label==0).sum())}) "
                f"from {gdf['group_key'].nunique()} unique source files → {plots_path}")
    return gdf


# ────────────────── Stage 2: extract ──────────────────

def stage_extract(gdf, processed_dir: Path, force: bool = False) -> pd.DataFrame:
    out = processed_dir / "extraction_wide.csv"
    if out.exists() and not force:
        df = pd.read_csv(out)
        logger.info(f"[extract] cached: {out} ({df.shape})")
        return df

    from data.polygon_extractor import PolygonExtractor

    ext = PolygonExtractor(config_path="config.yaml")
    df = ext.extract(gdf)
    df.to_csv(out, index=False)
    logger.info(f"[extract] saved {df.shape} to {out}")
    return df


# ────────────────── Stage 3: features ──────────────────

def stage_features(extraction_df, processed_dir: Path, force: bool = False) -> Path:
    out = processed_dir / "sugarcane_features.csv"
    if out.exists() and not force:
        logger.info(f"[features] cached: {out}")
        return out

    from features.feature_table_builder import FeatureTableBuilder
    builder = FeatureTableBuilder(config_path="config.yaml")
    feat = builder.build(extraction_df)
    feat.to_csv(out, index=False)
    logger.info(f"[features] saved {feat.shape} to {out}")
    return out


# ────────────────── Stage 4: train ──────────────────

def stage_train(features_csv: Path, centroids_csv: Path, model_out: Path,
                tune: bool = False, tune_trials: int = 50, n_splits: int = 5,
                seed: int = 42, quick: bool = False, season_invariant: bool = False):
    """Invoke train.py logic in-process (so we don't pay subprocess startup)."""
    import importlib
    
    metrics_name = model_out.stem.replace("best", "metrics")
    if "metrics" not in metrics_name:
        metrics_name = model_out.stem + "_metrics"
    metrics_out = model_out.parent / f"{metrics_name}.json"
    
    sys.argv = [
        "train.py",
        "--features",  str(features_csv),
        "--centroids", str(centroids_csv),
        "--model_out", str(model_out),
        "--metrics_out", str(metrics_out),
        "--n_splits",  str(n_splits),
        "--seed",      str(seed),
        "--tune_trials", str(tune_trials),
    ]
    if tune:
        sys.argv.append("--tune")
    if quick:
        sys.argv.append("--quick")
    if season_invariant:
        sys.argv.append("--season_invariant")

    train_mod = importlib.import_module("train")
    train_mod.main()


# ────────────────── Main ──────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kml_dir",  default="data/kml")
    ap.add_argument("--output",   default="data/processed/sugarcane_features.csv",
                    help="path to write the final feature CSV")
    ap.add_argument("--model_out", default="models/saved/best.pkl")
    ap.add_argument("--train",    action="store_true",
                    help="train models after building features")
    ap.add_argument("--tune",     action="store_true",
                    help="enable Optuna 50-trial tuning on the best base learner")
    ap.add_argument("--tune_trials", type=int, default=50)
    ap.add_argument("--force",    action="store_true",
                    help="force re-run of all stages (ignore caches)")
    ap.add_argument("--balance",  default="one_per_file",
                    choices=["one_per_file", "subsample_n", "none"],
                    help=("balance strategy: 'one_per_file' (default, 200:200), "
                          "'subsample_n', or 'none' (keep all 855)."))
    ap.add_argument("--quick",    action="store_true",
                    help="skip the stacker ensemble (RF/XGB/LGB only); faster iteration")
    ap.add_argument("--season_invariant", action="store_true",
                    help="train with ONLY season-invariant features. "
                         "Produces a model that works in any season.")
    ap.add_argument("--predict",  default=None,
                    help="run inference on a single KML (skip extraction/training)")
    ap.add_argument("--crop_date", default=None,
                    help="confirmed crop date YYYY-MM-DD (required with --predict)")
    args = ap.parse_args()

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("sampling", {}).get("random_seed", 42))
    set_seed(seed)

    processed_dir = Path(cfg.get("data", {}).get("processed_dir", "data/processed"))
    models_dir    = Path(cfg.get("data", {}).get("models_dir", "models/saved"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Inference-only mode
    if args.predict:
        if not args.crop_date:
            raise SystemExit("--crop_date YYYY-MM-DD is required with --predict")
        from inference.predictor import SugarcanePredictor
        p = SugarcanePredictor(model_path=str(models_dir / "best.pkl"),
                                config_path="config.yaml")
        res = p.predict(args.predict, args.crop_date)
        print(json.dumps(res, indent=2, default=str))
        return

    logger.info("=" * 64)
    logger.info(f"PIPELINE START  seed={seed}  force={args.force}")
    logger.info("=" * 64)

    # 1. parse + balance
    cfg_anchor = cfg.get("default_anchor_date", "2025-09-01")
    gdf = stage_parse(Path(args.kml_dir), processed_dir,
                      cfg_anchor=cfg_anchor,
                      balance_strategy=args.balance,
                      seed=seed,
                      force=args.force)

    # 2. extract (per-polygon zonal stats)
    extraction_df = stage_extract(gdf, processed_dir, force=args.force)

    # 3. features (indices + phenology + temporal stats + window-anchored)
    features_csv = stage_features(extraction_df, processed_dir, force=args.force)

    # Copy / symlink to user-requested --output if different
    if Path(args.output).resolve() != features_csv.resolve():
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copyfile(features_csv, args.output)
        logger.info(f"Feature CSV also written to {args.output}")

    # 4. train
    if args.train:
        centroids_csv = processed_dir / "centroids.csv"
        stage_train(
            features_csv=Path(args.output),
            centroids_csv=centroids_csv,
            model_out=Path(args.model_out),
            tune=args.tune,
            tune_trials=args.tune_trials,
            n_splits=int(cfg.get("models", {}).get("cv_folds", 5)),
            seed=seed,
            quick=args.quick,
            season_invariant=args.season_invariant,
        )

    logger.info("=" * 64)
    logger.info("PIPELINE DONE.")
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
