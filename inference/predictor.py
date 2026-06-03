"""
inference/predictor.py
======================
End-to-end sugarcane crop-detection inference for Uttar Pradesh.

This is a complete rewrite of the previous predictor. Key design points:

1. **Positive-only mode** — does NOT require external negative KMLs.
2. **Polygon-level inference** — uses ``data.polygon_extractor.PolygonExtractor``
   to compute zonal medians, then ``features.feature_table_builder`` to
   assemble the same feature set used during training.
3. **Persisted threshold** — loads the Youden-J optimal threshold from
   the saved model artefact (``optimal_threshold`` key).
4. **Real cloud-quality reporting** — uses ``valid_pixel_count_*`` columns
   from the extractor, no hard-coded values.

CLI
---
    python inference/predictor.py path/to/farm.kml 2025-09-01

Programmatic
------------
    from inference.predictor import SugarcanePredictor
    p = SugarcanePredictor()
    res = p.predict("farm.kml", crop_date="2025-09-01")
    print(res["sugarcane_probability"], res["classification"])
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import joblib
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def _crop_date_to_window(crop_date: str, months_before: int = 5, months_after: int = 5
                         ) -> Tuple[str, str]:
    """Convert single date → (start, end) bracketing the sugarcane phenology."""
    from dateutil.relativedelta import relativedelta
    anchor = datetime.strptime(crop_date, "%Y-%m-%d").date()
    start = anchor - relativedelta(months=months_before)
    end = anchor + relativedelta(months=months_after)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


class SugarcanePredictor:
    """Polygon-level sugarcane predictor."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: str = "config.yaml",
    ):
        self.config_path = config_path
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.model_path = (
            model_path
            or self.cfg.get("inference", {}).get("model_path", "models/saved/best.pkl")
        )
        self.optimal_threshold = float(
            self.cfg.get("inference", {}).get("probability_threshold", 0.5)
        )
        self.model_type = self.cfg.get("inference", {}).get("active_model", "unknown")

        self.model = None
        self.feature_names: List[str] = []
        self._models_loaded = False

    # ------------------------------------------------------------------
    def load_models(self):
        if self._models_loaded:
            return
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Model artefact missing: {self.model_path}\n"
                "Run  `python pipeline.py --train`  first."
            )
        artefact = joblib.load(self.model_path)
        if isinstance(artefact, dict) and "model" in artefact:
            self.model = artefact["model"]
            self.feature_names = artefact.get("feature_names", [])
            self.optimal_threshold = float(artefact.get(
                "optimal_threshold", self.optimal_threshold
            ))
            self.model_type = str(artefact.get("best_model_name", self.model_type))
        else:
            self.model = artefact
        self._models_loaded = True
        logger.info(
            f"Loaded model: {self.model_type}  "
            f"({len(self.feature_names)} features, threshold={self.optimal_threshold:.4f})"
        )

    # ------------------------------------------------------------------
    def predict(
        self,
        kml_path: str,
        crop_date: str,
        months_before: Optional[int] = None,
        months_after: Optional[int] = None,
    ) -> Dict:
        """
        Run end-to-end inference on a single KML containing one or more
        sugarcane-candidate polygons.
        """
        self.load_models()

        # Resolve window length from config if not explicitly passed
        if months_before is None:
            months_before = int(self.cfg.get("compositing", {}).get("months_before", 5))
        if months_after is None:
            months_after = int(self.cfg.get("compositing", {}).get("months_after", 5))

        # 1. Parse KML -> GeoDataFrame
        from data.kml_parser import KMLParser
        parser = KMLParser(months_before=months_before, months_after=months_after)
        gdf = parser.parse_file(kml_path)

        # Override anchor & window to the user-supplied date
        ws, we = _crop_date_to_window(crop_date, months_before, months_after)
        logger.info(
            f"Inference window: -{months_before}m / +{months_after}m around {crop_date} "
            f"\u2192 [{ws} \u2192 {we}]"
        )
        gdf["anchor_date"] = crop_date
        gdf["date_start"] = ws
        gdf["date_end"] = we
        gdf["label"] = 1  # placeholder for the extractor schema; not used for prediction

        # 2. Extract polygon-level zonal stats
        from data.polygon_extractor import PolygonExtractor
        ext = PolygonExtractor(config_path=self.config_path)
        wide = ext.extract(gdf)

        # 3. Build features
        from features.feature_table_builder import FeatureTableBuilder
        builder = FeatureTableBuilder(config_path=self.config_path)
        feat_df = builder.build(wide)

        # 4. Align to training feature names (add missing → 0, drop extras)
        from features.feature_table_builder import META_COLS
        # Detach meta to keep for response payload
        meta = feat_df[[c for c in META_COLS if c in feat_df.columns]].copy()
        X = feat_df.drop(columns=[c for c in META_COLS if c in feat_df.columns],
                          errors="ignore")
        for c in self.feature_names:
            if c not in X.columns:
                X[c] = 0.0
        # Reorder to training order
        X = X[self.feature_names].astype(np.float32)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 5. Predict probability per polygon
        proba = self.model.predict_proba(X.values)[:, 1]
        binary = (proba >= self.optimal_threshold).astype(int)

        # 6. Cloud-quality stats (real, not hardcoded)
        n_total = int(wide["n_total_windows"].iloc[0]) if "n_total_windows" in wide.columns else None
        cloud_pct = float(wide["pct_windows_with_valid_optical"].mean()) if "pct_windows_with_valid_optical" in wide.columns else None
        n_cloudy = int(wide["n_cloudy_windows"].mean()) if "n_cloudy_windows" in wide.columns else None

        # 7. Aggregate to plot-level summary
        n_polys = len(meta)
        sugarcane_polys = int(binary.sum())
        sugarcane_fraction = sugarcane_polys / max(n_polys, 1)

        result = {
            # Headline
            "sugarcane_probability_mean": float(np.mean(proba)),
            "sugarcane_probability_max":  float(np.max(proba)),
            "sugarcane_probability_min":  float(np.min(proba)),
            "classification":             "SUGARCANE" if sugarcane_fraction >= 0.5 else "NON-SUGARCANE",
            "label":                      int(sugarcane_fraction >= 0.5),
            "is_sugarcane":               bool(sugarcane_fraction >= 0.5),
            "confidence":                 float(np.mean(proba)),

            # Plot-level
            "n_polygons":         n_polys,
            "sugarcane_polygons": sugarcane_polys,
            "sugarcane_fraction": sugarcane_fraction,
            "per_polygon": [
                {
                    "plot_id":     str(meta.iloc[i]["plot_id"]),
                    "area_ha":     float(meta.iloc[i].get("area_ha", 0.0)),
                    "probability": float(proba[i]),
                    "label":       int(binary[i]),
                }
                for i in range(n_polys)
            ],

            # Window / date metadata
            "crop_date":    crop_date,
            "window_start": ws,
            "window_end":   we,
            "model_name":   self.model_type,
            "threshold":    float(self.optimal_threshold),

            # Cloud quality (REAL, not hardcoded)
            "cloud_quality": {
                "pct_windows_with_valid_optical": cloud_pct,
                "n_cloudy_windows":               n_cloudy,
                "n_total_windows":                n_total,
            },

            # Geo metadata
            "state": str(meta["state"].iloc[0]) if "state" in meta.columns and n_polys else "",
            "n_pixels": int(wide.get("valid_pixel_count_" + (sorted([c.replace("valid_pixel_count_","") for c in wide.columns if c.startswith("valid_pixel_count_")])[0] if any(c.startswith("valid_pixel_count_") for c in wide.columns) else ""), 0).sum()) if any(c.startswith("valid_pixel_count_") for c in wide.columns) else 0,
            "status": "success",
        }
        logger.info(
            f"Result: {result['classification']}  "
            f"prob_mean={result['sugarcane_probability_mean']:.3f}  "
            f"fraction={result['sugarcane_fraction']*100:.1f}%"
        )
        return result


# ────────────────── CLI ──────────────────

def _main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    ap = argparse.ArgumentParser(
        description="Sugarcane crop detection inference — provide a KML and a confirmed crop date."
    )
    ap.add_argument("kml_path",   help="path to .kml or .kmz file")
    ap.add_argument("crop_date",  help="confirmed crop date YYYY-MM-DD")
    ap.add_argument("--config",     default="config.yaml")
    ap.add_argument("--model_path", default=None)
    args = ap.parse_args()

    p = SugarcanePredictor(model_path=args.model_path, config_path=args.config)
    res = p.predict(args.kml_path, args.crop_date)
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    _main()
