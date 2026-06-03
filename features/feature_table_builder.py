"""
features/feature_table_builder.py
=================================
Assemble the **final per-polygon feature CSV** from raw extractor output.

Inputs
------
A wide DataFrame produced by ``data.polygon_extractor.PolygonExtractor.extract``
with columns:
    plot_id, label, area_ha, source_file, anchor_date,
    date_start, date_end, state,
    B2_<tag>..B12_<tag>, VV_<tag>, VH_<tag>, valid_pixel_count_<tag>

Outputs
-------
A wide DataFrame with the *same* row count, augmented with:
    1. Spectral indices per window  (NDVI, EVI, NDRE, NDMI, GNDVI, LSWI, SAVI,
                                     MSAVI, NBR, NDWI, CIre, IRECI, CCCI,
                                     GRVI, PSRI, RVI, RFDI, CR, NRPB)
    2. Phenology curve-shape features (per VI):
         auc, peak_value, peak_month, greenup_half_month,
         max_greenup_slope, max_senescence_slope,
         season_length, n_growing_seasons, amplitude, asymmetry, smoothness
    3. SAR phenology features (VV, VH, RVI):
         sar_auc, sar_peak_value, sar_season_length, sar_std
    4. NDVI–VV temporal correlation
    5. Window-anchored sugarcane features (per VI in {NDVI, NDRE, NDMI}):
         {vi}_may, {vi}_aug_sep, {vi}_dec_jan, {vi}_amp_window,
         {vi}_rate_rise, {vi}_rate_fall
    6. Temporal stats per band/index (min/max/mean/median/p10/p25/p75/p90/std/range/cv)
    7. Seasonal contrast (monsoon vs dry mean & std)
    8. Inter-annual diff & stability (when window crosses 2 calendar years)
    9. SAR-optical ratio stats (VV/NDVI, VH/NDVI)

Final feature matrix dimensions
-------------------------------
Approximately N × ~1100 columns. We deliberately **drop**:
    - longitude / latitude  (prevents geographic-shortcut learning)
    - source_file, anchor_date, date_start, date_end (stored separately)
    - state (kept only for diagnostics, not for training)

Usage
-----
    from features.feature_table_builder import FeatureTableBuilder

    builder = FeatureTableBuilder(config_path="config.yaml")
    final_df = builder.build(extraction_df)
    final_df.to_csv("data/processed/sugarcane_features.csv", index=False)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Columns we strip from the final feature matrix (kept only as metadata).
# The training pipeline reads label as y and everything else (excluding
# these META_COLS) as X.
META_COLS = [
    # NOTE: ``area_ha`` is intentionally NOT here — it's a legitimate
    # agronomic feature (sugarcane fields in UP have a distinctive size
    # distribution). It goes into X.
    "plot_id", "group_key", "label",
    "source_file", "anchor_date", "date_start", "date_end", "state",
]

# Columns we explicitly drop — geographic shortcut prevention.
DROP_COLS = ["longitude", "latitude"]


class FeatureTableBuilder:
    """Assemble the final wide feature table from raw extractor output."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path

    # ------------------------------------------------------------------
    def build(self, extraction_df: pd.DataFrame) -> pd.DataFrame:
        from features.spectral_indices import SpectralIndexCalculator
        from features.phenology_features import PhenologyExtractor
        from features.temporal_stats import TemporalStatsExtractor
        from utils import (
            detect_time_tags, detect_absolute_time_tags,
        )

        df = extraction_df.copy()

        # Drop any geographic-shortcut columns if present
        for c in DROP_COLS:
            if c in df.columns:
                df = df.drop(columns=[c])

        # ── YEAR-INDEPENDENCE STEP ────────────────────────────────────────
        # Rename absolute calendar tags  <band>_YYYY_MM_DD  ->  <band>_tNN
        # using chronological order. The resulting model will be a function
        # of the SHAPE of the satellite time-series within the window, not
        # the specific year. This is what makes the model usable across
        # different growing seasons (2023, 2024, 2025, ...).
        abs_tags = detect_absolute_time_tags(df)
        if abs_tags:
            abs_tags_sorted = sorted(abs_tags)  # YYYY_MM_DD sorts chronologically
            rename = {}
            for i, atag in enumerate(abs_tags_sorted):
                new_tag = f"t{i:02d}"
                suffix_old = f"_{atag}"
                suffix_new = f"_{new_tag}"
                for col in df.columns:
                    if col.endswith(suffix_old):
                        rename[col] = col[: -len(suffix_old)] + suffix_new
            if rename:
                df = df.rename(columns=rename)
                logger.info(
                    f"Renamed {len(abs_tags_sorted)} absolute time tags "
                    f"-> relative tNN (year-independent feature schema)"
                )

        time_tags = detect_time_tags(df)
        if not time_tags:
            raise ValueError(
                "No time tags found in extraction DataFrame (expected tNN or YYYY_MM_DD)."
            )
        logger.info(f"Building features over {len(time_tags)} timesteps "
                    f"({time_tags[0]} \u2026 {time_tags[-1]})")

        # 1) Spectral indices — adds NDVI_<tag>, ..., NRPB_<tag>
        calc = SpectralIndexCalculator()
        df = calc.compute_all(df, time_tags=time_tags, inplace=True)

        # 2) Phenology features (vectorised) — adds curve-shape + window-anchored
        pheno = PhenologyExtractor(config_path=self.config_path)
        pheno_df = pheno.compute(df, time_tags=time_tags)
        pheno_df.index = df.index

        # 3) Temporal stats per band/index + seasonal + interannual + SAR-optical ratio
        tstats = TemporalStatsExtractor(config_path=self.config_path)
        stats_df = tstats.compute(df, time_tags=time_tags)
        # tstats might have prepended metadata columns — strip them so we don't dup
        stats_df = stats_df.loc[:, ~stats_df.columns.duplicated()]
        for c in META_COLS:
            if c in stats_df.columns:
                stats_df = stats_df.drop(columns=[c])
        stats_df.index = df.index

        # 4) SEASON-INVARIANT features — the core innovation.
        #    These features characterize WHAT the crop IS (duration of greenness,
        #    curve shape, spectral stability, harmonic signature) regardless of
        #    WHEN you look at it. This makes the model work in any season.
        from features.season_invariant_features import SeasonInvariantExtractor
        si_extractor = SeasonInvariantExtractor()
        si_df = si_extractor.compute(df, time_tags=time_tags)
        si_df.index = df.index
        logger.info(f"Season-invariant features: {si_df.shape[1]} columns")

        # Combine: per-window features + scalar features + season-invariant features
        final = pd.concat([df, pheno_df, stats_df, si_df], axis=1)

        # Final dedup
        final = final.loc[:, ~final.columns.duplicated()]

        # Columns audit
        n_total = final.shape[1]
        n_meta = sum(1 for c in META_COLS if c in final.columns)
        n_si = si_df.shape[1]
        logger.info(
            f"Final feature table: {final.shape[0]} rows × {n_total} columns "
            f"({n_meta} metadata + {n_si} season-invariant + "
            f"{n_total - n_meta - n_si} timestep-indexed)"
        )
        return final

    # ------------------------------------------------------------------
    @staticmethod
    def split_X_y_groups(
        df: pd.DataFrame,
        drop_quality_cols: bool = True,
        season_invariant_only: bool = False,
    ):
        """
        Helper to split a final feature DataFrame into (X, y, groups, feature_names).

        Parameters
        ----------
        drop_quality_cols : if True (DEFAULT), drop ``valid_pixel_count_*``,
                             ``pct_windows_with_valid_optical``, etc. These
                             leak plot-level data quality (geographic proxy).
        season_invariant_only : if True, keep ONLY columns prefixed with
                                 ``si_``, ``sar_``, ``cross_``, and ``area_ha``.
                                 This drops ALL timestep-indexed features and
                                 gives a model that works in any season.

        Returns
        -------
        X (DataFrame), y (np.ndarray int), groups (np.ndarray), feature_names (list[str])
        """
        df = df.copy()
        # Drop metadata cols
        feature_df = df.drop(columns=[c for c in META_COLS if c in df.columns])

        # Always drop quality/pixel-count columns (location/quality proxy leak)
        if drop_quality_cols:
            qcols = [c for c in feature_df.columns if c.startswith("valid_pixel_count_")]
            qcols += [c for c in ("pct_windows_with_valid_optical",
                                  "n_cloudy_windows", "n_total_windows")
                      if c in feature_df.columns]
            feature_df = feature_df.drop(columns=qcols, errors="ignore")

        # Season-invariant mode: keep only season-invariant features
        if season_invariant_only:
            si_prefixes = ("si_", "sar_", "cross_")
            si_cols = [c for c in feature_df.columns
                       if c.startswith(si_prefixes) or c == "area_ha"]
            # Also keep existing phenology features that are purely shape/stats-based
            pheno_suffixes = ("_auc", "_peak_value", "_season_length",
                             "_amplitude", "_smoothness", "_n_growing_seasons",
                             "_asymmetry", "_greenup_half_month", "_peak_month",
                             "_max_greenup_slope", "_max_senescence_slope",
                             "_sar_auc", "_sar_peak_value", "_sar_season_length", "_sar_std",
                             "_temporal_correlation",
                             "_mean", "_max", "_min", "_std", "_range", "_cv",
                             "_p10", "_p25", "_p75", "_p90", "_median",
                             "_interannual_diff", "_interannual_stability")
            for c in feature_df.columns:
                if c not in si_cols and any(c.endswith(s) for s in pheno_suffixes):
                    # Only keep if it's NOT a timestep-indexed column
                    # (timestep columns look like BAND_t00, BAND_2024_09_01)
                    import re
                    if not re.search(r'_t\d{2,3}$', c) and not re.search(r'_\d{4}_\d{2}_\d{2}$', c):
                        si_cols.append(c)
            feature_df = feature_df[si_cols] if si_cols else feature_df

        # Drop any non-numeric remnants (e.g. stray strings)
        feature_df = feature_df.select_dtypes(include=[np.number]).copy()
        # Final NaN/Inf cleanup
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        y = df["label"].astype(int).values
        # Prefer file-level grouping (group_key) over plot_id so multi-polygon
        # KMLs do not leak across folds. Fall back to plot_id if absent.
        if "group_key" in df.columns:
            groups = df["group_key"].astype(str).values
        elif "source_file" in df.columns:
            groups = df["source_file"].astype(str).values
        else:
            groups = df["plot_id"].astype(str).values
        feature_names = feature_df.columns.tolist()
        return feature_df, y, groups, feature_names
