"""
temporal_stats.py
=================
Compute multi-temporal aggregation statistics from a wide-format pixel
time-series DataFrame.

For each band/index, the following statistics are computed across all
available time steps (NaN-safe):
  min, max, mean, median, p10, p25, p75, p90, std, range, cv (coeff. of variation)

Additionally computes:
  - SAR-optical ratio statistics (VV/NDVI, VH/NDVI) to capture crop structure
  - Monsoon vs. dry season split statistics (Jun–Sep vs. Oct–May)
  - Inter-annual difference statistics (year2 - year1 mean)

Output: a 2D feature table (n_pixels × n_stat_features) suitable for RF/XGBoost.

Usage
-----
    from features.temporal_stats import TemporalStatsExtractor

    extractor = TemporalStatsExtractor(config_path="config.yaml")
    df_stats = extractor.compute(df_wide, time_tags=["2022_01", ...])
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Monsoon months (June–September)
_MONSOON_MONTHS = {6, 7, 8, 9}
# Dry season months (October–May)
_DRY_MONTHS = {10, 11, 12, 1, 2, 3, 4, 5}


def _parse_tag(tag: str, n_total_steps: int = 21) -> Tuple[int, int]:
    """
    Parse a time tag -> (year, month).

    Supports two formats:
      * Absolute YYYY_MM_DD -> exact (year, month).
      * Relative tNN        -> synthetic (year, month) so seasonal /
        inter-annual code keeps working.  The middle step is mapped to
        ``anchor_month = 9`` (September) so the first half of the
        window is monsoon-equivalent and the second half is dry-equivalent.
    """
    from utils import parse_tag_to_ymd, parse_tag_to_step, step_to_synthetic_month

    ymd = parse_tag_to_ymd(tag)
    if ymd is not None:
        return ymd[0], ymd[1]

    step = parse_tag_to_step(tag)
    if step is not None:
        synth_month = step_to_synthetic_month(step, n_total_steps, anchor_month=9)
        synth_year = 2025 if step < n_total_steps // 2 else (
            2025 if synth_month >= 9 else 2026
        )
        return synth_year, synth_month

    # Legacy YYYY_MM fallback
    parts = tag.split("_")
    try:
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return 2025, 1


class TemporalStatsExtractor:
    """
    Extract temporal statistics from a wide-format pixel DataFrame.

    Parameters
    ----------
    config_path : path to config.yaml
    bands       : list of band/index names to process; if None, auto-detected
    stats       : list of stat names to compute; if None, uses config defaults
    """

    DEFAULT_STATS = [
        "min", "max", "mean", "median",
        "p10", "p25", "p75", "p90",
        "std", "range", "cv",
    ]

    def __init__(
        self,
        config_path: str = "config.yaml",
        bands: Optional[List[str]] = None,
        stats: Optional[List[str]] = None,
    ):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.stats = stats or self.cfg["features"].get("temporal_stats", self.DEFAULT_STATS)
        self.bands = bands  # None = auto-detect from DataFrame

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        df: pd.DataFrame,
        time_tags: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute temporal statistics for all bands across all time steps.

        Parameters
        ----------
        df        : wide-format DataFrame (pixels × band_YYYY_MM columns)
        time_tags : list of "YYYY_MM" strings; if None, auto-detected

        Returns
        -------
        DataFrame of shape (n_pixels, n_bands × n_stats + seasonal_stats)
        Preserves longitude, latitude, state, label, plot_id if present.
        """
        if time_tags is None:
            time_tags = self._detect_time_tags(df)

        if not time_tags:
            raise ValueError("No time tags found in DataFrame columns.")

        bands = self.bands or self._detect_bands(df, time_tags)
        logger.info(
            f"Computing temporal stats: {len(bands)} bands × "
            f"{len(self.stats)} stats × {len(time_tags)} time steps"
        )

        stat_frames = []

        # ---- Per-band statistics ----
        for band in bands:
            band_cols = [f"{band}_{tag}" for tag in time_tags if f"{band}_{tag}" in df.columns]
            if not band_cols:
                continue
            band_arr = df[band_cols].values.astype(np.float32)
            stat_df = self._compute_stats_for_band(band_arr, band)
            stat_frames.append(stat_df)

        # ---- Seasonal split statistics ----
        seasonal_df = self._compute_seasonal_stats(df, bands, time_tags)
        stat_frames.append(seasonal_df)

        # ---- Inter-annual difference statistics ----
        interannual_df = self._compute_interannual_stats(df, bands, time_tags)
        stat_frames.append(interannual_df)

        # ---- SAR-optical ratio statistics ----
        sar_optical_df = self._compute_sar_optical_stats(df, time_tags)
        stat_frames.append(sar_optical_df)

        # Combine all stat features
        result = pd.concat(stat_frames, axis=1)

        # Prepend metadata columns if present
        meta_cols = [c for c in ["longitude", "latitude", "state", "label", "plot_id"]
                     if c in df.columns]
        if meta_cols:
            result = pd.concat([df[meta_cols].reset_index(drop=True), result], axis=1)

        logger.info(f"Temporal stats shape: {result.shape}")
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_stats_for_band(
        self, arr: np.ndarray, band_name: str
    ) -> pd.DataFrame:
        """
        Compute configured statistics for a (n_pixels, n_timesteps) array.
        NaN-safe using numpy nanfunctions.
        """
        rows = {}
        with np.errstate(all="ignore"):
            for stat in self.stats:
                col = f"{band_name}_{stat}"
                if stat == "min":
                    rows[col] = np.nanmin(arr, axis=1)
                elif stat == "max":
                    rows[col] = np.nanmax(arr, axis=1)
                elif stat == "mean":
                    rows[col] = np.nanmean(arr, axis=1)
                elif stat == "median":
                    rows[col] = np.nanmedian(arr, axis=1)
                elif stat == "p10":
                    rows[col] = np.nanpercentile(arr, 10, axis=1)
                elif stat == "p25":
                    rows[col] = np.nanpercentile(arr, 25, axis=1)
                elif stat == "p75":
                    rows[col] = np.nanpercentile(arr, 75, axis=1)
                elif stat == "p90":
                    rows[col] = np.nanpercentile(arr, 90, axis=1)
                elif stat == "std":
                    rows[col] = np.nanstd(arr, axis=1)
                elif stat == "range":
                    rows[col] = np.nanmax(arr, axis=1) - np.nanmin(arr, axis=1)
                elif stat == "cv":
                    mean = np.nanmean(arr, axis=1)
                    std = np.nanstd(arr, axis=1)
                    rows[col] = np.where(np.abs(mean) > _EPS, std / (np.abs(mean) + _EPS), 0.0)

        return pd.DataFrame(rows)

    def _compute_seasonal_stats(
        self,
        df: pd.DataFrame,
        bands: List[str],
        time_tags: List[str],
    ) -> pd.DataFrame:
        """
        Compute mean and std separately for monsoon (Jun–Sep) and dry (Oct–May) months.
        This captures the seasonal phenological contrast that distinguishes sugarcane
        from other crops.
        """
        rows = {}
        n_steps = len(time_tags)
        for band in bands:
            monsoon_cols = [
                f"{band}_{tag}" for tag in time_tags
                if f"{band}_{tag}" in df.columns and _parse_tag(tag, n_steps)[1] in _MONSOON_MONTHS
            ]
            dry_cols = [
                f"{band}_{tag}" for tag in time_tags
                if f"{band}_{tag}" in df.columns and _parse_tag(tag, n_steps)[1] in _DRY_MONTHS
            ]

            if monsoon_cols:
                arr = df[monsoon_cols].values.astype(np.float32)
                rows[f"{band}_monsoon_mean"] = np.nanmean(arr, axis=1)
                rows[f"{band}_monsoon_std"] = np.nanstd(arr, axis=1)

            if dry_cols:
                arr = df[dry_cols].values.astype(np.float32)
                rows[f"{band}_dry_mean"] = np.nanmean(arr, axis=1)
                rows[f"{band}_dry_std"] = np.nanstd(arr, axis=1)

            # Seasonal contrast: dry_mean - monsoon_mean
            if monsoon_cols and dry_cols:
                m_arr = df[monsoon_cols].values.astype(np.float32)
                d_arr = df[dry_cols].values.astype(np.float32)
                rows[f"{band}_seasonal_contrast"] = (
                    np.nanmean(d_arr, axis=1) - np.nanmean(m_arr, axis=1)
                )

        return pd.DataFrame(rows)

    def _compute_interannual_stats(
        self,
        df: pd.DataFrame,
        bands: List[str],
        time_tags: List[str],
    ) -> pd.DataFrame:
        """
        Compute year-over-year difference in mean values.
        Captures perennial crop stability (sugarcane is perennial/ratooned → low inter-annual change).
        """
        rows = {}
        n_steps = len(time_tags)
        years = sorted(set(_parse_tag(t, n_steps)[0] for t in time_tags))

        if len(years) < 2:
            return pd.DataFrame()

        year1, year2 = years[0], years[1]

        for band in bands:
            y1_cols = [
                f"{band}_{tag}" for tag in time_tags
                if f"{band}_{tag}" in df.columns and _parse_tag(tag, n_steps)[0] == year1
            ]
            y2_cols = [
                f"{band}_{tag}" for tag in time_tags
                if f"{band}_{tag}" in df.columns and _parse_tag(tag, n_steps)[0] == year2
            ]

            if y1_cols and y2_cols:
                y1_mean = np.nanmean(df[y1_cols].values.astype(np.float32), axis=1)
                y2_mean = np.nanmean(df[y2_cols].values.astype(np.float32), axis=1)
                rows[f"{band}_interannual_diff"] = y2_mean - y1_mean
                rows[f"{band}_interannual_stability"] = np.exp(
                    -np.abs(y2_mean - y1_mean)
                )

        return pd.DataFrame(rows)

    def _compute_sar_optical_stats(
        self,
        df: pd.DataFrame,
        time_tags: List[str],
    ) -> pd.DataFrame:
        """
        Compute SAR-optical ratio statistics.
        VV/NDVI and VH/NDVI ratios help distinguish sugarcane canopy structure
        from other vegetation types.
        """
        rows = {}
        for tag in time_tags:
            vv_col = f"VV_{tag}"
            vh_col = f"VH_{tag}"
            ndvi_col = f"NDVI_{tag}"

            if vv_col in df.columns and ndvi_col in df.columns:
                vv = df[vv_col].values.astype(np.float32)
                ndvi_vals = df[ndvi_col].values.astype(np.float32)
                rows[f"VV_NDVI_ratio_{tag}"] = np.clip(vv / (ndvi_vals + _EPS), -50.0, 50.0)

            if vh_col in df.columns and ndvi_col in df.columns:
                vh = df[vh_col].values.astype(np.float32)
                ndvi_vals = df[ndvi_col].values.astype(np.float32)
                rows[f"VH_NDVI_ratio_{tag}"] = np.clip(vh / (ndvi_vals + _EPS), -50.0, 50.0)

        if not rows:
            return pd.DataFrame()

        ratio_df = pd.DataFrame(rows)
        # Aggregate ratio stats
        stat_rows = {}
        for prefix in ["VV_NDVI_ratio", "VH_NDVI_ratio"]:
            cols = [c for c in ratio_df.columns if c.startswith(prefix)]
            if cols:
                arr = ratio_df[cols].values.astype(np.float32)
                stat_rows[f"{prefix}_mean"] = np.nanmean(arr, axis=1)
                stat_rows[f"{prefix}_std"] = np.nanstd(arr, axis=1)
                stat_rows[f"{prefix}_p50"] = np.nanmedian(arr, axis=1)

        return pd.DataFrame(stat_rows)

    @staticmethod
    def _detect_time_tags(df: pd.DataFrame) -> List[str]:
        """Detect canonical YYYY_MM_DD time tags via the shared helper."""
        from utils import detect_time_tags as _detect
        return _detect(df)

    @staticmethod
    def _detect_bands(df: pd.DataFrame, time_tags: List[str]) -> List[str]:
        """Detect band/index prefixes via the shared helper."""
        from utils import detect_bands as _bands
        return _bands(df, time_tags)


# Small epsilon for division
_EPS = 1e-8
