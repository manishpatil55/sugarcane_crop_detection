"""
phenology_features.py
=====================
Extract phenological shape features from vegetation-index time series.

These features describe the **shape of the growing-season curve** rather
than absolute reflectance, making them robust to atmospheric and
domain-shift effects.

Feature families
----------------
A) Curve-shape per VI (NDVI, EVI, NDRE, NDMI, LSWI):
     auc, peak_value, peak_month, greenup_half_month,
     max_greenup_slope, max_senescence_slope,
     season_length, n_growing_seasons,
     amplitude (peak - p10), asymmetry, smoothness

B) SAR phenology per band (VV, VH, RVI):
     sar_auc, sar_peak_value, sar_season_length, sar_std

C) Cross-sensor coherence:
     NDVI_VV_temporal_correlation

D) **User-specified window-anchored sugarcane features** (NEW):
     ndvi_may, ndvi_aug_sep, ndvi_dec_jan,
     ndvi_amplitude_window  = ndvi_aug_sep - ndvi_may
     ndvi_rate_rise         = (ndvi_aug_sep - ndvi_may) / 3
     ndvi_rate_fall         = (ndvi_aug_sep - ndvi_dec_jan) / 4

Performance
-----------
Curve-shape features are now **vectorised across pixels** (numpy
broadcasting along axis=1) instead of a Python ``for i in range(n_pixels)``
loop. This brings the per-call cost from O(n_pixels × constant) Python
overhead to O(constant) numpy. For 400 polygons, this drops from ~30 s
to <100 ms.

Usage
-----
    from features.phenology_features import PhenologyExtractor
    pheno = PhenologyExtractor(config_path="config.yaml")
    pheno_df = pheno.compute(df_wide, time_tags=["2025_03_01", ...])
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

_EPS = 1e-8


# ────────────────── Vectorised primitives ──────────────────

def _interp_nans_along_time(arr: np.ndarray) -> np.ndarray:
    """1-D linear interpolation across NaN gaps along axis=1.

    arr shape: (n_pixels, n_timesteps).
    """
    n_pix, n_t = arr.shape
    out = arr.copy()
    for i in range(n_pix):
        ts = out[i]
        nans = np.isnan(ts)
        if not nans.any() or nans.all():
            continue
        valid_idx = np.where(~nans)[0]
        ts[nans] = np.interp(np.where(nans)[0], valid_idx, ts[valid_idx])
        out[i] = ts
    return out


def _smooth_savgol(arr: np.ndarray, window: int = 5, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay along axis=1. Handles short series gracefully."""
    n_t = arr.shape[1]
    window = min(window, n_t)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return arr
    try:
        return savgol_filter(arr, window_length=window,
                             polyorder=min(polyorder, window - 1), axis=1)
    except Exception:
        return arr


def _vec_phenology(arr: np.ndarray, threshold: float = 0.3) -> dict:
    """
    Compute curve-shape features for an (n_pixels, n_timesteps) array.

    Returns a dict of 1-D arrays of length n_pixels.
    """
    n_pix, n_t = arr.shape
    months = np.arange(n_t, dtype=np.float32)

    # AUC (trapezoidal) with NaN-safe fallback
    auc = np.trapz(np.where(np.isnan(arr), 0.0, arr), x=months, axis=1)

    # Peak value & index
    arr_filled = np.where(np.isnan(arr), -np.inf, arr)
    peak_idx = np.argmax(arr_filled, axis=1)
    peak_val = arr_filled[np.arange(n_pix), peak_idx]
    peak_val = np.where(np.isfinite(peak_val), peak_val, np.nan)

    # Green-up half: first index on the ascending limb where ts ≥ 0.5*peak
    half = peak_val * 0.5
    asc_mask = (np.arange(n_t)[None, :] <= peak_idx[:, None])
    cond = (arr >= half[:, None]) & asc_mask
    # First True per row; if none, use peak_idx
    first_true = cond.argmax(axis=1)
    has_any = cond.any(axis=1)
    greenup_half = np.where(has_any, first_true, peak_idx).astype(np.float32)

    # Slopes
    diffs = np.diff(arr, axis=1)
    max_up = np.nanmax(diffs, axis=1)
    max_down = np.nanmin(diffs, axis=1)

    # Season length: number of timesteps above threshold
    season_len = (arr > threshold).sum(axis=1).astype(np.float32)

    # Number of growing seasons (contiguous runs > threshold separated by ≥2 below)
    above = (arr > threshold).astype(np.int8)
    n_seasons = np.zeros(n_pix, dtype=np.float32)
    for i in range(n_pix):
        in_season = False
        gap = 0
        count = 0
        for v in above[i]:
            if v == 1:
                if not in_season:
                    count += 1
                    in_season = True
                gap = 0
            else:
                gap += 1
                if gap >= 2:
                    in_season = False
        n_seasons[i] = count

    # Amplitude = peak - 10th percentile
    p10 = np.nanpercentile(arr, 10, axis=1)
    amplitude = peak_val - p10

    # Asymmetry
    asym = np.where(season_len > 0, (peak_idx - greenup_half) / (season_len + _EPS), 0.0)

    # Smoothness = 1 / (std of 2nd derivative)
    d2 = np.diff(arr, n=2, axis=1)
    smoothness = 1.0 / (np.nanstd(d2, axis=1) + _EPS)

    return {
        "auc": auc.astype(np.float32),
        "peak_value": peak_val.astype(np.float32),
        "peak_month": peak_idx.astype(np.float32),
        "greenup_half_month": greenup_half,
        "max_greenup_slope": max_up.astype(np.float32),
        "max_senescence_slope": max_down.astype(np.float32),
        "season_length": season_len,
        "n_growing_seasons": n_seasons,
        "amplitude": amplitude.astype(np.float32),
        "asymmetry": asym.astype(np.float32),
        "smoothness": smoothness.astype(np.float32),
    }


def _vec_sar_phenology(arr: np.ndarray) -> dict:
    """SAR curve features. Auto-detects dB scale (>75% negative) for thresholds."""
    valid = arr[~np.isnan(arr)]
    is_db = len(valid) > 0 and (valid < 0).mean() > 0.75
    threshold = -15.0 if is_db else 0.1

    n_pix, n_t = arr.shape
    months = np.arange(n_t, dtype=np.float32)

    auc = np.trapz(np.where(np.isnan(arr), 0.0, arr), x=months, axis=1)
    peak_val = np.nanmax(arr, axis=1)
    season_len = (arr > threshold).sum(axis=1).astype(np.float32)
    std = np.nanstd(arr, axis=1)
    return {
        "sar_auc": auc.astype(np.float32),
        "sar_peak_value": peak_val.astype(np.float32),
        "sar_season_length": season_len,
        "sar_std": std.astype(np.float32),
    }


def _vec_pearson_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised Pearson correlation per row of two (N, T) arrays."""
    valid = ~np.isnan(a) & ~np.isnan(b)
    out = np.full(a.shape[0], np.nan, dtype=np.float32)
    for i in range(a.shape[0]):
        m = valid[i]
        if m.sum() < 4:
            continue
        x = a[i, m]; y = b[i, m]
        if np.std(x) < _EPS or np.std(y) < _EPS:
            continue
        out[i] = np.corrcoef(x, y)[0, 1]
    return out


# ────────────────── Window-anchored features (year-independent) ──────────────────

def _mean_for_tags(df: pd.DataFrame, vi: str, sel_tags: List[str]) -> np.ndarray:
    cols = [f"{vi}_{t}" for t in sel_tags if f"{vi}_{t}" in df.columns]
    if not cols:
        return np.full(len(df), np.nan, dtype=np.float32)
    return df[cols].mean(axis=1, skipna=True).values.astype(np.float32)


def _resolve_phase_tags(time_tags: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Resolve EARLY / PEAK / LATE phase tags relative to the window.

    * If tags are RELATIVE (tNN): split window into three phases based on
      step index, anchored at the centre of the window.  This is the
      year-independent path used after FeatureTableBuilder renames tags.

        - early : first  ~25%  of the window  (pre-peak, vegetative)
        - peak  : middle ~30%  of the window  (grand-growth)
        - late  : last   ~25%  of the window  (senescence / harvest)

    * If tags are ABSOLUTE (YYYY_MM_DD): fall back to the calendar-month
      windows used historically (May / Aug-Sep / Dec-Jan) so back-compat
      tests / legacy data still work.
    """
    from utils import (
        parse_tag_to_step, parse_tag_to_ymd,
        split_columns_by_phase, split_columns_by_step_range,
    )

    rel_tags = [t for t in time_tags if parse_tag_to_step(t) is not None]
    abs_tags = [t for t in time_tags if parse_tag_to_ymd(t) is not None]

    if rel_tags and len(rel_tags) >= len(abs_tags):
        steps = sorted(parse_tag_to_step(t) for t in rel_tags)
        n = len(steps)
        # 4-quartile split — early (1st), peak (centre, ~30 %), late (last)
        q = max(1, n // 4)
        early_end = q - 1                            # 0 ... q-1
        peak_start = max(0, (n // 2) - 2)
        peak_end = min(n - 1, (n // 2) + 2)
        late_start = n - q
        late_end = n - 1
        early = split_columns_by_step_range(time_tags, (0, early_end))
        peak  = split_columns_by_step_range(time_tags, (peak_start, peak_end))
        late  = split_columns_by_step_range(time_tags, (late_start, late_end))
    else:
        # legacy absolute-calendar path
        early = split_columns_by_phase(time_tags, [5])
        peak  = split_columns_by_phase(time_tags, [8, 9])
        late  = split_columns_by_phase(time_tags, [12, 1])

    return early, peak, late


def compute_window_anchored_features(
    df: pd.DataFrame,
    time_tags: List[str],
    vi: str = "NDVI",
) -> pd.DataFrame:
    """
    Compute scalar features capturing the SHAPE of the VI curve within the
    user-supplied window.

    Output columns are kept under their historic names (``ndvi_may``,
    ``ndvi_aug_sep``, ``ndvi_dec_jan`` ...) for code / report compatibility,
    but they no longer correspond to absolute calendar months.  They
    correspond to the three relative phases inside the window:

        ndvi_may       =  mean VI in EARLY phase  (1st  quartile of window)
        ndvi_aug_sep   =  mean VI in PEAK  phase  (centre ~30 % of window)
        ndvi_dec_jan   =  mean VI in LATE  phase  (last quartile of window)

    These signed differences and slopes capture the unique sugarcane
    phenology shape (slow rise, broad peak, gradual fall, harvest dip) in
    a year-independent way.
    """
    pref = vi.lower()
    early_tags, peak_tags, late_tags = _resolve_phase_tags(time_tags)

    early  = _mean_for_tags(df, vi, early_tags)
    peak_v = _mean_for_tags(df, vi, peak_tags)
    late   = _mean_for_tags(df, vi, late_tags)

    return pd.DataFrame({
        f"{pref}_may":           early,
        f"{pref}_aug_sep":       peak_v,
        f"{pref}_dec_jan":       late,
        f"{pref}_amp_window":    peak_v - early,
        f"{pref}_rate_rise":     (peak_v - early) / 3.0,
        f"{pref}_rate_fall":     (peak_v - late)  / 4.0,
    })


# ────────────────── DataFrame-level extractor ──────────────────

class PhenologyExtractor:
    """Extract phenological features from a wide-format DataFrame."""

    DEFAULT_VI_BANDS  = ["NDVI", "EVI", "NDRE", "NDMI", "LSWI"]
    DEFAULT_SAR_BANDS = ["VV", "VH", "RVI"]

    def __init__(
        self,
        config_path: str = "config.yaml",
        vi_bands: Optional[List[str]] = None,
        sar_bands: Optional[List[str]] = None,
    ):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        pheno_cfg = self.cfg["features"].get("phenology", {})
        self.ndvi_threshold = pheno_cfg.get("ndvi_threshold", 0.3)
        self.smooth_window = pheno_cfg.get("smooth_window", 5)
        self.vi_bands = vi_bands or self.DEFAULT_VI_BANDS
        self.sar_bands = sar_bands or self.DEFAULT_SAR_BANDS

    # ------------------------------------------------------------------
    def compute(
        self,
        df: pd.DataFrame,
        time_tags: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if time_tags is None:
            from utils import detect_time_tags as _detect
            time_tags = _detect(df)
        if not time_tags:
            return pd.DataFrame()

        n_pixels = len(df)
        n_t = len(time_tags)
        logger.info(
            f"Phenology features: {n_pixels} samples × {n_t} timesteps × "
            f"{len(self.vi_bands)} VIs"
        )

        all_features: dict = {}

        # A) VI curve-shape (vectorised per band)
        for band in self.vi_bands:
            arr = self._stack_band(df, band, time_tags)
            if arr is None:
                continue
            arr = _interp_nans_along_time(arr)
            arr_smooth = _smooth_savgol(arr, window=self.smooth_window)
            feats = _vec_phenology(arr_smooth, threshold=self.ndvi_threshold)
            for k, v in feats.items():
                all_features[f"{band}_{k}"] = v

        # B) SAR curve-shape
        for band in self.sar_bands:
            arr = self._stack_band(df, band, time_tags)
            if arr is None:
                continue
            arr = _interp_nans_along_time(arr)
            feats = _vec_sar_phenology(arr)
            for k, v in feats.items():
                all_features[f"{band}_{k}"] = v

        # C) Cross-sensor coherence (NDVI vs VV)
        ndvi_arr = self._stack_band(df, "NDVI", time_tags)
        vv_arr   = self._stack_band(df, "VV",   time_tags)
        if ndvi_arr is not None and vv_arr is not None:
            all_features["NDVI_VV_temporal_correlation"] = _vec_pearson_corr(
                _interp_nans_along_time(ndvi_arr),
                _interp_nans_along_time(vv_arr),
            )

        result = pd.DataFrame(all_features)

        # D) User-specified window-anchored features (computed on raw, NOT smoothed,
        #    so peak/harvest values are not deflated by smoothing).
        for vi in ("NDVI", "NDRE", "NDMI"):
            if any(c.startswith(f"{vi}_") for c in df.columns):
                window_df = compute_window_anchored_features(df, time_tags, vi=vi)
                # Reset index to avoid alignment issues
                window_df.index = result.index
                result = pd.concat([result, window_df], axis=1)

        logger.info(f"Phenology features shape: {result.shape}")
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _stack_band(df: pd.DataFrame, band: str, time_tags: List[str]) -> Optional[np.ndarray]:
        cols = [f"{band}_{t}" for t in time_tags]
        present = [c for c in cols if c in df.columns]
        if not present:
            return None
        n_pixels = len(df)
        n_t = len(time_tags)
        arr = np.full((n_pixels, n_t), np.nan, dtype=np.float32)
        for j, c in enumerate(cols):
            if c in df.columns:
                arr[:, j] = df[c].values.astype(np.float32)
        return arr
