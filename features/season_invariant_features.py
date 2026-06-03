"""
features/season_invariant_features.py
=====================================
Season-invariant feature extraction for sugarcane detection.

DESIGN PHILOSOPHY
-----------------
The previous feature set used timestep-indexed features (NDVI_t01, NDVI_t02, ...)
which are POSITION-DEPENDENT — they only work if the temporal window is aligned
to the same calendar months as training. This makes the model fail when queried
at a different time of year.

This module extracts features that describe WHAT the crop IS, regardless of
WHEN you look at it:

1. **Duration Features**: How long does the crop maintain high NDVI?
   Sugarcane: 6-10 months continuously above 0.5
   Rice: 3-4 months, Wheat: 3-4 months, Bare land: 0

2. **Shape Features**: What does the growth curve LOOK like?
   Sugarcane: slow rise → long plateau → gradual fall → sharp harvest dip
   Rice: rapid rise → sharp peak → rapid fall

3. **Stability Features**: How stable is the vegetation signal?
   Sugarcane: very low coefficient of variation (CV <0.3) — steady green
   Rice-Wheat rotation: high CV (>0.5) — alternating green/bare

4. **SAR Structure Features**: What does the canopy structure look like?
   Sugarcane: high VH (dense, tall canopy), high cross-ratio
   Rice: lower VH, different VH/VV dynamics

5. **Harmonic Features**: What is the periodicity of the signal?
   Sugarcane: low harmonic amplitude (quasi-constant signal)
   Rice-Wheat: high 1st/2nd harmonic (strong annual/bi-annual cycle)

All features are SCALAR (one value per plot) and do NOT depend on which
timestep is "t01" — they are invariant to temporal alignment.

Usage
-----
    from features.season_invariant_features import SeasonInvariantExtractor
    extractor = SeasonInvariantExtractor()
    feat_df = extractor.compute(df_wide, time_tags=["t00", "t01", ...])
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EPS = 1e-8


def _stack_band(df: pd.DataFrame, band: str, time_tags: List[str]) -> Optional[np.ndarray]:
    """Stack a band across time into (n_pixels, n_timesteps) array."""
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


def _interp_nans(arr: np.ndarray) -> np.ndarray:
    """Linear interpolation across NaN gaps along axis=1."""
    out = arr.copy()
    for i in range(out.shape[0]):
        ts = out[i]
        nans = np.isnan(ts)
        if not nans.any() or nans.all():
            continue
        valid_idx = np.where(~nans)[0]
        ts[nans] = np.interp(np.where(nans)[0], valid_idx, ts[valid_idx])
        out[i] = ts
    return out


# ────────────────── Duration Features ──────────────────

def compute_duration_features(arr: np.ndarray, band_prefix: str) -> Dict[str, np.ndarray]:
    """
    Compute how long the crop maintains high vegetation values.
    
    Key insight: Sugarcane stays green (NDVI > 0.5) for 6-10+ months
    continuously. No other annual crop in UP does this.
    """
    n_pix, n_t = arr.shape
    feats = {}
    
    for thresh_name, thresh_val in [("04", 0.4), ("05", 0.5), ("06", 0.6), ("07", 0.7)]:
        above = (arr > thresh_val).astype(np.int8)
        
        # Total timesteps above threshold
        feats[f"{band_prefix}_days_above_{thresh_name}"] = above.sum(axis=1).astype(np.float32)
        
        # Fraction of time above threshold
        feats[f"{band_prefix}_frac_above_{thresh_name}"] = (above.sum(axis=1) / n_t).astype(np.float32)
        
        # Longest consecutive streak above threshold (THE key sugarcane feature)
        max_streak = np.zeros(n_pix, dtype=np.float32)
        for i in range(n_pix):
            streak = 0
            best = 0
            for v in above[i]:
                if v == 1:
                    streak += 1
                    best = max(best, streak)
                else:
                    streak = 0
            max_streak[i] = best
        feats[f"{band_prefix}_max_streak_{thresh_name}"] = max_streak
    
    # Time from first to last above-0.5 reading (growing season span)
    above_05 = arr > 0.5
    first_above = np.full(n_pix, n_t, dtype=np.float32)
    last_above = np.zeros(n_pix, dtype=np.float32)
    for i in range(n_pix):
        indices = np.where(above_05[i])[0]
        if len(indices) > 0:
            first_above[i] = indices[0]
            last_above[i] = indices[-1]
    feats[f"{band_prefix}_green_span"] = (last_above - first_above).astype(np.float32)
    
    return feats


# ────────────────── Shape Features ──────────────────

def compute_shape_features(arr: np.ndarray, band_prefix: str) -> Dict[str, np.ndarray]:
    """
    Capture the SHAPE of the growth curve regardless of temporal position.
    
    Sugarcane: positively skewed (long period at high values), high kurtosis
    Rice/Wheat: more symmetric, lower kurtosis
    """
    n_pix, n_t = arr.shape
    feats = {}
    
    # Basic statistics (position-invariant)
    feats[f"{band_prefix}_mean"] = np.nanmean(arr, axis=1).astype(np.float32)
    feats[f"{band_prefix}_max"] = np.nanmax(arr, axis=1).astype(np.float32)
    feats[f"{band_prefix}_min"] = np.nanmin(arr, axis=1).astype(np.float32)
    feats[f"{band_prefix}_std"] = np.nanstd(arr, axis=1).astype(np.float32)
    feats[f"{band_prefix}_range"] = (
        np.nanmax(arr, axis=1) - np.nanmin(arr, axis=1)
    ).astype(np.float32)
    
    # Percentiles
    for p in [10, 25, 50, 75, 90]:
        feats[f"{band_prefix}_p{p}"] = np.nanpercentile(arr, p, axis=1).astype(np.float32)
    
    # IQR (inter-quartile range)
    feats[f"{band_prefix}_iqr"] = (
        np.nanpercentile(arr, 75, axis=1) - np.nanpercentile(arr, 25, axis=1)
    ).astype(np.float32)
    
    # Coefficient of variation (LOW for sugarcane — stable signal)
    mean_vals = np.nanmean(arr, axis=1)
    std_vals = np.nanstd(arr, axis=1)
    feats[f"{band_prefix}_cv"] = np.where(
        np.abs(mean_vals) > _EPS,
        std_vals / (np.abs(mean_vals) + _EPS),
        0.0
    ).astype(np.float32)
    
    # Skewness (sugarcane: negative skew — clustered at high values)
    from scipy.stats import skew, kurtosis
    feats[f"{band_prefix}_skewness"] = np.array([
        skew(arr[i, ~np.isnan(arr[i])]) if np.sum(~np.isnan(arr[i])) > 3 else 0.0
        for i in range(n_pix)
    ], dtype=np.float32)
    
    # Kurtosis (sugarcane: high kurtosis — concentrated distribution)
    feats[f"{band_prefix}_kurtosis"] = np.array([
        kurtosis(arr[i, ~np.isnan(arr[i])]) if np.sum(~np.isnan(arr[i])) > 3 else 0.0
        for i in range(n_pix)
    ], dtype=np.float32)
    
    # AUC (area under curve — total "greenness" over time)
    feats[f"{band_prefix}_auc"] = np.trapz(
        np.where(np.isnan(arr), 0.0, arr), axis=1
    ).astype(np.float32)
    
    # Amplitude = max - p10
    feats[f"{band_prefix}_amplitude"] = (
        np.nanmax(arr, axis=1) - np.nanpercentile(arr, 10, axis=1)
    ).astype(np.float32)
    
    # Peak-to-trough ratio
    p90 = np.nanpercentile(arr, 90, axis=1)
    p10 = np.nanpercentile(arr, 10, axis=1)
    feats[f"{band_prefix}_peak_trough_ratio"] = np.where(
        np.abs(p10) > _EPS, p90 / (np.abs(p10) + _EPS), 0.0
    ).astype(np.float32)
    
    return feats


# ────────────────── Temporal Dynamics Features ──────────────────

def compute_dynamics_features(arr: np.ndarray, band_prefix: str) -> Dict[str, np.ndarray]:
    """
    Capture the dynamics of change — how fast does NDVI rise/fall?
    
    Sugarcane: gradual changes (low max slope)
    Rice: rapid changes (high max slope at planting and harvest)
    """
    n_pix, n_t = arr.shape
    feats = {}
    
    # First derivative (rate of change)
    diffs = np.diff(arr, axis=1)  # (n_pix, n_t-1)
    
    feats[f"{band_prefix}_max_rise_rate"] = np.nanmax(diffs, axis=1).astype(np.float32)
    feats[f"{band_prefix}_max_fall_rate"] = np.nanmin(diffs, axis=1).astype(np.float32)
    feats[f"{band_prefix}_mean_abs_change"] = np.nanmean(np.abs(diffs), axis=1).astype(np.float32)
    
    # Smoothness = 1 / std(2nd derivative)  — higher = smoother curve
    d2 = np.diff(arr, n=2, axis=1)
    feats[f"{band_prefix}_smoothness"] = (
        1.0 / (np.nanstd(d2, axis=1) + _EPS)
    ).astype(np.float32)
    
    # Roughness = std of first derivative
    feats[f"{band_prefix}_roughness"] = np.nanstd(diffs, axis=1).astype(np.float32)
    
    # Autocorrelation at lag-1 (high for sugarcane — smooth, persistent signal)
    ac1 = np.zeros(n_pix, dtype=np.float32)
    for i in range(n_pix):
        ts = arr[i]
        valid = ~np.isnan(ts)
        if valid.sum() > 4:
            ts_clean = ts[valid]
            ts_centered = ts_clean - np.mean(ts_clean)
            var = np.var(ts_centered)
            if var > _EPS:
                ac1[i] = np.sum(ts_centered[:-1] * ts_centered[1:]) / ((len(ts_centered) - 1) * var)
    feats[f"{band_prefix}_autocorr_lag1"] = ac1
    
    # Number of peaks (sugarcane: 1 broad peak; rice-wheat: 2 sharp peaks)
    n_peaks = np.zeros(n_pix, dtype=np.float32)
    for i in range(n_pix):
        ts = arr[i]
        valid = ~np.isnan(ts)
        if valid.sum() > 4:
            ts_clean = ts[valid]
            mean_val = np.mean(ts_clean)
            above = ts_clean > mean_val
            # Count transitions from below-mean to above-mean
            transitions = np.sum(np.diff(above.astype(int)) == 1)
            n_peaks[i] = transitions
    feats[f"{band_prefix}_n_peaks"] = n_peaks
    
    return feats


# ────────────────── Harmonic Features ──────────────────

def compute_harmonic_features(arr: np.ndarray, band_prefix: str) -> Dict[str, np.ndarray]:
    """
    Fourier decomposition captures periodicity.
    
    Sugarcane: LOW harmonic amplitude (quasi-constant signal)
    Rice-Wheat rotation: HIGH 1st harmonic (annual cycle) + HIGH 2nd harmonic (bi-annual)
    """
    n_pix, n_t = arr.shape
    feats = {}
    
    # Replace NaN with per-row mean for FFT
    row_means = np.nanmean(arr, axis=1, keepdims=True)
    arr_filled = np.where(np.isnan(arr), row_means, arr)
    
    # Remove mean (DC component)
    arr_centered = arr_filled - np.nanmean(arr_filled, axis=1, keepdims=True)
    
    # FFT
    fft_vals = np.fft.rfft(arr_centered, axis=1)
    amplitudes = np.abs(fft_vals) / max(n_t, 1)
    
    # 1st harmonic (fundamental frequency = annual cycle)
    if amplitudes.shape[1] > 1:
        feats[f"{band_prefix}_harmonic1_amp"] = amplitudes[:, 1].astype(np.float32)
        feats[f"{band_prefix}_harmonic1_phase"] = np.angle(fft_vals[:, 1]).astype(np.float32)
    
    # 2nd harmonic (semi-annual cycle)
    if amplitudes.shape[1] > 2:
        feats[f"{band_prefix}_harmonic2_amp"] = amplitudes[:, 2].astype(np.float32)
        feats[f"{band_prefix}_harmonic2_phase"] = np.angle(fft_vals[:, 2]).astype(np.float32)
    
    # 3rd harmonic
    if amplitudes.shape[1] > 3:
        feats[f"{band_prefix}_harmonic3_amp"] = amplitudes[:, 3].astype(np.float32)
    
    # Ratio of 1st harmonic to total energy (low = no strong periodicity = sugarcane)
    total_energy = np.sum(amplitudes[:, 1:] ** 2, axis=1) + _EPS
    if amplitudes.shape[1] > 1:
        feats[f"{band_prefix}_harmonic1_dominance"] = (
            amplitudes[:, 1] ** 2 / total_energy
        ).astype(np.float32)
    
    # Spectral entropy (high = flat spectrum = sugarcane; low = peaked = seasonal crop)
    amp_norm = amplitudes[:, 1:] ** 2
    amp_sum = amp_norm.sum(axis=1, keepdims=True) + _EPS
    amp_prob = amp_norm / amp_sum
    spectral_entropy = -np.sum(
        amp_prob * np.log(amp_prob + _EPS), axis=1
    )
    feats[f"{band_prefix}_spectral_entropy"] = spectral_entropy.astype(np.float32)
    
    return feats


# ────────────────── SAR Structure Features ──────────────────

def compute_sar_features(df: pd.DataFrame, time_tags: List[str]) -> Dict[str, np.ndarray]:
    """
    SAR features that capture canopy structure.
    These are cloud-immune and critical for monsoon-season discrimination.
    """
    feats = {}
    
    vv_arr = _stack_band(df, "VV", time_tags)
    vh_arr = _stack_band(df, "VH", time_tags)
    
    if vv_arr is None or vh_arr is None:
        return feats
    
    vv_arr = _interp_nans(vv_arr)
    vh_arr = _interp_nans(vh_arr)
    
    # Cross-polarization ratio (VH/VV) — captures volume scattering
    cr = vh_arr / (vv_arr + _EPS)
    feats["sar_cr_mean"] = np.nanmean(cr, axis=1).astype(np.float32)
    feats["sar_cr_std"] = np.nanstd(cr, axis=1).astype(np.float32)
    feats["sar_cr_max"] = np.nanmax(cr, axis=1).astype(np.float32)
    
    # Radar Vegetation Index: 4*VH/(VV+VH)
    rvi = 4.0 * vh_arr / (vv_arr + vh_arr + _EPS)
    feats["sar_rvi_mean"] = np.nanmean(rvi, axis=1).astype(np.float32)
    feats["sar_rvi_std"] = np.nanstd(rvi, axis=1).astype(np.float32)
    feats["sar_rvi_max"] = np.nanmax(rvi, axis=1).astype(np.float32)
    feats["sar_rvi_p75"] = np.nanpercentile(rvi, 75, axis=1).astype(np.float32)
    
    # VH stats (sugarcane has consistently high VH due to tall dense canopy)
    feats["sar_vh_mean"] = np.nanmean(vh_arr, axis=1).astype(np.float32)
    feats["sar_vh_std"] = np.nanstd(vh_arr, axis=1).astype(np.float32)
    feats["sar_vh_cv"] = np.where(
        np.abs(np.nanmean(vh_arr, axis=1)) > _EPS,
        np.nanstd(vh_arr, axis=1) / (np.abs(np.nanmean(vh_arr, axis=1)) + _EPS),
        0.0
    ).astype(np.float32)
    
    # VV stats
    feats["sar_vv_mean"] = np.nanmean(vv_arr, axis=1).astype(np.float32)
    feats["sar_vv_std"] = np.nanstd(vv_arr, axis=1).astype(np.float32)
    
    # Duration of high RVI (> 0.7) — sugarcane has long periods
    rvi_above_07 = (rvi > 0.7).sum(axis=1).astype(np.float32)
    feats["sar_rvi_days_above_07"] = rvi_above_07
    
    # SAR temporal autocorrelation (high = stable canopy = sugarcane)
    n_pix = vh_arr.shape[0]
    sar_ac1 = np.zeros(n_pix, dtype=np.float32)
    for i in range(n_pix):
        ts = vh_arr[i]
        valid = ~np.isnan(ts)
        if valid.sum() > 4:
            ts_c = ts[valid] - np.mean(ts[valid])
            var = np.var(ts_c)
            if var > _EPS:
                sar_ac1[i] = np.sum(ts_c[:-1] * ts_c[1:]) / ((len(ts_c) - 1) * var)
    feats["sar_vh_autocorr_lag1"] = sar_ac1
    
    return feats


# ────────────────── Cross-Sensor Features ──────────────────

def compute_cross_sensor_features(
    df: pd.DataFrame, time_tags: List[str]
) -> Dict[str, np.ndarray]:
    """
    Features combining optical and SAR signals.
    The relationship between NDVI and VH changes differently for different crops.
    """
    feats = {}
    
    ndvi_arr = _stack_band(df, "NDVI", time_tags)
    vh_arr = _stack_band(df, "VH", time_tags)
    vv_arr = _stack_band(df, "VV", time_tags)
    
    if ndvi_arr is None:
        return feats
    
    ndvi_arr = _interp_nans(ndvi_arr)
    
    if vh_arr is not None:
        vh_arr = _interp_nans(vh_arr)
        # NDVI-VH correlation (crop-specific relationship)
        n_pix = ndvi_arr.shape[0]
        corr_ndvi_vh = np.zeros(n_pix, dtype=np.float32)
        for i in range(n_pix):
            v1 = ndvi_arr[i]; v2 = vh_arr[i]
            valid = ~np.isnan(v1) & ~np.isnan(v2)
            if valid.sum() > 4:
                if np.std(v1[valid]) > _EPS and np.std(v2[valid]) > _EPS:
                    corr_ndvi_vh[i] = np.corrcoef(v1[valid], v2[valid])[0, 1]
        feats["cross_ndvi_vh_corr"] = corr_ndvi_vh
        
        # VH/NDVI ratio statistics
        ratio = vh_arr / (ndvi_arr + _EPS)
        feats["cross_vh_ndvi_ratio_mean"] = np.nanmean(ratio, axis=1).astype(np.float32)
        feats["cross_vh_ndvi_ratio_std"] = np.nanstd(ratio, axis=1).astype(np.float32)
    
    if vv_arr is not None:
        vv_arr = _interp_nans(vv_arr)
        # NDVI-VV correlation
        n_pix = ndvi_arr.shape[0]
        corr_ndvi_vv = np.zeros(n_pix, dtype=np.float32)
        for i in range(n_pix):
            v1 = ndvi_arr[i]; v2 = vv_arr[i]
            valid = ~np.isnan(v1) & ~np.isnan(v2)
            if valid.sum() > 4:
                if np.std(v1[valid]) > _EPS and np.std(v2[valid]) > _EPS:
                    corr_ndvi_vv[i] = np.corrcoef(v1[valid], v2[valid])[0, 1]
        feats["cross_ndvi_vv_corr"] = corr_ndvi_vv
    
    return feats


# ────────────────── Main Extractor Class ──────────────────

class SeasonInvariantExtractor:
    """
    Extract season-invariant features from a wide-format satellite time-series DataFrame.
    
    These features characterize the CROP IDENTITY (what it is) rather than
    TEMPORAL POSITION (when it was observed), making the model work in any season.
    """
    
    # Optical vegetation indices to extract duration/shape/dynamics/harmonic features for
    VI_BANDS = ["NDVI", "EVI", "NDRE", "NDMI", "LSWI", "GNDVI", "SAVI"]
    
    # Raw S2 bands to extract shape features for
    RAW_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    
    def __init__(self):
        pass
    
    def compute(
        self,
        df: pd.DataFrame,
        time_tags: List[str],
    ) -> pd.DataFrame:
        """
        Compute all season-invariant features.
        
        Parameters
        ----------
        df        : wide-format DataFrame with columns like NDVI_t00, VH_t01, etc.
        time_tags : list of time tags (e.g. ["t00", "t01", ..., "t27"])
        
        Returns
        -------
        DataFrame with ~200-300 scalar features per plot (no timestep-indexed columns)
        """
        all_features: Dict[str, np.ndarray] = {}
        
        # ── Vegetation Index features ──
        for vi in self.VI_BANDS:
            arr = _stack_band(df, vi, time_tags)
            if arr is None:
                continue
            arr = _interp_nans(arr)
            
            # Duration features (THE key sugarcane discriminators)
            all_features.update(compute_duration_features(arr, f"si_{vi}"))
            
            # Shape features
            all_features.update(compute_shape_features(arr, f"si_{vi}"))
            
            # Temporal dynamics
            all_features.update(compute_dynamics_features(arr, f"si_{vi}"))
            
            # Harmonic/frequency features
            all_features.update(compute_harmonic_features(arr, f"si_{vi}"))
        
        # ── Raw band shape features (lighter — just stats, no duration) ──
        for band in self.RAW_BANDS:
            arr = _stack_band(df, band, time_tags)
            if arr is None:
                continue
            arr = _interp_nans(arr)
            all_features.update(compute_shape_features(arr, f"si_{band}"))
        
        # ── SAR structure features ──
        all_features.update(compute_sar_features(df, time_tags))
        
        # ── Cross-sensor features ──
        all_features.update(compute_cross_sensor_features(df, time_tags))
        
        # ── Plot area (legitimate agronomic feature) ──
        if "area_ha" in df.columns:
            all_features["area_ha"] = df["area_ha"].values.astype(np.float32)
        
        result = pd.DataFrame(all_features, index=df.index)
        
        # Replace inf/nan
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        n_features = result.shape[1]
        logger.info(
            f"Season-invariant features: {result.shape[0]} samples × "
            f"{n_features} features ({len(self.VI_BANDS)} VIs + "
            f"{len(self.RAW_BANDS)} bands + SAR + cross-sensor)"
        )
        return result
