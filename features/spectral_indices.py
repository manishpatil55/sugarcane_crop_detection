"""
spectral_indices.py
===================
Compute spectral vegetation and water indices from Sentinel-2 and Sentinel-1
band values stored in a pandas DataFrame (wide format).

Supported indices
-----------------
  Sentinel-2 derived:
    NDVI  — Normalised Difference Vegetation Index
    EVI   — Enhanced Vegetation Index
    NDWI  — Normalised Difference Water Index (Gao)
    LSWI  — Land Surface Water Index
    SAVI  — Soil-Adjusted Vegetation Index
    MSAVI — Modified SAVI
    NBR   — Normalised Burn Ratio
    NDRE  — Normalised Difference Red Edge  (B8A − B5) / (B8A + B5)
    GNDVI — Green Normalised Difference Vegetation Index
    NDMI  — Normalised Difference Moisture Index  (B8A − B11) / (B8A + B11)

  Sentinel-1 derived:
    RVI   — Radar Vegetation Index
    RFDI  — Radar Forest Degradation Index
    CR    — Cross-Ratio (VH/VV)

Usage
-----
    from features.spectral_indices import SpectralIndexCalculator

    calc = SpectralIndexCalculator()
    df_with_indices = calc.compute_all(df, time_tags=["2022_01", "2022_02", ...])
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Small epsilon to avoid division by zero
_EPS = 1e-8


# ---------------------------------------------------------------------------
# Pure-numpy index functions (operate on arrays)
# ---------------------------------------------------------------------------

def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - RED) / (NIR + RED)"""
    return (nir - red) / (nir + red + _EPS)


def evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """EVI = 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)"""
    return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + _EPS)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDWI (McFeeters) = (GREEN - NIR) / (GREEN + NIR)"""
    return (green - nir) / (green + nir + _EPS)


def lswi(nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """LSWI = (NIR - SWIR1) / (NIR + SWIR1)"""
    return (nir - swir1) / (nir + swir1 + _EPS)


def savi(nir: np.ndarray, red: np.ndarray, L: float = 0.5) -> np.ndarray:
    """SAVI = (NIR - RED) / (NIR + RED + L) * (1 + L)"""
    return (nir - red) / (nir + red + L + _EPS) * (1 + L)


def msavi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """MSAVI = (2*NIR + 1 - sqrt((2*NIR+1)^2 - 8*(NIR-RED))) / 2"""
    inner = np.maximum(0, (2 * nir + 1) ** 2 - 8 * (nir - red))
    return (2 * nir + 1 - np.sqrt(inner)) / 2


def nbr(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """NBR = (NIR - SWIR2) / (NIR + SWIR2)"""
    return (nir - swir2) / (nir + swir2 + _EPS)


def ndre(nir2: np.ndarray, re1: np.ndarray) -> np.ndarray:
    """NDRE = (NIR2 - RE1) / (NIR2 + RE1)  — Normalised Difference Red Edge.

    Uses B8A (Narrow NIR, 865nm) and B5 (Red Edge 1, 705nm).
    Resistant to saturation at high LAI (>4), critical for sugarcane grand growth.
    """
    return (nir2 - re1) / (nir2 + re1 + _EPS)


def ndmi(nir2: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """NDMI = (NIR2 - SWIR1) / (NIR2 + SWIR1)  — Normalised Difference Moisture Index.

    Uses B8A (Narrow NIR, 865nm) and B11 (SWIR1, 1610nm).
    Tracks canopy moisture content; critical for irrigated sugarcane in UP.
    """
    return (nir2 - swir1) / (nir2 + swir1 + _EPS)


def gndvi(nir: np.ndarray, green: np.ndarray) -> np.ndarray:
    """GNDVI = (NIR - GREEN) / (NIR + GREEN)"""
    return (nir - green) / (nir + green + _EPS)


# ── Sugarcane-specific red-edge / chlorophyll indices ─────────────────────

def cire(b7: np.ndarray, b5: np.ndarray) -> np.ndarray:
    """Chlorophyll Red-Edge index. CIre = B7 / B5 - 1.

    Sugarcane's grand-growth phase produces very high B7 / B5 ratios
    (canopy chlorophyll concentration) — much higher than rice or maize.
    """
    return b7 / (b5 + 1e-10) - 1.0


def ireci(b7: np.ndarray, b6: np.ndarray, b5: np.ndarray, b4: np.ndarray) -> np.ndarray:
    """Inverted Red-Edge Chlorophyll Index.

    IRECI = (B7 - B4) / (B5 / (B6 + 1e-10) + 1e-10)

    More sensitive than NDRE for high-LAI crops; widely used in operational
    sugarcane monitoring.
    """
    return (b7 - b4) / (b5 / (b6 + 1e-10) + 1e-10)


def ccci(ndre_arr: np.ndarray, ndvi_arr: np.ndarray) -> np.ndarray:
    """Canopy Chlorophyll Content Index.

    CCCI = NDRE / NDVI

    Decouples chlorophyll concentration from canopy density — high CCCI
    is a hallmark of sugarcane during August–September.
    """
    return ndre_arr / (ndvi_arr + 1e-10)


def grvi(b3: np.ndarray, b4: np.ndarray) -> np.ndarray:
    """Green-Red Vegetation Index. GRVI = (B3 - B4) / (B3 + B4)."""
    return (b3 - b4) / (b3 + b4 + 1e-10)


def psri(b4: np.ndarray, b2: np.ndarray, b7: np.ndarray) -> np.ndarray:
    """Plant Senescence Reflectance Index. PSRI = (B4 - B2) / B7.

    Spikes during sugarcane harvest (Dec–Feb) when the crop senesces;
    very useful for detecting the harvest signal.
    """
    return (b4 - b2) / (b7 + 1e-10)


def nrpb(vh: np.ndarray, vv: np.ndarray) -> np.ndarray:
    """Normalised Ratio Polarisation Bands (linear scale).

    NRPB = (VH - VV) / (VH + VV)

    Inputs **must be in linear power scale**, not dB. Use
    ``_ensure_linear`` upstream if not sure.
    """
    vv_lin, vh_lin = _ensure_linear(vv, vh)
    return (vh_lin - vv_lin) / (vh_lin + vv_lin + 1e-10)


def _ensure_linear(vv: np.ndarray, vh: np.ndarray) -> tuple:
    """Convert SAR values from dB to linear power scale if needed.
    Uses a robust check: if >75% of non-NaN values are negative, assume dB."""
    vv_valid = vv[~np.isnan(vv)]
    if len(vv_valid) > 0 and (vv_valid < 0).mean() > 0.75:
        vv = np.power(10, vv / 10.0)
        vh = np.power(10, vh / 10.0)
    return vv, vh


def rvi_sar(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    """Radar Vegetation Index (SAR). RVI = 4*VH / (VV + VH). Input: linear power scale."""
    vv, vh = _ensure_linear(vv, vh)
    return 4 * vh / (vv + vh + _EPS)


def rfdi(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    """Radar Forest Degradation Index. RFDI = (VV - VH) / (VV + VH)"""
    vv, vh = _ensure_linear(vv, vh)
    return (vv - vh) / (vv + vh + _EPS)


def cross_ratio(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    """CR = VH / VV (linear scale)"""
    vv, vh = _ensure_linear(vv, vh)
    return vh / (vv + _EPS)


# ---------------------------------------------------------------------------
# DataFrame-level calculator
# ---------------------------------------------------------------------------

# Mapping from Sentinel-2 band name → column prefix in wide DataFrame
_S2_BAND_MAP = {
    "blue": "B2",
    "green": "B3",
    "red": "B4",
    "re1": "B5",
    "re2": "B6",
    "re3": "B7",
    "nir": "B8",
    "nir2": "B8A",
    "swir1": "B11",
    "swir2": "B12",
}

_S1_BAND_MAP = {
    "vv": "VV",
    "vh": "VH",
}


class SpectralIndexCalculator:
    """
    Compute spectral indices for each time step in a wide-format DataFrame.

    The DataFrame is expected to have columns named:
        <BAND>_<YYYY_MM>   e.g. B8_2022_01, B4_2022_01, VV_2022_01

    After calling compute_all(), new columns are added:
        NDVI_2022_01, EVI_2022_01, NDWI_2022_01, ...
    """

    # Indices to compute by default
    DEFAULT_S2_INDICES = [
        "NDVI", "EVI", "NDWI", "LSWI", "SAVI", "MSAVI", "NBR",
        "NDRE", "GNDVI", "NDMI",
        # Sugarcane-specific red-edge / chlorophyll / senescence
        "CIre", "IRECI", "CCCI", "GRVI", "PSRI",
    ]
    DEFAULT_S1_INDICES = ["RVI", "RFDI", "CR", "NRPB"]

    def __init__(
        self,
        s2_indices: Optional[List[str]] = None,
        s1_indices: Optional[List[str]] = None,
    ):
        self.s2_indices = s2_indices or self.DEFAULT_S2_INDICES
        self.s1_indices = s1_indices or self.DEFAULT_S1_INDICES

    def compute_all(
        self,
        df: pd.DataFrame,
        time_tags: Optional[List[str]] = None,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        Compute all configured indices for every time step.

        Parameters
        ----------
        df        : wide-format DataFrame with band columns
        time_tags : list of "YYYY_MM" strings; if None, auto-detected
        inplace   : if True, modify df in place; else return a copy

        Returns
        -------
        DataFrame with additional index columns
        """
        if not inplace:
            df = df.copy()

        if time_tags is None:
            time_tags = self._detect_time_tags(df)

        logger.info(f"Computing indices for {len(time_tags)} time steps...")

        # Build all new index columns into a side dict, then concat once.
        # This avoids the "DataFrame is highly fragmented" warning and is
        # ~10x faster than column-by-column df[col] = ... assignment.
        new_cols: dict = {}
        for tag in time_tags:
            self._compute_s2_indices_for_tag(df, tag, out_dict=new_cols)
            self._compute_s1_indices_for_tag(df, tag, out_dict=new_cols)
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        return df

    def _detect_time_tags(self, df: pd.DataFrame) -> List[str]:
        """Detect canonical YYYY_MM_DD time tags using the shared helper."""
        from utils import detect_time_tags as _detect
        return _detect(df)

    def _get_band(self, df: pd.DataFrame, band: str, tag: str) -> np.ndarray:
        """Safely retrieve a band array, returning NaN array if missing."""
        col = f"{band}_{tag}"
        if col in df.columns:
            return df[col].values.astype(np.float32)
        return np.full(len(df), np.nan, dtype=np.float32)

    def _compute_s2_indices_for_tag(self, df: pd.DataFrame, tag: str, out_dict: dict = None):
        """Compute all S2 indices for a single time tag."""
        nir = self._get_band(df, "B8", tag)
        red = self._get_band(df, "B4", tag)
        green = self._get_band(df, "B3", tag)
        blue = self._get_band(df, "B2", tag)
        swir1 = self._get_band(df, "B11", tag)
        swir2 = self._get_band(df, "B12", tag)
        re1 = self._get_band(df, "B5", tag)
        re2 = self._get_band(df, "B6", tag)
        re3 = self._get_band(df, "B7", tag)
        nir2 = self._get_band(df, "B8A", tag)

        ndvi_arr = ndvi(nir, red)
        ndre_arr = ndre(nir2, re1)

        index_map = {
            "NDVI":  lambda: ndvi_arr,
            "EVI":   lambda: evi(nir, red, blue),
            "NDWI":  lambda: ndwi(green, nir),
            "LSWI":  lambda: lswi(nir, swir1),
            "SAVI":  lambda: savi(nir, red),
            "MSAVI": lambda: msavi(nir, red),
            "NBR":   lambda: nbr(nir, swir2),
            "NDRE":  lambda: ndre_arr,
            "GNDVI": lambda: gndvi(nir, green),
            "NDMI":  lambda: ndmi(nir2, swir1),
            # New sugarcane-specific indices
            "CIre":  lambda: cire(re3, re1),
            "IRECI": lambda: ireci(re3, re2, re1, red),
            "CCCI":  lambda: ccci(ndre_arr, ndvi_arr),
            "GRVI":  lambda: grvi(green, red),
            "PSRI":  lambda: psri(red, blue, re3),
        }

        for idx_name in self.s2_indices:
            if idx_name in index_map:
                col = f"{idx_name}_{tag}"
                if col in df.columns or col in out_dict:
                    continue
                out_dict[col] = index_map[idx_name]()

    def _compute_s1_indices_for_tag(self, df: pd.DataFrame, tag: str, out_dict: dict = None):
        """Compute all S1 indices for a single time tag.

        Inputs VV/VH may be in dB *or* linear; ``_ensure_linear`` (used
        inside each helper) normalises them.
        """
        vv = self._get_band(df, "VV", tag)
        vh = self._get_band(df, "VH", tag)

        index_map = {
            "RVI":  lambda: rvi_sar(vv, vh),
            "RFDI": lambda: rfdi(vv, vh),
            "CR":   lambda: cross_ratio(vv, vh),
            "NRPB": lambda: nrpb(vh, vv),
        }

        for idx_name in self.s1_indices:
            if idx_name in index_map:
                col = f"{idx_name}_{tag}"
                if col in df.columns or col in out_dict:
                    continue
                out_dict[col] = index_map[idx_name]()

    # ------------------------------------------------------------------
    # Utility: compute indices on raw arrays (for GEE-independent use)
    # ------------------------------------------------------------------

    @staticmethod
    def from_arrays(
        nir: np.ndarray,
        red: np.ndarray,
        green: np.ndarray,
        blue: np.ndarray,
        swir1: np.ndarray,
        swir2: Optional[np.ndarray] = None,
        re1: Optional[np.ndarray] = None,
        nir2: Optional[np.ndarray] = None,
        vv: Optional[np.ndarray] = None,
        vh: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Compute all indices from raw band arrays.

        Returns
        -------
        dict mapping index_name → np.ndarray
        """
        result = {
            "NDVI": ndvi(nir, red),
            "EVI": evi(nir, red, blue),
            "NDWI": ndwi(green, nir),
            "LSWI": lswi(nir, swir1),
            "SAVI": savi(nir, red),
            "MSAVI": msavi(nir, red),
        }
        if swir2 is not None:
            result["NBR"] = nbr(nir, swir2)
        if nir2 is not None and re1 is not None:
            result["NDRE"] = ndre(nir2, re1)
        if green is not None and nir is not None:
            result["GNDVI"] = gndvi(nir, green)
        if nir2 is not None and swir1 is not None:
            result["NDMI"] = ndmi(nir2, swir1)
        if vv is not None and vh is not None:
            result["RVI"] = rvi_sar(vv, vh)
            result["RFDI"] = rfdi(vv, vh)
            result["CR"] = cross_ratio(vv, vh)
        return result
