"""
Unit tests for spectral_indices.

The previous test_ndre passed for the wrong reason — it called
``ndre(re1, red)`` where the function signature is ``ndre(nir2, re1)``.
Because (a-b)/(a+b) is symmetric in magnitude, the numerical result was
correct but the test did NOT verify the documented contract.

These tests verify the **signatures** documented in the module docstrings.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.spectral_indices import (
    ndvi, evi, ndre, gndvi, ndmi, lswi, savi, msavi, nbr,
    rvi_sar, rfdi, cross_ratio,
    cire, ireci, ccci, grvi, psri, nrpb,
)


# ────────────────── Existing indices ──────────────────

def test_ndvi():
    nir = np.array([0.8, 0.4])
    red = np.array([0.2, 0.4])
    np.testing.assert_almost_equal(ndvi(nir, red),
                                   np.array([0.6, 0.0]), decimal=5)


def test_ndre_signature():
    """NDRE = (B8A - B5) / (B8A + B5).  ndre(nir2=B8A, re1=B5)."""
    nir2 = np.array([0.85, 0.55, 0.30])  # B8A — narrow NIR
    re1  = np.array([0.30, 0.55, 0.45])  # B5  — red edge 1
    expected = (nir2 - re1) / (nir2 + re1 + 1e-8)
    np.testing.assert_almost_equal(ndre(nir2, re1), expected, decimal=5)


def test_gndvi():
    nir = np.array([0.9, 0.5])
    green = np.array([0.1, 0.5])
    np.testing.assert_almost_equal(gndvi(nir, green),
                                   np.array([0.8, 0.0]), decimal=5)


def test_ndmi():
    """NDMI = (B8A - B11) / (B8A + B11)."""
    nir2 = np.array([0.7, 0.3])
    swir1 = np.array([0.2, 0.4])
    expected = (nir2 - swir1) / (nir2 + swir1 + 1e-8)
    np.testing.assert_almost_equal(ndmi(nir2, swir1), expected, decimal=5)


def test_lswi_savi_msavi_nbr():
    nir = np.array([0.6])
    red = np.array([0.1])
    swir1 = np.array([0.2])
    swir2 = np.array([0.15])
    np.testing.assert_almost_equal(
        lswi(nir, swir1),
        (nir - swir1) / (nir + swir1 + 1e-8),
        decimal=5,
    )
    np.testing.assert_almost_equal(
        savi(nir, red),
        (nir - red) / (nir + red + 0.5 + 1e-8) * 1.5,
        decimal=5,
    )
    expected_msavi = (2 * nir + 1 - np.sqrt(np.maximum(0, (2 * nir + 1) ** 2 - 8 * (nir - red)))) / 2
    np.testing.assert_almost_equal(msavi(nir, red), expected_msavi, decimal=5)
    np.testing.assert_almost_equal(
        nbr(nir, swir2),
        (nir - swir2) / (nir + swir2 + 1e-8),
        decimal=5,
    )


# ────────────────── New indices (CIre, IRECI, CCCI, GRVI, PSRI, NRPB) ──────────

def test_cire():
    """CIre = B7 / B5 - 1."""
    b7 = np.array([0.6, 0.3])
    b5 = np.array([0.2, 0.3])
    expected = b7 / (b5 + 1e-10) - 1
    np.testing.assert_almost_equal(cire(b7, b5), expected, decimal=5)


def test_ireci():
    """IRECI = (B7 - B4) / (B5 / (B6 + 1e-10) + 1e-10)."""
    b7 = np.array([0.6])
    b4 = np.array([0.1])
    b5 = np.array([0.3])
    b6 = np.array([0.5])
    expected = (b7 - b4) / (b5 / (b6 + 1e-10) + 1e-10)
    np.testing.assert_almost_equal(ireci(b7, b6, b5, b4), expected, decimal=5)


def test_ccci():
    """CCCI = NDRE / NDVI."""
    ndre_arr = np.array([0.4, 0.2])
    ndvi_arr = np.array([0.8, 0.4])
    expected = ndre_arr / (ndvi_arr + 1e-10)
    np.testing.assert_almost_equal(ccci(ndre_arr, ndvi_arr), expected, decimal=5)


def test_grvi():
    """GRVI = (B3 - B4) / (B3 + B4)."""
    b3 = np.array([0.3])
    b4 = np.array([0.1])
    expected = (b3 - b4) / (b3 + b4 + 1e-10)
    np.testing.assert_almost_equal(grvi(b3, b4), expected, decimal=5)


def test_psri():
    """PSRI = (B4 - B2) / B7."""
    b4 = np.array([0.3])
    b2 = np.array([0.1])
    b7 = np.array([0.5])
    expected = (b4 - b2) / (b7 + 1e-10)
    np.testing.assert_almost_equal(psri(b4, b2, b7), expected, decimal=5)


def test_nrpb_linear_scale():
    """NRPB = (VH - VV) / (VH + VV) on linear scale."""
    vh = np.array([0.05, 0.1])
    vv = np.array([0.10, 0.2])
    expected = (vh - vv) / (vh + vv + 1e-10)
    np.testing.assert_almost_equal(nrpb(vh, vv), expected, decimal=5)


# ────────────────── SAR conversions ──────────────────

def test_rvi_sar_linear():
    """RVI = 4*VH / (VV + VH) on linear scale."""
    vv_lin = np.array([0.1, 0.2])
    vh_lin = np.array([0.05, 0.1])
    expected = 4 * vh_lin / (vv_lin + vh_lin + 1e-8)
    np.testing.assert_almost_equal(rvi_sar(vv_lin, vh_lin), expected, decimal=5)


def test_rvi_sar_db_autoconvert():
    """rvi_sar should detect dB inputs (>75% negative values) and convert."""
    vv_db = np.array([-10.0] * 4)
    vh_db = np.array([-13.0103] * 4)  # ≈ 0.05 linear
    result = rvi_sar(vv_db, vh_db)
    # vv_lin=0.1, vh_lin≈0.05 → 4*0.05/(0.1+0.05) ≈ 1.3333
    np.testing.assert_almost_equal(result, np.array([1.3333333] * 4), decimal=4)


def test_rfdi_and_cr():
    """RFDI = (VV - VH) / (VV + VH).  CR = VH / VV."""
    vv = np.array([0.10])
    vh = np.array([0.05])
    np.testing.assert_almost_equal(rfdi(vv, vh),
                                   (vv - vh) / (vv + vh + 1e-8),
                                   decimal=5)
    np.testing.assert_almost_equal(cross_ratio(vv, vh),
                                   vh / (vv + 1e-8),
                                   decimal=5)


def test_evi():
    nir = np.array([0.7])
    red = np.array([0.1])
    blue = np.array([0.05])
    expected = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + 1e-8)
    np.testing.assert_almost_equal(evi(nir, red, blue), expected, decimal=5)
