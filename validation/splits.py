"""
validation/splits.py
====================
Honest cross-validation splitters for the sugarcane pipeline.

All splitters operate at the **polygon** level, never at the pixel level.
This eliminates within-plot autocorrelation leakage that random pixel CV
would introduce.

Splitters
---------
- ``polygon_group_kfold``     — StratifiedGroupKFold with plot_id as group.
                                Primary CV strategy for headline metrics.
- ``buffered_holdout_indices`` — drop test polygons within `buffer_m` of any
                                  train polygon's centroid.  Robustness check.
- ``spatial_block_kfold``     — bin polygons by lat/lon block (default 0.05°)
                                  and assign whole blocks to folds.  Useful
                                  when positives span multiple districts.
"""
from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError as exc:
    raise ImportError("scikit-learn >= 1.0 required") from exc


# ────────────────── PolygonGroupKFold (primary) ──────────────────

def polygon_group_kfold(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield (train_idx, val_idx) pairs from StratifiedGroupKFold.

    Each fold's val set contains *whole polygons* (no plot_id appears in
    both train and val of the same fold).

    Parameters
    ----------
    y      : (n,) int — class labels
    groups : (n,) str/int — plot_id per row (groups)
    n_splits : number of folds
    seed   : random seed
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in cv.split(np.zeros(len(y)), y, groups):
        yield np.asarray(tr), np.asarray(va)


# ────────────────── Buffered hold-out ──────────────────

def buffered_holdout_indices(
    centroids: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    buffer_m: float = 500.0,
    utm_epsg: int = 32644,
) -> np.ndarray:
    """
    Drop test polygons whose centroid is within `buffer_m` of any train centroid.

    Parameters
    ----------
    centroids : DataFrame with columns 'lon' (deg) and 'lat' (deg), one row per
                polygon, indexed by integer position matching X.
    train_idx, val_idx : output of ``polygon_group_kfold``
    buffer_m : exclusion radius in metres (default 500)
    utm_epsg : UTM zone for Euclidean distance (32644 = UTM 44N covers UP)

    Returns
    -------
    Filtered val_idx (subset).
    """
    from pyproj import Transformer

    if len(val_idx) == 0:
        return val_idx

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)

    def _project(idx):
        sub = centroids.iloc[idx]
        x, y = transformer.transform(sub["lon"].values, sub["lat"].values)
        return np.column_stack([x, y])

    train_xy = _project(train_idx)
    val_xy = _project(val_idx)

    keep = []
    for i, (vx, vy) in enumerate(val_xy):
        d = np.hypot(train_xy[:, 0] - vx, train_xy[:, 1] - vy)
        if d.min() >= buffer_m:
            keep.append(i)
    return val_idx[np.asarray(keep, dtype=int)] if keep else np.asarray([], dtype=int)


# ────────────────── Spatial block KFold ──────────────────

def spatial_block_kfold(
    centroids: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
    block_deg: float = 0.05,
    seed: int = 42,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    KFold where each fold consists of whole spatial blocks.

    Polygons are binned by floor(lat/block_deg), floor(lon/block_deg);
    each block becomes a group, then StratifiedGroupKFold is applied.

    Parameters
    ----------
    centroids : DataFrame with 'lon' and 'lat' columns
    y         : labels
    n_splits  : folds (clipped if there are fewer blocks than splits)
    block_deg : block size in degrees (0.05° ≈ 5.5 km at UP latitude)
    """
    lat_arr = pd.to_numeric(centroids["lat"], errors="coerce").values.astype(float)
    lon_arr = pd.to_numeric(centroids["lon"], errors="coerce").values.astype(float)
    lat_bin = np.floor(lat_arr / block_deg).astype(int)
    lon_bin = np.floor(lon_arr / block_deg).astype(int)
    block_id = np.array([f"{a}_{b}" for a, b in zip(lat_bin, lon_bin)])

    n_unique = len(set(block_id))
    n_splits = min(n_splits, max(2, n_unique // 2))  # need at least 2 blocks per fold

    yield from polygon_group_kfold(y, block_id, n_splits=n_splits, seed=seed)
