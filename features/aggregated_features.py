"""
aggregated_features.py
======================
Aggregates a 3D pixel timeseries tensor (N, T, F) into a 2D feature matrix (N, F_agg)
for use by Tree-based models (Random Forest, XGBoost).

For each feature in F (e.g. NDVI, VV, VH, etc.), it computes:
  - Temporal Mean
  - Temporal Std
  - Temporal Max
  - Temporal Min

Resulting in F * 4 aggregated features per pixel.
"""

import numpy as np

class TensorAggregator:
    def __init__(self, feature_names=None):
        self.feature_names = feature_names

    def aggregate(self, arr_3d: np.ndarray) -> np.ndarray:
        """
        Aggregate (N, T, F) array to (N, F * 4).
        
        Args:
            arr_3d: np.ndarray of shape (N, T, F)
            
        Returns:
            np.ndarray of shape (N, F * 4) containing Mean, Std, Max, Min
            for each feature across the time axis (T).
        """
        if arr_3d.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape {arr_3d.shape}")
            
        # Ignore NaNs during aggregation if any exist
        with np.errstate(invalid='ignore'):
            feat_mean = np.nanmean(arr_3d, axis=1)
            feat_std = np.nanstd(arr_3d, axis=1)
            feat_max = np.nanmax(arr_3d, axis=1)
            feat_min = np.nanmin(arr_3d, axis=1)
            
        # Concatenate along feature dimension
        arr_2d = np.concatenate([feat_mean, feat_std, feat_max, feat_min], axis=1)
        
        # Replace remaining NaNs (e.g. if an entire timeseries was NaN) with 0
        arr_2d = np.nan_to_num(arr_2d, nan=0.0)
        
        return arr_2d

    def get_feature_names(self):
        """Returns the generated column names for the 2D matrix."""
        if not self.feature_names:
            return None
        
        agg_names = []
        for stat in ["mean", "std", "max", "min"]:
            for f in self.feature_names:
                agg_names.append(f"{f}_{stat}")
        return agg_names
