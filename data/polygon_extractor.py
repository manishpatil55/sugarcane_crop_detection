"""
data/polygon_extractor.py
=========================
Per-polygon zonal-statistics extractor for Sentinel-1 + Sentinel-2.

This is the **canonical extraction module** for the rebuilt sugarcane
pipeline. Unlike ``gee_downloader.extract_pixel_timeseries_wide`` (which
samples N pixels per plot — yielding pseudo-replication), this module
produces **one row per polygon** by computing the median of all valid
pixels inside the polygon for each 15-day window.

Output schema (one row per polygon)
-----------------------------------
``plot_id, label, area_ha, source_file, anchor_date, date_start, date_end,``
``B2_<YYYY_MM_DD>, B3_<YYYY_MM_DD>, ..., B12_<YYYY_MM_DD>,``
``VV_<YYYY_MM_DD>, VH_<YYYY_MM_DD>,``
``valid_pixel_count_<YYYY_MM_DD>``

Cloud masking (per-pixel)
-------------------------
S2 valid pixel ⇔
    SCL ∈ {4, 5, 6, 7}                                  (vegetation, bare, water, unclassified)
    AND  S2_CLOUD_PROBABILITY < 30                       (per-pixel)
    AND  QA60 cloud + cirrus bits = 0
    AND  scene CLOUDY_PIXEL_PERCENTAGE < 30

VV/VH stored in **linear power** scale (NOT dB).

Window-level fallback
---------------------
If a window has zero valid optical pixels, optical bands are NaN
(filled later by 1-D temporal interpolation). SAR is independent.
If a window has < ``min_valid_pixels`` (default 5) valid pixels, that
window is also flagged NaN for that polygon.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import yaml

logger = logging.getLogger(__name__)

# Bands we extract from S2 + S1
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S1_BANDS = ["VV", "VH"]


# ────────────────── Helpers ──────────────────

def _interval_range(start_date: str, end_date: str, interval_days: int = 15
                    ) -> List[Tuple[str, str, str]]:
    """Return list of (start, end, tag='YYYY_MM_DD') for each window."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    out = []
    cur = start
    while cur < end:
        nxt = cur + timedelta(days=interval_days)
        out.append((cur.strftime("%Y-%m-%d"),
                    nxt.strftime("%Y-%m-%d"),
                    cur.strftime("%Y_%m_%d")))
        cur = nxt
    return out


def _ee_init(project_id: str):
    """Idempotent GEE init."""
    import ee
    try:
        # If already initialised, this is a no-op
        ee.Number(1).getInfo()
    except Exception:
        try:
            ee.Initialize(project=project_id)
        except Exception as exc:
            raise RuntimeError(
                f"GEE init failed: {exc}\n"
                "Run  `earthengine authenticate`  and confirm "
                f"project_id='{project_id}' in config.yaml."
            ) from exc


# ────────────────── GEE composite builders ──────────────────

def _mask_s2(image, cloud_prob_threshold: int = 30):
    """Per-pixel S2 cloud mask: SCL ∈ {4,5,6,7} & cloud_prob < threshold & QA60."""
    import ee
    qa = image.select("QA60")
    scl = image.select("SCL")
    qa_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    valid_scl = ee.List([4, 5, 6, 7])
    scl_mask = scl.remap(valid_scl, ee.List.repeat(1, valid_scl.size()), 0)
    # Cloud prob via the joined property added in _link_cloud_prob
    cloud_prob = ee.Image(image.get("cloud_probability"))
    cp_mask = cloud_prob.lt(cloud_prob_threshold)
    return image.updateMask(qa_mask.And(scl_mask).And(cp_mask)) \
                .divide(10000.0) \
                .copyProperties(image, ["system:time_start"])


def _link_cloud_prob(s2_coll, cprob_coll):
    """Join S2 SR with S2_CLOUD_PROBABILITY by system:index."""
    import ee
    cond = ee.Filter.equals(leftField="system:index", rightField="system:index")
    return ee.ImageCollection(
        ee.Join.saveFirst("cloud_probability").apply(
            primary=s2_coll, secondary=cprob_coll, condition=cond
        )
    ).map(lambda img: img.set("cloud_probability", img.get("cloud_probability")))


def _build_s2_composite(geometry, start: str, end: str, cloud_threshold: int):
    """Return median composite of S2 SR over [start,end] for the geometry."""
    import ee
    s2_sr = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
             .filterBounds(geometry).filterDate(start, end)
             .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold)))
    s2_cp = (ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
             .filterBounds(geometry).filterDate(start, end))

    joined = _link_cloud_prob(s2_sr, s2_cp)
    masked = joined.map(lambda img: _mask_s2(img, cloud_prob_threshold=30))
    return masked.select(S2_BANDS).median()


def _build_s1_composite(geometry, start: str, end: str, orbit_pass: str = "DESCENDING"):
    """Return median composite of S1 GRD over [start,end] in **linear** power."""
    import ee
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(geometry).filterDate(start, end)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .select(["VV", "VH"]))

    # Convert dB → linear power, then take median in linear domain
    def _to_linear(img):
        lin = ee.Image(10).pow(img.divide(10)).rename(["VV", "VH"])
        return lin.copyProperties(img, ["system:time_start"])

    return s1.map(_to_linear).median()


# ────────────────── Per-polygon zonal stats ──────────────────

def _zonal_stats_for_window(
    geometry,
    s2_img,
    s1_img,
    scale: int = 10,
    max_pixels: int = int(1e8),
) -> Dict[str, Any]:
    """Compute median + count for both S2 and S1 over the polygon."""
    import ee

    out: Dict[str, Any] = {}

    # ── S2: median + valid-pixel count (count of unmasked pixels in any band)
    if s2_img is not None:
        s2_dict = (s2_img.reduceRegion(
            reducer=ee.Reducer.median(),
            geometry=geometry, scale=scale, maxPixels=max_pixels,
            bestEffort=True,
        ).getInfo())
        for b in S2_BANDS:
            v = s2_dict.get(b)
            out[b] = float(v) if v is not None else float("nan")

        # Valid-pixel count using NDVI proxy band (NIR has same mask as others post-mask)
        count_dict = (s2_img.select("B8").reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=geometry, scale=scale, maxPixels=max_pixels,
            bestEffort=True,
        ).getInfo())
        out["valid_pixel_count"] = int(count_dict.get("B8") or 0)
    else:
        for b in S2_BANDS:
            out[b] = float("nan")
        out["valid_pixel_count"] = 0

    # ── S1: median in linear scale
    if s1_img is not None:
        s1_dict = (s1_img.reduceRegion(
            reducer=ee.Reducer.median(),
            geometry=geometry, scale=scale, maxPixels=max_pixels,
            bestEffort=True,
        ).getInfo())
        for b in S1_BANDS:
            v = s1_dict.get(b)
            out[b] = float(v) if v is not None else float("nan")
    else:
        for b in S1_BANDS:
            out[b] = float("nan")

    return out


# ────────────────── 1-D temporal cleaning ──────────────────

def _interpolate_and_smooth(
    df_long: pd.DataFrame,
    bands: List[str],
    n_windows: int,
    smooth_window: int = 5,
    min_valid_pixels: int = 5,
) -> pd.DataFrame:
    """
    Take a long DataFrame with one row per (plot_id, window_idx) and
    columns [plot_id, window_idx, B2..B12, VV, VH, valid_pixel_count],
    apply per-plot 1-D linear interpolation across `window_idx`, then
    Savitzky-Golay smoothing.
    """
    from scipy.signal import savgol_filter

    out_frames = []
    for plot_id, sub in df_long.groupby("plot_id"):
        sub = sub.sort_values("window_idx").reset_index(drop=True)
        # NaN out optical bands where valid_pixel_count < min_valid_pixels
        too_few = sub["valid_pixel_count"] < min_valid_pixels
        for b in S2_BANDS:
            sub.loc[too_few, b] = np.nan

        for b in bands:
            ts = sub[b].astype(np.float32).values
            nans = np.isnan(ts)
            if nans.any() and not nans.all():
                valid = np.where(~nans)[0]
                ts[nans] = np.interp(np.where(nans)[0], valid, ts[valid])
            sub[b] = ts

            # Smooth (only if at least `smooth_window` points are present)
            sw = min(smooth_window, len(ts))
            if sw % 2 == 0:
                sw -= 1
            if sw >= 3:
                try:
                    sub[b] = savgol_filter(
                        ts, window_length=sw,
                        polyorder=min(2, sw - 1),
                    )
                except Exception:
                    sub[b] = ts

        out_frames.append(sub)
    return pd.concat(out_frames, ignore_index=True)


# ────────────────── Pivot long → wide ──────────────────

def _long_to_wide(
    df_long: pd.DataFrame,
    bands: List[str],
    time_tags: List[str],
) -> pd.DataFrame:
    """Pivot long (plot_id × window) DataFrame into one row per plot."""
    rows = []
    for plot_id, sub in df_long.groupby("plot_id"):
        sub = sub.sort_values("window_idx").reset_index(drop=True)
        meta = {
            "plot_id":     plot_id,
            "group_key":   (sub["group_key"].iloc[0] if "group_key" in sub.columns
                            else sub["source_file"].iloc[0]),
            "label":       int(sub["label"].iloc[0]),
            "area_ha":     float(sub["area_ha"].iloc[0]),
            "source_file": sub["source_file"].iloc[0],
            "anchor_date": sub["anchor_date"].iloc[0],
            "date_start":  sub["date_start"].iloc[0],
            "date_end":    sub["date_end"].iloc[0],
            "state":       sub["state"].iloc[0],
        }
        for j, tag in enumerate(time_tags):
            row_j = sub[sub["window_idx"] == j]
            if len(row_j) == 0:
                for b in bands:
                    meta[f"{b}_{tag}"] = np.nan
                meta[f"valid_pixel_count_{tag}"] = 0
            else:
                r = row_j.iloc[0]
                for b in bands:
                    meta[f"{b}_{tag}"] = float(r[b]) if pd.notna(r[b]) else np.nan
                meta[f"valid_pixel_count_{tag}"] = int(r["valid_pixel_count"])
        rows.append(meta)
    return pd.DataFrame(rows)


# ────────────────── Public class ──────────────────

class PolygonExtractor:
    """
    Per-polygon zonal-statistics extractor.

    Parameters
    ----------
    config_path : path to config.yaml (reads gee.project_id, sentinel2.cloud_threshold, etc.)
    cache_dir   : optional directory to cache per-window per-plot extraction results.
                  If a window-result CSV exists, it is loaded instead of re-fetching.

    Usage
    -----
        ext = PolygonExtractor(config_path="config.yaml", cache_dir="data/cache")
        wide_df = ext.extract(gdf)        # gdf has plot_id, label, area_ha,
                                          # date_start, date_end, geometry, state
        wide_df.to_csv("data/processed/extraction_wide.csv", index=False)
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        cache_dir: Optional[str] = "data/cache",
    ):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.project_id = self.cfg["gee"]["project_id"]
        self.cloud_threshold = int(self.cfg.get("sentinel2", {}).get("cloud_threshold", 30))
        self.scale = int(self.cfg.get("sentinel2", {}).get("scale", 10))
        self.interval_days = int(self.cfg.get("compositing", {}).get("interval_days", 15))
        self.orbit_pass = self.cfg.get("sentinel1", {}).get("orbit_pass", "DESCENDING")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def extract(
        self,
        gdf: gpd.GeoDataFrame,
        min_valid_pixels: int = 5,
        chunk_size: int = 100,
    ) -> pd.DataFrame:
        """
        Batched zonal-statistic extractor.

        Algorithm
        ---------
        For each 15-day window:
            1. Build masked S2 composite + S1 composite over the union geometry.
            2. Build a FeatureCollection of all (or a chunk of) polygons.
            3. One call to ``image.reduceRegions(reducer=median+count)`` returns
                all polygon medians for the window in a single GEE round-trip.

        Result: ~28 windows * (chunks) reduceRegions calls instead of
        N_polygons * N_windows * 2 = ~47,000 calls for the previous design.

        Parameters
        ----------
        chunk_size : split the FeatureCollection into chunks of this size
                     to stay under GEE's getInfo() payload limit (~10 MB).
        """
        import ee
        _ee_init(self.project_id)

        if gdf.empty:
            raise ValueError("Empty GeoDataFrame.")
        if "date_start" not in gdf.columns or "date_end" not in gdf.columns:
            raise ValueError("gdf must have date_start, date_end columns.")

        ds = gdf["date_start"].dropna().iloc[0]
        de = gdf["date_end"].dropna().iloc[0]
        windows = _interval_range(ds, de, self.interval_days)
        time_tags = [t for _, _, t in windows]
        logger.info(
            f"Batched polygon extraction: {len(gdf)} plots \u00d7 {len(windows)} "
            f"15-day windows  (window: {ds} \u2192 {de})  chunk_size={chunk_size}"
        )

        # Pre-build EE FeatureCollections (one per chunk) so we don't reconstruct per window
        gdf = gdf.reset_index(drop=True)
        chunks = []
        for start in range(0, len(gdf), chunk_size):
            sub = gdf.iloc[start:start + chunk_size]
            features = []
            for _, r in sub.iterrows():
                try:
                    geom = ee.Geometry(r.geometry.__geo_interface__)
                    feat = ee.Feature(geom, {"plot_id": str(r["plot_id"])})
                    features.append(feat)
                except Exception as exc:
                    logger.warning(f"  [SKIP] {r['plot_id']}: bad geom \u2192 {exc}")
            if features:
                chunks.append((sub.reset_index(drop=True), ee.FeatureCollection(features)))

        # ── Run extraction: one reduceRegions per (chunk × window)
        long_rows = []
        for chunk_idx, (sub_gdf, fc) in enumerate(chunks):
            # Map plot_id \u2192 row dict so we can attach metadata later
            meta_map = {str(r["plot_id"]): r for _, r in sub_gdf.iterrows()}
            fc_geom = fc.geometry()

            for j, (w_start, w_end, tag) in enumerate(windows):
                try:
                    s2 = _build_s2_composite(fc_geom, w_start, w_end, self.cloud_threshold)
                    s1 = _build_s1_composite(fc_geom, w_start, w_end, self.orbit_pass)
                    # combined image: S2 bands + S1 bands. Counts come from B8.
                    combined = s2.addBands(s1)
                    reducer = (ee.Reducer.median()
                               .combine(ee.Reducer.count(), sharedInputs=True))
                    result_fc = combined.reduceRegions(
                        collection=fc, reducer=reducer, scale=self.scale,
                    )
                    feats = result_fc.getInfo()["features"]
                except Exception as exc:
                    logger.debug(f"[chunk{chunk_idx}/{tag}] failed: {exc}")
                    feats = []

                # Build a lookup by plot_id
                got = {f["properties"]["plot_id"]: f["properties"] for f in feats}

                for plot_id, row in meta_map.items():
                    props = got.get(plot_id, {})
                    stats = {b: float(props[b + "_median"])
                              if props.get(b + "_median") is not None else float("nan")
                              for b in S2_BANDS + S1_BANDS}
                    # valid_pixel_count via B8_count
                    stats["valid_pixel_count"] = int(props.get("B8_count") or 0)

                    stats.update({
                        "plot_id":     plot_id,
                        "group_key":   str(row.get("group_key", row.get("source_file", ""))),
                        "window_idx":  j,
                        "tag":         tag,
                        "label":       int(row["label"]),
                        "area_ha":     float(row.get("area_ha", 0.0)),
                        "source_file": str(row.get("source_file", "")),
                        "anchor_date": str(row.get("anchor_date", "")),
                        "date_start":  str(row.get("date_start", "")),
                        "date_end":    str(row.get("date_end", "")),
                        "state":       str(row.get("state", "")),
                    })
                    long_rows.append(stats)

                if (j + 1) % 4 == 0:
                    logger.info(
                        f"  chunk {chunk_idx + 1}/{len(chunks)}  "
                        f"window {j + 1}/{len(windows)} -> {tag}"
                    )

            # Cache per-chunk intermediate (optional)
            if self.cache_dir is not None:
                chunk_df = pd.DataFrame([r for r in long_rows
                                          if r["plot_id"] in meta_map])
                chunk_df.to_csv(self.cache_dir / f"chunk_{chunk_idx:03d}.csv",
                                 index=False)

        if not long_rows:
            raise RuntimeError("No polygon data extracted.")
        df_long = pd.DataFrame(long_rows)

        # Clean: NaN out optical when valid_pixel_count < min, then interp + smooth
        df_long = _interpolate_and_smooth(
            df_long,
            bands=S2_BANDS + S1_BANDS,
            n_windows=len(time_tags),
            smooth_window=5,
            min_valid_pixels=min_valid_pixels,
        )

        # Pivot to wide
        df_wide = _long_to_wide(df_long, bands=S2_BANDS + S1_BANDS, time_tags=time_tags)

        # Cloud-quality summary columns (used by api.py)
        cnt_cols = [c for c in df_wide.columns if c.startswith("valid_pixel_count_")]
        if cnt_cols:
            counts = df_wide[cnt_cols].values
            df_wide["pct_windows_with_valid_optical"] = (
                (counts >= min_valid_pixels).mean(axis=1) * 100
            )
            df_wide["n_cloudy_windows"] = (counts < min_valid_pixels).sum(axis=1)
            df_wide["n_total_windows"] = len(time_tags)

        logger.info(f"Wide table built: {df_wide.shape}")
        return df_wide
