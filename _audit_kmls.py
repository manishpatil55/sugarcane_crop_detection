"""
Comprehensive KML audit script for the sugarcane detection project.

Outputs:
  - Polygon counts per class
  - Area statistics (min/max/mean/median in hectares)
  - Geographic distribution (state, lat/lon bbox, district approximation)
  - Pixel count estimates at 10m S2 resolution
  - Geometry validity (self-intersecting, too small, outside UP bbox)
  - Negative class composition (from filenames)
  - Spatial leakage risk: nearest-neighbor distance distribution
"""
from __future__ import annotations

import json
import math
import re
import sys
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union, transform as shapely_transform
from shapely.validation import make_valid, explain_validity
from pyproj import Transformer


# ────────────────────────────── Helpers ──────────────────────────────

def parse_kml_polygons(path: Path):
    """Return list of shapely Polygon objects (lon/lat) from a KML."""
    polygons = []
    try:
        tree = ET.parse(path)
    except Exception:
        return polygons
    root = tree.getroot()

    def strip_ns(tag):
        return tag.split("}")[-1] if "}" in tag else tag

    def find_all(el, tag):
        return [e for e in el.iter() if strip_ns(e.tag) == tag]

    for placemark in find_all(root, "Placemark"):
        for poly_el in find_all(placemark, "Polygon"):
            outer = find_all(poly_el, "outerBoundaryIs")
            if not outer:
                continue
            coords_el = find_all(outer[0], "coordinates")
            if not coords_el or coords_el[0].text is None:
                continue
            coords_text = coords_el[0].text.strip()
            coords = []
            for pt in coords_text.split():
                parts = pt.split(",")
                if len(parts) >= 2:
                    try:
                        lon = float(parts[0])
                        lat = float(parts[1])
                        coords.append((lon, lat))
                    except ValueError:
                        pass
            if len(coords) >= 4:
                try:
                    polygons.append(Polygon(coords))
                except Exception:
                    pass
    return polygons


def area_hectares(geom, transformer):
    try:
        proj = shapely_transform(transformer.transform, geom)
        return proj.area / 10_000.0  # m² → ha
    except Exception:
        return float("nan")


# Approximate UP district bounding boxes for major sugarcane districts.
# Source: rough public-domain bbox estimates (lat_min, lat_max, lon_min, lon_max).
UP_DISTRICTS = [
    ("Saharanpur",       29.50, 30.20, 77.30, 78.10),
    ("Muzaffarnagar",    29.20, 29.80, 77.40, 78.10),
    ("Shamli",           29.30, 29.65, 77.10, 77.55),
    ("Meerut",           28.80, 29.30, 77.40, 78.10),
    ("Bijnor",           29.10, 29.80, 78.00, 78.80),
    ("Moradabad",        28.50, 29.20, 78.40, 79.00),
    ("Rampur",           28.60, 29.20, 78.80, 79.40),
    ("Bareilly",         28.10, 28.85, 78.80, 79.80),
    ("Pilibhit",         28.30, 28.85, 79.50, 80.20),
    ("Lakhimpur Kheri",  27.70, 28.70, 80.10, 81.30),
    ("Sitapur",          27.20, 27.85, 80.40, 81.30),
    ("Hardoi",           26.70, 27.60, 79.70, 80.80),
    ("Shahjahanpur",     27.65, 28.15, 79.40, 80.20),
    ("Ghaziabad",        28.55, 28.85, 77.30, 77.80),
    ("Bulandshahr",      28.10, 28.70, 77.55, 78.30),
    ("Aligarh",          27.70, 28.30, 77.60, 78.40),
    ("Mathura",          27.30, 27.80, 77.50, 78.10),
    ("Agra",             26.85, 27.50, 77.50, 78.40),
    ("Etah",             27.30, 28.00, 78.30, 79.00),
    ("Mainpuri",         27.00, 27.50, 78.70, 79.30),
    ("Farrukhabad",      27.05, 27.70, 79.20, 79.90),
    ("Etawah",           26.50, 27.05, 78.70, 79.40),
    ("Kanpur Nagar",     26.20, 26.70, 80.00, 80.50),
    ("Lucknow",          26.65, 27.10, 80.70, 81.30),
    ("Barabanki",        26.70, 27.30, 81.00, 81.80),
    ("Faizabad",         26.55, 27.00, 81.65, 82.40),
    ("Gonda",            26.85, 27.50, 81.60, 82.30),
    ("Bahraich",         27.30, 28.30, 81.10, 82.00),
    ("Balrampur",        27.20, 27.85, 82.00, 82.55),
    ("Gorakhpur",        26.45, 27.20, 83.00, 83.85),
    ("Maharajganj",      26.95, 27.45, 83.00, 83.90),
    ("Kushinagar",       26.60, 27.20, 83.65, 84.15),
    ("Deoria",           26.30, 27.00, 83.55, 84.20),
    ("Basti",            26.50, 27.20, 82.40, 83.20),
    ("Sant Kabir Nagar", 26.55, 27.20, 82.90, 83.55),
    ("Siddharthnagar",   27.00, 27.65, 82.65, 83.50),
    ("Ballia",           25.70, 26.40, 83.85, 84.50),
    ("Mau",              25.85, 26.40, 83.30, 84.05),
    ("Azamgarh",         25.85, 26.55, 82.90, 83.60),
    ("Jaunpur",          25.50, 26.05, 82.40, 83.20),
    ("Varanasi",         25.20, 25.65, 82.85, 83.40),
    ("Mirzapur",         24.50, 25.40, 82.35, 83.30),
    ("Sonbhadra",        23.85, 25.05, 82.65, 83.85),
    ("Allahabad",        25.10, 25.85, 81.45, 82.40),
    ("Kausambi",         25.30, 25.80, 81.10, 81.55),
    ("Pratapgarh",       25.65, 26.20, 81.50, 82.30),
    ("Sultanpur",        26.00, 26.70, 81.80, 82.55),
    ("Ambedkar Nagar",   26.30, 26.75, 82.30, 83.05),
]


def assign_district(lat: float, lon: float):
    for name, lat_min, lat_max, lon_min, lon_max in UP_DISTRICTS:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return name
    return "OTHER/OUTSIDE_UP"


def classify_negative_label(filename: str) -> str:
    """Best-effort label inference from the filename."""
    f = filename.lower()
    if "rice"   in f: return "rice"
    if "maize"  in f or "corn" in f: return "maize"
    if "wheat"  in f: return "wheat"
    if "mustard" in f: return "mustard"
    if "potato" in f: return "potato"
    if "mango"  in f: return "mango/orchard"
    if "poplar" in f: return "poplar/forestry"
    if "banana" in f: return "banana"
    if "guava"  in f: return "guava/orchard"
    if "fallow" in f or "barren" in f: return "fallow"
    if "urban"  in f or "city" in f or "built" in f: return "urban"
    if "water"  in f or "river" in f or "pond" in f or "lake" in f: return "water"
    if "sponge" in f: return "spongegourd/vegetable"
    if "untitled map" in f: return "unlabeled (Untitled map)"
    return "unlabeled (numeric)"


# ────────────────────────────── Audit ──────────────────────────────

def audit_class(class_dir: Path, label_name: str):
    files = sorted(list(class_dir.glob("*.kml")) + list(class_dir.glob("*.kmz")))
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True)  # UTM 44N covers central UP

    rows = []
    for fp in files:
        polys = parse_kml_polygons(fp)
        if not polys:
            rows.append({
                "file": fp.name, "class": label_name, "n_polygons": 0,
                "valid": False, "validity_reason": "No polygons parsed",
                "area_ha": None, "centroid_lat": None, "centroid_lon": None,
                "in_up_bbox": False, "district": None, "approx_pixels_10m": 0,
                "neg_subtype": classify_negative_label(fp.name) if label_name == "non_sugarcane" else None,
            })
            continue
        for i, poly in enumerate(polys):
            valid = poly.is_valid
            reason = explain_validity(poly) if not valid else "Valid"
            if not valid:
                fixed = make_valid(poly)
                # use fixed for area/centroid calc but flag invalid
                use_geom = fixed
            else:
                use_geom = poly
            try:
                ar = area_hectares(use_geom, transformer)
            except Exception:
                ar = float("nan")
            try:
                c = use_geom.centroid
                clat, clon = c.y, c.x
            except Exception:
                clat = clon = None
            in_up = (
                clat is not None and clon is not None
                and 23.9 <= clat <= 30.4 and 77.1 <= clon <= 84.6
            )
            district = assign_district(clat, clon) if (clat is not None and clon is not None) else None
            # 10m Sentinel-2 pixels  (approx)  =  area_ha * 100 (since 1ha = 10_000 m² = 100 px @ 100 m²)
            approx_pixels_10m = int(round(ar * 100)) if not math.isnan(ar) else 0

            rows.append({
                "file": fp.name,
                "class": label_name,
                "polygon_idx": i,
                "n_polygons": len(polys),
                "valid": valid,
                "validity_reason": reason,
                "area_ha": ar,
                "centroid_lat": clat,
                "centroid_lon": clon,
                "in_up_bbox": in_up,
                "district": district,
                "approx_pixels_10m": approx_pixels_10m,
                "neg_subtype": classify_negative_label(fp.name) if label_name == "non_sugarcane" else None,
            })
    return pd.DataFrame(rows)


def neighbour_distance_stats(df: pd.DataFrame, label: str):
    sub = df[(df["class"] == label) & df["centroid_lat"].notna()].copy()
    if len(sub) < 2:
        return None
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True)
    coords = []
    for _, r in sub.iterrows():
        x, y = transformer.transform(r["centroid_lon"], r["centroid_lat"])
        coords.append((x, y))
    coords = np.asarray(coords)
    # nearest-neighbour distance for each point
    nn = []
    for i, (x, y) in enumerate(coords):
        d = np.hypot(coords[:, 0] - x, coords[:, 1] - y)
        d[i] = np.inf
        nn.append(d.min())
    nn = np.asarray(nn)
    return {
        "count": int(len(nn)),
        "min_m": float(nn.min()),
        "median_m": float(np.median(nn)),
        "mean_m": float(nn.mean()),
        "max_m": float(nn.max()),
        "frac_within_500m": float((nn < 500).mean()),
        "frac_within_2km": float((nn < 2_000).mean()),
        "frac_within_10km": float((nn < 10_000).mean()),
    }


def main():
    root = Path("data/kml")
    pos_dir = root / "sugarcane"
    neg_dir = root / "non_sugarcane"

    print(f"\n{'═'*72}\nPROJECT KML AUDIT  —  Sugarcane Detection (UP)\n{'═'*72}")

    pos_df = audit_class(pos_dir, "sugarcane")
    neg_df = audit_class(neg_dir, "non_sugarcane")
    df_all = pd.concat([pos_df, neg_df], ignore_index=True)

    # ── Per-class summary ────────────────────────────────────────────
    print("\n── 1.  POLYGON COUNTS ──────────────────────────────────────────────")
    for c in ["sugarcane", "non_sugarcane"]:
        sub = df_all[df_all["class"] == c]
        n_files = sub["file"].nunique()
        n_polys = len(sub)
        n_invalid = (~sub["valid"]).sum()
        print(f"  {c:20s}  files={n_files:4d}  polygons={n_polys:4d}  "
              f"invalid_geom={n_invalid:3d}")

    # ── Area statistics ─────────────────────────────────────────────
    print("\n── 2.  POLYGON AREA STATISTICS (hectares) ─────────────────────────")
    print(f"  {'class':18s} {'count':>6s} {'min':>8s} {'p10':>8s} "
          f"{'median':>8s} {'mean':>8s} {'p90':>8s} {'max':>10s}")
    for c in ["sugarcane", "non_sugarcane"]:
        a = df_all[(df_all["class"] == c) & df_all["area_ha"].notna()]["area_ha"].values
        if len(a) == 0:
            continue
        print(f"  {c:18s} {len(a):6d} {a.min():8.3f} {np.percentile(a,10):8.3f} "
              f"{np.median(a):8.3f} {a.mean():8.3f} {np.percentile(a,90):8.3f} "
              f"{a.max():10.3f}")

    # ── Pixel counts at 10m ─────────────────────────────────────────
    print("\n── 3.  PIXEL COUNTS @ 10m (Sentinel-2 native) ─────────────────────")
    for c in ["sugarcane", "non_sugarcane"]:
        sub = df_all[df_all["class"] == c]["approx_pixels_10m"]
        if len(sub) == 0:
            continue
        print(f"  {c:18s}  total_pixels={sub.sum():>10,d}   "
              f"median_per_polygon={int(np.median(sub)):>5,d}   "
              f"min={sub.min():>4d}  max={sub.max():>6,d}")

    # ── Geographic distribution ─────────────────────────────────────
    print("\n── 4.  GEOGRAPHIC DISTRIBUTION ─────────────────────────────────────")
    for c in ["sugarcane", "non_sugarcane"]:
        sub = df_all[df_all["class"] == c]
        in_up = sub["in_up_bbox"].sum()
        out_up = (~sub["in_up_bbox"]).sum()
        print(f"\n  Class: {c}")
        print(f"    inside UP bbox: {in_up}  |  outside UP bbox: {out_up}")
        if in_up > 0:
            lats = sub.loc[sub["in_up_bbox"], "centroid_lat"]
            lons = sub.loc[sub["in_up_bbox"], "centroid_lon"]
            print(f"    lat range: [{lats.min():.3f}, {lats.max():.3f}]  "
                  f"span={lats.max()-lats.min():.2f}°")
            print(f"    lon range: [{lons.min():.3f}, {lons.max():.3f}]  "
                  f"span={lons.max()-lons.min():.2f}°")
        # district histogram
        d_counts = sub["district"].value_counts()
        print(f"    Top districts (top 10):")
        for d, n in d_counts.head(10).items():
            print(f"      {str(d):28s} {n:4d}")
        if len(d_counts) > 10:
            print(f"      ... and {len(d_counts)-10} more districts")

    # ── Negative class diversity ────────────────────────────────────
    print("\n── 5.  NEGATIVE CLASS COMPOSITION (inferred from filename) ────────")
    neg_subtypes = neg_df["neg_subtype"].value_counts()
    for k, v in neg_subtypes.items():
        print(f"  {k:30s} {v:5d}")

    # ── Spatial leakage risk ────────────────────────────────────────
    print("\n── 6.  SPATIAL LEAKAGE RISK — nearest-neighbour distances ─────────")
    print(f"  (Random pixel/polygon CV will leak if neighbour distance < ~500 m)")
    for c in ["sugarcane", "non_sugarcane"]:
        nn = neighbour_distance_stats(df_all, c)
        if nn is None:
            continue
        print(f"\n  Class: {c}")
        print(f"    median NN dist  : {nn['median_m']:>10,.0f} m")
        print(f"    mean NN dist    : {nn['mean_m']:>10,.0f} m")
        print(f"    min NN dist     : {nn['min_m']:>10,.0f} m")
        print(f"    max NN dist     : {nn['max_m']:>10,.0f} m")
        print(f"    frac < 500m     : {nn['frac_within_500m']*100:>9.1f}%   "
              f"(spatial autocorrelation HIGH if >5%)")
        print(f"    frac < 2 km     : {nn['frac_within_2km']*100:>9.1f}%")
        print(f"    frac < 10 km    : {nn['frac_within_10km']*100:>9.1f}%")

    # ── Validity issues ──────────────────────────────────────────────
    print("\n── 7.  GEOMETRY VALIDITY ───────────────────────────────────────────")
    invalid = df_all[~df_all["valid"]]
    if len(invalid) == 0:
        print("  All polygons are topologically valid.")
    else:
        print(f"  {len(invalid)} invalid polygon(s):")
        for _, r in invalid.head(15).iterrows():
            print(f"    {r['class']:14s} {r['file']:40s}  reason={r['validity_reason'][:60]}")

    # ── Tiny polygons (<0.5 ha) ──────────────────────────────────────
    print("\n── 8.  TINY POLYGONS (<0.5 ha — risk of pixel-count starvation) ───")
    tiny = df_all[df_all["area_ha"] < 0.5]
    print(f"  Total tiny polygons: {len(tiny)} ({100*len(tiny)/max(1,len(df_all)):.1f}%)")
    print(f"    sugarcane     : {((tiny['class']=='sugarcane')).sum()}")
    print(f"    non_sugarcane : {((tiny['class']=='non_sugarcane')).sum()}")

    # ── Final pixel-budget summary ──────────────────────────────────
    print("\n── 9.  TRAINING-PIXEL BUDGET (with max_pixels_per_plot=100 cap) ───")
    cap = 100
    pos_pix = df_all[df_all["class"] == "sugarcane"]["approx_pixels_10m"].clip(upper=cap).sum()
    neg_pix = df_all[df_all["class"] == "non_sugarcane"]["approx_pixels_10m"].clip(upper=cap).sum()
    print(f"  sugarcane pixels (capped @ {cap}/plot)     : {pos_pix:,}")
    print(f"  non_sugarcane pixels (capped @ {cap}/plot) : {neg_pix:,}")
    print(f"  TOTAL training pixels                      : {pos_pix + neg_pix:,}")
    print(f"  Effective independent samples (~plot count): {df_all['file'].nunique()} files / "
          f"~{df_all.groupby('class')['file'].nunique().to_dict()}")

    # Save full audit table
    out_path = Path("kml_audit_full.csv")
    df_all.to_csv(out_path, index=False)
    print(f"\n[Saved full per-polygon audit table → {out_path}]")
    print(f"{'═'*72}\n")


if __name__ == "__main__":
    main()
