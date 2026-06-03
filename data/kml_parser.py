"""
kml_parser.py
=============
Parse KML / KMZ files into geopandas GeoDataFrames.

Supports:
  - Single KML with one or many Placemark polygons
  - KMZ (zipped KML)
  - Batch parsing of a directory of KML/KMZ files
  - State/region tagging from filename or embedded metadata
  - Coordinate validation and CRS normalisation (EPSG:4326)

Anchor-date extraction
----------------------
KML filenames may embed the survey/confirmation date,
e.g. "sugarcane_2025-09-01_001.kml", "1_3aug2023.kml".

``_extract_date_from_filename()`` parses these patterns and stores the result
as ``anchor_date`` (a ``datetime.date``) in the GeoDataFrame.  The anchor date
is then used by ``SampleGenerator`` to define the GEE download window:
  start = anchor_date − 6 months
  end   = anchor_date + 7 months
This guarantees a full growth cycle of confirmed crop presence is captured.

If no date is found in the filename, the parser falls back to
``default_anchor_date`` from config.yaml (if set).

KML file location
-----------------
Place all your KML files in:
    sugarcane_detection_UP/data/kml/sugarcane/       (positive)
    sugarcane_detection_UP/data/kml/non_sugarcane/    (negative)

Example:
    sugarcane_detection_UP/
    └── data/
        └── kml/
            ├── sugarcane/
            │   ├── 1.kml
            │   └── ...
            └── non_sugarcane/
                ├── rice_2025-06-15_001.kml
                └── ...

Then run:
    from data.kml_parser import KMLParser
    parser = KMLParser()
    gdf = parser.parse_directory("data/kml/sugarcane/")
    print(gdf[["plot_id", "anchor_date", "state", "area_ha"]])

Usage
-----
    from data.kml_parser import KMLParser

    parser = KMLParser()
    gdf = parser.parse_file("data/kml/sugarcane/1.kml", state="Uttar Pradesh")
    gdf = parser.parse_directory("data/kml/sugarcane/")   # all KMLs in folder
    gdf = parser.parse_files(["a.kml", "b.kmz"])
"""

from __future__ import annotations

import logging
import os
import re
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

logger = logging.getLogger(__name__)

import fiona
fiona.drvsupport.supported_drivers['KML'] = 'rw'
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'

# fiona driver for KML
_KML_DRIVER = "KML"

# ---------------------------------------------------------------------------
# Anchor-date extraction
# ---------------------------------------------------------------------------

# Month abbreviation → month number (handles English + common Hindi romanisation)
_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    # common alternate spellings found in field data
    "january": 1, "february": 2, "march": 3, "april": 4,
    "june": 6, "july": 7, "august": 8, "september": 9,
    "october": 10, "november": 11, "december": 12,
}

# Regex patterns tried in order of specificity
# Matches: 3aug2023, 15jan2024, 20sep2023, 2023-08-03, 03/08/2023, 20230803
_DATE_PATTERNS: List[Tuple[str, str]] = [
    # "3aug2023" or "15jan2024"  (day + month-abbr + 4-digit year)
    (r"(\d{1,2})([a-z]{3,9})(\d{4})", "dmy_abbr"),
    # "aug2023" or "august2023"  (month-abbr + 4-digit year, no day → day=1)
    (r"([a-z]{3,9})(\d{4})", "my_abbr"),
    # ISO: "2023-08-03" or "2023_08_03"
    (r"(\d{4})[-_](\d{2})[-_](\d{2})", "iso"),
    # "03/08/2023" or "03-08-2023"  (DD/MM/YYYY)
    (r"(\d{2})[/\-](\d{2})[/\-](\d{4})", "dmy_num"),
    # Compact: "20230803"
    (r"(\d{8})", "compact"),
]


def _extract_date_from_filename(filename: str) -> Optional[date]:
    """
    Extract an anchor date from a KML filename.

    Handles various date formats in KML filenames:
      "sugarcane_2025-09-01_001.kml" → 2025-09-01
      "1_3aug2023.kml"              → 2023-08-03
      "survey_2023-08-03"           → 2023-08-03
      "20230803_field"              → 2023-08-03

    Returns None if no recognisable date is found.
    """
    stem = Path(filename).stem.lower()

    for pattern, fmt in _DATE_PATTERNS:
        m = re.search(pattern, stem)
        if not m:
            continue
        try:
            if fmt == "dmy_abbr":
                day = int(m.group(1))
                month = _MONTH_MAP.get(m.group(2))
                year = int(m.group(3))
                if month and 1 <= day <= 31 and 2000 <= year <= 2100:
                    return date(year, month, day)

            elif fmt == "my_abbr":
                month = _MONTH_MAP.get(m.group(1))
                year = int(m.group(2))
                if month and 2000 <= year <= 2100:
                    return date(year, month, 1)

            elif fmt == "iso":
                year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if 2000 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                    return date(year, month, day)

            elif fmt == "dmy_num":
                day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if 2000 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                    return date(year, month, day)

            elif fmt == "compact":
                s = m.group(1)
                year, month, day = int(s[:4]), int(s[4:6]), int(s[6:8])
                if 2000 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                    return date(year, month, day)

        except (ValueError, TypeError):
            continue

    logger.debug(f"No anchor date found in filename: {filename}")
    return None


def _anchor_to_date_range(
    anchor: date,
    months_before: int = 3,
    months_after: int = 3,
) -> Tuple[str, str]:
    """
    Convert an anchor date to a (start_date, end_date) string pair.

    Default window: 6 months before → 7 months after the anchor date.
    This captures a full sugarcane phenological cycle centred on the confirmed
    crop presence date (14 months total).

    Returns
    -------
    (start_date, end_date) as "YYYY-MM-DD" strings
    """
    from dateutil.relativedelta import relativedelta

    start = anchor - relativedelta(months=months_before)
    end = anchor + relativedelta(months=months_after)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

# ---------------------------------------------------------------------------
# State detection via centroid coordinates (approximate bounding boxes)
# ---------------------------------------------------------------------------
# These are approximate lat/lon bounding boxes for Indian states relevant
# to sugarcane cultivation in UP.
# Format: (state_name, min_lat, max_lat, min_lon, max_lon)
_STATE_BOUNDS = [
    ("Tamil Nadu",       8.0,  13.6,  76.2,  80.4),
    ("Karnataka",       11.5,  18.5,  74.0,  78.6),
    ("Kerala",           8.2,  12.8,  74.8,  77.4),
    ("Gujarat",         20.1,  24.7,  68.2,  74.5),
    ("Andhra Pradesh",  12.4,  19.9,  76.7,  84.8),
    ("Telangana",       15.8,  19.9,  77.2,  81.3),
    ("Madhya Pradesh",  21.1,  26.9,  74.0,  82.8),
    ("Uttar Pradesh",   23.9,  30.4,  77.1,  84.6),
    ("Bihar",           24.3,  27.5,  83.3,  88.2),
    ("West Bengal",     21.5,  27.2,  86.0,  89.9),
    ("Odisha",          17.8,  22.6,  81.4,  87.5),
    ("Assam",           24.0,  28.2,  89.7,  96.0),
    ("Rajasthan",       23.0,  30.2,  69.5,  78.3),
    ("Chhattisgarh",    17.8,  24.1,  80.2,  84.4),
    ("Jharkhand",       21.9,  25.3,  83.3,  87.9),
    ("Punjab",          29.5,  32.5,  73.9,  76.9),
    ("Haryana",         27.4,  30.9,  74.5,  77.6),
    ("Goa",             14.9,  15.8,  73.7,  74.3),
]


def _infer_state_from_coordinates(lat: float, lon: float) -> Optional[str]:
    """
    Infer state from centroid coordinates using offline reverse geocoding.
    Highly accurate and works globally.
    """
    try:
        import reverse_geocoder as rg
        results = rg.search(((lat, lon),), verbose=False)
        if results and len(results) > 0:
            return results[0].get("admin1", None)
    except Exception:
        pass
    return None


def _read_kml(path: Union[str, Path]) -> gpd.GeoDataFrame:
    """
    Read a KML or KMZ file and return a GeoDataFrame.
    Handles KMZ by extracting the inner doc.kml first.
    Enables the fiona KML driver explicitly (required on Windows).
    Falls back to xml-based parsing if fiona KML driver is unavailable.
    """
    import fiona
    # Enable KML driver — required on Windows where it's off by default
    fiona.supported_drivers["KML"]  = "rw"
    fiona.supported_drivers["LIBKML"] = "rw"

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".kmz":
        with zipfile.ZipFile(path, "r") as zf:
            kml_names = [n for n in zf.namelist() if n.lower().endswith(".kml")]
            if not kml_names:
                raise ValueError(f"No KML found inside KMZ: {path}")
            kml_name = next((n for n in kml_names if "doc.kml" in n.lower()), kml_names[0])
            with zf.open(kml_name) as kml_file:
                kml_bytes = kml_file.read()
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".kml", delete=False) as tmp:
            tmp.write(kml_bytes)
            tmp_path = tmp.name
        try:
            gdf = _read_kml_file(tmp_path)
        finally:
            os.unlink(tmp_path)
    else:
        gdf = _read_kml_file(str(path))

    return gdf


def _read_kml_file(path: str) -> gpd.GeoDataFrame:
    """Try fiona KML driver first, fall back to xml parsing."""
    import fiona

    # Try KML driver
    for driver in ["KML", "LIBKML"]:
        if driver in fiona.supported_drivers:
            try:
                gdf = gpd.read_file(path, driver=driver)
                if not gdf.empty:
                    return gdf
            except Exception:
                pass

    # Fallback: parse KML XML directly with shapely
    return _parse_kml_xml(path)


def _parse_kml_xml(path: str) -> gpd.GeoDataFrame:
    """
    Parse KML file using Python's built-in xml.etree.ElementTree.
    Extracts all Polygon coordinates without needing fiona KML driver.
    """
    import xml.etree.ElementTree as ET
    from shapely.geometry import Polygon as ShapelyPolygon

    ns = {
        "kml": "http://www.opengis.net/kml/2.2",
        "": "http://www.opengis.net/kml/2.2",
    }

    tree = ET.parse(path)
    root = tree.getroot()

    # Strip namespace for easier searching
    def _strip_ns(tag):
        return tag.split("}")[-1] if "}" in tag else tag

    def _find_all(element, tag):
        return [e for e in element.iter() if _strip_ns(e.tag) == tag]

    def _coords_from_text(text: str):
        coords = []
        for point in text.split():
            parts = point.split(",")
            if len(parts) >= 2:
                try:
                    lon, lat = float(parts[0]), float(parts[1])
                    coords.append((lon, lat))
                except ValueError:
                    pass
        return coords

    rows = []
    for placemark in _find_all(root, "Placemark"):
        name_el = next(iter(_find_all(placemark, "name")), None)
        name = name_el.text if name_el is not None else ""

        # 1) Real Polygon placemarks
        for polygon_el in _find_all(placemark, "Polygon"):
            outer = _find_all(polygon_el, "outerBoundaryIs")
            if not outer:
                continue
            coords_el = _find_all(outer[0], "coordinates")
            if not coords_el or coords_el[0].text is None:
                continue
            coords = _coords_from_text(coords_el[0].text.strip())
            if len(coords) >= 3:
                try:
                    geom = ShapelyPolygon(coords)
                    if geom.area > 0:
                        rows.append({"geometry": geom, "Name": name, "Description": ""})
                except Exception:
                    pass

        # 2) LineString placemarks that form a *closed ring* — many KMLs hand-drawn
        #    in Google Earth Pro come back as LineString with first==last point.
        #    Treat these as polygon outer boundaries.
        for line_el in _find_all(placemark, "LineString"):
            coords_el = _find_all(line_el, "coordinates")
            if not coords_el or coords_el[0].text is None:
                continue
            coords = _coords_from_text(coords_el[0].text.strip())
            if len(coords) < 4:
                continue
            # Auto-close if first ≈ last within ~10 m (1e-4 deg)
            _CLOSE_TOL = 1e-4
            if coords[0] != coords[-1]:
                if (abs(coords[0][0] - coords[-1][0]) > _CLOSE_TOL
                        or abs(coords[0][1] - coords[-1][1]) > _CLOSE_TOL):
                    continue  # truly open path
                coords[-1] = coords[0]
            try:
                geom = ShapelyPolygon(coords)
                if not geom.is_valid:
                    geom = make_valid(geom)
                if geom.area > 0:
                    rows.append({"geometry": geom, "Name": name or "from_linestring",
                                 "Description": "closed_linestring"})
            except Exception:
                pass

    if not rows:
        raise ValueError(f"No polygon geometries found in KML: {path}")

    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def _extract_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Keep only Polygon / MultiPolygon geometries.
    Validates and repairs invalid geometries.
    Forces 2D coordinates (drops Z/altitude) — required for GEE compatibility.

    Closed-ring LineStrings (a common Google Earth Pro export pattern) are
    promoted to Polygons before filtering.
    """
    from shapely.geometry import LineString as _LS, Polygon as _Poly
    from shapely.ops import transform as shapely_transform

    # Explode geometry collections
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)

    # Promote closed LineStrings to Polygons. Hand-drawn paths in Google Earth
    # Pro often miss perfect closure by 1–10 m, so we auto-close when the gap
    # is < ~10 m (~1e-4 deg).
    _CLOSE_TOL = 1e-4  # ≈ 10 m at UP latitude

    def _promote_linestring(geom):
        if isinstance(geom, _LS):
            coords = list(geom.coords)
            if len(coords) >= 4:
                if coords[0] != coords[-1]:
                    if (abs(coords[0][0] - coords[-1][0]) <= _CLOSE_TOL
                            and abs(coords[0][1] - coords[-1][1]) <= _CLOSE_TOL):
                        coords[-1] = coords[0]
                    else:
                        return geom  # truly open → leave alone
                try:
                    poly = _Poly([(c[0], c[1]) for c in coords])
                    if poly.area > 0:
                        return poly
                except Exception:
                    pass
        return geom

    gdf["geometry"] = gdf["geometry"].apply(_promote_linestring)

    # Keep only polygon types
    mask = gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    gdf = gdf[mask].copy()

    if gdf.empty:
        raise ValueError("No polygon geometries found in KML.")

    # Force 2D — drop Z coordinate (KML files include altitude=0 which breaks GEE)
    def _force_2d(geom):
        return shapely_transform(lambda x, y, z=None: (x, y), geom)

    gdf["geometry"] = gdf["geometry"].apply(_force_2d)

    # Validate / repair
    gdf["geometry"] = gdf["geometry"].apply(make_valid)

    # Ensure CRS is WGS84
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    return gdf


class KMLParser:
    """
    Parse KML/KMZ files into a unified GeoDataFrame with metadata columns:
      - geometry    : Shapely Polygon/MultiPolygon (EPSG:4326)
      - plot_id     : Unique identifier per polygon
      - state       : Indian state name (inferred or provided)
      - source_file : Original filename
      - area_ha     : Approximate area in hectares
      - anchor_date : Survey/confirmation date extracted from filename
                      (e.g. "sugarcane_2025-09-01_001.kml" → 2025-09-01)
                      Falls back to default_anchor_date from config.yaml
      - date_start  : GEE download start = anchor_date − 6 months
      - date_end    : GEE download end   = anchor_date + 7 months

    KML file location
    -----------------
    Place all KML files in:  sugarcane_detection_UP/data/kml/
    """

    def __init__(self, months_before: int = 5, months_after: int = 5, config_path: str = "config.yaml"):
        """
        Parameters
        ----------
        months_before : months before anchor_date to include in GEE window
        months_after  : months after anchor_date to include in GEE window
        config_path   : path to config.yaml (for default_anchor_date)
        """
        self._plot_counter = 0
        self.months_before = months_before
        self.months_after = months_after
        # Load default anchor date from config if available
        self._default_anchor_date = None
        try:
            import yaml as _yaml
            with open(config_path) as f:
                cfg = _yaml.safe_load(f)
            raw = cfg.get("default_anchor_date", "")
            if raw:
                self._default_anchor_date = datetime.strptime(str(raw), "%Y-%m-%d").date()
                logger.info(f"Default anchor date from config: {self._default_anchor_date}")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_file(
        self,
        kml_path: Union[str, Path],
        state: Optional[str] = None,
        label: int = 1,
        anchor_date_override: Optional[date] = None,
    ) -> gpd.GeoDataFrame:
        """
        Parse a single KML/KMZ file.

        Parameters
        ----------
        kml_path             : path to .kml or .kmz file
        state                : override state name; if None, inferred from filename
        label                : class label (1 = sugarcane, 0 = non-sugarcane)
        anchor_date_override : manually supply anchor date (overrides filename parse)

        Returns
        -------
        GeoDataFrame with columns: geometry, plot_id, state, source_file,
                                   area_ha, label, anchor_date,
                                   date_start, date_end
        """
        kml_path = Path(kml_path)
        if not kml_path.exists():
            raise FileNotFoundError(f"KML file not found: {kml_path}")

        logger.info(f"Parsing KML: {kml_path}")
        raw_gdf = _read_kml(kml_path)
        gdf = _extract_polygons(raw_gdf)

        # Infer state from centroid coordinates (much more reliable than filename)
        if state:
            inferred_state = state
        else:
            centroid = gdf.geometry.unary_union.centroid
            inferred_state = _infer_state_from_coordinates(centroid.y, centroid.x)

        # Extract anchor date from filename
        anchor = anchor_date_override or _extract_date_from_filename(kml_path.name)
        # Fallback to default anchor date from config if no date in filename
        if anchor is None and self._default_anchor_date is not None:
            anchor = self._default_anchor_date
            logger.info(f"  Using default_anchor_date from config: {anchor}")
        if isinstance(anchor, str):
            from datetime import datetime
            try:
                anchor = datetime.strptime(anchor, "%Y-%m-%d").date()
            except ValueError:
                try:
                    anchor = datetime.fromisoformat(anchor).date()
                except ValueError:
                    pass

        if anchor:
            date_start, date_end = _anchor_to_date_range(
                anchor, self.months_before, self.months_after
            )
            logger.info(
                f"  Anchor date: {anchor} -> "
                f"GEE window [{date_start} -> {date_end}]"
            )
        else:
            date_start = date_end = None
            logger.warning(
                f"  No anchor date found in '{kml_path.name}'. "
                f"date_start/date_end will be None — "
                f"pass explicit dates to generate() or set anchor_date_override."
            )

        # Build metadata rows
        rows = []
        for _, row in gdf.iterrows():
            self._plot_counter += 1
            geom = row.geometry
            area_ha = self._compute_area_ha(geom)
            rows.append(
                {
                    "geometry": geom,
                    "plot_id": f"plot_{self._plot_counter:05d}",
                    "state": inferred_state or "Unknown",
                    "source_file": kml_path.name,
                    "area_ha": round(area_ha, 4),
                    "label": label,
                    "anchor_date": anchor,
                    "date_start": date_start,
                    "date_end": date_end,
                    # Preserve original Name/Description if present
                    "name": row.get("Name", ""),
                    "description": row.get("Description", ""),
                }
            )

        result = gpd.GeoDataFrame(rows, crs="EPSG:4326")
        logger.info(
            f"  -> {len(result)} polygon(s) | state={inferred_state} | "
            f"anchor={anchor} | area={result['area_ha'].sum():.2f} ha"
        )
        return result

    def parse_files(
        self,
        kml_paths: List[Union[str, Path]],
        state_map: Optional[dict] = None,
        label: int = 1,
    ) -> gpd.GeoDataFrame:
        """
        Parse a list of KML/KMZ files and concatenate into one GeoDataFrame.

        Anchor dates are extracted automatically from each filename.
        Plots without a parseable date will have anchor_date=None and
        date_start/date_end=None — you must supply explicit dates to
        SampleGenerator.generate() for those plots.

        Parameters
        ----------
        kml_paths : list of file paths
        state_map : dict mapping filename stem → state name (optional override)
        label     : default class label

        Returns
        -------
        Combined GeoDataFrame
        """
        state_map = state_map or {}
        frames = []
        for p in kml_paths:
            p = Path(p)
            state_override = state_map.get(p.stem) or state_map.get(p.name)
            try:
                gdf = self.parse_file(p, state=state_override, label=label)
                frames.append(gdf)
            except Exception as exc:
                logger.warning(f"Skipping {p}: {exc}")

        if not frames:
            raise ValueError("No valid KML files could be parsed.")

        combined = gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True), crs="EPSG:4326"
        )
        logger.info(
            f"Parsed {len(combined)} total polygons from {len(frames)} files."
        )
        return combined

    def parse_directory(
        self,
        directory: Union[str, Path],
        state_map: Optional[dict] = None,
        label: int = 1,
        recursive: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Parse all KML/KMZ files in a directory.

        Parameters
        ----------
        directory : path to folder
        state_map : optional filename→state override dict
        label     : default class label
        recursive : if True, search subdirectories too

        Returns
        -------
        Combined GeoDataFrame
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        pattern = "**/*.kml" if recursive else "*.kml"
        kml_files = list(directory.glob(pattern))
        kmz_pattern = "**/*.kmz" if recursive else "*.kmz"
        kml_files += list(directory.glob(kmz_pattern))

        if not kml_files:
            raise FileNotFoundError(f"No KML/KMZ files found in: {directory}")

        logger.info(f"Found {len(kml_files)} KML/KMZ files in {directory}")
        return self.parse_files(kml_files, state_map=state_map, label=label)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_bbox(gdf: gpd.GeoDataFrame) -> dict:
        """Return bounding box dict {minx, miny, maxx, maxy} for a GeoDataFrame."""
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        return {
            "minx": bounds[0],
            "miny": bounds[1],
            "maxx": bounds[2],
            "maxy": bounds[3],
        }

    @staticmethod
    def to_ee_geometry(gdf: gpd.GeoDataFrame):
        """
        Convert GeoDataFrame to a Google Earth Engine geometry (union of all polygons).
        Requires earthengine-api to be installed and authenticated.
        """
        try:
            import ee
        except ImportError:
            raise ImportError("earthengine-api not installed. Run: pip install earthengine-api")

        union_geom = unary_union(gdf.geometry.values)
        geojson = union_geom.__geo_interface__
        return ee.Geometry(geojson)

    @staticmethod
    def _compute_area_ha(geom) -> float:
        """
        Approximate area in hectares using an equal-area projection.
        Uses EPSG:32643 (UTM Zone 43N) as a reasonable default for India.
        """
        try:
            from pyproj import Transformer
            from shapely.ops import transform as shapely_transform

            transformer = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
            projected = shapely_transform(transformer.transform, geom)
            return projected.area / 10_000  # m² → ha
        except Exception:
            # Fallback: rough degree-based estimate
            return geom.area * 111_320 * 111_320 / 10_000

    @staticmethod
    def split_by_state(gdf: gpd.GeoDataFrame) -> dict:
        """
        Split a combined GeoDataFrame into per-state sub-DataFrames.

        Returns
        -------
        dict mapping state_name → GeoDataFrame
        """
        return {
            state: sub_gdf.reset_index(drop=True)
            for state, sub_gdf in gdf.groupby("state")
        }

    @staticmethod
    def validate(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Validate and repair geometries in-place.
        Removes empty / null geometries.
        """
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
        gdf["geometry"] = gdf["geometry"].apply(make_valid)
        return gdf.reset_index(drop=True)

    @staticmethod
    def anchor_to_date_range(
        anchor: date,
        months_before: int = 12,
        months_after: int = 12,
    ) -> Tuple[str, str]:
        """
        Public wrapper around ``_anchor_to_date_range``.
        Useful when you want to compute the GEE window outside of parse_file.

        Returns
        -------
        (start_date, end_date) as "YYYY-MM-DD" strings
        """
        return _anchor_to_date_range(anchor, months_before, months_after)

    @staticmethod
    def get_date_range_for_plot(row: pd.Series) -> Tuple[Optional[str], Optional[str]]:
        """
        Return (date_start, date_end) for a single GeoDataFrame row.
        Falls back to (None, None) if anchor_date is missing.
        """
        return row.get("date_start"), row.get("date_end")


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python kml_parser.py <path_to_kml_or_directory>")
        sys.exit(1)

    target = Path(sys.argv[1])
    parser = KMLParser()

    if target.is_dir():
        gdf = parser.parse_directory(target)
    else:
        gdf = parser.parse_file(target)

    print(gdf[["plot_id", "anchor_date", "date_start", "date_end", "state", "area_ha", "label"]].to_string())
    print(f"\nTotal plots : {len(gdf)}")
    print(f"Total area  : {gdf['area_ha'].sum():.2f} ha")
    print(f"States      : {gdf['state'].unique().tolist()}")
    missing_dates = gdf["anchor_date"].isna().sum()
    if missing_dates:
        print(f"\nWARNING: {missing_dates} plot(s) have no anchor date in filename.")
        print("  Rename files to include a date, e.g. '1_3aug2023.kml'.")
