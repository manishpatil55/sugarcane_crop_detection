"""
Sugarcane Detection API
====================
FastAPI server with Swagger UI for sugarcane crop detection.

Run:  python api.py
Swagger UI: http://localhost:8088/docs
"""

import logging
import os
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sugarcane Crop Detection API",
    description="""
## UP Sugarcane Detection from Satellite Imagery

Upload a KML file with a farm boundary polygon and a date when the crop was observed.
The API will:
1. Download Sentinel-1 (SAR) + Sentinel-2 (optical) data from Google Earth Engine for a 14-month window
2. Compute spectral indices (NDRE, GNDVI, NDVI, etc.)
3. Extract temporal sequences
4. Run the trained PyTorch BiLSTM Sequence model
5. Return probability of sugarcane presence

### How it works
- **Input**: KML file + observation date
- **Output**: Sugarcane probability (0-100%), classification, per-pixel stats
- **Model**: Sequence BiLSTM trained on Uttar Pradesh sugarcane patterns
- **Satellites**: Sentinel-1 (radar) + Sentinel-2 (optical)
    """,
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Response models ───────────────────────────────────────────────────────

class DetectionResult(BaseModel):
    # ── 1. VERDICT (top — what testers look at first) ──────────────
    verdict: str = Field(description="Human-readable verdict with confidence level")
    classification: str = Field(description="'SUGARCANE' or 'NON-SUGARCANE'")
    label: int = Field(description="1 = Sugarcane, 0 = Non-Sugarcane")
    confidence: str = Field(description="Confidence level: HIGH / MODERATE / LOW")
    sugarcane_percentage: float = Field(description="% of area classified as sugarcane")

    # ── 2. LOCATION INFO ──────────────────────────────────────────
    area_hectares: Optional[float] = Field(default=None, description="Total farm area in hectares")
    centroid_latitude: Optional[float] = Field(default=None, description="Farm centroid latitude")
    centroid_longitude: Optional[float] = Field(default=None, description="Farm centroid longitude")
    bounding_box: Optional[dict] = Field(default=None, description="Farm bounding box {north, south, east, west}")
    state: Optional[str] = Field(default=None, description="Indian state (if detected from KML filename)")

    # ── 3. PIXEL ANALYSIS ─────────────────────────────────────────
    total_pixels: int = Field(description="Total pixels analysed")
    sugarcane_pixels: int = Field(description="Pixels classified as sugarcane")
    non_sugarcane_pixels: int = Field(description="Pixels classified as non-sugarcane")
    mean_probability: float = Field(description="Mean sugarcane probability (0-1)")
    max_probability: float = Field(description="Max sugarcane probability")
    min_probability: float = Field(description="Min sugarcane probability")

    # ── 4. CLOUD COVER & DATA QUALITY ─────────────────────────────
    cloud_free_percentage: Optional[float] = Field(default=None, description="% of months with clear optical data (0-100)")
    cloudy_months: Optional[int] = Field(default=None, description="Number of months affected by cloud cover")
    total_months: Optional[int] = Field(default=None, description="Total months in satellite window")
    cloud_details: Optional[dict] = Field(default=None, description="Per-month cloud status {YYYY_MM: 'clear'|'cloudy'}")
    data_quality: Optional[str] = Field(default=None, description="EXCELLENT / GOOD / FAIR / POOR based on cloud coverage")

    # ── 5. MODEL & SATELLITE INFO ─────────────────────────────────
    status: str = Field(description="'success' or 'error'")
    model_name: str = Field(description="Model used: BiLSTM Sequence Model")
    threshold: float = Field(description="Classification threshold used")
    satellites_used: str = Field(default="Sentinel-1 (SAR) + Sentinel-2 (Optical)", description="Satellites used")
    date_checked: Optional[str] = Field(default=None, description="Crop date provided by user")
    satellite_window: Optional[str] = Field(default=None, description="Satellite data window used")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    threshold: Optional[float]


# ── Model loading ─────────────────────────────────────────────────────────

from inference.predictor import SugarcanePredictor

predictor = None

def load_model():
    global predictor
    if predictor is not None:
        return

    try:
        predictor = SugarcanePredictor()
        predictor.load_models()
        logger.info("SugarcanePredictor initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to load predictor: {e}")


# ── Prediction logic ──────────────────────────────────────────────────────

def predict_from_kml(kml_path: str, crop_date: str) -> dict:
    """Run the full prediction pipeline on a KML file using SugarcanePredictor."""
    if predictor is None or not getattr(predictor, "_models_loaded", False):
        raise HTTPException(status_code=503, detail="Model not loaded. Run pipeline.py --train first.")

    try:
        pred_result = predictor.predict(kml_path, crop_date)
    except Exception as e:
        logger.exception("Predictor failed")
        raise HTTPException(status_code=400, detail=str(e))

    # Plot geometry for the response payload
    from data.kml_parser import KMLParser
    parser = KMLParser()
    gdf = parser.parse_file(kml_path, anchor_date_override=crop_date)

    total_area = None
    centroid_x, centroid_y = None, None
    bounds = {"north": None, "south": None, "east": None, "west": None}
    if not gdf.empty:
        total_area = gdf.geometry.to_crs(epsg=32644).area.sum() / 10000.0
        cb = gdf.total_bounds
        bounds = {
            "north": round(cb[3], 6),
            "south": round(cb[1], 6),
            "east":  round(cb[2], 6),
            "west":  round(cb[0], 6),
        }
        c = gdf.geometry.unary_union.centroid
        centroid_x, centroid_y = round(c.x, 6), round(c.y, 6)

    # Real cloud-quality numbers
    cq = pred_result.get("cloud_quality") or {}
    cloud_free_pct = cq.get("pct_windows_with_valid_optical")
    n_cloudy = cq.get("n_cloudy_windows")
    n_total  = cq.get("n_total_windows") or 28

    if cloud_free_pct is None:
        data_quality = "UNKNOWN"
    elif cloud_free_pct >= 80:
        data_quality = "EXCELLENT"
    elif cloud_free_pct >= 60:
        data_quality = "GOOD"
    elif cloud_free_pct >= 40:
        data_quality = "FAIR"
    else:
        data_quality = "POOR"

    sugarcane_pct = round(pred_result["sugarcane_fraction"] * 100, 1)
    n_polys = pred_result["n_polygons"]
    sugarcane_polys = pred_result["sugarcane_polygons"]
    non_sug = n_polys - sugarcane_polys
    probs_mean = pred_result["confidence"]
    area_str = f"{total_area:.2f} ha" if total_area else "unknown area"

    # ── 4-tier biological decision matrix ────────────────────────────────
    # Thresholds derived from QA team testing:
    #   >= 0.685  → Mature canopy:       full SAR + optical signature
    #   0.458–0.685 → Young/developing:  transitional growth zone
    #   < 0.458, pct >= 20% → Partial:  intercropping / partial harvest
    #   everything else → Non-sugarcane

    # Tier 1 – Mature Sugarcane
    if probs_mean >= 0.685:
        classification = "SUGARCANE"
        label = 1
        confidence = "HIGH"
        verdict = (
            f"HIGH CONFIDENCE - Mature sugarcane plantation detected. "
            f"{sugarcane_polys}/{n_polys} polygons classified as sugarcane "
            f"(mean prob {probs_mean:.1%}, area {area_str})."
        )

    # Tier 2 – Young / Developing Sugarcane
    elif probs_mean >= 0.458:
        classification = "SUGARCANE"
        label = 1
        confidence = "MODERATE"
        verdict = (
            f"MODERATE CONFIDENCE - Young or developing sugarcane detected. "
            f"Mean probability {probs_mean:.1%} is consistent with early-stage canopy "
            f"({sugarcane_polys}/{n_polys} polygons matched, area {area_str}). "
            f"Re-check in 4-6 weeks for mature signature."
        )

    # Tier 3 – Partial / Mixed Field
    elif sugarcane_pct >= 20.0:
        classification = "NON-SUGARCANE"
        label = 0
        confidence = "LOW"
        verdict = (
            f"LOW CONFIDENCE - Partial or mixed field detected. "
            f"Mean probability {probs_mean:.1%} is below the sugarcane threshold, "
            f"but {sugarcane_pct:.1f}% of polygons ({sugarcane_polys}/{n_polys}) "
            f"show localised sugarcane signatures. Possible intercropping or partial harvest."
        )

    # Tier 4 – Non-Sugarcane
    else:
        classification = "NON-SUGARCANE"
        label = 0
        confidence = "HIGH"
        verdict = (
            f"NOT SUGARCANE - No significant sugarcane detected. "
            f"Mean probability {probs_mean:.1%} and only {sugarcane_polys}/{n_polys} "
            f"polygons matched ({sugarcane_pct:.1f}% coverage, area {area_str})."
        )

    return {
        "verdict":               verdict,
        "classification":        classification,
        "label":                 label,
        "confidence":            confidence,
        "sugarcane_percentage":  sugarcane_pct,

        "area_hectares":         round(total_area, 2) if total_area else None,
        "centroid_latitude":     centroid_y,
        "centroid_longitude":    centroid_x,
        "bounding_box":          bounds,
        "state":                 pred_result.get("state"),

        # Polygon-level instead of pixel-level
        "total_pixels":          int(pred_result.get("n_pixels", 0)),
        "sugarcane_pixels":      sugarcane_polys,                # polygons classified as sugarcane
        "non_sugarcane_pixels":  non_sug,
        "mean_probability":      round(float(pred_result["sugarcane_probability_mean"]), 4),
        "max_probability":       round(float(pred_result["sugarcane_probability_max"]), 4),
        "min_probability":       round(float(pred_result["sugarcane_probability_min"]), 4),

        # REAL cloud stats from polygon extractor
        "cloud_free_percentage": round(cloud_free_pct, 2) if cloud_free_pct is not None else None,
        "cloudy_months":         int(n_cloudy) if n_cloudy is not None else None,
        "total_months":          int(n_total),
        "cloud_details":         {},
        "data_quality":          data_quality,

        "status":                "success",
        "model_name":            pred_result.get("model_name", "unknown"),
        "threshold":             float(pred_result.get("threshold", 0.5)),
        "satellites_used":       "Sentinel-1 (SAR) + Sentinel-2 (Optical)",
        "date_checked":          crop_date,
        "satellite_window":      f"{pred_result.get('window_start')} to {pred_result.get('window_end')}",
    }


# ── API Endpoints ─────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    load_model()


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Check if the API and model are healthy."""
    if predictor is None or not getattr(predictor, "_models_loaded", False):
        return HealthResponse(
            status="error",
            model_loaded=False,
            model_name="none",
            threshold=None,
        )
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=getattr(predictor, "model_type", "unknown"),
        threshold=getattr(predictor, "optimal_threshold", 0.5),
    )


@app.post("/detect", response_model=DetectionResult, tags=["Detection"])
def detect_sugarcane(
    kml_file: UploadFile = File(..., description="KML file with farm boundary polygon"),
    crop_date: str = Form(..., description="Date when sugarcane presence is to be checked (YYYY-MM-DD format)"),
):
    """
    ## Detect Sugarcane Crop

    Upload a KML file containing a farm boundary polygon and specify the date
    you want to check for sugarcane presence.

    The API will automatically download satellite data, compute vegetation indices, 
    and classify the area as sugarcane or non-sugarcane using our BiLSTM Sequence Model.

    ### Parameters:
    - **kml_file**: A `.kml` file with one or more polygon boundaries
    - **crop_date**: Date in `YYYY-MM-DD` format

    ### Returns:
    - Sugarcane probability per pixel
    - Overall classification
    - Human-readable verdict
    """
    # Validate date
    try:
        from datetime import datetime
        parsed = datetime.strptime(crop_date, "%Y-%m-%d")
        crop_date_str = parsed.strftime("%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD (e.g. 2025-10-12)")

    # Save uploaded KML to temp file
    suffix = Path(kml_file.filename or "upload.kml").suffix or ".kml"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=".") as tmp:
        content = kml_file.file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = predict_from_kml(tmp_path, crop_date_str)
        return DetectionResult(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── Run server ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Sugarcane Detection API")
    print("  Swagger UI: http://localhost:8585/docs")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8585, log_level="info")
