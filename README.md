# 🌾 Sugarcane Crop Detection System — Uttar Pradesh

State-wide sugarcane crop detection using **Sentinel-1 SAR + Sentinel-2 optical** satellite data with a **BiLSTM + Temporal Attention** model to capture the full sugarcane phenological cycle using high-resolution 15-day composites (28 timesteps).

## Quick Start

### 1. Install

```bash
cd sugarcane_detection_UP
pip install -r requirements.txt
earthengine authenticate       # One-time GEE setup
```

### 2. Prepare Data
Place KML files in `data/kml/sugarcane/` (positive samples) and `data/kml/non_sugarcane/` (negative samples).

**Positive KMLs (250 files):** Sugarcane field boundaries from UP. All use anchor date `2025-09-01` (configured in `config.yaml → default_anchor_date`).

**Negative KMLs:** Must include confuser crops common in UP:
- **Rice** (Kharif) — similar NDVI during monsoon but harvested Oct/Nov
- **Wheat** (Rabi) — sown Nov, green Dec–Mar, harvested Apr
- **Maize** (Kharif) — short cycle, harvested by Oct
- **Mustard** (Rabi) — flowers yellow Dec–Jan, harvested Mar
- **Fallow/Bare land** — no vegetation signature

### 3. Train the BiLSTM Model

```bash
python train.py
```
This will:
1. Parse all KMLs and extract anchor dates
2. Download 14-month Sentinel-1/2 time-series (in 15-day intervals = 28 timesteps) per plot via GEE
3. Compute 26 features per timestep (10 S2 bands + 10 spectral indices + 2 S1 bands + 3 SAR indices + 1 Optical Mask)
4. Train a BiLSTM with temporal attention (with Random Forest fallback logic) using GroupKFold cross-validation
5. Save `models/saved/best_model.pth` with calibrated classification threshold

### 4. Run API Server

```bash
python api.py
# Server: http://localhost:8088
# Swagger UI: http://localhost:8088/docs
```

### 5. Detect Sugarcane (API)

```bash
curl -X POST http://localhost:8088/detect \
  -F "kml_file=@farm_boundary.kml" \
  -F "crop_date=2025-09-01"
```

### 6. Detect Sugarcane (CLI)

```bash
python inference/predictor.py farm_boundary.kml 2025-09-01
```

---

## Architecture — Sugarcane-Specific Design

### 1. Temporal Window: 14 Months (6 before + 7 after anchor)
Sugarcane in UP has a 10–18 month growth cycle. The 14-month window captures:
- **Planting phase** (Feb–Apr) — low NDVI, bare soil
- **Tillering** (May–Jun) — rapid canopy closure
- **Grand growth** (Jul–Sep) — peak biomass, NDVI > 0.7
- **Maturation** (Oct–Dec) — sustained high NDVI (key discriminator: rice/wheat fields are bare)
- **Harvest** (Jan–Mar) — sharp NDVI drop, ratoon regrowth

### 2. 25-Feature Spectral Stack Per Timestep

| Source | Features | Count |
|--------|----------|-------|
| Sentinel-2 bands | B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 | 10 |
| S2 vegetation indices | NDVI, EVI, NDWI, LSWI, SAVI, MSAVI, NBR, NDRE, GNDVI, NDMI | 10 |
| Sentinel-1 bands | VV, VH | 2 |
| SAR indices | RVI, RFDI, CR | 3 |
| Quality Control | Optical Cloud Mask | 1 |
| **Total per timestep** | | **26** |

**Key sugarcane-specific indices:**
- **NDRE** `(B8A−B5)/(B8A+B5)` — Resistant to saturation at high LAI (>4), critical during grand growth
- **NDMI** `(B8A−B11)/(B8A+B11)` — Tracks canopy moisture; irrigated sugarcane in UP has distinctly high NDMI
- **GNDVI** `(B8−B3)/(B8+B3)` — Chlorophyll concentration during peak growth

### 3. BiLSTM with Temporal Attention
- **Encoder:** 2-layer Bidirectional LSTM (64 hidden units × 2 directions = 128)
- **Attention:** Learnable month-level weights — the model discovers which months are most discriminative (typically Nov–Dec when sugarcane stands green while rice-wheat rotation fields are bare)
- **Classifier:** Linear(128→32) → ReLU → Dropout → Linear(32→1) → Sigmoid

### 4. Post-Processing NDVI Sanity Check
If mean NDVI during Oct–Jan (standing sugarcane period) is below 0.25, probabilities are penalized by 40%. This guards against false positives on bare/harvested land.

### 5. Cloud Handling (Monsoon Robustness)
- Optical Masking: Any 15-day composite timestep with <30% cloud-free pixels zeroes out all optical features.
- Savitzky-Golay temporal smoothing fills minor gaps.
- SAR features (VV, VH, RVI) are cloud-immune and dominate the BiLSTM's attention during the Jun–Sep monsoon.

---

## Project Structure

```
sugarcane_detection_UP/
├── config.yaml                       # Configuration (crop=sugarcane, port=8088)
├── train.py                          # BiLSTM training with GroupKFold + early stopping
├── api.py                            # FastAPI REST server (port 8088)
├── requirements.txt                  # Python dependencies (PyTorch, not TensorFlow)
│
├── data/
│   ├── kml_parser.py                 # KML → GeoDataFrame (default_anchor_date fallback)
│   ├── gee_downloader.py             # S1/S2 monthly composites (config cloud threshold)
│   ├── sample_generator.py           # 14-month sequence + negative ring sampling
│   └── kml/
│       ├── sugarcane/                # 250 positive sugarcane KMLs (anchor: 2025-09-01)
│       └── non_sugarcane/            # Negative KMLs (Rice, Wheat, Maize, etc.)
│
├── features/
│   ├── spectral_indices.py           # 10 S2 indices (NDRE, NDMI, GNDVI...) + 3 SAR
│   ├── temporal_stats.py             # Monsoon/dry season split statistics
│   └── phenology_features.py         # Peak/trough detection, growing season length
│
├── models/
│   ├── sequence_models.py            # SugarcaneAttentionLSTM (PyTorch BiLSTM)
│   └── saved/                        # Model checkpoints (bilstm_best.pt, rf_best.joblib, xgb_best.joblib)
│
├── inference/
│   └── predictor.py                  # SugarcanePredictor (CLI + programmatic)
│
└── utils.py                          # Shared utilities
```

---

## Configuration

All tunable parameters are in `config.yaml`:
- `sentinel2.cloud_threshold: 40` — max cloud % per S2 scene
- `compositing.months_before: 6` / `months_after: 7` — 14-month window
- `default_anchor_date: "2025-09-01"` — fallback for KMLs without dates
- `models.bilstm.epochs: 100` / `patience: 15` — training hyperparameters
- `inference.probability_threshold: 0.5` — overridden by calibrated threshold from training
