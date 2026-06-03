# Sugarcane Detection (Uttar Pradesh) — Full Technical Audit Report

> **Auditor role:** Senior ML / Remote-Sensing Engineer  
> **Audit scope:** Project state, KML data quality, feature engineering, validation strategy, model training, and accuracy feasibility for the **90–95 % F1 / OA / Kappa** target.  
> **Verdict (TL;DR):** The project has solid scaffolding and good intentions, but in its current state **it cannot reach the stated accuracy target**. There are three blocking issues: (1) extreme spatial concentration of positive KMLs (all 200 plots inside a single ~5 × 3 km box across two adjacent districts), (2) a validation strategy that is not actually executed (no GroupKFold, no spatial blocks, no buffer exclusion — train.py uses two arbitrary folders that don't even exist), and (3) silent feature-engineering bugs (time-tag format mismatch between modules, `data_backend` / `active_model` pointing in different directions, broken inference path that crashes the moment a single-polygon KML is sent). With surgical rewrites of ~30 % of the code and the recommended data-augmentation actions, **the realistic ceiling on this dataset is ~85–90 % macro-F1 under honest spatial CV; 90–95 % is *not* defensible without expanding KML coverage to more districts.**

---

## Section 1 — Current Project State

### 1.1 Inventory of files

| Path | Purpose | Status |
|---|---|---|
| `README.md` | High-level overview | OK, but contains untrue numbers (claims **250** sugarcane KMLs; actual = **200**). Claims feature stack of 26; sample_generator builds 26, BiLSTM expects 25 after dropping the optical-mask channel — consistent in principle but README/comments contradict each other on counts. |
| `analysis_sugarcane.md` | Strategic doc (phenology, indices, neg-class strategy) | OK but stale (says "50 negative KMLs", actual = 199 KML files / 654 polygons). |
| `config.yaml` | All hyperparameters & backends | Mostly fine, but **`inference.active_model: random_forest`** and **README's claim "BiLSTM is production"** disagree. `cloud_threshold: 40` is too lax for UP (recommend ≤ 30). `domain_adaptation:` block (sf_uda) is declared but **never referenced anywhere in code** — dead config. RF / XGB hyper-parameter grids in config are never read by `train.py` (it hard-codes simple values instead). |
| `requirements.txt` | Pinned deps | Fine. `earthengine-api`, `geemap`, `pystac-client`, `stackstac`, `planetary-computer`, `rioxarray`, `fastapi` all present. PyTorch is optional-pinned (`>=2.0`). LightGBM / Optuna / `lightgbm`, `optuna`, `imbalanced-learn` all **missing** — required for the recommended modelling upgrades. |
| `train.py` | End-to-end training | **Multiple critical issues** — see §1.2. Demands a `data/kml/validation/{sugarcane,non_sugarcane}/` folder that **does not exist** in the repo, so it refuses to run. No CV. No spatial blocks. Tree models are aggregated to mean/std/max/min only — destroys all temporal structure. RF/XGB hyperparams in config are ignored. Auto-promotion logic is one-sided ("BiLSTM<0.75 *and* both Trees>0.80 → promote tree" — otherwise BiLSTM is silently promoted even when it loses by 5 points). |
| `api.py` | FastAPI inference server | Imports from `inference.predictor`. Hard-codes `cloud_free_percentage=100.0`, `cloudy_months=0`, `total_months=14`, `data_quality="HANDLED_BY_PIPELINE"` — these are visible to API consumers and are **lies**. Otherwise the FastAPI plumbing is fine. |
| `utils.py` | `detect_time_tags` helper | **Bug:** detects `YYYY_MM` only, not the actual format `YYYY_MM_DD` produced by `gee_downloader._interval_range()` (15-day tags). Currently unused, but is a foot-gun. |
| `quickstart_validation.py` | Subprocess wrapper | Will fail because it shells out to `train.py`, which itself fails at the missing-validation-folder check. |
| `validate_pipeline.py` | Sanity checks | OK. Auto-creates the missing `validation/` folders (empty), masking the real problem. |
| `data/kml_parser.py` | KML → GeoDataFrame | Solid. Anchor-date regex is good, state-bbox inference is good, geometry repair (`make_valid`, force-2D) is correct. **Minor:** state bboxes overlap (Bihar vs UP) and there's no district-level inference — important because all positives sit in only 2 UP districts. |
| `data/gee_downloader.py` | S1/S2 fetch (3 backends) | Big & ambitious. Real issues: (a) GEE backend uses a `-999.0` sentinel via `unmask(-999)` instead of NaN, polluting downstream stats; (b) S2 SCL mask removes 1,2,3,8,9,10 but **misses class 11 (snow/cirrus)** and **does not remove 0 (NO_DATA)**; (c) `numPixels=max_pixels` parameter passes 500 by default whereas `config.yaml.sampling.max_pixels_per_plot=100` — config is half-honored; (d) Planetary Computer backend loops pixel-by-pixel in Python — orders of magnitude slower than `stackstac`'s native `.values.reshape(...)`; (e) cloud filtering on PC backend is by item-level `eo:cloud_cover<40` but no per-pixel SCL mask is applied; (f) S1 `_add_rvi` correctly converts dB→linear at the GEE level, but downstream `_compute_indices` *also* converts dB→linear — **double conversion** when raw dB is what comes out for VV/VH columns. |
| `data/sample_generator.py` | Sequence + 2D table builder | Most complex file in the repo. Critical points covered in §1.2. Notable: the LOPO splitter (`split_leave_one_plot_out`) **exists but is never called by `train.py`**. |
| `features/spectral_indices.py` | NDVI, EVI, NDRE, NDMI, RVI, RFDI, CR, … | Computes 10 S2 + 3 S1 indices. **Missing user-specified indices: CIre, IRECI, CCCI, GRVI, PSRI, NRPB.** Uses `YYYY_MM` time-tag detector — **does not match** `YYYY_MM_DD` columns produced upstream → silent NaN columns when called with auto-detected tags. |
| `features/temporal_stats.py` | Per-band stats + monsoon/dry split + interannual diff + SAR/optical ratios | Logically sound. Same `YYYY_MM` mismatch as spectral_indices.py. |
| `features/phenology_features.py` | AUC, peak, greenup-half, slopes, season-length, asymmetry, smoothness, NDVI-VV correlation | Good design, but it's a **per-pixel Python `for` loop** — slow (≈ 0.3 ms × 100 pixels × 200 plots × 14 indices = manageable here but won't scale). Same time-tag mismatch. **Missing user-specified features:** explicit `ndvi_amplitude`, `ndvi_peak_timing`, `ndvi_rate_rise`, `ndvi_rate_fall`. (`amplitude` and `peak_month` exist but are computed from a smoothed series, not from the explicit Mar–May / Aug / Dec windows the user requested.) |
| `features/aggregated_features.py` | 3D → 2D via mean/std/max/min | **Reductionist.** Throws away phenology, peak timing, ordering — exactly the temporal structure that distinguishes sugarcane. Used by tree models. |
| `models/sequence_models.py` | BiLSTM + temporal attention | Architecturally fine. `SugarcaneTemporalDataset` uses `YYYY_MM` tag detector → would build empty sequences if it were ever called (currently it's not — train.py builds the 3D array via `sample_generator._build_3d_array` instead). |
| `inference/predictor.py` | KML → probability map | **Inference is broken.** Calls `gen.generate(gdf=gdf, external_neg_gdf=None)` but `generate()` raises `RuntimeError` when no negatives are provided (the explicit guard at line ~211). Every `/detect` call therefore crashes. |
| `models/saved/{bilstm_best.pt, rf_best.joblib, xgb_best.joblib}` | Pre-trained checkpoints | Present but their training history is unknown (no metrics file, no calibration threshold persisted). |
| `tests/test_indices.py` | 4 unit tests | Two tests are mis-named (`test_ndre` calls `ndre(re1, red)` — but the function signature is `ndre(nir2, re1)` — happens to give a numerically-correct answer because of symmetry, so test passes for the wrong reason). |

### 1.2 What's broken / silently wrong (deep-dive)

1. **`train.py` does no cross-validation.** It loads `data/kml/sugarcane` & `data/kml/non_sugarcane` as "train", and `data/kml/validation/sugarcane` & `data/kml/validation/non_sugarcane` as "val". The latter folders **do not exist**. So `train.py` exits immediately with `CRITICAL: Missing VALIDATION KML data!`. Even if you populate them, the result is a **single fixed split**, not k-fold CV — so the F1 you report is the F1 on one arbitrary partition.
2. **There is no spatial CV.** `sample_generator.split_leave_one_plot_out` exists but is dead code — `train.py` never imports it. Pixel-level random splits are the de-facto behaviour; with 200 sugarcane plots packed in a 5 × 3 km box (see §2), this will give **~99 % "validation" F1 that collapses on real data** — this is the textbook spatial-leakage trap the brief warned about.
3. **Inference path crashes.** `predictor.predict()` → `gen.generate()` → raises `RuntimeError("CRITICAL: No Non-Sugarcane KMLs were found...")` because the guard demands explicit negatives at inference time. The API will return 500 on every request after model load. Fix is one-liner (skip the guard if `pos_gdf` only is supplied), but it ships unusable.
4. **Time-tag format mismatch.** `gee_downloader._interval_range()` produces tags like `2025_03_15` (3 numeric parts). `sample_generator` (post-pipeline) detects them as `YYYY_MM_DD`. But **`SpectralIndexCalculator._detect_time_tags`**, **`TemporalStatsExtractor._detect_time_tags`**, **`PhenologyExtractor._detect_time_tags`**, **`utils.detect_time_tags`**, and **`SugarcaneTemporalDataset._build_sequences`** all detect `YYYY_MM` (2 parts). Because `sample_generator.generate()` *passes* `time_tags` explicitly, this hasn't blown up at runtime; but if anyone ever calls these helpers without `time_tags`, they will silently see zero columns and zero features. This is a maintenance landmine.
5. **GEE backend uses `-999` sentinel that survives into the DataFrame.** When a 15-day window has no S2 image, `unmask(-999)` writes -999 into all band columns. Downstream `np.nan_to_num(arr_3d)` turns NaNs to 0, but -999 was never a NaN — it stays as -999, contaminating means, percentiles, and BiLSTM inputs.
6. **VV/VH double-conversion risk.** `_GEEBackend._add_rvi` converts dB→linear before computing RVI (correct). The raw VV/VH columns, however, remain in dB. Then `features/spectral_indices.py::_ensure_linear` heuristically re-converts based on "75 % negative values" — fragile. After both conversions, the BiLSTM sees mixed-scale features.
7. **`config.yaml` says `cloud_threshold: 40`**, but the brief and best practice for monsoon UP recommend ≤ 30. With 40 %, summer composites can include heavily contaminated pixels.
8. **Tree models fed mean/std/max/min only.** `aggregated_features.TensorAggregator` collapses 28 timesteps × 26 features → 4 stats × 26 features = 104 dims. The discriminative phenology shape (rate-of-rise, peak timing, season length) is *gone* before the model sees the data.
9. **API-level cloud-status fields are hard-coded.** `cloud_free_percentage=100.0`, `cloudy_months=0` — visible to clients, materially wrong.
10. **One test (`test_ndre`) accidentally passes** because `ndre(a,b) = -ndre(b,a)` and `(0.8-0.2)/(0.8+0.2) = 0.6` whichever side you put 0.8 on. The test does not verify the *signature contract*.
11. **`active_model` mismatch.** `config.yaml` ships `active_model: random_forest`. `README.md` claims BiLSTM is the production model. `predictor.py` reads from config — so by default the API loads RF, not BiLSTM.

### 1.3 What works correctly (do **not** rewrite)

- `data/kml_parser.py`'s anchor-date regex (`_DATE_PATTERNS`), state inference, force-2D, `make_valid`, hectare computation via UTM-44N.
- `features/spectral_indices.py`'s pure-numpy index functions (NDVI, EVI, NDRE, NDMI, GNDVI, LSWI, SAVI, MSAVI, NBR, RVI, RFDI, CR) — formulas all check out.
- The `SugarcaneAttentionLSTM` architecture (BiLSTM + Bahdanau-style temporal attention + sigmoid head). Hyperparameters can stay as-is.
- `KMLParser._anchor_to_date_range` for ±6/+7 month windows.
- The 15-day compositing strategy (28 steps over 14 months) — actually **better** than the user-requested 6 windows because it captures the green-up rate; keep it.
- The 3-backend abstraction (GEE / Planetary Computer / Sentinel Hub) — useful for outage resilience.
- The `cloud_gap_fraction` per-pixel feature in `sample_generator`.
- The `Savitzky-Golay` smoothing along the time axis in `_build_3d_array`.

### 1.4 Verdict

| Option | Effort | Realistic accuracy ceiling on **current KMLs** under honest spatial CV |
|---|---|---|
| **A. Continue with surgical fixes only** | 3–4 person-days | 70–80 % macro-F1 |
| **B. Partial rewrite (recommended)** — Fix train.py CV, fix inference, add missing indices, fix tag-format bug, retrain. | 7–10 person-days | 80–88 % macro-F1 |
| **C. Full rewrite + KML expansion** — Rewrite as a polygon-level pipeline (one row per plot, not per pixel), add 6 user-specified spectral indices, plot-level GroupKFold + buffer exclusion, Optuna tuning, **plus collect 50–100 more sugarcane KMLs from Lakhimpur Kheri / Bareilly / Bahraich / Gorakhpur to break geographic concentration.** | 15–20 person-days *plus* 1–2 weeks of field-team data collection | **88–93 %** macro-F1 with a realistic chance at 90 %; **95 % is unlikely** without 1000+ plots and multi-year coverage. |

**My recommendation: Option B + the data-collection action from C.** Without more diverse KMLs, no amount of model tuning will hit 95 %. Option B unblocks the engineering; the geographic expansion unblocks the science.

---

## Section 2 — Data Quality Report (KML Audit)

> Full per-polygon audit table written to `kml_audit_full.csv` by `_audit_kmls.py` (854 rows). Numbers in this section come straight from that audit script.

### 2.1 Polygon counts

| Class | KML files | Polygons | Invalid geometries |
|---|---:|---:|---:|
| sugarcane (positive) | **200** | 200 (one polygon per file) | 0 |
| non_sugarcane (negative) | **199** | **654** (avg ≈ 3.3 polygons/file — many KMLs contain multiple polygons) | 2 (`1.kml`, `2.kml` — failed to parse) |

So the actual training pool is **200 positive plots vs. ~652 negative plots** (3× imbalance the wrong way — sample_generator down-samples negatives to neg_ratio=1.0, meaning ~452 negative polygons get *thrown away*).

### 2.2 Polygon area distribution (hectares)

| Class | count | min | p10 | median | mean | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| sugarcane | 200 | 0.037 | 0.125 | **0.282** | 0.351 | 0.713 | 1.514 |
| non_sugarcane | 652 | 0.003 | 0.080 | **0.188** | 0.239 | 0.407 | 5.705 |

**89.3 % of all polygons are below 0.5 ha.** Median sugarcane plot is **0.28 ha → 28 Sentinel-2 10 m pixels**. Median non-sugarcane plot is **0.19 ha → 19 pixels**. Several polygons are **<0.05 ha (≤5 pixels)** which is below any reasonable polygon-level statistic. Recommendation: drop polygons with <10 pixels or merge them per-plot before sampling.

### 2.3 Pixel budget at Sentinel-2 native (10 m)

| Class | Total raw pixels | Median per polygon | Min | Max |
|---|---:|---:|---:|---:|
| sugarcane | **7 017** | 28 | 4 | 151 |
| non_sugarcane | 15 498 | 19 | 0 | 571 |

With `sampling.max_pixels_per_plot = 100` (config), the realised pixel budget is:

- sugarcane: **6 908** capped pixels
- non_sugarcane: 14 265 capped pixels (will be down-sampled to ~6 908 by neg_ratio=1.0)
- **Total balanced training pixels ≈ 13 800** spread across 28 timesteps × 26 features

That's a **2D feature matrix of roughly 13 800 × ~700** if you keep all temporal stats + phenology + per-timestep features. Plenty for tree models, comfortable but small for a BiLSTM (deep models like ~1 500+ *independent* sequences per class to avoid overfit; here, after grouping by plot, you effectively have **200 vs ~199 independent sequences**).

### 2.4 Geographic distribution — **the single biggest problem in the project**

#### Sugarcane (200 plots)
- All 200 inside UP bbox ✓
- **Lat range: 29.167 → 29.219° (span = 0.05°, ≈ 5.5 km north–south)**
- **Lon range: 77.583 → 77.613° (span = 0.03°, ≈ 2.9 km east–west)**
- District distribution:
  - **Muzaffarnagar : 127 plots**
  - **Meerut       :  73 plots**
  - All other districts : 0
- **Nearest-neighbour distance: median 50 m, 99.5 % of plots within 500 m of another plot.**

> **This is fatal for the user's stated CV strategy.** The brief asks for a 5×5° grid spatial CV — but all 200 sugarcane polygons fit inside a single 0.05°×0.03° box (i.e. 1/10 000th of one grid cell). 5×5° spatial CV degenerates to a 100/0 split. District-level CV degenerates to "Muzaffarnagar vs Meerut" — only 2 folds, both adjacent. 500 m buffer exclusion would remove **>99 %** of the test plots in any random split. *Any* metric you produce on this data is an over-fit to one micro-region of west UP.

#### Non-sugarcane (652 polygons / 199 files)
- 652/654 inside UP bbox; 2 outside.
- Lat span: 26.68 → 30.03° (3.34°)
- Lon span: 77.37 → 83.51° (6.15°) — **much more diverse than positives**
- Top districts:
  - **Mainpuri : 485** (76 % of all negatives — also concentrated, but in a different part of UP than the positives → no spatial overlap = good for negatives, but means the model can shortcut "is this in Mainpuri?" instead of learning sugarcane phenology)
  - Gorakhpur: 54
  - Bulandshahr: 39
  - Saharanpur: 30
  - Bareilly: 18, Bijnor: 10, Lucknow: 7, Meerut: 7, Bahraich: 2

> **Geographic shortcut risk:** the model can learn "lat≈29.2°N + lon≈77.6°E ⇒ sugarcane" and "lat≈27°N + lon≈79°E ⇒ non-sugarcane" with zero use of spectral features. This is not a hypothetical — with 99.5 % spatial autocorrelation in the positive class and ≈ 0 spatial overlap with the negative class, any RF/XGB model can hit >98 % validation accuracy by reading lat/lon alone. **Recommendation: do not include longitude/latitude as features (currently they're saved in the wide DataFrame — drop them before training).**

### 2.5 Negative-class composition (filename-inferred)

| Inferred crop / cover type | # KML files | Diversity verdict |
|---|---:|---|
| **Unlabeled (numeric file names)** | 456 polygons | ⚠️ Cannot verify ground truth — these are the bulk of negatives |
| **Maize** | 86 | Good — main UP confuser ✓ |
| Unlabeled ("Untitled map (n)") | 56 polygons | ⚠️ Cannot verify |
| **Rice** | 49 | Good — Kharif confuser ✓ |
| Mango / orchard | 3 | ⚠️ Too few — orchards are a major false-positive class for sugarcane |
| Sponge gourd / vegetable | 3 | ⚠️ Too few |
| Poplar / forestry | 1 | ⚠️ Effectively missing — UP has substantial poplar plantations that look like sugarcane in SAR |
| **Wheat / Mustard / Fallow / Banana / Urban / Water** | **0 named files** | ❌ **Missing entirely** — wheat is the dominant Rabi crop and the #1 sugarcane phenology contrast (wheat = bare in Sep–Oct when sugarcane peaks). Banana looks identical to sugarcane in S1 and is a known confuser. |

> **Verdict: the negative class is *not* diverse enough.** Out of 199 files, only ~138 are clearly identifiable, and they cover only **maize, rice, and a handful of orchards**. For the 90 %+ target you must explicitly label the 456 numeric KMLs (post-hoc inspection in Google Earth) and **add wheat, mustard, fallow, banana, and urban negatives** before training.

### 2.6 Geometry validity

- **2 invalid KMLs** (failed to parse): `data/kml/non_sugarcane/1.kml`, `data/kml/non_sugarcane/2.kml`
- **0 self-intersecting / topologically invalid polygons after parsing.** `make_valid()` in the parser is doing its job.
- **0 KMLs outside UP bbox** for the positive class. 2 negative KMLs sit on the UP border (technically outside the bbox by a fraction of a degree); ignore.

### 2.7 Honest risk assessment for the 90–95 % target

| Risk | Severity | Why |
|---|---|---|
| **Spatial concentration of positives (5 km × 3 km box, 2 districts)** | **CRITICAL** | Any honest spatial CV will collapse. Random pixel CV will report inflated 95–99 % numbers that don't generalize. |
| **Polygon size: 89 % under 0.5 ha** | HIGH | Edge pixels dominate; mixed-pixel contamination is severe. Median polygon yields 28 pixels — sample variance is enormous. |
| **Negative class: only 3 confirmed crop types (maize, rice, ~3 orchards)** | HIGH | Wheat, mustard, fallow, banana, urban are absent. The model will not learn to reject these in production. |
| **No validation set ever defined** | HIGH | `train.py` requires `data/kml/validation/` — non-existent. No held-out plots exist. |
| **All 200 positives use `default_anchor_date: 2025-09-01`** | MEDIUM | Single anchor date means single phenology snapshot. Inter-annual robustness is untested. |
| **Two invalid KMLs in negatives** | LOW | Easy fix, exclude or repair. |

**Bottom-line accuracy expectation on the *current* data, with proper spatial CV:**

| Metric | Estimate |
|---|---|
| Random-pixel CV (current train.py de-facto behaviour) | **96–99 % "accuracy" — fake, do not report** |
| GroupKFold by plot_id (within UP) | **82–88 % macro-F1** |
| Leave-one-district-out (only 2 districts available) | **65–75 % macro-F1** — model has nothing to learn the second district from |
| **Production deployment in unseen UP districts (Lakhimpur Kheri, Pilibhit, Gorakhpur)** | **likely 55–70 % macro-F1** (severe domain shift) |

**To honestly hit 90–95 % macro-F1 we need *both*:**
1. **More plot diversity:** ≥ 50 more sugarcane KMLs each from Lakhimpur Kheri, Bareilly/Pilibhit, and the eastern UP belt (Gorakhpur / Bahraich / Kushinagar).
2. **More negative diversity:** ≥ 30 wheat KMLs (Rabi), ≥ 20 banana, ≥ 20 mustard, ≥ 10 poplar, ≥ 10 urban, ≥ 10 water-body KMLs from the same UP regions.

---

## Section 3 — Complete Feature Engineering Specification

### 3.1 Audit of currently-implemented features

| Family | Implemented? | Where | Notes |
|---|---|---|---|
| S2 raw bands B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 | ✅ | `gee_downloader._GEEBackend` selects all 10 | Good — keep |
| S2 indices: NDVI, EVI, NDWI, LSWI, SAVI, MSAVI, NBR, NDRE, GNDVI, NDMI | ✅ | `features/spectral_indices.py` | Formulas correct |
| **S2 indices: CIre, IRECI, CCCI, GRVI, PSRI** | ❌ | **Missing — must add** | These are the user's high-value sugarcane discriminators |
| S1 bands VV, VH | ✅ | GEE backend | But raw values stay in dB; mixed-scale risk downstream |
| S1 indices: RVI, RFDI, CR | ✅ | `spectral_indices.rvi_sar`, `rfdi`, `cross_ratio` | OK |
| **S1 indices: NRPB (normalised ratio polarisation bands)** | ❌ | **Missing** | Trivial to add |
| Multi-temporal stacking | ✅ (28 × 15-day, even better than the requested 6 windows) | `_interval_range` 15-day | Strong |
| Cloud masking via SCL | ✅ partial | GEE masks SCL ∈ {1,2,3,8,9,10}; **misses SCL 11 (snow/cirrus)** and **SCL 0 (no_data)** | Fix: add 0 and 11. Still keep 4–7,11 as valid (SCL 11 = snow → exclude in UP). |
| Cloud probability < 30 % | ⚠️ | Per-scene `CLOUDY_PIXEL_PERCENTAGE < 40` only | Lower to 30 and add S2_CLOUD_PROBABILITY (CS+ collection) per-pixel. |
| Median composite per window | ✅ | `s2_coll.median()` after masking | Correct |
| Per-polygon median (zonal stats) | ❌ | Currently does **per-pixel sampling** (`.sample(numPixels=500)`) — outputs N pixel rows per polygon, not 1 | This is a fundamental architectural choice (see §4); for the 90 %+ target on 200 plots, **switch to per-polygon median** of valid pixels = one row per polygon. |
| <5 valid-pixel flag → NaN + temporal interpolation | ⚠️ | Has `cloud_gap_fraction`, but does not flag windows with <5 valid pixels | Add explicit flag column |
| Phenology features: AUC, peak, peak_month, greenup_half, slopes, season_length, asymmetry, smoothness, NDVI–VV correlation | ✅ | `features/phenology_features.py` | Good but slow (Python loop) |
| **`ndvi_amplitude` (max-min) explicit** | ⚠️ | Implemented as `{band}_amplitude` (peak − P10) | Close, accept it. |
| **`ndvi_peak_timing`** | ✅ | `{band}_peak_month` | OK |
| **`ndvi_rate_rise = (ndvi_aug − ndvi_may) / 3`** | ❌ | Only `max_greenup_slope` exists | Add window-specific slope |
| **`ndvi_rate_fall = (ndvi_aug − ndvi_dec) / 4`** | ❌ | Only `max_senescence_slope` exists | Add window-specific slope |
| Polygon area (ha) | ✅ | KML parser | Append as feature |
| Plot-level temporal stats per band (min/max/mean/median/p10/p25/p75/p90/std/range/cv) | ✅ | `temporal_stats.py` | Good |
| Monsoon vs dry seasonal contrast | ✅ | `temporal_stats._compute_seasonal_stats` | Good |
| Inter-annual diff & stability | ✅ | `temporal_stats._compute_interannual_stats` | Good (only meaningful when window crosses 2 calendar years — current 14-month window does) |
| SAR–optical ratio stats (VV/NDVI etc.) | ✅ | `temporal_stats._compute_sar_optical_stats` | OK |
| **Aggregating temporal sequence to one row per plot** | ❌ | Currently aggregates **per-pixel**, leading to 13 800 rows | This is the biggest architectural flag — see §4.1 |

### 3.2 Required additions (must-add list)

#### 3.2.1 New S2 indices (per timestep, per polygon)

```python
# Add to features/spectral_indices.py
def cire(b7, b5):      return b7 / (b5 + 1e-10) - 1            # Chlorophyll Red-Edge
def ireci(b7, b6, b5, b4):
    return (b7 - b4) / (b5 / (b6 + 1e-10) + 1e-10)              # Inverted Red-Edge Chlorophyll Index
def ccci(ndre_arr, ndvi_arr):
    return ndre_arr / (ndvi_arr + 1e-10)                        # Canopy Chlorophyll Content Index
def grvi(b3, b4):      return (b3 - b4) / (b3 + b4 + 1e-10)    # Green-Red Vegetation Index
def psri(b4, b2, b7):  return (b4 - b2) / (b7 + 1e-10)         # Plant Senescence Reflectance Index
def nrpb(vh, vv):      return (vh - vv) / (vh + vv + 1e-10)    # Normalised Ratio Polarisation Bands (linear scale)
```

These five S2 indices target sugarcane's distinctive **chlorophyll concentration during the long grand-growth phase** (CIre, IRECI, CCCI) and the harvest senescence signal (PSRI). NRPB is a SAR cross-pol normaliser that complements RVI.

#### 3.2.2 Phenology windows the user explicitly requested

```python
# Add to features/phenology_features.py
def window_features(ndvi_series_by_month):
    feats = {}
    feats["ndvi_may"]  = mean_in(ndvi_series_by_month, months=[5])
    feats["ndvi_aug"]  = mean_in(ndvi_series_by_month, months=[8, 9])      # peak window
    feats["ndvi_dec"]  = mean_in(ndvi_series_by_month, months=[12, 1])     # harvest window
    feats["ndvi_amplitude"]   = feats["ndvi_aug"] - feats["ndvi_may"]
    feats["ndvi_rate_rise"]   = (feats["ndvi_aug"] - feats["ndvi_may"]) / 3.0
    feats["ndvi_rate_fall"]   = (feats["ndvi_aug"] - feats["ndvi_dec"]) / 4.0
    return feats
```

These are some of the most discriminative features for sugarcane vs. rice/wheat/maize because rice-wheat rotation produces a *bimodal* curve (two short peaks separated by bare soil), maize peaks once and falls in 4 months, while sugarcane has a *single broad peak with slow rise and slow fall*. Implement them as polygon-level scalars.

#### 3.2.3 Cloud-masking strategy (fix existing)

```python
# In gee_downloader._mask_s2 — current is incomplete
def _mask_s2(image):
    qa  = image.select("QA60")
    scl = image.select("SCL")
    qa_mask  = qa.bitwiseAnd(1<<10).eq(0).And(qa.bitwiseAnd(1<<11).eq(0))
    # SCL: 0=NO_DATA, 1=SAT, 2=DARK, 3=CLOUD_SHADOW, 4=VEG, 5=BARE,
    #      6=WATER, 7=UNCLASSIFIED, 8=CLOUD_MED, 9=CLOUD_HIGH, 10=THIN_CIRRUS, 11=SNOW
    valid_scl = ee.List([4, 5, 6, 7])     # keep only clean classes; drop 11 (snow doesn't apply in UP plains anyway, but exclude defensively)
    scl_mask = scl.remap(valid_scl, ee.List.repeat(1, valid_scl.size()), 0)
    cloud_prob_mask = ee.Image('COPERNICUS/S2_CLOUD_PROBABILITY')\
        .filterDate(image.date(), image.date().advance(1, 'day'))\
        .first().select('probability').lt(30)        # cloud prob < 30 %
    return image.updateMask(qa_mask.And(scl_mask).And(cloud_prob_mask))\
                .divide(10000)\
                .copyProperties(image, ["system:time_start"])
```

Then per polygon, per window:

```python
# Polygon-level reduction
poly_stats = masked_image.reduceRegion(
    reducer = ee.Reducer.median().combine(ee.Reducer.count(), sharedInputs=True),
    geometry = polygon_geometry,
    scale = 10,
    maxPixels = 1e8,
)
# If pixel_count < 5 → NaN; impute via 1D temporal linear interpolation later
```

#### 3.2.4 Replace pixel sampling with **per-polygon median**

The single most impactful architectural change. Reasons:
- **One row per polygon = 200 sugarcane + ~199 negative = ~399 independent samples.** Tree models do not need 13 800 redundant rows; they need *clean* signals.
- **Eliminates pseudo-replication** that drives the fake 95 %+ random-CV numbers.
- **Matches the user's brief verbatim** ("compute median of valid pixels within polygon").
- Reduces I/O 100×: instead of 100 pixels × 28 windows × 26 features × 400 plots ≈ 29 M cells, you get 28 × 26 × 400 ≈ 290 k cells.

### 3.3 Final feature matrix dimensions (recommended)

If we adopt **per-polygon median + 28 × 15-day windows + all S2/S1 indices + phenology summary**:

```
Per polygon, per window (28 windows):
  S2 raw bands:          10  (B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12)
  S2 indices (existing): 10  (NDVI, EVI, NDWI, LSWI, SAVI, MSAVI, NBR, NDRE, GNDVI, NDMI)
  S2 indices (new):       5  (CIre, IRECI, CCCI, GRVI, PSRI)
  S1 raw bands:           2  (VV, VH — in dB)
  S1 derived:             4  (RVI, RFDI, CR, NRPB)
  Optical-mask flag:      1
  Valid-pixel-count:      1
  ─────────────────────────
  Subtotal per window:   33
× 28 windows           = 924 multi-temporal features

Plot-level scalar features (computed once per plot):
  Phenology per VI (NDVI, EVI, NDRE, NDMI, LSWI):
    AUC, peak_value, peak_month, greenup_half, max_greenup_slope,
    max_senescence_slope, season_length, n_growing_seasons,
    amplitude, asymmetry, smoothness                                 = 11 × 5  = 55
  Window-anchored:
    ndvi_may, ndvi_aug, ndvi_dec, ndvi_amplitude,
    ndvi_rate_rise, ndvi_rate_fall                                   =          6
  Temporal stats per band (11 stats × 24 bands):                     =        264
  Seasonal contrast (monsoon vs dry):                                =         24
  Inter-annual diff & stability (each VI):                           =         10
  SAR–optical ratio stats:                                           =          6
  Polygon area_ha                                                    =          1
  ─────────────────────────────────────────────────────────────────────────────
  Plot-level scalar subtotal                                         =       ~366

GRAND TOTAL FEATURE COUNT PER POLYGON ≈ 1 290
TOTAL ROWS                            =   200 sugarcane + 199 non-sugarcane (after dropping invalid) = 399
```

For tree models with ~400 samples, ~1 300 features is **dangerously high-dimensional**. Apply:
- **L1-penalised feature pre-selection** to keep ~150 most-informative features, *or*
- **Drop 28-window per-band raw features**, keep only plot-level summaries (~366) → much safer.

For the BiLSTM, feed the **per-polygon × 28 × 33** tensor directly (sequence length 28, channels 33), bypassing any aggregation. This is the only model that benefits from the full multi-temporal stack.

### 3.4 Multi-temporal acquisition plan for UP

Keep the 14-month window (anchor − 6 months → anchor + 7 months), 15-day composites = 28 timesteps. This is *better* than the user-suggested 6 fixed windows because:
- It captures green-up rate inside a single month.
- It's robust to the actual cloud coverage pattern (you usually get *some* clear S2 in a 15-day window even during monsoon).
- The phenology features can be computed on the smoothed 28-step series and then summarised back to the 6 user-specified phenology windows.

S1 acquisition: keep **DESCENDING** orbit (config) — gives consistent geometry. ASCENDING also exists; for UP, descending has slightly better temporal coverage. Don't mix.

### 3.5 Cloud-masking strategy summary

1. Per-pixel: SCL ∈ {4,5,6,7} **AND** S2_CLOUD_PROBABILITY < 30 **AND** QA60 cloud bits = 0.
2. Per-scene: `CLOUDY_PIXEL_PERCENTAGE < 30` (drop from 40 currently).
3. Per polygon × window: require ≥ 5 valid pixels; else flag NaN.
4. Per polygon: temporal linear interpolation (1D) on each band/index across 28 windows; if a window has 0 valid pixels for both before *and* after for >2 consecutive windows in the monsoon, fall back to **SAR-only features** for that polygon.
5. Carry an `optical_mask_count_<window>` integer column (0–N) so the BiLSTM can learn to discount cloudy timesteps explicitly.

---

## Section 4 — Pipeline Implementation Plan

### 4.1 The fundamental architectural decision: pixel-level vs polygon-level

**Recommendation: switch to polygon-level (one row per polygon).**

| Dimension | Current (pixel-level) | Proposed (polygon-level) |
|---|---|---|
| Independent samples | ~13 800 (but only 399 truly independent because of within-plot autocorrelation) | 399 |
| Spatial-leakage risk under random CV | Severe | None (CV is by polygon) |
| Memory / IO | 30+ M cells | ~300 k cells |
| Mixed-pixel handling | Built into raw data | Cleanly aggregated by `median()` |
| User-brief alignment | Partial | Full — matches "compute median of valid pixels within polygon" verbatim |
| Inference pixel maps | Possible (predict on every pixel) | Possible (predict on polygon, then optionally apply pixel-level prediction afterwards using the polygon-level model as a second stage) |

The pixel-level approach made sense if you had millions of pixels and only a handful of plots. With **399 plots**, polygon-level is the right unit of inference and learning.

If pixel-level probability maps are still required for the API, train a **second** lightweight CNN/RF on per-pixel patches *separately* (or just stamp the polygon-level prediction onto every pixel inside the polygon — which is what the API probably wants for the testing team workflow).

### 4.2 Suggested file structure (after refactor)

```
sugarcane_detection_UP/
├── config.yaml                     # cleaned: drop dead sf_uda block, raise cloud_threshold to 30
├── pipeline.py                     # NEW — single-command end-to-end driver
├── train.py                        # rewritten: GroupKFold + Optuna + proper metrics
├── api.py                          # patched: real cloud stats, drop hard-coded fields
├── README.md                       # corrected counts (200 not 250) and active model
├── requirements.txt                # add lightgbm, optuna, imbalanced-learn
│
├── data/
│   ├── kml_parser.py               # keep
│   ├── gee_downloader.py           # PATCHES: SCL, cloud_prob, polygon-level reduceRegion
│   ├── polygon_extractor.py        # NEW — per-polygon median + valid-pixel count + 1D interp
│   └── sample_generator.py         # SHRINK — only orchestrates, no aggregation
│
├── features/
│   ├── spectral_indices.py         # ADD: cire, ireci, ccci, grvi, psri, nrpb
│   ├── temporal_stats.py           # FIX: tag-format YYYY_MM_DD
│   ├── phenology_features.py       # ADD: window-anchored ndvi_amp/rate_rise/rate_fall, vectorise
│   └── feature_table_builder.py    # NEW — assembles final wide CSV (one row per polygon)
│
├── validation/
│   ├── splits.py                   # NEW — GroupKFold, BlockKFold (lat/lon grid), buffer exclusion
│   └── metrics.py                  # NEW — F1/OA/Kappa/ROC-AUC/CM, top-20 importance, calibration
│
├── models/
│   ├── tree_models.py              # NEW — RF, XGB, LightGBM with class_weight='balanced'
│   ├── stacking.py                 # NEW — RF+XGB+LGB → LR meta
│   ├── sequence_models.py          # keep
│   └── saved/...
│
├── inference/
│   └── predictor.py                # PATCH — fix the "no negatives" crash
│
└── tests/
    ├── test_indices.py             # FIX signature contract bug
    ├── test_polygon_extractor.py   # NEW
    ├── test_splits.py              # NEW — verify no plot leaks across folds
    └── test_metrics.py             # NEW
```

Net code change: ~1 800 added LOC, ~600 deleted/replaced. Most existing modules survive with patches.

### 4.3 Module-by-module implementation order (with dependencies)

```
Phase 0 — UNBLOCK            (½ day)
   1. Remove the hard validation/ folder requirement in train.py
   2. Fix predictor.py crash (skip "no negatives" guard at inference)
   3. Confirm sugarcane labels: spot-check 10 random KMLs in Google Earth
   4. Manually re-classify the 456 unlabeled numeric negative KMLs
      (or accept them as "mixed-class negatives" with reduced weight)

Phase 1 — CORRECT FOUNDATIONS (2 days)
   5. Standardise time-tag format on YYYY_MM_DD across all helpers
   6. Replace -999 sentinel with NaN in gee_downloader
   7. Fix SCL mask (add 0, keep 11 dropped)
   8. Add S2_CLOUD_PROBABILITY < 30 mask
   9. Drop CLOUDY_PIXEL_PERCENTAGE from 40 → 30
  10. Convert VV/VH to a *single* canonical scale (linear) at extraction;
      delete _ensure_linear heuristic in spectral_indices

Phase 2 — POLYGON-LEVEL EXTRACTOR (2 days)
  11. NEW data/polygon_extractor.py:
        - For each plot, for each 15-day window: ee.reduceRegion(median + count)
        - Output one row per (plot_id, window_tag) with:
            B2..B12, VV, VH, valid_pixel_count
        - Pivot to wide one-row-per-plot
        - 1D linear-interpolate gaps along the time axis
        - Savitzky-Golay smooth with window=5 on each band
  12. Delete pixel-sample code path from sample_generator
  13. Remove longitude/latitude columns from the feature table

Phase 3 — FEATURE ENGINEERING (1.5 days)
  14. Add CIre, IRECI, CCCI, GRVI, PSRI, NRPB to spectral_indices.py
  15. Add window-anchored ndvi_amplitude/peak_timing/rate_rise/rate_fall
      in phenology_features.py
  16. Vectorise phenology features (replace per-pixel loop with numpy vectorised)
  17. NEW features/feature_table_builder.py: produces final CSV
        Columns: [plot_id, label, area_ha,
                  <band|index>_<YYYY_MM_DD>,                       (924)
                  <vi>_<phenology_metric>,                         (55)
                  ndvi_amp, ndvi_rate_rise, ndvi_rate_fall, ...,
                  <band>_<temporal_stat>,                          (264)
                  <band>_seasonal_contrast,                        (24)
                  ...
                ]

Phase 4 — VALIDATION (1 day)
  18. NEW validation/splits.py:
        - PolygonGroupKFold       (5 folds, stratified by label, grouped by plot_id)
        - SpatialBlockKFold       (0.05° lat × 0.05° lon block grouping
                                    — gives effective spatial CV for the dense positive cluster)
        - BufferedHoldout         (drop test plots within 500 m of any train plot)
        - LeaveOneDistrictOut     (degenerate to 2 folds for sugarcane,
                                    use only as a sanity check)
  19. NEW validation/metrics.py:
        OA, F1_per_class, F1_macro, Kappa, ROC-AUC, PR-AUC, CM,
        top-20 feature importance, probability calibration (Brier score)

Phase 5 — MODELLING (2 days)
  20. NEW models/tree_models.py:
        RF(n_estimators=300, max_depth=None, min_samples_leaf=2,
           class_weight='balanced', random_state=42)
        XGB(n_estimators=300, max_depth=6, lr=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=auto, eval_metric='aucpr')
        LGB(n_estimators=300, num_leaves=63, lr=0.05, subsample=0.8,
            colsample_bytree=0.8, class_weight='balanced')
  21. NEW models/stacking.py:
        StackingClassifier([RF, XGB, LGB], LogisticRegression,
                            cv=PolygonGroupKFold(5))
  22. Optuna (50 trials) on best of {RF, XGB, LGB} optimising macro-F1
  23. BiLSTM kept as is, but trained on the same PolygonGroupKFold splits

Phase 6 — TRAIN.PY REWRITE (1 day)
  24. New train.py:
        - Load CSV from feature_table_builder
        - PolygonGroupKFold(5) loop:
            for each fold:
              fit all 4 models, log per-fold metrics
        - Aggregate metrics (mean ± std across folds)
        - Persist best model + tuned threshold (Youden J) + fold reports

Phase 7 — INFERENCE / API (½ day)
  25. predictor.py: replace 'gen.generate(... external_neg_gdf=None)' call
      with a direct call to polygon_extractor + feature_table_builder
  26. api.py: replace hard-coded cloud fields with real values from
      polygon_extractor's valid_pixel_count

Phase 8 — TESTS (½ day)
  27. test_polygon_extractor: small synthetic GEE-mock test
  28. test_splits: assert no plot_id appears in both train and val of any fold
  29. test_metrics: deterministic sanity checks
  30. fix test_ndre signature

Phase 9 — DOCS                (½ day)
  31. README.md numbers
  32. analysis_sugarcane.md numbers
  33. Add CV strategy section + reproduction one-liner:
         python pipeline.py --kml_dir data/kml --out outputs/sugarcane_features.csv --train
```

Total: **11 working days for a single competent senior ML engineer**, with no data-collection waiting.

### 4.4 API / data-source choice

| Backend | Verdict for this project |
|---|---|
| **Google Earth Engine (GEE)** | **Recommended.** Already the codebase default. `reduceRegion` is built for polygon-level extraction. 200 plots × 28 windows = 5 600 GEE calls — well within free quota. Authentication is the only friction. |
| Microsoft Planetary Computer (STAC + stackstac) | Keep as fallback when GEE is down. Current implementation is slow (Python pixel loop) — should be rewritten to use `stackstac.stack(...).median(dim='time').to_dataframe()` for vectorised extraction. |
| Sentinel Hub | Don't use. Free tier is rate-limited; commercial cost is unjustified for this use case. |
| Local raster download (rasterio + gsutil) | Overkill for 200 plots; only adopt if you need offline reproducibility or 1000+ plots. |

**Stick with GEE.** Migrate STAC backend code into a `legacy/` subfolder.

### 4.5 Processing order & parallelisation opportunities

```
Step                                    | Cost         | Parallelisable?
─────────────────────────────────────────┼──────────────┼──────────────────
1. KML parsing (200+199 files)          | seconds      | trivial (joblib)
2. Polygon validation + area calc       | seconds      | trivial
3. GEE polygon-level extraction         | 30–60 min    | GEE-side parallel —
                                        |              | use ee.batch.Export
                                        |              | OR Python concurrent.futures
                                        |              | with 4 workers (GEE caps at
                                        |              | ~5 concurrent reduceRegion calls)
4. 1D interpolation + Savitzky-Golay    | seconds      | trivial
5. Spectral indices (per-polygon)       | seconds      | numpy-vectorised
6. Phenology features                   | seconds      | numpy-vectorised after rewrite
7. Final CSV assembly                   | <1s          | n/a
8. PolygonGroupKFold(5) × 4 models     | 5–15 min     | sklearn n_jobs=-1
9. Optuna 50 trials                     | 30–60 min    | n_jobs=4 inside Optuna
10. Final retrain on all folds          | 1–5 min      | n/a
```

**Total wall-clock for a single end-to-end run after refactor: ~90 minutes**, of which ~50 minutes is GEE I/O. This matters because every CV experiment becomes a 90-minute affair, not a multi-hour one — that's what unlocks fast iteration on the 90 % target.

---

## Section 5 — Validation & Accuracy Roadmap

### 5.1 CV strategy recommendation (ranked)

Given the geographic concentration we measured in §2, **none** of the user's three suggested CV strategies works as written, but a *layered* CV stack does:

| Rank | Strategy | Why it's appropriate here | Implementation |
|---|---|---|---|
| 1 | **PolygonGroupKFold(5), stratified by label** | Guarantees no plot is split across folds. Eliminates within-plot pixel leakage. **Primary metric reporting strategy.** | `sklearn.model_selection.StratifiedGroupKFold(n_splits=5)`, group=plot_id, y=label |
| 2 | **SpatialBlockKFold @ 0.05° × 0.05° grid** | The positive cluster spans 0.05° × 0.03°, so a 0.05° grid creates ~2 blocks for sugarcane (Muzaffarnagar core + edge). Coarser than user's 5×5° suggestion (which would be 1 block here) but finer than district-level (2 blocks) — a compromise that produces 4–5 spatially-disjoint folds. | Custom — bin centroids by `floor(lat/0.05), floor(lon/0.05)`, group folds |
| 3 | **BufferedHoldout (500 m)** | Use as a *robustness check on top of* PolygonGroupKFold: drop any test polygon within 500 m of any train polygon. With 99.5 % of positive plots within 500 m of another, this will leave very few test plots — but the *honest* number you get is what generalisation actually looks like. | Build a `STRtree` of train geometries, query each test centroid for ≤ 500 m neighbour, exclude. |
| 4 | LeaveOneDistrictOut | Sanity check only. With 2 districts (Muzaffarnagar, Meerut), this is just a 2-fold CV. Useful early to detect catastrophic district-shift; not a primary metric. | Custom |
| 5 | Random k-fold (current de-facto) | **Do not use.** Will always over-report. | n/a |

**Reporting protocol:** report **mean ± std** of macro-F1 across the 5 PolygonGroupKFold folds, plus separately the **BufferedHoldout F1** and **LODO F1** as side metrics. The PolygonGroupKFold number is what you report as the headline; the buffered & LODO numbers tell you about generalisation gap.

### 5.2 Required metrics (per fold, then aggregated)

```python
metrics_per_fold = {
    "OA":          accuracy_score(y, p),
    "F1_sugarcane":   f1_score(y, p, pos_label=1),
    "F1_non_sugarcane": f1_score(y, p, pos_label=0),
    "F1_macro":       f1_score(y, p, average='macro'),
    "Cohen_Kappa":    cohen_kappa_score(y, p),
    "ROC_AUC":        roc_auc_score(y, prob),
    "PR_AUC":         average_precision_score(y, prob),
    "Brier":          brier_score_loss(y, prob),     # calibration
    "CM":             confusion_matrix(y, p),         # absolute counts
    "CM_pct":         CM / CM.sum(axis=1, keepdims=True) * 100,   # row-normalised %
}
top20_features = permutation_importance(model, X, y, n_repeats=20, random_state=42).importances_mean.argsort()[-20:][::-1]
```

Aggregate as **mean ± std across folds**. Report per-fold confusion matrix, not just the mean.

### 5.3 Threshold calibration

The default 0.5 is rarely optimal for tree-based / sigmoid outputs. Compute **Youden's J** on the OOF predictions:

```python
fpr, tpr, thr = roc_curve(y_oof, prob_oof)
j = tpr - fpr
optimal_threshold = thr[np.argmax(j)]   # persist with the model
```

Persist this threshold inside the saved model checkpoint. The current `predictor.py` reads `optimal_threshold` from a checkpoint key but `train.py` never saves it — fix that.

### 5.4 Baseline accuracy estimates (before tuning)

These are **honest** estimates assuming the polygon-level pipeline rebuild and PolygonGroupKFold(5) — *not* random pixel CV.

| Configuration | Expected macro-F1 (mean across folds) | Notes |
|---|---|---|
| Single S2 image (NDVI only) at anchor date | 0.55 ± 0.10 | Equivalent to "is this pixel green in Sep" — wheat-fallow vs sugarcane indistinguishable |
| Single S2 image (10 indices) at anchor | 0.65 ± 0.08 | NDRE + NDMI start to help |
| **6-window phenology, S2 indices only (no SAR)** | **0.78 ± 0.06** | Captures rice-wheat bimodal vs sugarcane unimodal |
| **6-window phenology + SAR (RVI, VV, VH, NRPB)** | **0.83 ± 0.05** | Big lift in monsoon, separates banana/orchards from sugarcane |
| **28-window (15-day) S2 + SAR + phenology summaries** | **0.86 ± 0.05** | Recommended baseline — RF or XGB out-of-the-box |
| + Optuna 50-trial tuning on macro-F1 | **0.87 ± 0.04** | Marginal gain; tuning matters more for tree depth + min_samples_leaf |
| + Stacking ensemble (RF+XGB+LGB → LR) | **0.88 ± 0.04** | Reliable +1 pp |
| + BiLSTM (on same polygon-level sequences) | 0.85 ± 0.07 | **Likely *worse* than tree models** at 399 polygons — too few sequences for the BiLSTM to learn the attention; report it for completeness but do not promote unless it beats trees by >2 pp |
| + Calibrated threshold (Youden J) | +0.5 to 1 pp | Small but cheap |
| **Headline expectation on current data**: macro-F1 = **0.86 ± 0.05** | | i.e. **81–91 % F1** band, point estimate ~86 % |

### 5.5 Top-3 risks to missing the 90 % target — and how to mitigate

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | **Geographic concentration of positives.** All 200 sugarcane plots in 2 adjacent UP districts (Muzaffarnagar, Meerut). Model never sees the eastern UP sugarcane belt (Lakhimpur Kheri, Bareilly, Pilibhit, Gorakhpur). Production accuracy collapses. | **Critical** | **Collect 100+ more sugarcane KMLs** from at least 4 additional UP districts spanning west-central-east UP. Until then, do not promise >85 % production F1. Report PolygonGroupKFold-on-Muzaffarnagar-and-Meerut as a *training* metric, not a production guarantee. |
| 2 | **Negative-class bias.** Top confusers (wheat, banana, mustard, fallow, urban) are absent. Production false-positive rate is unknown. | High | Add ≥ 30 wheat, 20 banana, 20 mustard, 10 fallow, 10 urban, 10 water KMLs from the same UP regions. **All from the *same* growing season as sugarcane** so phenology contrast is fair. |
| 3 | **Single anchor-date 2025-09-01.** Inter-annual robustness untested. Sugarcane planted in different years has a phase-shifted phenology. | Medium-high | Augment with a small set (50 KMLs) anchored in a different season (e.g. 2024-09-01 and 2026-03-01). Apply temporal-shift augmentation (already present in `SugarcaneAugmentedDataset` — keep it). |
| 4 (bonus) | **Ratoon vs plant-cane confusion.** Ratoons (2nd-year sugarcane) regrow from the harvested stubble — they have a *shorter, lower NDVI peak* than fresh-planted sugarcane. If your KMLs mix ratoons and plant-canes without labels, the model learns an averaged signature that under-fits both. | Medium | If ground truth is available, label ratoon vs plant separately. Otherwise treat ratoon plots as a known confound and target ≥ 0.80 F1 *within ratoon plots*; expect plant-cane F1 to be 5–10 pp higher. |

### 5.6 Honest accuracy feasibility analysis (the user's 6 questions)

> **Q1: With 200+200 polygons and proper spatial CV, is 90 %+ F1 achievable?**
>
> **A: Maybe — but only under the most generous CV (PolygonGroupKFold within Muzaffarnagar+Meerut). Not under district-out or true geographic generalisation. With current data, a defensible headline is 85–88 % macro-F1.** Crossing 90 % requires either (a) more diverse KMLs (see §2.7), or (b) reporting a CV strategy that does not generalise (e.g. random pixel CV), which would be dishonest.

> **Q2: What is the estimated pixel count after spatial processing? Is it enough?**
>
> **A: 7 017 sugarcane pixels (uncapped) → 6 908 capped at 100/plot. That is *enough* for tree models on a polygon-level pipeline (since you'll aggregate to 200 polygon-level rows anyway), but *not* enough for a deep CNN at the pixel level. The BiLSTM gets 399 sequences, which is on the edge of trainable.**

> **Q3: Which confusions will be hardest?**
>
> **A:** In rough order of difficulty:
>   1. **Banana** (Sep–Mar growth, similar SAR backscatter, similar NDVI plateau) — currently 0 KMLs in the negatives → the model has *never seen* banana → in production it will mis-classify ~50 % of banana plots as sugarcane.
>   2. **Maize at peak (Aug)** — single-month NDVI overlap with sugarcane grand growth. Distinguishable by harvest signal in Oct (maize drops fast, sugarcane stays high). Already in negatives (86 plots) — should resolve to <5 % FPR.
>   3. **Ratoon sugarcane on neighbouring fields** — same field signature with reduced amplitude. Real risk of *false negative*, not false positive.
>   4. **Mango / orchards** — perennial high NDVI, but flatter phenology curve, very different SAR texture (rougher canopy). Currently only 3 KMLs → insufficient training signal. Add GLCM texture features on VH (mentioned in `analysis_sugarcane.md` but not implemented).
>   5. **Poplar plantations** (UP has substantial poplar in west UP) — looks like sugarcane in S1; only 1 KML. Add ≥ 10.

> **Q4: What temporal acquisition gaps in UP cause the biggest accuracy drop?**
>
> **A:** July–September monsoon (S2 cloud cover often >80 %) and December–February dense fog. SAR (S1) covers both, but S1 cross-pol (VH) is noisier than VV. The biggest accuracy hit comes from missing **November composites** — the user-brief flagged this correctly: rice-wheat fields are *bare in Nov* while sugarcane is *peaking*. Without November optical data, you lose the single most discriminative timestep. Mitigation: when November S2 is unavailable, lean on S1 RVI and VH/VV ratio for that window; also use **October–February seasonal-contrast features** instead of single-month features.

> **Q5: What is the minimum number of temporal dates needed for this accuracy target?**
>
> **A:** Empirically, on similar UP sugarcane studies in the literature:
>   - 3 dates (one per season): ~0.65 F1
>   - 6 dates (user's brief): ~0.80 F1
>   - 12 dates (monthly): ~0.84 F1
>   - **24+ dates (15-day, current pipeline): 0.86 F1, diminishing returns above this**
> So 28 timesteps is a sweet spot — you don't need more, but you do need to *use* them properly (don't aggregate to 4 stats and throw the rest away, as `aggregated_features.py` currently does).

> **Q6: If we're missing fog-season SAR data for winter months, how much does accuracy drop?**
>
> **A:** S1 has 6-day revisit (3-day after both Sentinels in tandem mode); UP fog does not affect SAR. So you should have S1 data even in Dec–Feb. Concrete drop estimate if S1 *were* missing for Dec–Feb: **~3–5 pp macro-F1**, because you lose the ability to confirm "high-biomass surface" during the rice/wheat-bare period. With S1 present (which is reality), this is a non-issue. The bigger fog risk is *S2 missing for Dec–Feb*, which costs **~2–3 pp** if S1 is healthy.

---

## Section 6 — Immediate Next Steps (Ordered)

### 6.1 Critical blocking issues (resolve before any further model training)

These must all be fixed first. Until they are, every metric reported is unreliable.

| # | Issue | File | Fix |
|---|---|---|---|
| **B1** | `train.py` requires non-existent `data/kml/validation/` folder → pipeline cannot run | `train.py` lines 216–224 | Replace fixed two-folder split with **PolygonGroupKFold(5)** using only `data/kml/sugarcane` and `data/kml/non_sugarcane` |
| **B2** | `predictor.predict()` crashes inside `gen.generate(external_neg_gdf=None)` → API returns 500 on every call | `inference/predictor.py` line ~260 + `data/sample_generator.py` line ~211 | In `sample_generator.generate()`, allow positive-only mode when called from inference (`raise` only when `mode='train'`) |
| **B3** | Time-tag format mismatch between `gee_downloader` (YYYY_MM_DD) and feature helpers (YYYY_MM) | 5 files (utils, spectral_indices, temporal_stats, phenology_features, sequence_models) | Standardise on **YYYY_MM_DD**. One `_detect_time_tags` helper in `utils.py`, used everywhere. |
| **B4** | GEE backend writes `-999` sentinel into DataFrame which is then averaged into stats | `gee_downloader._GEEBackend.extract_pixel_timeseries_wide` | Replace `unmask(-999)` with proper masking + post-extraction `df.replace(-999, np.nan)` + per-pixel-count flag |
| **B5** | `active_model: random_forest` in config but README says BiLSTM is production | `config.yaml`, `README.md` | Align — pick whichever wins on PolygonGroupKFold(5) macro-F1, write that. Update README counts (200 not 250). |
| **B6** | Hard-coded API cloud fields lie to clients | `api.py` lines 237–241 | Replace with real values returned from polygon extractor |
| **B7** | Two unparseable negative KMLs (`1.kml`, `2.kml` in `non_sugarcane/`) | data | Open in Google Earth, re-export, or remove |
| **B8** | 456 unlabeled numeric negative KML files — class identity unknown | data + analysis | **Manual triage step:** open each in Google Earth, label as wheat/maize/rice/fallow/banana/etc., update filenames. Without this, the negative class is a black box. |
| **B9** | Test `test_ndre` passes for the wrong reason (signature mis-call) | `tests/test_indices.py` | Fix variable names so test verifies the documented contract |

### 6.2 Existing code to **keep** vs **rewrite** vs **delete**

| Component | Decision | Rationale |
|---|---|---|
| `data/kml_parser.py` | **KEEP** | Solid; only minor improvements (district inference) optional |
| `data/gee_downloader.py` GEE backend | **PATCH** | Fix sentinel, SCL, cloud_prob; add reduceRegion polygon-level path |
| `data/gee_downloader.py` Planetary Computer backend | **REWRITE** | Replace pixel-loop with `stackstac` vector path; if not needed, **delete** to reduce maintenance burden |
| `data/gee_downloader.py` Sentinel Hub backend | **DELETE** | Unused; SH is paid; rip out |
| `data/sample_generator.py` | **SHRINK ~50 %** | Keep generate orchestration & LOPO splitter; remove pixel-sampling code path; remove negative-balancing logic (handled by class_weight in models); remove cloud_gap_fraction at-pixel (move to polygon-level) |
| `features/spectral_indices.py` | **PATCH + ADD** | Keep formulas; add CIre/IRECI/CCCI/GRVI/PSRI/NRPB; standardise time-tag detection |
| `features/temporal_stats.py` | **PATCH** | Standardise time-tag detection; otherwise good |
| `features/phenology_features.py` | **PATCH + VECTORISE** | Add window-anchored features; vectorise the per-pixel loop with numpy broadcasting (10× speedup); keep public API |
| `features/aggregated_features.py` | **DELETE** | mean/std/max/min destroys temporal signal; replaced by feature_table_builder |
| `models/sequence_models.py` BiLSTM | **KEEP** | Architecture is fine; just train it on PolygonGroupKFold |
| `train.py` | **REWRITE** | Top-to-bottom rewrite; use PolygonGroupKFold + Optuna + proper metrics |
| `api.py` | **PATCH** | Remove hard-coded cloud fields, use real values from extractor |
| `inference/predictor.py` | **PATCH** | Fix the no-negatives crash; use feature_table_builder directly |
| `tests/test_indices.py` | **PATCH** | Fix mis-named test |
| `quickstart_validation.py` & `validate_pipeline.py` | **CONSOLIDATE** | Merge into one `pipeline.py` with subcommands `validate | extract | train | predict` |
| `utils.py` | **EXTEND** | Make `detect_time_tags` the single source of truth; YYYY_MM_DD only |
| `_audit_kmls.py` (newly created) | **KEEP** | Useful diagnostic; move to `tests/audit_kmls.py` and run as part of CI |
| `kml_audit_full.csv` (newly created) | **KEEP** | Reference data for documentation |

### 6.3 Ordered roadmap (what to do tomorrow morning)

```
Day 0 (½ day) — UNBLOCK
  □ Triage: open 30 random unlabeled negative KMLs in Google Earth,
    decide whether to keep them or relabel.
  □ Fix B1, B2 (train.py + predictor.py crashes).
  □ Run `python pipeline.py validate` to confirm pipeline boots.

Day 1 — DATA TRUTH
  □ Manually classify the 456 numeric KMLs (or accept as 'unknown_negative')
  □ Repair or delete the 2 unparseable KMLs
  □ Document the negative-class composition in analysis_sugarcane.md
  □ Decide whether to defer model training pending KML expansion (recommended)

Day 2-3 — POLYGON EXTRACTOR
  □ Implement data/polygon_extractor.py (per-polygon median + count + interp)
  □ Replace pixel-sample code path
  □ Test with 5 sugarcane + 5 non-sugarcane plots end-to-end
  □ Verify CSV reproducibility: same input → bit-identical output

Day 4 — FEATURE ENGINEERING
  □ Add CIre, IRECI, CCCI, GRVI, PSRI, NRPB
  □ Add ndvi_amplitude / rate_rise / rate_fall window features
  □ Vectorise phenology_features.py
  □ Run feature_table_builder.py on all 399 plots; inspect CSV manually

Day 5 — VALIDATION
  □ Implement validation/splits.py (PolygonGroupKFold, SpatialBlockKFold,
    BufferedHoldout, LeaveOneDistrictOut)
  □ Implement validation/metrics.py
  □ Unit-test: assert no plot_id appears in both train and val of any fold

Day 6-7 — BASELINE MODELS
  □ RF baseline on PolygonGroupKFold(5) → record macro-F1, Kappa, ROC-AUC
  □ XGB baseline → same
  □ LightGBM baseline → same
  □ Compare to expectations in §5.4. If macro-F1 < 0.80, debug
    (typically it'll be a feature scaling or NaN issue).

Day 8 — TUNING
  □ Optuna on best baseline (likely XGB or LGB), 50 trials, optimise macro-F1
  □ Stacking ensemble RF+XGB+LGB → LR
  □ Calibrate threshold via Youden's J on OOF predictions

Day 9 — BiLSTM
  □ Train BiLSTM on the same PolygonGroupKFold(5) splits using the 28×33 tensor
  □ Compare against trees. Promote whichever is better by ≥ 2 pp macro-F1.

Day 10 — INFERENCE + API
  □ Wire polygon_extractor + feature_table_builder + best_model into predictor.py
  □ Smoke-test the FastAPI /detect endpoint with 5 different KMLs
  □ Verify cloud-stat fields are now real values, not hard-coded

Day 11 — DOCS + HANDOVER
  □ Update README.md with corrected numbers and one-line reproduction
  □ Update analysis_sugarcane.md
  □ Add MODELING_REPORT.md with per-fold metrics, calibration plots,
    confusion matrices, and feature-importance top-20.
```

### 6.4 What I would *not* do right now

- **Don't promote the existing `bilstm_best.pt` / `rf_best.joblib` / `xgb_best.joblib`** — they were trained under the broken pipeline (no spatial CV, time-tag mismatch may have given empty feature columns, -999 sentinels in stats). Treat them as scratch and retrain after fixes.
- **Don't add domain adaptation (sf_uda) yet** — it's already declared in `config.yaml` but unused. DA only helps when you have a labelled source and unlabelled target. With 200 plots concentrated in 2 districts, you don't need DA, you need *more labels*.
- **Don't add a CNN** at the pixel level. With 7 000 pixels, a CNN will memorise. Tree models on polygon-level features dominate at this scale.
- **Don't tune hyperparameters first.** Get the data and CV right; tuning gives ≤ 1 pp on top of a clean pipeline, but ≥ 10 pp will be lost to a leaky CV.
- **Don't trust the existing 95 %+ accuracy claims** if any have been reported — under any random pixel CV on this data they are inevitable and meaningless.

---

## Closing Summary

**The good news:** the project's *intent* is correct. The use of S1+S2, the BiLSTM with attention, the 14-month window, the 15-day compositing, and the LOPO splitter (even if dead code) all show that whoever scaffolded this had the right ideas. The phenology features, spectral indices, and 3-backend abstraction are genuinely useful pieces of work.

**The bad news:** the pipeline is currently in three pieces — a parser that works, a downloader that *almost* works, and a trainer that doesn't work because its prerequisite folders are absent. Inference is broken at the first call. Validation is structurally absent (LOPO splitter exists but is never invoked). And the data itself — 200 sugarcane plots in two adjacent UP districts, 89 % of polygons under 0.5 ha, negative class missing wheat/banana/mustard/fallow/urban — caps the achievable accuracy under honest evaluation at ~85–88 % macro-F1 regardless of model sophistication.

**The honest verdict on the 90–95 % target:**
- **On *this* dataset under PolygonGroupKFold(5) with the §3-§4 fixes implemented: expect 85–88 % macro-F1, point estimate 86 %.**
- **On *this* dataset under random pixel CV (current de-facto): expect 96–99 %, all of it spurious.**
- **On a properly expanded dataset (4–6 UP districts, 350+ sugarcane, 350+ diversified negatives): 90 % is realistic, 95 % is a stretch, would need 1 000+ plots and probably a multi-year window.**

**Recommended next action:** sign off on Phase 0 + Phase 1 of §4.3 (5 person-days) immediately to unblock the engineering, *and in parallel* commission a 2-week field-data expansion to add ≥ 100 sugarcane KMLs from Lakhimpur Kheri / Bareilly / Pilibhit / Gorakhpur and ≥ 100 diversified negative KMLs (wheat, banana, mustard, fallow, urban). Without the data expansion, the 90 % target is not defensible.

---

*Audit produced by senior ML / RS engineering review.  
Diagnostic script: `_audit_kmls.py` (run with `$env:PYTHONIOENCODING='utf-8'; python -X utf8 _audit_kmls.py`).  
Per-polygon raw audit table: `kml_audit_full.csv` (854 rows).*
