# Sugarcane Detection (Uttar Pradesh) - Analysis & Architecture

> **Target Crop:** Sugarcane
> **Region:** Uttar Pradesh (UP), India
> **Data:** Sentinel-1 (SAR) & Sentinel-2 (Optical)

---

## 1. Phenology & Region Context (Uttar Pradesh)

Sugarcane is a long-duration, high-biomass crop. In Uttar Pradesh, it is the dominant cash crop. Unlike most annual crops that complete their cycle in 3-5 months, sugarcane remains in the field for 10-18 months.

### Crop Calendar (UP)
- **Spring Planting:** February - March (Harvest: Nov - April next year; ~12-14 months)
- **Autumn Planting:** October - November (Harvest: Jan - April year after next; ~16-18 months)
- **Ratoon Crop:** Sprouting from the previous harvest (Harvest: Nov - March; ~11 months)

### Critical Phenological Stages
1. **Germination / Establishment (0-60 days):** Low NDVI (<0.3). Soil background dominates.
2. **Tillering & Canopy Closure (60-150 days):** Rapid NDVI increase. Usually corresponds to pre-monsoon (April-June).
3. **Grand Growth (150-270 days):** Maximum biomass, peak NDVI (0.7 - 0.9). Corresponds to Monsoon/Post-monsoon (July-October). High volume scattering in SAR.
4. **Ripening & Maturation (270-360 days):** Slight senescence, NDVI dips slightly (0.6 - 0.7). Harvest begins.

### Negative Crop Separation Strategy (The Confusion Matrix)
The primary challenge is distinguishing sugarcane from other crops grown in the same region, especially the dominant Kharif-Rabi rotation (e.g., Rice-Wheat).

| Confusing Crop | Phenology / Distinguishing Feature | Separation Logic |
|:---|:---|:---|
| **Rice (Kharif)** | Planted June/July, Harvested Oct/Nov. | Rice drops to bare soil (NDVI <0.2) in Nov/Dec. Sugarcane maintains high NDVI (0.7+) during this period. |
| **Wheat / Mustard (Rabi)** | Planted Nov/Dec, Harvested March/April. | Wheat has bare soil in Oct/Nov. Sugarcane has peak biomass in Oct/Nov. |
| **Maize (Kharif/Zaid)** | 3-4 month short cycle. | Senesces rapidly. Sugarcane remains green. |
| **Orchards (Mango/Guava)** | Permanent tree cover, high NDVI year-round. | SAR texture (GLCM) and VH/VV ratios. Trees have rougher canopies compared to dense, uniform sugarcane. |

---

## 2. Spectral Signature & Features

### Sentinel-2 (Optical) Recommendations
1. **NDVI (Normalized Difference Vegetation Index):** Core metric. Sugarcane's unique signature is the *duration* of high NDVI (>0.6 for 6+ months continuously).
2. **NDRE (Normalized Difference Red Edge):** `(B8A - B05) / (B8A + B05)`. Sugarcane has high chlorophyll content during its long grand growth phase. NDRE is less prone to saturation than NDVI for high-biomass crops.
3. **LSWI (Land Surface Water Index):** `(B8 - B11) / (B8 + B11)`. Sugarcane is highly irrigated in UP. LSWI helps track crop moisture and stress.
4. **GNDVI (Green NDVI):** `(B8 - B3) / (B8 + B3)`. Sensitive to chlorophyll concentration.

### Sentinel-1 (SAR) Recommendations
1. **VH Backscatter:** High sensitivity to volume scattering. Sugarcane canopy structure causes high multiple scattering, leading to high VH values during grand growth.
2. **VV Backscatter:** Tracks soil moisture early on, then gets attenuated by the dense canopy.
3. **RVI (Radar Vegetation Index):** `4 * VH / (VV + VH)`. Excellent proxy for biomass, especially during the cloudy monsoon season when optical data is missing.
4. **CR (Cross-Polarization Ratio - VH/VV):** Tracks canopy structure evolution.
5. **GLCM Texture (Optional):** To separate sugarcane from permanent orchards, texture metrics on VH can be highly effective.

---

## 3. Data Pipeline & Multi-Temporal Logic

### Temporal Window
- **Requirement:** A minimum 12-month window is required to capture the difference between a sugarcane crop and a Rice-Wheat rotation. 
- **Window:** Anchor Date minus 6 months to Anchor Date plus 6 months.
- **Handling Cloud Cover:** UP has heavy clouds from July to September. 
  - *Optical:* Apply Savitzky-Golay filtering or linear interpolation across the time dimension to fill gaps.
  - *SAR:* SAR is cloud-penetrating. During monsoon gaps, the model must learn to rely heavily on S1 VH/VV features.

### Negative Set Rationale
To train a robust model for UP, the 50 negative KMLs must specifically target the local alternatives:
- 15x Rice-Wheat rotation plots
- 10x Maize plots
- 10x Mustard/Potato (Rabi)
- 5x Mango/Orchard plots
- 10x Fodder/Sorghum (Chari) - *Highly confusing as it resembles sugarcane grass, but has a shorter cycle.*

---

## 4. Model Architecture & Evaluation

### Shift from Tabular to Sequence Modeling
The previous legacy model aggregated 7 months of data into static statistics (min, max, mean). Because sugarcane's identity is defined by its long *sequence* of growth (staying green across Kharif and Rabi), a sequence model is superior.

**Proposed Architecture: BiLSTM**
- **Input:** Sequence of shape `(14, C)` where 14 is the number of monthly timesteps and C is the number of features (bands + indices).
- **Encoder:** Bidirectional LSTM (e.g., 64 units) to capture forward and backward phenological patterns.
- **Attention:** Temporal attention layer to let the model focus on critical months (e.g., November/December when Rice-Wheat fields are bare but Sugarcane is green).
- **Classification Head:** Dense layers with Sigmoid output.

### Evaluation Metrics
- **F1-Score:** Primary metric due to class imbalance in real-world deployment.
- **Confusion Matrix Focus:** We must specifically monitor False Positives coming from the "Fodder/Sorghum" and "Orchard" negative classes.

### Post-Processing Rule (Secondary Filter)
If the LSTM predicts "Sugarcane" (Prob > 0.5), apply a sanity check:
`IF min(NDVI_Nov, NDVI_Dec) < 0.25 THEN Reject (Likely Kharif harvest / Rabi sowing)`
