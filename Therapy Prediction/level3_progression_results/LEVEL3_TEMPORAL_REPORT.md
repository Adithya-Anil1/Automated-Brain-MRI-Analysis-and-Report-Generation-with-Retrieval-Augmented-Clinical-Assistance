# Level 3: Temporal Consistency-Based Progression Detection

## Summary Report
**Generated:** 2026-02-02 01:48:05  
**Dataset:** LUMIERE Longitudinal Brain MRI  
**Patients Analyzed:** 77

---

## 1. Executive Summary

Level 3 implements **temporal consistency-based progression detection** where the unit of 
reasoning shifts from individual scan pairs (Level 2) to **patient-level longitudinal trends** 
across multiple (≥3) chronologically ordered scans.

### Key Results

| Metric | Value |
|--------|-------|
| Patients Analyzed | 77 |
| Accuracy | 93.5% |
| ROC-AUC | 0.975 |
| Precision | 0.972 |
| Recall | 0.958 |
| F1 Score | 0.965 |

---

## 2. Methodology Comparison

### Level 2: Single Baseline-Followup Pair Analysis

```
Scan_1 ──────────► Scan_2
         │
         └── Compare volumes, predict progression
```

**Characteristics:**
- Analyzes ONE pair at a time
- Multiple predictions per patient (one per pair)
- Sensitive to transient fluctuations
- Cannot distinguish true progression from pseudoprogression
- Prone to segmentation noise

### Level 3: Multi-Scan Temporal Trend Analysis

```
Scan_1 ──► Scan_2 ──► Scan_3 ──► Scan_4 ──► ...
                         │
                         └── Analyze TREND across all scans
                             ↓
                         Single patient-level prediction
```

**Characteristics:**
- Requires ≥3 chronologically ordered scans
- ONE prediction per patient (patient-level)
- Focuses on CONSISTENT trends, not single-point changes
- Filters out transient spikes through temporal consistency
- Growth slope, consecutive increases, and fraction-based analysis

---

## 3. Dataset Characteristics

### Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Progression | 72 | 93.5% |
| Non-Progression | 5 | 6.5% |

### Scan Statistics

| Statistic | Value |
|-----------|-------|
| Minimum scans/patient | 3 |
| Maximum scans/patient | 20 |
| Average scans/patient | 7.5 |
| Total scan records | 575 |

**Note:** The heavy class imbalance (93.5% Progression) reflects the clinical reality of 
glioma patients, who frequently progress despite treatment. This is NOT data leakage.

---

## 4. Temporal Features Extracted

Level 3 extracts **46 temporal features** organized into categories:

### 4.1 Trend Features (per region: WT, TC, ET)

| Feature | Description |
|---------|-------------|
| `*_slope` | Growth slope (ml/week) via linear regression |
| `*_r_squared` | Goodness of fit for slope |
| `*_net_change` | Volume change from first to last scan |
| `*_net_change_pct` | Percentage change from first to last |
| `*_max_increase` | Maximum single-interval increase |
| `*_max_decrease` | Maximum single-interval decrease |

### 4.2 Consistency Features (per region)

| Feature | Description |
|---------|-------------|
| `*_frac_increasing` | Fraction of intervals showing increase |
| `*_frac_decreasing` | Fraction of intervals showing decrease |
| `*_max_consecutive_inc` | Longest streak of consecutive increases |
| `*_monotonic_increase` | Flag: all intervals increasing |
| `*_monotonic_decrease` | Flag: all intervals decreasing |

### 4.3 Composition Trends

| Feature | Description |
|---------|-------------|
| `TC_WT_trend_slope` | Slope of TC/WT ratio over time |
| `TC_WT_net_change` | Change in TC/WT ratio |
| `ET_WT_trend_slope` | Slope of ET/WT ratio over time |
| `ET_WT_net_change` | Change in ET/WT ratio |

### 4.4 Noise Detection Features

| Feature | Description |
|---------|-------------|
| `*_volatility` | Standard deviation of intervals |
| `*_direction_changes` | Number of sign flips (noise indicator) |
| `*_newly_persistent` | Region appeared and sustained |

---

## 5. Noise Handling (REQUIRED)

Level 3 implements **strict noise thresholds**:

1. **Absolute Volume Threshold:** 50 mm³ (0.05 ml)
   - Changes smaller than this are ignored
   - Prevents segmentation artifact influence

2. **Consecutive Scan Requirement:**
   - Single-scan spikes are NOT treated as progression
   - Requires trend confirmation across subsequent scans

3. **Direction Change Detection:**
   - High volatility (frequent sign flips) indicates noise
   - Penalizes oscillating patterns

---

## 6. Model Performance

### Best Model: Logistic Regression

| Metric | Score |
|--------|-------|
| Accuracy | 0.9351 |
| Precision | 0.9718 |
| Recall | 0.9583 |
| F1 Score | 0.9650 |
| ROC-AUC | 0.9750 |

### Feature Importance Insights

Key discriminating features between classes:

| Feature | Progression (mean) | Non-Progression (mean) |
|---------|-------------------|----------------------|
| WT_frac_increasing | 50.6% | 10.0% |
| TC_frac_increasing | 50.2% | 20.0% |
| TC_max_consecutive_inc | 2.25 | 0.40 |
| WT_net_change_pct | +183% | -82% |

---

## 7. Error Analysis

### Overall Performance
- **Correct Predictions:** 72 (93.5%)
- **Incorrect Predictions:** 5 (6.5%)

### Error Patterns

**False Negatives (Predicted Non-Progression, True Progression):**
- Patients with decreasing volumes but RANO-rated PD
- Clinical progression via non-volumetric factors (infiltration, functional decline)
- Example: Patient-001 with WT slope = -0.49 ml/week but RANO trajectory SD→PD

**False Positives (Predicted Progression, True Non-Progression):**
- Short time spans (3 weeks) with ambiguous trends
- High TC/WT ratio changes due to edema reduction
- Low confidence predictions (50-53%)

---

## 8. How Temporal Consistency Improves Robustness

### 8.1 Problem: Single-Scan Spike False Positives (Level 2)

Level 2 may react to:
```
Week 0: 50 ml ──► Week 8: 65 ml (+30% 🔴 SPIKE)
                              │
                              └── Level 2: PROGRESSION ❌
```

But with more scans:
```
Week 0: 50 ml ──► Week 8: 65 ml ──► Week 16: 48 ml ──► Week 24: 50 ml
                     │                  │
                     └── Spike          └── Return to baseline
                     
                     Level 3: Analyze full trend = STABLE ✓
```

### 8.2 Solution: Temporal Trend Analysis (Level 3)

**Consecutive Increase Detection:**
```
Week 0: 50 ml ──► Week 8: 52 ml ──► Week 16: 55 ml ──► Week 24: 59 ml
                     ↑                  ↑                   ↑
                     +2                 +3                  +4
                     
                     Level 3: 3 consecutive increases = PROGRESSION ✓
```

**Fraction-Based Analysis:**
- If 75% of intervals show increase → likely true progression
- If only 25% show increase with oscillations → likely noise/stable

**Slope Analysis:**
- Positive slope (ml/week) → progressive trend
- Near-zero slope → stable
- Negative slope → responding

---

## 9. Comparison Summary: Level 2 vs Level 3

| Aspect | Level 2 | Level 3 |
|--------|---------|---------|
| **Unit of Analysis** | Scan pair | Patient trajectory |
| **Scans Required** | 2 | ≥3 |
| **Predictions per Patient** | Multiple | One |
| **Temporal Consistency** | None | Required |
| **Noise Robustness** | Low | High |
| **Spike Handling** | React immediately | Require confirmation |
| **Explainability** | Pair-based | Trend-based |

---

## 10. Clinical Implications

### Advantages for Clinical Decision Support

1. **Reduced False Alarms:**
   - Transient post-treatment changes (pseudoprogression) filtered
   - Segmentation variability smoothed out

2. **Confidence Calibration:**
   - More scans → Higher confidence (capped to avoid bias)
   - Short trajectories flagged as lower confidence

3. **Trajectory-Based Reasoning:**
   - "3 consecutive increases" is clinically interpretable
   - Aligns with RANO assessment methodology

### Limitations

1. **Non-Volumetric Progression:**
   - Infiltrative growth without volume change
   - Functional decline without structural change

2. **Class Imbalance:**
   - Only 5 Non-Progression patients in dataset
   - Larger validation cohort needed

3. **Minimum Scan Requirement:**
   - Cannot apply to new patients with <3 scans
   - Level 2 still needed for initial assessments

---

## 11. Files Generated

| File | Description |
|------|-------------|
| `level3_temporal_features.csv` | Extracted features for all patients |
| `clinical_summaries.csv` | Per-patient predictions and explanations |
| `model_metrics.csv` | Cross-validation performance metrics |
| `LEVEL3_TEMPORAL_REPORT.md` | This report |

---

## 12. Constraint Verification

✅ **Multi-scan temporal analysis** (≥3 scans per patient)  
✅ **Patient-level predictions** (one per patient)  
✅ **Labels from RANO** (not derived from features)  
✅ **Noise threshold** (50 mm³) applied  
✅ **No RNNs/LSTMs/Transformers**  
✅ **No CNNs**  
✅ **No future prediction/forecasting**  
✅ **Temporal explanations** (Decision, Reason, Confidence)  

---

*Report generated by Level 3 Temporal Progression Detection Pipeline*
