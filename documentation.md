# Wildfire Modeling and Prediction Project

## Overview
This repository documents the evolution of wildfire prediction approaches, starting from previously published work (Wildfire Prediction-3, “WP3”) and extending it with improved training protocols, additional features, sequential modeling, and risk-based classification methods.

The project was developed to address shortcomings in earlier studies, which relied on **non–group-aware training** and thus risked data leakage across fire events. This documentation outlines:
- The analysis of prior research and its limitations.  
- The improvements and new approaches introduced.  
- The datasets and preprocessing pipelines.  
- The models and their results.  
- The file and folder structure for reproducibility.  

---

## 1. Background and Previous Research

### 1.1 Prior Work (Wildfire Prediction-3)
The WP3 project created a large-scale wildfire dataset (~7.25M fire-day records), combining satellite fire detections with gridded meteorological and emissions data.  
They applied clustering (DBSCAN) to identify unique fire events and trained models (e.g., Random Forests, Gradient Boosting) to predict **event duration**.

**Key limitations of prior work:**
1. **Non-group-aware splitting**:  
   - All models were trained/tested on random splits.  
   - This caused data leakage because fire events were split across training and test sets.  
   - WP3 authors acknowledged this but did not apply group-aware training.  

2. **Duration as static target**:  
   - Collapsed all temporal daily variability into one label (total duration).  
   - Reduced signal-to-noise ratio and biased models towards short fires.

3. **Limited feature enrichment**:  
   - Relied only on core meteorological, emissions, and fire geometry data.  
   - No regional context (e.g., state/county, urban factors) was included.  

---

## 2. Contributions in This Project

### 2.1 Data Enhancements
- **Added State and County columns** using Google’s public geo-database.  
- Attempted city-level mapping (via ~32k polygons), but the join with 7.25M fire-day rows was computationally infeasible (~230B operations).  
  - Skipped this step due to high cost/time.  
- Produced feature-augmented datasets including **33 optimal features** (original + state, county, duration).  

All preprocessing pipelines are implemented in **BigQuery notebooks** (in the instance).  

### 2.2 Basic Exploration
- Conducted preliminary EDA on seasonal fire activity and distribution.  
- Observed peaks in fire activity between **days 80–110 (spring)** and **days 200–250 (summer)**.  
- Validated group-aware splitting preserved complete fire events across train/test sets.

---

## 3. Approach 1: Group-Aware Duration Regression

### 3.1 Motivation
Previous regression models achieved inflated performance due to random splits.  
We re-trained models with **GroupKFold (by fire event ID)** to prevent leakage.  

### 3.2 Baseline Results
- **Group-aware baseline** for XGBoost/HistGradientBoosting:  
  - XGBoost: R² ≈ 0.41  
  - HistGradientBoosting: R² ≈ 0.31  

### 3.3 Optimizations
- Applied **RandomizedSearchCV** and **GridSearchCV** for hyperparameter tuning.  
- Tuned hyperparameters:  
  - XGB: `max_depth, learning_rate, subsample, colsample_bytree, n_estimators`  
  - HistGB: `max_depth, learning_rate, min_samples_leaf`  

### 3.4 Results
- **XGBoost (tuned):** R² ≈ 0.74, RMSE ≈ 6.1 days, MAE ≈ 2.8 days.  
- **HistGradientBoosting (tuned):** R² ≈ 0.58, RMSE ≈ 7.8 days, MAE ≈ 4.2 days.  

### 3.5 Notes
- Models trained on **data shards** stored in:  
```bash
gs://data_housee/wildfire_ml_data/
├── wildfire_final_data_csv/
├── wildfire_final_data_parquet/
├── featured_data_csv/
├── featured_data_parquet/
├── sequential_data/
├── featured_data_with_risk_csv/
├── featured_data_with_risk_parquet/
├── train_test_regular/
└── train_test_group/
```

- Code file: `XGB_HISTGRAD-1` notebook (in instance).  
- Final models saved under:  
```gs://data_housee/wildfire_ml_models/```


---

## 4. Approach 2: Sequential Modeling (ConvLSTM)

### 4.1 Motivation
Static duration regression discards daily variability.  
Sequential modeling preserves **temporal fire-weather dynamics**.  

### 4.2 Data Preparation
- Reduced ~7.25M rows → ~1M fire events.  
- Each fire represented as a sequence of **1–113 daily feature vectors**.  
- Long fires (>46 days, top 1%) trimmed for stability.  
- Final sequences padded to uniform length.  

### 4.3 Model
- Architecture: **ConvLSTM** (64 hidden units).  
- Loss: SmoothL1Loss (Huber).  
- Optimizer: Adam (lr=0.001).  
- Training: 15 epochs, WeightedRandomSampler for imbalance.  

### 4.4 Results
- Validation MAE: ~0.42–0.50 days.  
- Validation RMSE: ~3.2–3.7 days.  
- R²: ~0.66.  
- Sub-day accuracy achieved for short fires, but long-duration fires (>30 days) remained challenging.  

### 4.5 Notes
- Code file: `sequential_modeling` notebook (in instance).  
- Data stored in `sequential_data/` under `data_housee/wildfire_ml_data/`.

---

## 5. Approach 3: Binary Classification (Hazard-Based)

### 5.1 Motivation
Emergency responders need **near-term forecasts**, not static durations.  
Reframed the task as: *“Will the fire end tomorrow?”*  

### 5.2 Formulation
- Label:  
y_t = 1 if fire ends on day t+1
0 otherwise

- Loss: Binary Cross-Entropy.  

### 5.3 Models
- **XGBoost Classifier** (baseline).  
- Applied **isotonic calibration** to align probabilities with observed frequencies.  

### 5.4 Results
- Accuracy: ~92.4%.  
- ROC-AUC: ~0.93.  
- Precision: ~0.77, Recall: ~0.60 (after calibration).  
- Output probabilities became interpretable as true extinction risks.  

### 5.5 Notes
- Code file: `risk_binary_classification` notebook.  
- Models saved under:  
```gs://data_housee/wildfire_ml_models/```


---

## 6. Additional Notes on Data

- **Feature engineering**: Added *state*, *county*, *duration*.  
- **City-level mapping skipped** due to computational cost.  
- **Feature sets** derived (33 features, including new ones).  
- **Shards**: All large datasets split into 100 shards (CSV + Parquet) for efficient processing.  

---

## 7. Repository and Code Layout

- **BigQuery Notebooks**: For all data preprocessing, cleaning, and feature engineering.  
- **Python Notebooks**: For modeling experiments.  
- **Models Folder**: Saved models (`.pkl`) in GCS bucket.  

---

## 8. Results Summary

| Approach                  | Model                  | R²   | RMSE (days) | MAE (days) | ROC-AUC | Notes                          |
|---------------------------|------------------------|------|-------------|------------|---------|--------------------------------|
| Duration Regression       | XGBoost (tuned)        | 0.74 | 6.1         | 2.8        | –       | Strong baseline, group-aware.  |
| Duration Regression       | HistGradientBoosting   | 0.58 | 7.8         | 4.2        | –       | Efficient, modest accuracy.    |
| Sequential Modeling       | ConvLSTM               | 0.66 | 3.2–3.7     | 0.42–0.50  | –       | Best for short fires.          |
| Hazard Classification     | XGB Classifier (calib) | –    | –           | –          | 0.93    | Actionable daily risk.         |

---

## 9. Risk Scoring System

In addition to predictive modeling, this project incorporates a **risk scoring framework** that synthesizes multiple environmental variables into a single interpretable index.  
The purpose of this system is to provide **real-time situational awareness** of wildfire conditions that can be used alongside model outputs.

### 9.1 Calculation Method

The risk score is calculated in four steps:

1. **Normalization**  
   Each feature is scaled into the range [0,1] using min–max normalization:  

   \[
   x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}
   \]

2. **Sub-score Calculation**  
   - **Temperature**: Average of normalized max and min temperature.  
   - **Humidity**: Average of normalized min and max humidity, **inverted** as (1 – score) since lower humidity increases risk.  
   - **Wind, fuel, debris, and duff**: Directly use normalized values.  

3. **Weighted Aggregation**  
   The sub-scores are combined into a weighted linear index:  

   \[
   risk\_score = 0.25 \cdot Temp + 0.20 \cdot Humidity + 0.20 \cdot Wind + 0.20 \cdot Fuel + 0.10 \cdot Debris + 0.05 \cdot Duff
   \]

4. **Scaling to 0–10**  
   The final risk level is scaled into a 0–10 range for interpretability:  

   \[
   risk\_level = risk\_score \times 10
   \]

### 9.2 Example

Suppose a fire-day record has normalized values:  

- Temperature = 0.7  
- Humidity = 0.8  
- Wind = 0.5  
- Fuel = 0.6  
- Debris = 0.3  
- Duff = 0.4  

Then:

\[
risk\_score = (0.7 \times 0.25) + (0.8 \times 0.20) + (0.5 \times 0.20) + (0.6 \times 0.20) + (0.3 \times 0.10) + (0.4 \times 0.05) = 0.605
\]

\[
risk\_level = 0.605 \times 10 = 6.05
\]

Thus, this day would be assigned a **risk level of ~6.1 on a 0–10 scale**.

### 9.3 Distribution Insights

When applied to the dataset:
- Most scores fall between **2.5–5.5**, with a concentration near ~3.0.  
- The distribution is **right-skewed**, with fewer high-risk cases.  
- A tail of **elevated risk (>5)** cases exists, representing situations where conditions are most conducive to fire spread.

### 9.4 Applications

- **Decision support**: Can be visualized in dashboards for fire managers.  
- **Model feature**: Can be incorporated as an additional predictor in regression/classification models.  
- **Public communication**: Scaled 0–10 risk levels are interpretable for non-technical stakeholders.

---

## 10. Limitations and Future Work
1. **Data Imbalance**: One-day fires dominate. Need resampling or survival analysis for long fires.  
2. **Long-duration fires**: Sequential models still underperform for rare 30+ day events.  
3. **City-level enrichment**: Adding urban-level context requires distributed spatial joins.  
4. **Operational integration**: Risk scores and probabilities must be integrated into fire management dashboards.  
5. **Explainability**: Future work will apply SHAP/attention maps to identify key drivers.  

---

## 11. References
1. Wildfire Prediction-3 (previous research).  
2. Shi, X. et al. (2015). *Convolutional LSTM Network for Precipitation Nowcasting*. NIPS.  
3. Hochreiter, S., Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.  
4. Kratzert, F. et al. (2019). *Hydrology and Earth System Sciences*.  
5. Fang, K., Shen, C. (2020). *Near-real-time forecast of soil moisture using RNNs*. Nature Communications.  

---
