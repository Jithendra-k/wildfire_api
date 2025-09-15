# Wildfire API Documentation

This API provides functionality for imputing missing wildfire features and predicting fire outcomes using trained machine learning models.

---

## 1. POST `/impute`

### Description
The `/impute` endpoint takes in partial wildfire feature data (e.g., state, county, prefire fuel, weather metrics) and imputes the missing values using a trained offline imputer model. The output includes a fully imputed feature set consistent with historical wildfire records.

### Input
- JSON body with partial wildfire attributes.
- Example:

```json
{
  "state": "Wyoming",
  "county": "Park",
  "prefire_fuel": 50,
  "rmax_value": 25,
  "covertype": 3,
  "doy": 208
}
```
### Output
- JSON object containing: The original input & The imputed feature set with all missing values filled
- Example:
```json
{
  "input": {
    "state": "Wyoming",
    "county": "Park",
    "prefire_fuel": 50,
    "rmax_value": 25,
    "covertype": 3,
    "doy": 208
  },
  "imputed": {
    "longitude": -111.875,
    "prefire_fuel": 50,
    "cwd_frac": 0.0,
    "duff_frac": 0.0,
    "day_of_year_sin": -0.4227806579826773,
    "fm1000_value": 10.16,
    "srad_value": 282.67,
    "day_of_year_cos": -0.9062,
    "bi_value": 55.5,
    "fm100_value": 7.14,
    "rmax_value": 25,
    "month": 8,
    "season": 3,
    "covertype": 3,
    "th_value": 233,
    "latitude": 44.83,
    "burn_source": 1,
    "rmin_value": 16.54,
    "vs_value": 3.58,
    "burnday_source": 15,
    "vpd_value": 1.59,
    "sph_value": 0.00455,
    "pet_value": 5.94,
    "BSEV": 2,
    "tmmn_value": 282.02,
    "fuelcode": 1,
    "fuel_moisture_class": 2,
    "state": "Wyoming",
    "county": "Park",
    "global_fire_event_id": null,
    "duration": null,
    "doy": 208,
    "risk": 3.19
  }
}

```
## 2. POST `/predict`

### Description
The `/predict` endpoint uses the fully imputed wildfire features (such as the output from `/impute`) to predict two outcomes:
1. **Fire Duration** (regression) — Predicted number of days the fire will last, using the `xgb_best_model.pkl` model.
2. **End Tomorrow Probability** (classification) — Probability that the fire will end the next day, using the `xgb_hazard_calibrated.pkl` model.

---

### Input
Provide a JSON object containing wildfire attributes. These can be partial user inputs or the output from the `/impute` endpoint.  

**Example:**
```json
{
  "input": {
    "state": "Wyoming",
    "county": "Park",
    "prefire_fuel": 50,
    "rmax_value": 25,
    "covertype": 3,
    "doy": 208
  }
}
```
### Output
- JSON object containing: The original input, The imputed features & Predictions for duration and end-tomorrow probability

**Example:**
```json
{
  "input": {
    "state": "Wyoming",
    "county": "Park",
    "prefire_fuel": 50,
    "rmax_value": 25,
    "covertype": 3,
    "doy": 208
  },
  "imputed": {
    "longitude": -111.875,
    "prefire_fuel": 50,
    "cwd_frac": 0.0,
    "duff_frac": 0.0,
    "day_of_year_sin": -0.4228,
    "fm1000_value": 10.16,
    "srad_value": 282.67,
    "day_of_year_cos": -0.9062,
    "bi_value": 55.5,
    "fm100_value": 7.14,
    "rmax_value": 25,
    "month": 8,
    "season": 3,
    "covertype": 3,
    "th_value": 233,
    "latitude": 44.83,
    "burn_source": 1,
    "rmin_value": 16.54,
    "vs_value": 3.58,
    "burnday_source": 15,
    "vpd_value": 1.59,
    "sph_value": 0.00455,
    "pet_value": 5.94,
    "BSEV": 2,
    "tmmn_value": 282.02,
    "fuelcode": 1,
    "fuel_moisture_class": 2,
    "state": "Wyoming",
    "county": "Park",
    "global_fire_event_id": null,
    "duration": null,
    "doy": 208,
    "risk": 3.19
  },
  "predictions": {
    "duration": 6.77,
    "end_tomorrow_prob": 0.05,
    "end_tomorrow_label": 0
  }
}
```
### Notes
- Always call /impute first if input data has missing values.
- The end_tomorrow_label is derived by applying a probability threshold (default: 0.5).
- Models are pre-trained and loaded from Google Cloud Storage at startup.