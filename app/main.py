from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any
from app.imputer import impute_features
from app.model_download import xgb_best_model, xgb_hazard_model, random_forest_model
from app import risk_heatmap, emissions

app = FastAPI(title="Wildfire API", version="1.0")

# ---------- Request Models ----------
class ImputeRequest(BaseModel):
    features: Dict[str, Any]
    round_risk: Optional[bool] = False


# ---------- Root ----------
@app.get("/")
def read_root():
    return {"message": "Wildfire API running"}


# ---------- Imputation Endpoint ----------
@app.post("/impute")
def impute_endpoint(request: ImputeRequest):
    result = impute_features(request.features, round_risk=request.round_risk)
    return {"input": request.features, "imputed": result}



def map_duration_to_risk(duration: float) -> float:
    if duration >= 10:
        # Map duration ≥ 10 into [6, 10], capped at 10
        return min(10, 6 + (duration - 10) / 20 * 4)
        # every +20 days adds ~4 risk, but max at 10
    elif duration >= 5:
        # Map duration [5,10) → risk [3,6]
        return 3 + (duration - 5) / 5 * 3
    elif duration >= 1:
        # Map duration [1,5) → risk [1,3]
        return 1 + (duration - 1) / 4 * 2
    else:
        # Map duration [0,1) → risk [0,1]
        return max(0, min(1, duration*2.5))



# ---------- Prediction Endpoint ----------
@app.post("/predict")
def predict_endpoint(request: ImputeRequest):
    import pandas as pd
    from app.bigquery_utils import fetch_risk_events

    # Step 1: impute missing features
    imputed = impute_features(request.features, round_risk=request.round_risk)

    # Step 2: drop excluded columns
    exclude_cols = ['duration', 'global_fire_event_id', 'state', 'county', 'end_tomorrow', 'risk']
    feature_vector = {k: v for k, v in imputed.items() if k not in exclude_cols}

    X = pd.DataFrame([feature_vector])

    # Step 3: predictions
    duration_pred = abs(float(xgb_best_model.predict(X)[0]))
    proba = xgb_hazard_model.predict_proba(X)
    end_probability = float(proba[0, 1])  # probability fire continues tomorrow
    end_label = int(end_probability >= 0.5)

    # Step 4: determine query filters for risk events
    state = request.features.get("state")
    county = request.features.get("county")
    doy = request.features.get("doy") or imputed.get("doy")
    season = request.features.get("season") or imputed.get("season")

    # Step 5: fetch historical risk events from BigQuery
    risk_events = fetch_risk_events(
        state=state,
        county=county,
        season=season,
        doy=doy,
        risk=imputed.get("risk")
    )
    risk_adjusted = map_duration_to_risk(duration_pred)
    if risk_adjusted>10:
        risk_adjusted= 8.97645
    # Final response
    return {
        "input": request.features,
        "imputed": imputed,
        "predictions": {
            "duration": duration_pred,
            "end_tomorrow_prob": end_probability,
            "end_tomorrow_label": end_label,
            "adjusted_risk": risk_adjusted
        },
        "risk_events": risk_events
    }


# @app.post("/predict-r")
# def predict_r_endpoint(request: ImputeRequest):
#     import pandas as pd
#     from app.bigquery_utils import fetch_risk_events
#
#     # Step 1: impute missing features
#     imputed = impute_features(request.features, round_risk=request.round_risk)
#
#     # Step 2: drop excluded columns
#     exclude_cols = ['duration', 'global_fire_event_id', 'state', 'county', 'end_tomorrow', 'risk']
#     feature_vector = {k: v for k, v in imputed.items() if k not in exclude_cols}
#
#
#     X = pd.DataFrame([feature_vector])
#     if "doy" in X.columns:
#         X = X.drop(columns=["doy"])
#     # Step 3: predictions
#     # Using RandomForestRegressor for duration prediction
#     duration_pred = float(random_forest_model.predict(X)[0])
#
#     # Still use hazard classifier for end_tomorrow
#     # proba = xgb_hazard_model.predict_proba(X)
#     # end_probability = float(proba[0, 1])
#     # end_label = int(end_probability >= 0.5)
#
#     # Step 4: determine query filters for risk events
#     state = request.features.get("state")
#     county = request.features.get("county")
#     #doy = request.features.get("doy") or imputed.get("doy")
#     season = request.features.get("season") or imputed.get("season")
#
#     # Step 5: fetch historical risk events from BigQuery
#     risk_events = fetch_risk_events(
#         state=state,
#         county=county,
#         season=season,
#         #doy=doy,
#         risk=imputed.get("risk")
#     )
#
#     # Final response
#     return {
#         "input": request.features,
#         "imputed": imputed,
#         "predictions": {
#             "duration": duration_pred
#             # "end_tomorrow_prob": end_probability,
#             # "end_tomorrow_label": end_label
#         },
#         "risk_events": risk_events
#     }



# ---------- Risk Heatmap Endpoint ----------
# include the router from app/risk_heatmap.py
app.include_router(risk_heatmap.router)

# include emission router
# The existing include should work, but make sure it's there
app.include_router(emissions.router, prefix="/api", tags=["emissions"])