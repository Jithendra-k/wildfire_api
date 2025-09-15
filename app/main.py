from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any
from app.imputer import impute_features
from app.model_download import xgb_best_model, xgb_hazard_model

app = FastAPI(title="Wildfire API", version="1.0")

class ImputeRequest(BaseModel):
    features: Dict[str, Any]
    round_risk: Optional[bool] = False

@app.get("/")
def read_root():
    return {"message": "Wildfire API running"}

@app.post("/impute")
def impute_endpoint(request: ImputeRequest):
    result = impute_features(request.features, round_risk=request.round_risk)
    return {"input": request.features, "imputed": result}


# Prediction endpoint
@app.post("/predict")
def predict_endpoint(request: ImputeRequest):
    # Step 1: impute missing features
    imputed = impute_features(request.features, round_risk=request.round_risk)

    # Step 2: drop excluded columns
    exclude_cols = ['duration','global_fire_event_id','state','county','end_tomorrow','risk']
    feature_vector = {k: v for k, v in imputed.items() if k not in exclude_cols}

    import pandas as pd
    X = pd.DataFrame([feature_vector])

    # Step 3: predictions
    duration_pred = float(xgb_best_model.predict(X)[0])

    proba = xgb_hazard_model.predict_proba(X)
    end_probability = float(proba[0, 1])  # probability fire continues tomorrow
    end_label = int(end_probability >= 0.5)

    return {
        "input": request.features,
        "imputed": imputed,
        "predictions": {
            "duration": duration_pred,
            "end_tomorrow_prob": end_probability,
            "end_tomorrow_label": end_label
        }
    }

