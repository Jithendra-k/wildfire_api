from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
from app.imputer import impute_features

# Initialize FastAPI
app = FastAPI(title="Wildfire API", version="1.0")

# Request schema
class ImputeRequest(BaseModel):
    features: Dict[str, Any]
    round_risk: Optional[bool] = False

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Wildfire API running"}

# Imputation endpoint
@app.post("/impute")
def impute_endpoint(request: ImputeRequest):
    result = impute_features(request.features, k=10, round_risk=request.round_risk)
    return {"input": request.features, "imputed": result}
