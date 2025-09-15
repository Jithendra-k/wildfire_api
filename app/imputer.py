import os
import math
import joblib
from google.cloud import storage
from app.imputer_model import WildfireImputer  # make sure class is registered

# Local model path (inside container or local dev)
LOCAL_MODEL_PATH = os.getenv("IMPUTER_PATH", "models/wildfire_imputer.pkl")

# GCS bucket + blob path
BUCKET_NAME = "data_housee"
BLOB_PATH = "wildfire_ml_models/wildfire_imputer.pkl"

def download_from_gcs(bucket_name: str, blob_path: str, local_path: str):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    print(f"Downloaded {blob_path} from gs://{bucket_name} to {local_path}")

# Ensure local model exists
if not os.path.exists(LOCAL_MODEL_PATH):
    print(f"Model not found at {LOCAL_MODEL_PATH}, downloading from GCS...")
    download_from_gcs(BUCKET_NAME, BLOB_PATH, LOCAL_MODEL_PATH)
else:
    print(f"Using local model at {LOCAL_MODEL_PATH}")

# Load imputer
wildfire_imputer = joblib.load(LOCAL_MODEL_PATH)

def clean_for_json(data: dict) -> dict:
    """Replace NaN/Inf with None so JSON serialization works."""
    clean = {}
    for k, v in data.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            clean[k] = None
        else:
            clean[k] = v
    return clean

def impute_features(user_json, k=10, round_risk=False):
    result = wildfire_imputer.transform(user_json, round_risk=round_risk)
    return clean_for_json(result)
