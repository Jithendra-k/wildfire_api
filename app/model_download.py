import os
import joblib
import warnings
from google.cloud import storage

BUCKET_NAME = "data_housee"
MODEL_DIR = "wildfire_ml_models"
LOCAL_MODEL_DIR = "models"

os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

def download_if_needed(model_name: str):
    local_path = os.path.join(LOCAL_MODEL_DIR, model_name)
    if not os.path.exists(local_path):
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"{MODEL_DIR}/{model_name}")
        blob.download_to_filename(local_path)
        print(f"Downloaded {model_name} from GCS to {local_path}")
    else:
        print(f"Using cached model: {local_path}")
    return local_path


# Suppress sklearn/xgboost warnings while loading
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    warnings.simplefilter("ignore", category=FutureWarning)
    warnings.simplefilter("ignore", category=Warning)  # broad filter if needed

    xgb_best_path = download_if_needed("xgb_best_model.pkl")
    random_forest_model_path = download_if_needed("RandomForestRegressor_model.pkl")
    xgb_hazard_path = download_if_needed("xgb_hazard_calibrated.pkl")

    xgb_best_model = joblib.load(xgb_best_path)
    random_forest_model = joblib.load(random_forest_model_path)
    xgb_hazard_model = joblib.load(xgb_hazard_path)
