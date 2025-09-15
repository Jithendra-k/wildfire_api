import os
import pandas as pd
import numpy as np
import joblib
from google.cloud import storage
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from app.imputer_model import WildfireImputer  # <-- IMPORTANT: ensures pickle saves correctly


def load_data():
    bucket_name = "data_housee"
    prefix = "wildfire_ml_data/featured_data_with_risk_parquet/"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # List parquet files
    blobs = list(bucket.list_blobs(prefix=prefix))
    parquet_files = [f"gs://{bucket_name}/{b.name}" for b in blobs if b.name.endswith(".parquet")]

    if not parquet_files:
        raise RuntimeError(f"No parquet files found in gs://{bucket_name}/{prefix}")

    print(f"Found {len(parquet_files)} shards. Loading dataset...")
    df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
    print(f"Dataset loaded: {df.shape}")
    return df


def train_imputer(df):
    # Feature definitions
    drop_cols = ["duration", "global_fire_event_id"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    geo_block = ["state", "county", "latitude", "longitude"]

    categorical_cols = [c for c in feature_cols if df[c].dtype == "object"]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols + geo_block]

    print("Numeric features:", len(numeric_cols))
    print("Categorical features:", len(categorical_cols))
    print("Geo-block features:", geo_block)

    # Fit scaler
    X_num = df[numeric_cols].dropna()
    scaler = StandardScaler()
    scaler.fit(X_num)

    # Train NearestNeighbors on scaled numerics
    X_all = df[numeric_cols].fillna(df[numeric_cols].median())
    X_all_scaled = scaler.transform(X_all)

    knn_index = NearestNeighbors(metric="euclidean")
    knn_index.fit(X_all_scaled)

    # Build custom imputer
    wildfire_imputer = WildfireImputer(
        df=df,
        numeric_cols=numeric_cols,
        geo_block=geo_block,
        scaler=scaler,
        knn_index=knn_index,
        k=10,
    )

    return wildfire_imputer


def save_and_upload(model, local_path="models/wildfire_imputer.pkl"):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    joblib.dump(model, local_path)
    print(f"Saved model locally at {local_path}")

    # Upload to GCS
    bucket_name = "data_housee"
    blob_path = "wildfire_ml_models/wildfire_imputer.pkl"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)

    print(f"Uploaded model to gs://{bucket_name}/{blob_path}")


if __name__ == "__main__":
    df = load_data()
    imputer = train_imputer(df)
    save_and_upload(imputer)
