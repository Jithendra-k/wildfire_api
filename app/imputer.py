import os
import pandas as pd
import numpy as np
from google.cloud import storage
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# -----------------------------
# Load Dataset from GCS (all shards)
# -----------------------------
bucket_name = "data_housee"
prefix = "wildfire_ml_data/featured_data_with_risk_parquet/"

storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)

# List parquet files in GCS
blobs = list(bucket.list_blobs(prefix=prefix))
parquet_files = [b.name for b in blobs if b.name.endswith(".parquet")]

if not parquet_files:
    raise RuntimeError(f"No parquet files found under gs://{bucket_name}/{prefix}")

print(f"Found {len(parquet_files)} shards. Loading into DataFrame...")

df_list = []
for f in parquet_files:
    local_path = os.path.basename(f)
    blob = bucket.blob(f)
    blob.download_to_filename(local_path)
    df_list.append(pd.read_parquet(local_path))

df = pd.concat(df_list, ignore_index=True)
print(f"Dataset loaded: {df.shape}")

# -----------------------------
# Feature Lists
# -----------------------------
drop_cols = ['duration', 'risk', 'global_fire_event_id']
feature_cols = [c for c in df.columns if c not in drop_cols]

# Numeric features (exclude state/county)
num_features = [
    c for c in feature_cols
    if df[c].dtype != 'object' and c not in ['state', 'county']
]

# -----------------------------
# Fit Scaler Once
# -----------------------------
X_num = df[num_features].fillna(df[num_features].median())
scaler = StandardScaler()
scaler.fit(X_num)

print(f"Scaler fitted on {len(num_features)} numeric features.")

# -----------------------------
# Imputation Function
# -----------------------------
def impute_features(user_json, k=10, round_risk=False):
    """
    Impute missing wildfire features using kNN + rule-based corrections.
    - Preserves user-provided inputs.
    - Handles zero-heavy fractions.
    - Optionally rounds risk to int.
    """
    # Extract state/county if provided
    state_filter = user_json.get("state")
    county_filter = user_json.get("county")

    df_subset = df.copy()
    if state_filter is not None:
        df_subset = df_subset[df_subset["state"] == state_filter]
    if county_filter is not None:
        df_subset = df_subset[df_subset["county"] == county_filter]

    # Fallback if no matches
    if df_subset.empty:
        df_subset = df.copy()

    # Numeric subset for kNN
    X_subset = df_subset[num_features].fillna(df[num_features].median())
    X_subset_scaled = scaler.transform(X_subset)

    knn_local = NearestNeighbors(n_neighbors=min(k, len(X_subset)), metric='euclidean')
    knn_local.fit(X_subset_scaled)

    # Build user row
    user_df = pd.DataFrame([user_json], columns=feature_cols)

    # Fill missing numerics with medians for scaling
    temp_num = user_df[num_features].fillna(df[num_features].median())
    temp_num_scaled = scaler.transform(temp_num)

    distances, indices = knn_local.kneighbors(temp_num_scaled)
    neighbor_values = df_subset.iloc[indices[0]]

    # ---- Numeric Imputation (mean of neighbors) ----
    for col in num_features:
        if pd.isna(user_df.loc[0, col]):  # only impute if missing
            user_df.loc[0, col] = neighbor_values[col].mean()

    # ---- Categorical Imputation (mode of neighbors) ----
    categorical_cols = [c for c in feature_cols if c not in num_features]
    for col in categorical_cols:
        if pd.isna(user_df.loc[0, col]) or user_df.loc[0, col] is None:
            user_df.loc[0, col] = neighbor_values[col].mode().iloc[0]

    # ---- Derived Features ----
    if "doy" in user_df.columns and not pd.isna(user_df.loc[0, "doy"]):
        doy = int(round(user_df.loc[0, "doy"]))
        user_df.loc[0, "doy"] = doy
        user_df.loc[0, "day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
        user_df.loc[0, "day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)

        # Ensure month & season consistency
        if "month" in user_df.columns and pd.isna(user_df.loc[0, "month"]):
            user_df.loc[0, "month"] = pd.to_datetime(f"2020-{doy}", format="%Y-%j").month
        if "season" in user_df.columns:
            user_df.loc[0, "season"] = (int(user_df.loc[0, "month"]) % 12 // 3) + 1

    # ---- Special Handling ----
    # Prefire_fuel: clip to 99th percentile
    if "prefire_fuel" in user_df.columns and not pd.isna(user_df.loc[0, "prefire_fuel"]):
        p99 = df["prefire_fuel"].quantile(0.99)
        user_df.loc[0, "prefire_fuel"] = min(user_df.loc[0, "prefire_fuel"], p99)

    # Zero-heavy fractions: replace 0 if rare
    for frac_col in ["cwd_frac", "duff_frac"]:
        if frac_col in user_df.columns:
            if user_df.loc[0, frac_col] == 0:
                zero_ratio = (df[frac_col] == 0).mean()
                if zero_ratio < 0.05:  # if zeros are rare
                    user_df.loc[0, frac_col] = neighbor_values[frac_col].median()

    # Risk: continuous mean or rounded
    risk_val = neighbor_values["risk"].mean()
    user_df["risk"] = int(round(risk_val)) if round_risk else risk_val

    # ---- Type Enforcement ----
    int_cats = ["covertype", "fuelcode", "fuel_moisture_class", "burn_source",
                "burnday_source", "BSEV", "month", "season", "doy"]
    for col in int_cats:
        if col in user_df.columns and not pd.isna(user_df.loc[0, col]):
            user_df.loc[0, col] = int(round(user_df.loc[0, col]))

    # State/county as strings
    if "state" in user_df.columns:
        user_df["state"] = user_df["state"].astype(str)
    if "county" in user_df.columns:
        user_df["county"] = user_df["county"].astype(str)

    return user_df.iloc[0].to_dict()
