# app/imputer_model.py
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class WildfireImputer:
    def __init__(self, df, numeric_cols, geo_block, scaler, knn_index, k=10):
        self.df = df
        self.numeric_cols = numeric_cols
        self.geo_block = geo_block
        self.scaler = scaler
        self.knn_index = knn_index
        self.k = k

    def transform(self, user_json, round_risk=False):
        user_df = pd.DataFrame([user_json], columns=self.df.columns)

        temp_num = user_df[self.numeric_cols].fillna(self.df[self.numeric_cols].median())
        temp_num_scaled = self.scaler.transform(temp_num)

        distances, indices = self.knn_index.kneighbors(temp_num_scaled, n_neighbors=self.k)
        neighbor_values = self.df.iloc[indices[0]]

        # numeric fill
        for col in self.numeric_cols:
            if pd.isna(user_df.loc[0, col]):
                user_df.loc[0, col] = neighbor_values[col].mean()

        # geo-block logic
        state_filter = user_json.get("state")
        county_filter = user_json.get("county")
        if state_filter and county_filter:
            if pd.isna(user_json.get("latitude")) or pd.isna(user_json.get("longitude")):
                nearest = neighbor_values.iloc[0]
                user_df.loc[0, "latitude"] = nearest["latitude"]
                user_df.loc[0, "longitude"] = nearest["longitude"]
            user_df.loc[0, "state"] = state_filter
            user_df.loc[0, "county"] = county_filter
        else:
            nearest = neighbor_values.iloc[0]
            for col in self.geo_block:
                if col in ["state", "county"]:
                    user_df[col] = user_df[col].astype("object")
                user_df.loc[0, col] = nearest[col]

        # derived DOY features
        if "doy" in user_df.columns and not pd.isna(user_df.loc[0, "doy"]):
            doy = int(round(user_df.loc[0, "doy"]))
            user_df.loc[0, "doy"] = doy
            user_df.loc[0, "day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
            user_df.loc[0, "day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)
            if "month" in user_df.columns and pd.isna(user_df.loc[0, "month"]):
                user_df.loc[0, "month"] = pd.to_datetime(f"2020-{doy}", format="%Y-%j").month
            if "season" in user_df.columns:
                user_df.loc[0, "season"] = (int(user_df.loc[0, "month"]) % 12 // 3) + 1

        # prefire_fuel clip
        if "prefire_fuel" in user_df.columns and not pd.isna(user_df.loc[0, "prefire_fuel"]):
            p99 = self.df["prefire_fuel"].quantile(0.99)
            user_df.loc[0, "prefire_fuel"] = min(user_df.loc[0, "prefire_fuel"], p99)

        # risk
        if "risk" in user_df.columns:
            risk_val = neighbor_values["risk"].mean()
            user_df.loc[0, "risk"] = int(round(risk_val)) if round_risk else risk_val

        # enforce integer categories
        int_cats = ["covertype", "fuelcode", "fuel_moisture_class", "burn_source",
                    "burnday_source", "BSEV", "month", "season", "doy"]
        for col in int_cats:
            if col in user_df.columns and not pd.isna(user_df.loc[0, col]):
                user_df.loc[0, col] = int(round(user_df.loc[0, col]))

        return user_df.iloc[0].to_dict()
