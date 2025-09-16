import math
import pandas as pd
from google.cloud import bigquery

client = bigquery.Client()

def fetch_risk_events(state=None, county=None, season=None, doy=None, risk=None, limit=None):
    """
    Fetch wildfire events with adaptive sample size.
    If limit is not provided, compute n adaptively based on risk + doy.
    """
    # --- Step 1: adaptive limit ---
    if not limit:
        # normalize risk (0–10 scale assumed)
        r_norm = min((risk or 5) / 10.0, 1.0)
        # seasonality curve (peaks mid-year)
        s_norm = (math.sin(2 * math.pi * (doy or 180) / 365.0) + 1) / 2
        base, max_n = 50, 1000
        alpha, beta = 0.6, 0.4
        limit = base + int((alpha * r_norm + beta * s_norm) * (max_n - base))
        limit = min(max_n, max(base, limit))

    # --- Step 2: query ---
    query = """
    SELECT latitude, longitude, state, county, risk, season, doy
    FROM `code-for-planet.data_housee.featured_data_risk_csv`
    WHERE risk IS NOT NULL
    """
    if state:
        query += " AND state = @state"
    if county:
        query += " AND county = @county"
    if season:
        query += " AND season = @season"
    if doy:
        query += " AND doy BETWEEN @doy - 7 AND @doy + 7"

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("state", "STRING", state),
            bigquery.ScalarQueryParameter("county", "STRING", county),
            bigquery.ScalarQueryParameter("season", "INT64", season),
            bigquery.ScalarQueryParameter("doy", "INT64", doy),
        ]
    )

    df = client.query(query, job_config=job_config).to_dataframe()

    # --- Step 3: downsample if needed ---
    if len(df) > limit:
        bins = [0, 3, 6, 10]
        df["risk_bin"] = pd.cut(df["risk"], bins=bins, labels=["low", "med", "high"], include_lowest=True)
        df = (
            df.groupby("risk_bin", group_keys=False, observed=False)
            .apply(lambda x: x.sample(n=min(limit // 3, len(x)), random_state=42), include_groups=False)
        )

    return df[["latitude", "longitude", "state", "county", "risk"]].to_dict(orient="records")
