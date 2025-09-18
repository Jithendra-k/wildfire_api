from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from google.cloud import bigquery
import pandas as pd

router = APIRouter()
bq_client = bigquery.Client()

TABLE = "code-for-planet.data_housee.wildfire_event_emissions_clean"


@router.get("/emissions")
def get_emissions(
        state: Optional[str] = Query(None),
        county: Optional[str] = Query(None),
        year: Optional[int] = Query(None),
        emission_intensity: Optional[str] = Query(None),
        size_category: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=5000)
):
    """Retrieve wildfire emission events with optional filters."""

    filters = []
    params = []

    if state:
        filters.append("UPPER(state) = UPPER(@state)")
        params.append(bigquery.ScalarQueryParameter("state", "STRING", state))

    if county and state:
        filters.append("UPPER(county) LIKE UPPER(@county)")
        params.append(bigquery.ScalarQueryParameter("county", "STRING", f"%{county}%"))

    if year:
        filters.append("year = @year")
        params.append(bigquery.ScalarQueryParameter("year", "INT64", year))

    if emission_intensity and emission_intensity.lower() in ['low', 'medium', 'high', 'very_high']:
        filters.append("emission_intensity = @emission_intensity")
        params.append(bigquery.ScalarQueryParameter("emission_intensity", "STRING", emission_intensity.lower()))

    if size_category and size_category.lower() in ['small', 'medium', 'large', 'very_large']:
        filters.append("size_category = @size_category")
        params.append(bigquery.ScalarQueryParameter("size_category", "STRING", size_category.lower()))

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

    # Separate queries for better reliability
    events_query = f"""
    SELECT 
        ROUND(lat, 6) as lat,
        ROUND(lon, 6) as lng,
        state,
        county,
        year,
        fire_type,
        ROUND(duration_days, 1) as duration_days,
        ROUND(spatial_extent_km, 2) as spatial_extent_km,
        ROUND(fire_size, 1) as fire_size,
        fire_size_original as fire_size_category,
        ROUND(emission_value, 2) as emission_value,
        ROUND(total_emissions, 2) as total_emissions,
        emission_intensity,
        size_category,
        ROUND(avg_eco2, 4) as co2,
        ROUND(avg_ech4, 4) as ch4,
        ROUND(avg_eco, 4) as co,
        ROUND(avg_epm2_5, 4) as pm2_5
    FROM `{TABLE}`
    {where_clause}
    ORDER BY emission_value DESC, total_emissions DESC
    LIMIT {limit}
    """

    try:
        job_config = bigquery.QueryJobConfig(query_parameters=params) if params else None
        df = bq_client.query(events_query, job_config=job_config).to_dataframe()

        if df.empty:
            return {
                "message": "No emission events found",
                "events": [],
                "count": 0,
                "summary": {}
            }

        # Convert to events list
        events = []
        for _, row in df.iterrows():
            events.append({
                "lat": row["lat"],
                "lng": row["lng"],
                "state": row["state"],
                "county": row["county"],
                "year": int(row["year"]),
                "fire_type": row["fire_type"],
                "duration_days": row["duration_days"],
                "spatial_extent_km": row["spatial_extent_km"],
                "fire_size": row["fire_size"],
                "fire_size_category": row["fire_size_category"],
                "emission_value": row["emission_value"],
                "total_emissions": row["total_emissions"],
                "emission_intensity": row["emission_intensity"],
                "size_category": row["size_category"],
                "emissions": {
                    "co2": row["co2"],
                    "ch4": row["ch4"],
                    "co": row["co"],
                    "pm2_5": row["pm2_5"]
                }
            })

        # Calculate summary
        summary = {
            "total_events": len(events),
            "total_fire_size": float(df["fire_size"].sum()),
            "avg_duration": float(df["duration_days"].mean()),
            "avg_spatial_extent": float(df["spatial_extent_km"].mean()),
            "total_emissions": float(df["total_emissions"].sum()),
            "max_emission_value": float(df["emission_value"].max()),
            "years_covered": sorted(df["year"].unique().tolist()),
            "states_covered": sorted(df["state"].unique().tolist())
        }

        return {
            "message": f"Successfully retrieved {len(events)} emission events",
            "events": events,
            "count": len(events),
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving emission data: {str(e)}")


@router.get("/emissions/summary")
def get_emissions_summary(
        state: Optional[str] = Query(None),
        year: Optional[int] = Query(None)
):
    """Get aggregated emissions summary by state and year."""

    filters = []
    params = []

    if state:
        filters.append("UPPER(state) = UPPER(@state)")
        params.append(bigquery.ScalarQueryParameter("state", "STRING", state))

    if year:
        filters.append("year = @year")
        params.append(bigquery.ScalarQueryParameter("year", "INT64", year))

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

    query = f"""
    SELECT 
        state,
        year,
        COUNT(*) as event_count,
        ROUND(AVG(duration_days), 2) as avg_duration,
        ROUND(AVG(spatial_extent_km), 2) as avg_spatial_extent,
        ROUND(SUM(avg_eco2), 2) as total_co2,
        ROUND(SUM(avg_ech4), 4) as total_ch4,
        ROUND(SUM(avg_eco), 2) as total_co,
        ROUND(SUM(avg_epm2_5), 4) as total_pm25,
        ROUND(SUM(emission_value), 2) as total_emission_value,
        ROUND(SUM(fire_size), 2) as total_fire_size,
        ROUND(SUM(total_emissions), 2) as total_all_emissions,
        COUNTIF(emission_intensity = 'very_high') as very_high_events,
        COUNTIF(size_category = 'large') as large_fires,
        COUNTIF(size_category = 'very_large') as very_large_fires
    FROM `{TABLE}`
    {where_clause}
    GROUP BY state, year 
    ORDER BY total_emission_value DESC 
    LIMIT 500
    """

    try:
        job_config = bigquery.QueryJobConfig(query_parameters=params) if params else None
        df = bq_client.query(query, job_config=job_config).to_dataframe()

        if df.empty:
            return {"message": "No summary data found", "data": [], "count": 0}

        data = df.to_dict('records')
        for row in data:
            row['total_emissions'] = {
                "co2": row.pop('total_co2'),
                "ch4": row.pop('total_ch4'),
                "co": row.pop('total_co'),
                "pm2_5": row.pop('total_pm25')
            }
            row['high_impact_events'] = {
                "very_high_emission": row.pop('very_high_events'),
                "large_fires": row.pop('large_fires'),
                "very_large_fires": row.pop('very_large_fires')
            }

        return {
            "message": f"Retrieved summary for {len(data)} state-year combinations",
            "data": data,
            "count": len(data)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving summary: {str(e)}")


@router.get("/emissions/states")
def get_available_states():
    """Get list of available states with emission statistics."""

    query = f"""
    SELECT 
        state,
        COUNT(*) as event_count,
        MIN(year) as first_year,
        MAX(year) as last_year,
        ROUND(SUM(total_emissions), 2) as total_emissions,
        ROUND(AVG(emission_value), 2) as avg_emission_value,
        COUNTIF(emission_intensity = 'very_high') as high_impact_events
    FROM `{TABLE}`
    GROUP BY state
    ORDER BY total_emissions DESC
    """

    try:
        df = bq_client.query(query).to_dataframe()

        states = []
        for _, row in df.iterrows():
            states.append({
                "state": row["state"],
                "event_count": int(row["event_count"]),
                "year_range": f"{int(row['first_year'])}-{int(row['last_year'])}",
                "total_emissions": row["total_emissions"],
                "avg_emission_value": row["avg_emission_value"],
                "high_impact_events": int(row["high_impact_events"])
            })

        return {"states": states, "count": len(states)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving states: {str(e)}")


@router.get("/emissions/counties")
def get_available_counties(state: str = Query(...)):
    """Get list of available counties for a specific state."""

    query = f"""
    SELECT 
        county,
        COUNT(*) as event_count,
        ROUND(SUM(total_emissions), 2) as total_emissions,
        ROUND(AVG(duration_days), 2) as avg_duration,
        COUNTIF(emission_intensity = 'very_high') as high_impact_events
    FROM `{TABLE}`
    WHERE UPPER(state) = UPPER(@state)
    GROUP BY county
    ORDER BY total_emissions DESC
    """

    try:
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("state", "STRING", state)]
        )
        df = bq_client.query(query, job_config=job_config).to_dataframe()

        counties = df.to_dict('records')
        for county in counties:
            county['event_count'] = int(county['event_count'])
            county['high_impact_events'] = int(county['high_impact_events'])

        return {"state": state, "counties": counties, "count": len(counties)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving counties: {str(e)}")


@router.get("/emissions/years")
def get_available_years():
    """Get list of available years with comprehensive statistics."""

    query = f"""
    SELECT 
        year,
        COUNT(*) as event_count,
        COUNT(DISTINCT state) as states_affected,
        ROUND(SUM(total_emissions), 2) as total_emissions,
        ROUND(AVG(duration_days), 2) as avg_duration,
        ROUND(SUM(fire_size), 2) as total_fire_size,
        COUNTIF(emission_intensity = 'very_high') as very_high_events,
        COUNTIF(size_category IN ('large', 'very_large')) as large_fires
    FROM `{TABLE}`
    GROUP BY year
    ORDER BY year DESC
    """

    try:
        df = bq_client.query(query).to_dataframe()

        years = df.to_dict('records')
        for year_data in years:
            year_data['event_count'] = int(year_data['event_count'])
            year_data['states_affected'] = int(year_data['states_affected'])
            year_data['very_high_events'] = int(year_data['very_high_events'])
            year_data['large_fires'] = int(year_data['large_fires'])

        year_values = [y["year"] for y in years]

        return {
            "years": years,
            "count": len(years),
            "range": {
                "min": min(year_values) if year_values else None,
                "max": max(year_values) if year_values else None
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving years: {str(e)}")


@router.get("/emissions/filters")
def get_available_filters():
    """Get all available filter options and endpoint information."""
    return {
        "filters": {
            "emission_intensity": ["low", "medium", "high", "very_high"],
            "size_category": ["small", "medium", "large", "very_large"],
            "state": "US state name (e.g., CALIFORNIA)",
            "county": "County name - requires state",
            "year": "Year (2003-2015)"
        },
        "endpoints": {
            "/emissions": "Get emission events with filters",
            "/emissions/summary": "Get aggregated summary by state/year",
            "/emissions/states": "Get available states",
            "/emissions/counties?state=X": "Get counties for state",
            "/emissions/years": "Get available years",
            "/emissions/filters": "Get filter options"
        }
    }