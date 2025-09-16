from fastapi import APIRouter, Query
from typing import Optional
from app.bigquery_utils import fetch_risk_events

router = APIRouter()

@router.get("/risk-heatmap")
def get_risk_heatmap(
    state: Optional[str] = Query(None, description="State name"),
    county: Optional[str] = Query(None, description="County name"),
    season: Optional[int] = Query(None, description="Season (1=Winter,2=Spring,3=Summer,4=Fall)"),
    doy: Optional[int] = Query(None, description="Day of year (1-365)"),
    limit: int = Query(5000, description="Max number of events to return")
):
    """
    Returns wildfire events (lat/lon + risk) filtered by state/county/season/doy.
    Automatically adapts result size to historical density.
    """
    events = fetch_risk_events(state, county, season, doy, limit)
    return {"events": events, "count": len(events)}
