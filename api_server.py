"""FastAPI server for NFL betting data.

Local API server that:
- Fetches data from external sources (nflverse, Odds API, ESPN)
- Stores in SQLite database (append-only for history)
- Serves data to MCP server via localhost

Run with: uvicorn api_server:app --reload --port 8000
"""

import os
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel

# Import dynamic season/week detection
from backend.utils.nfl_calendar import get_current_season, get_current_week

# Get current season and week dynamically
CURRENT_SEASON = get_current_season()
CURRENT_WEEK = get_current_week()

# Import database
from backend.database.local_db import (
    init_database, get_database_status,
    OddsRepository, ProjectionsRepository, InjuriesRepository,
    GamesRepository, ResultsRepository, ValuePropsRepository,
    ModelRunsRepository, PlayerStatsRepository, TeamStatsRepository,
    RostersRepository, SchedulesRepository
)

# Import ingestion modules
from backend.ingestion.fetch_prop_lines import PropLineFetcher
from backend.ingestion.fetch_injuries import InjuryFetcher

# Import external APIs
from backend.api.external_apis import open_meteo_api

# Project root
PROJECT_ROOT = Path(__file__).parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    init_database()
    yield


app = FastAPI(
    title="NFL Betting API",
    description="Local API for NFL prop betting data",
    version="1.0.0",
    lifespan=lifespan
)


# ============ RESPONSE MODELS ============

class StatusResponse(BaseModel):
    status: str
    database: dict
    timestamp: str


class FetchResponse(BaseModel):
    success: bool
    source: str
    records_inserted: int
    message: str
    timestamp: str


# ============ STATUS ENDPOINTS ============

@app.get("/", response_model=StatusResponse)
async def root():
    """API status and database overview."""
    return StatusResponse(
        status="running",
        database=get_database_status(),
        timestamp=datetime.now().isoformat()
    )


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ============ WEATHER ENDPOINTS ============

@app.get("/weather/impact")
async def get_weather_impact(
    game_id: str = Query(..., description="Game ID in format {season}_{week}_{away}_{home}"),
    game_time: Optional[str] = Query(None, description="Game time in ISO format")
):
    """Get weather forecast with prop impact analysis for a game.

    Uses Open-Meteo (FREE, no API key needed) to get:
    - Temperature, wind, precipitation forecast
    - Impact analysis on passing, rushing, kicking props
    - Expected total points adjustment
    - Notes on weather conditions

    Essential for outdoor games - skip for dome games.
    """
    try:
        weather = open_meteo_api.get_weather_for_game(game_id, game_time)

        return {
            "game_id": game_id,
            "weather": weather,
            "betting_notes": _generate_weather_betting_notes(weather),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching weather: {str(e)}")


def _generate_weather_betting_notes(weather: dict) -> list:
    """Generate betting notes based on weather conditions."""
    notes = []
    impact = weather.get('weather_impact', {})

    # Dome check
    if weather.get('is_dome'):
        notes.append("Indoor dome - weather is not a factor")
        return notes

    # Wind notes
    wind = weather.get('wind_speed', 0)
    gusts = weather.get('wind_gusts', 0)
    if gusts >= 30 or wind >= 20:
        notes.append(f"HIGH WIND ALERT: {wind} mph (gusts {gusts}) - AVOID long FGs and deep passing props")
        notes.append("Consider UNDER on passing yards props")
        notes.append("QB interceptions may increase")
    elif gusts >= 20 or wind >= 15:
        notes.append(f"Moderate winds ({wind} mph) may affect 45+ yard FGs")

    # Precipitation notes
    precip_chance = weather.get('precipitation_chance', 0)
    precip_mm = weather.get('precipitation_mm', 0)
    if precip_chance >= 60 or precip_mm >= 2:
        notes.append(f"Rain/Snow likely ({precip_chance}%) - rushing props may have value")
        notes.append("Completion percentage will likely decrease")

    # Temperature notes
    temp = weather.get('temperature', 70)
    if temp <= 32:
        notes.append(f"Freezing conditions ({temp}F) - ball handling issues possible")
        notes.append("Consider UNDER on game total")
    elif temp >= 90:
        notes.append(f"Hot weather ({temp}F) - fatigue factor in 4th quarter")

    # Total points adjustment
    total_adj = impact.get('total_adjustment', 0)
    if total_adj <= -2:
        notes.append(f"Weather suggests {abs(total_adj):.1f} fewer points than expected - lean UNDER")
    elif total_adj >= 1:
        notes.append(f"Favorable conditions may add {total_adj:.1f} points")

    if not notes:
        notes.append("Weather conditions are favorable - no significant impact expected")

    return notes


# ============ FETCH ENDPOINTS (append to DB) ============

@app.post("/fetch/odds", response_model=FetchResponse)
async def fetch_odds(
    week: int = Query(CURRENT_WEEK, description="NFL week number"),
    season: int = Query(CURRENT_SEASON, description="NFL season"),
    force: bool = Query(False, description="Force fetch even if recent data exists"),
    min_hours: float = Query(2.0, description="Minimum hours between fetches"),
    background_tasks: BackgroundTasks = None
):
    """Fetch odds from The Odds API and store in database.

    Uses smart time-based checking to avoid redundant API calls.
    Will skip fetching if data was retrieved within min_hours (default: 2 hours).
    Use force=True to bypass this check.
    """
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="ODDS_API_KEY not set")

    try:
        fetcher = PropLineFetcher(api_key)

        # Fetch from API with time-based check
        output_dir = PROJECT_ROOT / "outputs" / "prop_lines"
        output_dir.mkdir(parents=True, exist_ok=True)

        result = fetcher.fetch_snapshot_with_movement(output_dir, week, min_hours=min_hours, force=force)

        # Check if fetch was skipped
        if result.get('skipped'):
            return FetchResponse(
                success=True,
                source="ODDS_API",
                records_inserted=0,
                message=f"Skipped: {result.get('reason', 'Data is current')}",
                timestamp=result.get('snapshot_time', datetime.now().isoformat())
            )

        # Convert to list of dicts for database
        odds_data = []
        for game in result.get("raw_data", []):
            game_id = game.get("id", "")
            for bookmaker in game.get("bookmakers", []):
                book = bookmaker.get("key", "draftkings")
                for market in bookmaker.get("markets", []):
                    prop_type = market.get("key", "")
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name") in ["Over", "Under"]:
                            # Find matching Over/Under pair
                            pass
                        else:
                            player_name = outcome.get("description", "")
                            line = outcome.get("point", 0)
                            odds_data.append({
                                "game_id": game_id,
                                "player_id": f"{player_name.replace(' ', '_')}_{prop_type}",
                                "player_name": player_name,
                                "team": "",
                                "prop_type": prop_type,
                                "line": line,
                                "over_odds": outcome.get("price", -110) if outcome.get("name") == "Over" else -110,
                                "under_odds": -110,
                                "book": book
                            })

        # Insert into database
        inserted = OddsRepository.insert_snapshot(odds_data, week, season)

        return FetchResponse(
            success=True,
            source="ODDS_API",
            records_inserted=inserted,
            message=f"Fetched {result.get('props', 0)} props for week {week}",
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fetch/injuries", response_model=FetchResponse)
async def fetch_injuries(
    week: Optional[int] = Query(None, description="NFL week number"),
    season: int = Query(CURRENT_SEASON, description="NFL season")
):
    """Fetch injury reports from ESPN and store in database."""
    try:
        output_dir = PROJECT_ROOT / "inputs" / "injuries"
        output_dir.mkdir(parents=True, exist_ok=True)

        fetcher = InjuryFetcher()
        injuries_raw = fetcher.fetch_all_injuries(output_dir, week)

        # Convert to flat list for database
        injuries_data = []
        for team, players in injuries_raw.get("teams", {}).items():
            for player in players:
                injuries_data.append({
                    "player_name": player.get("name"),
                    "player_id": player.get("id", ""),
                    "team": team,
                    "position": player.get("position"),
                    "status": player.get("status"),
                    "injury_type": player.get("injury")
                })

        # Insert into database
        inserted = InjuriesRepository.insert_injuries(
            injuries_data, week or 0, season
        )

        return FetchResponse(
            success=True,
            source="ESPN_INJURIES",
            records_inserted=inserted,
            message=f"Fetched {len(injuries_data)} injury reports",
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fetch/nflverse")
async def fetch_nflverse(
    year: int = Query(CURRENT_SEASON, description="NFL season year"),
    include_all: bool = Query(True, description="Include all datasets")
):
    """Fetch nflverse data (play-by-play, stats, rosters).

    Downloads to inputs/ directory where all other code expects to find the files.
    """
    try:
        from backend.ingestion.fetch_nflverse import fetch_nflverse

        # Save directly to inputs/ where all other code expects files
        output_dir = PROJECT_ROOT / "inputs"
        cache_dir = PROJECT_ROOT / "cache"

        output_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        fetch_nflverse(year, output_dir, cache_dir, include_all)

        # List fetched files
        files = list(output_dir.glob(f"*{year}*.csv"))

        return {
            "success": True,
            "source": "NFLVERSE",
            "year": year,
            "files_fetched": len(files),
            "file_names": [f.name for f in files],
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fetch/all")
async def fetch_all(
    week: int = Query(CURRENT_WEEK, description="NFL week"),
    year: int = Query(CURRENT_SEASON, description="NFL season"),
    background_tasks: BackgroundTasks = None
):
    """Fetch all data sources."""
    results = {
        "odds": None,
        "injuries": None,
        "nflverse": None
    }

    # Odds
    try:
        odds_result = await fetch_odds(week, year)
        results["odds"] = "success"
    except Exception as e:
        results["odds"] = f"error: {str(e)}"

    # Injuries
    try:
        injury_result = await fetch_injuries(week, year)
        results["injuries"] = "success"
    except Exception as e:
        results["injuries"] = f"error: {str(e)}"

    # NFLverse (optional, can be slow)
    # try:
    #     nfl_result = await fetch_nflverse(year, True)
    #     results["nflverse"] = "success"
    # except Exception as e:
    #     results["nflverse"] = f"error: {str(e)}"

    return {
        "success": all(v == "success" for v in results.values() if v),
        "results": results,
        "week": week,
        "year": year,
        "timestamp": datetime.now().isoformat()
    }


# ============ AUTO-REFRESH ENDPOINTS ============

@app.get("/refresh/check")
async def check_data_freshness(
    odds_max_age_hours: int = Query(4, description="Max age for odds data in hours"),
    injuries_max_age_hours: int = Query(CURRENT_WEEK, description="Max age for injury data in hours"),
    nflverse_max_age_days: int = Query(1, description="Max age for nflverse data in days")
):
    """Check freshness of all data sources and recommend updates."""
    from backend.database.local_db import get_db

    now = datetime.now()
    freshness = {}
    recommendations = []

    with get_db() as conn:
        cursor = conn.cursor()

        # Check odds freshness
        cursor.execute("SELECT MAX(snapshot_time) FROM odds_snapshots")
        odds_last = cursor.fetchone()[0]
        if odds_last:
            odds_time = datetime.fromisoformat(odds_last.replace('Z', '+00:00').replace('+00:00', ''))
            odds_age_hours = (now - odds_time).total_seconds() / 3600
            freshness["odds"] = {
                "last_update": odds_last,
                "age_hours": round(odds_age_hours, 1),
                "is_stale": odds_age_hours > odds_max_age_hours
            }
            if odds_age_hours > odds_max_age_hours:
                recommendations.append("odds")
        else:
            freshness["odds"] = {"last_update": None, "is_stale": True}
            recommendations.append("odds")

        # Check injuries freshness
        cursor.execute("SELECT MAX(reported_at) FROM injuries")
        injuries_last = cursor.fetchone()[0]
        if injuries_last:
            injuries_time = datetime.fromisoformat(injuries_last.replace('Z', '+00:00').replace('+00:00', ''))
            injuries_age_hours = (now - injuries_time).total_seconds() / 3600
            freshness["injuries"] = {
                "last_update": injuries_last,
                "age_hours": round(injuries_age_hours, 1),
                "is_stale": injuries_age_hours > injuries_max_age_hours
            }
            if injuries_age_hours > injuries_max_age_hours:
                recommendations.append("injuries")
        else:
            freshness["injuries"] = {"last_update": None, "is_stale": True}
            recommendations.append("injuries")

        # Check projections freshness
        cursor.execute("SELECT MAX(generated_at) FROM projections")
        proj_last = cursor.fetchone()[0]
        if proj_last:
            freshness["projections"] = {
                "last_update": proj_last,
                "is_stale": False  # Projections are regenerated manually
            }
        else:
            freshness["projections"] = {"last_update": None, "is_stale": True}

        # Check nflverse files (in main inputs/ dir, not subdirectory)
        nflverse_dir = PROJECT_ROOT / "inputs"
        if nflverse_dir.exists():
            files = list(nflverse_dir.glob("*.csv"))
            if files:
                latest = max(files, key=lambda f: f.stat().st_mtime)
                nflverse_time = datetime.fromtimestamp(latest.stat().st_mtime)
                nflverse_age_days = (now - nflverse_time).total_seconds() / 86400
                freshness["nflverse"] = {
                    "last_update": nflverse_time.isoformat(),
                    "age_days": round(nflverse_age_days, 1),
                    "is_stale": nflverse_age_days > nflverse_max_age_days
                }
                if nflverse_age_days > nflverse_max_age_days:
                    recommendations.append("nflverse")
            else:
                freshness["nflverse"] = {"last_update": None, "is_stale": True}
                recommendations.append("nflverse")
        else:
            freshness["nflverse"] = {"last_update": None, "is_stale": True}
            recommendations.append("nflverse")

    return {
        "freshness": freshness,
        "stale_sources": recommendations,
        "needs_refresh": len(recommendations) > 0,
        "timestamp": now.isoformat()
    }


@app.post("/refresh/auto")
async def auto_refresh(
    week: int = Query(CURRENT_WEEK, description="NFL week"),
    year: int = Query(CURRENT_SEASON, description="NFL season"),
    odds_max_age_hours: int = Query(4, description="Max age for odds before refresh"),
    injuries_max_age_hours: int = Query(CURRENT_WEEK, description="Max age for injuries before refresh"),
    force: bool = Query(False, description="Force refresh even if data is fresh")
):
    """Automatically refresh stale data sources."""
    # Check freshness
    freshness_check = await check_data_freshness(
        odds_max_age_hours, injuries_max_age_hours
    )

    stale = freshness_check["stale_sources"] if not force else ["odds", "injuries"]
    results = {}

    # Refresh stale sources
    if "odds" in stale:
        try:
            odds_result = await fetch_odds(week, year)
            results["odds"] = "refreshed"
        except Exception as e:
            results["odds"] = f"error: {str(e)}"
    else:
        results["odds"] = "fresh"

    if "injuries" in stale:
        try:
            injuries_result = await fetch_injuries(week, year)
            results["injuries"] = "refreshed"
        except Exception as e:
            results["injuries"] = f"error: {str(e)}"
    else:
        results["injuries"] = "fresh"

    # Note: nflverse not auto-refreshed due to size
    if "nflverse" in stale:
        results["nflverse"] = "stale - refresh manually with /fetch/nflverse"

    refreshed_count = sum(1 for v in results.values() if v == "refreshed")

    return {
        "success": True,
        "refreshed_sources": refreshed_count,
        "results": results,
        "was_forced": force,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/refresh/schedule")
async def get_refresh_schedule():
    """Get recommended refresh schedule for data sources."""
    return {
        "recommended_schedule": {
            "odds": {
                "frequency": "Every 4 hours during game days",
                "reason": "Lines move frequently, especially close to game time"
            },
            "injuries": {
                "frequency": "Every 12 hours, more on Wed-Fri",
                "reason": "Injury reports typically update mid-week"
            },
            "nflverse": {
                "frequency": "Once per week after games complete",
                "reason": "Stats update after games, no need for frequent refresh"
            },
            "projections": {
                "frequency": "After each data refresh",
                "reason": "Re-run models with fresh data"
            }
        },
        "tips": [
            "Run /refresh/auto before betting sessions",
            "Fetch odds more frequently on game days (2-4 hours)",
            "Wednesday-Friday are key for injury report updates",
            "After Sunday games, fetch nflverse for updated stats"
        ]
    }


# ============ QUERY ENDPOINTS (for MCP tools) ============

@app.get("/odds/latest")
async def get_latest_odds(
    game_id: Optional[str] = None,
    player_name: Optional[str] = None,
    prop_type: Optional[str] = None
):
    """Get latest odds snapshot."""
    odds = OddsRepository.get_latest_odds(game_id, player_name, prop_type)
    return {
        "source": "LOCAL_DB",
        "count": len(odds),
        "odds": odds,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/odds/movement")
async def get_line_movement(
    player_name: str = Query(..., description="Player name"),
    prop_type: str = Query(..., description="Prop type"),
    days: int = Query(7, description="Days to look back")
):
    """Get line movement for a player/prop."""
    movement = OddsRepository.get_line_movement(player_name, prop_type, days)
    return {
        "source": "LOCAL_DB",
        "player": player_name,
        "prop_type": prop_type,
        "snapshots": len(movement),
        "movement_history": movement,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/odds/movers")
async def get_hot_movers(
    min_movement: float = Query(1.5, description="Minimum line movement"),
    hours: int = Query(48, description="Hours to look back")
):
    """Get props with significant line movement."""
    movers = OddsRepository.get_hot_movers(min_movement, hours)
    return {
        "source": "LOCAL_DB",
        "count": len(movers),
        "hot_movers": movers,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/projections/latest")
async def get_latest_projections(
    game_id: Optional[str] = None,
    player_name: Optional[str] = None,
    prop_type: Optional[str] = None
):
    """Get latest projections."""
    projections = ProjectionsRepository.get_latest_projections(
        game_id, player_name, prop_type
    )
    return {
        "source": "YOUR_MODEL",
        "count": len(projections),
        "projections": projections,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/projections/history")
async def get_projection_history(
    player_name: str = Query(..., description="Player name"),
    prop_type: str = Query(..., description="Prop type"),
    limit: int = Query(10, description="Number of records")
):
    """Get projection history for a player."""
    history = ProjectionsRepository.get_projection_history(
        player_name, prop_type, limit
    )
    return {
        "source": "YOUR_MODEL",
        "player": player_name,
        "prop_type": prop_type,
        "history": history,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/projections/store")
async def store_projections(
    projections: List[dict],
    week: int = Query(..., description="NFL week"),
    season: int = Query(CURRENT_SEASON, description="NFL season")
):
    """Store generated projections in database."""
    inserted = ProjectionsRepository.insert_projections(projections, week, season)
    return {
        "success": True,
        "records_inserted": inserted,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/injuries/latest")
async def get_latest_injuries(
    team: Optional[str] = None,
    status: Optional[str] = None
):
    """Get latest injury reports."""
    injuries = InjuriesRepository.get_latest_injuries(team, status)
    return {
        "source": "ESPN_INJURIES",
        "count": len(injuries),
        "injuries": injuries,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/injuries/history")
async def get_injury_history(
    player_name: str = Query(..., description="Player name"),
    weeks: int = Query(4, description="Weeks to look back")
):
    """Get injury history for a player."""
    history = InjuriesRepository.get_injury_history(player_name, weeks)
    return {
        "source": "ESPN_INJURIES",
        "player": player_name,
        "history": history,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/games")
async def get_games(
    week: Optional[int] = None,
    season: int = Query(CURRENT_SEASON, description="NFL season")
):
    """Get games/schedule."""
    games = GamesRepository.get_games(week, season)
    return {
        "source": "LOCAL_DB",
        "count": len(games),
        "games": games,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/games/store")
async def store_games(games: List[dict]):
    """Store/update games in database."""
    count = GamesRepository.upsert_games(games)
    return {
        "success": True,
        "games_stored": count,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/results")
async def get_results(
    game_id: Optional[str] = None,
    player_name: Optional[str] = None
):
    """Get actual results for backtesting."""
    results = ResultsRepository.get_results(game_id, player_name)
    return {
        "source": "LOCAL_DB",
        "count": len(results),
        "results": results,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/results/store")
async def store_results(results: List[dict]):
    """Store actual results."""
    count = ResultsRepository.insert_results(results)
    return {
        "success": True,
        "results_stored": count,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/value-props/history")
async def get_value_props_history(
    days: int = Query(7, description="Days to look back"),
    min_edge: float = Query(0, description="Minimum edge filter")
):
    """Get historical value props found."""
    props = ValuePropsRepository.get_value_props_history(days, min_edge)
    return {
        "source": "YOUR_MODEL",
        "count": len(props),
        "value_props": props,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/value-props/store")
async def store_value_props(
    props: List[dict],
    week: int = Query(..., description="NFL week"),
    season: int = Query(CURRENT_SEASON, description="NFL season")
):
    """Store value props found."""
    count = ValuePropsRepository.insert_value_props(props, week, season)
    return {
        "success": True,
        "props_stored": count,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model/runs")
async def get_model_runs(limit: int = Query(20, description="Number of runs")):
    """Get model training run history."""
    runs = ModelRunsRepository.get_all_runs(limit)
    return {
        "source": "LOCAL_DB",
        "count": len(runs),
        "runs": runs,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/model/log-run")
async def log_model_run(
    season: int,
    model_type: str,
    models_trained: int,
    prop_types: List[str],
    training_time: float,
    notes: str = ""
):
    """Log a model training run."""
    run_id = ModelRunsRepository.log_run(
        season, model_type, models_trained, prop_types, training_time, notes
    )
    return {
        "success": True,
        "run_id": run_id,
        "timestamp": datetime.now().isoformat()
    }


# ============ ANALYSIS ENDPOINTS ============

@app.get("/analysis/quick-props")
async def quick_props(
    min_edge: float = Query(5.0, description="Minimum edge %"),
    limit: int = Query(10, description="Number of props"),
    prop_types: Optional[str] = Query(None, description="Comma-separated prop types")
):
    """Quick value scan - find props with edge."""
    # Get latest odds and projections
    odds = OddsRepository.get_latest_odds()
    projections = ProjectionsRepository.get_latest_projections()

    if not odds or not projections:
        return {
            "source": "YOUR_MODEL",
            "error": "No data available",
            "message": "Fetch odds and generate projections first",
            "odds_count": len(odds),
            "projection_count": len(projections)
        }

    # Match odds with projections and calculate edge
    value_props = []
    proj_lookup = {
        (p['player_name'].lower(), p['prop_type']): p
        for p in projections
    }

    for odd in odds:
        key = (odd['player_name'].lower(), odd['prop_type'])
        if key in proj_lookup:
            proj = proj_lookup[key]

            # Calculate edge (simplified)
            line = odd['line']
            projection = proj['projection']
            prob_over = proj.get('hit_prob_over', 0.5)
            prob_under = proj.get('hit_prob_under', 0.5)

            # Implied probability from odds (-110 = 52.4%)
            implied_over = 110 / (110 + 100)
            implied_under = 110 / (110 + 100)

            edge_over = (prob_over - implied_over) * 100
            edge_under = (prob_under - implied_under) * 100

            best_edge = max(edge_over, edge_under)
            if best_edge >= min_edge:
                value_props.append({
                    "player": odd['player_name'],
                    "prop_type": odd['prop_type'],
                    "line": line,
                    "projection": round(projection, 1),
                    "edge_over": round(edge_over, 2),
                    "edge_under": round(edge_under, 2),
                    "best_edge": round(best_edge, 2),
                    "recommendation": "OVER" if edge_over > edge_under else "UNDER",
                    "confidence": round(max(prob_over, prob_under), 3)
                })

    # Filter by prop type if specified
    if prop_types:
        allowed = [t.strip() for t in prop_types.split(",")]
        value_props = [p for p in value_props if p['prop_type'] in allowed]

    # Sort by edge and limit
    value_props.sort(key=lambda x: x['best_edge'], reverse=True)
    value_props = value_props[:limit]

    return {
        "source": "YOUR_MODEL",
        "count": len(value_props),
        "min_edge_filter": min_edge,
        "props": value_props,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/analysis/game/{game_id}")
async def game_deep_dive(game_id: str):
    """Deep dive analysis for a specific game."""
    # Get all data for this game
    odds = OddsRepository.get_latest_odds(game_id=game_id)
    projections = ProjectionsRepository.get_latest_projections(game_id=game_id)

    # Parse game_id
    parts = game_id.split("_")
    if len(parts) >= 4:
        matchup = f"{parts[2]} @ {parts[3]}"
        week = parts[1]
    else:
        matchup = game_id
        week = "?"

    # Organize by player
    players = {}
    proj_lookup = {
        (p['player_name'].lower(), p['prop_type']): p
        for p in projections
    }

    for odd in odds:
        player = odd['player_name']
        if player not in players:
            players[player] = {"props": [], "best_edge": 0}

        key = (player.lower(), odd['prop_type'])
        proj = proj_lookup.get(key, {})

        if proj:
            edge = abs(proj.get('projection', odd['line']) - odd['line'])
            players[player]["props"].append({
                "prop_type": odd['prop_type'],
                "line": odd['line'],
                "projection": proj.get('projection'),
                "edge": round(edge, 2)
            })
            players[player]["best_edge"] = max(
                players[player]["best_edge"], edge
            )

    # Convert to list and sort
    player_list = [
        {"player": name, **data}
        for name, data in players.items()
    ]
    player_list.sort(key=lambda x: x['best_edge'], reverse=True)

    return {
        "source": "YOUR_MODEL",
        "game_id": game_id,
        "matchup": matchup,
        "week": week,
        "players": player_list[:15],
        "total_props": len(odds),
        "timestamp": datetime.now().isoformat()
    }


# ============ COMPREHENSIVE INTELLIGENCE ENDPOINTS ============

@app.get("/intelligence/matchup/{game_id}")
async def full_matchup_analysis(game_id: str):
    """
    COMPREHENSIVE matchup analysis combining ALL data sources:
    - Team injuries and their impact
    - All player projections with edges
    - Line movement signals
    - Situational factors
    - Correlated props for parlays
    """
    # Parse game_id
    parts = game_id.split("_")
    if len(parts) >= 4:
        season, week, away_team, home_team = parts[0], parts[1], parts[2], parts[3]
    else:
        return {"error": "Invalid game_id format. Use: YYYY_WK_AWAY_HOME"}

    # Get ALL relevant data
    odds = OddsRepository.get_latest_odds(game_id=game_id)
    projections = ProjectionsRepository.get_latest_projections(game_id=game_id)

    # Get injuries for both teams
    away_injuries = InjuriesRepository.get_latest_injuries(team=away_team)
    home_injuries = InjuriesRepository.get_latest_injuries(team=home_team)

    # Get line movement
    movers = OddsRepository.get_hot_movers(min_movement=1.0, hours=48)
    game_movers = [m for m in movers if m.get('game_id') == game_id]

    # Organize projections by team
    away_players = []
    home_players = []

    proj_lookup = {(p['player_name'].lower(), p['prop_type']): p for p in projections}

    for odd in odds:
        key = (odd['player_name'].lower(), odd['prop_type'])
        proj = proj_lookup.get(key, {})

        if proj:
            player_data = {
                "player": odd['player_name'],
                "team": odd.get('team', ''),
                "prop_type": odd['prop_type'],
                "line": odd['line'],
                "projection": proj.get('projection'),
                "std_dev": proj.get('std_dev'),
                "edge": round(abs(proj.get('projection', odd['line']) - odd['line']), 2),
                "recommendation": "OVER" if proj.get('projection', 0) > odd['line'] else "UNDER",
                "confidence": proj.get('hit_prob_over') if proj.get('projection', 0) > odd['line'] else proj.get('hit_prob_under')
            }

            # Determine team (simplified - would need roster data for accuracy)
            if odd.get('team') == away_team:
                away_players.append(player_data)
            else:
                home_players.append(player_data)

    # Sort by edge
    away_players.sort(key=lambda x: x['edge'], reverse=True)
    home_players.sort(key=lambda x: x['edge'], reverse=True)

    # Identify key injuries (OUT or DOUBTFUL)
    key_away_injuries = [i for i in away_injuries if i.get('status') in ['OUT', 'DOUBTFUL']]
    key_home_injuries = [i for i in home_injuries if i.get('status') in ['OUT', 'DOUBTFUL']]

    # Find correlated props for parlays
    correlations = []

    # QB + WR correlations
    away_qbs = [p for p in away_players if 'pass' in p['prop_type']]
    away_wrs = [p for p in away_players if 'rec' in p['prop_type']]
    if away_qbs and away_wrs:
        correlations.append({
            "type": "passing_stack",
            "team": away_team,
            "players": [away_qbs[0]['player'], away_wrs[0]['player']],
            "props": [away_qbs[0]['prop_type'], away_wrs[0]['prop_type']],
            "combined_edge": away_qbs[0]['edge'] + away_wrs[0]['edge'],
            "correlation": "positive"
        })

    home_qbs = [p for p in home_players if 'pass' in p['prop_type']]
    home_wrs = [p for p in home_players if 'rec' in p['prop_type']]
    if home_qbs and home_wrs:
        correlations.append({
            "type": "passing_stack",
            "team": home_team,
            "players": [home_qbs[0]['player'], home_wrs[0]['player']],
            "props": [home_qbs[0]['prop_type'], home_wrs[0]['prop_type']],
            "combined_edge": home_qbs[0]['edge'] + home_wrs[0]['edge'],
            "correlation": "positive"
        })

    return {
        "source": "COMPREHENSIVE_ANALYSIS",
        "game_id": game_id,
        "matchup": f"{away_team} @ {home_team}",
        "week": week,

        "injury_impact": {
            away_team: {
                "key_out": [{"player": i['player_name'], "position": i['position'], "injury": i['injury_type']}
                           for i in key_away_injuries[:5]],
                "total_injuries": len(away_injuries)
            },
            home_team: {
                "key_out": [{"player": i['player_name'], "position": i['position'], "injury": i['injury_type']}
                           for i in key_home_injuries[:5]],
                "total_injuries": len(home_injuries)
            }
        },

        "value_props": {
            away_team: away_players[:10],
            home_team: home_players[:10]
        },

        "line_movement_signals": game_movers,

        "parlay_correlations": correlations,

        "summary": {
            "total_props_analyzed": len(odds),
            "props_with_edge": len([p for p in away_players + home_players if p['edge'] >= 3]),
            "sharp_movers": len(game_movers)
        },

        "timestamp": datetime.now().isoformat()
    }


@app.get("/intelligence/daily-brief")
async def get_daily_betting_brief(
    week: int = Query(CURRENT_WEEK, description="NFL week"),
    min_edge: float = Query(3.0, description="Minimum edge for top props"),
    auto_refresh: bool = Query(True, description="Auto-refresh stale data first")
):
    """
    DAILY BETTING INTELLIGENCE - Everything you need before betting:
    - Auto-refreshes stale data
    - Top value props across all games
    - Key injuries league-wide
    - Sharp line movements
    - Best parlay candidates
    """
    results = {"data_refreshed": False}

    # Auto-refresh if requested
    if auto_refresh:
        freshness = await check_data_freshness()
        if freshness["needs_refresh"]:
            refresh_result = await auto_refresh_endpoint(week=week, year=2025)
            results["data_refreshed"] = True
            results["refresh_details"] = refresh_result

    # Get all latest data
    all_odds = OddsRepository.get_latest_odds()
    all_projections = ProjectionsRepository.get_latest_projections()
    all_injuries = InjuriesRepository.get_latest_injuries()
    all_movers = OddsRepository.get_hot_movers(min_movement=1.5, hours=48)

    # Find top value props
    top_props = []
    proj_lookup = {(p['player_name'].lower(), p['prop_type']): p for p in all_projections}

    for odd in all_odds:
        key = (odd['player_name'].lower(), odd['prop_type'])
        proj = proj_lookup.get(key)

        if proj:
            edge = abs(proj.get('projection', odd['line']) - odd['line'])
            if edge >= min_edge:
                top_props.append({
                    "player": odd['player_name'],
                    "game_id": odd.get('game_id', ''),
                    "prop_type": odd['prop_type'],
                    "line": odd['line'],
                    "projection": proj.get('projection'),
                    "edge": round(edge, 2),
                    "side": "OVER" if proj.get('projection', 0) > odd['line'] else "UNDER",
                    "confidence": proj.get('hit_prob_over') if proj.get('projection', 0) > odd['line'] else proj.get('hit_prob_under')
                })

    top_props.sort(key=lambda x: x['edge'], reverse=True)

    # Key injuries (OUT only)
    key_injuries = [i for i in all_injuries if i.get('status') == 'OUT']
    injuries_by_team = {}
    for inj in key_injuries:
        team = inj.get('team', 'UNK')
        if team not in injuries_by_team:
            injuries_by_team[team] = []
        injuries_by_team[team].append({
            "player": inj['player_name'],
            "position": inj['position'],
            "injury": inj['injury_type']
        })

    # Sharp action signals (big movers)
    sharp_signals = [m for m in all_movers if abs(m.get('movement', 0)) >= 2.0]

    return {
        "source": "DAILY_INTELLIGENCE",
        "week": week,
        "generated_at": datetime.now().isoformat(),

        "top_value_props": top_props[:15],

        "key_injuries_by_team": injuries_by_team,

        "sharp_line_movement": [
            {
                "player": m['player_name'],
                "prop": m['prop_type'],
                "movement": m['movement'],
                "direction": "UP" if m['movement'] > 0 else "DOWN",
                "signal": "Sharp money likely"
            }
            for m in sharp_signals[:10]
        ],

        "parlay_candidates": top_props[:6],  # Top edge props for parlays

        "summary": {
            "total_props_analyzed": len(all_odds),
            "props_with_edge": len(top_props),
            "teams_with_key_injuries": len(injuries_by_team),
            "sharp_signals": len(sharp_signals)
        },

        "action_items": [
            f"Found {len(top_props)} props with {min_edge}%+ edge",
            f"{len(injuries_by_team)} teams have key players OUT",
            f"{len(sharp_signals)} props showing sharp action"
        ]
    }


# Alias for the auto_refresh function to avoid naming conflict
async def auto_refresh_endpoint(week: int, year: int):
    return await auto_refresh(week=week, year=year)


@app.get("/intelligence/player/{player_name}")
async def player_full_outlook(
    player_name: str,
    include_history: bool = Query(True, description="Include projection history")
):
    """
    COMPLETE PLAYER OUTLOOK - Everything about one player:
    - All current props with projections
    - Injury status and history
    - Projection trends over time
    - Line movement on their props
    - Usage/efficiency metrics
    """
    # Get all props for this player
    odds = OddsRepository.get_latest_odds(player_name=player_name)
    projections = ProjectionsRepository.get_latest_projections(player_name=player_name)
    injuries = InjuriesRepository.get_injury_history(player_name, weeks=4)

    # Get projection history for main props
    proj_history = {}
    if include_history and projections:
        for prop_type in set(p['prop_type'] for p in projections):
            history = ProjectionsRepository.get_projection_history(
                player_name, prop_type, limit=5
            )
            if history:
                proj_history[prop_type] = history

    # Match odds with projections
    props_analysis = []
    proj_lookup = {p['prop_type']: p for p in projections}

    for odd in odds:
        proj = proj_lookup.get(odd['prop_type'], {})

        # Check for line movement
        movement = OddsRepository.get_line_movement(
            player_name, odd['prop_type'], days=3
        )

        line_change = 0
        if len(movement) >= 2:
            line_change = movement[-1]['line'] - movement[0]['line']

        props_analysis.append({
            "prop_type": odd['prop_type'],
            "line": odd['line'],
            "projection": proj.get('projection'),
            "std_dev": proj.get('std_dev'),
            "edge": round(abs(proj.get('projection', odd['line']) - odd['line']), 2) if proj else 0,
            "side": "OVER" if proj.get('projection', odd['line']) > odd['line'] else "UNDER",
            "line_movement_3d": round(line_change, 1),
            "confidence": proj.get('hit_prob_over') if proj.get('projection', 0) > odd['line'] else proj.get('hit_prob_under')
        })

    props_analysis.sort(key=lambda x: x['edge'], reverse=True)

    # Current injury status
    current_injury = None
    if injuries:
        latest = injuries[0]
        current_injury = {
            "status": latest['status'],
            "injury": latest['injury_type'],
            "reported": latest['reported_at']
        }

    return {
        "source": "PLAYER_OUTLOOK",
        "player": player_name,
        "generated_at": datetime.now().isoformat(),

        "injury_status": current_injury,
        "injury_history": injuries[:5] if injuries else [],

        "current_props": props_analysis,

        "projection_trends": proj_history,

        "best_bet": props_analysis[0] if props_analysis else None,

        "summary": {
            "total_props": len(props_analysis),
            "props_with_edge": len([p for p in props_analysis if p['edge'] >= 3]),
            "is_injured": current_injury is not None and current_injury['status'] in ['OUT', 'DOUBTFUL']
        }
    }


# ============ STATS ENDPOINTS (for general knowledge queries) ============

@app.get("/stats/player/{player_name}")
async def get_player_stats(
    player_name: str,
    season: int = Query(CURRENT_SEASON, description="NFL season")
):
    """
    FULL PLAYER STATS - Like ESPN player page:
    - Season totals
    - Weekly breakdowns
    - Bio information
    - All relevant stats
    """
    # Get season totals
    totals = PlayerStatsRepository.get_player_season_totals(player_name, season)

    # Get weekly stats
    weekly = PlayerStatsRepository.get_player_stats(player_name, season)

    # Get player bio
    bio = RostersRepository.get_player_info(player_name)

    if not totals and not weekly and not bio:
        return {
            "source": "LOCAL_DB",
            "player": player_name,
            "error": "Player not found",
            "message": "Player stats not in database. Run /populate/stats first."
        }

    return {
        "source": "LOCAL_DB",
        "player": player_name,
        "season": season,

        "bio": {
            "team": bio.get('team') if bio else totals.get('team') if totals else None,
            "position": bio.get('position') if bio else totals.get('position') if totals else None,
            "jersey": bio.get('jersey_number') if bio else None,
            "height": bio.get('height') if bio else None,
            "weight": bio.get('weight') if bio else None,
            "college": bio.get('college') if bio else None,
            "years_exp": bio.get('years_exp') if bio else None
        } if bio else None,

        "season_totals": totals,

        "weekly_stats": weekly,

        "timestamp": datetime.now().isoformat()
    }


@app.get("/stats/team/{team}")
async def get_team_stats(
    team: str,
    season: int = Query(CURRENT_SEASON, description="NFL season")
):
    """
    FULL TEAM STATS - Team profile and stats:
    - Team record and rankings
    - Key players by position
    - Roster information
    """
    # Get team stats
    stats = TeamStatsRepository.get_team_stats(team, season)

    # Get team players
    players = PlayerStatsRepository.get_team_players(team, season)

    # Get roster
    roster = RostersRepository.get_team_roster(team, season)

    # Get team schedule
    schedule = SchedulesRepository.get_team_schedule(team, season)

    return {
        "source": "LOCAL_DB",
        "team": team,
        "season": season,

        "team_stats": stats,

        "key_players": {
            "offense": [p for p in players if p['position'] in ['QB', 'RB', 'WR', 'TE']][:10],
            "all": players[:20]
        },

        "roster_count": len(roster),
        "roster_by_position": _group_roster_by_position(roster) if roster else {},

        "schedule": {
            "total_games": len(schedule),
            "completed": len([g for g in schedule if g.get('home_score') is not None]),
            "upcoming": [g for g in schedule if g.get('home_score') is None][:4],
            "results": [g for g in schedule if g.get('home_score') is not None][-4:]
        },

        "timestamp": datetime.now().isoformat()
    }


def _group_roster_by_position(roster):
    """Helper to group roster by position."""
    grouped = {}
    for player in roster:
        pos = player.get('position', 'UNK')
        if pos not in grouped:
            grouped[pos] = []
        grouped[pos].append({
            "name": player['player_name'],
            "number": player.get('jersey_number'),
            "depth": player.get('depth_chart_position')
        })
    return grouped


@app.get("/stats/leaders/{stat_type}")
async def get_league_leaders(
    stat_type: str,
    season: int = Query(CURRENT_SEASON, description="NFL season"),
    limit: int = Query(20, description="Number of leaders")
):
    """
    LEAGUE LEADERS - Top players in each stat category:
    - passing_yards, passing_tds
    - rushing_yards, rushing_tds
    - receiving_yards, receiving_tds, receptions
    - fantasy, fantasy_ppr
    """
    valid_stats = [
        'passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds',
        'receiving_yards', 'receiving_tds', 'receptions', 'fantasy', 'fantasy_ppr'
    ]

    if stat_type not in valid_stats:
        return {
            "error": f"Invalid stat_type. Use one of: {', '.join(valid_stats)}"
        }

    leaders = PlayerStatsRepository.get_league_leaders(stat_type, season, limit)

    return {
        "source": "LOCAL_DB",
        "stat_type": stat_type,
        "season": season,
        "leaders": [
            {
                "rank": i + 1,
                "player": l['player_name'],
                "team": l['team'],
                "position": l['position'],
                "value": l['value']
            }
            for i, l in enumerate(leaders)
        ],
        "timestamp": datetime.now().isoformat()
    }


@app.get("/stats/schedule")
async def get_schedule(
    season: int = Query(CURRENT_SEASON, description="NFL season"),
    week: Optional[int] = Query(None, description="Specific week")
):
    """Get full season schedule."""
    schedule = SchedulesRepository.get_schedule(season, week)

    return {
        "source": "LOCAL_DB",
        "season": season,
        "week": week,
        "games": schedule,
        "count": len(schedule),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/stats/rankings")
async def get_team_rankings(season: int = Query(CURRENT_SEASON, description="NFL season")):
    """Get all teams ranked by record."""
    teams = TeamStatsRepository.get_all_teams(season)

    return {
        "source": "LOCAL_DB",
        "season": season,
        "standings": [
            {
                "rank": i + 1,
                "team": t['team'],
                "wins": t['wins'],
                "losses": t['losses'],
                "ties": t.get('ties', 0),
                "points_scored": t.get('points_scored'),
                "points_allowed": t.get('points_allowed')
            }
            for i, t in enumerate(teams)
        ],
        "timestamp": datetime.now().isoformat()
    }


# ============ DATA POPULATION ENDPOINTS ============

@app.post("/populate/stats")
async def populate_player_stats(
    season: int = Query(CURRENT_SEASON, description="NFL season"),
    week: Optional[int] = Query(None, description="Specific week to load")
):
    """
    Populate player stats from nflverse data files.
    Loads weekly stats for all players.
    """
    import pandas as pd

    stats_file = PROJECT_ROOT / "inputs" / f"player_stats_{season}.csv"
    if not stats_file.exists():
        # Try combined file
        stats_file = PROJECT_ROOT / "inputs" / f"player_stats_2024_{season}.csv"

    if not stats_file.exists():
        return {
            "error": f"Stats file not found: {stats_file}",
            "message": "Run /fetch/nflverse first to download stats"
        }

    try:
        df = pd.read_csv(stats_file)

        # Filter by week if specified
        if week:
            df = df[df['week'] == week]

        stats_data = []
        for _, row in df.iterrows():
            stats_data.append({
                'player_id': row.get('player_id'),
                'player_name': row.get('player_name') or row.get('player_display_name'),
                'team': row.get('recent_team') or row.get('team'),
                'position': row.get('position'),
                'season': int(row.get('season', season)),
                'week': int(row.get('week', 0)),
                'games_played': 1,
                'pass_attempts': row.get('attempts'),
                'pass_completions': row.get('completions'),
                'pass_yards': row.get('passing_yards'),
                'pass_tds': row.get('passing_tds'),
                'interceptions': row.get('interceptions'),
                'sacks': row.get('sacks'),
                'sack_yards': row.get('sack_yards'),
                'pass_rating': row.get('passer_rating'),
                'rush_attempts': row.get('carries'),
                'rush_yards': row.get('rushing_yards'),
                'rush_tds': row.get('rushing_tds'),
                'rush_yards_per_attempt': row.get('rushing_yards') / row.get('carries') if row.get('carries') else None,
                'targets': row.get('targets'),
                'receptions': row.get('receptions'),
                'rec_yards': row.get('receiving_yards'),
                'rec_tds': row.get('receiving_tds'),
                'yards_per_reception': row.get('receiving_yards') / row.get('receptions') if row.get('receptions') else None,
                'fantasy_points': row.get('fantasy_points'),
                'fantasy_points_ppr': row.get('fantasy_points_ppr'),
                'air_yards': row.get('receiving_air_yards'),
                'yards_after_catch': row.get('receiving_yards_after_catch'),
                'epa': row.get('receiving_epa') or row.get('rushing_epa'),
                'cpoe': None,
                'snap_pct': None,
                'route_participation': None
            })

        count = PlayerStatsRepository.upsert_player_stats(stats_data)

        return {
            "success": True,
            "source": "NFLVERSE",
            "records_loaded": count,
            "season": season,
            "week": week,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/populate/schedule")
async def populate_schedule(season: int = Query(CURRENT_SEASON, description="NFL season")):
    """Populate full season schedule from nflverse."""
    import pandas as pd

    # Look for schedule file
    schedule_file = None
    for pattern in [f"schedules.csv", f"schedule_{season}.csv", "schedules_*.csv"]:
        files = list((PROJECT_ROOT / "inputs").glob(pattern))
        if files:
            schedule_file = files[0]
            break

    if not schedule_file:
        # Try fetching directly from nflverse
        try:
            from backend.ingestion.fetch_nflverse_schedules import fetch_schedule
            output_dir = PROJECT_ROOT / "inputs"
            fetch_schedule(season, output_dir)
            schedule_file = output_dir / f"schedule_{season}.csv"
        except Exception as e:
            return {
                "error": f"Schedule file not found and fetch failed: {str(e)}",
                "message": "Download schedule data first"
            }

    try:
        df = pd.read_csv(schedule_file)

        # Filter to season
        if 'season' in df.columns:
            df = df[df['season'] == season]

        schedules = df.to_dict('records')
        count = SchedulesRepository.upsert_schedules(schedules)

        return {
            "success": True,
            "source": "NFLVERSE",
            "games_loaded": count,
            "season": season,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/populate/rosters")
async def populate_rosters(
    season: int = Query(CURRENT_SEASON, description="NFL season"),
    week: int = Query(CURRENT_WEEK, description="Week for roster snapshot")
):
    """Populate rosters from nflverse."""
    import pandas as pd

    roster_file = PROJECT_ROOT / "inputs" / f"weekly_rosters_{season}.csv"
    if not roster_file.exists():
        roster_file = PROJECT_ROOT / "inputs" / "weekly_rosters.csv"

    if not roster_file.exists():
        return {
            "error": "Roster file not found",
            "message": "Run /fetch/nflverse first"
        }

    try:
        df = pd.read_csv(roster_file)

        # Filter to season/week
        if 'season' in df.columns:
            df = df[df['season'] == season]
        if 'week' in df.columns:
            df = df[df['week'] == week]

        rosters = []
        for _, row in df.iterrows():
            rosters.append({
                'player_id': row.get('player_id') or row.get('gsis_id'),
                'player_name': row.get('player_name') or row.get('full_name'),
                'team': row.get('team'),
                'position': row.get('position'),
                'jersey_number': row.get('jersey_number'),
                'status': row.get('status'),
                'height': row.get('height'),
                'weight': row.get('weight'),
                'birth_date': row.get('birth_date'),
                'college': row.get('college'),
                'years_exp': row.get('years_exp'),
                'season': season,
                'week': week,
                'depth_chart_position': row.get('depth_chart_position')
            })

        count = RostersRepository.upsert_rosters(rosters)

        return {
            "success": True,
            "source": "NFLVERSE",
            "players_loaded": count,
            "season": season,
            "week": week,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/populate/all")
async def populate_all_data(
    season: int = Query(CURRENT_SEASON, description="NFL season"),
    week: int = Query(CURRENT_WEEK, description="Current week"),
    fetch_first: bool = Query(False, description="Fetch from nflverse first"),
    include_odds: bool = Query(True, description="Fetch DraftKings odds from OddsAPI")
):
    """
    POPULATE ALL DATA - Load complete database for season:
    - Season schedule
    - Player stats (all weeks)
    - Current rosters
    - Injuries
    - DraftKings odds (for betting analysis)

    This is the recommended way to initialize the database.
    """
    results = {}

    # Optionally fetch data first
    if fetch_first:
        try:
            nfl_result = await fetch_nflverse(season, include_all=True)
            results["fetch"] = "success"
        except Exception as e:
            results["fetch"] = f"error: {str(e)}"

    # Populate schedule
    try:
        schedule_result = await populate_schedule(season)
        results["schedule"] = f"loaded {schedule_result.get('games_loaded', 0)} games"
    except Exception as e:
        results["schedule"] = f"error: {str(e)}"

    # Populate player stats
    try:
        stats_result = await populate_player_stats(season)
        results["player_stats"] = f"loaded {stats_result.get('records_loaded', 0)} records"
    except Exception as e:
        results["player_stats"] = f"error: {str(e)}"

    # Populate rosters
    try:
        roster_result = await populate_rosters(season, week)
        results["rosters"] = f"loaded {roster_result.get('players_loaded', 0)} players"
    except Exception as e:
        results["rosters"] = f"error: {str(e)}"

    # Populate injuries
    try:
        injury_result = await fetch_injuries(week, season)
        results["injuries"] = f"fetched {injury_result.records_inserted} injuries"
    except Exception as e:
        results["injuries"] = f"error: {str(e)}"

    # Fetch DraftKings odds from OddsAPI
    if include_odds:
        try:
            odds_result = await fetch_odds(week, season)
            results["odds"] = f"fetched {odds_result.records_inserted} prop lines"
        except Exception as e:
            results["odds"] = f"error: {str(e)}"

    return {
        "success": True,
        "results": results,
        "season": season,
        "week": week,
        "timestamp": datetime.now().isoformat(),
        "next_steps": [
            "Check /stats/leaders/passing_yards for QB leaders",
            "Check /stats/team/KC for team profile",
            "Check /stats/schedule for full season schedule",
            "Check /odds/latest for current prop lines",
            "Check /odds/movers for line movement signals"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
