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

# Import database
from backend.database.local_db import (
    init_database, get_database_status,
    OddsRepository, ProjectionsRepository, InjuriesRepository,
    GamesRepository, ResultsRepository, ValuePropsRepository,
    ModelRunsRepository
)

# Import ingestion modules
from backend.ingestion.fetch_prop_lines import PropLineFetcher
from backend.ingestion.fetch_injuries import InjuryFetcher

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


# ============ FETCH ENDPOINTS (append to DB) ============

@app.post("/fetch/odds", response_model=FetchResponse)
async def fetch_odds(
    week: int = Query(12, description="NFL week number"),
    season: int = Query(2024, description="NFL season"),
    background_tasks: BackgroundTasks = None
):
    """Fetch odds from The Odds API and store in database."""
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="ODDS_API_KEY not set")

    try:
        fetcher = PropLineFetcher(api_key)

        # Fetch from API
        output_dir = PROJECT_ROOT / "outputs" / "prop_lines"
        output_dir.mkdir(parents=True, exist_ok=True)

        result = fetcher.fetch_snapshot_with_movement(output_dir, week)

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
    season: int = Query(2024, description="NFL season")
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
    year: int = Query(2024, description="NFL season year"),
    include_all: bool = Query(True, description="Include all datasets")
):
    """Fetch nflverse data (play-by-play, stats, rosters)."""
    try:
        from backend.ingestion.fetch_nflverse import fetch_nflverse

        output_dir = PROJECT_ROOT / "inputs" / "nflverse"
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
    week: int = Query(12, description="NFL week"),
    year: int = Query(2024, description="NFL season"),
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
    injuries_max_age_hours: int = Query(12, description="Max age for injury data in hours"),
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

        # Check nflverse files
        nflverse_dir = PROJECT_ROOT / "inputs" / "nflverse"
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
    week: int = Query(12, description="NFL week"),
    year: int = Query(2024, description="NFL season"),
    odds_max_age_hours: int = Query(4, description="Max age for odds before refresh"),
    injuries_max_age_hours: int = Query(12, description="Max age for injuries before refresh"),
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
    season: int = Query(2024, description="NFL season")
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
    season: int = Query(2024, description="NFL season")
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
    season: int = Query(2024, description="NFL season")
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
    week: int = Query(12, description="NFL week"),
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
            refresh_result = await auto_refresh_endpoint(week=week, year=2024)
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
