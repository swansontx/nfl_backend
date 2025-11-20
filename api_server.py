"""FastAPI server for NFL betting data.

Local API server that:
- Fetches data from external sources (nflverse, Odds API, ESPN)
- Stores in SQLite database (append-only for history)
- Serves data to MCP server via localhost

Run with: uvicorn api_server:app --reload --port 8000
"""

import os
import asyncio
from datetime import datetime
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
