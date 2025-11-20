"""MCP Server for NFL Prop Betting System.

Run with: python mcp_server.py
Then add to Claude Desktop config or use with Claude Code.

This exposes your betting system as tools Claude can call naturally.
All tools return YOUR MODEL'S data, not generic analysis.
"""

import json
import asyncio
import os
from typing import Optional
from datetime import datetime
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Import your backend modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from backend.api.prop_analyzer import prop_analyzer, PropLine, PropProjection
from backend.api.model_loader import model_loader
from backend.api.odds_api import odds_api
from backend.config import settings
from backend.ingestion.fetch_prop_lines import PropLineFetcher
from backend.ingestion.fetch_injuries import InjuryFetcher

# Create MCP server
server = Server("nfl-betting")

# Project root for paths
PROJECT_ROOT = Path(__file__).parent


def add_source_label(data: dict, source: str = "YOUR_MODEL") -> dict:
    """Add source label to output data."""
    return {
        "_source": source,
        "_generated_at": datetime.now().isoformat(),
        **data
    }


@server.list_tools()
async def list_tools():
    """List available NFL betting tools."""
    return [
        # ========== DATA FETCH TOOLS ==========
        Tool(
            name="fetch_nflverse_data",
            description="Fetch fresh NFL data from nflverse (play-by-play, stats, rosters, snap counts). Use this to update your training data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "year": {
                        "type": "integer",
                        "description": "NFL season year (e.g., 2024)",
                        "default": 2024
                    },
                    "include_all": {
                        "type": "boolean",
                        "description": "Include all datasets (PFR advanced, snap counts, depth charts)",
                        "default": True
                    }
                }
            }
        ),
        Tool(
            name="fetch_injuries",
            description="Fetch current injury reports from ESPN. Returns injuries by team and status.",
            inputSchema={
                "type": "object",
                "properties": {
                    "week": {
                        "type": "integer",
                        "description": "NFL week number (optional, defaults to current)"
                    }
                }
            }
        ),
        Tool(
            name="fetch_dk_odds",
            description="Fetch fresh DraftKings prop odds from The Odds API. Captures line movement and hot movers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "week": {
                        "type": "integer",
                        "description": "NFL week number"
                    }
                }
            }
        ),
        Tool(
            name="sync_all_data",
            description="Sync all data sources at once (nflverse, injuries, odds). Use before generating fresh projections.",
            inputSchema={
                "type": "object",
                "properties": {
                    "year": {
                        "type": "integer",
                        "description": "NFL season year",
                        "default": 2024
                    },
                    "week": {
                        "type": "integer",
                        "description": "NFL week number"
                    }
                }
            }
        ),

        # ========== MODEL TRAINING & PREDICTION TOOLS ==========
        Tool(
            name="train_models",
            description="Train prop prediction models using your data. Trains 60+ models for all prop types.",
            inputSchema={
                "type": "object",
                "properties": {
                    "season": {
                        "type": "integer",
                        "description": "Season to train on (e.g., 2024)",
                        "default": 2024
                    },
                    "model_type": {
                        "type": "string",
                        "description": "Model type: 'xgboost', 'lightgbm', or 'ridge'",
                        "default": "xgboost"
                    }
                }
            }
        ),
        Tool(
            name="generate_projections",
            description="Generate projections for a week or specific game using your trained models.",
            inputSchema={
                "type": "object",
                "properties": {
                    "week": {
                        "type": "integer",
                        "description": "NFL week number",
                        "default": 12
                    },
                    "game_id": {
                        "type": "string",
                        "description": "Specific game ID (optional, e.g., '2024_12_BUF_MIA')"
                    },
                    "season": {
                        "type": "integer",
                        "description": "NFL season year",
                        "default": 2024
                    }
                }
            }
        ),
        Tool(
            name="get_model_status",
            description="Check status of trained models and data freshness.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),

        # ========== PURPOSE-SPECIFIC ANALYSIS TOOLS ==========
        Tool(
            name="quick_props",
            description="Fast value scan - returns your model's top props with highest edge. Raw projections, no fluff.",
            inputSchema={
                "type": "object",
                "properties": {
                    "min_edge": {
                        "type": "number",
                        "description": "Minimum edge percentage (default 5.0)",
                        "default": 5.0
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of props to return (default 10)",
                        "default": 10
                    },
                    "prop_types": {
                        "type": "string",
                        "description": "Filter by prop types (comma-separated, e.g., 'pass_yards,rush_yards')"
                    }
                }
            }
        ),
        Tool(
            name="game_deep_dive",
            description="Comprehensive analysis for a specific game - all props, correlations, game script scenarios.",
            inputSchema={
                "type": "object",
                "properties": {
                    "game_id": {
                        "type": "string",
                        "description": "Game ID (e.g., '2024_12_BUF_MIA')"
                    }
                },
                "required": ["game_id"]
            }
        ),
        Tool(
            name="parlay_builder",
            description="Build correlation-aware parlays using your model's projections. Returns raw edge/probability data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "game_ids": {
                        "type": "string",
                        "description": "Comma-separated game IDs to build from"
                    },
                    "max_legs": {
                        "type": "integer",
                        "description": "Maximum legs per parlay (default 3)",
                        "default": 3
                    },
                    "min_leg_edge": {
                        "type": "number",
                        "description": "Minimum edge per leg (default 5.0)",
                        "default": 5.0
                    },
                    "correlation_boost": {
                        "type": "boolean",
                        "description": "Prefer positively correlated legs (same-game stacks)",
                        "default": True
                    }
                }
            }
        ),

        # ========== RAW DATA ACCESS TOOLS ==========
        Tool(
            name="get_player_projection",
            description="Get your model's raw projection for a specific player and prop type.",
            inputSchema={
                "type": "object",
                "properties": {
                    "player_name": {
                        "type": "string",
                        "description": "Player name (e.g., 'Josh Allen')"
                    },
                    "prop_type": {
                        "type": "string",
                        "description": "Prop type (e.g., 'pass_yards', 'rush_yards', 'receptions')"
                    },
                    "game_id": {
                        "type": "string",
                        "description": "Game ID for context (optional)"
                    }
                },
                "required": ["player_name", "prop_type"]
            }
        ),
        Tool(
            name="get_line_movement",
            description="Get line movement data for props - shows opening vs current lines and sharp action signals.",
            inputSchema={
                "type": "object",
                "properties": {
                    "week": {
                        "type": "integer",
                        "description": "NFL week number"
                    },
                    "min_movement": {
                        "type": "number",
                        "description": "Minimum line movement in points (default 1.5)",
                        "default": 1.5
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of movers to return (default 15)",
                        "default": 15
                    }
                }
            }
        ),
        Tool(
            name="get_schedule",
            description="Get upcoming NFL games with your model's available projections.",
            inputSchema={
                "type": "object",
                "properties": {
                    "week": {
                        "type": "integer",
                        "description": "NFL week number (optional)"
                    }
                }
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Handle tool calls."""

    handlers = {
        # Data fetch
        "fetch_nflverse_data": handle_fetch_nflverse,
        "fetch_injuries": handle_fetch_injuries,
        "fetch_dk_odds": handle_fetch_odds,
        "sync_all_data": handle_sync_all,

        # Training/prediction
        "train_models": handle_train_models,
        "generate_projections": handle_generate_projections,
        "get_model_status": handle_model_status,

        # Purpose-specific
        "quick_props": handle_quick_props,
        "game_deep_dive": handle_game_deep_dive,
        "parlay_builder": handle_parlay_builder,

        # Raw data
        "get_player_projection": handle_player_projection,
        "get_line_movement": handle_line_movement,
        "get_schedule": handle_get_schedule,
    }

    handler = handlers.get(name)
    if handler:
        return await handler(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


# ========== DATA FETCH HANDLERS ==========

async def handle_fetch_nflverse(args: dict):
    """Fetch nflverse data."""
    year = args.get("year", 2024)
    include_all = args.get("include_all", True)

    try:
        from backend.ingestion.fetch_nflverse import fetch_nflverse

        output_dir = PROJECT_ROOT / "inputs" / "nflverse"
        cache_dir = PROJECT_ROOT / "cache"

        output_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        fetch_nflverse(year, output_dir, cache_dir, include_all)

        # List what was fetched
        fetched_files = list(output_dir.glob(f"*{year}*.csv"))

        result = add_source_label({
            "success": True,
            "year": year,
            "output_dir": str(output_dir),
            "files_fetched": [f.name for f in fetched_files],
            "file_count": len(fetched_files),
            "datasets": [
                "play_by_play (EPA, WPA, CPOE)",
                "player_stats",
                "rosters",
                "snap_counts" if include_all else None,
                "depth_charts" if include_all else None,
                "pfr_advanced" if include_all else None
            ]
        }, source="NFLVERSE")

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "_source": "ERROR",
            "error": str(e),
            "suggestion": "Check that nflverse data is available for this year"
        }))]


async def handle_fetch_injuries(args: dict):
    """Fetch injury reports."""
    week = args.get("week")

    try:
        output_dir = PROJECT_ROOT / "inputs" / "injuries"
        output_dir.mkdir(parents=True, exist_ok=True)

        fetcher = InjuryFetcher()
        injuries = fetcher.fetch_all_injuries(output_dir, week)

        # Summarize by status
        summary = {"OUT": [], "DOUBTFUL": [], "QUESTIONABLE": [], "PROBABLE": []}

        for team, players in injuries.get("teams", {}).items():
            for player in players:
                status = player.get("status", "UNKNOWN").upper()
                if status in summary:
                    summary[status].append({
                        "player": player.get("name"),
                        "team": team,
                        "position": player.get("position"),
                        "injury": player.get("injury")
                    })

        result = add_source_label({
            "week": week or "current",
            "total_injuries": sum(len(v) for v in summary.values()),
            "by_status": {
                "OUT": len(summary["OUT"]),
                "DOUBTFUL": len(summary["DOUBTFUL"]),
                "QUESTIONABLE": len(summary["QUESTIONABLE"]),
                "PROBABLE": len(summary["PROBABLE"])
            },
            "key_injuries": {
                "out": summary["OUT"][:10],
                "doubtful": summary["DOUBTFUL"][:10]
            },
            "output_file": str(output_dir / f"injuries_week{week or 'current'}.json")
        }, source="ESPN_INJURIES")

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "_source": "ERROR",
            "error": str(e)
        }))]


async def handle_fetch_odds(args: dict):
    """Fetch fresh odds from The Odds API."""
    week = args.get("week", 12)

    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        return [TextContent(type="text", text=json.dumps({
            "_source": "ERROR",
            "error": "ODDS_API_KEY not set",
            "message": "Set ODDS_API_KEY environment variable"
        }))]

    output_dir = PROJECT_ROOT / "outputs" / "prop_lines"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        fetcher = PropLineFetcher(api_key)
        result = fetcher.fetch_snapshot_with_movement(output_dir, week)

        response = add_source_label({
            "success": True,
            "games": result.get("games", 0),
            "props": result.get("props", 0),
            "markets": result.get("markets", []),
            "hot_movers": result.get("hot_movers", 0),
            "output_file": result.get("output_file", ""),
            "api_calls_remaining": result.get("remaining_requests", "unknown")
        }, source="ODDS_API_DRAFTKINGS")

        return [TextContent(type="text", text=json.dumps(response, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "_source": "ERROR",
            "error": str(e)
        }))]


async def handle_sync_all(args: dict):
    """Sync all data sources."""
    year = args.get("year", 2024)
    week = args.get("week", 12)

    results = {
        "nflverse": None,
        "injuries": None,
        "odds": None
    }

    # Fetch nflverse
    try:
        nfl_result = await handle_fetch_nflverse({"year": year, "include_all": True})
        results["nflverse"] = "success"
    except Exception as e:
        results["nflverse"] = f"error: {str(e)}"

    # Fetch injuries
    try:
        inj_result = await handle_fetch_injuries({"week": week})
        results["injuries"] = "success"
    except Exception as e:
        results["injuries"] = f"error: {str(e)}"

    # Fetch odds
    try:
        odds_result = await handle_fetch_odds({"week": week})
        results["odds"] = "success"
    except Exception as e:
        results["odds"] = f"error: {str(e)}"

    response = add_source_label({
        "sync_results": results,
        "year": year,
        "week": week,
        "all_success": all(v == "success" for v in results.values())
    }, source="SYNC_ALL")

    return [TextContent(type="text", text=json.dumps(response, indent=2))]


# ========== TRAINING & PREDICTION HANDLERS ==========

async def handle_train_models(args: dict):
    """Train prop prediction models."""
    season = args.get("season", 2024)
    model_type = args.get("model_type", "xgboost")

    try:
        from backend.modeling.train_multi_prop_models import train_multi_prop_models

        input_dir = PROJECT_ROOT / "inputs"
        output_dir = PROJECT_ROOT / "outputs" / "models"
        output_dir.mkdir(parents=True, exist_ok=True)

        result = train_multi_prop_models(season, input_dir, output_dir, model_type)

        response = add_source_label({
            "success": True,
            "season": season,
            "model_type": model_type,
            "models_trained": result.get("models_trained", 0),
            "prop_types": result.get("prop_types", []),
            "output_dir": str(output_dir),
            "training_time": result.get("training_time", "unknown")
        }, source="YOUR_MODEL_TRAINING")

        return [TextContent(type="text", text=json.dumps(response, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "_source": "ERROR",
            "error": str(e),
            "suggestion": "Ensure training data exists in inputs/ directory"
        }))]


async def handle_generate_projections(args: dict):
    """Generate projections using trained models."""
    week = args.get("week", 12)
    game_id = args.get("game_id")
    season = args.get("season", 2024)

    try:
        from backend.modeling.generate_projections import ProjectionGenerator

        generator = ProjectionGenerator(
            models_dir=PROJECT_ROOT / "outputs" / "models",
            features_dir=PROJECT_ROOT / "inputs",
            output_dir=PROJECT_ROOT / "outputs" / "projections"
        )

        if game_id:
            output_file = generator.generate_for_game(game_id, week)
            scope = f"game {game_id}"
        else:
            output_file = generator.generate_for_week(week, season)
            scope = f"week {week}"

        response = add_source_label({
            "success": True,
            "scope": scope,
            "season": season,
            "output_file": output_file,
            "message": f"Projections generated for {scope}"
        }, source="YOUR_MODEL_PROJECTIONS")

        return [TextContent(type="text", text=json.dumps(response, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "_source": "ERROR",
            "error": str(e),
            "suggestion": "Ensure models are trained first (use train_models)"
        }))]


async def handle_model_status(args: dict):
    """Check model and data status."""
    models_dir = PROJECT_ROOT / "outputs" / "models"
    projections_dir = PROJECT_ROOT / "outputs" / "projections"
    odds_dir = PROJECT_ROOT / "outputs" / "prop_lines"
    inputs_dir = PROJECT_ROOT / "inputs"

    # Check models
    model_files = list(models_dir.glob("**/*.pkl")) if models_dir.exists() else []

    # Check projections
    projection_files = list(projections_dir.glob("*.csv")) if projections_dir.exists() else []

    # Check odds
    odds_files = list(odds_dir.glob("*.json")) if odds_dir.exists() else []

    # Check input data
    nflverse_files = list((inputs_dir / "nflverse").glob("*.csv")) if (inputs_dir / "nflverse").exists() else []

    # Get latest timestamps
    def get_latest_mtime(files):
        if not files:
            return None
        latest = max(files, key=lambda f: f.stat().st_mtime)
        return datetime.fromtimestamp(latest.stat().st_mtime).isoformat()

    response = add_source_label({
        "models": {
            "count": len(model_files),
            "last_trained": get_latest_mtime(model_files),
            "directory": str(models_dir)
        },
        "projections": {
            "count": len(projection_files),
            "last_generated": get_latest_mtime(projection_files),
            "files": [f.name for f in projection_files[-5:]]
        },
        "odds_data": {
            "snapshots": len(odds_files),
            "last_fetched": get_latest_mtime(odds_files)
        },
        "input_data": {
            "nflverse_files": len(nflverse_files),
            "last_updated": get_latest_mtime(nflverse_files)
        },
        "available_games": model_loader.get_available_games()[:10]
    }, source="SYSTEM_STATUS")

    return [TextContent(type="text", text=json.dumps(response, indent=2))]


# ========== PURPOSE-SPECIFIC HANDLERS ==========

async def handle_quick_props(args: dict):
    """Fast value scan - raw model projections with edge."""
    min_edge = args.get("min_edge", 5.0)
    limit = args.get("limit", 10)
    prop_types_filter = args.get("prop_types", "")

    # Load all available projections
    projections = []
    available = model_loader.get_available_games()
    for gid in available[:10]:
        projections.extend(model_loader.load_projections_for_game(gid))

    # Get odds
    prop_lines = odds_api.get_player_props()

    if not prop_lines or not projections:
        return [TextContent(type="text", text=json.dumps(add_source_label({
            "error": "No data available",
            "message": "Fetch odds first with fetch_dk_odds, then generate projections",
            "projections_loaded": len(projections),
            "odds_loaded": len(prop_lines) if prop_lines else 0
        }, source="YOUR_MODEL")))]

    # Find value props
    value_props = prop_analyzer.find_best_props(
        prop_lines, projections, min_edge=min_edge
    )

    # Filter by prop type if specified
    if prop_types_filter:
        allowed_types = [t.strip() for t in prop_types_filter.split(",")]
        value_props = [v for v in value_props if v.prop_line.prop_type in allowed_types]

    # Build raw output
    props_data = []
    for v in value_props[:limit]:
        props_data.append({
            "player": v.prop_line.player_name,
            "team": v.prop_line.player_id.split("_")[0] if "_" in v.prop_line.player_id else "",
            "prop_type": v.prop_line.prop_type,
            "line": v.prop_line.line,
            "book_odds": {
                "over": v.prop_line.over_odds,
                "under": v.prop_line.under_odds
            },
            "your_model": {
                "projection": round(v.projection.projection, 1),
                "std_dev": round(v.projection.std_dev, 2),
                "prob_over": round(1 - (v.edge_under / 100 + prop_analyzer.american_to_implied_probability(v.prop_line.under_odds)) if v.recommendation == "OVER" else v.projection.hit_probability_over, 3),
                "prob_under": round(v.projection.hit_probability_under, 3)
            },
            "edge": {
                "over": round(v.edge_over, 2),
                "under": round(v.edge_under, 2),
                "best_side": v.recommendation,
                "best_edge": round(max(v.edge_over, v.edge_under), 2)
            },
            "confidence": round(v.confidence, 3),
            "grade": v.value_grade
        })

    response = add_source_label({
        "total_found": len(value_props),
        "returned": len(props_data),
        "min_edge_filter": min_edge,
        "props": props_data
    }, source="YOUR_MODEL")

    return [TextContent(type="text", text=json.dumps(response, indent=2))]


async def handle_game_deep_dive(args: dict):
    """Comprehensive game analysis with all props and correlations."""
    game_id = args.get("game_id", "")

    if not game_id:
        return [TextContent(type="text", text=json.dumps({
            "_source": "ERROR",
            "error": "game_id required"
        }))]

    # Parse game_id
    parts = game_id.split("_")
    if len(parts) >= 4:
        season, week, away_team, home_team = parts[0], parts[1], parts[2], parts[3]
    else:
        away_team, home_team = "AWAY", "HOME"
        week = "?"

    # Load projections for this game
    projections = model_loader.load_projections_for_game(game_id)
    prop_lines = odds_api.get_player_props()

    # Get all props for this game
    all_props = []
    if prop_lines and projections:
        value_props = prop_analyzer.find_best_props(prop_lines, projections, min_edge=0)

        # Organize by player
        players = {}
        for v in value_props:
            player = v.prop_line.player_name
            if player not in players:
                players[player] = []
            players[player].append({
                "prop_type": v.prop_line.prop_type,
                "line": v.prop_line.line,
                "projection": round(v.projection.projection, 1),
                "edge": round(max(v.edge_over, v.edge_under), 2),
                "side": v.recommendation,
                "grade": v.value_grade
            })

        # Sort by max edge per player
        all_props = [
            {
                "player": player,
                "props": sorted(props, key=lambda x: x["edge"], reverse=True),
                "best_edge": max(p["edge"] for p in props)
            }
            for player, props in players.items()
        ]
        all_props.sort(key=lambda x: x["best_edge"], reverse=True)

    # Identify correlated stacks
    stacks = []
    if all_props:
        # QB + WR stack potential
        qb_props = [p for p in all_props if any("pass" in prop["prop_type"] for prop in p["props"])]
        wr_props = [p for p in all_props if any("rec" in prop["prop_type"] for prop in p["props"])]

        if qb_props and wr_props:
            stacks.append({
                "type": "passing_stack",
                "players": [qb_props[0]["player"], wr_props[0]["player"]],
                "correlation": "positive",
                "note": "QB pass yards + WR receiving yards tend to correlate"
            })

    response = add_source_label({
        "game_id": game_id,
        "matchup": f"{away_team} @ {home_team}",
        "week": week,
        "total_players": len(all_props),
        "players_with_edge": [p for p in all_props if p["best_edge"] >= 3.0],
        "all_players": all_props[:15],
        "correlation_stacks": stacks,
        "value_summary": {
            "a_grade_props": len([p for p in all_props for prop in p["props"] if prop["grade"] in ["A", "A+"]]),
            "b_grade_props": len([p for p in all_props for prop in p["props"] if prop["grade"] in ["B", "B+"]])
        }
    }, source="YOUR_MODEL")

    return [TextContent(type="text", text=json.dumps(response, indent=2))]


async def handle_parlay_builder(args: dict):
    """Build correlation-aware parlays."""
    game_ids = args.get("game_ids", "")
    max_legs = args.get("max_legs", 3)
    min_leg_edge = args.get("min_leg_edge", 5.0)
    correlation_boost = args.get("correlation_boost", True)

    # Load projections
    projections = []
    if game_ids:
        for gid in game_ids.split(","):
            projections.extend(model_loader.load_projections_for_game(gid.strip()))
    else:
        available = model_loader.get_available_games()
        for gid in available[:5]:
            projections.extend(model_loader.load_projections_for_game(gid))

    prop_lines = odds_api.get_player_props()

    if not prop_lines or not projections:
        return [TextContent(type="text", text=json.dumps(add_source_label({
            "error": "No data for parlay building",
            "suggestion": "Fetch odds and generate projections first"
        }, source="YOUR_MODEL")))]

    # Get value props
    value_props = prop_analyzer.find_best_props(prop_lines, projections, min_edge=min_leg_edge)

    if len(value_props) < max_legs:
        return [TextContent(type="text", text=json.dumps(add_source_label({
            "error": f"Not enough props with {min_leg_edge}%+ edge",
            "found": len(value_props),
            "needed": max_legs
        }, source="YOUR_MODEL")))]

    # Build parlays
    parlays = []

    # Parlay 1: Top edge props
    top_legs = value_props[:max_legs]
    combined_prob = 1.0
    for v in top_legs:
        combined_prob *= v.confidence

    parlays.append({
        "name": "Top Edge Parlay",
        "legs": [
            {
                "player": v.prop_line.player_name,
                "prop": v.prop_line.prop_type,
                "line": v.prop_line.line,
                "side": v.recommendation,
                "projection": round(v.projection.projection, 1),
                "edge": round(max(v.edge_over, v.edge_under), 2),
                "confidence": round(v.confidence, 3)
            }
            for v in top_legs
        ],
        "combined_probability": round(combined_prob, 4),
        "expected_odds": round((1 / combined_prob - 1) * 100) if combined_prob > 0 else 0
    })

    # Parlay 2: Same-game stack (if correlation_boost)
    if correlation_boost and game_ids:
        # Group by game
        by_game = {}
        for v in value_props:
            gid = getattr(v.prop_line, 'game_id', 'unknown')
            if gid not in by_game:
                by_game[gid] = []
            by_game[gid].append(v)

        # Find best same-game stack
        for gid, props in by_game.items():
            if len(props) >= 2:
                stack_legs = props[:min(max_legs, len(props))]
                stack_prob = 1.0
                for v in stack_legs:
                    stack_prob *= v.confidence

                # Boost for positive correlation (simplified)
                stack_prob *= 1.05  # 5% correlation boost

                parlays.append({
                    "name": f"Same-Game Stack ({gid})",
                    "legs": [
                        {
                            "player": v.prop_line.player_name,
                            "prop": v.prop_line.prop_type,
                            "line": v.prop_line.line,
                            "side": v.recommendation,
                            "projection": round(v.projection.projection, 1),
                            "edge": round(max(v.edge_over, v.edge_under), 2),
                            "confidence": round(v.confidence, 3)
                        }
                        for v in stack_legs
                    ],
                    "combined_probability": round(stack_prob, 4),
                    "correlation_adjustment": "+5%",
                    "expected_odds": round((1 / stack_prob - 1) * 100) if stack_prob > 0 else 0
                })
                break

    response = add_source_label({
        "parlays_built": len(parlays),
        "max_legs": max_legs,
        "min_edge_per_leg": min_leg_edge,
        "parlays": parlays
    }, source="YOUR_MODEL")

    return [TextContent(type="text", text=json.dumps(response, indent=2))]


# ========== RAW DATA HANDLERS ==========

async def handle_player_projection(args: dict):
    """Get raw projection for a specific player."""
    player_name = args.get("player_name", "")
    prop_type = args.get("prop_type", "")
    game_id = args.get("game_id", "")

    if not player_name or not prop_type:
        return [TextContent(type="text", text=json.dumps({
            "_source": "ERROR",
            "error": "player_name and prop_type required"
        }))]

    # Load projections
    projections = []
    if game_id:
        projections = model_loader.load_projections_for_game(game_id)
    else:
        available = model_loader.get_available_games()
        for gid in available:
            projections.extend(model_loader.load_projections_for_game(gid))

    # Find player's projection
    player_proj = None
    for p in projections:
        if player_name.lower() in p.player_name.lower() and p.prop_type == prop_type:
            player_proj = p
            break

    if not player_proj:
        return [TextContent(type="text", text=json.dumps(add_source_label({
            "error": f"No projection found for {player_name} - {prop_type}",
            "available_players": list(set(p.player_name for p in projections))[:20]
        }, source="YOUR_MODEL")))]

    # Get odds if available
    prop_lines = odds_api.get_player_props()
    odds_data = None
    if prop_lines:
        for line in prop_lines:
            if player_name.lower() in line.player_name.lower() and line.prop_type == prop_type:
                odds_data = {
                    "line": line.line,
                    "over_odds": line.over_odds,
                    "under_odds": line.under_odds,
                    "book": line.book
                }
                break

    response = add_source_label({
        "player": player_proj.player_name,
        "prop_type": prop_type,
        "projection": {
            "mean": round(player_proj.projection, 2),
            "std_dev": round(player_proj.std_dev, 2),
            "confidence_interval_80": [
                round(player_proj.confidence_interval[0], 1),
                round(player_proj.confidence_interval[1], 1)
            ],
            "prob_over_line": round(player_proj.hit_probability_over, 3) if odds_data else None,
            "prob_under_line": round(player_proj.hit_probability_under, 3) if odds_data else None
        },
        "quality_metrics": {
            "games_sampled": player_proj.games_sampled,
            "model_quality": round(player_proj.model_quality, 3),
            "usage_metric": round(player_proj.usage_metric, 3) if player_proj.usage_metric else None
        },
        "current_odds": odds_data
    }, source="YOUR_MODEL")

    return [TextContent(type="text", text=json.dumps(response, indent=2))]


async def handle_line_movement(args: dict):
    """Get line movement data."""
    week = args.get("week")
    min_movement = args.get("min_movement", 1.5)
    limit = args.get("limit", 15)

    snapshots_dir = PROJECT_ROOT / "outputs" / "prop_lines"

    if not snapshots_dir.exists():
        return [TextContent(type="text", text=json.dumps(add_source_label({
            "error": "No snapshots found",
            "message": "Run fetch_dk_odds first to capture prop line snapshots"
        }, source="ODDS_API")))]

    api_key = os.getenv("ODDS_API_KEY", "")
    fetcher = PropLineFetcher(api_key)
    trends = fetcher.get_trending_props(snapshots_dir, week)

    if "error" in trends:
        return [TextContent(type="text", text=json.dumps(add_source_label(trends, source="ODDS_API")))]

    # Filter by minimum movement
    hot_movers = [
        m for m in trends.get("hot_movers", [])
        if abs(m.get("line_movement", 0)) >= min_movement
    ][:limit]

    response = add_source_label({
        "min_movement_filter": min_movement,
        "movers_found": len(hot_movers),
        "hot_movers": [
            {
                "player": m["player"],
                "market": m["market"],
                "opening_line": m["opening_line"],
                "current_line": m["current_line"],
                "movement": m["line_movement"],
                "direction": m["direction"],
                "sharp_signal": "likely" if abs(m["line_movement"]) >= 3 else "possible"
            }
            for m in hot_movers
        ],
        "sustained_trends": [
            {
                "player": t["player"],
                "market": t["market"],
                "total_movement": t["total_movement"],
                "direction": t["direction"],
                "weeks_trending": t["consistency"]
            }
            for t in trends.get("sustained_trends", [])[:limit]
        ]
    }, source="ODDS_API_LINE_MOVEMENT")

    return [TextContent(type="text", text=json.dumps(response, indent=2))]


async def handle_get_schedule(args: dict):
    """Get game schedule with projection availability."""
    week = args.get("week")

    available_games = model_loader.get_available_games()

    games_data = []
    for gid in available_games[:16]:
        parts = gid.split("_")
        if len(parts) >= 4:
            games_data.append({
                "game_id": gid,
                "season": parts[0],
                "week": parts[1],
                "away": parts[2],
                "home": parts[3],
                "has_projections": True
            })
        else:
            games_data.append({
                "game_id": gid,
                "has_projections": True
            })

    response = add_source_label({
        "week": week or "all available",
        "games_count": len(games_data),
        "games": games_data
    }, source="YOUR_MODEL")

    return [TextContent(type="text", text=json.dumps(response, indent=2))]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
