"""MCP Server for NFL Prop Betting System.

This MCP server calls the local FastAPI server (localhost:8000) for all data operations.
The API server handles fetching, storage, and calculations.

Run the API server first: python start_server.py
Then use this with Claude Desktop.
"""

import json
import asyncio
from datetime import datetime
from pathlib import Path

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Create MCP server
server = Server("nfl-betting")

# API server URL
API_BASE = "http://localhost:8000"
DEFAULT_WEEK = 12
DEFAULT_FETCH_SEASON = 2024
DEFAULT_STATS_SEASON = 2025


def _int_property(default: int, description: str | None = None) -> dict:
    prop = {"type": "integer", "default": default}
    if description:
        prop["description"] = description
    return prop


def week_property(description: str = "NFL week number") -> dict:
    return _int_property(DEFAULT_WEEK, description)


def fetch_season_property(description: str = "NFL season year") -> dict:
    return _int_property(DEFAULT_FETCH_SEASON, description)


def stats_season_property(description: str = "NFL season year") -> dict:
    return _int_property(DEFAULT_STATS_SEASON, description)


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
            name="fetch_odds",
            description="Fetch fresh DraftKings prop odds and store in local database. Tracks line movement over time.",
            inputSchema={
                "type": "object",
                "properties": {
                    "week": week_property(),
                    "season": fetch_season_property()
                }
            }
        ),
        Tool(
            name="fetch_injuries",
            description="Fetch current injury reports from ESPN and store in local database.",
            inputSchema={
                "type": "object",
                "properties": {
                    "week": {"type": "integer", "description": "NFL week number"},
                    "season": fetch_season_property()
                }
            }
        ),
        Tool(
            name="fetch_nflverse",
            description="Fetch nflverse data (play-by-play, stats, rosters). This can take several minutes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "year": fetch_season_property()
                }
            }
        ),
        Tool(
            name="sync_all_data",
            description="Fetch all data sources at once (odds, injuries).",
            inputSchema={
                "type": "object",
                "properties": {
                    "week": week_property(),
                    "year": fetch_season_property()
                }
            }
        ),
        Tool(
            name="check_data_freshness",
            description="Check how fresh your data is and see what needs refreshing.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="auto_refresh",
            description="Automatically refresh only stale data sources. Smart refresh that skips fresh data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "week": week_property(),
                    "year": fetch_season_property(),
                    "force": {
                        "type": "boolean",
                        "description": "Force refresh even if data is fresh",
                        "default": False
                    }
                }
            }
        ),

        # ========== QUERY TOOLS ==========
        Tool(
            name="get_status",
            description="Check database status and data freshness.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="quick_props",
            description="Fast value scan - find props with highest edge from your model.",
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
                        "description": "Filter by prop types (comma-separated)"
                    }
                }
            }
        ),
        Tool(
            name="game_deep_dive",
            description="Comprehensive analysis for a specific game - all props, projections, edges.",
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

        # ========== COMPREHENSIVE INTELLIGENCE TOOLS ==========
        Tool(
            name="full_matchup_analysis",
            description="COMPREHENSIVE matchup analysis - injuries, projections, line movement, correlations, everything for one game.",
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
            name="daily_betting_brief",
            description="DAILY INTELLIGENCE - Auto-refreshes data, shows top props, injuries, sharp action across ALL games. Start here!",
            inputSchema={
                "type": "object",
                "properties": {
                    "week": week_property(),
                    "min_edge": {
                        "type": "number",
                        "description": "Minimum edge for top props (default 3.0)",
                        "default": 3.0
                    },
                    "auto_refresh": {
                        "type": "boolean",
                        "description": "Auto-refresh stale data first",
                        "default": True
                    }
                }
            }
        ),
        Tool(
            name="player_outlook",
            description="COMPLETE player analysis - all props, injury status/history, projection trends, line movement.",
            inputSchema={
                "type": "object",
                "properties": {
                    "player_name": {
                        "type": "string",
                        "description": "Player name (e.g., 'Josh Allen')"
                    }
                },
                "required": ["player_name"]
            }
        ),
        Tool(
            name="get_situational_analysis",
            description="Situational analysis for a game (weather, rest, momentum, positional edges).",
            inputSchema={
                "type": "object",
                "properties": {
                    "game_id": {"type": "string", "description": "Game ID (e.g., '2024_12_BUF_MIA')"},
                    "home_team": {"type": "string", "description": "Home team abbreviation"},
                    "away_team": {"type": "string", "description": "Away team abbreviation"},
                    "season": stats_season_property(),
                    "week": week_property()
                },
                "required": ["game_id", "home_team", "away_team"]
            }
        ),
        Tool(
            name="get_positional_matchups",
            description="Positional matchup grades and prop targets for a game.",
            inputSchema={
                "type": "object",
                "properties": {
                    "game_id": {"type": "string", "description": "Game ID (e.g., '2024_12_BUF_MIA')"},
                    "home_team": {"type": "string", "description": "Home team abbreviation"},
                    "away_team": {"type": "string", "description": "Away team abbreviation"},
                    "season": stats_season_property(),
                    "week": week_property()
                },
                "required": ["game_id", "home_team", "away_team"]
            }
        ),
        Tool(
            name="evaluate_game",
            description="Full evaluation pipeline for a single game (situational, matchups, injuries, prop value).",
            inputSchema={
                "type": "object",
                "properties": {
                    "game_id": {"type": "string", "description": "Game ID (e.g., '2024_12_BUF_MIA')"},
                    "home_team": {"type": "string", "description": "Home team abbreviation"},
                    "away_team": {"type": "string", "description": "Away team abbreviation"},
                    "season": stats_season_property(),
                    "week": week_property()
                },
                "required": ["game_id", "home_team", "away_team"]
            }
        ),
        Tool(
            name="evaluate_week",
            description="Evaluate and rank all games in a week using the full pipeline.",
            inputSchema={
                "type": "object",
                "properties": {
                    "week": week_property(),
                    "season": stats_season_property()
                },
                "required": ["week"]
            }
        ),
        Tool(
            name="get_team_trending_form",
            description="Momentum and trending form for a team (last 3 vs season averages).",
            inputSchema={
                "type": "object",
                "properties": {
                    "team": {"type": "string", "description": "Team abbreviation (e.g., 'BUF')"},
                    "season": stats_season_property(),
                    "week": week_property()
                },
                "required": ["team"]
            }
        ),
        Tool(
            name="get_rush_defense",
            description="Rush defense performance with RB matchups and trends.",
            inputSchema={
                "type": "object",
                "properties": {
                    "team": {"type": "string", "description": "Team abbreviation"},
                    "season": stats_season_property(),
                    "last_n_games": {"type": "integer", "default": 5}
                },
                "required": ["team"]
            }
        ),
        Tool(
            name="get_pass_defense",
            description="Pass defense performance with QB matchups and trends.",
            inputSchema={
                "type": "object",
                "properties": {
                    "team": {"type": "string", "description": "Team abbreviation"},
                    "season": stats_season_property(),
                    "last_n_games": {"type": "integer", "default": 5}
                },
                "required": ["team"]
            }
        ),
        Tool(
            name="get_defense_summary",
            description="Combined rush/pass defense summary with matchup notes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "team": {"type": "string", "description": "Team abbreviation"},
                    "season": stats_season_property()
                },
                "required": ["team"]
            }
        ),
        Tool(
            name="get_line_movement",
            description="Get line movement history for a player/prop.",
            inputSchema={
                "type": "object",
                "properties": {
                    "player_name": {
                        "type": "string",
                        "description": "Player name"
                    },
                    "prop_type": {
                        "type": "string",
                        "description": "Prop type (e.g., 'pass_yards')"
                    },
                    "days": {
                        "type": "integer",
                        "description": "Days to look back (default 7)",
                        "default": 7
                    }
                },
                "required": ["player_name", "prop_type"]
            }
        ),
        Tool(
            name="get_hot_movers",
            description="Find props with significant line movement (sharp action signals).",
            inputSchema={
                "type": "object",
                "properties": {
                    "min_movement": {
                        "type": "number",
                        "description": "Minimum line movement in points (default 1.5)",
                        "default": 1.5
                    },
                    "hours": {
                        "type": "integer",
                        "description": "Hours to look back (default 48)",
                        "default": 48
                    }
                }
            }
        ),
        Tool(
            name="get_latest_odds",
            description="Get latest odds snapshot from database.",
            inputSchema={
                "type": "object",
                "properties": {
                    "game_id": {
                        "type": "string",
                        "description": "Filter by game ID"
                    },
                    "player_name": {
                        "type": "string",
                        "description": "Filter by player name"
                    },
                    "prop_type": {
                        "type": "string",
                        "description": "Filter by prop type"
                    }
                }
            }
        ),
        Tool(
            name="get_latest_projections",
            description="Get latest projections from your model.",
            inputSchema={
                "type": "object",
                "properties": {
                    "game_id": {
                        "type": "string",
                        "description": "Filter by game ID"
                    },
                    "player_name": {
                        "type": "string",
                        "description": "Filter by player name"
                    },
                    "prop_type": {
                        "type": "string",
                        "description": "Filter by prop type"
                    }
                }
            }
        ),
        Tool(
            name="get_injuries",
            description="Get latest injury reports.",
            inputSchema={
                "type": "object",
                "properties": {
                    "team": {
                        "type": "string",
                        "description": "Filter by team"
                    },
                    "status": {
                        "type": "string",
                        "description": "Filter by status (OUT, DOUBTFUL, QUESTIONABLE)"
                    }
                }
            }
        ),
        Tool(
            name="get_games",
            description="Get schedule/games.",
            inputSchema={
                "type": "object",
                "properties": {
                    "week": {
                        "type": "integer",
                        "description": "NFL week number"
                    },
                    "season": stats_season_property()
                }
            }
        ),
        Tool(
            name="get_value_props_history",
            description="Get historical value props found by your model.",
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Days to look back (default 7)",
                        "default": 7
                    },
                    "min_edge": {
                        "type": "number",
                        "description": "Minimum edge filter (default 0)",
                        "default": 0
                    }
                }
            }
        ),
        Tool(
            name="get_model_runs",
            description="Get model training run history.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of runs to return (default 20)",
                        "default": 20
                    }
                }
            }
        ),

        # ========== STATS/KNOWLEDGE TOOLS (for general queries) ==========
        Tool(
            name="get_player_stats",
            description="FULL PLAYER PROFILE - Like ESPN page: season totals, weekly stats, bio. Use for quick knowledge queries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "player_name": {
                        "type": "string",
                        "description": "Player name (e.g., 'Patrick Mahomes')"
                    },
                    "season": stats_season_property()
                },
                "required": ["player_name"]
            }
        ),
        Tool(
            name="get_team_profile",
            description="FULL TEAM PROFILE - Stats, roster, key players, schedule. Use for quick team knowledge queries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "team": {
                        "type": "string",
                        "description": "Team abbreviation (e.g., 'KC', 'BUF', 'SF')"
                    },
                    "season": stats_season_property()
                },
                "required": ["team"]
            }
        ),
        Tool(
            name="get_league_leaders",
            description="LEAGUE LEADERS - Top players in passing yards, rushing yards, receiving yards, TDs, fantasy points, etc.",
            inputSchema={
                "type": "object",
                "properties": {
                    "stat_type": {
                        "type": "string",
                        "description": "Stat category: passing_yards, passing_tds, rushing_yards, rushing_tds, receiving_yards, receiving_tds, receptions, fantasy, fantasy_ppr",
                        "enum": ["passing_yards", "passing_tds", "rushing_yards", "rushing_tds", "receiving_yards", "receiving_tds", "receptions", "fantasy", "fantasy_ppr"]
                    },
                    "season": stats_season_property(),
                    "limit": {
                        "type": "integer",
                        "description": "Number of leaders to return",
                        "default": 20
                    }
                },
                "required": ["stat_type"]
            }
        ),
        Tool(
            name="get_schedule",
            description="Get full NFL schedule for season or specific week.",
            inputSchema={
                "type": "object",
                "properties": {
                    "season": stats_season_property(),
                    "week": {
                        "type": "integer",
                        "description": "Specific week (optional - all weeks if not specified)"
                    }
                }
            }
        ),
        Tool(
            name="get_team_rankings",
            description="Get all teams ranked by wins/losses - league standings.",
            inputSchema={
                "type": "object",
                "properties": {
                    "season": stats_season_property()
                }
            }
        ),
        Tool(
            name="populate_database",
            description="POPULATE ALL DATA - Load schedule, player stats, rosters, injuries, and odds for the season. Run this first!",
            inputSchema={
                "type": "object",
                "properties": {
                    "season": stats_season_property(),
                    "week": week_property(),
                    "fetch_first": {
                        "type": "boolean",
                        "description": "Fetch from nflverse first (slower)",
                        "default": False
                    }
                }
            }
        ),

        # ========== CROSS-GAME PARLAY TOOLS ==========
        Tool(
            name="best_props_all_games",
            description="BEST PROPS ACROSS ALL GAMES - Find the highest edge props across every game in a week for cross-game parlays. Returns top props sorted by edge with game context. USE THIS when asked for 'best props', 'parlay legs', or props across multiple games.",
            inputSchema={
                "type": "object",
                "properties": {
                    "week": week_property(),
                    "min_edge": {
                        "type": "number",
                        "description": "Minimum edge percentage (default 3.0)",
                        "default": 3.0
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of props to return (default 20)",
                        "default": 20
                    },
                    "prop_types": {
                        "type": "string",
                        "description": "Filter by prop types (comma-separated, e.g., 'pass_yards,rush_yards')"
                    }
                }
            }
        )
    ]



async def _call_api(client: httpx.AsyncClient, method: str, path: str, *, params=None, timeout: float = 60.0):
    """Shared wrapper to hit the API server."""
    return await client.request(
        method,
        f"{API_BASE}{path}",
        params=params or {},
        timeout=timeout,
    )


async def _handle_fetch_odds(client, args):
    params = {
        "week": args.get("week", DEFAULT_WEEK),
        "season": args.get("season", DEFAULT_FETCH_SEASON),
    }
    return await _call_api(client, "post", "/fetch/odds", params=params)


async def _handle_fetch_injuries(client, args):
    params = {"season": args.get("season", DEFAULT_FETCH_SEASON)}
    if args.get("week") is not None:
        params["week"] = args["week"]
    return await _call_api(client, "post", "/fetch/injuries", params=params)


async def _handle_fetch_nflverse(client, args):
    params = {"year": args.get("year", DEFAULT_FETCH_SEASON), "include_all": True}
    return await _call_api(client, "post", "/fetch/nflverse", params=params, timeout=300.0)


async def _handle_sync_all_data(client, args):
    params = {
        "week": args.get("week", DEFAULT_WEEK),
        "year": args.get("year", DEFAULT_FETCH_SEASON),
    }
    return await _call_api(client, "post", "/fetch/all", params=params)


async def _handle_check_data_freshness(client, _args):
    return await _call_api(client, "get", "/refresh/check")


async def _handle_auto_refresh(client, args):
    params = {
        "week": args.get("week", DEFAULT_WEEK),
        "year": args.get("year", DEFAULT_FETCH_SEASON),
        "force": args.get("force", False),
    }
    return await _call_api(client, "post", "/refresh/auto", params=params)


async def _handle_get_status(client, _args):
    return await _call_api(client, "get", "/")


async def _handle_quick_props(client, args):
    params = {"min_edge": args.get("min_edge", 5.0), "limit": args.get("limit", 10)}
    if args.get("prop_types"):
        params["prop_types"] = args["prop_types"]
    return await _call_api(client, "get", "/analysis/quick-props", params=params)


async def _handle_game_deep_dive(client, args):
    return await _call_api(client, "get", f"/analysis/game/{args.get('game_id', '')}")


async def _handle_full_matchup_analysis(client, args):
    return await _call_api(client, "get", f"/intelligence/matchup/{args.get('game_id', '')}")


async def _handle_daily_betting_brief(client, args):
    params = {
        "week": args.get("week", DEFAULT_WEEK),
        "min_edge": args.get("min_edge", 3.0),
        "auto_refresh": args.get("auto_refresh", True),
    }
    return await _call_api(client, "get", "/intelligence/daily-brief", params=params)


async def _handle_player_outlook(client, args):
    return await _call_api(client, "get", f"/intelligence/player/{args.get('player_name', '')}")


async def _handle_get_situational_analysis(client, args):
    params = {
        "game_id": args.get("game_id", ""),
        "home_team": args.get("home_team", ""),
        "away_team": args.get("away_team", ""),
        "season": args.get("season", DEFAULT_FETCH_SEASON),
        "week": args.get("week", DEFAULT_WEEK),
    }
    return await _call_api(client, "get", "/analysis/situational", params=params)


async def _handle_get_positional_matchups(client, args):
    params = {
        "home_team": args.get("home_team", ""),
        "away_team": args.get("away_team", ""),
        "season": args.get("season", DEFAULT_FETCH_SEASON),
        "week": args.get("week", DEFAULT_WEEK),
    }
    return await _call_api(
        client,
        "get",
        f"/analysis/game/{args.get('game_id', '')}/positional",
        params=params,
    )


async def _handle_evaluate_game(client, args):
    params = {
        "game_id": args.get("game_id", ""),
        "home_team": args.get("home_team", ""),
        "away_team": args.get("away_team", ""),
        "season": args.get("season", DEFAULT_FETCH_SEASON),
        "week": args.get("week", DEFAULT_WEEK),
    }
    return await _call_api(client, "get", "/analysis/evaluate/game", params=params, timeout=90.0)


async def _handle_evaluate_week(client, args):
    params = {
        "week": args.get("week", DEFAULT_WEEK),
        "season": args.get("season", DEFAULT_FETCH_SEASON),
    }
    return await _call_api(client, "get", "/analysis/evaluate/week", params=params, timeout=120.0)


async def _handle_get_team_trending_form(client, args):
    params = {
        "team": args.get("team", ""),
        "season": args.get("season", DEFAULT_FETCH_SEASON),
        "week": args.get("week", DEFAULT_WEEK),
    }
    return await _call_api(client, "get", "/analysis/team/form", params=params)


async def _handle_get_rush_defense(client, args):
    params = {
        "season": args.get("season", DEFAULT_FETCH_SEASON),
        "last_n_games": args.get("last_n_games", 5),
    }
    return await _call_api(client, "get", f"/analysis/defense/{args.get('team', '')}/rush", params=params)


async def _handle_get_pass_defense(client, args):
    params = {
        "season": args.get("season", DEFAULT_FETCH_SEASON),
        "last_n_games": args.get("last_n_games", 5),
    }
    return await _call_api(client, "get", f"/analysis/defense/{args.get('team', '')}/pass", params=params)


async def _handle_get_defense_summary(client, args):
    params = {"season": args.get("season", DEFAULT_FETCH_SEASON)}
    return await _call_api(client, "get", f"/analysis/defense/{args.get('team', '')}", params=params)


async def _handle_get_line_movement(client, args):
    params = {
        "player_name": args.get("player_name", ""),
        "prop_type": args.get("prop_type", ""),
        "days": args.get("days", 7),
    }
    return await _call_api(client, "get", "/odds/movement", params=params)


async def _handle_get_hot_movers(client, args):
    params = {"min_movement": args.get("min_movement", 1.5), "hours": args.get("hours", 48)}
    return await _call_api(client, "get", "/odds/movers", params=params)


async def _handle_get_latest_odds(client, args):
    params = {}
    if args.get("game_id"):
        params["game_id"] = args["game_id"]
    if args.get("player_name"):
        params["player_name"] = args["player_name"]
    if args.get("prop_type"):
        params["prop_type"] = args["prop_type"]
    return await _call_api(client, "get", "/odds/latest", params=params)


async def _handle_get_latest_projections(client, args):
    params = {}
    if args.get("game_id"):
        params["game_id"] = args["game_id"]
    if args.get("player_name"):
        params["player_name"] = args["player_name"]
    if args.get("prop_type"):
        params["prop_type"] = args["prop_type"]
    return await _call_api(client, "get", "/projections/latest", params=params)


async def _handle_get_injuries(client, args):
    params = {"season": args.get("season", DEFAULT_FETCH_SEASON)}
    if args.get("week") is not None:
        params["week"] = args["week"]
    return await _call_api(client, "get", "/injuries", params=params)


async def _handle_get_games(client, args):
    params = {"season": args.get("season", DEFAULT_FETCH_SEASON)}
    if args.get("week") is not None:
        params["week"] = args["week"]
    return await _call_api(client, "get", "/games", params=params)


async def _handle_get_value_props_history(client, args):
    params = {"days": args.get("days", 7), "min_edge": args.get("min_edge", 0)}
    return await _call_api(client, "get", "/value-props/history", params=params)


async def _handle_get_model_runs(client, args):
    return await _call_api(client, "get", "/model/runs", params={"limit": args.get("limit", 20)})


async def _handle_get_player_stats(client, args):
    params = {"season": args.get("season", DEFAULT_STATS_SEASON)}
    return await _call_api(client, "get", f"/stats/player/{args.get('player_name', '')}", params=params)


async def _handle_get_team_profile(client, args):
    params = {"season": args.get("season", DEFAULT_STATS_SEASON)}
    return await _call_api(client, "get", f"/stats/team/{args.get('team', '')}", params=params)


async def _handle_get_league_leaders(client, args):
    params = {
        "season": args.get("season", DEFAULT_STATS_SEASON),
        "limit": args.get("limit", 20),
    }
    return await _call_api(client, "get", f"/stats/leaders/{args.get('stat_type', 'passing_yards')}", params=params)


async def _handle_get_schedule(client, args):
    params = {"season": args.get("season", DEFAULT_STATS_SEASON)}
    if args.get("week") is not None:
        params["week"] = args["week"]
    return await _call_api(client, "get", "/stats/schedule", params=params)


async def _handle_get_team_rankings(client, args):
    params = {"season": args.get("season", DEFAULT_STATS_SEASON)}
    return await _call_api(client, "get", "/stats/rankings", params=params)


async def _handle_populate_database(client, args):
    params = {
        "season": args.get("season", DEFAULT_STATS_SEASON),
        "week": args.get("week", DEFAULT_WEEK),
        "fetch_first": args.get("fetch_first", False),
        "include_odds": True,
    }
    return await _call_api(client, "post", "/populate/all", params=params, timeout=180.0)


async def _handle_best_props_all_games(client, args):
    params = {
        "min_edge": args.get("min_edge", 3.0),
        "limit": args.get("limit", 20),
        "week": args.get("week", DEFAULT_WEEK),
    }
    if args.get("prop_types"):
        params["prop_types"] = args["prop_types"]
    return await _call_api(client, "get", "/analysis/quick-props", params=params)


TOOL_HANDLERS = {
    "fetch_odds": _handle_fetch_odds,
    "fetch_injuries": _handle_fetch_injuries,
    "fetch_nflverse": _handle_fetch_nflverse,
    "sync_all_data": _handle_sync_all_data,
    "check_data_freshness": _handle_check_data_freshness,
    "auto_refresh": _handle_auto_refresh,
    "get_status": _handle_get_status,
    "quick_props": _handle_quick_props,
    "game_deep_dive": _handle_game_deep_dive,
    "full_matchup_analysis": _handle_full_matchup_analysis,
    "daily_betting_brief": _handle_daily_betting_brief,
    "player_outlook": _handle_player_outlook,
    "get_situational_analysis": _handle_get_situational_analysis,
    "get_positional_matchups": _handle_get_positional_matchups,
    "evaluate_game": _handle_evaluate_game,
    "evaluate_week": _handle_evaluate_week,
    "get_team_trending_form": _handle_get_team_trending_form,
    "get_rush_defense": _handle_get_rush_defense,
    "get_pass_defense": _handle_get_pass_defense,
    "get_defense_summary": _handle_get_defense_summary,
    "get_line_movement": _handle_get_line_movement,
    "get_hot_movers": _handle_get_hot_movers,
    "get_latest_odds": _handle_get_latest_odds,
    "get_latest_projections": _handle_get_latest_projections,
    "get_injuries": _handle_get_injuries,
    "get_games": _handle_get_games,
    "get_value_props_history": _handle_get_value_props_history,
    "get_model_runs": _handle_get_model_runs,
    "get_player_stats": _handle_get_player_stats,
    "get_team_profile": _handle_get_team_profile,
    "get_league_leaders": _handle_get_league_leaders,
    "get_schedule": _handle_get_schedule,
    "get_team_rankings": _handle_get_team_rankings,
    "populate_database": _handle_populate_database,
    "best_props_all_games": _handle_best_props_all_games,
}


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Handle tool calls by calling the local API server."""

    async with httpx.AsyncClient(timeout=60.0) as client:
        handler = TOOL_HANDLERS.get(name)
        if handler is None:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown tool: {name}"})
            )]

        try:
            response = await handler(client, arguments)

            if response.status_code == 200:
                result = response.json()
                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]

            return [TextContent(
                type="text",
                text=json.dumps({
                    "_source": "ERROR",
                    "status_code": response.status_code,
                    "detail": response.text
                })
            )]

        except httpx.ConnectError:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "_source": "ERROR",
                    "error": "Cannot connect to API server",
                    "message": "Start the API server first: python start_server.py",
                    "api_url": API_BASE
                })
            )]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "_source": "ERROR",
                    "error": str(e)
                })
            )]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
