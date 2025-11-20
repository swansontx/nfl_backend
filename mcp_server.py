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
                    "week": {
                        "type": "integer",
                        "description": "NFL week number",
                        "default": 12
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
            name="fetch_injuries",
            description="Fetch current injury reports from ESPN and store in local database.",
            inputSchema={
                "type": "object",
                "properties": {
                    "week": {
                        "type": "integer",
                        "description": "NFL week number"
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
            name="fetch_nflverse",
            description="Fetch nflverse data (play-by-play, stats, rosters). This can take several minutes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "year": {
                        "type": "integer",
                        "description": "NFL season year",
                        "default": 2024
                    }
                }
            }
        ),
        Tool(
            name="sync_all_data",
            description="Fetch all data sources at once (odds, injuries).",
            inputSchema={
                "type": "object",
                "properties": {
                    "week": {
                        "type": "integer",
                        "description": "NFL week number",
                        "default": 12
                    },
                    "year": {
                        "type": "integer",
                        "description": "NFL season year",
                        "default": 2024
                    }
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
                    "week": {
                        "type": "integer",
                        "description": "NFL week number",
                        "default": 12
                    },
                    "year": {
                        "type": "integer",
                        "description": "NFL season year",
                        "default": 2024
                    },
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
                    "week": {
                        "type": "integer",
                        "description": "NFL week number",
                        "default": 12
                    },
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
                    "season": {
                        "type": "integer",
                        "description": "NFL season year",
                        "default": 2024
                    }
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
                    "season": {
                        "type": "integer",
                        "description": "NFL season year",
                        "default": 2025
                    }
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
                    "season": {
                        "type": "integer",
                        "description": "NFL season year",
                        "default": 2025
                    }
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
                    "season": {
                        "type": "integer",
                        "description": "NFL season year",
                        "default": 2025
                    },
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
                    "season": {
                        "type": "integer",
                        "description": "NFL season year",
                        "default": 2025
                    },
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
                    "season": {
                        "type": "integer",
                        "description": "NFL season year",
                        "default": 2025
                    }
                }
            }
        ),
        Tool(
            name="populate_database",
            description="POPULATE ALL DATA - Load schedule, player stats, rosters, injuries, and odds for the season. Run this first!",
            inputSchema={
                "type": "object",
                "properties": {
                    "season": {
                        "type": "integer",
                        "description": "NFL season year",
                        "default": 2025
                    },
                    "week": {
                        "type": "integer",
                        "description": "Current week number",
                        "default": 12
                    },
                    "fetch_first": {
                        "type": "boolean",
                        "description": "Fetch from nflverse first (slower)",
                        "default": False
                    }
                }
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Handle tool calls by calling the local API server."""

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            # ========== FETCH TOOLS ==========
            if name == "fetch_odds":
                week = arguments.get("week", 12)
                season = arguments.get("season", 2024)
                response = await client.post(
                    f"{API_BASE}/fetch/odds",
                    params={"week": week, "season": season}
                )

            elif name == "fetch_injuries":
                week = arguments.get("week")
                season = arguments.get("season", 2024)
                params = {"season": season}
                if week:
                    params["week"] = week
                response = await client.post(
                    f"{API_BASE}/fetch/injuries",
                    params=params
                )

            elif name == "fetch_nflverse":
                year = arguments.get("year", 2024)
                response = await client.post(
                    f"{API_BASE}/fetch/nflverse",
                    params={"year": year, "include_all": True},
                    timeout=300.0  # 5 minutes for large download
                )

            elif name == "sync_all_data":
                week = arguments.get("week", 12)
                year = arguments.get("year", 2024)
                response = await client.post(
                    f"{API_BASE}/fetch/all",
                    params={"week": week, "year": year}
                )

            elif name == "check_data_freshness":
                response = await client.get(f"{API_BASE}/refresh/check")

            elif name == "auto_refresh":
                week = arguments.get("week", 12)
                year = arguments.get("year", 2024)
                force = arguments.get("force", False)
                response = await client.post(
                    f"{API_BASE}/refresh/auto",
                    params={"week": week, "year": year, "force": force}
                )

            # ========== QUERY TOOLS ==========
            elif name == "get_status":
                response = await client.get(f"{API_BASE}/")

            elif name == "quick_props":
                min_edge = arguments.get("min_edge", 5.0)
                limit = arguments.get("limit", 10)
                prop_types = arguments.get("prop_types")
                params = {"min_edge": min_edge, "limit": limit}
                if prop_types:
                    params["prop_types"] = prop_types
                response = await client.get(
                    f"{API_BASE}/analysis/quick-props",
                    params=params
                )

            elif name == "game_deep_dive":
                game_id = arguments.get("game_id", "")
                response = await client.get(f"{API_BASE}/analysis/game/{game_id}")

            # ========== COMPREHENSIVE INTELLIGENCE ==========
            elif name == "full_matchup_analysis":
                game_id = arguments.get("game_id", "")
                response = await client.get(
                    f"{API_BASE}/intelligence/matchup/{game_id}"
                )

            elif name == "daily_betting_brief":
                week = arguments.get("week", 12)
                min_edge = arguments.get("min_edge", 3.0)
                do_refresh = arguments.get("auto_refresh", True)
                response = await client.get(
                    f"{API_BASE}/intelligence/daily-brief",
                    params={
                        "week": week,
                        "min_edge": min_edge,
                        "auto_refresh": do_refresh
                    }
                )

            elif name == "player_outlook":
                player_name = arguments.get("player_name", "")
                response = await client.get(
                    f"{API_BASE}/intelligence/player/{player_name}"
                )

            elif name == "get_line_movement":
                player_name = arguments.get("player_name", "")
                prop_type = arguments.get("prop_type", "")
                days = arguments.get("days", 7)
                response = await client.get(
                    f"{API_BASE}/odds/movement",
                    params={
                        "player_name": player_name,
                        "prop_type": prop_type,
                        "days": days
                    }
                )

            elif name == "get_hot_movers":
                min_movement = arguments.get("min_movement", 1.5)
                hours = arguments.get("hours", 48)
                response = await client.get(
                    f"{API_BASE}/odds/movers",
                    params={"min_movement": min_movement, "hours": hours}
                )

            elif name == "get_latest_odds":
                params = {}
                if arguments.get("game_id"):
                    params["game_id"] = arguments["game_id"]
                if arguments.get("player_name"):
                    params["player_name"] = arguments["player_name"]
                if arguments.get("prop_type"):
                    params["prop_type"] = arguments["prop_type"]
                response = await client.get(f"{API_BASE}/odds/latest", params=params)

            elif name == "get_latest_projections":
                params = {}
                if arguments.get("game_id"):
                    params["game_id"] = arguments["game_id"]
                if arguments.get("player_name"):
                    params["player_name"] = arguments["player_name"]
                if arguments.get("prop_type"):
                    params["prop_type"] = arguments["prop_type"]
                response = await client.get(
                    f"{API_BASE}/projections/latest",
                    params=params
                )

            elif name == "get_injuries":
                params = {}
                if arguments.get("team"):
                    params["team"] = arguments["team"]
                if arguments.get("status"):
                    params["status"] = arguments["status"]
                response = await client.get(
                    f"{API_BASE}/injuries/latest",
                    params=params
                )

            elif name == "get_games":
                week = arguments.get("week")
                season = arguments.get("season", 2024)
                params = {"season": season}
                if week:
                    params["week"] = week
                response = await client.get(f"{API_BASE}/games", params=params)

            elif name == "get_value_props_history":
                days = arguments.get("days", 7)
                min_edge = arguments.get("min_edge", 0)
                response = await client.get(
                    f"{API_BASE}/value-props/history",
                    params={"days": days, "min_edge": min_edge}
                )

            elif name == "get_model_runs":
                limit = arguments.get("limit", 20)
                response = await client.get(
                    f"{API_BASE}/model/runs",
                    params={"limit": limit}
                )

            # ========== STATS/KNOWLEDGE TOOLS ==========
            elif name == "get_player_stats":
                player_name = arguments.get("player_name", "")
                season = arguments.get("season", 2025)
                response = await client.get(
                    f"{API_BASE}/stats/player/{player_name}",
                    params={"season": season}
                )

            elif name == "get_team_profile":
                team = arguments.get("team", "")
                season = arguments.get("season", 2025)
                response = await client.get(
                    f"{API_BASE}/stats/team/{team}",
                    params={"season": season}
                )

            elif name == "get_league_leaders":
                stat_type = arguments.get("stat_type", "passing_yards")
                season = arguments.get("season", 2025)
                limit = arguments.get("limit", 20)
                response = await client.get(
                    f"{API_BASE}/stats/leaders/{stat_type}",
                    params={"season": season, "limit": limit}
                )

            elif name == "get_schedule":
                season = arguments.get("season", 2025)
                week = arguments.get("week")
                params = {"season": season}
                if week:
                    params["week"] = week
                response = await client.get(f"{API_BASE}/stats/schedule", params=params)

            elif name == "get_team_rankings":
                season = arguments.get("season", 2025)
                response = await client.get(
                    f"{API_BASE}/stats/rankings",
                    params={"season": season}
                )

            elif name == "populate_database":
                season = arguments.get("season", 2025)
                week = arguments.get("week", 12)
                fetch_first = arguments.get("fetch_first", False)
                response = await client.post(
                    f"{API_BASE}/populate/all",
                    params={
                        "season": season,
                        "week": week,
                        "fetch_first": fetch_first,
                        "include_odds": True
                    },
                    timeout=180.0  # 3 minutes for full population
                )

            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Unknown tool: {name}"})
                )]

            # Handle response
            if response.status_code == 200:
                result = response.json()
                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]
            else:
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
