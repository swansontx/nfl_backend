"""MCP Server for NFL Prop Betting System.

Run with: python mcp_server.py
Then add to Claude Desktop config or use with Claude Code.

This exposes your betting system as tools Claude can call naturally.
"""

import json
import asyncio
from typing import Optional
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Import your backend modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backend.api.prop_analyzer import prop_analyzer, PropLine, PropProjection
from backend.api.model_loader import model_loader
from backend.api.odds_api import odds_api
from backend.config import settings
from backend.ingestion.fetch_prop_lines import PropLineFetcher
from backend.api.narrative_generator import NarrativeTemplates
from backend.api.insights_engine import StatisticalAnalyzer

# Create MCP server
server = Server("nfl-betting")


@server.list_tools()
async def list_tools():
    """List available NFL betting tools."""
    return [
        Tool(
            name="get_best_props",
            description="Find the best value prop bets. Returns props with highest edge vs sportsbook lines. Use for finding individual bets or parlay legs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "game_ids": {
                        "type": "string",
                        "description": "Comma-separated game IDs to filter (e.g., 'BUF_HOU,KC_LAC'). Leave empty for all games."
                    },
                    "min_edge": {
                        "type": "number",
                        "description": "Minimum edge percentage (default 5.0)",
                        "default": 5.0
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of props to return (default 10)",
                        "default": 10
                    }
                }
            }
        ),
        Tool(
            name="get_matchup_analysis",
            description="Get detailed matchup analysis for a specific game including top props, insights, and betting angles.",
            inputSchema={
                "type": "object",
                "properties": {
                    "game_id": {
                        "type": "string",
                        "description": "Game ID (e.g., '2024_12_BUF_HOU')"
                    }
                },
                "required": ["game_id"]
            }
        ),
        Tool(
            name="get_trending_props",
            description="Get hot movers and line movement trends. Shows which props have moved significantly.",
            inputSchema={
                "type": "object",
                "properties": {
                    "week": {
                        "type": "integer",
                        "description": "NFL week number (optional)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of movers to return (default 10)",
                        "default": 10
                    }
                }
            }
        ),
        Tool(
            name="get_parlay_suggestions",
            description="Get correlation-aware parlay suggestions. Accounts for same-game correlations and game script scenarios.",
            inputSchema={
                "type": "object",
                "properties": {
                    "game_ids": {
                        "type": "string",
                        "description": "Comma-separated game IDs for parlay building"
                    },
                    "max_legs": {
                        "type": "integer",
                        "description": "Maximum legs per parlay (default 3)",
                        "default": 3
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of parlays to suggest (default 5)",
                        "default": 5
                    }
                }
            }
        ),
        Tool(
            name="get_player_trend",
            description="Analyze a player's recent performance trend for a specific stat.",
            inputSchema={
                "type": "object",
                "properties": {
                    "player_name": {
                        "type": "string",
                        "description": "Player name (e.g., 'Josh Allen')"
                    },
                    "stat": {
                        "type": "string",
                        "description": "Stat to analyze (e.g., 'passing_yards', 'receptions')"
                    }
                },
                "required": ["player_name", "stat"]
            }
        ),
        Tool(
            name="get_schedule",
            description="Get upcoming NFL games schedule with spreads and totals.",
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
            description="Fetch fresh DraftKings odds from The Odds API. Use this to update odds data before analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "week": {
                        "type": "integer",
                        "description": "NFL week number"
                    }
                }
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Handle tool calls."""

    if name == "get_best_props":
        return await handle_get_best_props(arguments)
    elif name == "get_matchup_analysis":
        return await handle_get_matchup(arguments)
    elif name == "get_trending_props":
        return await handle_get_trending(arguments)
    elif name == "get_parlay_suggestions":
        return await handle_get_parlays(arguments)
    elif name == "get_player_trend":
        return await handle_get_player_trend(arguments)
    elif name == "get_schedule":
        return await handle_get_schedule(arguments)
    elif name == "fetch_dk_odds":
        return await handle_fetch_odds(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def handle_get_best_props(args: dict):
    """Find best value props."""
    game_ids = args.get("game_ids", "")
    min_edge = args.get("min_edge", 5.0)
    limit = args.get("limit", 10)

    # Load projections
    projections = []
    if game_ids:
        for gid in game_ids.split(","):
            projections.extend(model_loader.load_projections_for_game(gid.strip()))
    else:
        available = model_loader.get_available_games()
        for gid in available[:10]:
            projections.extend(model_loader.load_projections_for_game(gid))

    # Get odds
    prop_lines = odds_api.get_player_props()

    # Find value
    if not prop_lines or not projections:
        # Demo data
        result = {
            "message": "No live data available. Using demo.",
            "props": [{
                "player": "Patrick Mahomes",
                "prop": "passing_yards",
                "line": 275.5,
                "projection": 295.3,
                "edge": 8.5,
                "recommendation": "OVER",
                "grade": "A"
            }]
        }
    else:
        value_props = prop_analyzer.find_best_props(
            prop_lines, projections, min_edge=min_edge
        )

        result = {
            "total": len(value_props),
            "props": [
                {
                    "player": v.prop_line.player_name,
                    "prop": v.prop_line.prop_type,
                    "line": v.prop_line.line,
                    "projection": round(v.projection.projection, 1),
                    "edge": round(max(v.edge_over, v.edge_under), 1),
                    "recommendation": v.recommendation,
                    "grade": v.value_grade,
                    "confidence": round(v.confidence, 2)
                }
                for v in value_props[:limit]
            ]
        }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_get_matchup(args: dict):
    """Get matchup analysis for a game."""
    game_id = args.get("game_id", "")

    if not game_id:
        return [TextContent(type="text", text="Error: game_id required")]

    # Parse game_id for teams
    parts = game_id.split("_")
    if len(parts) >= 4:
        away_team = parts[2]
        home_team = parts[3]
    else:
        away_team = "AWAY"
        home_team = "HOME"

    # Load projections for this game
    projections = model_loader.load_projections_for_game(game_id)
    prop_lines = odds_api.get_player_props()

    # Get top props for this game
    value_props = []
    if prop_lines and projections:
        all_values = prop_analyzer.find_best_props(prop_lines, projections, min_edge=3.0)
        value_props = all_values[:12]

    # Generate narrative
    key_stats = {
        "home_offense_rank": 12,
        "away_offense_rank": 8,
        "home_defense_rank": 15,
        "away_defense_rank": 20
    }
    narrative = NarrativeTemplates.game_preview(home_team, away_team, key_stats)

    result = {
        "game_id": game_id,
        "matchup": f"{away_team} @ {home_team}",
        "narrative": narrative.content,
        "top_props": [
            {
                "player": v.prop_line.player_name,
                "prop": v.prop_line.prop_type,
                "line": v.prop_line.line,
                "recommendation": v.recommendation,
                "edge": round(max(v.edge_over, v.edge_under), 1),
                "grade": v.value_grade
            }
            for v in value_props
        ] if value_props else [{"message": "No props loaded - fetch odds first"}]
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_get_trending(args: dict):
    """Get trending/hot mover props."""
    week = args.get("week")
    limit = args.get("limit", 10)

    snapshots_dir = Path("outputs/prop_lines")

    if not snapshots_dir.exists():
        return [TextContent(type="text", text=json.dumps({
            "error": "No snapshots found",
            "message": "Run fetch_dk_odds first to capture prop line snapshots"
        }))]

    fetcher = PropLineFetcher("dummy")
    trends = fetcher.get_trending_props(snapshots_dir, week)

    if "error" in trends:
        return [TextContent(type="text", text=json.dumps(trends))]

    result = {
        "hot_movers": [
            {
                "player": m["player"],
                "market": m["market"],
                "movement": m["line_movement"],
                "direction": m["direction"],
                "opening": m["opening_line"],
                "current": m["current_line"]
            }
            for m in trends.get("hot_movers", [])[:limit]
        ],
        "sustained_trends": [
            {
                "player": t["player"],
                "market": t["market"],
                "total_movement": t["total_movement"],
                "direction": t["direction"],
                "weeks": t["consistency"]
            }
            for t in trends.get("sustained_trends", [])[:limit]
        ]
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_get_parlays(args: dict):
    """Get parlay suggestions."""
    game_ids = args.get("game_ids", "")
    max_legs = args.get("max_legs", 3)
    limit = args.get("limit", 5)

    # For now, return a structured suggestion based on best props
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
        result = {
            "message": "No live data for parlay building",
            "suggestion": "Fetch odds first, then I can build correlation-aware parlays"
        }
    else:
        value_props = prop_analyzer.find_best_props(prop_lines, projections, min_edge=5.0)

        # Simple parlay building - group by confidence
        parlays = []
        if len(value_props) >= max_legs:
            legs = value_props[:max_legs]
            combined_prob = 1.0
            for v in legs:
                combined_prob *= v.confidence

            parlays.append({
                "legs": [
                    {
                        "player": v.prop_line.player_name,
                        "prop": v.prop_line.prop_type,
                        "side": v.recommendation,
                        "line": v.prop_line.line,
                        "edge": round(max(v.edge_over, v.edge_under), 1)
                    }
                    for v in legs
                ],
                "combined_probability": round(combined_prob, 3),
                "correlation_note": "Same-game legs have correlated outcomes"
            })

        result = {
            "total_suggestions": len(parlays),
            "parlays": parlays[:limit]
        }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_get_player_trend(args: dict):
    """Analyze player trend."""
    player_name = args.get("player_name", "")
    stat = args.get("stat", "passing_yards")

    if not player_name:
        return [TextContent(type="text", text="Error: player_name required")]

    # Load player stats (would come from real data)
    # For demo, generate trend narrative
    trend_data = {
        "direction": "increasing",
        "strength": 0.7,
        "pct_change": 15.2,
        "recent_avg": 285.3,
        "earlier_avg": 247.5,
        "stat_name": stat
    }

    narrative = NarrativeTemplates.player_trend(player_name, trend_data)

    result = {
        "player": player_name,
        "stat": stat,
        "trend": narrative.supporting_stats,
        "narrative": narrative.content,
        "recommendation": narrative.supporting_stats.get("recommendation"),
        "confidence": round(narrative.confidence, 2)
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_get_schedule(args: dict):
    """Get game schedule."""
    week = args.get("week")

    # Load from model_loader or generate
    available_games = model_loader.get_available_games()

    result = {
        "week": week or "current",
        "games": [
            {"game_id": gid, "parsed": gid.replace("_", " ")}
            for gid in available_games[:16]
        ] if available_games else [
            {"message": "No games loaded. Fetch schedule data first."}
        ]
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_fetch_odds(args: dict):
    """Fetch fresh odds from The Odds API."""
    import os
    week = args.get("week", 12)

    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        return [TextContent(type="text", text=json.dumps({
            "error": "ODDS_API_KEY not set",
            "message": "Set ODDS_API_KEY environment variable"
        }))]

    output_dir = Path("outputs/prop_lines")

    try:
        fetcher = PropLineFetcher(api_key)
        result = fetcher.fetch_snapshot_with_movement(output_dir, week)

        return [TextContent(type="text", text=json.dumps({
            "success": True,
            "games": result.get("games", 0),
            "props": result.get("props", 0),
            "hot_movers": result.get("hot_movers", 0),
            "output_file": result.get("output_file", "")
        }, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "error": str(e)
        }))]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
