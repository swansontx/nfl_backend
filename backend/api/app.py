from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime

from backend.api.external_apis import weather_api, sleeper_api
from backend.api.insights_engine import insight_generator, PlayerStats, Insight as InsightData
from backend.api.narrative_generator import narrative_generator
from backend.api.prop_analyzer import prop_analyzer, PropLine, PropProjection, PropValue
from backend.api.odds_api import odds_api
from backend.api.model_loader import model_loader
from backend.api.team_database import get_team, get_all_teams, get_division_teams, get_conference_teams
from backend.api.schedule_loader import schedule_loader, Game
from backend.api.boxscore_generator import boxscore_generator

app = FastAPI(
    title='NFL Props Backend API',
    version='1.0.0',
    description='Backend API for NFL props predictions, insights, and content'
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Pydantic Models
# ============================================================================

class RecomputeRequest(BaseModel):
    game_id: str


class InjuryRecord(BaseModel):
    player_id: str
    player_name: str
    team: str
    position: str
    injury_status: str
    injury_body_part: Optional[str] = None
    injury_notes: Optional[str] = None
    last_updated: str


class NewsItem(BaseModel):
    id: str
    title: str
    summary: str
    source: str
    published_at: str
    url: Optional[str] = None
    category: str  # "injury", "news", "analysis"
    related_players: List[str] = []
    related_teams: List[str] = []


class WeatherData(BaseModel):
    temperature: int
    temp_unit: str
    condition: str
    wind_speed: int
    wind_unit: str
    humidity: int
    precipitation_chance: int
    is_dome: bool


class MatchupInsight(BaseModel):
    insight_type: str  # "trend", "stat", "matchup"
    title: str
    description: str
    confidence: float
    supporting_data: Dict


class GameNarrative(BaseModel):
    narrative_type: str  # "preview", "key_matchups", "betting_angle"
    content: str
    generated_at: str


class ContentItem(BaseModel):
    content_type: str  # "article", "video", "podcast"
    title: str
    source: str
    url: str
    published_at: str
    thumbnail_url: Optional[str] = None


# ============================================================================
# Core Endpoints (Existing)
# ============================================================================

@app.get('/health')
async def health():
    """Health check endpoint."""
    return {'status': 'ok', 'timestamp': datetime.now().isoformat()}


@app.post('/admin/recompute')
async def recompute(req: RecomputeRequest):
    """Trigger model recomputation for a specific game."""
    # TODO: Integrate with orchestration pipeline
    return {'status': 'started', 'game_id': req.game_id}


@app.get('/admin/odds-api-usage')
async def check_odds_api_usage():
    """Check Odds API usage and remaining requests."""
    usage = odds_api.check_usage()
    return usage


@app.post('/admin/create-sample-projections/{game_id}')
async def create_sample_projections(game_id: str):
    """Create sample projection file for development/testing."""
    model_loader.create_sample_projections(game_id)
    return {
        'status': 'created',
        'game_id': game_id,
        'file': f'outputs/predictions/props_{game_id}.csv'
    }


@app.get('/game/{game_id}/projections')
async def get_projections(game_id: str):
    """Get player prop projections for a game."""
    projections = model_loader.load_projections_for_game(game_id)

    return {
        'game_id': game_id,
        'total_projections': len(projections),
        'projections': [
            {
                'player_id': p.player_id,
                'player_name': p.player_name,
                'prop_type': p.prop_type,
                'projection': p.projection,
                'std_dev': p.std_dev,
                'confidence_interval': p.confidence_interval
            }
            for p in projections
        ]
    }


# ============================================================================
# Phase 1: News & Injuries (Core Frontend Features)
# ============================================================================

@app.get('/api/v1/news', response_model=List[NewsItem])
async def get_news(
    limit: int = 20,
    category: Optional[str] = None,
    team: Optional[str] = None
):
    """Get latest NFL news and injury updates.

    Args:
        limit: Number of items to return (default 20)
        category: Filter by category ("injury", "news", "analysis")
        team: Filter by team abbreviation

    Returns:
        List of news items with injuries prominently featured
    """
    # TODO: Integrate with actual news API or RSS feeds
    # Sources could include:
    # - ESPN API
    # - NFL.com RSS
    # - Twitter API for beat reporters
    # - FantasyPros news feed

    # Get current injuries from Sleeper
    injuries = sleeper_api.get_injuries()

    # Convert injuries to news items
    news_items = []
    for injury in injuries[:limit]:
        if team and injury['team'] != team:
            continue

        news_items.append(NewsItem(
            id=f"injury_{injury['player_id']}",
            title=f"{injury['player_name']} - {injury['injury_status']}",
            summary=f"{injury['injury_body_part']} - {injury['injury_notes']}",
            source="Sleeper API",
            published_at=injury['last_updated'],
            category="injury",
            related_players=[injury['player_name']],
            related_teams=[injury['team']]
        ))

    # TODO: Add non-injury news items
    # Placeholder for demonstration
    if not category or category == "news":
        news_items.append(NewsItem(
            id="news_001",
            title="NFL Week 12 Preview",
            summary="Key matchups and storylines for Week 12",
            source="NFL.com",
            published_at=datetime.now().isoformat(),
            category="news",
            related_players=[],
            related_teams=[]
        ))

    return news_items[:limit]


@app.get('/api/v1/games/{game_id}/injuries')
async def get_game_injuries(game_id: str):
    """Get injury report for both teams in a specific game.

    Args:
        game_id: Game ID in format {season}_{week}_{away}_{home}

    Returns:
        Injury data for both teams including:
        - Current injury status
        - Injury designation (Out, Doubtful, Questionable, Probable)
        - Body part and notes
        - Impact on team
    """
    try:
        injury_data = sleeper_api.get_injuries_for_game(game_id)

        return {
            'game_id': game_id,
            'away_team': injury_data['away_team'],
            'home_team': injury_data['home_team'],
            'away_injuries': injury_data['away_injuries'],
            'home_injuries': injury_data['home_injuries'],
            'last_updated': datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching injuries: {str(e)}")


# ============================================================================
# Phase 2: Enhanced Features (ML Insights & Narratives)
# ============================================================================

@app.get('/api/v1/games/{game_id}/insights', response_model=List[MatchupInsight])
async def get_game_insights(game_id: str):
    """Get ML-powered matchup insights for a game.

    Insights include:
    - Historical performance trends
    - Matchup advantages/disadvantages
    - Key statistical edges
    - Weather impact analysis
    - Injury impact analysis

    Args:
        game_id: Game ID in format {season}_{week}_{away}_{home}
    """
    # TODO: Integrate with ML models and feature analysis
    # This should pull from:
    # - backend/features/ for player trends
    # - backend/modeling/ for predictions
    # - Historical matchup data
    # - Weather data

    insights = []

    # Example insights (placeholder)
    insights.append(MatchupInsight(
        insight_type="trend",
        title="QB Performance vs. Defense",
        description="Away team QB has averaged 285 passing yards vs. similar defenses",
        confidence=0.82,
        supporting_data={
            "avg_yards": 285,
            "sample_size": 6,
            "trend": "increasing"
        }
    ))

    insights.append(MatchupInsight(
        insight_type="matchup",
        title="Run Defense Vulnerability",
        description="Home team allows 4.8 yards per carry, 2nd worst in league",
        confidence=0.91,
        supporting_data={
            "ypc_allowed": 4.8,
            "league_rank": 30,
            "last_3_games": 5.2
        }
    ))

    # Add weather insight if applicable
    # TODO: Get stadium location from game_id/schedule
    # weather = weather_api.get_game_weather(lat, lon, game_time)
    # if weather['wind_speed'] > 15:
    #     insights.append(weather_impact_insight)

    return insights


@app.get('/api/v1/games/{game_id}/narrative', response_model=List[GameNarrative])
async def get_game_narrative(game_id: str):
    """Get AI-generated game narratives and storylines.

    Narratives include:
    - Game preview
    - Key player matchups
    - Betting angles and value props
    - Historical context
    - Weather and external factors

    Args:
        game_id: Game ID in format {season}_{week}_{away}_{home}
    """
    # TODO: Integrate with LLM (OpenAI, Claude, etc.)
    # This should generate narratives based on:
    # - Team stats and trends
    # - Player props and projections
    # - Injury reports
    # - Weather conditions
    # - Historical matchups

    narratives = []

    # Placeholder narratives
    narratives.append(GameNarrative(
        narrative_type="preview",
        content=(
            "This AFC showdown features two high-powered offenses. "
            "The away team's passing attack ranks 2nd in the league, "
            "while the home team's defense has struggled against elite QBs."
        ),
        generated_at=datetime.now().isoformat()
    ))

    narratives.append(GameNarrative(
        narrative_type="key_matchups",
        content=(
            "Watch for the battle in the trenches. The away team's offensive line "
            "has allowed just 12 sacks this season, while the home team's pass rush "
            "leads the league with 42 sacks. This matchup will dictate the game flow."
        ),
        generated_at=datetime.now().isoformat()
    ))

    narratives.append(GameNarrative(
        narrative_type="betting_angle",
        content=(
            "Value opportunity on the away team's RB receiving props. "
            "He's averaged 6.2 receptions in games where the team is favored, "
            "and books have his line at 4.5. Weather forecast shows clear conditions."
        ),
        generated_at=datetime.now().isoformat()
    ))

    return narratives


# ============================================================================
# Phase 3: Content Aggregation
# ============================================================================

@app.get('/api/v1/games/{game_id}/content', response_model=List[ContentItem])
async def get_game_content(
    game_id: str,
    content_type: Optional[str] = None,
    limit: int = 10
):
    """Get aggregated content (articles, videos, podcasts) for a game.

    Args:
        game_id: Game ID in format {season}_{week}_{away}_{home}
        content_type: Filter by type ("article", "video", "podcast")
        limit: Number of items to return

    Returns:
        Aggregated content from various sources
    """
    # TODO: Integrate with content APIs:
    # - YouTube API for game previews/highlights
    # - RSS feeds from major sports sites
    # - Podcast APIs (Apple Podcasts, Spotify)
    # - Twitter for embedded video content

    content_items = []

    # Placeholder content
    content_items.append(ContentItem(
        content_type="article",
        title="Week 12 Preview: Key Matchups to Watch",
        source="ESPN",
        url="https://espn.com/nfl/preview",
        published_at=datetime.now().isoformat(),
        thumbnail_url="https://placeholder.com/thumbnail.jpg"
    ))

    content_items.append(ContentItem(
        content_type="video",
        title="Film Breakdown: Offensive Schemes",
        source="YouTube",
        url="https://youtube.com/watch?v=example",
        published_at=datetime.now().isoformat(),
        thumbnail_url="https://placeholder.com/video-thumb.jpg"
    ))

    if content_type:
        content_items = [
            item for item in content_items
            if item.content_type == content_type
        ]

    return content_items[:limit]


# ============================================================================
# Bonus: Weather Endpoint
# ============================================================================

@app.get('/api/v1/games/{game_id}/weather', response_model=WeatherData)
async def get_game_weather(game_id: str):
    """Get weather forecast for a game.

    Args:
        game_id: Game ID in format {season}_{week}_{away}_{home}

    Returns:
        Weather forecast including temperature, wind, precipitation

    Note: Now using stadium database for accurate coordinates!
          Automatically handles dome stadiums (returns controlled conditions)
    """
    # Use new convenience method with stadium database
    weather_data = weather_api.get_weather_for_game(game_id)

    return WeatherData(**weather_data)


# ============================================================================
# Enhanced Endpoints: Prop Value & Player Insights
# ============================================================================

@app.get('/api/v1/props/value')
async def find_prop_value(
    game_id: Optional[str] = None,
    player_id: Optional[str] = None,
    min_edge: float = 5.0,
    min_grade: str = "B",
    limit: int = 10
):
    """Find high-value prop bets.

    Args:
        game_id: Filter by game ID (optional)
        player_id: Filter by player ID (optional)
        min_edge: Minimum edge percentage (default 5%)
        min_grade: Minimum value grade (default "B")
        limit: Number of results to return

    Returns:
        List of high-value prop recommendations with:
        - Sportsbook lines vs model projections
        - Expected value calculations
        - Edge percentages
        - Bet sizing recommendations
        - Value grades (A+, A, B+, B, C, F)
    """
    # Load real sportsbook lines from The Odds API
    prop_lines = odds_api.get_player_props()

    # Filter by game_id if provided
    if game_id:
        prop_lines = [line for line in prop_lines if game_id in line.timestamp]  # TODO: Better filtering

    # Load real model projections
    projections = []
    if game_id:
        projections = model_loader.load_projections_for_game(game_id)
    else:
        # Load all recent projections if no game_id specified
        available_games = model_loader.get_available_games()
        for gid in available_games[:5]:  # Limit to 5 most recent games
            projections.extend(model_loader.load_projections_for_game(gid))

    # If no real data available, use sample data for demo
    if not prop_lines or not projections:
        print("No real odds or projections found, using sample data")
        prop_lines = [
            PropLine(
                player_id="player_001",
                player_name="Patrick Mahomes",
                prop_type="passing_yards",
                line=275.5,
                over_odds=-110,
                under_odds=-110,
                book="DraftKings",
                timestamp=datetime.now().isoformat()
            )
        ]
        projections = [
            PropProjection(
                player_id="player_001",
                player_name="Patrick Mahomes",
                prop_type="passing_yards",
                projection=295.3,
                std_dev=42.5,
                confidence_interval=(252.8, 337.8),
                hit_probability_over=0.68,
                hit_probability_under=0.32
            )
        ]

    # Find value props using real data
    value_props = prop_analyzer.find_best_props(
        prop_lines,
        projections,
        min_edge=min_edge,
        min_grade=min_grade
    )

    # Convert to response format
    results = []
    for value in value_props[:limit]:
        stake = prop_analyzer.calculate_kelly_stake(
            max(value.edge_over, value.edge_under),
            value.confidence,
            bankroll=1000  # Placeholder bankroll
        )

        results.append({
            "player_name": value.prop_line.player_name,
            "prop_type": value.prop_line.prop_type,
            "sportsbook_line": value.prop_line.line,
            "model_projection": value.projection.projection,
            "confidence_interval": value.projection.confidence_interval,
            "recommendation": value.recommendation,
            "edge_over": round(value.edge_over, 2),
            "edge_under": round(value.edge_under, 2),
            "value_grade": value.value_grade,
            "confidence": round(value.confidence, 3),
            "suggested_stake_pct": round(stake / 1000 * 100, 2),  # As percentage
            "sportsbook": value.prop_line.book,
            "odds": value.prop_line.over_odds if value.recommendation == "OVER" else value.prop_line.under_odds
        })

    return {
        "total_opportunities": len(results),
        "best_values": results,
        "filters_applied": {
            "min_edge": min_edge,
            "min_grade": min_grade
        }
    }


@app.get('/api/v1/players/{player_id}/insights')
async def get_player_insights(player_id: str):
    """Get comprehensive insights for a specific player.

    Args:
        player_id: Player ID

    Returns:
        Player-specific insights including:
        - Recent performance trends
        - Statistical consistency
        - Matchup history
        - Prop recommendations
    """
    # TODO: Load actual player data from database/files
    # Placeholder player data
    sample_player = PlayerStats(
        player_id=player_id,
        player_name="Patrick Mahomes",
        position="QB",
        team="KC",
        recent_games=[
            {"passing_yards": 320, "passing_tds": 3},
            {"passing_yards": 295, "passing_tds": 2},
            {"passing_yards": 278, "passing_tds": 2},
            {"passing_yards": 310, "passing_tds": 4},
            {"passing_yards": 285, "passing_tds": 2}
        ],
        season_avg={"passing_yards": 297.6, "passing_tds": 2.6}
    )

    # Generate insights
    insights = []

    # Trend insight
    trend_insight = insight_generator.generate_player_trend_insight(
        sample_player,
        "passing_yards"
    )
    if trend_insight:
        insights.append({
            "insight_type": trend_insight.insight_type,
            "title": trend_insight.title,
            "description": trend_insight.description,
            "confidence": trend_insight.confidence,
            "impact_level": trend_insight.impact_level,
            "recommendation": trend_insight.recommendation,
            "supporting_data": trend_insight.supporting_data
        })

    return {
        "player_id": player_id,
        "player_name": sample_player.player_name,
        "position": sample_player.position,
        "team": sample_player.team,
        "insights": insights,
        "season_avg": sample_player.season_avg,
        "recent_performance": sample_player.recent_games[-5:]
    }


@app.get('/api/v1/props/compare')
async def compare_props(
    player_ids: str,  # Comma-separated
    prop_type: str = "passing_yards"
):
    """Compare props across multiple players.

    Args:
        player_ids: Comma-separated list of player IDs
        prop_type: Type of prop to compare

    Returns:
        Comparative analysis of players for the specified prop
    """
    player_id_list = player_ids.split(',')

    # TODO: Load actual player projections
    # Placeholder data
    comparisons = []
    for pid in player_id_list[:5]:  # Limit to 5 players
        comparisons.append({
            "player_id": pid,
            "player_name": f"Player {pid}",
            "projection": 275.5,
            "std_dev": 35.2,
            "recent_avg": 282.3,
            "trend": "increasing",
            "matchup_grade": "B+",
            "recommendation": "Consider OVER"
        })

    return {
        "prop_type": prop_type,
        "players_compared": len(comparisons),
        "comparisons": comparisons,
        "best_value": comparisons[0] if comparisons else None
    }


@app.get('/api/v1/games/{game_id}/prop-sheet')
async def get_game_prop_sheet(game_id: str):
    """Get comprehensive prop sheet for a game.

    Args:
        game_id: Game ID in format {season}_{week}_{away}_{home}

    Returns:
        Complete prop sheet with:
        - All available props
        - Model projections
        - Value assessments
        - Recommended plays
    """
    # TODO: Aggregate all props for the game
    # TODO: Generate projections for all players
    # TODO: Calculate value for each prop

    return {
        "game_id": game_id,
        "total_props": 150,
        "high_value_props": 12,
        "categories": {
            "passing": 50,
            "rushing": 35,
            "receiving": 45,
            "scoring": 20
        },
        "top_plays": [
            {
                "player": "Player A",
                "prop": "passing_yards OVER 285.5",
                "edge": 8.5,
                "grade": "A"
            },
            {
                "player": "Player B",
                "prop": "receiving_yards UNDER 72.5",
                "edge": 6.2,
                "grade": "B+"
            }
        ]
    }


# ============================================================================
# Team Pages Endpoints
# ============================================================================

@app.get('/api/v1/teams', tags=['Teams'])
def get_teams(conference: Optional[str] = None, division: Optional[str] = None):
    """Get all NFL teams or filter by conference/division.

    Args:
        conference: Optional conference filter (AFC or NFC)
        division: Optional division filter (e.g., 'AFC East')

    Returns:
        List of teams with full metadata
    """
    if division:
        teams = get_division_teams(division)
        return {'teams': teams, 'count': len(teams)}
    elif conference:
        teams = get_conference_teams(conference.upper())
        return {'teams': teams, 'count': len(teams)}
    else:
        teams = get_all_teams()
        return {'teams': list(teams.values()), 'count': len(teams)}


@app.get('/api/v1/teams/{team_id}', tags=['Teams'])
def get_team_info(team_id: str):
    """Get detailed info for a specific team.

    Args:
        team_id: Team abbreviation (e.g., 'KC', 'BUF')

    Returns:
        Team info with stadium, colors, historical records
    """
    team = get_team(team_id.upper())

    if not team:
        raise HTTPException(status_code=404, detail=f'Team not found: {team_id}')

    return team


@app.get('/api/v1/teams/{team_id}/schedule', tags=['Teams'])
def get_team_schedule_endpoint(
    team_id: str,
    year: Optional[int] = None,
    game_type: str = 'REG'
):
    """Get schedule for a team.

    Args:
        team_id: Team abbreviation
        year: Season year (defaults to current year)
        game_type: Game type filter ('REG', 'POST', 'PRE', 'ALL')

    Returns:
        List of games for this team
    """
    if year is None:
        year = datetime.now().year

    team = get_team(team_id.upper())
    if not team:
        raise HTTPException(status_code=404, detail=f'Team not found: {team_id}')

    schedule = schedule_loader.get_team_schedule(team_id.upper(), year, game_type)

    # Convert Game objects to dicts
    schedule_dicts = []
    for game in schedule:
        game_dict = {
            'game_id': game.game_id,
            'season': game.season,
            'week': game.week,
            'game_type': game.game_type,
            'gameday': game.gameday,
            'weekday': game.weekday,
            'gametime': game.gametime,
            'away_team': game.away_team,
            'home_team': game.home_team,
            'away_score': game.away_score,
            'home_score': game.home_score,
            'result': game.result,
            'is_home': game.home_team == team_id.upper(),
            'opponent': game.away_team if game.home_team == team_id.upper() else game.home_team,
            'location': game.location,
            'stadium': game.stadium,
        }
        schedule_dicts.append(game_dict)

    return {
        'team': team,
        'season': year,
        'game_type': game_type,
        'games': schedule_dicts,
        'count': len(schedule_dicts)
    }


@app.get('/api/v1/teams/{team_id}/news', tags=['Teams'])
def get_team_news(team_id: str, limit: int = 20):
    """Get news for a specific team.

    Args:
        team_id: Team abbreviation
        limit: Max number of news items

    Returns:
        News items filtered for this team
    """
    team = get_team(team_id.upper())
    if not team:
        raise HTTPException(status_code=404, detail=f'Team not found: {team_id}')

    # Get all news and filter by team
    all_news = sleeper_api.get_injuries()

    team_news = [
        item for item in all_news
        if item.get('team', '').upper() == team_id.upper()
    ]

    return {
        'team': team,
        'news': team_news[:limit],
        'count': len(team_news[:limit])
    }


@app.get('/api/v1/games/{game_id}/boxscore', tags=['Games'])
def get_game_boxscore(game_id: str):
    """Get complete boxscore for a game (similar to ESPN boxscore).

    Args:
        game_id: Game ID (format: YYYY_WW_AWAY_HOME)

    Returns:
        Comprehensive boxscore with team stats, player stats, and scoring summary

    Example:
        /api/v1/games/2024_10_KC_BUF/boxscore
    """
    boxscore = boxscore_generator.generate_boxscore(game_id)

    if not boxscore:
        raise HTTPException(status_code=404, detail=f'Boxscore not found for game: {game_id}')

    # Convert to dict for JSON response
    def player_stats_to_dict(stats):
        return {
            'player_id': stats.player_id,
            'player_name': stats.player_name,
            'team': stats.team,
            'passing': {
                'attempts': stats.pass_attempts,
                'completions': stats.pass_completions,
                'yards': stats.pass_yards,
                'touchdowns': stats.pass_tds,
                'interceptions': stats.interceptions,
                'sacks': stats.sacks,
                'sack_yards': stats.sack_yards,
            },
            'rushing': {
                'attempts': stats.rush_attempts,
                'yards': stats.rush_yards,
                'touchdowns': stats.rush_tds,
                'fumbles': stats.fumbles,
                'fumbles_lost': stats.fumbles_lost,
            },
            'receiving': {
                'receptions': stats.receptions,
                'targets': stats.targets,
                'yards': stats.rec_yards,
                'touchdowns': stats.rec_tds,
            }
        }

    def team_stats_to_dict(stats):
        return {
            'team': stats.team,
            'score': {
                'total': stats.total_points,
                'q1': stats.q1_points,
                'q2': stats.q2_points,
                'q3': stats.q3_points,
                'q4': stats.q4_points,
                'ot': stats.ot_points,
            },
            'offense': {
                'total_yards': stats.total_yards,
                'passing_yards': stats.passing_yards,
                'rushing_yards': stats.rushing_yards,
                'first_downs': stats.first_downs,
            },
            'conversions': {
                'third_down': f'{stats.third_down_conversions}/{stats.third_down_attempts}',
                'fourth_down': f'{stats.fourth_down_conversions}/{stats.fourth_down_attempts}',
            },
            'turnovers': {
                'total': stats.turnovers,
                'fumbles_lost': stats.fumbles_lost,
                'interceptions': stats.interceptions_thrown,
            },
            'penalties': {
                'count': stats.penalties,
                'yards': stats.penalty_yards,
            },
            'time_of_possession': stats.time_of_possession,
        }

    # Convert players dicts
    away_players_list = [player_stats_to_dict(p) for p in boxscore.away_players.values()]
    home_players_list = [player_stats_to_dict(p) for p in boxscore.home_players.values()]

    # Sort players by position relevance
    def player_sort_key(p):
        # Passers first, then rushers, then receivers
        if p['passing']['attempts'] > 0:
            return (0, -p['passing']['attempts'])
        elif p['rushing']['attempts'] > 0:
            return (1, -p['rushing']['attempts'])
        else:
            return (2, -p['receiving']['receptions'])

    away_players_list.sort(key=player_sort_key)
    home_players_list.sort(key=player_sort_key)

    # Convert scoring plays
    scoring_plays_list = [
        {
            'quarter': play.quarter,
            'time': play.time,
            'team': play.team,
            'description': play.description,
            'away_score': play.away_score,
            'home_score': play.home_score,
            'play_type': play.play_type,
        }
        for play in boxscore.scoring_plays
    ]

    return {
        'game_id': boxscore.game_id,
        'away_team': boxscore.away_team,
        'home_team': boxscore.home_team,
        'final_score': {
            'away': boxscore.away_score,
            'home': boxscore.home_score,
        },
        'away_stats': team_stats_to_dict(boxscore.away_stats),
        'home_stats': team_stats_to_dict(boxscore.home_stats),
        'away_players': away_players_list,
        'home_players': home_players_list,
        'scoring_plays': scoring_plays_list,
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
