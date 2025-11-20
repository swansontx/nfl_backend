from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import logging

from backend.api.external_apis import weather_api, sleeper_api
from backend.api.data_refresh import data_refresh_manager, startup_refresh

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup: Auto-refresh all data
    logger.info("Server starting - initiating data refresh...")
    try:
        results = await startup_refresh()
        logger.info(f"Startup data refresh complete: {results.get('database_status', {})}")
    except Exception as e:
        logger.error(f"Startup refresh failed: {e}")
        # Don't prevent server from starting

    yield

    # Shutdown
    logger.info("Server shutting down...")
from backend.api.insights_engine import insight_generator, PlayerStats, Insight as InsightData
from backend.api.narrative_generator import narrative_generator
from backend.api.prop_analyzer import prop_analyzer, PropLine, PropProjection, PropValue
from backend.api.odds_api import odds_api
from backend.api.model_loader import model_loader
from backend.api.team_database import get_team, get_all_teams, get_division_teams, get_conference_teams
from backend.api.schedule_loader import schedule_loader, Game
from backend.api.boxscore_generator import boxscore_generator
from backend.api.injury_impact_analyzer import injury_analyzer, get_injury_impact_for_game
from backend.api.situational_analyzer import situational_analyzer, analyze_game_situation, get_top_situations
from backend.config import settings, check_environment

app = FastAPI(
    title='NFL Props Backend API',
    version='1.0.0',
    description='Backend API for NFL props predictions, insights, and content',
    lifespan=lifespan
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


@app.get('/health/data')
async def health_data():
    """Check environment setup and data availability.

    Returns detailed status about:
    - Environment mode (development/production)
    - Missing required API keys
    - Available output directories (predictions, odds, reports)
    - Configuration thresholds
    - Warnings about setup issues
    """
    env_status = check_environment()

    # Add API-specific checks
    predictions_count = 0
    if env_status['data_available']['predictions']:
        predictions_dir = settings.get_predictions_dir()
        predictions_count = len(list(predictions_dir.glob('props_*.csv')))

    odds_count = 0
    if env_status['data_available']['odds']:
        odds_dir = settings.get_odds_dir()
        odds_count = len(list(odds_dir.glob('*.json')))

    return {
        'status': 'ok' if not env_status['warnings'] else 'degraded',
        'timestamp': datetime.now().isoformat(),
        'environment': env_status['environment'],
        'is_production': env_status['is_production'],
        'api_keys': {
            'odds_api_key': bool(settings.odds_api_key),
            'openweather_api_key': bool(settings.openweather_api_key),
        },
        'missing_required_keys': env_status['missing_keys'],
        'data_status': {
            'predictions_available': env_status['data_available']['predictions'],
            'predictions_count': predictions_count,
            'odds_available': env_status['data_available']['odds'],
            'odds_snapshots_count': odds_count,
            'reports_available': env_status['data_available']['reports'],
        },
        'configuration': env_status['thresholds'],
        'warnings': env_status['warnings'],
        'recommendations': _get_setup_recommendations(env_status)
    }


def _get_setup_recommendations(env_status: dict) -> list[str]:
    """Generate setup recommendations based on environment status."""
    recommendations = []

    if env_status['missing_keys']:
        recommendations.append(
            f"Set missing API keys: {', '.join(env_status['missing_keys'])} in .env file"
        )

    if not env_status['data_available']['predictions']:
        recommendations.append(
            "Run data pipeline to generate predictions: python -m backend.workflow.workflow_multi_prop_system"
        )

    if not env_status['data_available']['odds']:
        recommendations.append(
            "Fetch odds snapshots: python -m backend.ingestion.fetch_prop_lines"
        )

    if env_status['is_production'] and env_status['warnings']:
        recommendations.append(
            "⚠️  Production mode has warnings - review configuration before deploying"
        )

    return recommendations


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


# ============================================================================
# Data Refresh Endpoints (for MCP/Claude to trigger updates)
# ============================================================================

@app.post('/admin/refresh/all', tags=['Admin', 'Refresh'])
async def refresh_all_data(force: bool = False):
    """Refresh all data sources (injuries, schedules, stats, play-by-play).

    This is the main endpoint for MCP/Claude to trigger comprehensive data updates.

    Args:
        force: Force refresh even if recently updated

    Returns:
        Dict with refresh results for each data type and database status
    """
    results = await data_refresh_manager.refresh_all(force=force)
    return results


@app.post('/admin/refresh/injuries', tags=['Admin', 'Refresh'])
async def refresh_injuries(force: bool = False):
    """Refresh injury data from Sleeper API.

    Args:
        force: Force refresh even if recently updated

    Returns:
        Dict with refresh results
    """
    result = await data_refresh_manager.refresh_injuries(force=force)
    return result


@app.post('/admin/refresh/schedules', tags=['Admin', 'Refresh'])
async def refresh_schedules(force: bool = False):
    """Refresh schedule data from CSV/parquet files.

    Args:
        force: Force refresh

    Returns:
        Dict with refresh results
    """
    result = await data_refresh_manager.refresh_schedules(force=force)
    return result


@app.post('/admin/refresh/player-stats', tags=['Admin', 'Refresh'])
async def refresh_player_stats(force: bool = False):
    """Refresh player stats from CSV files.

    Args:
        force: Force refresh

    Returns:
        Dict with refresh results
    """
    result = await data_refresh_manager.refresh_player_stats(force=force)
    return result


@app.post('/admin/refresh/team-stats', tags=['Admin', 'Refresh'])
async def refresh_team_stats(force: bool = False):
    """Refresh team stats from CSV files.

    Args:
        force: Force refresh

    Returns:
        Dict with refresh results
    """
    result = await data_refresh_manager.refresh_team_stats(force=force)
    return result


@app.post('/admin/refresh/rosters', tags=['Admin', 'Refresh'])
async def refresh_rosters(force: bool = False):
    """Refresh roster/depth chart data.

    Args:
        force: Force refresh

    Returns:
        Dict with refresh results
    """
    result = await data_refresh_manager.refresh_rosters(force=force)
    return result


@app.post('/admin/refresh/play-by-play', tags=['Admin', 'Refresh'])
async def refresh_play_by_play(force: bool = False):
    """Refresh play-by-play data from nflverse parquet files.

    Args:
        force: Force refresh

    Returns:
        Dict with refresh results
    """
    result = await data_refresh_manager.refresh_play_by_play(force=force)
    return result


@app.get('/admin/refresh/status', tags=['Admin', 'Refresh'])
async def get_refresh_status():
    """Get current data refresh status.

    Returns:
        Dict with last refresh times, in-progress status, and database stats
    """
    return data_refresh_manager.get_refresh_status()


@app.get('/admin/database/status', tags=['Admin'])
async def get_database_status():
    """Get SQLite database status and record counts.

    Returns:
        Dict with counts for all tables and last update times
    """
    from backend.database.local_db import get_database_status
    return get_database_status()


# ============================================================================
# Situational Analysis Endpoints
# ============================================================================

@app.get('/game/{game_id}/situation', tags=['Analysis'])
async def get_game_situation(game_id: str, home_team: str, away_team: str,
                             season: int = 2024, week: int = 12):
    """Get complete situational analysis for a game.

    Compounds all data to identify betting edges:
    - Trending form (last 3 games vs season)
    - Weather impact
    - Rest/schedule advantages
    - Positional matchup grades

    Returns detailed analysis with specific prop targets.
    """
    analysis = analyze_game_situation(game_id, home_team, away_team, season, week)

    return {
        'game_id': game_id,
        'matchup': f"{away_team} @ {home_team}",
        'season': season,
        'week': week,
        'home_form': {
            'team': analysis.home_form.team,
            'momentum': analysis.home_form.momentum,
            'form_grade': analysis.home_form.form_grade,
            'narrative': analysis.home_form.form_narrative,
            'recent_points': analysis.home_form.recent_points_avg,
            'season_points': analysis.home_form.season_points_avg
        },
        'away_form': {
            'team': analysis.away_form.team,
            'momentum': analysis.away_form.momentum,
            'form_grade': analysis.away_form.form_grade,
            'narrative': analysis.away_form.form_narrative,
            'recent_points': analysis.away_form.recent_points_avg,
            'season_points': analysis.away_form.season_points_avg
        },
        'weather': {
            'temperature': analysis.weather.temperature,
            'wind': analysis.weather.wind_speed,
            'is_dome': analysis.weather.is_dome,
            'narrative': analysis.weather.weather_narrative,
            'props': analysis.weather.weather_props
        },
        'schedule': {
            'home': {
                'days_rest': analysis.home_schedule.days_rest,
                'rest_advantage': analysis.home_schedule.rest_advantage,
                'narrative': analysis.home_schedule.schedule_narrative
            },
            'away': {
                'days_rest': analysis.away_schedule.days_rest,
                'rest_advantage': analysis.away_schedule.rest_advantage,
                'narrative': analysis.away_schedule.schedule_narrative
            }
        },
        'positional_edges': [
            {
                'team': edge.team,
                'position': edge.position,
                'grade': edge.grade,
                'edge_score': edge.edge_score,
                'insight': edge.insight,
                'props': edge.target_props
            }
            for edge in analysis.positional_edges
        ],
        'key_situations': analysis.key_situations,
        'prop_targets': analysis.prop_targets
    }


@app.get('/week/{week}/situations', tags=['Analysis'])
async def get_week_betting_situations(week: int, season: int = 2024, min_edge: float = 15.0):
    """Get top betting situations across all games in a week.

    Identifies SMASH SPOTS and favorable matchups for:
    - Positional advantages (QB vs weak pass D, RB vs weak rush D)
    - Hot/cold team momentum
    - Weather impacts
    - Rest advantages

    Returns situations sorted by edge score.
    """
    situations = get_top_situations(season, week, min_edge)

    return {
        'season': season,
        'week': week,
        'min_edge': min_edge,
        'total_situations': len(situations),
        'situations': situations
    }


@app.get('/team/{team}/form', tags=['Analysis'])
async def get_team_trending_form(team: str, season: int = 2024, week: int = 12):
    """Get trending form analysis for a team.

    Compares last 3 games vs season average for:
    - Scoring
    - Passing yards
    - Rushing yards
    - Points allowed

    Returns momentum indicator (hot/cold/neutral) and form grade.
    """
    form = situational_analyzer.get_trending_form(team.upper(), season, week)

    return {
        'team': team.upper(),
        'season': season,
        'week': week,
        'momentum': form.momentum,
        'form_grade': form.form_grade,
        'narrative': form.form_narrative,
        'scoring': {
            'recent_avg': form.recent_points_avg,
            'season_avg': form.season_points_avg,
            'trend': form.scoring_trend,
            'trend_pct': form.scoring_trend_pct
        },
        'passing': {
            'recent_avg': form.recent_pass_yards_avg,
            'season_avg': form.season_pass_yards_avg,
            'trend': form.pass_trend
        },
        'rushing': {
            'recent_avg': form.recent_rush_yards_avg,
            'season_avg': form.season_rush_yards_avg,
            'trend': form.rush_trend
        },
        'defense': {
            'recent_allowed': form.recent_points_allowed_avg,
            'season_allowed': form.season_points_allowed_avg,
            'trend': form.defense_trend
        }
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


@app.get('/api/v1/games/{game_id}/injury-impact', tags=['Games', 'Injuries'])
async def get_game_injury_impact(game_id: str):
    """Get comprehensive injury impact analysis for betting.

    Analyzes injuries and their cascading effects on:
    - Team performance and game outcomes
    - Player prop redistribution (targets, carries, etc.)
    - Backup player opportunities
    - Betting recommendations

    Args:
        game_id: Game ID in format {season}_{week}_{away}_{home}

    Returns:
        Full injury impact analysis including:
        - Team impact scores (0-100)
        - Replacement players from depth chart
        - Prop redistribution predictions (who gets more targets/carries)
        - Specific betting recommendations
        - High-confidence prop plays
    """
    try:
        # Get injury data
        injury_data = sleeper_api.get_injuries_for_game(game_id)

        # Analyze impact for both teams
        impact_analysis = get_injury_impact_for_game(
            game_id=game_id,
            home_injuries=injury_data['home_injuries'],
            away_injuries=injury_data['away_injuries']
        )

        # Convert dataclass objects to dicts for JSON serialization
        home_report = impact_analysis['home_report']
        away_report = impact_analysis['away_report']

        def serialize_injury_impact(impact):
            return {
                'injured_player': impact.injured_player,
                'position': impact.position,
                'team': impact.team,
                'status': impact.status,
                'replacement': impact.replacement,
                'replacement_depth': impact.replacement_depth,
                'team_impact_score': impact.team_impact_score,
                'prop_implications': impact.prop_implications,
                'narrative': impact.narrative,
                'betting_recommendations': impact.betting_recommendations
            }

        def serialize_team_report(report):
            return {
                'team': report.team,
                'total_impact_score': report.total_impact_score,
                'key_injuries': [serialize_injury_impact(i) for i in report.key_injuries],
                'prop_redistributions': report.prop_redistributions,
                'summary': report.summary,
                'betting_angle': report.betting_angle
            }

        return {
            'game_id': game_id,
            'home_team': impact_analysis['home_team'],
            'away_team': impact_analysis['away_team'],
            'home_impact': serialize_team_report(home_report),
            'away_impact': serialize_team_report(away_report),
            'injury_edge': impact_analysis['injury_edge'],
            'lean': impact_analysis['lean'],
            'impact_differential': impact_analysis['impact_differential'],
            'top_prop_plays': impact_analysis['all_prop_plays'][:10],
            'last_updated': datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing injury impact: {str(e)}")


@app.get('/api/v1/teams/{team_id}/injury-impact', tags=['Teams', 'Injuries'])
async def get_team_injury_impact(team_id: str):
    """Get injury impact analysis for a specific team.

    Args:
        team_id: Team abbreviation (e.g., 'KC', 'BUF')

    Returns:
        Team injury impact including:
        - Total impact score
        - Key injuries with replacements
        - Prop redistribution analysis
        - Betting angle
    """
    try:
        team = get_team(team_id.upper())
        if not team:
            raise HTTPException(status_code=404, detail=f'Team not found: {team_id}')

        # Get team injuries from Sleeper
        all_injuries = sleeper_api.get_injuries()
        team_injuries = [
            inj for inj in all_injuries
            if inj.get('team', '').upper() == team_id.upper()
        ]

        # Analyze team injuries
        report = injury_analyzer.analyze_team_injuries(team_id.upper(), team_injuries)

        # Serialize response
        def serialize_injury_impact(impact):
            return {
                'injured_player': impact.injured_player,
                'position': impact.position,
                'status': impact.status,
                'replacement': impact.replacement,
                'replacement_depth': impact.replacement_depth,
                'team_impact_score': impact.team_impact_score,
                'prop_implications': impact.prop_implications,
                'narrative': impact.narrative,
                'betting_recommendations': impact.betting_recommendations
            }

        return {
            'team': team_id.upper(),
            'team_name': team.get('name'),
            'total_impact_score': report.total_impact_score,
            'key_injuries': [serialize_injury_impact(i) for i in report.key_injuries],
            'prop_redistributions': report.prop_redistributions,
            'summary': report.summary,
            'betting_angle': report.betting_angle,
            'injury_count': len(team_injuries),
            'last_updated': datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing team injuries: {str(e)}")


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
    game_ids: Optional[str] = None,
    player_id: Optional[str] = None,
    min_edge: float = 5.0,
    min_grade: str = "B",
    limit: int = 10
):
    """Find high-value prop bets.

    Args:
        game_id: Filter by single game ID (optional)
        game_ids: Filter by multiple game IDs, comma-separated (optional, for parlay building)
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

    # Build list of game IDs to filter
    target_game_ids = []
    if game_ids:
        target_game_ids = [gid.strip() for gid in game_ids.split(',')]
    elif game_id:
        target_game_ids = [game_id]

    # Load real model projections
    projections = []
    if target_game_ids:
        for gid in target_game_ids:
            projections.extend(model_loader.load_projections_for_game(gid))
    else:
        # Load all recent projections if no game filter specified
        available_games = model_loader.get_available_games()
        for gid in available_games[:10]:  # Limit to 10 most recent games
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


@app.get('/api/v1/props/trending')
async def get_trending_props(
    week: Optional[int] = None,
    limit: int = 20
):
    """Get trending prop lines with 3-week movement tracking (DraftKings only).

    Returns "hot movers" (2+ point week-over-week changes) and "sustained trends"
    (3-week consistent direction patterns) for DraftKings prop betting.

    Args:
        week: Week number (optional, defaults to current week)
        limit: Number of trending props to return per category

    Returns:
        Dict with:

        hot_movers (week-over-week big changes):
        - player, market, bookmaker (always 'draftkings')
        - opening_line, current_line, line_movement, line_movement_pct
        - over_odds, under_odds (juice/vig)
        - direction ('up', 'down', 'neutral')
        - opening_timestamp, current_timestamp, days_tracked
        - badge: "+2.5 pts (7 days)" or "+22% (7 days)"
        - icon (⬆️⬇️), color (green/red), strength (🔥⚡📊)

        sustained_trends (3-week consistent patterns):
        - player, market, bookmaker
        - week_1_line, week_2_line, week_3_line, current_line
        - total_movement, total_movement_pct
        - direction, consistency ("3/3 weeks up")
        - badge: "+5.5 pts (3 weeks)" or "+15% (3 weeks)"
        - icon, color, strength (🔥🔥 for strong sustained trends)

        Snapshot frequency:
        - Week-over-week: Daily snapshots (7 data points)
        - 3-week: Every 2-3 days (10-15 data points recommended)
    """
    from pathlib import Path
    from backend.ingestion.fetch_prop_lines import PropLineFetcher

    # Use dummy API key for analysis (doesn't make API calls)
    fetcher = PropLineFetcher('dummy_key')
    snapshots_dir = Path('outputs/prop_lines')

    if not snapshots_dir.exists():
        return {
            "error": "No prop line snapshots found",
            "message": "Run prop line fetcher first to capture snapshots",
            "command": f"python -m backend.ingestion.fetch_prop_lines --week {week}"
        }

    # Analyze trends
    trends = fetcher.get_trending_props(snapshots_dir, week)

    if 'error' in trends:
        return trends

    # Format for frontend (already has icons/colors/strength from get_trending_props)
    all_movers = trends.get('hot_movers', [])
    sustained_trends = trends.get('sustained_trends', [])[:limit]

    # Categorize hot movers by direction
    # 🔥 Hottest movers: Biggest absolute line changes (regardless of direction)
    hottest_movers = sorted(
        all_movers,
        key=lambda m: abs(m.get('line_movement', 0)),
        reverse=True
    )[:limit]

    # ⬆️ Lines moving up: Harder to hit Over, easier to hit Under
    lines_moving_up = [
        m for m in all_movers
        if m.get('direction') == 'up'
    ][:limit]

    # ⬇️ Lines moving down: Easier to hit Over, harder to hit Under
    lines_moving_down = [
        m for m in all_movers
        if m.get('direction') == 'down'
    ][:limit]

    return {
        "week": week,
        "snapshot_count": trends.get('snapshots_analyzed', 0),
        "current_timestamp": trends.get('current_timestamp'),

        # 🔥 Hottest movers (biggest absolute line changes)
        "hottest_movers": hottest_movers,

        # ⬆️ Lines moving up (getting harder to hit Over)
        "lines_moving_up": lines_moving_up,

        # ⬇️ Lines moving down (getting easier to hit Over)
        "lines_moving_down": lines_moving_down,

        # Sustained 3-week trends (pattern validation)
        "sustained_trends": sustained_trends,

        # Summary stats with averages and counts for each category
        "summary": {
            "hottest_movers": {
                "total": len(hottest_movers),
                "strong": len([m for m in hottest_movers if abs(m.get('line_movement', 0)) >= 5.0]),
                "avg_movement": round(
                    sum(abs(m.get('line_movement', 0)) for m in hottest_movers) / max(len(hottest_movers), 1),
                    2
                )
            },
            "lines_moving_up": {
                "total": len(lines_moving_up),
                "strong": len([m for m in lines_moving_up if m.get('line_movement', 0) >= 5.0]),
                "avg_movement": round(
                    sum(m.get('line_movement', 0) for m in lines_moving_up) / max(len(lines_moving_up), 1),
                    2
                )
            },
            "lines_moving_down": {
                "total": len(lines_moving_down),
                "strong": len([m for m in lines_moving_down if m.get('line_movement', 0) <= -5.0]),
                "avg_movement": round(
                    sum(m.get('line_movement', 0) for m in lines_moving_down) / max(len(lines_moving_down), 1),
                    2
                )
            },
            "sustained_trends": {
                "total": len(sustained_trends),
                "strong": len([t for t in sustained_trends if abs(t.get('total_movement', 0)) >= 8.0]),
                "trending_up": len([t for t in sustained_trends if t.get('direction') == 'up']),
                "trending_down": len([t for t in sustained_trends if t.get('direction') == 'down'])
            }
        }
    }


@app.get('/api/v1/odds/current')
async def get_current_odds(
    week: Optional[int] = None,
    market: Optional[str] = None
):
    """Get current DraftKings prop odds for all upcoming games.

    Essential for displaying current betting lines to users.

    Args:
        week: Week number (optional, defaults to current week)
        market: Filter by specific market (e.g., 'player_pass_yds', 'player_rush_yds')

    Returns:
        Dict with current DraftKings odds:
        - games: List of games with props
        - Each prop includes: player, market, line, over_odds, under_odds, timestamp
        - Grouped by game for easy navigation
    """
    from pathlib import Path
    import json

    snapshots_dir = Path('outputs/prop_lines')
    week_str = f"week_{week}" if week else "current"
    latest_file = snapshots_dir / f"snapshot_{week_str}_latest.json"

    if not latest_file.exists():
        return {
            "error": "No current odds available",
            "message": "Run prop line fetcher to get current odds",
            "command": f"python -m backend.ingestion.fetch_prop_lines --week {week}"
        }

    # Load latest snapshot
    with open(latest_file, 'r') as f:
        snapshot = json.load(f)

    # Format for frontend
    games_with_odds = []

    for game_id, game_data in snapshot.items():
        game_info = {
            "game_id": game_id,
            "home_team": game_data.get('home_team'),
            "away_team": game_data.get('away_team'),
            "commence_time": game_data.get('commence_time'),
            "snapshot_timestamp": game_data.get('snapshot_timestamp'),
            "props_by_market": {}
        }

        # Group props by market
        for market_name, props in game_data.get('props', {}).items():
            # Filter by market if specified
            if market and market_name != market:
                continue

            # Only include DraftKings lines
            dk_props = [
                {
                    "player": p.get('player_name'),
                    "market": market_name,
                    "line": p.get('line'),
                    "over_odds": p.get('over_odds', -110),
                    "under_odds": p.get('under_odds', -110),
                    "timestamp": p.get('timestamp'),
                    "bookmaker": "draftkings"
                }
                for p in props
                if p.get('bookmaker') == 'draftkings'
            ]

            if dk_props:
                game_info['props_by_market'][market_name] = dk_props

        if game_info['props_by_market']:  # Only include games with props
            games_with_odds.append(game_info)

    return {
        "week": week,
        "total_games": len(games_with_odds),
        "total_props": sum(
            len(props)
            for game in games_with_odds
            for props in game['props_by_market'].values()
        ),
        "snapshot_timestamp": snapshot.get(list(snapshot.keys())[0], {}).get('snapshot_timestamp') if snapshot else None,
        "games": games_with_odds
    }


@app.get('/api/v1/standings')
async def get_standings(
    season: int = 2024,
    week: Optional[int] = None
):
    """Get NFL standings by division and conference.

    Essential context for understanding team strength and playoff implications.

    Args:
        season: Season year (default 2024)
        week: Week number (optional, returns current standings if not specified)

    Returns:
        Dict with standings:
        - afc_east, afc_north, afc_south, afc_west
        - nfc_east, nfc_north, nfc_south, nfc_west
        - Each division includes: team, wins, losses, ties, win_pct, division_record, conference_record
    """
    from pathlib import Path
    import pandas as pd

    # Try to load standings from nflverse data
    standings_file = Path(f'inputs/{season}_standings.csv')

    if standings_file.exists():
        # Load from nflverse standings
        standings_df = pd.read_csv(standings_file)

        # Filter by week if specified
        if week:
            standings_df = standings_df[standings_df['week'] == week]

        # Group by division
        divisions = {}
        for division in ['AFC East', 'AFC North', 'AFC South', 'AFC West',
                        'NFC East', 'NFC North', 'NFC South', 'NFC West']:
            div_teams = standings_df[standings_df['division'] == division].to_dict('records')
            divisions[division.lower().replace(' ', '_')] = div_teams

        return {
            "season": season,
            "week": week,
            "standings": divisions
        }

    # Fallback: Calculate from schedule/games data
    schedule_file = Path(f'inputs/{season}_schedule.parquet')

    if schedule_file.exists():
        import pandas as pd

        schedule = pd.read_parquet(schedule_file)

        # Calculate records from completed games
        teams_records = {}

        for _, game in schedule.iterrows():
            if pd.notna(game.get('home_score')) and pd.notna(game.get('away_score')):
                # Game is complete
                home_team = game['home_team']
                away_team = game['away_team']
                home_score = game['home_score']
                away_score = game['away_score']

                # Initialize teams if not exist
                for team in [home_team, away_team]:
                    if team not in teams_records:
                        teams_records[team] = {'wins': 0, 'losses': 0, 'ties': 0}

                # Update records
                if home_score > away_score:
                    teams_records[home_team]['wins'] += 1
                    teams_records[away_team]['losses'] += 1
                elif away_score > home_score:
                    teams_records[away_team]['wins'] += 1
                    teams_records[home_team]['losses'] += 1
                else:
                    teams_records[home_team]['ties'] += 1
                    teams_records[away_team]['ties'] += 1

        # Format standings by division (hardcoded NFL divisions)
        divisions = {
            'afc_east': ['BUF', 'MIA', 'NE', 'NYJ'],
            'afc_north': ['BAL', 'CIN', 'CLE', 'PIT'],
            'afc_south': ['HOU', 'IND', 'JAX', 'TEN'],
            'afc_west': ['DEN', 'KC', 'LV', 'LAC'],
            'nfc_east': ['DAL', 'NYG', 'PHI', 'WAS'],
            'nfc_north': ['CHI', 'DET', 'GB', 'MIN'],
            'nfc_south': ['ATL', 'CAR', 'NO', 'TB'],
            'nfc_west': ['ARI', 'LAR', 'SF', 'SEA']
        }

        standings = {}
        for division, teams in divisions.items():
            standings[division] = [
                {
                    'team': team,
                    'wins': teams_records.get(team, {}).get('wins', 0),
                    'losses': teams_records.get(team, {}).get('losses', 0),
                    'ties': teams_records.get(team, {}).get('ties', 0),
                    'win_pct': round(
                        teams_records.get(team, {}).get('wins', 0) /
                        max(sum(teams_records.get(team, {}).values()), 1),
                        3
                    )
                }
                for team in teams
            ]

            # Sort by win percentage
            standings[division].sort(key=lambda x: x['win_pct'], reverse=True)

        return {
            "season": season,
            "week": week,
            "standings": standings
        }

    # No data available
    return {
        "error": "No standings data available",
        "message": "Ingest nflverse data to get standings",
        "command": f"python -m backend.ingestion.fetch_nflverse --year {season}"
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


@app.get('/api/v1/teams/{team_id}/stats', tags=['Teams'])
def get_team_stats(team_id: str, season: int = 2024):
    """Get team statistics for a season.

    Args:
        team_id: Team abbreviation
        season: Season year (default: 2024)

    Returns:
        Team offensive/defensive stats and rankings
    """
    team = get_team(team_id.upper())
    if not team:
        raise HTTPException(status_code=404, detail=f'Team not found: {team_id}')

    # TODO: Load actual team stats from database or computed files
    # For now, return structure with mock data
    return {
        'team_id': team_id.upper(),
        'season': season,
        'offensive_stats': {
            'points_per_game': 0.0,
            'yards_per_game': 0.0,
            'pass_yards_per_game': 0.0,
            'rush_yards_per_game': 0.0,
            'turnovers_per_game': 0.0,
        },
        'defensive_stats': {
            'points_allowed_per_game': 0.0,
            'yards_allowed_per_game': 0.0,
            'sacks': 0,
            'turnovers_forced': 0,
            'interceptions': 0,
        },
        'rankings': {
            'offensive_rank': None,
            'defensive_rank': None,
            'scoring_rank': None,
            'pass_offense_rank': None,
            'rush_offense_rank': None,
            'pass_defense_rank': None,
            'rush_defense_rank': None,
        },
        'record': {
            'wins': 0,
            'losses': 0,
            'ties': 0,
        }
    }


@app.get('/api/v1/games', tags=['Games'])
def list_games(
    week: Optional[int] = None,
    season: int = 2024,
    team: Optional[str] = None,
    limit: int = 100
):
    """List NFL games (filterable by week, season, team).

    Args:
        week: Filter by week number
        season: Season year (default: 2024)
        team: Filter by team abbreviation
        limit: Max number of games to return

    Returns:
        List of games matching filters
    """
    # Load schedule for the season
    games = schedule_loader.load_schedule(season)

    # Apply filters
    filtered_games = games

    if week is not None:
        filtered_games = [g for g in filtered_games if g.week == week]

    if team:
        team_upper = team.upper()
        filtered_games = [
            g for g in filtered_games
            if g.home_team == team_upper or g.away_team == team_upper
        ]

    # Limit results
    filtered_games = filtered_games[:limit]

    # Convert to dict
    games_list = [
        {
            'game_id': g.game_id,
            'week': g.week,
            'season': g.season,
            'game_type': g.game_type,
            'game_date': g.gameday,
            'game_time': g.gametime,
            'home_team': g.home_team,
            'away_team': g.away_team,
            'home_score': g.home_score,
            'away_score': g.away_score,
            'completed': g.away_score is not None and g.home_score is not None,
            'stadium': get_team(g.home_team).get('stadium', {}).get('name') if get_team(g.home_team) else None,
        }
        for g in filtered_games
    ]

    return {
        'season': season,
        'week': week,
        'team': team,
        'count': len(games_list),
        'games': games_list
    }


@app.get('/api/v1/games/{game_id}', tags=['Games'])
def get_game_details(game_id: str):
    """Get detailed information for a specific game.

    Args:
        game_id: Game ID (format: YYYY_WW_AWAY_HOME)

    Returns:
        Comprehensive game details
    """
    # Parse game_id to extract season
    try:
        parts = game_id.split('_')
        if len(parts) < 4:
            raise HTTPException(status_code=400, detail='Invalid game_id format')

        season = int(parts[0])
        week = int(parts[1])
        away_team = parts[2]
        home_team = parts[3]
    except:
        raise HTTPException(status_code=400, detail='Invalid game_id format')

    # Load schedule and find the game
    games = schedule_loader.load_schedule(season)
    game = next((g for g in games if g.game_id == game_id), None)

    if not game:
        raise HTTPException(status_code=404, detail=f'Game not found: {game_id}')

    # Get team info
    home_team_info = get_team(game.home_team)
    away_team_info = get_team(game.away_team)

    return {
        'game_id': game.game_id,
        'week': game.week,
        'season': game.season,
        'game_type': game.game_type,
        'home_team': game.home_team,
        'away_team': game.away_team,
        'home_team_name': home_team_info.get('name') if home_team_info else game.home_team,
        'away_team_name': away_team_info.get('name') if away_team_info else game.away_team,
        'game_date': game.gameday,
        'game_time': game.gametime,
        'stadium': home_team_info.get('stadium', {}).get('name') if home_team_info else None,
        'roof': home_team_info.get('stadium', {}).get('roof') if home_team_info else None,
        'surface': home_team_info.get('stadium', {}).get('surface') if home_team_info else None,
        'home_score': game.home_score,
        'away_score': game.away_score,
        'completed': game.away_score is not None and game.home_score is not None,
        # TODO: Add betting lines when available
        'spread': None,
        'total': None,
        'moneyline_home': None,
        'moneyline_away': None,
    }


@app.get('/api/v1/players', tags=['Players'])
def search_players(
    team: Optional[str] = None,
    position: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50
):
    """Search and filter NFL players.

    Args:
        team: Filter by team abbreviation
        position: Filter by position (QB, RB, WR, TE, etc.)
        search: Search by player name
        limit: Max number of players to return

    Returns:
        List of players matching filters
    """
    # TODO: Load from player lookup JSON or database
    # For now, return empty structure
    return {
        'filters': {
            'team': team,
            'position': position,
            'search': search,
        },
        'count': 0,
        'players': []
    }


@app.get('/api/v1/players/{player_id}', tags=['Players'])
def get_player_details(player_id: str):
    """Get detailed information for a specific player.

    Args:
        player_id: Player ID (nflverse format)

    Returns:
        Player details and metadata
    """
    # TODO: Load from player lookup JSON or database
    return {
        'player_id': player_id,
        'player_name': 'Unknown Player',
        'position': None,
        'team': None,
        'number': None,
        'height': None,
        'weight': None,
        'college': None,
        'draft_year': None,
    }


@app.get('/api/v1/players/{player_id}/stats', tags=['Players'])
def get_player_stats(player_id: str, season: int = 2024):
    """Get player statistics for a season.

    Args:
        player_id: Player ID (nflverse format)
        season: Season year (default: 2024)

    Returns:
        Season statistics by position
    """
    # TODO: Load from player_stats CSV or database
    return {
        'player_id': player_id,
        'season': season,
        'games_played': 0,
        'passing': {
            'attempts': 0,
            'completions': 0,
            'yards': 0,
            'touchdowns': 0,
            'interceptions': 0,
        },
        'rushing': {
            'attempts': 0,
            'yards': 0,
            'touchdowns': 0,
        },
        'receiving': {
            'targets': 0,
            'receptions': 0,
            'yards': 0,
            'touchdowns': 0,
        }
    }


# ============================================================================
# Betting Infrastructure Endpoints (NEW!)
# ============================================================================

@app.get('/api/v1/betting/parlays/suggestions', tags=['Betting'])
async def get_parlay_suggestions(
    game_ids: Optional[str] = None,
    max_legs: int = 4,
    min_parlay_ev: float = 0.10,
    limit: int = 10
):
    """Get correlation-aware parlay suggestions.

    Uses the portfolio optimizer to build optimal parlays that account for:
    - Correlation between props (same-game, same-team, opposing teams)
    - Game script scenarios (shootout, defensive, RB game)
    - Risk-adjusted sizing with correlation penalties

    Args:
        game_ids: Comma-separated game IDs to analyze (optional, uses all if not specified)
        max_legs: Maximum legs per parlay (default 4)
        min_parlay_ev: Minimum EV to suggest parlay (default 10%)
        limit: Number of parlay suggestions to return

    Returns:
        List of parlay suggestions with:
        - Legs (individual props)
        - Combined odds and probabilities (raw vs correlation-adjusted)
        - EV and recommended stake
        - Correlation analysis and scenarios
    """
    from backend.betting.portfolio_optimizer import build_parlay_suggestions, PropValue

    # Load value props (from /api/v1/props/value endpoint logic)
    prop_lines = odds_api.get_player_props()
    projections = []

    if game_ids:
        for game_id in game_ids.split(','):
            projections.extend(model_loader.load_projections_for_game(game_id.strip()))
    else:
        # Load all available
        available_games = model_loader.get_available_games()
        for gid in available_games[:10]:  # Limit to 10 games
            projections.extend(model_loader.load_projections_for_game(gid))

    # Find value props
    value_props_raw = prop_analyzer.find_best_props(
        prop_lines,
        projections,
        min_edge=settings.min_edge_threshold
    )

    # Convert to PropValue format for optimizer
    value_props = []
    for vp in value_props_raw:
        # Extract game context (would come from projections in real implementation)
        value_props.append(PropValue(
            player_id=vp.prop_line.player_id,
            player_name=vp.prop_line.player_name,
            game_id="2024_12_KC_BUF",  # TODO: Extract from data
            team="KC",  # TODO: Extract from data
            opponent="BUF",
            prop_type=vp.prop_line.prop_type,
            line=vp.prop_line.line,
            side="over" if vp.recommendation == "OVER" else "under",
            odds=vp.prop_line.over_odds if vp.recommendation == "OVER" else vp.prop_line.under_odds,
            projection=vp.projection.projection,
            hit_probability=vp.projection.hit_probability_over if vp.recommendation == "OVER" else vp.projection.hit_probability_under,
            edge=max(vp.edge_over, vp.edge_under),
            ev=max(vp.edge_over, vp.edge_under) / 100,  # Convert to decimal
            is_home=True,  # TODO: Extract from data
            spread=-3.0,  # TODO: Extract from market data
            total=49.5,  # TODO: Extract from market data
        ))

    # Build parlay suggestions
    parlay_suggestions = build_parlay_suggestions(
        value_props=value_props,
        max_legs=max_legs,
        max_risk_per_game=settings.max_stake_pct,
        kelly_fraction=settings.kelly_fraction,
        min_parlay_ev=min_parlay_ev
    )

    # Convert to response format
    results = []
    for parlay in parlay_suggestions[:limit]:
        results.append({
            "legs": [
                {
                    "player_name": leg.player_name,
                    "prop_type": leg.prop_type,
                    "side": leg.side,
                    "line": leg.line,
                    "odds": leg.odds,
                    "projection": leg.projection,
                    "hit_probability": leg.hit_probability
                }
                for leg in parlay.legs
            ],
            "combined_odds": parlay.combined_odds,
            "probabilities": {
                "raw": round(parlay.raw_probability, 3),
                "adjusted": round(parlay.combined_probability, 3),
                "correlation_penalty": round(parlay.correlation_adjustment, 2)
            },
            "ev": round(parlay.ev, 2),
            "recommended_stake_pct": round(parlay.recommended_stake_pct, 2),
            "confidence": parlay.confidence,
            "scenarios": parlay.scenarios
        })

    return {
        "total_suggestions": len(results),
        "parlays": results,
        "filters": {
            "max_legs": max_legs,
            "min_parlay_ev": min_parlay_ev
        },
        "warning": "Parlay betting is high-risk. These suggestions account for correlation but variance is still significant."
    }


@app.get('/api/v1/betting/clv/report', tags=['Betting'])
async def get_clv_report(last_n: int = 50):
    """Get Closing Line Value (CLV) report for recent bets.

    CLV is the gold standard for evaluating betting model quality.
    Consistent positive CLV indicates a profitable long-term process
    regardless of short-term win/loss record.

    Args:
        last_n: Number of recent bets to analyze (default 50)

    Returns:
        CLV summary with:
        - Average CLV
        - Positive CLV rate
        - Win rate by CLV buckets
        - Top/worst CLV bets
        - Breakdown by prop type
    """
    from backend.betting.recommendation_manager import recommendation_manager

    # Get CLV summary
    summary = recommendation_manager.get_recent_clv_summary(last_n=last_n)

    if 'error' in summary:
        return {
            "error": summary['error'],
            "message": "No CLV data available. Log bets using /api/v1/betting/recommendations endpoint first."
        }

    # Get detailed CLV report from tracker
    clv_tracker = recommendation_manager.clv_tracker
    report = clv_tracker.generate_clv_report()

    return {
        "summary": summary,
        "overall_metrics": {
            "total_bets": report.get('total_bets', 0),
            "avg_clv": report.get('avg_clv', 0),
            "median_clv": report.get('median_clv', 0),
            "positive_clv_rate": report.get('positive_clv_rate', 0),
            "win_rate": report.get('win_rate', 0)
        },
        "by_prop_type": report.get('by_prop_type', {}),
        "clv_buckets": report.get('clv_buckets', {}),
        "top_clv_bets": report.get('top_clv_bets', [])[:10],
        "worst_clv_bets": report.get('worst_clv_bets', [])[:10],
        "interpretation": {
            "avg_clv": _interpret_clv(summary.get('avg_clv', 0)),
            "positive_clv_rate": _interpret_positive_rate(summary.get('positive_clv_rate', 0))
        }
    }


def _interpret_clv(avg_clv: float) -> str:
    """Interpret average CLV."""
    if avg_clv >= 2.0:
        return "EXCELLENT - Elite market-beating performance"
    elif avg_clv >= 1.0:
        return "GOOD - Sustainable edge, profitable long-term"
    elif avg_clv >= 0.5:
        return "FAIR - Slight edge, marginally profitable"
    elif avg_clv >= 0:
        return "NEUTRAL - Breaking even with closing lines"
    else:
        return "POOR - Consistently betting on wrong side of information"


def _interpret_positive_rate(rate: float) -> str:
    """Interpret positive CLV rate."""
    if rate >= 0.60:
        return "EXCELLENT - Beating closing line 60%+ of the time"
    elif rate >= 0.55:
        return "GOOD - Consistently finding value"
    elif rate >= 0.50:
        return "FAIR - Slightly better than market"
    else:
        return "POOR - Losing to closing line more often than winning"


@app.post('/api/v1/betting/recommendations', tags=['Betting'])
async def log_recommendation(recommendation: Dict):
    """Log a prop recommendation for CLV tracking.

    This endpoint should be called whenever a recommendation is generated,
    whether or not the bet is actually placed. This builds the CLV tracking
    history and meta trust model training data.

    Request body:
        {
            "player_name": "Patrick Mahomes",
            "game_id": "2024_12_KC_BUF",
            "prop_type": "player_pass_yds",
            "side": "over",
            "line": 275.5,
            "odds": -110,
            "projection": 295.3,
            "hit_probability": 0.68,
            "edge": 0.078,
            "ev": 0.05,
            "actually_bet": true,
            "trust_score": 0.72,
            "games_sampled": 12
        }

    Returns:
        Bet ID for tracking
    """
    from backend.betting.recommendation_manager import (
        recommendation_manager,
        PropRecommendation
    )

    # Create recommendation object
    rec = PropRecommendation(
        player_id=recommendation.get('player_id', 'unknown'),
        player_name=recommendation['player_name'],
        game_id=recommendation['game_id'],
        prop_type=recommendation['prop_type'],
        side=recommendation['side'],
        line=recommendation['line'],
        odds=recommendation['odds'],
        projection=recommendation['projection'],
        hit_probability=recommendation['hit_probability'],
        edge=recommendation['edge'],
        ev=recommendation['ev'],
        trust_score=recommendation.get('trust_score'),
        games_sampled=recommendation.get('games_sampled'),
        recommendation="BET" if recommendation.get('actually_bet') else "CONSIDER",
        stake_pct=recommendation.get('stake_pct')
    )

    # Log recommendation
    bet_id = recommendation_manager.log_prop_recommendation(
        recommendation=rec,
        actually_bet=recommendation.get('actually_bet', False)
    )

    return {
        "status": "logged",
        "bet_id": bet_id,
        "message": "Recommendation logged for CLV tracking" if recommendation.get('actually_bet') else "Recommendation logged (not bet)"
    }


@app.get('/api/v1/players/{player_id}/gamelogs', tags=['Players'])
def get_player_gamelogs(player_id: str, season: int = 2024, limit: int = 20):
    """Get game-by-game performance logs for a player.

    Args:
        player_id: Player ID (nflverse format)
        season: Season year (default: 2024)
        limit: Max number of games to return

    Returns:
        List of game performances
    """
    # TODO: Load from player_stats CSV (filtered by week)
    return {
        'player_id': player_id,
        'season': season,
        'count': 0,
        'games': []
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
