from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime

from backend.api.external_apis import weather_api, sleeper_api

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


@app.get('/game/{game_id}/projections')
async def get_projections(game_id: str):
    """Get player prop projections for a game."""
    # TODO: Load from outputs/predictions/props_{game_id}.csv
    return {'game_id': game_id, 'projections': []}


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
    """
    # TODO: Get stadium location and game time from schedule
    # For now, using placeholder coordinates
    # In production, you'd have a stadium database:
    # stadium_data = get_stadium_for_game(game_id)
    # weather = weather_api.get_game_weather(
    #     stadium_data['lat'],
    #     stadium_data['lon'],
    #     stadium_data['game_time']
    # )

    weather_data = weather_api.get_game_weather(
        stadium_lat=42.0909,  # Example: Buffalo
        stadium_lon=-78.7869,
        game_time="2025-11-24T18:00:00Z"
    )

    return WeatherData(**weather_data)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
