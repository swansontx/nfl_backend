# P4 External Service Requirements

**Last Updated:** 2025-11-24

This document details the remaining P4 API endpoint items that require external services, APIs, or additional infrastructure that are not currently implemented.

---

## Overview

**Status:** 8/17 P4 items require external services
**Priority:** LOW - These are enhancement features, core functionality works with real NFLverse data

---

## 1. CORS Origin Restriction (Line 59)

**File:** `backend/api/app.py`

**Current State:**
```python
allow_origins=["*"],  # TODO: Restrict in production
```

**Requirement:**
- Production deployment should restrict CORS origins to specific frontend domains
- Currently allows all origins for development convenience

**Implementation:**
```python
# Production example
allow_origins=[
    "https://your-frontend-domain.com",
    "https://app.your-domain.com"
]

# Or use environment variable
allow_origins=os.getenv("CORS_ORIGINS", "*").split(",")
```

**Priority:** HIGH for production deployment
**Blocker:** None - can be configured in environment variables

---

## 2. Orchestration Pipeline Integration (Line 218)

**File:** `backend/api/app.py`
**Endpoint:** `POST /admin/recompute`

**Current State:**
```python
@app.post('/admin/recompute')
async def recompute(req: RecomputeRequest):
    # TODO: Integrate with orchestration pipeline
    return {'status': 'started', 'game_id': req.game_id}
```

**Requirement:**
- Wire into `backend/orchestration/orchestrator.py`
- Trigger model regeneration for specific games on-demand
- Handle async job tracking and status updates

**Implementation Needed:**
```python
from backend.orchestration.orchestrator import Orchestrator

async def recompute(req: RecomputeRequest):
    orchestrator = Orchestrator()

    # Parse game_id to get season/week
    game_info = parse_game_id(req.game_id)

    # Run orchestration for specific game
    await orchestrator.run_for_game(
        season=game_info['season'],
        week=game_info['week'],
        game_id=req.game_id
    )

    return {'status': 'started', 'game_id': req.game_id}
```

**Dependencies:**
- Orchestrator needs to support single-game regeneration (currently does full weeks)
- Need job queue/status tracking (celery, redis, or simple async tasks)

**Priority:** MEDIUM
**Blocker:** Orchestrator refactoring needed

---

## 3. News API Integration (Lines 769, 796)

**File:** `backend/api/app.py`
**Endpoint:** `GET /api/v1/news`

**Current State:**
- Only returns injury news from Sleeper API
- Has placeholder for general NFL news

**Requirement:**
Integrate with news sources:
- ESPN API
- NFL.com RSS feeds
- Twitter API for beat reporters
- FantasyPros news feed
- The Athletic RSS

**Implementation Example:**
```python
import feedparser

def fetch_nfl_news():
    feeds = [
        'https://www.nfl.com/feed/news',
        'https://www.espn.com/espn/rss/nfl/news',
    ]

    news_items = []
    for feed_url in feeds:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:5]:
            news_items.append(NewsItem(
                id=entry.id,
                title=entry.title,
                summary=entry.summary,
                source=feed.feed.title,
                published_at=entry.published,
                category="news",
                url=entry.link
            ))

    return news_items
```

**Dependencies:**
- `feedparser` library for RSS
- Twitter API key for social content (optional)
- ESPN API access (may require credentials)

**Priority:** LOW - Not critical for props functionality
**Blocker:** API keys/access for some sources

---

## 4. ML Model Integration for Insights (Line 999)

**File:** `backend/api/app.py`
**Endpoint:** `GET /api/v1/games/{game_id}/insights`

**Current State:**
- Returns placeholder insights
- No connection to actual ML models or feature analysis

**Requirement:**
Pull insights from:
- `backend/features/` - Player trend analysis
- `backend/modeling/` - Model predictions and confidence
- Historical matchup data
- Weather impact analysis
- Injury impact analysis

**Implementation Needed:**
```python
from backend.features.player_trends import calculate_player_trends
from backend.api.evaluation_pipeline import EvaluationPipeline

async def get_game_insights(game_id: str):
    pipeline = EvaluationPipeline()

    # Load game data
    game_info = parse_game_id(game_id)

    # Generate insights from models
    insights = []

    # QB trends
    qb_trends = calculate_player_trends(
        position='QB',
        team=game_info['away_team'],
        opponent=game_info['home_team']
    )

    insights.append(MatchupInsight(
        insight_type="trend",
        title=f"QB Performance Trend",
        description=qb_trends['summary'],
        confidence=qb_trends['confidence'],
        supporting_data=qb_trends['data']
    ))

    # Weather impact
    weather = weather_api.get_weather_for_game(game_id)
    if weather['wind_speed'] > 15:
        insights.append(MatchupInsight(
            insight_type="weather",
            title="High Wind Alert",
            description=f"Wind speeds of {weather['wind_speed']} mph expected",
            confidence=0.95,
            supporting_data=weather
        ))

    return insights
```

**Dependencies:**
- Feature extraction pipeline in `backend/features/`
- Model loading from `outputs/models/`
- Historical matchup database

**Priority:** HIGH - Core value proposition
**Blocker:** Need to wire up feature extraction and model loading

---

## 5. Stadium Location Resolution (Line 1034)

**File:** `backend/api/app.py`
**Endpoint:** `GET /api/v1/games/{game_id}/insights`

**Current State:**
```python
# TODO: Get stadium location from game_id/schedule
# weather = weather_api.get_game_weather(lat, lon, game_time)
```

**Requirement:**
- Extract stadium from game_id (home team)
- Look up coordinates from `backend/api/stadium_database.py`
- Get game time from schedule CSVs

**Implementation:**
```python
from backend.api.stadium_database import get_stadium_for_game
from backend.canonical.schedule_loader import ScheduleLoader

def get_weather_for_game_insight(game_id: str):
    # Get stadium (already implemented in external_apis.py)
    weather = weather_api.get_weather_for_game(game_id)

    # Get game time from schedule
    schedule = ScheduleLoader()
    game_info = schedule.get_game(game_id)
    game_time = game_info.get('gameday')

    return weather
```

**Dependencies:**
- Stadium database (✅ already implemented)
- Schedule loader (✅ already implemented)
- Just need to wire them together

**Priority:** LOW - Can use existing weather endpoint
**Blocker:** None

---

## 6. LLM Integration for Narratives (Line 1056)

**File:** `backend/api/app.py`
**Endpoint:** `GET /api/v1/games/{game_id}/narratives`

**Current State:**
- Returns placeholder narrative text
- No LLM integration

**Requirement:**
Generate AI narratives using LLM based on:
- Team stats and trends
- Player props and projections
- Injury reports
- Weather conditions
- Historical matchups

**Implementation Example:**
```python
import openai  # or anthropic

async def get_game_narratives(game_id: str):
    # Gather context
    game_info = parse_game_id(game_id)
    weather = weather_api.get_weather_for_game(game_id)
    injuries = sleeper_api.get_injuries_for_game(game_id)

    # Build prompt
    prompt = f"""
    Generate a game preview narrative for:
    {game_info['away_team']} @ {game_info['home_team']}

    Weather: {weather['condition']}, {weather['temperature']}°F
    Key Injuries: {injuries}

    Include:
    1. Game preview (2-3 sentences)
    2. Key matchups to watch
    3. Betting angles and value opportunities
    """

    # Call LLM
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    narrative_text = response.choices[0].message.content

    return [GameNarrative(
        narrative_type="preview",
        content=narrative_text,
        generated_at=datetime.now().isoformat()
    )]
```

**Dependencies:**
- OpenAI API key (`OPENAI_API_KEY`) or Anthropic API key
- Cost: ~$0.01-0.05 per game narrative

**Priority:** LOW - Enhancement feature
**Blocker:** API key and cost approval

---

## 7. Content Aggregation APIs (Line 1120)

**File:** `backend/api/app.py`
**Endpoint:** `GET /api/v1/games/{game_id}/content`

**Current State:**
- Returns placeholder content items
- No integration with content sources

**Requirement:**
Aggregate content from:
- YouTube API for game previews/highlights
- RSS feeds from major sports sites
- Podcast APIs (Apple Podcasts, Spotify)
- Twitter for embedded video content

**Implementation Example:**
```python
from googleapiclient.discovery import build

async def get_game_content(game_id: str, content_type: str):
    content_items = []
    game_info = parse_game_id(game_id)

    # YouTube videos
    if not content_type or content_type == "video":
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        search_query = f"{game_info['away_team']} vs {game_info['home_team']} preview"

        request = youtube.search().list(
            q=search_query,
            type='video',
            part='id,snippet',
            maxResults=5
        )
        response = request.execute()

        for item in response['items']:
            content_items.append(ContentItem(
                content_type="video",
                title=item['snippet']['title'],
                url=f"https://youtube.com/watch?v={item['id']['videoId']}",
                source="YouTube",
                thumbnail_url=item['snippet']['thumbnails']['default']['url'],
                published_at=item['snippet']['publishedAt']
            ))

    # RSS feed articles
    if not content_type or content_type == "article":
        # Similar implementation with feedparser
        pass

    return content_items
```

**Dependencies:**
- YouTube API key
- Spotify API for podcasts
- Apple Podcasts API
- RSS feed URLs

**Priority:** LOW - Not critical for props
**Blocker:** Multiple API keys needed

---

## 8. Player Projections Model Loading (Line 1379)

**File:** `backend/api/app.py`
**Endpoint:** `GET /api/v1/tools/player-comparison`

**Current State:**
- Returns placeholder projection data
- No connection to trained models

**Requirement:**
Load actual player projections from:
- `outputs/predictions/props_{game_id}.csv`
- Or generate on-demand using `backend/modeling/generate_projections.py`

**Implementation:**
```python
from pathlib import Path
import pandas as pd

def get_player_projections(player_id: str, prop_type: str):
    # Try to load from cached predictions
    projections_dir = Path('outputs/predictions/')

    # Find relevant game file
    for proj_file in projections_dir.glob('props_*.csv'):
        df = pd.read_csv(proj_file)
        player_proj = df[df['player_id'] == player_id]

        if not player_proj.empty:
            return {
                'player_id': player_id,
                'projection': player_proj[prop_type].iloc[0],
                'std_dev': player_proj[f'{prop_type}_std'].iloc[0],
                'confidence': player_proj['confidence'].iloc[0]
            }

    # If not found, generate on-demand
    from backend.modeling.generate_projections import ProjectionGenerator
    generator = ProjectionGenerator()
    projection = generator.generate_for_player(player_id, prop_type)

    return projection
```

**Dependencies:**
- Model files in `outputs/models/`
- Projection files in `outputs/predictions/`
- Feature data in `inputs/`

**Priority:** HIGH - Core functionality
**Blocker:** Need to wire up model loading (models exist, just need integration)

---

## 9. Game Prop Sheet Generation (Lines 1770-1772)

**File:** `backend/api/app.py`
**Endpoint:** `POST /api/v1/analysis/game`

**Current State:**
```python
# TODO: Aggregate all props for the game
# TODO: Generate projections for all players
# TODO: Calculate value for each prop
```

**Requirement:**
- Load all available props for a game from odds API
- Generate model projections for each prop
- Calculate EV and value grade
- Rank by edge/confidence

**Implementation:**
```python
def generate_game_prop_sheet(game_id: str):
    # Load odds/props
    props = odds_api.get_props_for_game(game_id)

    # Generate projections
    from backend.modeling.generate_projections import ProjectionGenerator
    generator = ProjectionGenerator()

    prop_sheet = []
    for prop in props:
        # Get model projection
        projection = generator.generate_for_player(
            prop['player_id'],
            prop['prop_type']
        )

        # Calculate value
        line = prop['line']
        if projection['value'] > line:
            edge = projection['value'] - line
            grade = calculate_grade(edge, projection['std_dev'])

            prop_sheet.append({
                'player': prop['player_name'],
                'prop': f"{prop['prop_type']} {'OVER' if edge > 0 else 'UNDER'} {line}",
                'projection': projection['value'],
                'edge': abs(edge),
                'grade': grade,
                'ev': calculate_ev(projection, line, prop['odds'])
            })

    # Sort by edge
    prop_sheet.sort(key=lambda x: x['edge'], reverse=True)

    return {
        'game_id': game_id,
        'total_props': len(props),
        'high_value_props': len([p for p in prop_sheet if p['grade'] in ['A', 'A+']]),
        'top_plays': prop_sheet[:20]
    }
```

**Dependencies:**
- Odds API integration (✅ already implemented)
- Model loading and projection generation
- Value calculation logic

**Priority:** HIGH - Core product feature
**Blocker:** Model loading integration needed

---

## 10. Betting Lines in Standings (Line 2256)

**File:** `backend/api/app.py`
**Endpoint:** `GET /api/v1/standings`

**Current State:**
```python
# TODO: Add betting lines when available
```

**Requirement:**
- Add current week's betting lines to standings data
- Show spread, moneyline, over/under for each team's next game

**Implementation:**
```python
def get_standings_with_lines(season: int, division: str = None):
    standings = get_standings(season, division)

    # Get current week's lines
    current_week = get_current_nfl_week()
    odds = odds_api.fetch_odds_api()

    # Enrich standings with betting info
    for team in standings:
        team_odds = [o for o in odds if team['team'] in [o['home_team'], o['away_team']]]
        if team_odds:
            team['next_game'] = {
                'opponent': team_odds[0]['home_team' if team['team'] == team_odds[0]['away_team'] else 'away_team'],
                'spread': team_odds[0]['spread'],
                'total': team_odds[0]['total'],
                'moneyline': team_odds[0]['moneyline']
            }

    return standings
```

**Dependencies:**
- Odds API (✅ already implemented)
- Just need to wire it in

**Priority:** LOW - Nice to have
**Blocker:** None

---

## Summary Table

| Item | Priority | Blocker | Estimated Effort |
|------|----------|---------|------------------|
| 1. CORS Restriction | HIGH (prod) | None | 5 min |
| 2. Orchestration | MEDIUM | Orchestrator refactor | 2-4 hours |
| 3. News API | LOW | API keys | 2-3 hours |
| 4. ML Insights | HIGH | Feature pipeline wiring | 4-6 hours |
| 5. Stadium Location | LOW | None (already exists) | 30 min |
| 6. LLM Narratives | LOW | API key + cost | 1-2 hours |
| 7. Content APIs | LOW | Multiple API keys | 3-4 hours |
| 8. Model Loading | HIGH | Model integration | 3-4 hours |
| 9. Prop Sheet | HIGH | Model integration | 4-6 hours |
| 10. Betting Lines | LOW | None | 1 hour |

---

## Recommended Implementation Order

### Phase 1: Quick Wins (No external dependencies)
1. ✅ CORS restriction (environment variable)
2. ✅ Stadium location (already implemented, just wire up)
3. ✅ Betting lines in standings

### Phase 2: Core Product Features (High priority)
4. Model loading for projections (#8)
5. Game prop sheet generation (#9)
6. ML insights integration (#4)

### Phase 3: Infrastructure
7. Orchestration pipeline integration (#2)

### Phase 4: Content Enhancement (Low priority, requires API keys)
8. News API integration (#3)
9. LLM narratives (#6)
10. Content aggregation (#7)

---

## Environment Variables Needed

```bash
# Already supported
OPENWEATHER_API_KEY=<key>  # For weather data
ODDS_API_KEY=<key>          # For betting odds

# Would need to add
OPENAI_API_KEY=<key>        # For LLM narratives
YOUTUBE_API_KEY=<key>       # For video content
TWITTER_API_KEY=<key>       # For social content
ESPN_API_KEY=<key>          # For news (if required)
CORS_ORIGINS=https://app.example.com,https://example.com  # Production CORS
```

---

## Notes

- **No synthetic data used**: All placeholder returns documented as requiring external integration
- **Real data priority**: Items #4, #8, #9 should use real models and predictions when implemented
- **Cost considerations**: LLM integration (#6) has ongoing API costs
- **Scalability**: Content APIs (#7) may need caching layer for production use
