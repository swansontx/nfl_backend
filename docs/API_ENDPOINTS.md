# NFL Props Backend API Documentation

**Base URL:** `http://localhost:8000`
**Version:** v1.0.0

## Overview

The NFL Props Backend API provides comprehensive data for NFL game predictions, player props, injuries, weather, insights, and content aggregation. All endpoints return JSON responses.

---

## Table of Contents

- [Core Endpoints](#core-endpoints)
- [Phase 1: News & Injuries](#phase-1-news--injuries)
- [Phase 2: ML Insights & Narratives](#phase-2-ml-insights--narratives)
- [Phase 3: Content Aggregation](#phase-3-content-aggregation)
- [Phase 4: Advanced Prop Analytics](#phase-4-advanced-prop-analytics)
- [Bonus: Weather](#bonus-weather)
- [Data Models](#data-models)

---

## Core Endpoints

### Health Check

```http
GET /health
```

**Description:** Check API health status

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2025-11-19T10:30:00Z"
}
```

---

### Get Projections

```http
GET /game/{game_id}/projections
```

**Description:** Get player prop projections for a specific game

**Parameters:**
- `game_id` (path) - Game ID in format `{season}_{week}_{away}_{home}` (e.g., `2025_10_KC_BUF`)

**Response:**
```json
{
  "game_id": "2025_10_KC_BUF",
  "projections": []
}
```

---

### Recompute Models

```http
POST /admin/recompute
```

**Description:** Trigger model recomputation for a specific game

**Request Body:**
```json
{
  "game_id": "2025_10_KC_BUF"
}
```

**Response:**
```json
{
  "status": "started",
  "game_id": "2025_10_KC_BUF"
}
```

---

## Phase 1: News & Injuries

### Get News & Updates

```http
GET /api/v1/news
```

**Description:** Get latest NFL news and injury updates

**Query Parameters:**
- `limit` (optional, default: 20) - Number of items to return
- `category` (optional) - Filter by category: `injury`, `news`, `analysis`
- `team` (optional) - Filter by team abbreviation (e.g., `KC`, `BUF`)

**Example Request:**
```http
GET /api/v1/news?limit=10&category=injury&team=KC
```

**Response:**
```json
[
  {
    "id": "injury_abc123",
    "title": "Patrick Mahomes - Questionable",
    "summary": "Ankle - Limited in practice",
    "source": "Sleeper API",
    "published_at": "2025-11-19T10:30:00Z",
    "url": null,
    "category": "injury",
    "related_players": ["Patrick Mahomes"],
    "related_teams": ["KC"]
  }
]
```

---

### Get Game Injuries

```http
GET /api/v1/games/{game_id}/injuries
```

**Description:** Get injury report for both teams in a specific game

**Parameters:**
- `game_id` (path) - Game ID (e.g., `2025_10_KC_BUF`)

**Example Request:**
```http
GET /api/v1/games/2025_10_KC_BUF/injuries
```

**Response:**
```json
{
  "game_id": "2025_10_KC_BUF",
  "away_team": "KC",
  "home_team": "BUF",
  "away_injuries": [
    {
      "player_id": "sleeper_abc123",
      "player_name": "Patrick Mahomes",
      "team": "KC",
      "position": "QB",
      "injury_status": "Questionable",
      "injury_body_part": "Ankle",
      "injury_notes": "Limited in practice",
      "last_updated": "2025-11-19T10:30:00Z"
    }
  ],
  "home_injuries": [],
  "last_updated": "2025-11-19T10:30:00Z"
}
```

**Injury Status Values:**
- `Questionable` - Uncertain (25-75% chance to play)
- `Doubtful` - Unlikely (<25%)
- `Out` - Will not play
- `IR` - Injured reserve

---

## Phase 2: ML Insights & Narratives

### Get Game Insights

```http
GET /api/v1/games/{game_id}/insights
```

**Description:** Get ML-powered matchup insights for a game

**Parameters:**
- `game_id` (path) - Game ID (e.g., `2025_10_KC_BUF`)

**Features:**
- Historical performance trends
- Matchup advantages/disadvantages
- Key statistical edges
- Weather impact analysis
- Injury impact analysis

**Response:**
```json
[
  {
    "insight_type": "trend",
    "title": "QB Performance vs. Defense",
    "description": "Away team QB has averaged 285 passing yards vs. similar defenses",
    "confidence": 0.82,
    "supporting_data": {
      "avg_yards": 285,
      "sample_size": 6,
      "trend": "increasing"
    }
  },
  {
    "insight_type": "matchup",
    "title": "Run Defense Vulnerability",
    "description": "Home team allows 4.8 yards per carry, 2nd worst in league",
    "confidence": 0.91,
    "supporting_data": {
      "ypc_allowed": 4.8,
      "league_rank": 30,
      "last_3_games": 5.2
    }
  }
]
```

**Insight Types:**
- `trend` - Historical performance trends
- `stat` - Statistical analysis
- `matchup` - Head-to-head matchup data
- `weather` - Weather impact
- `injury` - Injury impact

---

### Get Game Narrative

```http
GET /api/v1/games/{game_id}/narrative
```

**Description:** Get AI-generated game narratives and storylines

**Parameters:**
- `game_id` (path) - Game ID (e.g., `2025_10_KC_BUF`)

**Features:**
- Game preview
- Key player matchups
- Betting angles and value props
- Historical context
- Weather and external factors

**Response:**
```json
[
  {
    "narrative_type": "preview",
    "content": "This AFC showdown features two high-powered offenses. The away team's passing attack ranks 2nd in the league, while the home team's defense has struggled against elite QBs.",
    "generated_at": "2025-11-19T10:30:00Z"
  },
  {
    "narrative_type": "key_matchups",
    "content": "Watch for the battle in the trenches. The away team's offensive line has allowed just 12 sacks this season, while the home team's pass rush leads the league with 42 sacks.",
    "generated_at": "2025-11-19T10:30:00Z"
  },
  {
    "narrative_type": "betting_angle",
    "content": "Value opportunity on the away team's RB receiving props. He's averaged 6.2 receptions in games where the team is favored, and books have his line at 4.5.",
    "generated_at": "2025-11-19T10:30:00Z"
  }
]
```

**Narrative Types:**
- `preview` - Game overview
- `key_matchups` - Player/unit matchups
- `betting_angle` - Betting insights
- `historical` - Historical context
- `weather` - Weather impact

---

## Phase 3: Content Aggregation

### Get Game Content

```http
GET /api/v1/games/{game_id}/content
```

**Description:** Get aggregated content (articles, videos, podcasts) for a game

**Query Parameters:**
- `content_type` (optional) - Filter by type: `article`, `video`, `podcast`
- `limit` (optional, default: 10) - Number of items to return

**Example Request:**
```http
GET /api/v1/games/2025_10_KC_BUF/content?content_type=video&limit=5
```

**Response:**
```json
[
  {
    "content_type": "article",
    "title": "Week 12 Preview: Key Matchups to Watch",
    "source": "ESPN",
    "url": "https://espn.com/nfl/preview",
    "published_at": "2025-11-19T08:00:00Z",
    "thumbnail_url": "https://placeholder.com/thumbnail.jpg"
  },
  {
    "content_type": "video",
    "title": "Film Breakdown: Offensive Schemes",
    "source": "YouTube",
    "url": "https://youtube.com/watch?v=example",
    "published_at": "2025-11-19T09:00:00Z",
    "thumbnail_url": "https://placeholder.com/video-thumb.jpg"
  }
]
```

**Content Types:**
- `article` - Written articles
- `video` - Video content
- `podcast` - Audio podcasts

---

## Phase 4: Advanced Prop Analytics

### Find High-Value Props

```http
GET /api/v1/props/value
```

**Description:** Find the best value prop bets using advanced analytics including edge calculation, confidence scoring, and Kelly criterion bet sizing

**Query Parameters:**
- `min_edge` (optional, default: 5.0) - Minimum edge percentage required (e.g., 7.0 for 7%+ edge)
- `min_grade` (optional, default: "B") - Minimum value grade: `A+`, `A`, `B+`, `B`, `C`, `F`
- `limit` (optional, default: 10) - Number of props to return

**Features:**
- Edge calculation (true probability vs implied probability from odds)
- Confidence-based value grading (A+ = elite value, F = no value)
- Kelly criterion bet sizing recommendations
- Confidence intervals for model projections
- Multiple sportsbook comparison ready

**Example Request:**
```http
GET /api/v1/props/value?min_edge=7.0&min_grade=A&limit=5
```

**Response:**
```json
{
  "total_opportunities": 45,
  "best_values": [
    {
      "player_name": "Patrick Mahomes",
      "prop_type": "passing_yards",
      "sportsbook_line": 285.5,
      "model_projection": 302.3,
      "confidence_interval": [285.1, 319.5],
      "recommendation": "OVER",
      "edge_over": 12.4,
      "edge_under": -8.2,
      "value_grade": "A",
      "confidence": 0.78,
      "suggested_stake_pct": 3.1,
      "sportsbook": "DraftKings",
      "odds": -110
    }
  ],
  "filters_applied": {
    "min_edge": 7.0,
    "min_grade": "A",
    "limit": 5
  }
}
```

**Value Grading System:**
- `A+` - Elite value (15%+ edge, 75%+ confidence)
- `A` - Excellent value (10-14.9% edge, 70%+ confidence)
- `B+` - Good value (7-9.9% edge, 65%+ confidence)
- `B` - Decent value (5-6.9% edge, 60%+ confidence)
- `C` - Marginal value (3-4.9% edge)
- `F` - No value (<3% edge)

**Kelly Criterion:**
The `suggested_stake_pct` is calculated using fractional Kelly (0.25):
```
Stake % = (Edge × Confidence) / 4
```

---

### Get Player Insights

```http
GET /api/v1/players/{player_id}/insights
```

**Description:** Get comprehensive data-driven insights for a specific player including trend analysis, consistency metrics, and prop recommendations

**Parameters:**
- `player_id` (path) - Player identifier

**Features:**
- Statistical trend detection (linear regression)
- Consistency analysis (coefficient of variation)
- Matchup-specific insights
- Weather impact analysis
- Injury context
- Actionable prop recommendations

**Example Request:**
```http
GET /api/v1/players/player_001/insights
```

**Response:**
```json
{
  "player_id": "player_001",
  "player_name": "Patrick Mahomes",
  "position": "QB",
  "team": "KC",
  "insights": [
    {
      "insight_type": "trend",
      "title": "Patrick Mahomes Trending Up in Passing Yards",
      "description": "Patrick Mahomes has been on an upward trend, averaging 315.0 passing yards in recent games vs 278.3 earlier (+13.2%). Performance is very_consistent.",
      "confidence": 0.82,
      "impact_level": "high",
      "supporting_data": {
        "stat": "passing_yards",
        "recent_avg": 315.0,
        "earlier_avg": 278.3,
        "pct_change": 13.2,
        "trend_strength": 0.68,
        "consistency": "very_consistent",
        "sample_size": 9,
        "recent_games": [320, 305, 320]
      },
      "recommendation": "Consider OVER on Patrick Mahomes passing_yards props"
    },
    {
      "insight_type": "matchup",
      "title": "Favorable Matchup: Patrick Mahomes vs Weak Defense",
      "description": "Patrick Mahomes (avg 295.5 passing yards) faces a #28 ranked defense. Historical performance suggests exploitable matchup. League avg: 245.0.",
      "confidence": 0.86,
      "impact_level": "high",
      "supporting_data": {
        "player_avg": 295.5,
        "league_avg": 245.0,
        "opponent_rank": 28,
        "percentile": 12.5
      },
      "recommendation": "Value play on Patrick Mahomes OVER"
    }
  ],
  "season_avg": {
    "passing_yards": 295.5,
    "passing_tds": 2.3,
    "completions": 26.8
  },
  "recent_performance": [
    {"game": "Week 10", "passing_yards": 320},
    {"game": "Week 9", "passing_yards": 305},
    {"game": "Week 8", "passing_yards": 320}
  ]
}
```

**Insight Types:**
- `trend` - Performance trend analysis (increasing/decreasing/stable)
- `matchup` - Opponent-specific advantages
- `weather` - Weather impact on performance
- `injury` - Injury impact on role/usage
- `consistency` - Reliability metrics

**Impact Levels:**
- `high` - Significant factor (trend strength >0.6, or top/bottom 25% matchup)
- `medium` - Moderate factor
- `low` - Minor consideration

---

### Compare Props

```http
GET /api/v1/props/compare
```

**Description:** Side-by-side comparison of props across multiple players for the same stat category

**Query Parameters:**
- `player_ids` (required) - Comma-separated player IDs (max 5)
- `prop_type` (optional, default: "passing_yards") - Stat to compare
- `sportsbook` (optional) - Filter by specific sportsbook

**Example Request:**
```http
GET /api/v1/props/compare?player_ids=p1,p2,p3&prop_type=rushing_yards
```

**Response:**
```json
{
  "prop_type": "rushing_yards",
  "players_compared": 3,
  "comparisons": [
    {
      "player_name": "Christian McCaffrey",
      "player_id": "p1",
      "line": 95.5,
      "model_projection": 108.2,
      "edge_over": 15.3,
      "value_grade": "A+",
      "recommendation": "OVER"
    },
    {
      "player_name": "Derrick Henry",
      "player_id": "p2",
      "line": 88.5,
      "model_projection": 91.7,
      "edge_over": 4.2,
      "value_grade": "C",
      "recommendation": "PASS"
    }
  ],
  "best_value": {
    "player_name": "Christian McCaffrey",
    "edge": 15.3,
    "grade": "A+"
  }
}
```

---

### Get Game Prop Sheet

```http
GET /api/v1/games/{game_id}/prop-sheet
```

**Description:** Comprehensive prop sheet for a specific game with all players, organized by category and ranked by value

**Parameters:**
- `game_id` (path) - Game ID (e.g., `2025_10_KC_BUF`)

**Example Request:**
```http
GET /api/v1/games/2025_10_KC_BUF/prop-sheet
```

**Response:**
```json
{
  "game_id": "2025_10_KC_BUF",
  "total_props": 67,
  "high_value_props": 12,
  "categories": {
    "passing": {
      "props": [
        {
          "player": "Patrick Mahomes",
          "stat": "passing_yards",
          "line": 285.5,
          "projection": 302.3,
          "edge": 12.4,
          "grade": "A",
          "recommendation": "OVER"
        }
      ]
    },
    "rushing": {
      "props": [...]
    },
    "receiving": {
      "props": [...]
    },
    "scoring": {
      "props": [...]
    }
  },
  "top_plays": [
    {
      "player": "Patrick Mahomes",
      "prop": "passing_yards OVER 285.5",
      "edge": 12.4,
      "grade": "A"
    },
    {
      "player": "Travis Kelce",
      "prop": "receiving_yards OVER 68.5",
      "edge": 10.1,
      "grade": "A"
    }
  ]
}
```

**Categories:**
- `passing` - QB stats (yards, TDs, completions, attempts)
- `rushing` - RB/QB rushing (yards, TDs, attempts)
- `receiving` - WR/TE/RB receiving (yards, TDs, receptions)
- `scoring` - Anytime TD, first TD scorer

---

## Bonus: Weather

### Get Game Weather

```http
GET /api/v1/games/{game_id}/weather
```

**Description:** Get weather forecast for a game

**Parameters:**
- `game_id` (path) - Game ID (e.g., `2025_10_KC_BUF`)

**Response:**
```json
{
  "temperature": 45,
  "temp_unit": "F",
  "condition": "Clear",
  "wind_speed": 12,
  "wind_unit": "mph",
  "humidity": 65,
  "precipitation_chance": 10,
  "is_dome": false
}
```

**Weather Conditions:**
- `Clear` - Clear skies
- `Clouds` - Cloudy
- `Rain` - Rainy
- `Snow` - Snowing
- `Fog` - Foggy

---

## Data Models

### NewsItem

```typescript
{
  id: string;
  title: string;
  summary: string;
  source: string;
  published_at: string;  // ISO 8601 timestamp
  url?: string;
  category: "injury" | "news" | "analysis";
  related_players: string[];
  related_teams: string[];
}
```

### InjuryRecord

```typescript
{
  player_id: string;
  player_name: string;
  team: string;  // Team abbreviation
  position: string;
  injury_status: "Questionable" | "Doubtful" | "Out" | "IR";
  injury_body_part?: string;
  injury_notes?: string;
  last_updated: string;  // ISO 8601 timestamp
}
```

### MatchupInsight

```typescript
{
  insight_type: "trend" | "stat" | "matchup" | "weather" | "injury";
  title: string;
  description: string;
  confidence: number;  // 0.0 to 1.0
  supporting_data: Record<string, any>;
}
```

### GameNarrative

```typescript
{
  narrative_type: "preview" | "key_matchups" | "betting_angle" | "historical" | "weather";
  content: string;
  generated_at: string;  // ISO 8601 timestamp
}
```

### ContentItem

```typescript
{
  content_type: "article" | "video" | "podcast";
  title: string;
  source: string;
  url: string;
  published_at: string;  // ISO 8601 timestamp
  thumbnail_url?: string;
}
```

### WeatherData

```typescript
{
  temperature: number;  // Integer
  temp_unit: "F" | "C";
  condition: string;
  wind_speed: number;  // Integer
  wind_unit: "mph" | "kph";
  humidity: number;  // 0-100
  precipitation_chance: number;  // 0-100
  is_dome: boolean;
}
```

### PropValue

```typescript
{
  player_name: string;
  prop_type: string;  // e.g., "passing_yards", "rushing_yards"
  sportsbook_line: number;
  model_projection: number;
  confidence_interval: [number, number];  // [lower, upper] bounds
  recommendation: "OVER" | "UNDER" | "PASS";
  edge_over: number;  // Percentage edge on OVER
  edge_under: number;  // Percentage edge on UNDER
  value_grade: "A+" | "A" | "B+" | "B" | "C" | "F";
  confidence: number;  // 0.0 to 1.0
  suggested_stake_pct: number;  // Kelly criterion stake percentage
  sportsbook: string;
  odds: number;  // American odds (e.g., -110, +150)
}
```

### PropComparison

```typescript
{
  player_name: string;
  player_id: string;
  line: number;
  model_projection: number;
  edge_over: number;
  value_grade: "A+" | "A" | "B+" | "B" | "C" | "F";
  recommendation: "OVER" | "UNDER" | "PASS";
}
```

### PlayerInsight

```typescript
{
  player_id: string;
  player_name: string;
  position: string;
  team: string;
  insights: MatchupInsight[];
  season_avg: Record<string, number>;
  recent_performance: Array<{
    game: string;
    [stat: string]: number | string;
  }>;
}
```

### GamePropSheet

```typescript
{
  game_id: string;
  total_props: number;
  high_value_props: number;  // Count of A/A+ graded props
  categories: {
    passing: { props: PropValue[] };
    rushing: { props: PropValue[] };
    receiving: { props: PropValue[] };
    scoring: { props: PropValue[] };
  };
  top_plays: Array<{
    player: string;
    prop: string;
    edge: number;
    grade: string;
  }>;
}
```

---

## Error Responses

All endpoints may return the following error responses:

### 400 Bad Request
```json
{
  "detail": "Invalid request parameters"
}
```

### 404 Not Found
```json
{
  "detail": "Resource not found"
}
```

### 422 Validation Error
```json
{
  "detail": [
    {
      "loc": ["body", "game_id"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error"
}
```

---

## Rate Limiting

Currently no rate limiting is enforced. In production, consider implementing rate limits:
- General endpoints: 100 requests/minute
- ML/AI endpoints: 20 requests/minute
- Admin endpoints: 10 requests/minute

---

## CORS

CORS is enabled for all origins in development (`*`). In production, restrict to your frontend domain:

```python
allow_origins=["https://your-frontend-domain.com"]
```

---

## Authentication

Currently, no authentication is required. For production, consider implementing:
- API keys for external integrations
- JWT tokens for user-specific data
- OAuth for admin endpoints

---

## External APIs Used

### Sleeper API
- **Purpose:** NFL player data and injuries
- **Docs:** https://docs.sleeper.app/
- **Rate Limit:** No official limit, be respectful
- **Authentication:** None required

### OpenWeather API
- **Purpose:** Game weather forecasts
- **Docs:** https://openweathermap.org/api
- **Rate Limit:** 60 calls/minute (free tier)
- **Authentication:** API key required (`OPENWEATHER_API_KEY`)

---

## Development Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

3. **Run the server:**
   ```bash
   uvicorn backend.api.app:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Access API docs:**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

---

## Testing

Run the test suite:
```bash
# Core API tests
pytest tests/test_api.py -v

# Enhanced endpoint tests
pytest tests/test_enhanced_endpoints.py -v

# Run all tests
pytest tests/ -v
```

**Test Coverage:**
- `test_api.py` - 20 tests for core endpoints, news, injuries, insights, narratives, content, weather
- `test_enhanced_endpoints.py` - 11 tests for prop value finder, player insights, prop comparisons, prop sheets

**Total:** 31 tests, all passing ✅

---

## Future Enhancements

### Completed ✅
- [x] Sleeper API integration for injuries (Phase 1)
- [x] OpenWeather API integration (Bonus)
- [x] Statistical insight generation engine (Phase 2)
- [x] Narrative template system with LLM scaffolding (Phase 2)
- [x] Prop value finder with edge calculations (Phase 4)
- [x] Kelly criterion bet sizing (Phase 4)
- [x] Player-specific trend analysis (Phase 4)
- [x] Comprehensive test coverage (31 tests)

### Short Term
- [ ] Implement actual news aggregation (ESPN, NFL.com RSS feeds)
- [ ] Add caching layer for Sleeper API responses (1-hour TTL)
- [ ] Connect to real sportsbook line feeds (DraftKings, FanDuel APIs)
- [ ] Add stadium database with coordinates for accurate weather lookups
- [ ] Integrate actual ML model outputs from `outputs/predictions/`

### Medium Term
- [ ] Implement full LLM integration for narratives (OpenAI GPT-4 or Claude)
- [ ] Add content aggregation (YouTube Data API, podcast RSS)
- [ ] Build WebSocket endpoint for real-time line movement alerts
- [ ] Add user authentication and personalized betting history
- [ ] Implement bet tracking and results analysis
- [ ] Add bankroll management recommendations

### Long Term
- [ ] GraphQL API alongside REST for flexible querying
- [ ] Real-time injury updates via webhooks
- [ ] Train custom ML models for prop predictions using nflverse data
- [ ] Mobile app-specific endpoints with reduced payloads
- [ ] Multi-sport expansion (NBA, MLB)
- [ ] Social features (share plays, leaderboards)

---

## Support

For issues or questions:
- GitHub Issues: [Create an issue](https://github.com/your-repo/issues)
- Documentation: See `docs/` directory
- API Status: Check `/health` endpoint
