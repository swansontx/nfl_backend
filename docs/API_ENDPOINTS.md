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
pytest tests/test_api.py -v
```

All 20 API tests should pass.

---

## Future Enhancements

### Short Term
- [ ] Implement actual news aggregation (ESPN, NFL.com RSS)
- [ ] Add caching for Sleeper API responses
- [ ] Integrate with actual ML models for insights
- [ ] Add stadium database for weather lookups

### Medium Term
- [ ] Implement LLM integration for narratives (OpenAI/Claude)
- [ ] Add content aggregation (YouTube, podcasts)
- [ ] Implement WebSocket for real-time updates
- [ ] Add user authentication and personalization

### Long Term
- [ ] GraphQL API alongside REST
- [ ] Real-time injury updates via webhooks
- [ ] Custom ML models for prop predictions
- [ ] Mobile app-specific endpoints

---

## Support

For issues or questions:
- GitHub Issues: [Create an issue](https://github.com/your-repo/issues)
- Documentation: See `docs/` directory
- API Status: Check `/health` endpoint
