# API Endpoints Documentation

Complete reference for all NFL Props Backend API endpoints.

## Base URL

```
http://localhost:8000
```

API Documentation (Swagger): http://localhost:8000/docs

---

## ✅ COMPLETE ENDPOINT COVERAGE

All endpoints required by frontend are now implemented!

### Teams Endpoints (COMPLETE)
- ✅ `GET /api/v1/teams` - List all teams
- ✅ `GET /api/v1/teams/{team_id}` - Team details  
- ✅ `GET /api/v1/teams/{team_id}/stats` - Team statistics
- ✅ `GET /api/v1/teams/{team_id}/schedule` - Team schedule
- ✅ `GET /api/v1/teams/{team_id}/news` - Team news

### Games Endpoints (COMPLETE)
- ✅ `GET /api/v1/games` - List games (filterable)
- ✅ `GET /api/v1/games/{game_id}` - Game details
- ✅ `GET /api/v1/games/{game_id}/boxscore` - Complete boxscore
- ✅ `GET /api/v1/games/{game_id}/weather` - Weather
- ✅ `GET /api/v1/games/{game_id}/insights` - Insights
- ✅ `GET /api/v1/games/{game_id}/projections` - Projections

### Players Endpoints (SCAFFOLDED)
- ✅ `GET /api/v1/players` - Search players
- ✅ `GET /api/v1/players/{player_id}` - Player details
- ✅ `GET /api/v1/players/{player_id}/stats` - Player stats
- ✅ `GET /api/v1/players/{player_id}/gamelogs` - Game logs
- ✅ `GET /api/v1/players/{player_id}/insights` - Player insights

### Prop Betting Endpoints (COMPLETE)
- ✅ `GET /api/v1/odds/current` - Current DraftKings odds
- ✅ `GET /api/v1/props/trending` - Trending props (line movement tracking)
- ✅ `GET /api/v1/props/value` - High-value prop recommendations
- ✅ `GET /api/v1/props/compare` - Compare props across players

### League Info Endpoints (COMPLETE)
- ✅ `GET /api/v1/standings` - NFL standings by division
- ✅ `GET /api/v1/news` - Latest NFL news and injuries

---

## Quick Examples

### Get All Teams
```bash
curl http://localhost:8000/api/v1/teams
```

### Get Team Schedule
```bash
curl http://localhost:8000/api/v1/teams/KC/schedule?season=2024
```

### Get Games for Week 12
```bash
curl http://localhost:8000/api/v1/games?week=12&season=2024
```

### Get Game Details
```bash
curl http://localhost:8000/api/v1/games/2024_12_KC_BUF
```

### Get Complete Boxscore
```bash
curl http://localhost:8000/api/v1/games/2024_10_KC_BUF/boxscore
```

### Get Current Odds (DraftKings)
```bash
curl http://localhost:8000/api/v1/odds/current?week=12
```

### Get Trending Props
```bash
curl http://localhost:8000/api/v1/props/trending?week=12&limit=20
```

### Get NFL Standings
```bash
curl http://localhost:8000/api/v1/standings?season=2024
```

---

## Detailed Endpoint Documentation

### Prop Betting Endpoints

#### GET /api/v1/odds/current

Get current DraftKings prop odds for all upcoming games.

**Query Parameters:**
- `week` (optional): Week number (defaults to current week)
- `market` (optional): Filter by specific market (e.g., 'player_pass_yds')

**Response:**
```json
{
  "week": 12,
  "total_games": 14,
  "total_props": 450,
  "snapshot_timestamp": "2024-11-19T12:00:00",
  "games": [
    {
      "game_id": "2024_12_KC_BUF",
      "home_team": "KC",
      "away_team": "BUF",
      "commence_time": "2024-11-24T20:20:00",
      "props_by_market": {
        "player_pass_yds": [
          {
            "player": "Patrick Mahomes",
            "market": "player_pass_yds",
            "line": 275.5,
            "over_odds": -110,
            "under_odds": -110,
            "timestamp": "2024-11-19T12:00:00",
            "bookmaker": "draftkings"
          }
        ]
      }
    }
  ]
}
```

#### GET /api/v1/props/trending

Get trending prop lines with directional categorization and 3-week movement tracking.

**Query Parameters:**
- `week` (optional): Week number
- `limit` (optional): Number of props per category (default: 20)

**Response:**
```json
{
  "week": 12,
  "snapshot_count": 15,
  "current_timestamp": "2024-11-19T12:00:00",

  "hottest_movers": [
    {
      "player": "Patrick Mahomes",
      "market": "player_pass_yds",
      "bookmaker": "draftkings",
      "opening_line": 265.5,
      "current_line": 275.5,
      "line_movement": 10.0,
      "line_movement_pct": 3.8,
      "over_odds": -110,
      "under_odds": -110,
      "direction": "up",
      "opening_timestamp": "2024-11-12T12:00:00",
      "current_timestamp": "2024-11-19T12:00:00",
      "days_tracked": 7,
      "badge": "+10.0 pts (7 days)",
      "badge_pct": "+4% (7 days)",
      "icon": "⬆️",
      "color": "green",
      "strength": "🔥"
    }
  ],

  "lines_moving_up": [
    {
      "player": "Josh Allen",
      "market": "player_pass_yds",
      "line_movement": 5.0,
      "direction": "up"
    }
  ],

  "lines_moving_down": [
    {
      "player": "Travis Kelce",
      "market": "player_reception_yds",
      "line_movement": -3.5,
      "direction": "down"
    }
  ],

  "sustained_trends": [
    {
      "player": "Christian McCaffrey",
      "market": "player_rush_yds",
      "week_1_line": 85.5,
      "week_2_line": 90.5,
      "week_3_line": 95.5,
      "current_line": 95.5,
      "total_movement": 10.0,
      "total_movement_pct": 11.7,
      "direction": "up",
      "consistency": "3/3 weeks up",
      "badge": "+10.0 pts (3 weeks)",
      "strength": "🔥🔥"
    }
  ],

  "summary": {
    "hottest_movers": {
      "total": 20,
      "strong": 8,
      "avg_movement": 4.2
    },
    "lines_moving_up": {
      "total": 12,
      "strong": 5,
      "avg_movement": 3.1
    },
    "lines_moving_down": {
      "total": 8,
      "strong": 2,
      "avg_movement": -2.8
    },
    "sustained_trends": {
      "total": 15,
      "strong": 6,
      "trending_up": 9,
      "trending_down": 6
    }
  }
}
```

**Trending Categories:**
- 🔥 **Hottest movers**: Biggest absolute line changes (regardless of direction)
- ⬆️ **Lines moving up**: Props getting harder to hit Over (line increased)
- ⬇️ **Lines moving down**: Props getting easier to hit Over (line decreased)
- **Sustained trends**: 3-week consistent directional patterns

### League Info Endpoints

#### GET /api/v1/standings

Get NFL standings by division and conference.

**Query Parameters:**
- `season` (optional): Season year (default: 2024)
- `week` (optional): Week number (returns current standings if not specified)

**Response:**
```json
{
  "season": 2024,
  "week": 12,
  "standings": {
    "afc_east": [
      {
        "team": "BUF",
        "wins": 9,
        "losses": 2,
        "ties": 0,
        "win_pct": 0.818
      }
    ],
    "afc_north": [...],
    "afc_south": [...],
    "afc_west": [...],
    "nfc_east": [...],
    "nfc_north": [...],
    "nfc_south": [...],
    "nfc_west": [...]
  }
}
```

---

## Complete Documentation

See Swagger docs for detailed request/response schemas:
http://localhost:8000/docs

All endpoints support:
- Proper HTTP status codes (200, 404, 400, 500)
- JSON responses
- CORS for frontend integration
- Query parameter filtering

---

## Implementation Notes

**Fully Functional:**
- Teams (all endpoints working with real data)
- Schedule/Games (working with nflverse schedules)
- Boxscores (generated from play-by-play data)

**Structured (Ready for Data Integration):**
- Team stats (structure ready, needs aggregation)
- Player endpoints (structure ready, needs player_lookup/stats CSV)
- Betting lines (structure ready, needs odds API)

See full API details in Swagger docs!
