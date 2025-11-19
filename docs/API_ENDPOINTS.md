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
