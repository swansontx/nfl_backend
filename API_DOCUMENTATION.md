# NFL Props API Documentation

FastAPI service for NFL prop recommendations, predictions, and backtesting.

## Base URL

```
Local: http://localhost:8000
Production: https://your-api.com
```

## Interactive Docs

- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`

Visit `http://localhost:8000/docs` to explore all endpoints interactively.

---

## Quick Start

### Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn backend.api.app:app --reload

# Or with Docker
docker build -t nfl-props-api .
docker run -p 8000:8000 nfl-props-api
```

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Get today's games
curl http://localhost:8000/api/v1/games/today

# Get recommendations for a game
curl http://localhost:8000/api/v1/recommendations/2024_17_KC_LV
```

---

## API Endpoints

### Health & Status

#### `GET /`
Root endpoint with API information.

```bash
curl http://localhost:8000/
```

Response:
```json
{
  "service": "NFL Props API",
  "version": "1.0.0",
  "status": "ok",
  "docs": "/docs"
}
```

#### `GET /health`
Health check with system status.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-12-24T10:00:00Z",
  "database": "connected",
  "games": 267,
  "players": 1247
}
```

#### `GET /status`
Detailed system status and configuration.

---

### Games

#### `GET /api/v1/games/`
Get list of games with filters.

**Query Parameters:**
- `season` (int): Filter by season
- `week` (int): Filter by week (1-18)
- `team` (str): Filter by team (e.g., "KC", "BUF")
- `date` (str): Filter by date (YYYY-MM-DD)
- `upcoming` (bool): Only upcoming games

```bash
# All games for week 17, 2024
curl "http://localhost:8000/api/v1/games/?season=2024&week=17"

# Upcoming KC games
curl "http://localhost:8000/api/v1/games/?team=KC&upcoming=true"
```

Response:
```json
{
  "games": [
    {
      "game_id": "2024_17_KC_LV",
      "season": 2024,
      "week": 17,
      "game_type": "REG",
      "home_team": "KC",
      "away_team": "LV",
      "game_date": "2024-12-25T13:00:00Z",
      "stadium": "Arrowhead Stadium",
      "completed": false
    }
  ],
  "total_count": 1,
  "season": 2024,
  "week": 17
}
```

#### `GET /api/v1/games/today`
Get today's games (convenience endpoint).

```bash
curl http://localhost:8000/api/v1/games/today
```

#### `GET /api/v1/games/{game_id}`
Get single game details.

```bash
curl http://localhost:8000/api/v1/games/2024_17_KC_LV
```

---

### Recommendations

#### `GET /api/v1/recommendations/{game_id}`
Get prop recommendations for a game.

**Query Parameters:**
- `limit` (int, 1-100): Max recommendations (default: 10)
- `min_confidence` (float, 0-1): Minimum confidence (default: 0.6)
- `markets` (list): Specific markets to analyze

```bash
# Top 10 recommendations
curl "http://localhost:8000/api/v1/recommendations/2024_17_KC_LV"

# Top 20 with high confidence only
curl "http://localhost:8000/api/v1/recommendations/2024_17_KC_LV?limit=20&min_confidence=0.75"

# Only receiving yards props
curl "http://localhost:8000/api/v1/recommendations/2024_17_KC_LV?markets=player_rec_yds"
```

Response:
```json
{
  "game_id": "2024_17_KC_LV",
  "game_time": "2024-12-25T13:00:00Z",
  "home_team": "KC",
  "away_team": "LV",
  "recommendations": [
    {
      "player_id": "mahomes_patrick",
      "player_name": "Patrick Mahomes",
      "position": "QB",
      "team": "KC",
      "market": "player_pass_yds",
      "line": 287.5,
      "model_prob": 0.65,
      "calibrated_prob": 0.62,
      "base_signal": 0.68,
      "matchup_signal": 0.55,
      "trend_signal": 0.72,
      "news_signal": 0.60,
      "roster_signal": 0.90,
      "overall_score": 0.742,
      "recommendation_strength": "strong",
      "confidence": 0.85,
      "reasoning": [
        "Model probability: 62.0%",
        "Hot streak: 5 game streak",
        "Favorable matchup vs LV (rank 28)"
      ],
      "flags": [],
      "edge": 0.10
    }
  ],
  "total_count": 1,
  "markets_analyzed": ["all"],
  "min_confidence": 0.6
}
```

#### `GET /api/v1/recommendations/{game_id}/parlays`
Get parlay recommendations.

**Query Parameters:**
- `parlay_size` (int, 2-6): Number of props (default: 3)
- `min_correlation` (float): Minimum correlation (default: 0.0)
- `max_correlation` (float): Maximum correlation (default: 0.8)
- `limit` (int, 1-20): Max parlays (default: 5)

```bash
# 3-leg parlays
curl "http://localhost:8000/api/v1/recommendations/2024_17_KC_LV/parlays"

# 4-leg parlays with positive correlation
curl "http://localhost:8000/api/v1/recommendations/2024_17_KC_LV/parlays?parlay_size=4&min_correlation=0.2"
```

Response:
```json
{
  "game_id": "2024_17_KC_LV",
  "parlays": [
    {
      "props": [
        {
          "player_name": "Patrick Mahomes",
          "market": "player_pass_yds",
          "line": 287.5,
          "probability": 0.62,
          "score": 0.74
        },
        {
          "player_name": "Travis Kelce",
          "market": "player_rec_yds",
          "line": 68.5,
          "probability": 0.58,
          "score": 0.69
        }
      ],
      "raw_probability": 0.36,
      "adjusted_probability": 0.42,
      "adjustment_factor": 1.17,
      "overall_score": 0.715,
      "correlation_impact": "positive",
      "confidence": 0.78,
      "reasoning": [
        "Parlay of 2 props",
        "Correlation: +0.65 (positive)",
        "QB-TE stack on same team"
      ]
    }
  ],
  "total_count": 1,
  "parlay_size": 2
}
```

#### `GET /api/v1/recommendations/player/{player_id}`
Get all recommendations for a player.

**Query Parameters:**
- `season` (int, required): Season year
- `week` (int): Specific week

```bash
# All Mahomes props for 2024
curl "http://localhost:8000/api/v1/recommendations/player/mahomes_patrick?season=2024"

# Week 17 only
curl "http://localhost:8000/api/v1/recommendations/player/mahomes_patrick?season=2024&week=17"
```

---

### Backtesting

#### `GET /api/v1/backtest/`
Run historical backtest.

**Query Parameters:**
- `season` (int, required): Season to test
- `kelly_fraction` (float, 0.1-1.0): Kelly fraction (default: 0.25)

```bash
# Backtest 2024 season
curl "http://localhost:8000/api/v1/backtest/?season=2024"

# With full Kelly
curl "http://localhost:8000/api/v1/backtest/?season=2024&kelly_fraction=1.0"
```

Response:
```json
{
  "start_date": "2024-09-01T00:00:00Z",
  "end_date": "2024-12-31T00:00:00Z",
  "total_games": 267,
  "total_projections": 8534,
  "markets_tested": ["player_rec_yds", "player_rush_yds", "player_pass_yds"],
  "metrics": {
    "brier_score": 0.218,
    "log_loss": 0.642,
    "roc_auc": 0.627,
    "total_bets": 247,
    "winning_bets": 134,
    "win_rate": 0.5425,
    "roi_percent": 7.84,
    "sharpe_ratio": 1.23,
    "max_drawdown_percent": 18.35,
    "initial_bankroll": 1000.0,
    "final_bankroll": 1078.40,
    "total_profit": 78.40
  },
  "kelly_fraction": 0.25
}
```

#### `GET /api/v1/backtest/signals`
Analyze signal effectiveness.

**Query Parameters:**
- `season` (int, required): Season to analyze

```bash
# Signal analysis for 2024
curl "http://localhost:8000/api/v1/backtest/signals?season=2024"
```

Response:
```json
{
  "signal_contributions": [
    {
      "signal_name": "base_signal",
      "standalone_auc": 0.623,
      "standalone_brier": 0.215,
      "correlation": 0.342,
      "optimal_weight": 0.350,
      "mean_value": 0.625,
      "std_value": 0.185
    }
  ],
  "combined_auc": 0.658,
  "combined_brier": 0.201,
  "best_signal_name": "base_signal",
  "best_signal_auc": 0.623,
  "optimal_weights": {
    "base_signal": 0.350,
    "trend_signal": 0.180,
    "matchup_signal": 0.120,
    "news_signal": 0.080,
    "roster_signal": 0.060
  },
  "n_samples": 8534
}
```

---

## Frontend Integration

### React Example

```javascript
// Get today's games
const getGames = async () => {
  const response = await fetch('http://localhost:8000/api/v1/games/today');
  const data = await response.json();
  return data.games;
};

// Get recommendations for a game
const getRecommendations = async (gameId) => {
  const response = await fetch(
    `http://localhost:8000/api/v1/recommendations/${gameId}?limit=20&min_confidence=0.7`
  );
  const data = await response.json();
  return data.recommendations;
};

// Get parlays
const getParlays = async (gameId) => {
  const response = await fetch(
    `http://localhost:8000/api/v1/recommendations/${gameId}/parlays?parlay_size=3`
  );
  const data = await response.json();
  return data.parlays;
};

// Run backtest
const runBacktest = async (season) => {
  const response = await fetch(
    `http://localhost:8000/api/v1/backtest/?season=${season}`
  );
  const data = await response.json();
  return data.metrics;
};
```

### TypeScript Types

```typescript
interface PropRecommendation {
  player_name: string;
  position: string;
  team: string;
  market: string;
  line: number;
  calibrated_prob: number;
  overall_score: number;
  recommendation_strength: 'elite' | 'strong' | 'moderate' | 'weak' | 'avoid';
  confidence: number;
  reasoning: string[];
  flags: string[];
  edge?: number;
}

interface Game {
  game_id: string;
  season: number;
  week: number;
  home_team: string;
  away_team: string;
  game_date: string;
  completed: boolean;
}

interface BacktestMetrics {
  roi_percent: number;
  sharpe_ratio: number;
  win_rate: number;
  total_bets: number;
  total_profit: number;
  max_drawdown_percent: number;
}
```

---

## Deployment

### Docker

```bash
# Build
docker build -t nfl-props-api .

# Run
docker run -p 8000:8000 \
  -e DATABASE_URL="postgresql://..." \
  nfl-props-api
```

### Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/nfl_props
    depends_on:
      - db

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: nfl_props
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

### Railway / Render / Heroku

```bash
# Push to Git
git push origin main

# Railway will auto-deploy from Dockerfile
# Set environment variables in dashboard
```

---

## Error Handling

All errors return consistent format:

```json
{
  "error": "Error type",
  "detail": "Detailed error message",
  "request_id": "uuid-here"
}
```

**Common Status Codes:**
- `200`: Success
- `400`: Bad request (invalid parameters)
- `404`: Not found (game/player doesn't exist)
- `500`: Internal server error

---

## Rate Limiting

(To be implemented)

- 100 requests/minute per IP
- Authenticated: 1000 requests/minute

---

## Authentication

(To be implemented)

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:8000/api/v1/recommendations/...
```

---

## Support

- **Issues**: GitHub Issues
- **Docs**: This file + `/docs` endpoint
- **Status**: `/health` endpoint
