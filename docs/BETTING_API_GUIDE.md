# Betting Infrastructure API Guide

This guide covers the new betting infrastructure APIs that provide professional-grade prop betting tools.

## Table of Contents

1. [Overview](#overview)
2. [Parlay Optimizer API](#parlay-optimizer-api)
3. [CLV Tracking API](#clv-tracking-api)
4. [Recommendations API](#recommendations-api)
5. [Configuration](#configuration)
6. [Complete Workflow Example](#complete-workflow-example)

---

## Overview

The betting infrastructure APIs provide:

- **Correlation-aware parlay optimization** - Build parlays that account for same-game correlation
- **CLV tracking** - The gold standard metric for evaluating betting model quality
- **Recommendation management** - Log bets for continuous improvement via meta trust model
- **Probability calibration** - Proper distributional predictions from quantile models

All endpoints are part of the `/api/v1/betting/` namespace and use the `Betting` tag.

---

## Parlay Optimizer API

### `GET /api/v1/betting/parlays/suggestions`

Get correlation-aware parlay suggestions that account for same-game correlation, game script scenarios, and risk-adjusted sizing.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `game_ids` | string (optional) | all games | Comma-separated game IDs to analyze |
| `max_legs` | int | 4 | Maximum legs per parlay |
| `min_parlay_ev` | float | 0.10 | Minimum EV to suggest (10%) |
| `limit` | int | 10 | Number of suggestions to return |

#### Example Request

```bash
curl "http://localhost:8000/api/v1/betting/parlays/suggestions?max_legs=3&min_parlay_ev=0.12&limit=5"
```

#### Example Response

```json
{
  "total_suggestions": 5,
  "parlays": [
    {
      "legs": [
        {
          "player_name": "Patrick Mahomes",
          "prop_type": "player_pass_yds",
          "side": "over",
          "line": 275.5,
          "odds": -110,
          "projection": 295.3,
          "hit_probability": 0.68
        },
        {
          "player_name": "Travis Kelce",
          "prop_type": "player_reception_yds",
          "side": "over",
          "line": 65.5,
          "odds": -110,
          "projection": 78.2,
          "hit_probability": 0.64
        },
        {
          "player_name": "Tyreek Hill",
          "prop_type": "player_receptions",
          "side": "over",
          "line": 6.5,
          "odds": -110,
          "projection": 7.8,
          "hit_probability": 0.61
        }
      ],
      "combined_odds": +450,
      "probabilities": {
        "raw": 0.265,
        "adjusted": 0.198,
        "correlation_penalty": -25.3
      },
      "ev": 12.5,
      "recommended_stake_pct": 2.8,
      "confidence": "MEDIUM",
      "scenarios": {
        "SHOOTOUT": 0.40,
        "DEFENSIVE_SLOG": 0.15,
        "RB_DOMINATION": 0.20,
        "BLOWOUT_FAVORITE": 0.15,
        "BLOWOUT_UNDERDOG": 0.10
      }
    }
  ],
  "filters": {
    "max_legs": 3,
    "min_parlay_ev": 0.12
  },
  "warning": "Parlay betting is high-risk. These suggestions account for correlation but variance is still significant."
}
```

#### Response Fields

**Probabilities:**
- `raw`: Probability assuming independence (incorrect for same-game parlays)
- `adjusted`: Correlation-adjusted probability (actual)
- `correlation_penalty`: How much correlation reduced probability (%)

**Scenarios:**
- Probability estimates for each game script scenario
- Used to adjust prop hit probabilities based on game flow

**Correlation Adjustment Examples:**
- Same-team QB + WR: ~25-35% penalty
- Opposing QBs: ~15-20% penalty
- Different games: 0% (independent)

---

## CLV Tracking API

### `GET /api/v1/betting/clv/report`

Get Closing Line Value (CLV) report for recent bets. **CLV is the gold standard for evaluating betting model quality.**

> "Bettors who beat closing lines by 1-2% have positive ROI over 10,000+ bets" - Pinnacle Sports

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `last_n` | int | 50 | Number of recent bets to analyze |

#### Example Request

```bash
curl "http://localhost:8000/api/v1/betting/clv/report?last_n=100"
```

#### Example Response

```json
{
  "summary": {
    "total_bets": 100,
    "avg_clv": 1.8,
    "median_clv": 1.5,
    "positive_clv_rate": 0.62,
    "max_clv": 7.5,
    "min_clv": -3.2
  },
  "overall_metrics": {
    "total_bets": 100,
    "avg_clv": 1.8,
    "median_clv": 1.5,
    "positive_clv_rate": 0.62,
    "win_rate": 0.547
  },
  "by_prop_type": {
    "player_pass_yds": {
      "count": 35,
      "avg_clv": 2.3,
      "positive_clv_rate": 0.68,
      "win_rate": 0.571
    },
    "player_rush_yds": {
      "count": 28,
      "avg_clv": 1.5,
      "positive_clv_rate": 0.59,
      "win_rate": 0.536
    },
    "player_receptions": {
      "count": 22,
      "avg_clv": 1.9,
      "positive_clv_rate": 0.64,
      "win_rate": 0.545
    }
  },
  "clv_buckets": {
    "very_positive (>2)": {
      "count": 28,
      "win_rate": 0.607
    },
    "positive (0-2)": {
      "count": 34,
      "win_rate": 0.529
    },
    "negative (<0)": {
      "count": 38,
      "win_rate": 0.500
    }
  },
  "top_clv_bets": [
    {
      "bet_id": "2024_12_KC_BUF_mahomes_pass_yds",
      "player_name": "Patrick Mahomes",
      "prop_type": "player_pass_yds",
      "clv": 7.5,
      "opening_line": 275.5,
      "closing_line": 283.0,
      "won": true
    }
  ],
  "interpretation": {
    "avg_clv": "GOOD - Sustainable edge, profitable long-term",
    "positive_clv_rate": "EXCELLENT - Beating closing line 60%+ of the time"
  }
}
```

#### CLV Interpretation

**Average CLV:**
- `≥ 2.0`: EXCELLENT - Elite market-beating performance
- `≥ 1.0`: GOOD - Sustainable edge, profitable long-term
- `≥ 0.5`: FAIR - Slight edge, marginally profitable
- `≥ 0.0`: NEUTRAL - Breaking even with closing lines
- `< 0.0`: POOR - Betting on wrong side of information

**Positive CLV Rate:**
- `≥ 60%`: EXCELLENT - Beating closing line most of the time
- `≥ 55%`: GOOD - Consistently finding value
- `≥ 50%`: FAIR - Slightly better than market
- `< 50%`: POOR - Losing to closing line more than winning

---

## Recommendations API

### `POST /api/v1/betting/recommendations`

Log a prop recommendation for CLV tracking and meta trust model training.

**Call this endpoint:**
- Every time a recommendation is generated (whether bet or not)
- To build CLV tracking history
- To generate training data for meta trust model

#### Request Body

```json
{
  "player_name": "Patrick Mahomes",
  "player_id": "player_001",
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
  "games_sampled": 12,
  "model_r2": 0.58,
  "stake_pct": 2.5
}
```

#### Response

```json
{
  "status": "logged",
  "bet_id": "2024_12_KC_BUF_player_001_player_pass_yds_over",
  "message": "Recommendation logged for CLV tracking"
}
```

#### Updating Closing Lines

After logging recommendations, update with closing lines before game starts:

```python
from backend.betting.recommendation_manager import recommendation_manager

# Load odds snapshot before game starts
odds_snapshot = {...}  # From odds API

# Update all pending bets with closing lines
clv_results = recommendation_manager.update_closing_lines_from_odds_snapshot(odds_snapshot)
# Returns: {"bet_123": +5.0, "bet_124": -1.5, ...}
```

#### Updating Results

After game completes, update with actual results:

```python
# Load boxscores
boxscores = {...}

# Update all closed bets with results
results = recommendation_manager.update_results_from_boxscores(boxscores)
# Returns: {"bet_123": True, "bet_124": False, ...}
```

---

## Configuration

### Environment Variables

Configure thresholds in `.env`:

```env
# Betting Configuration
MIN_EDGE_THRESHOLD=0.05  # 5% minimum edge
MIN_EV_THRESHOLD=0.02    # 2% minimum EV
MIN_TRUST_SCORE=0.60     # 60% meta model confidence
MIN_GAMES_SAMPLED=5      # Minimum training samples
KELLY_FRACTION=0.25      # Quarter Kelly (conservative)
MAX_STAKE_PCT=0.05       # Max 5% of bankroll per bet
```

### Config Object

```python
from backend.config import settings

# Access configuration
settings.min_edge_threshold  # 0.05
settings.min_ev_threshold    # 0.02
settings.kelly_fraction      # 0.25
settings.max_stake_pct       # 0.05

# Check environment
settings.is_production       # bool
settings.allow_fallback_data # bool
```

---

## Complete Workflow Example

### 1. Find Value Props

```bash
# Get high-value single props
curl "http://localhost:8000/api/v1/props/value?min_edge=5.0&limit=20"
```

### 2. Get Parlay Suggestions

```bash
# Get correlation-aware parlays
curl "http://localhost:8000/api/v1/betting/parlays/suggestions?max_legs=4&min_parlay_ev=0.10"
```

### 3. Log Recommendations

```python
import requests

# For each recommendation (single or parlay leg)
recommendation = {
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
    "actually_bet": True,  # Set to True if you actually placed the bet
    "trust_score": 0.72,
    "games_sampled": 12
}

response = requests.post(
    "http://localhost:8000/api/v1/betting/recommendations",
    json=recommendation
)

bet_id = response.json()["bet_id"]
```

### 4. Update Closing Lines (Before Game)

```python
from backend.betting.recommendation_manager import recommendation_manager

# Fetch closing lines from odds API right before game starts
odds_snapshot = {
    "2024_12_KC_BUF": {
        "props": [
            {
                "player_name": "Patrick Mahomes",
                "prop_type": "player_pass_yds",
                "over_line": 280.5,  # Moved from 275.5
                "over_odds": -110
            }
        ]
    }
}

# Update CLV tracker
clv_results = recommendation_manager.update_closing_lines_from_odds_snapshot(odds_snapshot)
# clv_results: {"bet_123": +5.0}  (beat closing by 5 yards!)
```

### 5. Update Results (After Game)

```python
# After game completes
boxscores = {
    "2024_12_KC_BUF": {
        "players": [
            {
                "player_name": "Patrick Mahomes",
                "passing_yards": 318  # Actual result
            }
        ]
    }
}

results = recommendation_manager.update_results_from_boxscores(boxscores)
# results: {"bet_123": True}  (bet won: 318 > 275.5)
```

### 6. Check CLV Report

```bash
# View CLV performance
curl "http://localhost:8000/api/v1/betting/clv/report?last_n=50"
```

### 7. Train Meta Trust Model

```python
from backend.betting.recommendation_manager import recommendation_manager
from backend.betting.meta_trust_model import train_meta_trust_model

# Generate training data from bet history
training_data = recommendation_manager.generate_meta_training_data()

# Train meta trust model
result = train_meta_trust_model(
    bet_history_file='outputs/betting/clv_bets.json',
    output_dir='outputs/models/meta_trust'
)

# Future recommendations will now use improved trust scores
```

---

## Health & Data Availability

### `GET /health/data`

Check if required data is available:

```bash
curl "http://localhost:8000/health/data"
```

Response shows:
- Environment mode (dev/prod)
- Missing API keys
- Available predictions/odds files
- Configuration thresholds
- Setup recommendations

---

## Best Practices

### 1. Always Log Recommendations

Log ALL recommendations, not just bets you place:
```python
# Even if not betting
recommendation["actually_bet"] = False
requests.post("/api/v1/betting/recommendations", json=recommendation)
```

This builds historical data for calibration and meta trust model.

### 2. Update Closing Lines Consistently

Set up automated job to fetch closing lines:
```bash
# Cron job 5 minutes before game time
*/5 * * * * python -m backend.betting.update_closing_lines
```

### 3. Review CLV Weekly

```bash
# Weekly CLV check
curl "/api/v1/betting/clv/report?last_n=100"
```

Target: `avg_clv >= 1.0`, `positive_clv_rate >= 0.55`

### 4. Use Correlation-Aware Parlays Sparingly

- Same-game parlays have 25-35% correlation penalty
- Only bet when `adjusted_probability × combined_odds > 1.15` (15% EV)
- Prefer 2-3 leg parlays over 4+ legs

### 5. Respect Stake Limits

```python
# Never exceed max_stake_pct
stake = min(stake, settings.max_stake_pct * bankroll)
```

---

## Troubleshooting

### "No CLV data available"

**Cause:** Haven't logged any bets yet

**Solution:** Use `POST /api/v1/betting/recommendations` to log recommendations

### "No projections found"

**Cause:** Data pipeline hasn't run

**Solution:**
```bash
python -m backend.workflow.workflow_multi_prop_system
```

### "No odds available"

**Cause:** Missing ODDS_API_KEY or haven't fetched odds

**Solution:**
```bash
# Set API key in .env
ODDS_API_KEY=your_key_here

# Fetch odds
python -m backend.ingestion.fetch_prop_lines
```

---

## API Reference Summary

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health/data` | GET | Check data availability |
| `/api/v1/props/value` | GET | Find high-value single props |
| `/api/v1/betting/parlays/suggestions` | GET | Get correlation-aware parlays |
| `/api/v1/betting/clv/report` | GET | View CLV performance |
| `/api/v1/betting/recommendations` | POST | Log recommendation for tracking |

---

## Further Reading

- [PREDICTIVE_SIGNALS_GUIDE.md](./PREDICTIVE_SIGNALS_GUIDE.md) - Model features and signals
- [MULTI_PROP_SYSTEM.md](./MULTI_PROP_SYSTEM.md) - System architecture
- [backend/betting/portfolio_optimizer.py](../backend/betting/portfolio_optimizer.py) - Parlay optimizer implementation
- [backend/betting/clv_tracker.py](../backend/betting/clv_tracker.py) - CLV tracking implementation
