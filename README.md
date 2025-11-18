# NFL Props Backend

> **Production-ready backend system for NFL player prop projections, modeling, and opportunity identification.**

Generates probabilistic projections for NFL player props across multiple markets, identifies +EV betting opportunities, and provides a REST API for accessing predictions.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0+-orange.svg)](https://www.sqlalchemy.org/)

---

## 🎯 Features

### Core Capabilities
- **Multi-Market Projections**: Player receiving yards, rushing yards, TDs, receptions, attempts
- **Probabilistic Modeling**: Poisson/Negative Binomial for counts, lognormal for yards, logistic for TDs
- **Injury Redistribution**: Automatically redistributes volume when players are inactive
- **Probability Calibration**: Platt scaling and isotonic regression for accurate probabilities
- **Opportunity Scoring**: Identifies best bets using edge, confidence, volatility, and usage
- **Tier Classification**: Core (N=8), Mid (N=12), Lotto (N=5) opportunity categories
- **Comprehensive Backtesting**: Historical validation with ROI, Sharpe, Brier score, log loss
- **REST API**: FastAPI endpoints for projections, admin recompute, job status

### Technical Features
- **Player Name Canonicalization**: 5-step fuzzy matching pipeline (95%+ accuracy)
- **Event→Game Mapping**: Confidence-scored mapping of odds API events to games
- **Roster/Injury Tracking**: Redis-cached status with confidence multipliers (Q/D/P)
- **Feature Engineering**: PBP-derived features with empirical Bayes smoothing
- **Database Persistence**: PostgreSQL for projections, outcomes, calibration params
- **Structured Logging**: JSON logs with request IDs for full observability
- **Background Jobs**: Async projection generation with job tracking

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 14+
- Redis 6+

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/nfl_backend
cd nfl_backend

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your database credentials and API keys
```

### Database Setup

```bash
# Start PostgreSQL and Redis (via Docker)
docker-compose up -d

# Initialize database schema
python -c "from backend.database import init_db; init_db()"
```

### Run API

```bash
# Development
uvicorn backend.api.app:app --reload

# Production
uvicorn backend.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

Visit http://localhost:8000/docs for interactive API documentation.

---

## 📊 System Components

### 1. **Canonicalization**
Player name matching with 5-step pipeline: exact match → team-scoped → token-based → nickname expansion → fuzzy matching. Achieves 95%+ match rate.

### 2. **Feature Engineering**
Extracts PBP features: snaps, routes, targets, carries, redzone usage. Smoothed with empirical Bayes and exponential rolling averages.

### 3. **Modeling**
- **Counts**: Poisson/Negative Binomial
- **Yards**: Lognormal (captures right-skew)
- **TDs**: Logistic/XGBoost

### 4. **Injury Redistribution**
Redistributes volume from OUT/IR/DEV players to active teammates using proportional, redzone-weighted, or QB-specific algorithms.

### 5. **Scoring & Ranking**
Scores opportunities using: `(edge × prob × conf) / (1 + k×vol) + λ×usage`. Assigns Core/Mid/Lotto tiers.

### 6. **Calibration**
Platt scaling or isotonic regression per market. Cross-validated to prevent overfitting.

### 7. **Backtesting**
Validates performance with Brier score, log loss, ROC AUC, ROI, Sharpe ratio, max drawdown. Kelly criterion betting simulation.

### 8. **API**
FastAPI endpoints for projections, recompute, job status. Bearer token auth for admin endpoints.

---

## 📖 Usage Examples

### Generate Projections

```python
from backend.models.prop_models import PropModelRunner

runner = PropModelRunner()
projections = runner.generate_projections(
    game_id="2024_10_BUF_KC",
    markets=["player_rec_yds", "player_anytime_td"],
    save_to_db=True
)
```

### Query via API

```bash
# Get projections for a game
curl "http://localhost:8000/projections/games/2024_10_BUF_KC?market=player_rec_yds&tier=core"

# Trigger recompute (requires admin token)
curl -X POST http://localhost:8000/admin/recompute \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"game_id": "2024_10_BUF_KC"}'
```

### Run Backtest

```python
from backend.backtest import BacktestEngine
from datetime import datetime

engine = BacktestEngine()
result = engine.run_backtest(
    start_date=datetime(2024, 9, 1),
    end_date=datetime(2025, 1, 1),
    simulate_betting=True
)
print(engine.generate_backtest_report(result))
```

---

## 🎲 Supported Markets

| Market                 | Distribution | Description              |
|------------------------|--------------|--------------------------|
| `player_rec_yds`       | Lognormal    | Receiving yards          |
| `player_rush_yds`      | Lognormal    | Rushing yards            |
| `player_pass_yds`      | Lognormal    | Passing yards            |
| `player_receptions`    | Poisson      | Number of receptions     |
| `player_rush_attempts` | Poisson      | Number of rush attempts  |
| `player_pass_tds`      | Poisson      | Passing touchdowns       |
| `player_anytime_td`    | Logistic     | Anytime TD probability   |

---

## 🗂️ Project Structure

```
backend/
├── api/              # FastAPI application
├── backtest/         # Backtesting framework
├── calibration/      # Probability calibration
├── canonical/        # Player/game canonicalization
├── config/           # Configuration & logging
├── database/         # SQLAlchemy models & session
├── features/         # Feature extraction & smoothing
├── models/           # Prop distribution models
├── redistribution/   # Injury volume redistribution
├── roster_injury/    # Roster & injury tracking
├── scoring/          # Scoring & tier assignment
└── utils/            # Date & validation utilities
```

---

## 🔧 Configuration

Key settings in `.env`:

```bash
# Database
POSTGRES_HOST=localhost
POSTGRES_PASSWORD=your_password

# Redis
REDIS_HOST=localhost

# API Keys
ODDS_API_KEY=your_api_key

# Model Parameters
LOOKBACK_GAMES=8
MIN_EDGE_THRESHOLD=0.03
CORE_PICKS_COUNT=8

# Admin
ADMIN_TOKEN=your_secret_token
```

---

## 📈 Performance Metrics

The system tracks:
- **Calibration**: Brier score, log loss, ROC AUC
- **Value**: MAE, RMSE, R², direction accuracy
- **Betting**: ROI, Sharpe ratio, max drawdown, hit rate
- **CLV**: Closing Line Value vs market

---

## 🧪 Testing

```bash
pytest                              # Run all tests
pytest --cov=backend                # With coverage
pytest tests/integration/           # Integration tests only
```

---

## 📦 Deployment

```bash
# Docker Compose (recommended)
docker-compose up -d

# Manual
uvicorn backend.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 📚 API Documentation

Interactive docs: http://localhost:8000/docs

**Key Endpoints**:
- `GET /health` - Health check
- `GET /projections/games/{game_id}` - Game projections
- `POST /admin/recompute` - Trigger projection job
- `GET /admin/jobs/{job_id}` - Job status

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Open a Pull Request

---

## 📄 License

MIT License

---

**Built with ⚡ for NFL props analysis**
