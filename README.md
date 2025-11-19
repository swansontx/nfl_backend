# NFL Props Prediction Backend

Comprehensive NFL player prop prediction system with data ingestion, feature engineering, machine learning models, and REST API.

## Features

### ✅ Data Ingestion
- **nflverse Integration**: Automated download of play-by-play, player stats, rosters, and schedules
- **Weather Data**: Real-time weather conditions for outdoor stadiums
- **Injury Reports**: Team injury reports with game-level indexing
- **Sportsbook Lines**: DraftKings odds integration (via The Odds API)
- **News Feeds**: ESPN news API integration

### ✅ Feature Engineering
- **Play-by-Play Features**: Extract player statistics from detailed game data
- **Smoothing**: Exponential Moving Average (EMA) for trend analysis
- **Rolling Windows**: Multi-week performance windows
- **Canonical Mapping**: Unified player/team ID system across data sources

### ✅ Team Pages & Boxscores
- **Team Database**: Complete metadata for all 32 NFL teams
- **Schedules**: Load and query team/week schedules
- **Boxscores**: ESPN-style boxscore generation from play-by-play data
- **Player Stats**: Comprehensive passing, rushing, receiving statistics

### ✅ API Service (FastAPI)
- **Player Stats**: Historical and current season statistics
- **Game Projections**: Player prop predictions with confidence scores
- **Insights Engine**: Trend analysis and betting angles
- **Team Endpoints**: Team info, schedules, news, and boxscores
- **Weather API**: Stadium weather conditions
- **Injury API**: Team injury reports
- **News API**: Latest team/player news headlines

### 🚧 Model Training (Scaffold)
- **Passing Model**: XGBoost/LightGBM for QB predictions
- **Rushing Model**: RB/QB rushing predictions (planned)
- **Receiving Model**: WR/TE receiving predictions (planned)

### 🚧 Backtest Framework (Scaffold)
- **Prediction Accuracy**: RMSE, MAE, R² metrics
- **Calibration Testing**: Probability calibration analysis
- **ROI Simulation**: Simulated betting performance
- **Position-Specific**: QB, RB, WR, TE accuracy tracking

### ✅ LLM Narrative Enhancement
- **Template-Based**: Data-driven narrative generation
- **LLM Integration**: OpenAI GPT-4 and Anthropic Claude support
- **Betting Angles**: Contrarian plays, weather factors, matchup analysis
- **Game Previews**: Automated game analysis and key matchups

### ✅ PostgreSQL Database (Documentation & Schema)
- **User Management**: Authentication, profiles, subscription tiers
- **Bet Tracking**: User bet history and outcomes
- **Bankroll Management**: ROI, win rate, performance tracking
- **Prediction Storage**: Historical predictions for backtesting
- **Line Caching**: Sportsbook odds caching
- **Watchlists**: Bet alerts and notifications

### ✅ Orchestration Pipeline
- **Full Workflow**: Coordinates all stages from ingestion to predictions
- **Stage Management**: Individual or full pipeline execution
- **Error Handling**: Robust error handling and logging

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/swansontx/nfl_backend.git
cd nfl_backend

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Run API Server

```bash
# Start FastAPI server
uvicorn backend.api.app:app --reload

# API available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Run Pipeline

```bash
# Full pipeline for 2024 season
python -m backend.orchestration.orchestrator --season 2024

# Individual stages
python -m backend.ingestion.fetch_nflverse --year 2024
python -m backend.features.extract_player_pbp_features --pbp inputs/play_by_play_2024.csv
python -m backend.features.smoothing_and_rolling --input outputs/player_pbp_features_by_id.json
```

## Project Structure

```
nfl_backend/
├── backend/
│   ├── api/                    # FastAPI REST API
│   │   ├── app.py             # Main API application
│   │   ├── team_database.py   # NFL team metadata
│   │   ├── schedule_loader.py # Schedule management
│   │   ├── boxscore_generator.py # Boxscore generation
│   │   ├── narrative_generator.py # LLM narratives
│   │   └── ...
│   ├── ingestion/             # Data ingestion scripts
│   │   ├── fetch_nflverse.py
│   │   ├── fetch_nflverse_schedules.py
│   │   ├── fetch_weather.py
│   │   └── ...
│   ├── features/              # Feature engineering
│   │   ├── extract_player_pbp_features.py
│   │   ├── smoothing_and_rolling.py
│   │   └── ...
│   ├── modeling/              # ML model training
│   │   ├── train_passing_model.py (scaffold)
│   │   └── ...
│   ├── calib_backtest/        # Backtesting framework
│   │   └── run_backtest.py (scaffold)
│   ├── roster_injury/         # Roster and injury indexing
│   │   ├── build_game_roster_index.py
│   │   └── build_injury_game_index.py
│   ├── orchestration/         # Pipeline orchestration
│   │   └── orchestrator.py
│   └── database/              # PostgreSQL integration
│       ├── __init__.py
│       ├── models.py          # SQLAlchemy models
│       └── crud.py            # Database operations
├── inputs/                    # Downloaded data
├── outputs/                   # Processed features and predictions
├── tests/                     # Test suite
├── DATABASE_SETUP.md          # Database documentation
├── requirements.txt
└── README.md
```

## Environment Variables

```env
# API Keys (optional)
ODDS_API_KEY=your_odds_api_key
OPENAI_API_KEY=your_openai_key  # For LLM narratives
ANTHROPIC_API_KEY=your_claude_key

# LLM Configuration
LLM_PROVIDER=openai  # or 'anthropic'
LLM_MODEL=gpt-4

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/nfl_props

# Cache Settings
CACHE_ENABLED=true
CACHE_TTL_INJURIES=900
CACHE_TTL_WEATHER=3600
```

## API Endpoints

### Players
- `GET /api/v1/players/{player_id}/stats` - Player statistics
- `GET /api/v1/players/search?name={name}` - Search players

### Teams
- `GET /api/v1/teams` - List all NFL teams
- `GET /api/v1/teams/{team_id}` - Team details
- `GET /api/v1/teams/{team_id}/schedule` - Team schedule
- `GET /api/v1/teams/{team_id}/news` - Team news

### Games
- `GET /api/v1/games/{game_id}/projections` - Game projections
- `GET /api/v1/games/{game_id}/insights` - Betting insights
- `GET /api/v1/games/{game_id}/boxscore` - Complete boxscore

### Utilities
- `GET /api/v1/weather/{stadium_id}` - Stadium weather
- `GET /api/v1/injuries/{team_id}` - Team injury report
- `GET /api/v1/news/{team_id}` - Team news feed

Full API documentation: http://localhost:8000/docs

## Database Setup

See [DATABASE_SETUP.md](DATABASE_SETUP.md) for complete PostgreSQL setup instructions, schema design, and migration guide.

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_api.py

# Run with coverage
pytest --cov=backend tests/
```

## Development Status

### ✅ Complete
- Data ingestion (nflverse, weather, injuries, news)
- Feature extraction and engineering
- Team pages and boxscores
- REST API with comprehensive endpoints
- Orchestration pipeline
- LLM narrative enhancement
- Database schema and documentation

### 🚧 In Progress
- Model training implementation
- Backtest framework implementation
- Frontend dashboard (separate repo)

### 📋 Planned
- Real-time game tracking
- Advanced analytics dashboard
- Multi-model ensemble predictions
- Automated bet placement integration

## Contributing

This is a private project. For questions or collaboration:
- Create an issue
- Submit a pull request
- Contact: swansontx

## License

Private - All Rights Reserved
