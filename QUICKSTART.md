# NFL Backend - Quick Start Guide

Complete guide to get the NFL prop prediction system up and running.

## Prerequisites

```bash
# Python 3.9+
python --version

# Install dependencies
pip install -r requirements.txt
```

## Step 1: Ingest Data

Download nflverse data (play-by-play + advanced metrics) for multiple seasons:

```bash
# Option A: Use the master workflow (recommended)
python workflow_train_and_backtest.py --seasons 2023 2024

# Option B: Manual ingestion
python -m backend.ingestion.fetch_nflverse --year 2023 --output inputs/
python -m backend.ingestion.fetch_nflverse --year 2024 --output inputs/
```

This downloads:
- **Play-by-play data** with 300+ columns (EPA, CPOE, success rate, WPA, air yards, YAC, QB pressure)
- **Player stats** (weekly aggregates)
- **Rosters** (player positions and team assignments)
- **Next Gen Stats** (player tracking data)
- **Snap counts** (participation percentages)
- **PFR advanced stats** (Pro Football Reference metrics)
- **Depth charts** (positional rankings)

**Output**: `inputs/2023_play_by_play.parquet`, `inputs/2024_play_by_play.parquet`, etc.

---

## Step 2: Extract Features

Extract player-level features from play-by-play data:

```bash
# Option A: Use workflow (recommended)
python workflow_train_and_backtest.py --seasons 2023 2024

# Option B: Manual extraction
python -m backend.features.extract_player_pbp_features \
    --pbp-file inputs/2023_play_by_play.parquet \
    --roster-file inputs/2023_weekly_rosters.parquet \
    --output outputs/features/2023_player_features.json
```

**Features extracted per player per game**:
- **Basic stats**: passing_yards, passing_tds, completions, attempts, interceptions
- **EPA metrics**: total_epa, qb_epa, rushing_epa, receiving_epa
- **CPOE**: cpoe_sum, cpoe_count (completion % over expected)
- **Success rate**: success_plays / total_plays
- **WPA**: win_probability_added
- **Air yards & YAC**: air_epa, yac_epa, xyac_sum, xyac_count
- **QB pressure**: qb_hits, qb_hurries, qb_pressures

**Output**: `outputs/features/2023_player_features.json`

---

## Step 3: Train Models

Train XGBoost/LightGBM model to predict passing yards using advanced metrics:

```bash
# Option A: Use workflow (recommended)
python workflow_train_and_backtest.py --seasons 2023 2024

# Option B: Manual training
python -m backend.modeling.train_passing_model \
    --season 2024 \
    --features-file outputs/features/2024_player_features.json \
    --output outputs/models/passing_model_2024.pkl \
    --model-type xgboost
```

**Model features**:
- **Primary signals** (30% weight): EPA avg, CPOE avg, success rate, WPA avg
- **Pressure metrics** (10% weight): QB pressure rate, hit rate
- **Rolling windows** (30% weight): 3-game and 5-game rolling averages
- **Traditional context** (20% weight): yards/attempt, completion %, air yards
- **Game context** (10% weight): games played, attempts

**Training approach**:
- Time-based split: Train on early weeks, validate on mid weeks, test on late weeks
- XGBoost with 200 estimators, max_depth=6, learning_rate=0.05
- Early stopping on validation set to prevent overfitting

**Output**:
- `outputs/models/passing_model_2024.pkl` (trained model)
- `outputs/models/passing_model_2024_metrics.json` (training metrics)

---

## Step 4: Run Backtest

Validate model accuracy on test data:

```bash
# Option A: Use workflow (recommended)
python workflow_train_and_backtest.py --seasons 2023 2024

# Option B: Manual backtest
python -m backend.calib_backtest.run_backtest \
    --season 2024 \
    --model-path outputs/models/passing_model_2024.pkl \
    --features-file outputs/features/2024_player_features.json \
    --actuals-file outputs/features/2024_player_features.json \
    --output outputs/backtest/backtest_report_2024.json
```

**Backtest analysis**:
- **Accuracy metrics**: RMSE, MAE, R², MAPE
- **Calibration**: Predicted vs actual bins
- **Weekly breakdown**: Error analysis by week
- **Position-specific**: Metrics per position

**Output**: `outputs/backtest/backtest_report_2024.json`

---

## Complete Workflow (Recommended)

Run the entire pipeline in one command:

```bash
# Train and backtest on 2024 data
python workflow_train_and_backtest.py --seasons 2024

# Multi-season training (train on 2023, backtest on 2024)
python workflow_train_and_backtest.py --seasons 2023 2024

# Skip data ingestion if already downloaded
python workflow_train_and_backtest.py --seasons 2024 --skip-ingestion

# Train only (skip backtest)
python workflow_train_and_backtest.py --seasons 2024 --train-only

# Backtest only (use existing model)
python workflow_train_and_backtest.py --seasons 2024 --backtest-only \
    --model-path outputs/models/passing_model_2024.pkl
```

---

## Optional: Injury Data

Fetch current injury reports (for adjusting predictions):

```bash
python -m backend.ingestion.fetch_injuries \
    --output inputs/injuries/ \
    --week 14
```

**Output**: `inputs/injuries/injuries_week_14.json`

Uses ESPN API (free, no API key required).

---

## Optional: Prop Betting Lines (Future)

For betting ROI analysis, fetch prop lines from The Odds API:

```bash
# Requires ODDS_API_KEY environment variable
export ODDS_API_KEY="your_api_key_here"

python -m backend.ingestion.fetch_prop_lines \
    --output outputs/prop_lines/ \
    --week 14
```

**Note**: Not required for initial model training/validation. Only needed for betting ROI calculation.

---

## API Server

Start the FastAPI server to serve predictions:

```bash
cd backend/api
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Available endpoints**:
- `GET /api/v1/teams` - List all NFL teams
- `GET /api/v1/teams/{team_id}/stats` - Team statistics
- `GET /api/v1/games` - List games (filterable by week, season, team)
- `GET /api/v1/games/{game_id}` - Game details
- `GET /api/v1/games/{game_id}/insights` - Matchup insights
- `GET /api/v1/players` - Search players
- `GET /api/v1/players/{player_id}` - Player details
- `GET /api/v1/players/{player_id}/stats` - Player statistics
- `GET /api/v1/players/{player_id}/gamelogs` - Player game logs

---

## Expected Results

After running the complete workflow on 2024 data, you should see:

**Training metrics**:
- Validation RMSE: ~25-35 yards
- Validation MAE: ~18-25 yards
- Validation R²: 0.40-0.55

**Backtest metrics**:
- Test RMSE: ~28-38 yards
- Test MAE: ~20-28 yards
- Test R²: 0.35-0.50

**Interpretation**:
- RMSE ~30 means predictions are typically within 30 yards of actual
- R² ~0.45 means model explains 45% of variance (good for NFL props)
- These metrics validate the model can accurately predict passing yards using advanced metrics

---

## Directory Structure After Workflow

```
nfl_backend/
├── inputs/                              # Raw data from nflverse
│   ├── 2023_play_by_play.parquet
│   ├── 2024_play_by_play.parquet
│   ├── 2023_player_stats.parquet
│   ├── 2023_weekly_rosters.parquet
│   ├── injuries/                        # Optional injury data
│   └── ...
├── outputs/
│   ├── features/                        # Extracted player features
│   │   ├── 2023_player_features.json
│   │   └── 2024_player_features.json
│   ├── models/                          # Trained models
│   │   ├── passing_model_2024.pkl
│   │   └── passing_model_2024_metrics.json
│   ├── backtest/                        # Backtest reports
│   │   └── backtest_report_2024.json
│   ├── prop_lines/                      # Optional prop lines
│   └── workflow_summary.json            # Overall workflow results
├── cache/                               # nflverse download cache
└── workflow.log                         # Execution log
```

---

## Next Steps

1. **Review backtest results**: Check `outputs/backtest/backtest_report_2024.json`
2. **Analyze feature importance**: See which advanced metrics (EPA, CPOE, success rate) matter most
3. **Build matchup analyzer**: Use trained model with matchup edges for game-specific predictions
4. **Add contextual splits**: Analyze performance vs elite defenses, high pressure situations
5. **Integrate with frontend**: Connect API endpoints to React frontend

---

## Troubleshooting

**Issue**: `FileNotFoundError: inputs/2024_play_by_play.parquet`
- **Fix**: Run data ingestion first: `python -m backend.ingestion.fetch_nflverse --year 2024`

**Issue**: `ModuleNotFoundError: No module named 'nfl_data_py'`
- **Fix**: Install dependencies: `pip install -r requirements.txt`

**Issue**: Training takes too long
- **Fix**: Use fewer seasons or reduce model complexity: `--model-type lightgbm`

**Issue**: Low R² score (<0.30)
- **Check**: Are you using time-based splits? (Don't shuffle data)
- **Check**: Are advanced metrics being extracted? (EPA, CPOE, success rate)
- **Check**: Is sample size sufficient? (Need at least 200+ player-games)

---

## Resources

- **Advanced Metrics Guide**: `docs/ADVANCED_METRICS.md`
- **Matchup Analysis Guide**: `docs/MATCHUP_ANALYSIS.md`
- **API Documentation**: `docs/API_ENDPOINTS.md`
- **nflverse Data**: https://github.com/nflverse/nflverse-data
