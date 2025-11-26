# NFL Backend System Map

**Purpose**: Central reference for all systems, models, APIs, and features.
**Last Updated**: 2025-11-26

---

## 📋 Table of Contents

1. [ML Models & Predictors](#ml-models--predictors)
2. [Backtesting Systems](#backtesting-systems)
3. [API Endpoints](#api-endpoints)
4. [Configuration & Weights](#configuration--weights)
5. [Orchestrators](#orchestrators)
6. [Analyzers & Engines](#analyzers--engines)
7. [Data Sources](#data-sources)
8. [Validation Status](#validation-status)

---

## 🤖 ML Models & Predictors

### Production Models (LIVE)

| Model | Location | Status | Accuracy | Features | Use Case |
|-------|----------|--------|----------|----------|----------|
| **Neural Network** | `models/game_totals_ml/neural_network.pkl` | ✅ **LIVE** | **10.93 MAE** (+10.8% vs baseline) | 26 features | Game totals prediction |
| Random Forest | `models/game_totals_ml/random_forest.pkl` | ⚠️ Trained but not used | 11.02 MAE (+10.0%) | 26 features | Alternative totals model |
| Gradient Boosting | `models/game_totals_ml/gradient_boosting_tuned.pkl` | ⚠️ Trained but not used | 11.22 MAE (+8.4%) | 26 features | Alternative totals model |

### ML Production Code

| File | Purpose | Status | Integrated? |
|------|---------|--------|-------------|
| `backend/ml/game_totals_predictor.py` | Production predictor using Neural Network | ✅ Working | ✅ Yes (via orchestrator) |
| `backend/ml/train_game_totals_models.py` | Training pipeline for game totals models | ✅ Working | N/A (training only) |

### Model Features (26 total)

**Recent Form (6)**:
- `home_recent_ppg`, `away_recent_ppg` (last 4 games)
- `home_l3_ppg`, `away_l3_ppg` (last 3 games)
- `home_l3_margin`, `away_l3_margin`

**Season Stats (4)**:
- `home_season_ppg`, `away_season_ppg`
- `home_def_ppg_allowed`, `away_def_ppg_allowed`

**Defense Matchups (2)**:
- `home_off_vs_away_def`, `away_off_vs_home_def`

**Momentum & Volatility (4)**:
- `home_trend`, `away_trend` (PPG change L3 vs season)
- `home_std`, `away_std` (scoring volatility)

**Rest & Schedule (3)**:
- `rest_differential` (home rest - away rest)
- `home_off_bye`, `away_off_bye`

**Weather (4)**:
- `temperature`, `wind_speed`
- `is_cold` (<32°F), `is_windy` (>15 mph)

**Situational (3)**:
- `is_primetime`, `is_division_game`, `week`

---

## 🧪 Backtesting Systems

### Validated Backtests ✅

| Backtest | File | Sample Size | Accuracy | Notes |
|----------|------|-------------|----------|-------|
| **Injury Impact** | `injury_impact_backtest.py` | 2,418 obs | 50-83% confidence | Different injury severities |
| **Weather (Wind)** | `weather_impact_backtest.py` | 43 obs | 80% confidence | Wind >15mph only |
| **ML Game Totals** | `ml_comprehensive_backtest.py` | 662 games | NN: 10.93 MAE | Neural Network validated |

### Experimental/Needs Work ⚠️

| Backtest | File | Status | Issue |
|----------|------|--------|-------|
| Defense Matchup | `defense_matchup_backtest.py` | ❌ **BROKEN** | Returns 0 observations (filtering too strict) |
| Situational Factors | `situational_factors_backtest.py` | ⚠️ Recently fixed | Was using fantasy points instead of real scores |
| EPA Enhanced | `epa_enhanced_backtest.py` | ⚠️ No improvement | EPA provides 0% improvement (redundant with recent PPG) |

### Other Backtests

| Backtest | File | Purpose |
|----------|------|---------|
| Player Props | `player_props_backtest.py` | Validates player projections |
| Overall Accuracy | `overall_accuracy_backtest.py` | System-wide accuracy metrics |
| Enhanced Totals | `enhanced_totals_backtest.py` | Legacy totals validation |

### Backtesting Framework

**Core**: `backend/backtesting/framework.py`
- Loads historical data (2021-2023 seasons)
- Provides game data, player stats, weather, injuries
- Used by all backtesting scripts

---

## 🌐 API Endpoints

### Game Predictions & Markets

| Endpoint | Method | Purpose | ML Model? |
|----------|--------|---------|-----------|
| `/api/v1/betting/game-markets/{game_id}` | GET | **Complete game analysis** (spreads, totals, moneylines) | ✅ **Neural Network** |
| `/api/v1/betting/game-markets/week/{week}` | GET | All games for a week | ✅ **Neural Network** |
| `/api/v1/games/{game_id}` | GET | Game details | No |
| `/api/v1/games` | GET | All games | No |

### Player Projections

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/game/{game_id}/projections` | GET | Player projections for game |
| `/api/v1/players/{player_id}/gamelogs` | GET | Player game logs |
| `/api/v1/games/{game_id}/prop-sheet` | GET | All props for game |

### Analysis & Insights

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/game/{game_id}/situation` | GET | Situational analysis |
| `/game/{game_id}/evaluate` | GET | Game evaluation |
| `/api/v1/games/{game_id}/insights` | GET | Matchup insights |
| `/api/v1/games/{game_id}/narrative` | GET | Game narratives |
| `/api/v1/games/{game_id}/weather` | GET | Weather data |
| `/api/v1/games/{game_id}/injuries` | GET | Injury reports |
| `/api/v1/games/{game_id}/injury-impact` | GET | Injury impact analysis |
| `/api/v1/games/{game_id}/boxscore` | GET | Game boxscore |

### Trends & Opportunities

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/betting/hot-movers` | GET | Props with line movement |
| `/api/v1/betting/opportunities` | GET | Betting opportunities |

---

## ⚙️ Configuration & Weights

| File | Status | Purpose |
|------|--------|---------|
| `backend/config/validated_weights.py` | ⚠️ **Mixed** | Contains BOTH validated and unvalidated configs |
| `backend/config/validated_weights_clean.py` | ✅ Clean | ONLY validated data (injury + wind) |

### Validated Sections ✅

1. **Injury Redistribution** (2,418 observations)
   - OUT: 50% confidence
   - Doubtful: 60% confidence
   - Questionable: 75% confidence
   - IR: 83% confidence

2. **Weather Impact** (43 observations)
   - Wind >15mph: -0.5 passing yards per attempt
   - 80% confidence

### Unvalidated/Broken Sections ❌

1. **Defense Matchups** - Backtest returns 0 results
2. **Situational Factors** - Recently fixed but needs revalidation
3. **Trend Weights** - No backtest exists
4. **Feature Weights** - Proven not to help

---

## 🎯 Orchestrators

### Game Outcome Orchestrator

**File**: `backend/orchestration/game_outcome_orchestrator.py`
**Status**: ✅ **LIVE with ML**

**What it does**:
- Predicts game outcomes (spreads, totals, win probabilities)
- Uses Neural Network for totals (+10.8% accuracy)
- Formula-based for spreads (for now)
- Collects 76+ features (most broken, but ML doesn't need them)

**How to use**:
```python
from backend.orchestration.game_outcome_orchestrator import game_outcome_orchestrator

prediction = game_outcome_orchestrator.predict_game(
    game_id="2025_12_BUF_KC",
    week=12
)

print(f"Predicted total: {prediction.predicted_total}")
```

**Integrated with**:
- `/api/v1/betting/game-markets/{game_id}` endpoint
- Loads Neural Network automatically

### Public Betting Orchestrator

**File**: `backend/orchestration/public_betting_orchestrator.py`
**Status**: ⚠️ Needs API key

**What it does**:
- Fetches public betting percentages
- Identifies sharp money vs public money
- Contrarian opportunities

### Picks Pipeline

**File**: `backend/orchestration/picks_pipeline.py`
**Status**: ⚠️ Unknown (needs investigation)

---

## 🔍 Analyzers & Engines

| File | Purpose | Used By |
|------|---------|---------|
| `backend/api/prediction_engine.py` | Player projection predictions | `/game/{game_id}/projections` |
| `backend/api/insights_engine.py` | Matchup insights generation | `/api/v1/games/{game_id}/insights` |
| `backend/api/narrative_generator.py` | Game narrative generation | `/api/v1/games/{game_id}/narrative` |
| `backend/api/injury_impact_analyzer.py` | Injury impact analysis | `/api/v1/games/{game_id}/injury-impact` |
| `backend/api/defense_analyzer.py` | Defense matchup analysis | Various |
| `backend/api/matchup_analyzer.py` | General matchup analysis | Various |
| `backend/api/situational_analyzer.py` | Situational factors | `/game/{game_id}/situation` |
| `backend/api/prop_analyzer.py` | Prop bet analysis | `/api/v1/games/{game_id}/prop-sheet` |
| `backend/analysis/game_markets.py` | Game market analysis | `/api/v1/betting/game-markets` |

---

## 💾 Data Sources

### Historical Data

| Source | Location | Coverage | Size |
|--------|----------|----------|------|
| Play-by-play data | `data/play_by_play/` | 2021-2023 | 551 MB |
| Games data | Loaded via framework | 2021-2023 | N/A |
| Player stats | Loaded via framework | 2021-2023 | N/A |
| Weather data | Loaded via framework | 2021-2023 | N/A |

### Live Data APIs

| API | File | Purpose | Key Required? |
|-----|------|---------|---------------|
| Odds API | `backend/ingestion/fetch_odds.py` | Current betting lines | Yes (ODDS_API_KEY) |
| Public Betting | Various | Public betting percentages | Yes (needs bs4) |

---

## ✅ Validation Status

### Summary Table

| Category | Validated | Broken | Theoretical | Notes |
|----------|-----------|--------|-------------|-------|
| **ML Models** | 1/3 | 0/3 | 2/3 | Neural Network in production, others trained but unused |
| **Backtests** | 3/9 | 1/9 | 5/9 | Injury, Weather, ML validated; Defense broken |
| **Config Sections** | 2/6 | 2/6 | 2/6 | Only injury + wind validated |
| **API Endpoints** | ~20 | 0 | 0 | All working (some need API keys) |

### Action Items

**High Priority** 🔴:
1. Fix or remove `defense_matchup_backtest.py` (returns 0 results)
2. Revalidate `situational_factors_backtest.py` after bug fix
3. Document picks_pipeline.py status

**Medium Priority** 🟡:
1. Consider training ML model for spreads (currently formula-based)
2. Clean up theoretical weights from `validated_weights.py`
3. Add more weather scenarios (cold, precipitation)

**Low Priority** 🟢:
1. Investigate if Random Forest or Gradient Boosting should be used
2. Add more features to Neural Network
3. Extend training data to 2024 season

---

## 🚀 Recent Improvements

**2025-11-26**:
- ✅ Integrated Neural Network into production (+10.8% accuracy)
- ✅ Created comprehensive ML training pipeline
- ✅ Fixed situational factors backtest (was using fantasy points bug)
- ✅ Created validation audit identifying broken systems
- ✅ Created clean config file with only validated data

---

## 📖 How to Use This Map

**Before building something new**:
1. Search this document for similar functionality
2. Check if a backtest/analyzer/model already exists
3. Check validation status before using in production

**When adding new features**:
1. Add entry to appropriate section
2. Mark validation status
3. Document integration points
4. Update "Recent Improvements"

**When finding broken systems**:
1. Update validation status to ❌
2. Add to "Action Items"
3. Document the issue

---

## 📞 Quick Reference

**Need game totals prediction?**
- Use: `game_outcome_orchestrator.predict_game()`
- Model: Neural Network (10.93 MAE)
- API: `/api/v1/betting/game-markets/{game_id}`

**Need player projection?**
- Use: `prediction_engine`
- API: `/game/{game_id}/projections`

**Need to validate a new feature?**
- Use: `backend/backtesting/framework.py`
- Reference: Other `*_backtest.py` files
- Seasons: 2021-2023

**Need to check what's validated?**
- See: `backend/config/validated_weights_clean.py`
- Or: Validation Status section above
