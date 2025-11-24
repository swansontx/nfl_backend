# NFL Backend - Hardcoded Values & Unimplemented Features

**Last Updated:** 2025-11-24

Track progress by marking items as:
- [ ] Not started
- [x] Completed
- [~] In progress / Partial

---

## Priority 1: Core Infrastructure (CRITICAL)

### backend/orchestration/orchestrator.py
- [x] Line 72: `def __init__(self, season: int = 2025, week: Optional[int] = None)`
  - **FIXED:** Now uses get_current_season_and_week()

### backend/modeling/generate_projections.py
- [x] Line 184: `def generate_for_week(self, week: int, season: int = 2025)`
  - **FIXED:** Now uses get_current_nfl_season()

### backend/tools/quick_picks.py
- [x] Line 14: `def load_projections(week: int, season: int = 2025)`
  - **FIXED:** Now uses get_current_nfl_season()

### backend/canonical/map_event_to_game.py
- [x] Line 17: `def map_event_to_game(event_json: dict, season: int = 2025)`
  - **FIXED:** Now uses get_current_nfl_season()

### backend/canonical/player_map.py
- [x] Line 29: `def load_player_lookup(year: int = 2025, lookup_dir: Path = Path('inputs'))`
  - **FIXED:** Now uses get_current_nfl_season()
- [x] Line 36-59: `TODO: Implement loading from inputs/player_lookup_YYYY.json`
  - **FIXED:** Implemented JSON loading with name variation mapping
- [x] Line 75-89: `TODO: Implement fuzzy matching logic`
  - **FIXED:** Implemented fuzzy matching with fuzzywuzzy (optional), team/position filtering

---

## Priority 2: Analysis Scripts (MEDIUM - Used for Training)

### backend/analysis/backtest_week.py
- [x] Line 78: `def backtest_week(week: int, season: int = 2025)`
  - **FIXED:** Now uses get_current_nfl_season()

### backend/analysis/backtest_with_models.py
- [x] Line 98: `def backtest_with_models(week: int, season: int = 2025)`
  - **FIXED:** Now uses get_current_nfl_season()

### backend/analysis/create_enhanced_player_stats.py
- [ ] Lines 19, 25, 38, 47-49, 57: Multiple hardcoded `season == 2025` filters
  - SKIP: One-time data processing script, intentionally hardcoded

### backend/analysis/train_baseline_model.py
- [ ] Line 168: Hardcoded `season == 2025`
  - SKIP: Specific week 10 training script, intentionally hardcoded

### backend/analysis/train_scoring_props.py
- [x] Line 39: Hardcoded `season == 2025`
  - **FIXED:** Now uses get_current_nfl_season()

### backend/analysis/train_game_derivative_markets.py
- [x] Line 38: Hardcoded `season == 2025`
  - **FIXED:** Now uses get_current_nfl_season()

### backend/analysis/generate_quarter_scores.py
- [x] Line 240: Hardcoded `season == 2025`
  - **FIXED:** Now uses get_current_nfl_season()

### backend/analysis/generate_synthetic_player_stats.py
- [x] DELETED: Synthetic data generator removed

### backend/analysis/fetch_2025_training_data.py
- [ ] Line 34: Hardcoded `season == 2025`
  - SKIP: File specifically for 2025 data fetching

### backend/analysis/aggregate_real_player_stats_from_pbp.py
- [ ] Line 28: `season = 2025  # We know this is 2025 data`
  - SKIP: Intentional for specific data processing

### backend/analysis/backtest_props_nov9.py
- [ ] Line 20: Hardcoded `season == 2025`
  - SKIP: Point-in-time backtest (Nov 9)

### backend/analysis/backtest_real_nov9.py
- [ ] Line 33: Hardcoded `season == 2025`
  - SKIP: Point-in-time backtest (Nov 9)

### backend/analysis/analyze_misses_nov9.py
- [ ] Line 22: Hardcoded `season == 2025`
  - SKIP: Point-in-time analysis (Nov 9)

---

## Priority 3: Unimplemented Features (TODO/FIXME)

### backend/calib_backtest/calibrate.py
- [x] Line 47-57: `TODO: Implement using sklearn` - Calibration fitting not implemented
  - **FIXED:** Implemented Platt scaling (LogisticRegression) and Isotonic regression
- [x] Line 80: `TODO: Apply calibration` - Not implemented
  - **FIXED:** Implemented transform method with both methods
- [x] Line 90-91: `TODO: Save calibrator` - Not implemented
  - **FIXED:** Implemented with joblib serialization
- [x] Line 99-100: `TODO: Load calibrator` - Not implemented
  - **FIXED:** Implemented with joblib deserialization
- [ ] Line 121: `TODO: Load historical data`
  - Remains: run_calibration function data loading

### backend/calib_backtest/backtest.py
- [x] Line 42-44: `TODO: Implement using sklearn.metrics` - Calibration error not implemented
  - **FIXED:** Implemented classification metrics (accuracy, precision, recall, F1, ROC-AUC, Brier)
- [x] Line 68-70: `TODO: Implement using sklearn.metrics or numpy` - Precision not implemented
  - **FIXED:** Implemented regression metrics (MAE, RMSE, R2, MAPE)
- [x] Line 136: `TODO: Load predictions and actuals`
  - **FIXED:** Implemented JSON file loading

### backend/features/extract_weather_features.py
- [x] Line 45: `TODO: Implement actual API call` - Weather API not connected
  - **FIXED:** Implemented OpenWeather API integration with error handling

### backend/features/hfa_impact_analysis.py
- [x] Line 39: `TODO: Replace with actual analysis from your data`
  - **FIXED:** Added calculate_hfa_impacts_from_data() method using pandas

### backend/api/external_apis.py
- [x] Line 75: `TODO: Match forecast to game_time`
  - **FIXED:** Added _find_closest_forecast() method
- [x] Line 86: `TODO: Determine from stadium data` - is_dome always False
  - **FIXED:** Now set by get_weather_for_game() from stadium database

### backend/ingestion/fetch_odds.py
- [x] Line 34-41: `TODO: Implement actual API call using requests`
  - **FIXED:** Implemented OddsAPI integration with quota tracking

### backend/betting/meta_trust_model.py
- [x] Line 114: `TODO: infer from bet history` - Player role
  - **FIXED:** Updated comment, inferred from prop type
- [x] Line 348: `TODO: add if available` - Spread
  - **FIXED:** Now extracts spread, total, dome, weather from bet record

### backend/modeling/train_usage_efficiency_models.py
- [ ] Line 413: `TODO: Need team totals to calculate actual shares`

---

## Priority 4: API Endpoints (LOW - Some have TODOs)

### backend/api/app.py
- [ ] Line 55: `TODO: Restrict in production` - CORS origins allow all
- [ ] Line 214: `TODO: Integrate with orchestration pipeline`
- [ ] Line 754: `TODO: Integrate with actual news API or RSS feeds`
- [ ] Line 781: `TODO: Add non-injury news items`
- [ ] Line 984: `TODO: Integrate with ML models and feature analysis`
- [ ] Line 1019: `TODO: Get stadium location from game_id/schedule`
- [ ] Line 1041: `TODO: Integrate with LLM (OpenAI, Claude, etc.)`
- [ ] Line 1105: `TODO: Integrate with content APIs`
- [ ] Line 1301: `TODO: Load actual player data from database/files`
- [ ] Line 1364: `TODO: Load actual player projections`
- [ ] Line 1753-1755: Multiple TODOs for game analysis
- [ ] Line 2058: `TODO: Load actual team stats from database`
- [ ] Line 2210: `TODO: Add betting lines when available`
- [ ] Line 2236: `TODO: Load from player lookup JSON or database`

### Remaining endpoints in app.py that need default resolution:
- [ ] Line 1618: `season: int = None` - needs resolution
- [ ] Line 2048: `get_team_stats` - needs resolution
- [ ] Line 2101: season parameter - needs resolution
- [ ] Line 2278: `get_player_stats` - needs resolution
- [ ] Line 2584: `get_player_gamelogs` - needs resolution

---

## Priority 5: Test Files (SKIP - Hardcoded values OK)

### tests/test_game_id_utils.py
- Lines 18, 33, 102: Test assertions with 2025 - INTENTIONAL

---

## Notes

### Files Intentionally Hardcoded:
- `fetch_player_stats_2025.py` - Specific to 2025 data fetching
- `fetch_injury_data.py` - Year detection logic
- Analysis/backtest files dated (e.g., `nov9`) - Specific point-in-time analysis

### Already Fixed:
- [x] api_server.py - All endpoints use dynamic defaults
- [x] backend/database/local_db.py - All repository methods use dynamic defaults
- [x] backend/api/defense_analyzer.py - Uses dynamic defaults
- [x] backend/api/situational_analyzer.py - Uses dynamic defaults
- [x] backend/api/evaluation_pipeline.py - Uses dynamic defaults
- [x] start_server.py - Uses dynamic defaults
- [x] mcp_server.py - Uses dynamic defaults
- [x] nfl_manager.py - Uses dynamic defaults
- [x] backend/config.py - Has current_season/current_week settings

---

## How to Fix Hardcoded Defaults

```python
# Import at top of file:
from backend.nfl_calendar import get_current_nfl_season, get_current_nfl_week

# Change function signature:
# FROM: def my_function(season: int = 2025, week: int = 12):
# TO:   def my_function(season: int = None, week: int = None):

# Add resolution at start of function:
def my_function(season: int = None, week: int = None):
    season = season or get_current_nfl_season()
    week = week or get_current_nfl_week()
    # ... rest of function
```

---

## Progress Tracking

| Priority | Total | Fixed | Remaining |
|----------|-------|-------|-----------|
| P1 Critical | 5 | 5 | 0 |
| P2 Analysis | 12 | 6 | 0 (6 intentionally skipped) |
| P3 Features | 15 | 14 | 1 |
| P4 API | 17 | 0 | 17 |

**Overall: 25/49 items fixed (P1 & P2 Complete, P3 Nearly Complete)**

Note: Synthetic data generators deleted, fallbacks removed.

### P3 Completed Items:
- calibrate.py: Platt scaling, isotonic regression, save/load
- backtest.py: Classification and regression metrics with sklearn
- player_map.py: JSON loading, fuzzy matching with fuzzywuzzy
- extract_weather_features.py: OpenWeather API integration
- hfa_impact_analysis.py: Data-driven HFA calculation method
- external_apis.py: Forecast time matching, stadium dome detection
- fetch_odds.py: OddsAPI integration with quota tracking
- meta_trust_model.py: Bet context extraction for game features
