# NFL Backend MCP Integration Guide

## Directory Structure Overview
- `/home/user/nfl_backend/backend/ingestion/` - Data fetching modules
- `/home/user/nfl_backend/backend/modeling/` - Training & prediction pipelines
- `/home/user/nfl_backend/backend/orchestration/` - Pipeline orchestration
- `/home/user/nfl_backend/backend/api/` - API & analysis engines
- `/home/user/nfl_backend/mcp_server.py` - Existing MCP server (betting-focused)

---

## INGESTION LAYER - Data Fetching Capabilities

### 1. **NFL Play-by-Play & Player Stats** (nflverse)
**File:** `/home/user/nfl_backend/backend/ingestion/fetch_nflverse.py`

**Main Function:** `fetch_nflverse(year, out_dir, cache_dir=None, include_all=True)`

**Parameters:**
- `year` (int): NFL season year (e.g., 2023, 2024, 2025)
- `out_dir` (Path): Output directory for processed CSV files
- `cache_dir` (Path, optional): Cache directory to avoid re-downloading
- `include_all` (bool): Download all datasets or core only (default: True)

**Returns:** None (writes files to disk)

**What It Downloads:**
```
player_stats_{year}.csv          - Weekly player statistics (all positions)
play_by_play_{year}.csv          - Play-by-play with 300+ metrics:
                                   * EPA (Expected Points Added)
                                   * WPA (Win Probability Added)
                                   * CPOE (Completion % Over Expected)
                                   * Air Yards, Yards After Catch
                                   * Success rates, QB pressure metrics
weekly_rosters_{year}.csv         - Roster data for injury tracking
nextgen_stats_{year}.csv          - Player tracking data (if include_all=True)
snap_counts_{year}.csv            - Snap counts & participation %
pfr_advstats_pass_{year}.csv      - PFR advanced passing stats
pfr_advstats_rush_{year}.csv      - PFR advanced rushing stats
pfr_advstats_rec_{year}.csv       - PFR advanced receiving stats
pfr_advstats_def_{year}.csv       - PFR advanced defensive stats
depth_charts_{year}.csv           - Weekly depth charts
player_lookup_{year}.json         - Player metadata lookup
```

**Data Source:** GitHub nflverse releases
**Advanced Metrics Available:** EPA, WPA, CPOE, air yards, success rate, pressure stats

---

### 2. **NFL Schedule Data**
**File:** `/home/user/nfl_backend/backend/ingestion/fetch_nflverse_schedules.py`

**Main Function:** `fetch_schedule(year, out_dir, cache_dir=None)`

**Parameters:**
- `year` (int): NFL season year
- `out_dir` (Path): Output directory
- `cache_dir` (Path, optional): Cache directory

**Returns:** None (writes schedule_{year}.csv)

**Output:** `schedule_{year}.csv` with columns: season, week, game_id, home_team, away_team, kickoff_time, results, spreads, totals

---

### 3. **Player Injury Reports** (ESPN API)
**File:** `/home/user/nfl_backend/backend/ingestion/fetch_injuries.py`

**Main Class:** `InjuryFetcher`

**Key Methods:**
- `fetch_all_injuries(output_dir, week=None) -> Dict`: Fetch injury reports for all 32 NFL teams
- `get_player_injury_status(player_name, team, injury_file) -> Optional[Dict]`
- `get_team_key_injuries(team, injury_file, positions=['QB','RB','WR','TE']) -> List[Dict]`

**Main Function:** `fetch_injuries(output_dir, week=None) -> Dict`

**Returns:**
```json
{
  "timestamp": "ISO timestamp",
  "total_teams": 32,
  "teams_with_injuries": N,
  "total_injuries": N,
  "by_status": {
    "OUT": count,
    "DOUBTFUL": count,
    "QUESTIONABLE": count
  }
}
```

**Output Files:**
- `injuries_{week}.json` - Team -> list of injury dicts
- `injuries_{week}_summary.json` - Summary stats

**Injury Status Values:** OUT, DOUBTFUL, QUESTIONABLE, PROBABLE

---

### 4. **Historical Injury Data** (nflverse)
**File:** `/home/user/nfl_backend/backend/ingestion/fetch_injury_data.py`

**Main Functions:**
- `fetch_injury_data(season, output_dir, cache_dir=None) -> Dict`
- `merge_injury_into_features(features_file, injury_file, output_file) -> Dict`

**Parameters:**
- `season` (int): Season year
- `output_dir` (Path): Where to save injury JSON
- `cache_dir` (Path, optional): Cache directory

**Returns:** Dict mapping `season_week_gsis_id` -> injury info

**Output:** `{season}_injuries.json` with injury status, type, practice status, and DNP reasons

---

### 5. **Prop Betting Lines with Line Movement Tracking**
**File:** `/home/user/nfl_backend/backend/ingestion/fetch_prop_lines.py`

**Main Class:** `PropLineFetcher(api_key)`

**Key Methods:**
- `fetch_upcoming_games() -> List[Dict]` - Get list of upcoming games
- `fetch_prop_odds_for_game(game_id, market, bookmakers=None) -> Dict` - Single market
- `fetch_all_prop_odds_for_game(game_id, markets, bookmakers=None) -> Dict` - Batch (efficient)
- `fetch_snapshot_with_movement(output_dir, week=None, load_previous=True) -> Dict` - Full snapshot
- `get_trending_props(snapshots_dir, week=None, lookback_days=7) -> Dict` - Trending analysis
- `analyze_sharp_action(props) -> Dict` - Detect smart money movement

**Main Function:** `fetch_prop_lines(api_key, output_dir, week=None) -> Dict`

**Parameters:**
- `api_key` (str): The Odds API key (from ODDS_API_KEY env var)
- `output_dir` (Path): Where to save snapshots
- `week` (int, optional): NFL week number

**Prop Markets Tracked (60+):**

**Full Game:**
- Passing: yards, TDs, completions, attempts, interceptions, longest
- Rushing: yards, TDs, attempts, longest
- Receiving: receptions, yards, TDs, longest
- Touchdowns: anytime, first, last
- Defense: tackles+assists, sacks, interceptions
- Combos: pass+rush yards, pass+rush TDs, etc.

**Quarter/Half Props:**
- 1H (First Half), 1Q, 2H, 3Q, 4Q with proportional models
- Pass/rush/rec yards and TDs for each quarter

**Returns:**
```json
{
  "games": number_of_games,
  "props": total_prop_count,
  "hot_movers": count_of_2plus_point_moves,
  "snapshot_time": "ISO timestamp",
  "output_file": "path/to/snapshot"
}
```

**Output Files:**
- `snapshot_{week}_{timestamp}.json` - Full prop snapshot with OVER/UNDER odds
- `snapshot_{week}_latest.json` - Latest snapshot for comparison
- `hot_movers_{week}_{timestamp}.json` - Props with 2+ point movement

**Sportsbooks Tracked:**
- DraftKings (PRIMARY - user's book)
- FanDuel, Caesars, BetMGM, PointsBet, Unibet (for sharp action detection)

**Line Movement Features:**
- Opening vs current line tracking
- Hot mover detection (2+ point moves)
- Sharp action signals (DK isolated vs sharp consensus)
- Sustained trend analysis (3-week consistent direction)

---

### 6. **Placeholder: Odds API (H2H, Spreads, Totals)**
**File:** `/home/user/nfl_backend/backend/ingestion/fetch_odds.py`

**Function:** `fetch_odds_api(sport='americanfootball_nfl', markets='h2h,spreads,totals', cache_dir=Path('cache')) -> List[Dict]`

**Status:** Placeholder - would fetch main game odds (not props)

---

## MODELING LAYER - Training & Prediction Pipelines

### 1. **Multi-Prop Model Training System** (Comprehensive)
**File:** `/home/user/nfl_backend/backend/modeling/train_multi_prop_models.py`

**Main Function:** `train_multi_prop_models(season, input_dir, output_dir, model_type='xgboost')`

**What It Does:**
- Trains separate models for 60+ prop types
- Handles full game props + quarter/half proportional models
- Accounts for player availability (DNP instances)
- Generates projections for ALL markets

**Prop Models Trained (60+):**

**Full Game:**
- player_pass_yds, player_pass_tds, player_pass_completions, player_pass_attempts, player_pass_interceptions
- player_rush_yds, player_rush_tds, player_rush_attempts
- player_receptions, player_reception_yds, player_reception_tds
- player_anytime_td, player_first_td, player_last_td
- player_tackles_assists, player_sacks, player_interceptions
- Combo props: player_pass_rush_yds, etc.

**Quarter/Half Props (with proportional scaling):**
- 1H: 52% of full game expectation
- 1Q: 25% of full game expectation
- 2H: 48% of full game expectation
- 3Q, 4Q: 24% each

**Features Used:**
- EPA-based metrics (QB EPA, total EPA)
- CPOE (Completion % Over Expected)
- Success Rate (positive EPA plays)
- Air Yards & YAC decomposition
- WPA (clutch performance)
- QB Pressure (hits, hurries, sacks)
- Rolling averages (3-game, 5-game, season)

**Output:**
- Trained models saved to `outputs/models/multi_prop/*.pkl`
- Feature importance analysis (JSON + visualization)
- Training metrics (RMSE, MAE, R², feature importance)

---

### 2. **Projection Generation Engine**
**File:** `/home/user/nfl_backend/backend/modeling/generate_projections.py`

**Main Class:** `ProjectionGenerator(models_dir="outputs/models", inputs_dir="inputs", outputs_dir="outputs/predictions")`

**Key Methods:**
- `generate_for_week(week, season=2025) -> str` - Generate for all players in a week
- `generate_for_game(game_id, week) -> str` - Generate for specific game
- `predict_prop(player_id, prop_type, features) -> Optional[Dict]` - Single prop prediction
- `get_player_features(player_id, week) -> Dict` - Get latest features

**Parameters:**
- `week` (int): NFL week number
- `season` (int): Season year (default 2025)
- `game_id` (str): Format "2025_10_KC_BUF"

**Returns:**
```python
{
  "projection": float,              # Mean prediction
  "std_dev": float,                 # Standard deviation
  "confidence_lower": float,        # 95% CI lower
  "confidence_upper": float,        # 95% CI upper
  "hit_probability_over": float,    # P(outcome > line)
  "hit_probability_under": float    # P(outcome < line)
}
```

**Output:** `props_{season}_{week}.csv` with columns:
```
player_id, player_name, team, opponent, position, game_id, 
prop_type, projection, std_dev, confidence_lower, confidence_upper,
hit_probability_over, hit_probability_under
```

**Prop Types Generated (by position):**
- QB: passing_yards, passing_tds, completions, attempts, interceptions
- RB: rushing_yards, rushing_tds, carries, receptions, receiving_yards
- WR/TE: receptions, receiving_yards, receiving_tds, targets

**Features Used:**
- Player stats from last 4 weeks
- Season averages and 3-game rolling averages
- Game context (home/away, spread, total)
- Injury adjustments (if available)

---

### 3. **Passing Prop Model Training**
**File:** `/home/user/nfl_backend/backend/modeling/train_passing_model.py`

**Function:** `train_passing_model(season, features_file, output_model_path, model_type='xgboost') -> Dict`

**Parameters:**
- `season` (int): Season year
- `features_file` (Path): Path to player features JSON
- `output_model_path` (Path): Where to save trained model
- `model_type` (str): 'xgboost' or 'lightgbm'

**Returns:** Dict with training metrics (RMSE, MAE, R², feature importance)

**Features:**
- EPA-based: qb_epa, total_epa (rolling averages)
- CPOE (QB accuracy metric)
- Success Rate (% positive EPA plays)
- Air Yards & YAC decomposition
- WPA (Win Probability Added)
- QB Pressure (hits, hurries, sacks)
- Traditional stats for context
- Rolling windows: 3-game, 5-game, season

---

### 4. **Model Runner (Framework)**
**File:** `/home/user/nfl_backend/backend/modeling/model_runner.py`

**Main Class:** `PropModel(prop_type)`

**Key Methods:**
- `load_features(features_path) -> Dict`
- `load_team_profiles(team_profiles_path) -> Dict`
- `predict(player_id, opponent, features) -> float`

**Function:** `run_model_pipeline(features_path, team_profiles_path, output_path, game_id) -> None`

**Output:** CSV with columns: game_id, player_id, prop_type, prediction, confidence

---

### 5. **Distribution Models for Probabilities**
**File:** `/home/user/nfl_backend/backend/modeling/distributions.py`

**Classes:**
- `NormalDistribution(mean, std)` - For yardage props
- `PoissonDistribution(lam)` - For count data (receptions, TDs)

**Methods:**
- `cdf(x) -> float` - Cumulative distribution
- `prob_over(line) -> float` - P(X > line)
- `prob_under(line) -> float` - P(X < line)

**Usage:** Converts projections (mean + std) to hit probabilities

---

### 6. **Other Training Models** (Available)
**Files:**
- `train_quantile_models.py` - Quantile regression for distribution modeling
- `train_quarter_share_models.py` - Quarter-specific proportional models
- `train_usage_efficiency_models.py` - Usage efficiency and snap metrics
- `probability_calibration.py` - Calibration of probability predictions

---

## ORCHESTRATION LAYER - Pipeline Coordination

### 1. **Full Pipeline Orchestrator**
**File:** `/home/user/nfl_backend/backend/orchestration/orchestrator.py`

**Class:** `PipelineStage(name, script_path, args)`

**Main Entry Point:** Full pipeline coordinates:
1. Data Ingestion (nflverse, schedules, odds, injuries)
2. Feature Extraction (player PBP features)
3. Feature Engineering (smoothing, rolling windows)
4. Roster/Injury Indexing
5. Model Training (passing/rushing/receiving models)
6. Prediction Generation (prop projections)
7. Backtest/Calibration (model validation)

**Usage Examples:**
```bash
# Full pipeline with training
python -m backend.orchestration.orchestrator --season 2024 --train

# Predictions only (skip data/features)
python -m backend.orchestration.orchestrator --season 2024 --predict-only

# Backtest on historical data
python -m backend.orchestration.orchestrator --season 2023 --backtest
```

---

### 2. **Picks Pipeline** (Betting-Focused)
**File:** `/home/user/nfl_backend/backend/orchestration/picks_pipeline.py`

**Main Class:** `PicksPipeline()`

**Key Methods:**
- `generate_picks(week) -> List[PickRecommendation]` - Generate betting picks
- Integrates: Model projections + Current odds + Prop analysis + Portfolio optimization

**Output:** `PickRecommendation` dataclass with:
- Player identity (player_id, player_name, game_id, team, opponent)
- Prop details (prop_type, line, side: OVER/UNDER, odds)
- Model outputs (projection, confidence, edge)
- Recommendation rationale

---

## EXISTING MCP SERVER - Already Exposed Tools

**File:** `/home/user/nfl_backend/mcp_server.py`

**Currently Exposed Tools:**
1. `get_best_props` - Find value bets (min_edge, game_ids, limit)
2. `get_matchup_analysis` - Detailed game analysis (game_id)
3. `get_trending_props` - Hot movers & line movement (week, limit)
4. `get_parlay_suggestions` - Correlation-aware parlays (game_ids, max_legs, limit)
5. `get_player_trend` - Player performance trends (player_name, stat)
6. `get_schedule` - Upcoming games schedule (week)
7. `fetch_dk_odds` - Fetch fresh DraftKings odds (week)

---

## RECOMMENDED MCP ADDITIONS - Data Fetching & Training

### Tier 1: Critical Data Fetching Functions
**Priority: HIGH** - Essential for keeping data fresh

1. **fetch_nflverse_data**
   - Parameters: year, include_all=True
   - Triggers: `fetch_nflverse()` with all advanced metrics
   - Returns: Confirmation + file list

2. **fetch_injury_reports**
   - Parameters: week (optional)
   - Triggers: `fetch_injuries()` from ESPN API
   - Returns: Injury summary by team + key injuries

3. **fetch_prop_snapshots**
   - Parameters: week (optional)
   - Triggers: `fetch_prop_lines()` with line movement tracking
   - Returns: Snapshot summary + hot movers count

4. **analyze_line_movement**
   - Parameters: week (optional), lookback_days=7
   - Triggers: `get_trending_props()` on existing snapshots
   - Returns: Hot movers + sustained trends

---

### Tier 2: Model Training & Predictions
**Priority: MEDIUM** - For periodic retraining

1. **train_prop_models**
   - Parameters: season, model_type='xgboost'
   - Triggers: `train_multi_prop_models()`
   - Returns: Training completion status + metrics summary

2. **generate_week_projections**
   - Parameters: week, season=2025
   - Triggers: `ProjectionGenerator.generate_for_week()`
   - Returns: Projection file location + stats by position

3. **generate_game_projections**
   - Parameters: game_id, week
   - Triggers: `ProjectionGenerator.generate_for_game()`
   - Returns: Projection file location

4. **get_model_status**
   - Parameters: None
   - Returns: Available models, last training date, data freshness

---

### Tier 3: Pipeline Orchestration
**Priority: MEDIUM** - For coordinated workflows

1. **run_full_pipeline**
   - Parameters: season, include_training=True, include_backtest=False
   - Triggers: Full orchestrator pipeline
   - Returns: Pipeline status + results summary

2. **validate_data_freshness**
   - Parameters: None
   - Returns: Age of each data source + recommendations

---

## Data Dependencies & Flow

```
┌─────────────────────────────────────────┐
│     DATA INGESTION LAYER                │
├─────────────────────────────────────────┤
│ ├─ fetch_nflverse()                     │
│ │  └─ play_by_play, player_stats,       │
│ │     weekly_rosters, snap_counts, etc. │
│ ├─ fetch_schedule()                     │
│ ├─ fetch_injuries()                     │
│ ├─ fetch_injury_data()                  │
│ └─ fetch_prop_lines()                   │
│    └─ Snapshots with line movement      │
└──────────────┬──────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────┐
│  FEATURE EXTRACTION/ENGINEERING         │
├─────────────────────────────────────────┤
│ ├─ Extract player features from PBP     │
│ ├─ Calculate rolling averages           │
│ ├─ Merge injury data                    │
│ └─ Build game-level indices             │
└──────────────┬──────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────┐
│     MODELING LAYER (Training)           │
├─────────────────────────────────────────┤
│ ├─ train_multi_prop_models()            │
│ ├─ train_passing_model()                │
│ ├─ train_quantile_models()              │
│ └─ probability_calibration()            │
└──────────────┬──────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────┐
│  PREDICTION/PROJECTION GENERATION       │
├─────────────────────────────────────────┤
│ ├─ generate_for_week()                  │
│ └─ generate_for_game()                  │
└──────────────┬──────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────┐
│     ANALYSIS & RECOMMENDATIONS          │
├─────────────────────────────────────────┤
│ ├─ Prop analyzer (edge detection)       │
│ ├─ Trend analysis (hot movers)          │
│ ├─ Sharp action detection               │
│ └─ Parlay construction                  │
└─────────────────────────────────────────┘
```

---

## Implementation Checklist for MCP

### Phase 1: Data Management Tools
- [ ] `fetch_nflverse_data(year, include_all=True)`
- [ ] `fetch_current_injuries(week=None)`
- [ ] `fetch_prop_snapshots(week=None)`
- [ ] `get_data_freshness_status()`

### Phase 2: Model Training Tools
- [ ] `train_all_prop_models(season, model_type='xgboost')`
- [ ] `generate_projections_for_week(week, season=2025)`
- [ ] `get_training_metrics(model_type=None)`

### Phase 3: Analysis & Pipeline Tools
- [ ] `run_full_pipeline(season, include_training=True)`
- [ ] `analyze_prop_trends(week=None, lookback_days=7)`
- [ ] `validate_model_calibration()`

### Phase 4: Advanced Tools
- [ ] `backtest_models(season, prop_types=None)`
- [ ] `compare_projections_to_lines(game_ids=None)`
- [ ] `optimize_parlay_portfolio(week, max_legs=3)`

