# NFL BETTING PROPS SYSTEM - COMPREHENSIVE AUDIT REPORT
Generated: 2025-11-20

---

## EXECUTIVE SUMMARY

The NFL betting props system is **partially integrated but has significant data flow gaps**. Key finding:

**Critical Issue**: Injury data is fetched and processed but **never integrated into predictions or pick generation**. Weather data infrastructure exists but is not used in models. Defensive rankings are calculated with a simplified approach that doesn't properly track opponent stats.

---

## 1. COMPONENT INVENTORY

### A. DATA INGESTION LAYER
**Status: ~70% complete**

Implemented:
- `fetch_nflverse.py` - Play-by-play, player stats, rosters, snap counts, depth charts
- `fetch_nflverse_schedules.py` - Game schedules  
- `fetch_odds.py` - Sportsbook odds from The Odds API
- `fetch_prop_lines.py` - Prop lines with extended markets
- `fetch_injury_data.py` - nflverse injury reports (Out, Doubtful, Questionable)
- `fetch_injuries.py` - Alternative injury data source

Missing/Incomplete:
- Weather data fetching (Only Open-Meteo stub exists)
- Depth chart parsing for snap allocation
- Target share tracking from snap data
- Player usage efficiency metrics from snap counts

### B. FEATURE EXTRACTION LAYER
**Status: ~60% complete**

Implemented:
- `extract_player_pbp_features.py` - Basic stats + EPA metrics
  - Passing: yards, TDs, completions, attempts, air yards, CPOE, EPA
  - Rushing: yards, TDs, attempts, success rate, EPA
  - Receiving: yards, TDs, receptions, targets, air yards, YAC, EPA
  - Advanced: WPA, xYAC, QB pressure metrics
- `extract_weather_features.py` - Weather framework (NOT integrated)
- `extract_context_features.py` - Game context features
- `extract_market_context.py` - Market spread/total features

Missing/Not Extracted:
- **Snap counts** - Downloaded but not extracted as features
- **Target share** - Not calculated from targets/team totals
- **Snap percentage** - Available in database but not in feature engineering
- **Injury impact** - Injury data exists but not merged into features
- **Game script** - Spread/total context limited
- **Weather** - API framework exists but data not fetched/used

### C. MODELING LAYER
**Status: ~80% complete**

Implemented:
- `train_multi_prop_models.py` - 60+ prop type models (passing, rushing, receiving, TDs, combos, quarters/halves)
- `train_passing_model.py` - QB-specific models
- `train_usage_efficiency_models.py` - Usage models (touches, snaps, targets)
- `train_quantile_models.py` - Confidence interval models
- `train_quarter_share_models.py` - Proportional quarter models
- Separate models: TD scorers, kicker props, game derivatives, longest plays

Features Used in Models:
- EPA metrics (team and player level)
- CPOE (QB accuracy vs secondary quality)
- Success rate (efficiency)
- Volume metrics (touches, targets)
- Historical averages and rolling windows

**GAP**: No injury impact features, no snap % changes, no target share changes

### D. PREDICTION ENGINE
**Status: ~70% complete**

Implemented:
- `prediction_engine.py` - Multi-signal system:
  1. Contextual Performance (30%) - splits by opponent tier
  2. EPA Matchup (25%) - offensive EPA vs defensive EPA allowed
  3. Success Rate Edge (15%)
  4. CPOE Edge (10%, QB only)
  5. Pressure Matchup (10%, QB only)
  6. Trend Momentum (5%) - recent EPA trajectory
  7. Game Script (5%) - spread/total adjustments

**Signals NOT in prediction engine**:
- Injury adjustments (star player out = backup opportunity)
- Weather impacts
- Depth chart changes
- Target share changes
- Snap count allocation changes

### E. RECOMMENDATIONS/PICKS GENERATION
**Status: ~50% complete**

Implemented:
- `generate_game_picks.py` - Narrative-based parlay generation
  - Uses player projections from CSV files
  - Applies defensive matchup context
  - Builds narrative parlays (Ground Game Stuffed, Secondary Lockdown, etc.)
  - Estimates hit rates and EV

**Major Gaps**:
- **Does NOT load injury data** - No check if player is active
- **Does NOT load depth chart data** - Assumes full starters
- **Does NOT use weather data** - Even though schema exists
- **Defensive ranking calculation is broken** - Uses own team stats instead of opponent stats allowed
- **No actual current odds** - Uses hardcoded projection-based lines

### F. ORCHESTRATION/PIPELINE
**Status: ~70% complete**

Pipeline stages implemented:
1. Data ingestion (PBP, schedules, odds, partial injuries)
2. Feature extraction (PBP features, partial context)
3. Feature engineering (smoothing, rolling windows)
4. Roster/Injury indexing (infrastructure exists, not fully populated)
5. Model training (comprehensive)
6. Prediction generation
7. Backtest/calibration
8. Pick generation

**Missing connections**:
- Injury data merge NOT in pipeline
- Weather data fetch NOT in pipeline
- Snap count extraction NOT in pipeline
- Depth chart processing NOT in pipeline
- No feedback loop from actual game results to injury adjustments

### G. DATABASE/STORAGE
**Status: ~60% complete**

Implemented:
- SQLite database with tables for:
  - Player stats (including snap_pct field)
  - Injuries
  - Rosters
  - Odds
  - Projections
  - Model runs

Missing:
- Actual snap count data population
- Historical target share tracking
- Weather data storage
- Game script adjustments
- Injury impact factors

---

## 2. CONNECTION MAP & DATA FLOW

### CURRENT DATA FLOW (What Works)

```
PBP Data (CSV/Parquet)
    ↓
extract_player_pbp_features.py
    ↓ (outputs: player_pbp_features_by_id.json)
smoothing_and_rolling.py
    ↓ (outputs: player_features_smoothed.json)
[Features stored in inputs/ and outputs/]
    ↓
train_multi_prop_models.py
    ↓ (outputs: models/*.pkl)
generate_projections.py
    ↓ (outputs: projections/*.json)
prediction_engine.py (or generate_game_picks.py)
    ↓
picks_pipeline.py / picks_generator
    ↓
API server / JSON output
```

### BROKEN/MISSING DATA FLOWS

1. **Injury Data → Predictions**: BROKEN
   ```
   fetch_injury_data.py → injuries.json
   ↓
   [DEAD END - not consumed by feature extraction or models]
   generate_game_picks.py does NOT load injury data
   ```

2. **Weather Data → Models**: NOT IMPLEMENTED
   ```
   extract_weather_features.py (stub only)
   ↓
   [NEVER CALLED - no weather data fetched]
   [If it ran, output wouldn't be consumed by models]
   ```

3. **Snap Counts → Features**: NOT EXTRACTED
   ```
   fetch_nflverse.py downloads snap_counts_YYYY.csv
   ↓
   [File exists in inputs/ but NEVER PROCESSED]
   [No feature extraction script for snap data]
   [build_game_roster_index.py expects snap data but doesn't get it]
   ```

4. **Target Share → Models**: NOT CALCULATED
   ```
   Targets are extracted from PBP
   ↓
   [Team target totals NOT calculated]
   [Target share % NOT computed for each receiver]
   [Models use absolute targets, not share]
   ```

5. **Depth Charts → Snap Allocation**: NOT USED
   ```
   fetch_nflverse.py downloads depth_charts_YYYY.csv (46MB file!)
   ↓
   [File exists but NEVER READ]
   [No processing for snap allocation by depth chart position]
   [Backup vs starter role NOT inferred from depth]
   ```

6. **Defensive Stats → Rankings**: INCORRECTLY CALCULATED
   ```
   generate_game_picks.py._calculate_defensive_rankings()
   ↓
   Uses TEAM'S OWN STATS grouped by team
   ↓
   WRONG: Should use OPPONENT STATS when that team was opponent
   ↓
   Result: Rankings meaningless
   ```

7. **Game Script → Adjustments**: MINIMAL
   ```
   Spread/total available in schedule/odds
   ↓
   game_script_adjustment in prediction_engine is only 5% weight
   ↓
   No volume reduction for trailing team (less carries/targets)
   ```

---

## 3. SPECIFIC ISSUES FOUND

### Issue #1: Injury Data Not Used in Picks Generation
**File**: `backend/recommendations/generate_game_picks.py`
**Line**: Entire file
**Problem**: No injury loading whatsoever

```python
# Current code:
class GamePicksGenerator:
    def __init__(self, inputs_dir: str = "inputs"):
        self.inputs_dir = Path(inputs_dir)
        self.stats = pd.read_csv(self.inputs_dir / "player_stats_2024_2025.csv")
        # ^^^ ONLY loads stats CSV, no injury data

# Missing code:
# self.injuries = load_injury_data()
# self.depth_charts = load_depth_charts()
```

**Impact**: 
- System doesn't know which starting players are OUT
- No backup opportunity detection
- Projections used even for inactive players

### Issue #2: Defensive Rankings Use Wrong Baseline
**File**: `backend/recommendations/generate_game_picks.py`
**Lines**: 149-202 (_calculate_defensive_rankings method)
**Problem**: Calculates rankings from team stats, not opponent stats

```python
# Current WRONG code:
for team in teams:
    team_games = recent_stats[recent_stats['team'] != team]  # Gets non-team stats
    opp_stats = recent_stats.groupby('team').agg({...})      # Groups by team itself
    
    # This calculates: "What were BUF's stats on average"
    # NOT: "What stats did opponents have AGAINST BUF defense"

# Correct approach would be:
# For each team, find games where they were opponent
# Sum OPPONENT stats in those games
# That shows what defenses ALLOW, not what teams produce
```

**Impact**:
- Defensive matchups are essentially random noise
- No real edge from defensive context
- High-quality defenses not correctly identified

### Issue #3: Weather Features Not Integrated
**Files**: 
- `backend/features/extract_weather_features.py` (stub only)
- `backend/modeling/train_multi_prop_models.py` (doesn't use weather)
- `generate_game_picks.py` (no weather data)

**Problem**: Infrastructure exists but never populated or used

```python
# extract_weather_features.py only has stubs:
def fetch_weather_from_api(...):
    weather = {
        'temp_f': 65,  # HARDCODED!
        'wind_mph': 8,
        'precipitation': False,
    }
    # TODO: Implement actual API call
    pass

# Never called from pipeline
# Not in training features
# Not in prediction logic
```

**Impact**:
- Wind >15mph can reduce passing yards by 20-30%
- Cold <32F reduces passing efficiency
- Rain/snow increases run attempts
- All of this is invisible to models

### Issue #4: Snap Count Data Downloaded But Never Processed
**Files**:
- `backend/ingestion/fetch_nflverse.py` (downloads snap_counts_YYYY.csv)
- No extraction script
- `backend/roster_injury/build_game_roster_index.py` (expects snap data, gets nothing)

**Problem**: File sits in inputs/ unused

**Impact**:
- No snap % tracking
- Can't detect reduced snap count (injury, benching)
- Can't calculate true usage (touches/snap)
- Can't identify rotational vs featured back

### Issue #5: Target Share Not Calculated
**File**: `backend/modeling/train_usage_efficiency_models.py` (line ~290)
**Problem**: Hardcoded approximation instead of real calculation

```python
# Current code:
df['target_share_rolling_3'] = df['targets_rolling_3'] / 35  # Approx team targets

# Should be:
# Sum all targets on team per game
# Divide each player's targets by that total
# Results will be very different (35 is wrong estimate)
```

**Impact**:
- Target share predictions are inaccurate
- Can't detect when player's share drops
- Can't identify breakout candidates when share increases

### Issue #6: Injury Status Never Checked in Pick Generation
**File**: `backend/recommendations/generate_game_picks.py`
**Lines**: 329-430 (generate_picks method)
**Problem**: No filtering for actual playing status

```python
# Current code:
players = self.stats[
    (self.stats['week'] >= max_week - 3) &
    (self.stats['team'].isin(teams)) &
    (self.stats['position'].isin(['QB', 'RB', 'WR', 'TE']))
][['player_id', 'player_display_name', 'team', 'position']].drop_duplicates()

# Missing:
# for each player, check if they're Out/Doubtful this week
# if Out: skip entirely
# if Doubtful: reduce projection or flag confidence

# Backup detection is crude (just based on projection threshold):
is_backup = False
if strategy['prop'] == 'rushing_yards' and projection < 40:
    is_backup = True  # WRONG: Injured starter might have low proj too
```

**Impact**:
- Recommends bets on players who won't play
- Missing backup opportunity bets
- Incorrect backup identification

### Issue #7: Database Has snap_pct Column But Never Populated
**File**: `backend/database/local_db.py`
**Lines**: Creates snap_pct column but never filled

```python
# Schema has:
snap_pct REAL,  # Exists but...

# But no code actually populates it from snap_counts CSV
# Players show 0 snap_pct or NULL
```

**Impact**:
- Snap data unusable even if someone adds snap extraction

---

## 4. MISSING VARIABLES NOT BEING TRACKED

| Variable | Status | Impact |
|----------|--------|--------|
| **Injury Status** | Fetched but not used | No injury adjustments |
| **Snap Count** | Downloaded but not processed | No usage tracking |
| **Target Share %** | Estimated (wrong) | Inaccurate usage models |
| **Snap % Changes** | Not tracked | Can't detect reduced availability |
| **Depth Chart Position** | Downloaded but not processed | Can't prioritize starters vs backups |
| **Game Script Adjustment** | Minimal (5% weight) | Not handling blowout scripts |
| **Weather (Wind, Temp)** | Infrastructure only | No weather impact on passing |
| **Pressure Rate vs OL** | Database field exists | Not fully integrated |
| **Weekly Ownership %** | Not tracked | Can't target contrarian plays |
| **Vegas Line Movement** | Not tracked | Can't detect sharp/soft money |
| **Actual Current Odds** | Not loaded | Using projections not real lines |

---

## 5. WHAT VARIABLES SHOULD BE TRACKED

### Must-Have (High Impact)
1. **Injury Status per Game**
   - Out/Doubtful/Questionable → projection multiplier
   - Backup role detection from roster
   - Impact factor by position (QB injury = huge)

2. **Snap Count Changes**
   - Week-over-week snap % trends
   - When snaps drop → projection reduction
   - Rotational role detection

3. **Target Share Trends**
   - % of team targets per receiver
   - When target share drops → red flag
   - When share increases → opportunity

4. **Game Script Dynamics**
   - Spread and total at kickoff
   - Predicted game flow (run vs pass)
   - Volume adjustments for blowouts

### Should-Have (Medium Impact)
5. **Weather Impact**
   - Wind speed → passing yards reduction
   - Temperature → efficiency
   - Precipitation → run emphasis

6. **Defensive Opponent Metrics**
   - EPA allowed vs position
   - Pressure rate generated
   - CPOE allowed (secondary quality)

7. **Vegas Movement**
   - Line opening vs current
   - Sharp vs soft indicators
   - Sharp bets on/against

### Nice-to-Have (Low-Medium Impact)
8. **Red Zone Efficiency**
   - Team red zone success %
   - Player goal-line role
   - TD probability by situation

9. **Ownership/Contrarian Value**
   - DFS ownership %
   - Sportsbook liability info
   - Fade high-owned props

10. **Historical H2H**
    - Player vs specific defense history
    - Divisional game context
    - Playoffs vs regular season

---

## 6. RECOMMENDATIONS FOR FIXING THE SYSTEM

### Phase 1: CRITICAL (Do First - 2-3 weeks)

**1A. Integrate Injury Data into Predictions**
- Load injury JSON in `generate_game_picks.py`
- Check `expected_to_play` flag for each player
- If Out: skip player completely
- If Doubtful: reduce projection by 30-50%
- Files to create:
  - `backend/data/injury_loader.py` - Load injury index
  - Update `generate_game_picks.py` to call it

**1B. Fix Defensive Rankings Calculation**
- Rebuild `_calculate_defensive_rankings()` properly
- For each team, find opponent-game records
- Calculate what opponents actually achieved against that defense
- Or better: Load pre-calculated defensive stats from PBP

**1C. Implement Actual Current Odds Loading**
- `generate_game_picks.py` currently uses projections as lines
- Add actual sportsbook odds fetch
- Compare projection vs real line for actual edge
- Use Odds API properly (don't just sample data)

### Phase 2: HIGH-PRIORITY (Next 3-4 weeks)

**2A. Extract and Use Snap Count Data**
- Create `backend/features/extract_snap_features.py`
- Load snap_counts_YYYY.csv
- Calculate snap % per game per player
- Add to player features JSON
- Track week-over-week changes
- Include in training features

**2B. Calculate Actual Target Share**
- Modify feature extraction to sum team targets per game
- Calculate receiver target % = player_targets / team_targets
- Track target share trends (3-week rolling)
- Add as features to models

**2C. Implement Weather Data Pipeline**
- Replace stub in `extract_weather_features.py` with real API
- Or use historical weather database
- Add weather features to model training
- Include in prediction engine (especially wind impact on passing)

**2D. Process Depth Charts**
- Create `backend/features/extract_depth_charts.py`
- Load depth_charts_YYYY.csv
- Determine starter vs backup role per player per game
- Add depth chart position to features
- Use for backup opportunity detection

### Phase 3: MEDIUM-PRIORITY (Weeks 5-6)

**3A. Add Game Script Adjustment**
- Increase weight from 5% to 15-20%
- Calculate expected game script from spread/total
- Predict if game will be run-heavy, pass-heavy, etc.
- Reduce volume projections for trailing teams
- Increase for leading teams (garbage time)

**3B. Connect Orchestrator Fully**
- Add injury data fetch to orchestrator pipeline
- Add snap extraction stage
- Add depth chart processing stage
- Add weather data stage
- Ensure all intermediate files created

**3C. Implement Backup Opportunity Detection**
- When starter is Out/Doubtful
- Check backup on depth chart
- Apply projection boost for backup role
- Compare backup's recent stats vs projection

### Phase 4: NICE-TO-HAVE (Weeks 7+)

**4A. Vegas Movement Tracking**
- Store line opening vs current spread/total
- Flag sharp money movements
- Use as confidence indicator

**4B. Red Zone Specific Models**
- Separate TD models for red zone touches
- By situation (1st&G, short yardage)
- More predictive than general TD models

**4C. Contrarian Value System**
- Track DFS ownership% where available
- Flag props with low public ownership
- Higher EV potential

---

## 7. IMPLEMENTATION PRIORITY MAP

### Must-Do (Blocks accurate picks)
- [ ] Load injury data in picks generator
- [ ] Fix defensive ranking calculation
- [ ] Load actual current odds (not projections)
- [ ] Skip injured/inactive players

### Should-Do (Improves accuracy)
- [ ] Extract snap counts and % changes
- [ ] Calculate actual target share
- [ ] Implement weather features
- [ ] Process depth charts
- [ ] Detect backup opportunities

### Audit Issues to Address
1. **`generate_game_picks.py`**: Major rewrite needed
   - Add injury loading
   - Add weather loading
   - Fix defensive rankings
   - Load actual odds
   - Check player active status

2. **Feature Extraction**: Add missing scripts
   - Snap count extraction
   - Depth chart processing
   - Weather features (real, not stub)
   - Game script signals

3. **Orchestrator**: Add missing stages
   - Snap feature extraction
   - Depth chart processing
   - Weather data fetch
   - Injury data merge

4. **Models**: Include new features
   - Injury impact factors
   - Snap % changes
   - Target share
   - Weather (especially wind)
   - Game script

---

## 8. DATA FLOW SUMMARY TABLE

| Stage | Input | Process | Output | Status | Gap |
|-------|-------|---------|--------|--------|-----|
| Ingestion | External APIs | Fetch data | CSVs in inputs/ | 70% | Weather, snap processing |
| PBP Features | play_by_play.csv | Extract EPA, targets | JSON features | 80% | Snap %, target share |
| Cleaning | Raw features | Smooth, rolling windows | Clean features | 60% | Injury flags, weather tags |
| Enrichment | Clean features | Merge injury, roster | Enriched features | 20% | NOT DONE |
| Training | Enriched features | Train models | Models .pkl | 70% | Missing injury/weather/snap inputs |
| Prediction | Models + Player data | Generate projections | Projections | 70% | Injury adjustments missing |
| Picks Gen | Projections + Odds | Filter edges, build parlays | Picks JSON | 50% | Injury checks, real odds, proper def rankings |
| Output | Picks JSON | Format for user | API/JSON | 80% | Missing confidence factors |

---

## FINAL ASSESSMENT

**System Completeness: 58%**

- Data ingestion: 70%
- Feature extraction: 60%
- Modeling: 80%
- Prediction: 70%
- Recommendations: 50%
- **Integration: 30% ← BIGGEST PROBLEM**

**What Works**: Model training, basic projections, narrative parlays
**What's Broken**: Injury integration, defensive matchups, weather, snap/target tracking, actual odds loading
**What's Missing**: Proper data flow, backup detection, game script adjustments, injury impact factors

**Recommendation**: Fix data integration before adding features. The pipes are broken, not the endpoints.

