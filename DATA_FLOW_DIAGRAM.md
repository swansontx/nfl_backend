# NFL Props System - Data Flow Architecture

## Working Data Flows (Green Path)

```
┌─────────────────────┐
│  nflverse PBP CSV   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│  extract_player_pbp_features.py         │
│  - Passing/Rushing/Receiving stats      │
│  - EPA, CPOE, Success Rate              │
│  - WPA, Air Yards, YAC                  │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│  smoothing_and_rolling.py               │
│  - Rolling 3/4 week averages            │
│  - Exponential smoothing                │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│  train_multi_prop_models.py             │
│  - Train 60+ prop types                 │
│  - Using extracted features             │
│  - Output: models/*.pkl                 │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│  generate_projections.py                │
│  - Load models                          │
│  - Generate projections                 │
│  - Output: projections/*.json           │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│  generate_game_picks.py                 │
│  - Load player stats CSV                │
│  - Calculate def rankings (BROKEN)      │
│  - Build narrative parlays              │
│  - Output: picks/*.json                 │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│  API Server / JSON Output               │
│  (Final picks delivered to user)        │
└─────────────────────────────────────────┘
```

---

## BROKEN Data Flows (Red Paths)

### 1. Injury Data Pipeline (FETCHED BUT NOT USED)
```
┌──────────────────────────┐
│  fetch_injury_data.py    │
│  nflverse injuries API   │
│  ✓ Works                 │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  injuries_YYYY.json      │
│  ✓ Created               │
└──────────┬───────────────┘
           │
    ❌ DEAD END ❌
           │
    Not loaded by:
    - extract_player_pbp_features.py
    - train_multi_prop_models.py
    - prediction_engine.py
    - generate_game_picks.py
           │
           ▼
    INJURY STATUS UNKNOWN IN PICKS
    ➜ Recommends bets on inactive players
    ➜ No backup opportunity detection
```

### 2. Weather Data Pipeline (INFRASTRUCTURE ONLY)
```
┌──────────────────────────┐
│  extract_weather_features│
│  .py (STUB ONLY)         │
│  ❌ No real implementation│
└──────────┬───────────────┘
           │
    ❌ NEVER CALLED ❌
           │
    Not in orchestrator
    Not in pipeline
    No real API calls
           │
           ▼
    WEATHER DATA UNUSED
    ➜ No wind impact on passing
    ➜ No cold impact on efficiency
    ➜ No rain impact on run/pass mix
```

### 3. Snap Count Pipeline (DOWNLOADED BUT NOT PROCESSED)
```
┌──────────────────────────────────┐
│  fetch_nflverse.py               │
│  Downloads:                      │
│  snap_counts_YYYY.csv ✓          │
└──────────┬───────────────────────┘
           │ (File size: ~5MB)
           ▼
┌──────────────────────────────────┐
│  inputs/snap_counts_2025.csv     │
│  ✓ File exists                   │
└──────────┬───────────────────────┘
           │
    ❌ NEVER PROCESSED ❌
           │
    No extraction script
    No feature engineering
    No model input
           │
           ▼
    SNAP DATA WASTED
    ➜ No snap % tracking
    ➜ Can't detect reduced availability
    ➜ Can't calculate usage efficiency
```

### 4. Depth Chart Pipeline (DOWNLOADED BUT NOT READ)
```
┌──────────────────────────────────┐
│  fetch_nflverse.py               │
│  Downloads:                      │
│  depth_charts_YYYY.csv ✓         │
└──────────┬───────────────────────┘
           │ (File size: 46MB!)
           ▼
┌──────────────────────────────────┐
│  inputs/depth_charts_2025.csv    │
│  ✓ File exists                   │
└──────────┬───────────────────────┘
           │
    ❌ NEVER READ ❌
           │
    No extraction script
    No depth chart position mapping
    No starter/backup inference
           │
           ▼
    DEPTH CHART UNUSED
    ➜ Can't prioritize starters
    ➜ No backup opportunity signals
    ➜ Snap allocation unknown
```

### 5. Target Share Pipeline (PARTIALLY BROKEN)
```
┌──────────────────────────┐
│  extract_player_pbp_     │
│  features.py            │
│  ✓ Extracts targets     │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  targets field in        │
│  feature JSON            │
│  ✓ Exists               │
└──────────┬───────────────┘
           │
    ❌ NOT AGGREGATED ❌
           │
    No team target sums
    No target share calculation
           │
           ▼
┌──────────────────────────┐
│  train_usage_efficiency_ │
│  models.py              │
│  ❌ Hardcoded estimate  │
│  / 35 (WRONG!)          │
└──────────┬───────────────┘
           │
           ▼
    INACCURATE TARGET SHARE
    ➜ Models trained on wrong data
    ➜ Can't detect share changes
    ➜ Breakout predictions unreliable
```

### 6. Defensive Rankings Pipeline (INCORRECTLY CALCULATED)
```
┌──────────────────────────────────┐
│  generate_game_picks.py          │
│  _calculate_defensive_rankings() │
│  ❌ WRONG LOGIC                  │
└──────────┬───────────────────────┘
           │
    Logic:
    1. Get recent games
    2. Group BY TEAM (own stats)
    3. Calculate team averages
           │
    ❌ SHOULD BE:
    1. For each defense
    2. Find games vs opponents
    3. Sum OPPONENT achievements
    4. That's what defense allowed
           │
           ▼
    MEANINGLESS RANKINGS
    ➜ Defensive context is noise
    ➜ No real matchup edge
    ➜ High-quality defenses not identified
```

### 7. Game Script Pipeline (MINIMAL)
```
┌──────────────────────────┐
│  schedule data           │
│  + odds data             │
│  ✓ Available            │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  prediction_engine.py    │
│  game_script_adjustment  │
│  ❌ Only 5% weight      │
└──────────┬───────────────┘
           │
    ❌ UNDERUTILIZED ❌
           │
    No volume reduction for losers
    No carry/target projection drops
    No blowout scenario modeling
           │
           ▼
    WEAK GAME SCRIPT SIGNALS
    ➜ Misses big game flow shifts
    ➜ Trailing team projections too high
```

---

## Missing Integrations Summary

| Input Data | Status | Flow | Impact |
|-----------|--------|------|--------|
| Injury Status | ✓ Fetched | ❌ Not consumed | HIGH |
| Snap Counts | ✓ Downloaded | ❌ Not processed | HIGH |
| Depth Charts | ✓ Downloaded | ❌ Not read | HIGH |
| Target Share | ✓ Partial | ⚠️ Wrong calc | HIGH |
| Weather Data | ⚠️ Stub only | ❌ Not fetched | MEDIUM |
| Game Script | ✓ Available | ⚠️ Minimal (5%) | MEDIUM |
| Defensive Stats | ✓ Available | ❌ Wrong calc | HIGH |
| Actual Odds | ❌ Not loaded | ❌ Uses projections | HIGH |

---

## Critical Bottlenecks

### 1. Injury Data Integration
**Location**: `generate_game_picks.py` (line 1-50)
```python
def __init__(self, inputs_dir: str = "inputs"):
    self.stats = pd.read_csv(...)  # Only this
    # Missing:
    # self.injuries = load_injuries_from_json()
    # self.depth_charts = load_depth_charts()
    # self.snap_counts = load_snap_counts()
```

### 2. Defensive Ranking Logic
**Location**: `generate_game_picks.py` (lines 149-202)
```python
def _calculate_defensive_rankings(self) -> Dict:
    # Current: Groups by own team stats
    # Result: Meaningless rankings
    # Fix: Calculate opponent achievements against each defense
```

### 3. Snap Data Processing
**Missing File**: `backend/features/extract_snap_features.py`
```python
# Should:
# 1. Load snap_counts_YYYY.csv
# 2. Calculate snap % per player per game
# 3. Track week-over-week changes
# 4. Add to feature JSON
```

### 4. Depth Chart Processing
**Missing File**: `backend/features/extract_depth_charts.py`
```python
# Should:
# 1. Load depth_charts_YYYY.csv
# 2. Map player IDs to depth positions
# 3. Infer starter vs backup role
# 4. Add to feature JSON
```

### 5. Weather Feature Engineering
**Location**: `backend/features/extract_weather_features.py`
```python
# Current: All stubs/hardcoded
# Fix: Implement real Open-Meteo API or historical DB
# Add to pipeline and training
```

---

## How to Fix the Pipes

### Step 1: Injury Integration (1-2 days)
1. Create `backend/data/injury_loader.py`
2. Load injury JSON in `generate_game_picks.py`
3. Check injury status for each player
4. Skip if Out, reduce projection if Doubtful

### Step 2: Snap Extraction (2-3 days)
1. Create `backend/features/extract_snap_features.py`
2. Load snap_counts CSV
3. Calculate snap % per game
4. Track week-over-week changes
5. Add to player features

### Step 3: Depth Chart Processing (1-2 days)
1. Create `backend/features/extract_depth_charts.py`
2. Load depth_charts CSV
3. Map to player IDs
4. Infer roles (starter/backup)
5. Add to features

### Step 4: Fix Defensive Rankings (1 day)
1. Rewrite `_calculate_defensive_rankings()` in `generate_game_picks.py`
2. OR load pre-calculated defensive stats from PBP

### Step 5: Weather Integration (2-3 days)
1. Implement real weather fetching
2. Add weather features to training
3. Include in prediction engine

### Step 6: Load Actual Odds (1 day)
1. Load real odds from Odds API
2. Compare to projections for edge
3. Don't use projections as sportsbook lines

**Total Effort: ~1-2 weeks for complete integration**

---

## Priority Order

1. **CRITICAL** (Today): Injury data + fix defensive rankings + load real odds
2. **HIGH** (This week): Snap extraction + depth charts + target share fix
3. **MEDIUM** (Next week): Weather integration + game script boost
4. **NICE-TO-HAVE** (Later): Vegas movement + contrarian signals

---

## Test to Verify Fixes

```python
# After fixing injury integration:
picks = generator.generate_picks("KC", "BUF")
for pick in picks:
    assert pick['player'] is not Out or Doubtful
    # (or confidence is low for Doubtful)

# After fixing defensive rankings:
def_rankings = generator._calculate_defensive_rankings()
assert def_rankings['BUF']['pass'] <= 8  # BUF actually has elite pass D
assert def_rankings['KC']['pass'] > 20   # KC has weak pass D

# After snap extraction:
snap_data = load_snap_features()
assert 'snap_pct' in snap_data
assert snap_data['player_1']['week_1']['snap_pct'] > 0

# After weather integration:
assert 'wind_mph' in feature_dict
assert 'temperature' in feature_dict
```

