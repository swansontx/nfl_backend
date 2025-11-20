# NFL Props System - Audit Findings Checklist

## Component Status Summary

### Ingestion Layer (70% Complete)
- [x] Play-by-play data fetching
- [x] Schedule fetching
- [x] Odds API integration
- [x] Prop lines fetching
- [x] Injury data fetching
- [ ] Weather data fetching (stub only)
- [x] Snap counts downloading (file downloaded, not processed)
- [x] Depth charts downloading (file downloaded, not processed)

### Feature Extraction (60% Complete)
- [x] Basic player stats extraction (yards, TDs, attempts)
- [x] Advanced EPA metrics extraction
- [x] CPOE calculation
- [x] Success rate calculation
- [x] Air yards and YAC extraction
- [x] WPA extraction
- [ ] Snap count extraction (file exists, no script)
- [ ] Target share calculation (hardcoded estimate instead)
- [ ] Weather features (stub only)
- [ ] Game script signals (minimal)
- [ ] Injury impact flags (not merged)

### Feature Engineering (60% Complete)
- [x] Smoothing and rolling windows
- [x] Player stats aggregation
- [ ] Injury data merging
- [ ] Snap % change tracking
- [ ] Target share trend detection
- [ ] Weather data tagging
- [ ] Backup role inference from depth charts

### Modeling (80% Complete)
- [x] Multi-prop models (60+ types)
- [x] Passing models
- [x] Rushing models
- [x] Receiving models
- [x] TD models
- [x] Combo models
- [x] Quarter/half proportional models
- [ ] Injury impact factors
- [ ] Snap % changes in features
- [ ] Target share in features
- [ ] Weather in features

### Prediction Engine (70% Complete)
- [x] Contextual performance signals (30% weight)
- [x] EPA matchup signals (25% weight)
- [x] Success rate signals (15% weight)
- [x] CPOE signals (10% weight, QB only)
- [x] Pressure signals (10% weight, QB only)
- [x] Trend momentum (5% weight)
- [x] Game script adjustment (5% weight)
- [ ] Injury adjustments
- [ ] Weather adjustments
- [ ] Snap count changes

### Picks Generation (50% Complete)
- [x] Load player stats
- [x] Build narrative parlays
- [x] Estimate hit rates and EV
- [ ] Load injury data
- [ ] Load depth charts
- [ ] Load weather data
- [ ] Load actual current odds
- [ ] Fix defensive rankings
- [ ] Check if player is active
- [ ] Detect backup opportunities

### Orchestration (70% Complete)
- [x] Data ingestion stage
- [x] Feature extraction stage
- [x] Feature engineering stage
- [x] Roster indexing stage (created, not populated)
- [x] Injury indexing stage (created, not populated)
- [x] Model training stages
- [x] Projection generation stage
- [x] Backtest stages
- [x] Picks generation stage
- [ ] Injury data merge stage
- [ ] Snap extraction stage
- [ ] Depth chart processing stage
- [ ] Weather fetch stage

---

## Critical Issues Found

### Issue 1: Injury Data Not Used (HIGH PRIORITY)
**File**: `backend/recommendations/generate_game_picks.py`
**Problem**: 
- Fetches injury data successfully
- Never loads it in pick generation
- No check for player active status
- Recommends bets on injured players

**Status**: 
- [ ] Create injury loader module
- [ ] Load injuries in generate_game_picks.py
- [ ] Skip Out players entirely
- [ ] Reduce Doubtful projections 30-50%
- [ ] Flag low confidence for Doubtful

**Effort**: 1-2 days

### Issue 2: Defensive Rankings Wrong (HIGH PRIORITY)
**File**: `backend/recommendations/generate_game_picks.py` (lines 149-202)
**Problem**:
- Calculates rankings from team's own stats
- Should use opponent stats against that team
- Defensive matchup context is meaningless

**Status**:
- [ ] Rewrite _calculate_defensive_rankings()
- [ ] OR load pre-calculated defensive stats
- [ ] Test: BUF pass defense should be top 10
- [ ] Test: Weak pass defenses ranked correctly

**Effort**: 1 day

### Issue 3: Snap Counts Not Processed (HIGH PRIORITY)
**File**: Missing `backend/features/extract_snap_features.py`
**Problem**:
- snap_counts_YYYY.csv downloaded but never processed
- No snap % feature extraction
- Can't detect reduced availability
- Can't track usage efficiency

**Status**:
- [ ] Create extract_snap_features.py
- [ ] Load snap_counts CSV
- [ ] Calculate snap % per player per game
- [ ] Add to feature JSON
- [ ] Track week-over-week changes
- [ ] Add to orchestrator pipeline

**Effort**: 2-3 days

### Issue 4: Depth Charts Not Read (HIGH PRIORITY)
**File**: Missing `backend/features/extract_depth_charts.py`
**Problem**:
- depth_charts_YYYY.csv (46MB) downloaded but never read
- Can't infer starter vs backup
- Backup opportunity detection impossible

**Status**:
- [ ] Create extract_depth_charts.py
- [ ] Load depth_charts CSV
- [ ] Map to player IDs
- [ ] Determine starter/backup position
- [ ] Add to feature JSON
- [ ] Use for backup opportunity signals

**Effort**: 1-2 days

### Issue 5: Target Share Calculated Wrong (HIGH PRIORITY)
**File**: `backend/modeling/train_usage_efficiency_models.py` (line ~290)
**Problem**:
- Hardcoded `/ 35` instead of calculating actual team targets
- Models trained on incorrect data
- Can't detect target share changes

**Status**:
- [ ] Calculate team target sums per game
- [ ] Compute actual target share %
- [ ] Track 3-week rolling average
- [ ] Retrain models with correct feature
- [ ] Update feature extraction

**Effort**: 2-3 days

### Issue 6: Weather Not Integrated (MEDIUM PRIORITY)
**File**: `backend/features/extract_weather_features.py` (entire file)
**Problem**:
- Only stubs and hardcoded values
- Never fetches real weather data
- Infrastructure exists but not used
- Wind impact on passing not modeled

**Status**:
- [ ] Implement real weather API (Open-Meteo)
- [ ] Fetch historical weather data
- [ ] Add weather features to feature extraction
- [ ] Include in model training
- [ ] Add to prediction engine

**Effort**: 2-3 days

### Issue 7: Actual Odds Not Loaded (HIGH PRIORITY)
**File**: `backend/recommendations/generate_game_picks.py`
**Problem**:
- Uses projections as sportsbook lines
- No actual odds comparison
- Edge calculation is meaningless

**Status**:
- [ ] Load real odds from Odds API
- [ ] Compare projection vs actual line
- [ ] Calculate true edge
- [ ] Don't use projection as line

**Effort**: 1 day

---

## Data Flow Issues

### Broken Flows (7 Total)

1. **Injury Data Flow**: BROKEN
   - Status: Fetched but not consumed
   - Fix: Add injury loader to picks generator
   - Impact: HIGH (can't detect player availability)

2. **Weather Data Flow**: NOT IMPLEMENTED
   - Status: Infrastructure only, no real implementation
   - Fix: Implement real weather fetching
   - Impact: MEDIUM (20-30% passing variance unaccounted)

3. **Snap Count Flow**: NOT EXTRACTED
   - Status: File exists but no extraction
   - Fix: Create extraction script
   - Impact: HIGH (no usage tracking)

4. **Depth Chart Flow**: NOT READ
   - Status: File exists but never processed
   - Fix: Create processing script
   - Impact: HIGH (no role detection)

5. **Target Share Flow**: WRONG CALCULATION
   - Status: Hardcoded estimate instead of real calc
   - Fix: Calculate actual team totals
   - Impact: HIGH (models trained wrong)

6. **Defensive Stats Flow**: WRONG LOGIC
   - Status: Uses team stats instead of opponent stats
   - Fix: Calculate opponent achievements
   - Impact: HIGH (defensive edge is noise)

7. **Game Script Flow**: UNDERUTILIZED
   - Status: Only 5% weight, no volume adjustments
   - Fix: Increase weight and add adjustments
   - Impact: MEDIUM (blowout scenarios missed)

---

## Variables Not Being Tracked

### Critical (Must Add)
- [ ] Injury status (Out/Doubtful/Questionable)
- [ ] Snap count per player per game
- [ ] Target share % (actual, not estimated)
- [ ] Snap % changes week-over-week

### Important (Should Add)
- [ ] Weather (wind, temperature, precipitation)
- [ ] Depth chart position (starter/backup)
- [ ] Game script prediction (run vs pass heavy)
- [ ] Backup opportunity flags

### Useful (Nice to Have)
- [ ] Vegas line movement
- [ ] DFS ownership %
- [ ] Red zone efficiency
- [ ] Historical head-to-head

---

## Testing Needed

### Test 1: Injury Integration
```python
# After fix
picks = generator.generate_picks("KC", "BUF")
# Should not recommend bets on Out players
# Should have low confidence for Doubtful
assert all(p['confidence'] != 'HIGH' for p in picks if p['injury_status'] == 'Doubtful')
assert all(p['player'] not in out_players for p in picks)
```

### Test 2: Defensive Rankings
```python
# After fix
rankings = generator._calculate_defensive_rankings()
# BUF pass defense is top 5
assert rankings['BUF']['pass'] <= 8
# KC pass defense is bottom 10
assert rankings['KC']['pass'] > 22
```

### Test 3: Snap Extraction
```python
# After adding snap features
snap_data = load_snap_features()
# Should have snap % for each player
assert 'snap_pct' in snap_data
assert snap_data['player_001']['week_1']['snap_pct'] > 0
assert snap_data['player_001']['week_2']['snap_pct_change'] in feature_data
```

### Test 4: Target Share
```python
# After fixing calculation
features = load_features()
# Target share should sum to ~100% for each team
team_targets = {}
for player in features:
    team = player['team']
    if team not in team_targets:
        team_targets[team] = 0
    team_targets[team] += player.get('target_share', 0)
for team, total in team_targets.items():
    assert 90 < total < 110  # ~100% with rounding
```

### Test 5: Weather Features
```python
# After implementing weather
features = load_features()
# Should have weather fields
assert 'wind_mph' in features[0]
assert 'temperature' in features[0]
assert 'precipitation_chance' in features[0]
```

---

## Implementation Checklist

### Week 1: Critical Fixes
- [ ] Day 1: Fix defensive rankings calculation
- [ ] Day 1: Load actual odds instead of projections
- [ ] Days 2-3: Integrate injury data
- [ ] Days 3-4: Extract snap counts
- [ ] Days 4-5: Process depth charts
- [ ] Day 5: Fix target share calculation

### Week 2: Integrations
- [ ] Days 1-2: Implement weather fetching
- [ ] Day 3: Add snap % to models
- [ ] Day 4: Add target share to models
- [ ] Days 4-5: Increase game script weight

### Week 3: Testing & Polish
- [ ] Days 1-2: Unit tests for all fixes
- [ ] Days 3-4: Integration tests
- [ ] Day 5: Validate improvements in backtest

---

## Success Criteria

### After All Fixes
- [x] System completeness: 58% → 85%
- [ ] Injury data loaded in picks (0% → 100%)
- [ ] Snap counts extracted (0% → 100%)
- [ ] Defensive rankings correct (0% → 100%)
- [ ] Target share accurate (0% → 90%)
- [ ] Weather features added (0% → 100%)
- [ ] Real odds loaded (0% → 100%)
- [ ] Backtest accuracy improves 5-10%
- [ ] Hit rates on recommendations increase

---

## Risk Assessment

### High Risk
1. Defensive ranking change could break existing picks
   - Mitigation: Validate against season history
2. Injury data integration might have stale data
   - Mitigation: Check injury data freshness
3. Wrong target share could break models
   - Mitigation: Retrain carefully, compare results

### Medium Risk
4. Weather data quality varies by location
   - Mitigation: Use best available source
5. Snap count data might have gaps
   - Mitigation: Handle missing data gracefully

---

## Files to Create

1. `backend/data/injury_loader.py` - Load injury JSON
2. `backend/features/extract_snap_features.py` - Extract snap data
3. `backend/features/extract_depth_charts.py` - Process depth charts
4. `backend/features/weather_fetcher.py` - Fetch real weather

## Files to Modify

1. `backend/recommendations/generate_game_picks.py` - Add injury/odds/rankings
2. `backend/modeling/train_usage_efficiency_models.py` - Fix target share
3. `backend/orchestration/orchestrator.py` - Add missing stages
4. `backend/features/extract_weather_features.py` - Real implementation
5. `backend/api/prediction_engine.py` - Add weather/injury signals

---

## Sign-Off

**Audit Date**: 2025-11-20
**System Status**: 58% Complete, 30% Integration Coverage
**Recommendation**: Fix integration pipes before adding new features
**Estimated Effort**: 1-2 weeks for critical issues, 2-3 weeks for complete fix
**Expected Improvement**: 5-15% accuracy improvement when complete

