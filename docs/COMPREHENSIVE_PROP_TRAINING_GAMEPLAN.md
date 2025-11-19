# Comprehensive Prop Training Gameplan - All DraftKings Markets

## Current Status

### ✅ Props Currently Trained
- **Pass Yards** (pass_yards) - 40-50% hit rate
- **Rush Yards** (rush_yards) - 75% hit rate Week 11
- **Rec Yards** (rec_yards) - 80% hit rate Week 10

### ❌ Props NOT Yet Trained
- Pass TDs, Completions, Pass Attempts, Interceptions
- Rush TDs, Rush Attempts
- Rec TDs, Receptions, Targets
- Anytime TD Scorer
- First TD Scorer
- Kicker props (FG Made, XP Made)
- Combo props (Pass+Rush Yards, Rec+Rush Yards)

### 🔴 Critical Missing Infrastructure
- **Injury/Active Status Tracking** - NO IMPLEMENTATION YET
- **Snap Count Data** - Not integrated
- **Depth Chart Data** - Not integrated
- **Weather Data** - Not integrated
- **Opponent Defense Stats** - Not integrated

---

## PRIORITY 1: Injury & Active Status Integration

### Why Critical
Cannot make prop predictions on inactive/injured players. This is the #1 cause of bad bets.

### Data Sources
1. **NFLverse Weekly Rosters** - `https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{year}.csv`
   - Contains: `status` column (ACT, INACT, IR, PUP, etc.)

2. **NFLverse Injuries** - `https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{year}.csv`
   - Contains: injury reports, practice status

### Implementation Plan
```python
# backend/analysis/fetch_injury_data.py
def fetch_weekly_injury_report(year, week):
    """Fetch injury report for a specific week."""
    # 1. Download rosters for the year
    # 2. Filter to the specific week
    # 3. Identify players: OUT, DOUBTFUL, QUESTIONABLE, PROBABLE
    # 4. Return dict: {player_id: status}

def filter_predictions_by_active_status(predictions_df, week):
    """Remove predictions for inactive players."""
    # 1. Fetch injury report for week
    # 2. Remove any player with status != ACT
    # 3. Log removed players
```

### Required Before ANY Prop Predictions
- ✅ Fetch weekly roster status
- ✅ Filter out OUT/DOUBTFUL players
- ✅ Add confidence penalty for QUESTIONABLE players (reduce edge by 50%)

---

## Prop Training Gameplans by Category

## Category 1: Passing Props

### 1.1 Pass Yards ✅ TRAINED
**Status:** Models trained, 40-50% hit rate
**Current Features:**
- season_avg_pass_yards
- l3_avg_pass_yards
- games_played

**Improvements Needed:**
- Add opponent pass defense EPA/play
- Add weather (wind, dome)
- Add home/away
- Add implied team total (game script)
- Add QB pressure rate faced

**Model:** Quantile Regression (Q10, Q25, Q50, Q75, Q90)

### 1.2 Pass TDs ❌ NOT TRAINED
**Target Variable:** `passing_tds`
**Data Available:** ✅ Yes (in synthetic data)

**Features Needed:**
- season_avg_pass_tds
- l3_avg_pass_tds
- games_played
- red_zone_attempts_per_game
- opponent_red_zone_td_rate_allowed
- team_implied_total (high total = more TDs)
- is_dome

**Model Type:** **Poisson Regression** (TDs are count data)
- Use `statsmodels.discrete.discrete_model.Poisson`
- Or quantile regression for P(TDs > 0.5), P(TDs > 1.5), etc.

**Typical Lines:** 0.5 TDs (will he throw 1+), 1.5 TDs (will he throw 2+), 2.5 TDs

**Training Priority:** HIGH (very popular prop)

### 1.3 Completions ❌ NOT TRAINED
**Target Variable:** `completions`
**Data Available:** ✅ Yes

**Features Needed:**
- season_avg_completions
- l3_avg_completions
- completion_percentage (season)
- attempts_per_game
- opponent_completion_percentage_allowed
- is_high_total (more pass attempts)

**Model Type:** Quantile Regression

**Typical Lines:** 18.5, 22.5, 25.5 completions

**Training Priority:** MEDIUM

### 1.4 Pass Attempts ❌ NOT TRAINED
**Target Variable:** `attempts`
**Data Available:** ✅ Yes

**Features Needed:**
- season_avg_attempts
- l3_avg_attempts
- team_pass_rate (need to calculate from historical data)
- game_script_expectation (spread < -7 = fewer attempts)
- is_high_total
- pace_of_play

**Model Type:** Quantile Regression

**Typical Lines:** 30.5, 35.5, 38.5 attempts

**Training Priority:** LOW (less popular)

### 1.5 Interceptions ❌ NOT TRAINED
**Target Variable:** `interceptions` (need to add to synthetic data!)
**Data Available:** ❌ NO - need to add

**Features Needed:**
- season_int_rate
- l3_int_rate
- opponent_int_rate
- pressure_rate_faced

**Model Type:** **Bernoulli** (0 vs 1+) or **Poisson**

**Typical Lines:** 0.5 INTs (will he throw one?)

**Training Priority:** LOW (but easy to add)

---

## Category 2: Rushing Props

### 2.1 Rush Yards ✅ TRAINED
**Status:** 75% hit rate Week 11, +42.5% ROI
**Current Features:**
- season_avg_rush_yards
- l3_avg_rush_yards
- games_played

**Improvements Needed:**
- opponent_run_defense_epa
- game_script (winning = more rushing)
- is_high_total (>48 = less rushing)
- carries_per_game
- yards_per_carry

**Model:** Quantile Regression

### 2.2 Rush TDs ❌ NOT TRAINED
**Target Variable:** `rushing_tds`
**Data Available:** ✅ Yes

**Features Needed:**
- season_avg_rush_tds
- red_zone_carries_per_game
- goal_line_carry_share (need snap count data)
- opponent_goal_line_td_rate_allowed
- team_implied_total

**Model Type:** **Poisson** or **Bernoulli**

**Typical Lines:** 0.5 TDs

**Training Priority:** HIGH (very popular)

### 2.3 Rush Attempts ❌ NOT TRAINED
**Target Variable:** `carries`
**Data Available:** ✅ Yes

**Features Needed:**
- season_avg_carries
- l3_avg_carries
- snap_share (need snap count data)
- game_script_expectation
- is_goal_line_back

**Model Type:** Quantile Regression

**Typical Lines:** 15.5, 18.5, 22.5 carries

**Training Priority:** MEDIUM

---

## Category 3: Receiving Props

### 3.1 Rec Yards ✅ TRAINED
**Status:** 80% hit rate Week 10, +52% ROI
**Current Features:**
- season_avg_rec_yards
- l3_avg_rec_yards
- games_played

**Improvements Needed:**
- target_share
- air_yards_share
- opponent_pass_defense_vs_position (vs WR, vs TE, vs RB)
- qb_rating_when_targeting
- is_slot_wr vs outside_wr

**Model:** Quantile Regression

### 3.2 Receptions ❌ NOT TRAINED
**Target Variable:** `receptions`
**Data Available:** ✅ Yes

**Features Needed:**
- season_avg_receptions
- l3_avg_receptions
- target_share
- catch_rate (receptions / targets)
- team_pass_rate
- ppr_expectation

**Model Type:** Quantile Regression

**Typical Lines:** 3.5, 5.5, 7.5 receptions

**Training Priority:** HIGH (very popular)

### 3.3 Rec TDs ❌ NOT TRAINED
**Target Variable:** `receiving_tds`
**Data Available:** ✅ Yes

**Features Needed:**
- season_avg_rec_tds
- red_zone_target_share
- end_zone_targets_per_game
- td_rate
- opponent_red_zone_td_rate_allowed

**Model Type:** **Poisson** or **Bernoulli**

**Typical Lines:** 0.5 TDs

**Training Priority:** HIGH

### 3.4 Targets ❌ NOT TRAINED
**Target Variable:** `targets`
**Data Available:** ✅ Yes

**Features Needed:**
- season_avg_targets
- l3_avg_targets
- target_share
- team_pass_attempts
- snap_share

**Model Type:** Quantile Regression

**Typical Lines:** 5.5, 7.5, 9.5 targets

**Training Priority:** MEDIUM

---

## Category 4: Touchdown Scorer Props

### 4.1 Anytime TD ❌ NOT TRAINED
**Target Variable:** `scored_any_td` (binary: did they score?)
**Data Available:** ❌ Need to calculate (rushing_tds + receiving_tds > 0)

**Features Needed:**
- season_td_rate (TDs per game)
- red_zone_touches_per_game
- goal_line_role (need snap data)
- opponent_red_zone_defense
- team_implied_total

**Model Type:** **Bernoulli Classification**
- Predict P(scores TD)
- Compare to DK odds (convert -150 to implied prob)

**Typical Odds:** -150 to +300 depending on player

**Training Priority:** VERY HIGH (extremely popular bet)

### 4.2 First TD ❌ NOT TRAINED
**Target Variable:** `is_first_td_scorer` (binary)
**Data Available:** ❌ Need to parse game logs

**Features Needed:**
- team_scores_first_rate
- player_td_rate
- opening_drive_usage
- goal_line_role

**Model Type:** **Multinomial Classification** (all players compete)

**Typical Odds:** +800 to +2000

**Training Priority:** LOW (harder to model, lower volume)

---

## Category 5: Kicker Props

### 5.1 FG Made ❌ NOT TRAINED
**Target Variable:** `fg_made`
**Data Available:** ❌ Not in current data (need kicker stats)

**Features Needed:**
- kicker_fg_percentage
- team_red_zone_failure_rate (stalls = FGs)
- weather (wind, temp)
- is_dome

**Model Type:** Poisson

**Typical Lines:** 1.5 FGs

**Training Priority:** MEDIUM (need kicker data source)

### 5.2 XP Made ❌ NOT TRAINED
**Target Variable:** `xp_made`
**Data Available:** ❌ Not in current data

**Features Needed:**
- team_tds_per_game
- red_zone_efficiency

**Model Type:** Poisson

**Typical Lines:** 2.5 XPs

**Training Priority:** LOW

---

## Category 6: Combo Props

### 6.1 Pass + Rush Yards ❌ NOT TRAINED
**Target Variable:** `passing_yards + rushing_yards`
**Data Available:** ✅ Can calculate from existing data

**Features Needed:**
- Combine pass and rush features
- QB_rushing_yards_per_game (Lamar, Hurts, Allen)

**Model Type:** Quantile Regression on combined total

**Typical Lines:** 280.5 for mobile QBs

**Training Priority:** MEDIUM

### 6.2 Rec + Rush Yards ❌ NOT TRAINED
**Target Variable:** `receiving_yards + rushing_yards`
**Data Available:** ✅ Can calculate

**Features Needed:**
- Combine rec and rush features
- Multi-purpose back indicator

**Model Type:** Quantile Regression

**Typical Lines:** 100.5 for pass-catching RBs

**Training Priority:** LOW

---

## Implementation Roadmap

### Phase 1: Foundation (WEEK 1)
**Goal:** Get injury tracking working, prevent bad bets

1. ✅ Create `backend/analysis/fetch_injury_data.py`
2. ✅ Create `backend/analysis/filter_active_players.py`
3. ✅ Integrate into prop prediction pipeline
4. ✅ Test on Week 12 (exclude injured players)

**Success Metric:** 0 bets on inactive players

### Phase 2: High-Value Props (WEEK 2)
**Goal:** Train the most popular/profitable prop types

1. ✅ Pass TDs (Poisson model)
2. ✅ Rush TDs (Poisson model)
3. ✅ Rec TDs (Poisson model)
4. ✅ Anytime TD Scorer (Bernoulli model)
5. ✅ Receptions (Quantile model)

**Success Metric:** 5 additional prop types at >55% hit rate

### Phase 3: Volume Props (WEEK 3)
**Goal:** Add volume-based props

1. ✅ Completions
2. ✅ Pass Attempts
3. ✅ Rush Attempts
4. ✅ Targets

**Success Metric:** 9 total prop types trained

### Phase 4: Enhanced Features (WEEK 4)
**Goal:** Improve existing models with advanced features

1. ✅ Add opponent defensive stats (EPA allowed)
2. ✅ Add weather data (wind, dome, temp)
3. ✅ Add snap count data
4. ✅ Add target share / carry share
5. ✅ Add game script features

**Success Metric:** 5-10% improvement in hit rates

### Phase 5: Specialty Props (WEEK 5)
**Goal:** Round out coverage

1. ✅ Kicker props (FG Made, XP Made)
2. ✅ Combo props (Pass+Rush, Rec+Rush)
3. ✅ Interceptions
4. ⏸️ First TD (deprioritize - low volume)

**Success Metric:** Complete DraftKings market coverage

---

## Testing Requirements

### Unit Tests
- [ ] Test injury filtering (should remove OUT players)
- [ ] Test quantile probability calculation
- [ ] Test Poisson probability calculation
- [ ] Test feature engineering (no nulls, correct dtypes)

### Integration Tests
- [ ] End-to-end: raw data → features → predictions → evaluation
- [ ] Backtest framework for each prop type
- [ ] ROI calculation validation

### Validation Tests
Each prop type needs:
- [ ] **Calibration curve** - Are 60% predictions actually hitting 60%?
- [ ] **Hit rate by week** - Weeks 10, 11, 12 backtests
- [ ] **ROI by week**
- [ ] **Feature importance** - Which features matter most?
- [ ] **Edge distribution** - What edge thresholds work best?

### Quality Filters
For each prop prediction, verify:
- [ ] Player is ACTIVE (not OUT/DOUBTFUL)
- [ ] Player has 3+ games played (sample size)
- [ ] Edge > 5% (minimum threshold)
- [ ] Model confidence > 60%
- [ ] Usage stability (CV < 40%)

---

## File Structure

```
backend/
  analysis/
    # Data fetching
    fetch_injury_data.py              ✅ CREATE
    fetch_snap_counts.py              ✅ CREATE
    fetch_opponent_defense_stats.py   ✅ CREATE
    fetch_weather_data.py             ✅ CREATE

    # Feature engineering
    engineer_passing_features.py      ✅ CREATE
    engineer_rushing_features.py      ✅ CREATE
    engineer_receiving_features.py    ✅ CREATE
    engineer_td_features.py           ✅ CREATE

    # Model training
    train_passing_props.py            ✅ CREATE (pass_tds, completions, attempts)
    train_rushing_props.py            ✅ CREATE (rush_tds, rush_attempts)
    train_receiving_props.py          ✅ CREATE (rec_tds, receptions, targets)
    train_td_scorer_model.py          ✅ CREATE (anytime TD)
    train_kicker_props.py             ✅ CREATE

    # Modified existing
    train_all_prop_models.py          ✅ EXPAND (add all prop types)

  models/
    prop_models/
      pass_yards/
      pass_tds/
      rush_yards/
      rush_tds/
      rec_yards/
      rec_tds/
      anytime_td/
      ...

  tests/
    test_injury_filtering.py          ✅ CREATE
    test_prop_models.py               ✅ CREATE
    test_backtesting.py               ✅ CREATE
```

---

## Success Metrics

### By End of Phase 2 (2 weeks)
- ✅ 8 prop types trained (3 yards + 5 new)
- ✅ Injury filtering integrated (0 bets on OUT players)
- ✅ >55% hit rate average across all prop types
- ✅ >10% ROI average

### By End of Phase 4 (4 weeks)
- ✅ 13+ prop types trained
- ✅ >60% hit rate average
- ✅ >20% ROI average
- ✅ Opponent defense, weather, snap counts integrated

### Production Ready (6 weeks)
- ✅ 15+ prop types
- ✅ Full DraftKings market coverage
- ✅ Calibrated probabilities (Brier score < 0.20)
- ✅ Automated weekly backtesting
- ✅ API integration complete

---

## Next Immediate Actions

1. **Create injury filtering module** (CRITICAL)
2. **Expand synthetic data to include interceptions**
3. **Train TD models** (pass_tds, rush_tds, rec_tds)
4. **Train anytime TD model**
5. **Create comprehensive backtesting suite**
6. **Add opponent defense stats**
7. **Add weather data integration**

---

## Notes

- **Position filtering is CRITICAL** - Learned this the hard way (0% hit rate without it)
- **L3 average > season average** - Recency matters more (47-71% feature importance)
- **Edge threshold matters** - Need >5% edge to overcome juice
- **Sample size matters** - Require 3+ games before betting
- **Variance is high** - Even 60% models will have losing weeks
