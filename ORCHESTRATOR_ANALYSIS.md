# Game Outcome Orchestrator Analysis

**Date:** 2025-11-26
**Purpose:** Investigate if we should be using the game_outcome_orchestrator for better predictions

---

## Executive Summary

The game outcome orchestrator **exists and has excellent structure** (76+ features including EPA, market intelligence, and public betting data), BUT:

1. **Feature collection is BROKEN** - Looking for files that don't exist
2. **Using formulas, NOT ML models** - Despite having trained models available
3. **Performs WORSE than our baseline** - 29.0 MAE vs 5.6 MAE on test game
4. **The trained models use Vegas lines as inputs** - Not raw team stats

**Recommendation:** Fix the orchestrator's data loading, OR extract valuable features (EPA, market data) into our existing pipeline.

---

## Detailed Findings

### 1. Orchestrator Structure (EXCELLENT)

The orchestrator collects **76+ features** across 7 categories:

#### Team Stats
- `home_off_ppg`, `home_def_ppg`, `away_off_ppg`, `away_def_ppg`
- **EPA metrics** (Expected Points Added - the #1 NFL advanced stat):
  - `home_off_epa`, `home_def_epa`, `away_off_epa`, `away_def_epa`

#### Recent Form
- `home_l3_margin`, `away_l3_margin` (last 3 games margin)
- `home_l3_total`, `away_l3_total` (last 3 games total points)

#### Situational
- `rest_differential` (home rest days - away rest days)
- `is_division_game`, `is_primetime`, `is_dome`
- Weather: `temperature`, `wind_speed`, `precipitation`

#### Historical Matchup
- `h2h_home_margin_avg` (historical home team margin in this matchup)
- `h2h_total_avg` (historical total in this matchup)

#### Market Intelligence ⭐
- `opening_spread`, `current_spread` → **Line movement tracking**
- `opening_total`, `current_total` → **Line movement tracking**
- `line_movement_spread`, `line_movement_total`

#### Public Betting Data ⭐⭐
- `spread_bet_pct_home` (% of bets on home)
- `spread_money_pct_home` (% of money on home)
- `total_bet_pct_over`, `total_money_pct_over`
- `ml_bet_pct_home`, `ml_money_pct_home`

#### Sharp Money Indicators ⭐⭐⭐
- `spread_sharp_on_home`, `spread_sharp_on_away`
- `total_sharp_on_over`, `total_sharp_on_under`
- `spread_contrarian_home`, `spread_contrarian_away` (fade the public)
- `total_contrarian_over`, `total_contrarian_under`

**These are GOLD for betting markets!** Sharp money and contrarian indicators have proven edge.

### 2. Feature Collection (BROKEN)

The orchestrator looks for files that **don't exist**:

#### Expected Files
```
inputs/2023_team_stats_offense.csv  ❌ NOT FOUND
inputs/2023_team_stats_defense.csv  ❌ NOT FOUND
inputs/2023_schedule.parquet        ❌ NOT FOUND
```

#### Actual Files
```
inputs/schedules_2024_2025.csv              ✅ EXISTS (different format)
inputs/team_stats_2024_2025.csv             ✅ EXISTS (different format)
inputs/historical/games_2023.csv            ✅ EXISTS (different location)
inputs/historical/player_stats_2023_all.csv ✅ EXISTS (different location)
```

**Result:** All features return **0.0 or None** → terrible predictions

### 3. Test Results (BASELINE WINS)

Tested on CAR @ CHI, Week 10, 2023 (Actual: CAR 13, CHI 16, Total 29)

#### Orchestrator Prediction
- **Predicted Total:** 0.0 (Error: 29.0 points) ❌
- **Predicted Margin:** 2.5 (Error: 0.5 points) ✅
- **Why:** All team stats returned 0.0, so formula defaulted to minimal values

#### Simple Baseline (Recent Avg)
- **Predicted Total:** 34.6 (Error: 5.6 points) ✅
- **Predicted Margin:** 1.9 (Error: 1.1 points)
- **Why:** Successfully loaded historical games and calculated recent averages

**Winner:** Baseline by 23.4 points on total prediction

### 4. Prediction Method (FORMULAS, NOT ML)

Source: `game_outcome_orchestrator.py:562-565`

```python
# For now, use formula-based prediction (until models trained)
# This matches the current simple game_markets.py approach
predicted_margin, margin_std = self._predict_margin_formula(X, features)
predicted_total, total_std = self._predict_total_formula(X, features)
```

**The orchestrator is NOT using ML models**, despite models existing!

#### Total Formula (Lines 666-714)
```python
# Average of offense vs defense matchups
total = (
    (features.home_off_ppg + features.away_def_ppg) / 2 +
    (features.away_off_ppg + features.home_def_ppg) / 2
)

# Adjust for recent form (30% weight)
total = total * 0.7 + X['recent_total_avg'] * 0.3

# Adjust for historical matchup (15% weight)
total = total * 0.85 + X['h2h_total'] * 0.15

# Dome boost
if X['is_dome']:
    total += 2.0

# Division game (usually lower scoring)
if X['is_division_game']:
    total -= 1.5

# Sharp money adjustments
if features.total_sharp_on_over:
    total += 1.0
elif features.total_sharp_on_under:
    total -= 1.0

# Contrarian adjustments (fade heavy public)
if features.total_contrarian_under:
    total -= 0.7
elif features.total_contrarian_over:
    total += 0.7
```

**This is essentially our enhanced model**, but with:
- ✅ Sharp money indicators (we don't have)
- ✅ Public betting data (we don't have)
- ❌ Broken data loading (returns all zeros)

### 5. Trained Models (USE VEGAS LINES AS INPUTS)

Found in `outputs/models/`:

#### Vegas Markets
- `vegas/home_team_total_model.pkl` (199K)
- `vegas/away_team_total_model.pkl` (200K)
- `vegas/game_total_over_model.pkl` (1.3K)
- `vegas/home_covers_spread_model.pkl` (1.3K)

#### Derivative Markets
- `derivative/home_total_model.pkl` (37K)
- `derivative/away_total_model.pkl` (37K)
- `derivative/h1_total_model.pkl`, `h2_total_model.pkl`, `q1_total_model.pkl`

**Source:** `train_game_derivative_markets.py:89-100`

#### Model Features
```python
feature_cols = [
    'spread_line',           # Vegas spread (input!)
    'total_line',            # Vegas total (input!)
    'rest_advantage',
    'is_pickem',
    'is_high_total',
    'is_low_total',
    'div_game',
    'is_dome',
    'home_implied_total',    # Derived from Vegas lines
    'away_implied_total',    # Derived from Vegas lines
]
```

**These models are for DERIVATIVE markets** - They take Vegas lines as inputs to predict other markets (team totals, 1H totals, etc.)

**NOT useful for predicting game totals from scratch!**

---

## What We're Missing (Valuable Features)

### 1. EPA (Expected Points Added) ⭐⭐⭐
- **The #1 NFL advanced stat** for team performance
- Better than PPG because it measures quality, not just quantity
- Where to get: `nfl_data_py` has EPA data in play-by-play

### 2. Market Intelligence ⭐⭐
- **Line movement:** Opening spread/total → Current spread/total
- **Sharp money indicators:** When money % >> bet % (professional bettors)
- **Contrarian opportunities:** When public is heavily on one side (fade them)
- Where to get: The Odds API (requires API key), sports betting sites

### 3. Public Betting Percentages ⭐⭐
- **Bet %:** Percentage of bets on each side
- **Money %:** Percentage of money on each side
- **Discrepancies** indicate sharp vs public action
- Where to get: Action Network, The Athletic, ESPN

### 4. Rest Differential ⭐
- Already in orchestrator: `home_rest - away_rest`
- **Thursday games** (3 days rest) perform worse
- **Post-bye games** (14 days rest) perform better
- Easy to calculate from schedule

---

## Integration Plan

### Option 1: Fix Orchestrator Data Loading (HIGH EFFORT)

**Pros:**
- Get all 76+ features working
- Comprehensive ML-ready feature set
- Public betting data integration

**Cons:**
- Need to restructure data files or rewrite data loading
- Still using formulas, not ML (would need to train new models)
- Significant development time

**Effort:** 4-8 hours

### Option 2: Extract Valuable Features to Baseline (RECOMMENDED)

**Pros:**
- Quick wins with proven features
- Keep our working baseline
- Incremental improvement approach

**Cons:**
- Manual feature selection
- Doesn't get ALL orchestrator features

**Effort:** 1-2 hours

#### Implementation:
```python
# Add to our baseline backtest
def add_orchestrator_features(game, baseline_prediction):
    """Add valuable orchestrator features to baseline."""

    # 1. EPA (if available from nfl_data_py)
    home_off_epa = get_team_epa(game.home_team, game.season, game.week, 'offense')
    away_def_epa = get_team_epa(game.away_team, game.season, game.week, 'defense')

    if home_off_epa and away_def_epa:
        # EPA adjustment (better teams score more)
        epa_adj = (home_off_epa - away_def_epa) * 2.0  # 2 pts per EPA point
        baseline_prediction += epa_adj

    # 2. Rest differential
    rest_diff = get_rest_differential(game)
    if rest_diff >= 3:  # Home team well-rested
        baseline_prediction += 1.5
    elif rest_diff <= -3:  # Away team well-rested
        baseline_prediction -= 1.5

    # 3. Market data (if we add Odds API)
    sharp_on_over = check_sharp_money(game, 'total')
    if sharp_on_over:
        baseline_prediction += 1.0  # Follow sharp money

    return baseline_prediction
```

### Option 3: Train New ML Models with Orchestrator Features (LONG-TERM)

**Pros:**
- Best accuracy potential
- Learn optimal feature weights
- Scalable to new features

**Cons:**
- Requires comprehensive feature engineering
- Need to fix data loading first
- Training and validation time

**Effort:** 8-16 hours

---

## Recommendations (Prioritized)

### Immediate (Now)
1. ✅ **Add EPA to baseline** (1 hour)
   - Load from `nfl_data_py` play-by-play data
   - Test on backtests to measure improvement
   - Expected gain: +3-5% accuracy

2. ✅ **Add rest differential** (30 min)
   - Already have schedule data
   - Simple calculation
   - Expected gain: +1-2% accuracy

### Short-term (This week)
3. ⏳ **Integrate Odds API for market data** (2 hours)
   - Get API key (free tier: 500 requests/month)
   - Add line movement tracking
   - Add public betting percentages
   - Expected gain: +2-4% accuracy for betting recommendations

### Medium-term (Next sprint)
4. ⏳ **Fix orchestrator data loading** (4 hours)
   - Point to correct file paths
   - Or convert data to orchestrator's expected format
   - Enables comprehensive feature testing

5. ⏳ **Train ML model with all features** (8 hours)
   - Use fixed orchestrator features
   - Gradient boosting or neural network
   - Cross-validation on 2021-2023 data
   - Expected gain: +5-10% accuracy

---

## Conclusion

**Yes, we SHOULD be using orchestrator features**, but:
- The orchestrator itself is currently broken (data loading)
- The trained models are for derivative markets (use Vegas lines as inputs)
- Our baseline is already better than the broken orchestrator

**Best path forward:**
1. Extract valuable features (EPA, rest differential) to our working baseline
2. Add market intelligence if pursuing betting recommendations
3. Consider fixing orchestrator for comprehensive ML training later

**Biggest missed opportunity:** EPA data - it's the best NFL stat and we're not using it!
