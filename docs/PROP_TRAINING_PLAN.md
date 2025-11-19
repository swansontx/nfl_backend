# Prop Model Training Plan - DraftKings Lines

## Overview
Train models on actual DraftKings prop lines + historical player performance to find +EV props.

## Data Sources

### 1. Historical Prop Lines (Odds API - DraftKings)
```python
# Fetch from: https://the-odds-api.com/
# Endpoint: /v4/sports/americanfootball_nfl/odds/

{
  "game_id": "2025_10_BUF_MIA",
  "player": "Josh Allen",
  "prop_type": "pass_yards",
  "line": 265.5,
  "over_price": -110,
  "under_price": -110,
  "timestamp": "2025-11-08T10:00:00Z"  # Pre-game
}
```

### 2. Actual Player Stats (NFLverse)
```python
# From play-by-play data
{
  "game_id": "2025_10_BUF_MIA",
  "player": "Josh Allen",
  "pass_yards": 189,
  "rush_yards": 24,
  "pass_tds": 1
}
```

### 3. Game Context Features
- Opponent defensive rankings
- Vegas spread/total
- Weather (dome, wind, temp)
- Rest days
- Division game
- Home/away
- Time of season

## Training Pipeline

### Step 1: Fetch Historical Props (Weeks 1-9)
```bash
# For each week:
for week in 1..9:
  fetch_props_from_odds_api(
    sport='americanfootball_nfl',
    market='player_props',
    bookmaker='draftkings',
    week=week
  )
```

### Step 2: Match Props to Actual Results
```python
props_df = pd.DataFrame({
  'game_id': [...],
  'player': [...],
  'prop_type': ['pass_yards', 'rush_yards', ...],
  'dk_line': [265.5, 75.5, ...],
  'actual_result': [189, 72, ...],
  'hit_over': [False, False, ...]  # Target variable
})
```

### Step 3: Engineer Features
```python
features = {
  # Player features (from historical data through week-1)
  'player_season_avg': calculate_ytd_average(player, prop_type, through_week-1),
  'player_last_3_avg': calculate_recent_average(player, prop_type, last_n=3),
  'player_usage_rate': calculate_usage_rate(player),
  'player_target_share': calculate_target_share(player),  # For WRs

  # Opponent features
  'opp_def_rank_vs_position': get_defensive_rank(opponent, position),
  'opp_def_epa_allowed': get_epa_allowed(opponent, stat_type),
  'opp_pass_rush_rate': get_pressure_rate(opponent),

  # Game context
  'game_total': vegas_total,
  'team_implied_total': implied_total,
  'is_home': is_home_game,
  'rest_days': days_since_last_game,
  'is_division_game': is_division,
  'is_dome': is_dome,
  'wind_speed': wind_mph,

  # Market context
  'line_movement': dk_line - opening_line,
  'prop_type': prop_type,  # Categorical
}
```

### Step 4: Train Quantile Models
```python
from sklearn.ensemble import GradientBoostingRegressor

# For each prop type
for prop_type in ['pass_yards', 'rush_yards', 'rec_yards', 'pass_tds']:

    # Train quantile models (10th, 50th, 90th percentiles)
    model_q10 = GradientBoostingRegressor(
        loss='quantile',
        alpha=0.10,
        n_estimators=200,
        max_depth=5
    )

    model_q50 = GradientBoostingRegressor(
        loss='quantile',
        alpha=0.50,
        n_estimators=200,
        max_depth=5
    )

    model_q90 = GradientBoostingRegressor(
        loss='quantile',
        alpha=0.90,
        n_estimators=200,
        max_depth=5
    )

    # Train on Weeks 1-9 data
    model_q10.fit(X_train, y_train)
    model_q50.fit(X_train, y_train)
    model_q90.fit(X_train, y_train)
```

### Step 5: Calculate Edge
```python
def calculate_edge(player, prop_type, dk_line):
    """Calculate edge vs DraftKings line."""

    # Get our prediction distribution
    q10 = model_q10.predict(features)
    q50 = model_q50.predict(features)
    q90 = model_q90.predict(features)

    # Calculate P(X > line) from distribution
    prob_over = calculate_prob_from_quantiles(q10, q50, q90, line=dk_line)

    # DK's implied probability (from -110 odds)
    dk_implied_prob = 0.524  # -110 = 52.4%

    # Our edge
    edge = prob_over - dk_implied_prob

    # Expected value
    if prob_over > dk_implied_prob:
        ev = (prob_over * 0.91) - ((1 - prob_over) * 1.0)
        side = "OVER"
    else:
        ev = ((1 - prob_over) * 0.91) - (prob_over * 1.0)
        side = "UNDER"

    return {
        'prob': prob_over if side == "OVER" else 1 - prob_over,
        'edge': abs(edge),
        'ev': ev,
        'side': side
    }
```

### Step 6: Apply Filters
```python
def should_bet_prop(prop, edge_calc, quality_metrics):
    """6-layer quality filter."""

    # Layer 1: Minimum edge
    if edge_calc['edge'] < 0.05:
        return False, "Edge too small"

    # Layer 2: Minimum EV
    if edge_calc['ev'] < 0.02:
        return False, "EV too small"

    # Layer 3: Sample size
    if quality_metrics['games_played'] < 5:
        return False, "Insufficient sample"

    # Layer 4: Usage stability
    if quality_metrics['usage_cv'] > 0.40:
        return False, "High usage variance"

    # Layer 5: Model confidence
    if edge_calc['prob'] < 0.60:
        return False, "Low confidence"

    # Layer 6: Game script filter
    if is_high_total and prop_type == 'rush_yards':
        # High-scoring games = less rushing
        edge_calc['confidence'] *= 0.8

    return True, "PASS"
```

## Backtesting Process

### Week 10 Test
```python
# 1. Load DK prop lines from Nov 8, 2025 (pre-game)
dk_props_week10 = load_dk_props(week=10, date='2025-11-08')

# 2. Generate features using ONLY data through Week 9
features_week10 = generate_features(
    props=dk_props_week10,
    historical_data_through_week=9
)

# 3. Make predictions
predictions = []
for prop in dk_props_week10:
    edge = calculate_edge(prop.player, prop.prop_type, prop.line)

    if should_bet_prop(prop, edge, quality_metrics):
        predictions.append({
            'player': prop.player,
            'prop_type': prop.prop_type,
            'side': edge['side'],
            'line': prop.line,
            'our_prob': edge['prob'],
            'edge': edge['edge'],
            'ev': edge['ev']
        })

# 4. Compare to actual Week 10 results
actual_stats = load_actual_stats(week=10)

for pred in predictions:
    actual = actual_stats[pred['player']][pred['prop_type']]

    if pred['side'] == 'OVER':
        hit = actual > pred['line']
    else:
        hit = actual < pred['line']

    pred['hit'] = hit
    pred['actual'] = actual

# 5. Calculate ROI
win_rate = sum(p['hit'] for p in predictions) / len(predictions)
net_profit = sum(90 if p['hit'] else -100 for p in predictions)
roi = net_profit / (len(predictions) * 100)
```

## Expected Performance Targets

### Conservative Estimates
- **Win Rate:** 58-62% (need >52.4% to profit at -110)
- **ROI:** +15-25% per week
- **Volume:** 20-40 prop bets per week (after filters)
- **Brier Score:** <0.20 (calibrated probabilities)

### By Prop Type
| Prop Type | Expected Win Rate | Volume/Week |
|-----------|-------------------|-------------|
| Pass Yards | 62% | 8-12 |
| Rush Yards | 58% | 6-10 |
| Rec Yards | 60% | 8-12 |
| TDs | 55% | 4-8 |

## Next Steps

1. ✅ Fetch 2025 games data (DONE)
2. ✅ Train baseline game model (DONE)
3. ⏳ **Fetch DK prop lines for Weeks 1-10**
4. ⏳ **Match props to actual player stats**
5. ⏳ **Train prop-specific models**
6. ⏳ **Backtest on Week 10**
7. ⏳ **Compare to our earlier estimates**

## Key Insights to Incorporate

From our Week 10 analysis:
- ✅ Rest advantage matters (+5.9% home win rate)
- ✅ High totals (>48) favor passing props
- ❌ Division games are NOT more unpredictable (61.8% favorite cover rate)
- ✅ Pick'em games are true coin flips (avoid)
- ✅ Blowouts kill losing team's QB props
- ✅ Same-game stacks (QB + WR) have positive correlation

## Data Requirements

### Odds API
- API Key: Required
- Cost: ~$100/month for player props
- Coverage: DraftKings, FanDuel, BetMGM
- Historical: Need to fetch and store weekly

### NFLverse
- Cost: Free
- Coverage: All games since 1999
- Granularity: Play-by-play level
- Update frequency: Weekly

## File Structure
```
backend/
  data/
    props/
      week_01_dk_props.json
      week_02_dk_props.json
      ...
    player_stats/
      week_01_actuals.csv
      week_02_actuals.csv
      ...
  modeling/
    train_prop_models.py       # Main training script
    predict_props.py           # Week prediction script
    evaluate_props.py          # Backtest evaluation
  features/
    extract_player_features.py # Player-specific features
    extract_opponent_features.py # Defensive features
```
