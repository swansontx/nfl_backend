# Game Totals & Over/Under Tuning Guide

## Current Performance

**Baseline Model (Simple Recent Averaging):**
- MAE: 12.68 points
- Within 7: 31.2%
- Method: Average last 4 games + 2.5 HFA

**This is actually GOOD!** Most betting models aim for 30-35% within 7 points.

---

## Why Enhanced Model Failed

Tested adding all validated weights simultaneously:
- Weather adjustments
- Injury impacts
- Situational factors
- Weighted recent form

**Result:** -2.8% worse (MAE: 13.04 vs 12.68)

**Reason:** Too many adjustments = more noise than signal

---

## Optimal Tuning Strategy

### Phase 1: Test Features Individually

Test each feature in isolation to find which ones actually help:

```bash
# Test 1: Add ONLY weather
python test_feature.py --feature=weather
# Expected: Slight improvement on windy/cold games

# Test 2: Add ONLY primetime
python test_feature.py --feature=primetime
# Expected: Improvement on SNF/MNF games

# Test 3: Add ONLY major injuries
python test_feature.py --feature=injuries
# Expected: Improvement when QB/star player out
```

### Phase 2: Combine Winners

Only combine features that individually improved accuracy:

```python
def tuned_prediction(context):
    baseline = recent_average(last_4_games)

    # Only add features that tested positive
    if feature_X_helped:
        baseline += feature_X_adjustment

    return baseline
```

### Phase 3: Optimize Weights

Use regression to find optimal feature weights:

```python
from sklearn.linear_model import Ridge

# Features that individually helped
features = ['recent_avg', 'wind', 'primetime', 'qb_injury']

# Learn optimal weights
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Model automatically learns:
# - recent_avg: weight = 0.95 (most important!)
# - wind: weight = 0.15 (small effect)
# - primetime: weight = 0.30 (moderate effect)
# - qb_injury: weight = 0.50 (significant effect)
```

---

## Specific Improvement Ideas

### 1. Team-Specific Baselines

Don't use NFL average (22/20), use team's season average:

```python
# Current (weak)
baseline = 22.0 if is_home else 20.0

# Better
KC_season_avg = 28.5  # Chiefs score a lot
CHI_season_avg = 17.2  # Bears don't

baseline = team_season_avg + (2.5 if is_home else 0)
```

**Expected Impact:** +3-5% accuracy

### 2. Weighted Recent Games

Weight recent games by quality of opponent:

```python
# Current (simple)
recent_avg = mean([24, 28, 21, 27])  # = 25.0

# Better (weighted by opponent)
games = [
    (24, vs_good_defense),   # weight: 1.2
    (28, vs_bad_defense),    # weight: 0.8
    (21, vs_elite_defense),  # weight: 1.5
    (27, vs_avg_defense)     # weight: 1.0
]
weighted_avg = weighted_mean(games)  # More accurate
```

**Expected Impact:** +2-3% accuracy

### 3. Pace Adjustments

Fast-paced teams = higher totals:

```python
# Get team's plays per game
team_pace = 67.5  # Plays per game
league_avg_pace = 64.0

pace_factor = team_pace / league_avg_pace  # 1.055
adjusted_baseline = baseline * pace_factor
```

**Expected Impact:** +1-2% accuracy

### 4. Vegas Line Integration

Use Vegas line as a signal (they're good!):

```python
our_prediction = 48.5
vegas_line = 52.5

# Regress toward Vegas (they know things we don't)
final_prediction = 0.7 * our_prediction + 0.3 * vegas_line
# = 0.7(48.5) + 0.3(52.5) = 49.7
```

**Expected Impact:** +5-8% accuracy (Vegas is smart!)

### 5. Ensemble Multiple Models

Combine different approaches:

```python
models = {
    'recent_average': predict_from_recent(),
    'regression': predict_from_regression(),
    'team_specific': predict_team_model(),
}

# Weighted average based on historical accuracy
final = (
    0.4 * models['recent_average'] +    # Most reliable
    0.35 * models['regression'] +       # Good at extremes
    0.25 * models['team_specific']      # Good for specific teams
)
```

**Expected Impact:** +3-5% accuracy

---

## Recommended Roadmap

### Week 1: Individual Feature Testing
- [ ] Test weather impact in isolation
- [ ] Test primetime impact in isolation
- [ ] Test injury impact in isolation
- [ ] Test pace adjustments
- [ ] Document which features help

### Week 2: Combine Winners
- [ ] Build model with only positive features
- [ ] Run backtest on 2021-2023 data
- [ ] Target: Beat baseline 12.68 MAE

### Week 3: Advanced Techniques
- [ ] Add team-specific baselines
- [ ] Add opponent-weighted recent games
- [ ] Test Vegas line integration
- [ ] Target: 35%+ within 7 points

### Week 4: Production Testing
- [ ] Paper trade for 2-3 weeks
- [ ] Track actual vs predicted
- [ ] Compare to Vegas lines
- [ ] Deploy if profitable

---

## Success Metrics

| Metric | Current | Good | Excellent |
|--------|---------|------|-----------|
| **MAE** | 12.68 | <12.0 | <10.0 |
| **Within 7** | 31.2% | >35% | >40% |
| **Within 3** | ~15% | >20% | >25% |
| **ROI vs Vegas** | TBD | +3% | +7% |

---

## Key Takeaways

1. **Simple baseline is strong** (31.2% within 7)
2. **Don't add all features at once** (creates noise)
3. **Test features individually first**
4. **Use regression to learn optimal weights**
5. **Team-specific models likely better than universal**
6. **Vegas line integration could be biggest win**

---

## Tools for Testing

### Feature Testing Script

```python
def test_feature_impact(feature_name):
    """Test if adding a feature improves accuracy."""

    baseline_predictions = []
    enhanced_predictions = []
    actuals = []

    for game in test_games:
        # Baseline
        baseline = predict_baseline(game)
        baseline_predictions.append(baseline)

        # With feature
        enhanced = predict_baseline(game)
        enhanced += apply_feature(game, feature_name)
        enhanced_predictions.append(enhanced)

        actuals.append(game.actual_total)

    baseline_mae = mean_absolute_error(baseline_predictions, actuals)
    enhanced_mae = mean_absolute_error(enhanced_predictions, actuals)

    improvement = (baseline_mae - enhanced_mae) / baseline_mae * 100

    print(f"{feature_name}: {improvement:+.1f}% improvement")
    return improvement > 0  # True if feature helps
```

### Regression Model Template

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Prepare features
X = pd.DataFrame({
    'home_recent': home_recent_avgs,
    'away_recent': away_recent_avgs,
    'wind': wind_speeds,
    'primetime': is_primetime,
    'qb_injury': has_qb_injury,
    # ... more features
})

y = actual_totals

# Train with cross-validation
model = Ridge(alpha=1.0)  # Regularization
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')

print(f"Cross-validated MAE: {-scores.mean():.2f} ± {scores.std():.2f}")

# Fit final model
model.fit(X, y)

# See feature importance
for feature, weight in zip(X.columns, model.coef_):
    print(f"{feature}: {weight:.3f}")
```

---

## Questions to Investigate

1. **Does weather help?** Test on games with wind >15 mph
2. **Does primetime hurt scoring?** Test SNF/MNF vs 1pm games
3. **Do injuries matter?** Test games with QB/star player out
4. **Does pace matter?** Test fast teams vs slow teams
5. **Does Vegas beat us?** Compare our predictions to Vegas lines

Run these tests systematically to find what actually works!
