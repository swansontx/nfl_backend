# System Validation Guide

## How Do We Know It Works?

This system is validated through multiple layers of testing:

## 1. Automated Validation Tests

Run the comprehensive validation suite:

```bash
# Full validation (all 7 tests)
python scripts/validate_system.py --season 2024

# Quick validation (data, features, models only)
python scripts/validate_system.py --season 2024 --quick
```

### What Gets Tested:

1. **Data Availability** ✅
   - Can we access games, players, features?
   - Is the database populated?

2. **Feature Engineering** ✅
   - Do smoothed features generate correctly?
   - Do matchup features extract?

3. **Model Predictions** ✅
   - Can models generate projections?
   - Do projections have mu, sigma, model_prob?

4. **Calibration** ✅
   - Are probabilities well-calibrated?
   - Brier score, log loss, ROC AUC

5. **Trend Analysis** ✅
   - Does trend detection work?
   - Recent form, streaks, consistency

6. **Recommendations** ✅
   - Can we generate ranked recommendations?
   - Do scores combine all signals?

7. **Historical Backtest** ✅
   - Would we have made money historically?
   - ROI, Sharpe ratio, win rate

---

## 2. Historical Backtesting

Test on past data to see if the system would have been profitable:

```bash
# Full season backtest
python scripts/run_historical_backtest.py --season 2024

# Date range
python scripts/run_historical_backtest.py \
    --start 2024-09-01 \
    --end 2024-12-31

# With signal analysis
python scripts/run_historical_backtest.py \
    --season 2024 \
    --analyze-signals

# Rolling window (detect drift)
python scripts/run_historical_backtest.py \
    --seasons 2023 2024 \
    --rolling

# Compare strategies
python scripts/run_historical_backtest.py \
    --season 2024 \
    --compare-strategies
```

### What to Look For:

**Good Performance:**
- **ROI > 5%** (consistent profit)
- **Sharpe Ratio > 1.0** (good risk-adjusted returns)
- **Win Rate > 52.4%** (beats -110 juice)
- **Max Drawdown < 30%** (manageable risk)
- **Positive CLV** (beating closing lines)

**Warning Signs:**
- Negative ROI (losing money)
- Sharpe < 0.5 (poor risk-adjusted returns)
- High variance in rolling windows (model drift)
- Win rate < 50% (below breakeven)

---

## 3. Calibration Validation

Probabilities should match reality:

```python
from backend.calibration import CalibrationValidator

validator = CalibrationValidator()

# Test calibration for a market
metrics = validator.evaluate_market(
    market='player_rec_yds',
    season=2024
)

print(f"Brier Score: {metrics.brier_score}")  # Lower = better (0.25 = random)
print(f"ECE: {metrics.expected_calibration_error}")  # < 0.05 = well calibrated
print(f"ROC AUC: {metrics.roc_auc}")  # > 0.6 = good discrimination
```

### Calibration Metrics:

- **Brier Score**: Mean squared error of probabilities
  - < 0.20 = excellent
  - 0.20-0.25 = good
  - \> 0.25 = poor (worse than random)

- **Expected Calibration Error (ECE)**: Deviation from perfect calibration
  - < 0.05 = excellent
  - 0.05-0.10 = acceptable
  - \> 0.10 = poorly calibrated

- **ROC AUC**: Discrimination ability
  - \> 0.70 = excellent
  - 0.60-0.70 = good
  - 0.50-0.60 = weak
  - = 0.50 = random guessing

---

## 4. Signal Effectiveness Analysis

Which signals actually help?

```bash
python scripts/run_historical_backtest.py \
    --season 2024 \
    --analyze-signals
```

This shows:
- **Standalone AUC**: How good is each signal alone?
- **Optimal Weights**: What weights maximize performance?
- **Incremental Value**: Does adding this signal improve the model?

### Expected Signal Performance:

| Signal | Expected AUC | Importance |
|--------|--------------|------------|
| Base Model | 0.60-0.65 | High |
| Trend | 0.55-0.60 | Medium |
| Matchup | 0.54-0.58 | Medium |
| News | 0.52-0.56 | Low-Medium |
| Roster | 0.51-0.54 | Low |

If a signal has AUC < 0.52, it's not adding value and should be removed or re-weighted.

---

## 5. Live Paper Trading

Test on current games without risking money:

```python
from backend.recommendations import RecommendationScorer
from backend.integrations import OddsAPIClient

# Get recommendations
scorer = RecommendationScorer()
recs = scorer.recommend_props(game_id="2024_17_KC_LV", limit=10)

# Compare to market
odds_client = OddsAPIClient()
market_odds = odds_client.fetch_player_props(event_id="...")

for rec in recs:
    # Find market line
    market_prop = find_matching_prop(rec, market_odds)

    # Calculate edge
    edge = rec.calibrated_prob - odds_client._odds_to_prob(market_prop.over_odds)

    print(f"{rec.player_name} {rec.market}")
    print(f"  Model: {rec.calibrated_prob:.1%}")
    print(f"  Market: {market_prob:.1%}")
    print(f"  Edge: {edge:+.1%}")

    # Track this pick and verify after game
```

Track picks in a spreadsheet:
- Player, market, line
- Model probability
- Market odds
- Actual outcome
- P/L

After 50-100 picks, calculate:
- Overall ROI
- Calibration (did 60% picks hit 60% of the time?)
- CLV (did you beat closing lines?)

---

## 6. Continuous Monitoring

Set up automated monitoring:

### Daily Checks:
```bash
# Run validation
python scripts/validate_system.py --quick

# Check calibration hasn't drifted
python -c "
from backend.calibration import CalibrationValidator
v = CalibrationValidator()
m = v.evaluate_market('player_rec_yds', 2024)
assert m.expected_calibration_error < 0.10, 'Calibration drift!'
"
```

### Weekly Checks:
- Review recent picks performance
- Check signal weights still optimal
- Monitor for model drift
- Update calibration parameters

### Monthly Checks:
- Full backtest on latest month
- Re-train calibration if needed
- Analyze which markets performing best
- Update signal weights based on recent data

---

## 7. Key Performance Indicators (KPIs)

### Model Quality:
- ✅ **Calibration ECE < 0.10** (probabilities accurate)
- ✅ **ROC AUC > 0.60** (can discriminate)
- ✅ **Brier Score < 0.25** (better than random)

### Betting Performance:
- ✅ **ROI > 5%** (profitable)
- ✅ **Sharpe > 1.0** (good risk-adjusted)
- ✅ **Win Rate > 52.4%** (beats juice)
- ✅ **Max Drawdown < 30%** (manageable risk)

### System Health:
- ✅ **All validation tests pass**
- ✅ **No model drift in rolling windows**
- ✅ **Positive CLV** (beating market)
- ✅ **Signal weights stable** (not changing drastically)

---

## 8. What Success Looks Like

### Month 1 (Paper Trading):
- Track 100+ picks
- Verify calibration holds (ECE < 0.10)
- Positive ROI in paper trading
- No major bugs or failures

### Month 2 (Small Stakes):
- Continue paper trading + small real money
- ROI > 3% on real picks
- Sharpe > 0.5
- System running reliably

### Month 3+ (Scale Up):
- Consistent ROI > 5%
- Sharpe > 1.0
- Win rate > 52%
- Positive CLV
- Gradual bankroll growth

---

## 9. Red Flags

Stop and investigate if:

❌ **ROI negative for 2+ weeks**
- Check calibration drift
- Verify data quality
- Review signal weights

❌ **Win rate < 50%**
- Model may be broken
- Re-calibrate immediately
- Check for data issues

❌ **ECE > 0.15**
- Probabilities not calibrated
- Re-train calibration
- May need more data

❌ **Sharpe < 0**
- Losing money with high variance
- Reduce position sizes
- Review strategy

❌ **Max drawdown > 40%**
- Position sizes too large
- Reduce Kelly fraction
- May need to pause

---

## 10. Iterative Improvement

The system gets better over time:

1. **Collect Data**: Track every pick (predicted, actual, outcome)
2. **Analyze**: Which markets/signals work best?
3. **Adjust**: Update weights, add new signals
4. **Re-calibrate**: Retrain calibration monthly
5. **Validate**: Re-run backtests
6. **Repeat**: Continuous improvement cycle

### Example Improvement Cycle:

```python
# Week 1: Collect data
picks = track_picks()

# Week 2: Analyze
from backend.backtest import SignalEffectivenessAnalyzer
analyzer = SignalEffectivenessAnalyzer()
results = analyzer.analyze_signals(picks)

# Week 3: Adjust weights
new_weights = results.optimal_weights

# Update SignalWeights in recommendation_scorer.py
# with new optimal weights

# Week 4: Re-validate
python scripts/validate_system.py --season 2024
```

---

## Quick Start Validation

```bash
# 1. Run validation suite
python scripts/validate_system.py --season 2024

# 2. Run historical backtest
python scripts/run_historical_backtest.py --season 2024 --analyze-signals

# 3. If all tests pass and ROI > 5%, start paper trading
# 4. Track 100 picks
# 5. Verify results match expectations
# 6. Scale up gradually
```

**Bottom line**: The system is validated when:
1. All automated tests pass ✅
2. Historical backtest shows ROI > 5% ✅
3. Calibration is tight (ECE < 0.10) ✅
4. Paper trading confirms real-world performance ✅
5. Continuous monitoring shows stability ✅
