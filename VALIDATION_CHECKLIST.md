# Validation Checklist

Use this checklist to verify the system works correctly.

## Pre-Validation Setup

- [ ] Database is populated with:
  - [ ] Games for 2024 season
  - [ ] Players (QB, RB, WR, TE)
  - [ ] PlayerGameFeatures
  - [ ] At least some completed games with outcomes

- [ ] Environment variables set (if using real APIs):
  - [ ] `ODDS_API_KEY` (The Odds API)
  - [ ] `TWITTER_BEARER_TOKEN` (optional)
  - [ ] `OPENWEATHER_API_KEY` (optional)
  - [ ] `FANTASYPROS_API_KEY` (optional)

## Step 1: Quick Validation (5 minutes)

```bash
python scripts/validate_system.py --season 2024 --quick
```

**Expected Output:**
```
✅ PASSED: Data Availability
✅ PASSED: Feature Engineering
✅ PASSED: Model Predictions
```

**If any fail:**
- Data Availability → Check database populated
- Feature Engineering → Check PlayerGameFeature data exists
- Model Predictions → Check models can access data

---

## Step 2: Full Validation (15 minutes)

```bash
python scripts/validate_system.py --season 2024
```

**Expected Output:**
```
✅ PASSED: Data Availability
✅ PASSED: Feature Engineering
✅ PASSED: Model Predictions
✅ PASSED: Calibration
✅ PASSED: Trend Analysis
✅ PASSED: Recommendations
✅ PASSED: Historical Backtest

🎉 ALL TESTS PASSED! System is working correctly.
```

**If Calibration fails:**
- Not enough projections + outcomes matched
- Need at least 50 matched pairs
- Run outcome extraction first:
  ```python
  from backend.calibration import OutcomeExtractor
  extractor = OutcomeExtractor()
  extractor.extract_season_outcomes(2024)
  ```

**If Historical Backtest fails:**
- Need completed games with outcomes
- Check `Game.game_date < now()`
- Verify outcomes extracted

---

## Step 3: Historical Backtest (30 minutes)

```bash
python scripts/run_historical_backtest.py \
    --season 2024 \
    --analyze-signals
```

**What to Check:**

### Betting Performance:
- [ ] **ROI**: Should be > 0% (positive)
  - Excellent: > 10%
  - Good: 5-10%
  - Acceptable: 0-5%
  - Poor: < 0% (losing money)

- [ ] **Win Rate**: Should be > 50%
  - Need > 52.4% to beat -110 juice
  - 55%+ is excellent

- [ ] **Sharpe Ratio**: Risk-adjusted returns
  - Excellent: > 1.5
  - Good: 1.0-1.5
  - Acceptable: 0.5-1.0
  - Poor: < 0.5

- [ ] **Max Drawdown**: Should be < 30%
  - If > 40%, reduce position sizes

### Calibration:
- [ ] **Brier Score**: < 0.25 (lower = better)
  - < 0.20 = excellent
  - 0.20-0.25 = good
  - \> 0.25 = poor

- [ ] **Log Loss**: < 0.7 (lower = better)

- [ ] **ROC AUC**: > 0.55 (higher = better)
  - \> 0.65 = excellent
  - 0.60-0.65 = good
  - 0.55-0.60 = acceptable
  - < 0.55 = poor

### Signal Analysis:
- [ ] Check which signals have highest AUC
- [ ] Note optimal weights
- [ ] Update `SignalWeights` in `backend/recommendations/recommendation_scorer.py` with optimal weights

**Example Good Results:**
```
BETTING SIMULATION (Kelly Criterion):
  Total Bets:    247
  Win Rate:      54.25%
  ROI:           7.84%
  Sharpe Ratio:  1.23
  Max Drawdown:  18.35%
  Final P/L:     $78.40

CALIBRATION METRICS:
  Brier Score:  0.218
  Log Loss:     0.642
  ROC AUC:      0.627
```

✅ This would pass validation!

---

## Step 4: Calibration Deep Dive

```python
from backend.calibration import CalibrationValidator

validator = CalibrationValidator()

# Test each key market
for market in ['player_rec_yds', 'player_rush_yds', 'player_pass_yds']:
    metrics = validator.evaluate_market(market, 2024)

    print(f"\n{market}:")
    print(f"  Brier: {metrics.brier_score:.3f}")
    print(f"  ECE: {metrics.expected_calibration_error:.3f}")
    print(f"  ROC AUC: {metrics.roc_auc:.3f}")
    print(f"  Samples: {metrics.n_samples}")

    # Generate calibration plot
    validator.plot_calibration_curve(
        metrics,
        save_path=f"calibration_{market}.png"
    )
```

**Checklist:**
- [ ] Each market has ECE < 0.10
- [ ] Calibration curves are close to diagonal
- [ ] Enough samples (> 50 per market)

---

## Step 5: Signal Effectiveness

```bash
python scripts/run_historical_backtest.py \
    --season 2024 \
    --analyze-signals
```

**Check Signal Rankings:**
```
SIGNAL RANKINGS (by standalone AUC):
Signal                    AUC        Brier      Corr       Weight
--------------------------------------------------------------------------------
base_signal              0.6234     0.2145     0.3421     0.3500
trend_signal             0.5823     0.2287     0.2134     0.1800
matchup_signal           0.5612     0.2356     0.1845     0.1200
news_signal              0.5334     0.2412     0.0923     0.0800
roster_signal            0.5156     0.2489     0.0512     0.0600
```

**Checklist:**
- [ ] Base signal has highest AUC (> 0.60)
- [ ] All signals have AUC > 0.50 (better than random)
- [ ] Weights roughly match importance
- [ ] Combined model AUC > best single signal

**If a signal has AUC < 0.52:**
- Consider removing it or reducing weight
- May be adding noise instead of signal

---

## Step 6: Live Paper Trading (Week 1)

**Setup tracking spreadsheet:**

| Date | Player | Market | Line | Model Prob | Odds | Edge | Outcome | P/L |
|------|--------|--------|------|------------|------|------|---------|-----|
| 12/15 | Mahomes | pass_yds | 275.5 | 0.62 | -110 | +0.10 | Over | +$9.09 |
| 12/15 | CMC | rush_yds | 85.5 | 0.58 | -110 | +0.06 | Under | -$10.00 |

**Daily Process:**
1. Generate recommendations for today's games
2. Compare to market odds
3. Calculate edge for each pick
4. Record picks with edge > 3%
5. After games, record outcomes
6. Track running ROI

**Week 1 Targets:**
- [ ] Track at least 20 picks
- [ ] ROI > 0% (positive)
- [ ] Win rate > 50%
- [ ] No major system errors

---

## Step 7: Ongoing Monitoring

### Daily (During Season):
- [ ] Run quick validation
  ```bash
  python scripts/validate_system.py --quick
  ```
- [ ] Check for errors in logs
- [ ] Generate picks for today's games

### Weekly:
- [ ] Review week's performance
- [ ] Check calibration hasn't drifted
- [ ] Update any stale data
- [ ] ROI tracking

### Monthly:
- [ ] Full backtest on latest data
  ```bash
  python scripts/run_historical_backtest.py \
      --start 2024-11-01 \
      --end 2024-11-30
  ```
- [ ] Re-train calibration if needed
- [ ] Analyze signal effectiveness
- [ ] Update weights if optimal changed significantly

---

## Red Flags - Stop and Investigate

🚨 **Immediate Action Required:**
- [ ] ROI negative for 2+ weeks → Re-calibrate
- [ ] Win rate < 48% for 50+ picks → Check model
- [ ] ECE > 0.15 → Probabilities broken
- [ ] System errors/crashes → Debug immediately

⚠️  **Review Needed:**
- [ ] ROI < 2% for 100+ picks → May need tuning
- [ ] Sharpe < 0.5 → Too much variance
- [ ] Max drawdown > 35% → Reduce position sizes
- [ ] Signal AUC dropped > 0.05 → Model drift

---

## Success Criteria

### Week 1:
- ✅ All validation tests pass
- ✅ Historical backtest ROI > 3%
- ✅ 20+ paper trading picks tracked
- ✅ No critical bugs

### Month 1:
- ✅ 100+ paper picks tracked
- ✅ Paper trading ROI > 0%
- ✅ ECE < 0.10 on live picks
- ✅ Positive CLV

### Month 2:
- ✅ 200+ picks total
- ✅ ROI > 3% on live picks
- ✅ Sharpe > 0.5
- ✅ System stable

### Month 3+:
- ✅ Consistent ROI > 5%
- ✅ Sharpe > 1.0
- ✅ Win rate > 52%
- ✅ Scaling up gradually

---

## Quick Commands Reference

```bash
# Full validation
python scripts/validate_system.py --season 2024

# Quick check
python scripts/validate_system.py --season 2024 --quick

# Historical backtest
python scripts/run_historical_backtest.py --season 2024

# With signal analysis
python scripts/run_historical_backtest.py --season 2024 --analyze-signals

# Rolling window (detect drift)
python scripts/run_historical_backtest.py --seasons 2023 2024 --rolling

# Specific date range
python scripts/run_historical_backtest.py \
    --start 2024-09-01 \
    --end 2024-12-31
```

---

## Summary

The system is **validated and ready** when:

1. ✅ All automated tests pass
2. ✅ Historical ROI > 5%
3. ✅ Calibration ECE < 0.10
4. ✅ ROC AUC > 0.60
5. ✅ Win rate > 52%
6. ✅ Sharpe > 1.0
7. ✅ Paper trading confirms real performance
8. ✅ System runs without errors

**Then and only then** → Start with small stakes and scale gradually.
