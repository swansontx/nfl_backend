# Game Predictions Backtest Findings

**Date:** 2025-11-28
**Season:** 2025, Weeks 5-11 (100 games)
**Framework:** `backend/backtesting/game_predictions_backtest.py`

---

## Executive Summary

Backtesting revealed that the **enhanced metrics (pace/turnover/efficiency) currently underperform baseline** predictions for spread accuracy, while showing modest improvement for total predictions.

### Key Findings

| Metric | Baseline | Enhanced | Change |
|--------|----------|----------|--------|
| **Spread MAE** | 11.45 pts | 18.73 pts | **-63.6% (worse)** ⚠️ |
| **Total MAE** | 10.52 pts | 10.95 pts | -4.1% (worse) |
| **Spread ATS Win %** | 46.0% | 39.0% | -7.0% (worse) |
| **Total O/U Win %** | 48.0% | 52.0% | **+4.0% (better)** ✓ |

**Conclusion:** Enhanced metrics need calibration before production use.

---

## Detailed Analysis

### Problem 1: Excessive Spread Adjustments

**Issue:** Enhanced spread predictions show extreme swings that don't match reality.

**Examples:**

1. **HOU @ BAL (Week 5)**
   - Actual: HOU 44 - BAL 10 (HOU +34)
   - Baseline: HOU 21.5 - BAL 24.0 (+2.5)
   - Enhanced: HOU 5.1 - BAL 41.3 (+36.2)
   - **Result:** Enhanced predicted the exact opposite winner!

2. **MIN @ CLE (Week 5)**
   - Actual: MIN 21 - CLE 17 (MIN +4)
   - Baseline: MIN 21.5 - CLE 24.0 (+2.5)
   - Enhanced: MIN 34.7 - CLE 12.0 (-22.7)
   - **Result:** Enhanced overcorrected by 27 points!

### Problem 2: Turnover Margin Adjustment Too Aggressive

**Current Formula:**
```python
turnover_adjustment = margin_differential * 2.5 points
```

**Issue:**
- A margin differential of +10 (common for recent week samples) = +25 point swing
- This is far too large for a 4-week sample
- NFL research suggests ~1.5 points per full-season turnover margin

**Recommendation:** Reduce multiplier to 0.8-1.0 points per margin for recent weeks.

### Problem 3: Efficiency Adjustments May Overcorrect

**Current Logic:**
```python
# EPA adjustment
epa_diff = (home_epa - away_epa) * 65 plays
spread_adj = epa_diff

# Success rate adjustment
rate_diff = home_success - away_success
if abs(rate_diff) > 0.05:
    spread_adj += (rate_diff / 0.05) * 1.0
```

**Issue:**
- EPA differences multiplied by 65 plays can create large adjustments
- Success rate threshold of 0.05 (5%) is very sensitive
- These stack on top of turnover adjustments

**Recommendation:** Add damping factors or use exponential smoothing.

### Problem 4: Small Sample Noise (4 Recent Weeks)

**Issue:**
- Using only 4 recent weeks for metrics creates high variance
- Teams can have anomalous 4-week stretches (injury, schedule luck)
- Turnovers especially are high-variance events

**Examples:**
- A team with 3 defensive TDs in 4 weeks looks elite but may regress
- A team facing backup QBs in 4 weeks has inflated defensive stats

**Recommendation:**
- Blend full-season metrics (70%) with recent metrics (30%)
- Use larger sample for turnovers (8+ weeks or full season)
- Add regression to league mean for small samples

---

## What Worked

### ✅ Total Predictions Improved Slightly

- **Over/Under accuracy:** 48.0% → 52.0% (+4%)
- **Pace adjustments** appear to help with total scoring
- The formula `(pace - 65) / 10 × 3.5 points` seems reasonable

**Why This Worked:**
- Pace is a more stable metric than turnovers
- Pace directly affects possessions → points
- The multiplier (3.5 pts per 10 plays) is conservative

---

## Recommendations for Fixes

### 1. **Reduce Turnover Multiplier** (High Priority)

```python
# Current
turnover_adjustment = margin_diff * 2.5

# Recommended
if recent_weeks <= 4:
    turnover_adjustment = margin_diff * 0.8  # Much smaller for small samples
else:
    turnover_adjustment = margin_diff * 1.5  # Moderate for larger samples
```

### 2. **Add Damping to Efficiency Adjustments** (High Priority)

```python
# Current
spread_adj = epa_diff * 65

# Recommended
spread_adj = epa_diff * 65 * 0.5  # 50% damping factor
```

### 3. **Blend Full Season + Recent Metrics** (Medium Priority)

```python
# Use weighted average
final_metric = (full_season_metric * 0.7) + (recent_metric * 0.3)
```

### 4. **Add Regression to Mean** (Medium Priority)

```python
# For metrics with high variance, regress toward league average
regressed_margin = (observed_margin * 0.6) + (league_avg * 0.4)
```

### 5. **Cap Maximum Adjustments** (Low Priority)

```python
# Prevent any single adjustment from exceeding reasonable bounds
total_adjustment = np.clip(total_adjustment, -10, +10)  # Cap at ±10 points
```

---

## Testing Protocol

After implementing fixes:

1. **Re-run backtest** on same 2025 weeks 5-11 data
2. **Target metrics:**
   - Spread MAE < 10.5 points (better than baseline's 11.45)
   - Total MAE < 10.0 points (better than baseline's 10.52)
   - Spread ATS Win % > 48% (better than baseline's 46%)
   - Total O/U Win % > 52% (maintain current improvement)

3. **Validate on different weeks:**
   - Test on weeks 1-4 (not used for tuning)
   - Test on 2024 data once play-by-play is available

---

## Data Availability Note

**Important:** 2024 backtest could not use enhanced metrics because:
- Play-by-play data only available for 2025
- Advanced metrics (EPA, success rate) require play-by-play data
- Baseline predictions work without play-by-play (simple PPG)

**Action Item:** Acquire 2024 play-by-play data for more robust backtesting.

---

## Files Created

1. `backend/backtesting/game_predictions_backtest.py` (630 lines)
   - Comprehensive backtesting framework
   - Compares baseline vs enhanced predictions
   - Calculates accuracy metrics and betting performance

2. `examples/run_backtest.py` (90 lines)
   - CLI tool for running backtests
   - Supports week filtering and CSV export
   - Usage: `python examples/run_backtest.py --season 2025 --weeks 5 6 7 8`

3. `outputs/backtest_2025_results.csv` (100 games)
   - Detailed results for every prediction
   - Includes errors, betting outcomes, actual scores

---

## Next Steps

1. **Immediate:** Implement recommended calibration fixes (items #1 and #2 above)
2. **Short-term:** Add blended metrics and regression to mean
3. **Medium-term:** Re-run backtest and validate improvements
4. **Long-term:** Acquire 2024 play-by-play for larger sample validation

---

## Conclusion

The backtesting framework is **working correctly** and has **identified real issues** with the current enhanced metrics implementation. The metrics are theoretically sound but need better calibration:

- ✅ Framework correctly compares baseline vs enhanced
- ✅ Identified specific problems (turnover multiplier too high)
- ✅ Pace adjustments show promise for totals (+4% on O/U)
- ⚠️ Spread adjustments need significant tuning
- ⚠️ Small sample sizes (4 weeks) create noise

**Status:** Enhanced metrics should NOT be used in production until calibration fixes are implemented and validated.

---

**Author:** Claude Code
**Version:** 1.0
**Last Updated:** 2025-11-28
