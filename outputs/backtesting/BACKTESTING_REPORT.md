# Historical Backtesting Report

**Generated:** 2025-11-27 04:49:19

**Seasons Analyzed:** 2021, 2022, 2023


---


## Executive Summary

- **Total Observations:** 0
- **Features Tested:** 5
- **Updates Recommended:** 0/5
- **Average Improvement:** 0.0%


## Results Summary

| Feature | Sample Size | RMSE | MAE | R² | Improvement | Update? |
|---------|------------:|-----:|----:|---:|------------:|:-------:|
| Injury Impact | 0 | 0.00 | 0.00 | 0.000 | +0.0% | ❌ |
| Defense Matchup | 0 | 0.00 | 0.00 | 0.000 | +0.0% | ❌ |
| Weather Impact | 0 | 0.00 | 0.00 | 0.000 | +0.0% | ❌ |
| Situational Factors | 0 | 0.00 | 0.00 | 0.000 | +0.0% | ❌ |
| Overall Accuracy | 0 | 0.00 | 0.00 | 0.000 | +0.0% | ❌ |

---


## Detailed Results


### Injury Impact

⚠️ Insufficient data for analysis

- Error: 'player'

### Defense Matchup

⚠️ Insufficient data for analysis

- Error: 'game_id'

### Weather Impact

⚠️ Insufficient data for analysis

- Error: 'game_id'

### Situational Factors

⚠️ Insufficient data for analysis

- Error: 'team'

### Overall Accuracy

⚠️ Insufficient data for analysis

- Error: 'team'

---


## Implementation Recommendations


No updates recommended at this time. Current factors are performing well.


## Next Steps

1. Review calculated factors for each feature
2. Update configuration files with data-driven coefficients
3. Re-run validation tests to confirm improvements
4. Monitor performance in production
5. Schedule quarterly backtesting to refine factors


---


## Appendix: Methodology


### Data Sources
- Historical game data: NFL official stats
- Player statistics: nfl-data-py package
- Injury reports: NFL injury reports
- Weather data: Historical weather APIs


### Validation Metrics
- **RMSE (Root Mean Square Error):** Lower is better
- **MAE (Mean Absolute Error):** Lower is better
- **Correlation:** Higher is better (closer to 1.0)
- **R² (Coefficient of Determination):** Higher is better (closer to 1.0)


### Statistical Significance
- Minimum sample size: 30 observations
- Confidence level: 95%
- Improvement threshold: 5% reduction in prediction error
