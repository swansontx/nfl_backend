# NFL Backend Metrics Registry

**Central catalog of all available metrics and where they're used across the system.**

Last Updated: 2025-11-27

---

## 📊 Metric Categories

### 1. **Efficiency Metrics** (Play-Level Performance)

| Metric | Definition | Calculation | Used In | Source |
|--------|-----------|-------------|---------|--------|
| **Success Rate** | % of plays that achieve situational goals | 1st: 45% yards, 2nd: 60%, 3rd/4th: conversion | Matchups, Deep Analysis | `advanced_team_metrics.py` |
| **EPA (Expected Points Added)** | Points added vs expectation per play | From pbp `epa` column | Player Props, Matchups | `train_passing_model.py`, `advanced_team_metrics.py` |
| **CPOE (Completion % Over Expected)** | QB accuracy vs expectation | From pbp `cpoe` column | Player Props (QB) | `train_passing_model.py` |
| **Yards Per Play** | Average yards gained per play | `total_yards / total_plays` | Matchups, Game Predictions | `team_matchup_analyzer.py` |

**Integration Points:**
- ✅ Player prop models: `train_passing_model.py` (EPA, CPOE)
- ✅ Team matchups: `advanced_team_metrics.py` (Success Rate, EPA)
- ⏳ Game predictions: Can integrate
- ⏳ Backtesting validation: Can integrate

---

### 2. **Red Zone Performance** (Scoring Efficiency)

| Metric | Definition | Calculation | Used In | Source |
|--------|-----------|-------------|---------|--------|
| **Red Zone TD%** | Touchdown rate in red zone | `TDs / Red Zone Attempts` | Matchups | `advanced_team_metrics.py` |
| **Red Zone Score%** | Any score rate in red zone | `(TDs + FGs) / Red Zone Attempts` | Matchups | `advanced_team_metrics.py` |
| **Red Zone Attempts** | Trips inside opponent 20 | Filter `yardline_100 <= 20` | Matchups | `advanced_team_metrics.py` |

**Integration Points:**
- ✅ Team matchups: `advanced_team_metrics.py`
- ⏳ Game totals predictions: Can integrate (scoring likelihood)
- ⏳ Player TD props: Can integrate (red zone target share)
- ⏳ Deep analysis: Can add to situational factors

---

### 3. **Third Down Efficiency** (Conversion Metrics)

| Metric | Definition | Calculation | Used In | Source |
|--------|-----------|-------------|---------|--------|
| **Third Down %** | Conversion rate on 3rd down | `Conversions / Attempts` | Matchups | `advanced_team_metrics.py` |
| **Third Down Attempts** | Total 3rd down plays | Filter `down == 3` | Matchups | `advanced_team_metrics.py` |
| **Third Down Conversions** | Successful conversions | Filter `third_down_converted == 1` | Matchups | `advanced_team_metrics.py` |

**Integration Points:**
- ✅ Team matchups: `advanced_team_metrics.py`
- ⏳ Game pace predictions: Can integrate (TOP factor)
- ⏳ Drive success models: Can integrate

---

### 4. **Passing Efficiency** (QB & Passing Game)

| Metric | Definition | Calculation | Used In | Source |
|--------|-----------|-------------|---------|--------|
| **Completion %** | Pass completion rate | `Completions / Attempts` | Matchups, Player Props | `advanced_team_metrics.py`, `train_passing_model.py` |
| **Yards Per Attempt** | Yards per pass attempt | `Passing Yards / Attempts` | Matchups, Player Props | Both |
| **Air Yards/Attempt** | Air yards per attempt | `Air Yards / Attempts` | Matchups | `advanced_team_metrics.py` |
| **YAC Per Completion** | Yards after catch per completion | `Total YAC / Completions` | Matchups | `advanced_team_metrics.py` |
| **QB EPA** | EPA on QB plays | From pbp `qb_epa` | Player Props (QB) | `train_passing_model.py` |
| **QB CPOE** | Completion % over expected | From pbp `cpoe` | Player Props (QB) | `train_passing_model.py` |

**Integration Points:**
- ✅ Player props (QB): `train_passing_model.py`
- ✅ Team matchups: `advanced_team_metrics.py`
- ⏳ WR/TE props: Can integrate YAC metrics
- ⏳ Game totals: Can integrate passing volume

---

### 5. **Rushing Efficiency** (Ground Game)

| Metric | Definition | Calculation | Used In | Source |
|--------|-----------|-------------|---------|--------|
| **Yards Per Carry** | Average yards per rush | `Rush Yards / Carries` | Matchups | `advanced_team_metrics.py` |
| **Rush Success Rate** | % successful rushes | `Successful Rushes / Total Rushes` | Matchups | `advanced_team_metrics.py` |
| **Explosive Rush %** | % rushes 10+ yards | `10+ Yd Rushes / Total Rushes` | Matchups | `advanced_team_metrics.py` |

**Integration Points:**
- ✅ Team matchups: `advanced_team_metrics.py`
- ⏳ RB prop models: Can integrate
- ⏳ Game script predictions: Can integrate (rush volume)

---

### 6. **Turnover Metrics** (Ball Security)

| Metric | Definition | Calculation | Used In | Source |
|--------|-----------|-------------|---------|--------|
| **Turnover Rate** | Turnovers per 100 plays | `(TOs / Plays) * 100` | Matchups | `advanced_team_metrics.py` |
| **Takeaway Rate** | Takeaways per 100 def plays | `(Takeaways / Def Plays) * 100` | Matchups | `advanced_team_metrics.py` |
| **Turnover Margin** | TO differential | `Takeaways - Turnovers` | Matchups, Game Predictions | `advanced_team_metrics.py` |
| **Interceptions** | Total INTs | Count from pbp | Matchups, Player Props | Both |
| **Fumbles Lost** | Total fumbles lost | Count from pbp | Matchups | `advanced_team_metrics.py` |

**Integration Points:**
- ✅ Team matchups: `advanced_team_metrics.py`
- ✅ Player props (QB): `train_passing_model.py` (INT tracking)
- ⏳ Game predictions: Can integrate (turnover impact on spread)
- ⏳ Defense matchups: Can integrate (takeaway likelihood)

---

### 7. **Explosive Plays** (Big Play Capability)

| Metric | Definition | Calculation | Used In | Source |
|--------|-----------|-------------|---------|--------|
| **Explosive Play Rate** | % plays 20+ pass or 10+ rush | `Big Plays / Total Plays` | Matchups | `advanced_team_metrics.py` |
| **Explosive Plays Allowed** | % opp explosive plays | `Opp Big Plays / Opp Plays` | Matchups | `advanced_team_metrics.py` |

**Integration Points:**
- ✅ Team matchups: `advanced_team_metrics.py`
- ⏳ Player props: Can integrate (boom/bust likelihood)
- ⏳ Game totals: Can integrate (variance factor)

---

### 8. **Pace & Time Control**

| Metric | Definition | Calculation | Used In | Source |
|--------|-----------|-------------|---------|--------|
| **Plays Per Game** | Offensive pace | `Total Plays / Games` | Matchups | `advanced_team_metrics.py` |
| **Seconds Per Play** | Play clock pace | `Game Time / Plays` | Matchups | `advanced_team_metrics.py` |
| **Time of Possession** | Minutes possessing ball | Drive-level calculation | Matchups | `advanced_team_metrics.py` (placeholder) |

**Integration Points:**
- ✅ Team matchups: `advanced_team_metrics.py`
- ⏳ Game totals: Can integrate (pace impacts total plays)
- ⏳ Player volume props: Can integrate (plays = opportunities)

---

### 9. **Team Performance** (Basic Stats)

| Metric | Definition | Calculation | Used In | Source |
|--------|-----------|-------------|---------|--------|
| **Points Per Game** | Scoring average | `Total Points / Games` | Matchups, Game Predictions | `team_matchup_analyzer.py` |
| **Yards Per Game** | Yardage average | `Total Yards / Games` | Matchups | Both analyzers |
| **Passing YPG** | Passing yards average | `Pass Yards / Games` | Matchups | Both analyzers |
| **Rushing YPG** | Rushing yards average | `Rush Yards / Games` | Matchups | Both analyzers |
| **Points Allowed** | Defensive scoring | `Opp Points / Games` | Matchups | `team_matchup_analyzer.py` |
| **Yards Allowed** | Defensive yardage | `Opp Yards / Games` | Matchups | `team_matchup_analyzer.py` |

**Integration Points:**
- ✅ Team matchups: `team_matchup_analyzer.py`
- ✅ Backtesting: `defense_matchup_backtest.py`, `weather_impact_backtest.py`
- ⏳ Game predictions: Can integrate

---

### 10. **Situational Performance**

| Metric | Definition | Calculation | Used In | Source |
|--------|-----------|-------------|---------|--------|
| **Home Points/Game** | Scoring at home | `Home Points / Home Games` | Matchups | `team_matchup_analyzer.py` |
| **Away Points/Game** | Scoring on road | `Away Points / Away Games` | Matchups | `team_matchup_analyzer.py` |
| **Recent Form (PPG)** | Last 4 games scoring | `Recent Points / 4` | Matchups | `team_matchup_analyzer.py` |
| **Division Game Record** | Performance vs division | Win-loss in division | Matchups | `team_matchup_analyzer.py` (placeholder) |
| **Primetime Performance** | Performance in primetime | Stats filtered by `is_primetime` | Deep Analysis | `situational_factors_backtest.py` |

**Integration Points:**
- ✅ Team matchups: `team_matchup_analyzer.py`
- ✅ Backtesting: `situational_factors_backtest.py`
- ⏳ Game predictions: Can integrate situational adjustments

---

### 11. **Matchup-Specific**

| Metric | Definition | Calculation | Used In | Source |
|--------|-----------|-------------|---------|--------|
| **Head-to-Head Win %** | Historical matchup wins | `Wins / H2H Games` | Matchups | `team_matchup_analyzer.py` |
| **H2H Avg Total** | Average combined score | `Avg(Home + Away Score)` | Matchups | `team_matchup_analyzer.py` |
| **Common Opponent Edge** | Performance vs shared foes | Compare scores vs same teams | Matchups | `team_matchup_analyzer.py` |
| **Momentum** | Recent trend | `Recent PPG - Season PPG` | Matchups | `team_matchup_analyzer.py` |

**Integration Points:**
- ✅ Team matchups: `team_matchup_analyzer.py`
- ⏳ Game predictions: Can integrate historical factors

---

### 12. **Player-Level** (Individual Performance)

| Metric | Definition | Calculation | Used In | Source |
|--------|-----------|-------------|---------|--------|
| **Rolling 3-Game Avg** | 3-game moving average | Last 3 games stats | Player Props | `train_passing_model.py` |
| **Rolling 5-Game Avg** | 5-game moving average | Last 5 games stats | Player Props | `train_passing_model.py` |
| **Season Average** | Full season average | All games this season | Player Props | `train_passing_model.py` |
| **Targets** | Receiving targets | From player stats | Player Props, Injury Impact | `train_passing_model.py`, `injury_impact_backtest.py` |
| **Target Share** | % of team targets | `Player Targets / Team Targets` | Player Props | `train_passing_model.py` |
| **Snap %** | % of offensive snaps | From snap count data | Player Props | (Not yet implemented) |

**Integration Points:**
- ✅ Player props: `train_passing_model.py`, `extract_player_pbp_features.py`
- ✅ Injury impact: `injury_impact_backtest.py` (target redistribution)
- ⏳ Usage tracking: Can expand

---

### 13. **Defense Matchup** (Positional Defense)

| Metric | Definition | Calculation | Used In | Source |
|--------|-----------|-------------|---------|--------|
| **Yards vs Position** | Yards allowed to position | `Positional Yards / Games` | Defense Matchups, Backtesting | `defense_matchup_backtest.py` |
| **Defense Adjustment Factor** | Multiplier for matchup | Based on league rank | Defense Matchups | `defense_matchup_backtest.py` |
| **Position Rank** | Defensive ranking vs position | League percentile | Defense Matchups | `defense_matchup_backtest.py` |

**Integration Points:**
- ✅ Backtesting: `defense_matchup_backtest.py` (20,202 observations)
- ⏳ Player props: Can integrate matchup difficulty
- ⏳ DFS optimization: Can integrate

---

### 14. **Weather Impact**

| Metric | Definition | Calculation | Used In | Source |
|--------|-----------|-------------|---------|--------|
| **Wind Speed** | MPH wind | From games data | Weather Analysis, Backtesting | `weather_impact_backtest.py` |
| **Temperature** | Degrees F | From games data | Weather Analysis | `weather_impact_backtest.py` |
| **Precipitation** | Rain/snow status | From games data | Weather Analysis | `weather_impact_backtest.py` |
| **Is Dome** | Indoor game | From games data | Weather Analysis | Games data |

**Integration Points:**
- ✅ Backtesting: `weather_impact_backtest.py` (1,328 observations)
- ⏳ Game predictions: Can integrate weather adjustments
- ⏳ Player props: Can integrate (passing impacted by wind)

---

### 15. **Injury Impact**

| Metric | Definition | Calculation | Used In | Source |
|--------|-----------|-------------|---------|--------|
| **Target Redistribution** | Target increase when player out | `Teammate Targets - Baseline` | Injury Analysis | `injury_impact_backtest.py` |
| **Usage Redistribution** | Carry increase when RB out | `Teammate Carries - Baseline` | Injury Analysis | `injury_impact_backtest.py` |
| **Injury Status** | OUT/DOUBTFUL/QUESTIONABLE | From injury reports | Injury Analysis | `injury_impact_backtest.py` |

**Integration Points:**
- ✅ Backtesting: `injury_impact_backtest.py` (validated patterns)
- ⏳ Player props: Can integrate (beneficiary boosts)
- ⏳ Game predictions: Can integrate (team impact)

---

## 🔗 System Integration Map

### Current Integration Status

```
┌─────────────────────────────────────────────────────────────┐
│                    METRICS SOURCES                           │
├─────────────────────────────────────────────────────────────┤
│ Play-by-Play (2025.parquet)                                 │
│  ├─ EPA, Success Rate, Down/Distance                        │
│  ├─ Passing: Completions, Air Yards, YAC                    │
│  ├─ Rushing: Carries, Yards, Success                        │
│  ├─ Turnovers: INTs, Fumbles                                │
│  └─ Red Zone: Plays inside 20                               │
│                                                              │
│ Player Stats (player_stats_YYYY_all.csv)                    │
│  ├─ Aggregated stats by week                                │
│  ├─ Fantasy points, yards, TDs                              │
│  └─ Team/opponent tracking                                  │
│                                                              │
│ Games (games_YYYY.csv)                                       │
│  ├─ Scores, dates, locations                                │
│  ├─ Weather data                                             │
│  └─ Game context (primetime, division, etc.)                │
│                                                              │
│ Injuries (injuries_YYYY.csv)                                 │
│  └─ Player status, reports                                  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│               METRICS CALCULATORS                            │
├─────────────────────────────────────────────────────────────┤
│ ✅ advanced_team_metrics.py                                 │
│    → Success Rate, EPA, Red Zone, 3rd Down, YPC, etc.       │
│                                                              │
│ ✅ team_matchup_analyzer.py                                 │
│    → PPG, YPG, H2H, Home/Away, Momentum                     │
│                                                              │
│ ✅ extract_player_pbp_features.py                           │
│    → Rolling avgs, EPA, CPOE, usage                         │
│                                                              │
│ ✅ defense_matchup_backtest.py                              │
│    → Positional defense rankings                            │
│                                                              │
│ ✅ injury_impact_backtest.py                                │
│    → Target/usage redistribution                            │
│                                                              │
│ ✅ weather_impact_backtest.py                               │
│    → Weather coefficients                                   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  USAGE POINTS                                │
├─────────────────────────────────────────────────────────────┤
│ Player Props                                                 │
│  ✅ train_passing_model.py (EPA, CPOE, rolling avgs)        │
│  ⏳ Can add: Success Rate, Red Zone, 3rd Down               │
│                                                              │
│ Team Matchups                                                │
│  ✅ team_matchup_analyzer.py (comprehensive)                │
│  ✅ advanced_team_metrics.py (efficiency metrics)           │
│                                                              │
│ Game Predictions                                             │
│  ⏳ Can integrate: All efficiency metrics                   │
│  ⏳ Can integrate: Pace, turnover margin                    │
│                                                              │
│ Backtesting                                                  │
│  ✅ 5 modules using various metrics                         │
│  ✅ Validated against 24,186 observations                   │
│                                                              │
│ Deep Analysis                                                │
│  ⏳ Can integrate: Advanced metrics throughout              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📈 Integration Opportunities

### High-Priority (Easy Wins)

**1. Add Efficiency Metrics to Player Props**
- **What:** Success Rate, Red Zone stats to player models
- **Why:** Better situational awareness
- **How:** Import from `advanced_team_metrics.py`
- **Impact:** Could improve QB/RB/WR prop accuracy

**2. Add Pace Metrics to Game Totals**
- **What:** Plays per game, time of possession
- **Why:** Pace directly impacts total scoring opportunities
- **How:** Use from `advanced_team_metrics.py`
- **Impact:** Better over/under predictions

**3. Add Turnover Margin to Game Spreads**
- **What:** Team turnover differential
- **Why:** Turnovers are ~4 points each
- **How:** Use from `advanced_team_metrics.py`
- **Impact:** Better spread predictions

### Medium-Priority (Moderate Effort)

**4. Integrate Defense Matchups to Player Props**
- **What:** Use positional defense rankings
- **Why:** Adjust player projections for matchup difficulty
- **How:** Join player with opponent defense metrics
- **Impact:** More accurate game-specific props

**5. Add Red Zone Metrics to TD Props**
- **What:** Team red zone TD%, player red zone targets
- **Why:** TD probability depends on red zone efficiency
- **How:** Filter pbp for red zone, calculate rates
- **Impact:** Better anytime TD scorer predictions

**6. Integrate Weather to Passing Props**
- **What:** Wind speed impact on passing yards
- **Why:** Strong correlation in backtesting
- **How:** Use weather coefficients from backtesting
- **Impact:** Better accuracy in bad weather games

### Long-Term (Complex Integration)

**7. Unified Metrics API**
- **What:** Single interface for all metrics
- **Why:** Avoid redundant calculations
- **How:** Create `MetricsRegistry` class
- **Impact:** Cleaner code, faster performance

**8. Real-Time Metric Updates**
- **What:** Update metrics as season progresses
- **Why:** Keep current with latest data
- **How:** Scheduled jobs to recalculate
- **Impact:** Always current predictions

**9. Cross-Metric Correlations**
- **What:** Analyze which metrics predict best
- **Why:** Focus on highest-signal metrics
- **How:** Statistical analysis across all metrics
- **Impact:** Optimize model features

---

## 🎯 Recommended Next Steps

1. **✅ COMPLETE:** Advanced metrics calculator created
2. **✅ COMPLETE:** Team matchup analyzer created
3. **⏳ TODO:** Integrate efficiency metrics into player prop models
4. **⏳ TODO:** Add pace/turnover metrics to game predictions
5. **⏳ TODO:** Create unified metrics API
6. **⏳ TODO:** Build metrics dashboard/visualization

---

## 📝 Usage Examples

### Get All Available Metrics for a Team

```python
from backend.analysis.advanced_team_metrics import AdvancedTeamMetricsCalculator
from backend.analysis.team_matchup_analyzer import TeamMatchupAnalyzer

# Advanced efficiency metrics
calc = AdvancedTeamMetricsCalculator(season=2025)
metrics = calc.calculate_team_metrics('KC')

print(f"Success Rate: {metrics['success_rate_offense']:.1%}")
print(f"EPA/play: {metrics['epa_per_play_offense']:+.3f}")
print(f"Red Zone TD%: {metrics['red_zone_td_pct']:.1%}")
print(f"3rd Down%: {metrics['third_down_pct']:.1%}")

# Basic team performance
analyzer = TeamMatchupAnalyzer(season=2025)
profile = analyzer.team_profiles['KC']

print(f"PPG: {profile.points_per_game:.1f}")
print(f"YPG: {profile.yards_per_game:.1f}")
print(f"Home PPG: {profile.home_ppg:.1f}")
```

### Compare Teams Across All Metrics

```python
comparison = calc.compare_teams('KC', 'BUF')

print("KC Advantages:")
for adv in comparison['advantages_a']:
    print(f"  {adv}")

print("BUF Advantages:")
for adv in comparison['advantages_b']:
    print(f"  {adv}")
```

### Use Metrics in Predictions

```python
# Example: Adjust player prop for matchup quality
from backend.modeling.train_passing_model import train_passing_model

# Get team metrics
kc_metrics = calc.calculate_team_metrics('KC')
opponent_metrics = calc.calculate_team_metrics('BUF')

# Adjust QB projection based on opponent pass defense
qb_projection = base_projection
if opponent_metrics['success_rate_defense'] < 0.45:  # Good defense
    qb_projection *= 0.95  # Reduce 5%
elif opponent_metrics['success_rate_defense'] > 0.52:  # Weak defense
    qb_projection *= 1.05  # Increase 5%
```

---

## 🔍 Data Quality Notes

### Metric Completeness

| Season | PBP Data | Player Stats | Games | Injuries |
|--------|----------|--------------|-------|----------|
| 2020 | ✅ Complete | ✅ Complete | ✅ Complete | ✅ Complete |
| 2021 | ✅ Complete | ✅ Complete | ✅ Complete | ✅ Complete |
| 2022 | ✅ Complete | ✅ Complete | ✅ Complete | ✅ Complete |
| 2023 | ✅ Complete | ✅ Complete | ✅ Complete | ✅ Complete |
| 2024 | ✅ Complete | ✅ Complete | ✅ Complete | ✅ Complete |
| 2025 | ⚠️ Partial (Weeks 1-12) | ⚠️ Partial | ⚠️ Partial | ⚠️ Partial |

### Known Limitations

- **Time of Possession:** Placeholder only, need drive-level calculation
- **Snap Counts:** Not currently tracked
- **Route Trees:** No route data available
- **Coverage Stats:** No coverage tracking data
- **O-Line/D-Line:** Need NextGen Stats for pressure/sacks allowed

---

## 📚 Documentation References

- **Team Matchups:** `team_matchup_analyzer.py`
- **Advanced Metrics:** `advanced_team_metrics.py`
- **Player Features:** `extract_player_pbp_features.py`
- **Backtesting:** `outputs/backtesting/BACKTESTING_REPORT.md`
- **Training Guide:** `WEEKLY_UPDATE_GUIDE.md`
