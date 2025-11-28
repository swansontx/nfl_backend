# NFL Backend Metrics Registry

**Central catalog of all available metrics and where they're used across the system.**

Last Updated: 2025-11-28
**Recent Updates:**
- ✅ Team efficiency metrics and defense matchups integrated into player props pipeline
- ✅ Pace, turnover margin, and efficiency metrics integrated into game predictions (spreads/totals)
- ✅ Unified Metrics API created for centralized access to all metrics

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

### ✅ Completed Integrations (Nov 28, 2025)

**1. Team Efficiency Metrics to Player Props** ✅
- **Status:** COMPLETED
- **What:** Success rate, EPA, red zone %, 3rd down %, passing/rushing efficiency
- **Implementation:** `TeamMetricsFeatureEngine` in `backend/features/team_metrics_features.py`
- **Features Added:** 23 new team-level metrics enriching player predictions
- **Prop Coverage:** Passing (8 metrics), Rushing (7 metrics), Receiving (8 metrics), TD (5 metrics)
- **Impact:** Contextual team performance now included in all player prop predictions

**4. Defense Matchups to Player Props** ✅
- **Status:** COMPLETED
- **What:** Position-specific defense rankings and matchup difficulty factors
- **Implementation:** Integrated `DefenseMatchupAnalyzer` into `TeamMetricsFeatureEngine`
- **Features Added:** `defense_matchup_factor`, `defense_matchup_rank`, `defense_yards_allowed_vs_pos`
- **Position Coverage:** QB, RB, WR, TE with position-specific defensive ratings
- **Impact:** Player projections now adjusted for opponent defensive strength

**2. Pace Metrics to Game Totals** ✅
- **Status:** COMPLETED (Nov 28, 2025)
- **What:** Plays per game, time of possession affecting total scoring
- **Implementation:** `GameMetricsEngine` in `backend/features/game_metrics_features.py`
- **Algorithm:** Each +10 plays/game ≈ +3.5 points to total
- **Impact:** Game totals now adjusted for team pace (+5.8 in fast-paced matchups)

**3. Turnover Margin to Game Spreads** ✅
- **Status:** COMPLETED (Nov 28, 2025)
- **What:** Team turnover differential impacting point spreads
- **Implementation:** Integrated into `GameMetricsEngine` and `GameMarketAnalyzer`
- **Algorithm:** Each +1 turnover margin ≈ +2.5 points to spread
- **Impact:** Spreads now reflect ball security advantage (conservative multiplier)

### High-Priority (Remaining)

### Medium-Priority (Moderate Effort)

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

**7. Unified Metrics API** ✅
- **Status:** COMPLETED (Nov 28, 2025)
- **What:** Single interface for all metrics across the system
- **Implementation:** `MetricsAPI` class in `backend/metrics/unified_metrics_api.py`
- **Features**:
  * Single point of access to all metric calculators
  * Automatic caching (335x speedup on repeated calls)
  * Team metrics, player enrichment, game metrics, matchup analysis
  * Metric discovery and documentation
  * Singleton pattern for easy reuse
- **Impact:** Simplified codebase, better performance, easier to use

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
3. **✅ COMPLETE:** Team metrics feature engineering module created (Nov 28, 2025)
4. **✅ COMPLETE:** Integrated efficiency + defense metrics into player props (Nov 28, 2025)
5. **✅ COMPLETE:** Game metrics feature engineering module created (Nov 28, 2025)
6. **✅ COMPLETE:** Integrated pace/turnover/efficiency metrics into game predictions (Nov 28, 2025)
7. **✅ COMPLETE:** Created unified metrics API (Nov 28, 2025)
8. **⏳ TODO:** Train models with new enhanced features
9. **⏳ TODO:** Build metrics dashboard/visualization

---

## 📝 Usage Examples

### ⭐ RECOMMENDED: Using the Unified Metrics API (Nov 28, 2025)

**The easiest way to access all metrics in the system:**

```python
from backend.metrics.unified_metrics_api import MetricsAPI, get_metrics_api

# Initialize the API (or use singleton)
api = MetricsAPI(season=2025)
# Or use singleton: api = get_metrics_api(season=2025)

# 1. Get team metrics (37+ metrics in one call)
kc_metrics = api.get_team_metrics('KC')
print(f"Success Rate: {kc_metrics['success_rate_offense']:.1%}")
print(f"EPA/play: {kc_metrics['epa_per_play_offense']:+.3f}")
print(f"Plays/Game: {kc_metrics['plays_per_game']:.1f}")
print(f"TO Margin: {kc_metrics['turnover_margin']:+d}")

# 2. Get recent metrics (last 4 weeks)
recent = api.get_team_metrics('KC', weeks=[9, 10, 11, 12])

# 3. Compare two teams
comparison = api.compare_teams('KC', 'BUF')
print(f"KC Advantages: {len(comparison['advantages_a'])} metrics")
print(f"BUF Advantages: {len(comparison['advantages_b'])} metrics")

# 4. Enrich player features with team context
import pandas as pd
player_df = pd.read_csv('inputs/player_stats_2025.csv')
enriched_df = api.enrich_player_features(player_df, recency_weeks=4)
# Adds 23 team metric columns automatically

# 5. Get player context for predictions
context = api.get_player_context(
    player_id='00-0036355',
    team='KC',
    opponent='BUF',
    position='QB',
    week=13
)
print(f"Defense Matchup Factor: {context['defense_matchup']['matchup_factor']:.2f}")

# 6. Get game prediction metrics
game_metrics = api.get_game_metrics('KC', 'BUF', week=13)
print(f"Combined Pace: {game_metrics['summary']['pace']['combined_pace']:.1f}")
print(f"TO Differential: {game_metrics['summary']['turnovers']['margin_differential']:+d}")

# 7. Full matchup analysis
matchup = api.analyze_matchup('KC', 'BUF', week=13)

# 8. Discover available metrics
available = api.get_available_metrics()
print(f"Available: {sum(len(m) for m in available.values())} metrics")

# 9. Get metric information
info = api.get_metric_info('success_rate_offense')
print(f"{info['name']}: {info['description']}")

# 10. Check API status
summary = api.get_summary()
print(f"Cached items: {summary['cached_items']}")
print(f"Calculators: {summary['calculators']}")
```

**Benefits of Unified API:**
- ✅ Single import, access to everything
- ✅ Automatic caching (335x faster on repeated calls)
- ✅ Consistent interface across all metrics
- ✅ Error handling built-in
- ✅ Easy metric discovery

---

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

### ✨ NEW: Enrich Player Features with Team Metrics (Nov 28, 2025)

```python
from backend.features.team_metrics_features import TeamMetricsFeatureEngine
import pandas as pd

# Initialize feature engine
engine = TeamMetricsFeatureEngine(season=2025, inputs_dir="inputs")

# Enrich entire DataFrame with team metrics
player_stats = pd.read_csv("inputs/player_stats_2025.csv")
enriched_df = engine.enrich_player_dataframe(player_stats, recency_weeks=4)

# New columns automatically added:
# - team_success_rate, team_epa_per_play, team_red_zone_td_pct
# - opp_def_success_rate, opp_def_epa_allowed
# - pass_efficiency_edge, rush_efficiency_edge, red_zone_matchup
# - defense_matchup_factor, defense_matchup_rank, defense_yards_allowed_vs_pos
# ... and 13 more team-level efficiency metrics

print(f"Original columns: {len(player_stats.columns)}")
print(f"Enriched columns: {len(enriched_df.columns)}")
print(f"New metrics added: {len(enriched_df.columns) - len(player_stats.columns)}")

# Or enrich a single player's features
player_features = {
    'player_id': '00-1234',
    'position': 'WR',
    'team': 'KC',
    'opponent_team': 'BUF',
    'week': 13,
    # ... other features
}

enriched = engine.enrich_player_features(
    player_features=player_features,
    team='KC',
    opponent='BUF',
    week=13,
    recency_weeks=4  # Use last 4 weeks for team metrics
)

# Access new metrics
print(f"Team Success Rate: {enriched['team_success_rate']:.1%}")
print(f"Defense Matchup Factor: {enriched['defense_matchup_factor']:.3f}")
print(f"Pass Efficiency Edge: {enriched['pass_efficiency_edge']:+.2f}")
```

### ✨ NEW: Use Enhanced Features in Picks Pipeline (Nov 28, 2025)

```python
from backend.orchestration.picks_pipeline import PicksPipeline

# Pipeline automatically enriches features with team metrics
pipeline = PicksPipeline(
    season=2025,
    models_dir="outputs/models",
    inputs_dir="inputs",
    bankroll=1000.0,
    min_edge=3.0
)

# Generate picks - features automatically enriched with 23 team metrics
picks_report = pipeline.generate_picks(week=13)

print(f"Generated {len(picks_report.single_picks)} value picks")
print(f"Total expected value: ${picks_report.total_edge_dollars:.2f}")

# Each pick now has team metrics incorporated:
# - Passing props: 8 team metrics (success rate, completion %, YPA, etc.)
# - Rushing props: 7 team metrics (YPC, rush efficiency, etc.)
# - Receiving props: 8 team metrics (completion %, explosive plays, etc.)
# - TD props: 5 team metrics (red zone %, matchup factors)
```

### ✨ NEW: Enhanced Game Predictions with Pace & Turnover Metrics (Nov 28, 2025)

```python
from backend.analysis.game_markets import GameMarketAnalyzer

# Initialize with enhanced metrics enabled (default)
analyzer = GameMarketAnalyzer(season=2025, use_enhanced_metrics=True)

# Predict game outcome - automatically includes pace, turnovers, efficiency
prediction = analyzer.predict_game_outcome(
    home_team='KC',
    away_team='BUF',
    week=13,
    recent_weeks=4  # Use last 4 weeks for team metrics
)

print(f"Predicted Score: {prediction.home_team} {prediction.home_score} - {prediction.away_team} {prediction.away_score}")
print(f"Predicted Spread: {prediction.predicted_spread:+.1f} ({prediction.home_team})")
print(f"Predicted Total: {prediction.predicted_total}")
print(f"Win Probabilities: {prediction.home_win_prob:.1%} / {prediction.away_win_prob:.1%}")

# Analyze betting markets
spread_analysis = analyzer.analyze_spread_market(
    prediction=prediction,
    market_spread=-3.0,  # Market has KC favored by 3
    market_odds=-110
)

print(f"Recommendation: {spread_analysis.recommendation}")
print(f"Edge: {spread_analysis.edge:.1f} points")
print(f"Reasoning: {spread_analysis.reasoning}")
```

### ✨ NEW: Direct Enhancement with GameMetricsEngine (Nov 28, 2025)

```python
from backend.features.game_metrics_features import enhance_game_prediction

# Enhance any base prediction with advanced metrics
result = enhance_game_prediction(
    home_team='KC',
    away_team='BUF',
    base_home_score=25.5,
    base_away_score=23.0,
    home_offensive_rating=26.0,
    home_defensive_rating=19.0,
    away_offensive_rating=25.0,
    away_defensive_rating=20.0,
    season=2025,
    recent_weeks=4
)

print(f"Base Total: {result['base_prediction']['total']:.1f}")
print(f"Enhanced Total: {result['enhanced_prediction']['total']:.1f}")
print(f"Pace Adjustment: {result['adjustments']['pace_total_adj']:+.1f} points")

print(f"\nBase Spread: {result['base_prediction']['spread']:+.1f}")
print(f"Enhanced Spread: {result['enhanced_prediction']['spread']:+.1f}")
print(f"Turnover Adjustment: {result['adjustments']['turnover_spread_adj']:+.1f} points")
print(f"Efficiency Adjustment: {result['adjustments']['efficiency_spread_adj']:+.1f} points")

print(f"\nReasoning:")
print(f"  {result['reasoning']['pace']}")
print(f"  {result['reasoning']['turnovers']}")
print(f"  {result['reasoning']['efficiency']}")

# Access detailed metrics breakdown
metrics = result['metrics_summary']
print(f"\nCombined Pace: {metrics['pace']['combined_pace']:.1f} plays/game")
print(f"Turnover Differential: {metrics['turnovers']['margin_differential']:+d}")
print(f"EPA Edge: Home {metrics['efficiency']['home_epa_edge']:+.3f}, Away {metrics['efficiency']['away_epa_edge']:+.3f}")
```

### Key Features in Game Predictions:

**Pace Adjustments to Totals:**
- Fast-paced games (70+ plays/game): +3-6 points to total
- Slow-paced games (60- plays/game): -3-6 points to total
- Algorithm: Each +10 plays ≈ +3.5 points

**Turnover Margin Adjustments to Spreads:**
- Team with +5 TO margin: +12.5 point edge (5 × 2.5)
- Conservative multiplier (2.5 pts/margin) vs. typical 4 pts/turnover
- Reflects season-long ball security advantage

**Efficiency Adjustments (EPA, Success Rate, Red Zone):**
- EPA differential: Direct point impact (EPA × 65 plays/game)
- Success rate edge: Each 5% ≈ 1 point
- Red zone efficiency: Each 10% ≈ 1.5 points
- Explosive play rate: Affects total variance

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
