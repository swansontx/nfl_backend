# NFL Metrics System - Quick Reference

One-page reference for the unified metrics API.

---

## 🚀 Instant Start

```python
from backend.metrics.unified_metrics_api import get_metrics_api
api = get_metrics_api(season=2025)

# Get metrics
metrics = api.get_team_metrics('KC')
```

---

## 📊 Available Metrics (37+)

### Team Efficiency (7)
```python
metrics['success_rate_offense']      # % successful plays
metrics['success_rate_defense']      # % opp successful plays
metrics['epa_per_play_offense']      # Expected points added
metrics['epa_per_play_defense']      # EPA allowed
metrics['completion_pct']            # Pass completion %
metrics['yards_per_attempt']         # Yards per pass attempt
metrics['yards_per_carry']           # Yards per rush
```

### Pace (2)
```python
metrics['plays_per_game']            # Offensive pace
metrics['time_of_possession_pct']    # % game possession
```

### Turnovers (3)
```python
metrics['turnover_margin']           # +/- differential
metrics['turnover_rate']             # TOs per 100 plays
metrics['takeaway_rate']             # Takeaways per 100 plays
```

### Red Zone (3)
```python
metrics['red_zone_td_pct']          # TD % in red zone
metrics['red_zone_score_pct']       # Any score % in red zone
metrics['red_zone_attempts']        # Trips inside 20
```

### Third Down (2)
```python
metrics['third_down_pct']           # Conversion rate
metrics['third_down_attempts']      # Total 3rd downs
```

### Explosive Plays (2)
```python
metrics['explosive_play_rate']           # % plays 20+ (pass) or 10+ (rush)
metrics['explosive_plays_allowed_rate']  # % opp explosive plays
```

### Team Performance (6)
```python
metrics['points_per_game']          # PPG
metrics['yards_per_game']           # YPG
metrics['passing_yards_per_game']   # Pass YPG
metrics['rushing_yards_per_game']   # Rush YPG
metrics['home_ppg']                 # Home PPG
metrics['away_ppg']                 # Away PPG
```

---

## 🎯 Common Tasks

### Get Team Metrics
```python
# Full season
all_metrics = api.get_team_metrics('KC')

# Recent 4 weeks
recent = api.get_team_metrics('KC', weeks=[9, 10, 11, 12])
```

### Compare Teams
```python
comp = api.compare_teams('KC', 'BUF')
print(f"KC advantages: {comp['advantages_a']}")
print(f"BUF advantages: {comp['advantages_b']}")
```

### Enrich Player Data
```python
import pandas as pd
df = pd.read_csv('inputs/player_stats_2025.csv')
enriched = api.enrich_player_features(df, recency_weeks=4)
# Adds 23 columns
```

### Get Player Context
```python
context = api.get_player_context(
    player_id='00-0036355',
    team='KC',
    opponent='BUF',
    position='QB',
    week=13
)
```

### Game Metrics
```python
game = api.get_game_metrics('KC', 'BUF', week=13)
pace = game['summary']['pace']['combined_pace']
to_diff = game['summary']['turnovers']['margin_differential']
```

### Full Matchup Analysis
```python
matchup = api.analyze_matchup('KC', 'BUF', week=13)
```

### Discover Metrics
```python
# List all available
available = api.get_available_metrics()

# Get metric info
info = api.get_metric_info('success_rate_offense')
print(info['description'])
```

### API Status
```python
summary = api.get_summary()
print(f"Cached: {summary['cached_items']} items")
print(f"Calculators: {summary['calculators']}")
```

---

## 🎮 Enhanced Predictions

### Player Props
```python
from backend.orchestration.picks_pipeline import PicksPipeline

pipeline = PicksPipeline(season=2025, bankroll=1000, min_edge=3.0)
report = pipeline.generate_picks(week=13)
# Automatically uses 23 team metrics per prediction
```

### Game Predictions
```python
from backend.analysis.game_markets import GameMarketAnalyzer

analyzer = GameMarketAnalyzer(season=2025, use_enhanced_metrics=True)
prediction = analyzer.predict_game_outcome('KC', 'BUF', week=13)
# Automatically applies pace, turnovers, efficiency
```

---

## 📐 Key Algorithms

### Pace → Total
```
adjustment = ((combined_pace - 65) / 10) × 3.5 points
```

### Turnover → Spread
```
adjustment = (margin_differential) × 2.5 points
```

### EPA → Spread
```
adjustment = (epa_differential) × 65 plays
```

### Success Rate → Spread
```
adjustment = (rate_diff / 0.05) × 1.0 points
```

### Defense Matchup → Props
```
factor = yards_allowed / league_average
clamped to [0.70, 1.30]
```

---

## 🏆 Best Practices

✅ **DO:**
- Use `get_metrics_api()` singleton
- Leverage caching (335x speedup)
- Use recency for current form
- Batch DataFrame enrichment

❌ **DON'T:**
- Create multiple API instances
- Enrich DataFrames row-by-row
- Forget to check metric availability
- Skip cache clearing between weeks

---

## 🐛 Quick Troubleshooting

**"No play-by-play data"**
→ Missing `inputs/play_by_play_2025.parquet`
→ System works but returns defaults

**Metric returns 0**
→ Check team abbreviation
→ Check week range

**Slow performance**
→ Use singleton: `get_metrics_api()`
→ Check cache: `api.get_summary()`

**"Could not initialize X analyzer"**
→ Calculator is optional
→ 30+ metrics still available

---

## 📦 Player Props Features (23)

When you use `enrich_player_features()`, these columns are added:

**Team Offense (9)**
- `team_success_rate`, `team_epa_per_play`, `team_red_zone_td_pct`
- `team_third_down_pct`, `team_explosive_play_rate`, `team_completion_pct`
- `team_yards_per_attempt`, `team_yards_per_carry`, `team_yards_after_catch_per_comp`

**Opponent Defense (5)**
- `opp_def_success_rate`, `opp_def_epa_allowed`, `opp_def_explosive_allowed_rate`
- `opp_def_third_down_allowed`, `opp_def_takeaway_rate`

**Pace (2)**
- `team_plays_per_game`, `opp_plays_per_game`

**Turnovers (1)**
- `team_turnover_rate`

**Matchup Edges (3)**
- `pass_efficiency_edge`, `rush_efficiency_edge`, `red_zone_matchup`

**Defense Matchup (3)**
- `defense_matchup_factor`, `defense_matchup_rank`, `defense_yards_allowed_vs_pos`

---

## 🎯 Prop-Type Specific Metrics

**Passing Props** (8 metrics)
- success_rate, completion_pct, yards_per_attempt
- opp_def_success_rate, pass_efficiency_edge, plays_per_game
- defense_matchup_factor, defense_yards_allowed_vs_pos

**Rushing Props** (7 metrics)
- yards_per_carry, success_rate, rush_efficiency_edge
- plays_per_game, opp_def_success_rate
- defense_matchup_factor, defense_yards_allowed_vs_pos

**Receiving Props** (8 metrics)
- completion_pct, yards_per_attempt, pass_efficiency_edge
- explosive_play_rate, opp_def_explosive_allowed_rate, plays_per_game
- defense_matchup_factor, defense_yards_allowed_vs_pos

**TD Props** (5 metrics)
- red_zone_td_pct, red_zone_matchup, success_rate
- opp_def_success_rate, defense_matchup_factor

---

## 📈 Performance Stats

- **Caching**: 335x faster on repeated calls
- **Coverage**: 37+ metrics per team
- **Enrichment**: 23 features per player
- **Calculators**: 5 integrated systems
- **Confidence**: 80% (enhanced) vs 75% (baseline)

---

## 🔗 Related Docs

- **Full Guide**: `docs/IMPLEMENTATION_GUIDE.md`
- **Metric Catalog**: `METRICS_REGISTRY.md`
- **Tests**: `test_unified_metrics_api.py`

---

**Last Updated:** 2025-11-28
