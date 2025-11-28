# NFL Backend Metrics System - Implementation Guide

**Complete guide for using the new unified metrics infrastructure**

Last Updated: 2025-11-28

---

## 🚀 Quick Start

The fastest way to get started with the metrics system:

```python
from backend.metrics.unified_metrics_api import get_metrics_api

# Initialize once (singleton pattern handles caching)
api = get_metrics_api(season=2025)

# Get all metrics for a team
metrics = api.get_team_metrics('KC')
print(f"Success Rate: {metrics['success_rate_offense']:.1%}")
print(f"Plays/Game: {metrics['plays_per_game']:.1f}")
```

That's it! You now have access to 37+ metrics for any team.

---

## 📋 Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Core Concepts](#core-concepts)
3. [Common Use Cases](#common-use-cases)
4. [Player Props Workflow](#player-props-workflow)
5. [Game Predictions Workflow](#game-predictions-workflow)
6. [Migration Guide](#migration-guide)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)
9. [API Reference](#api-reference)

---

## Installation & Setup

### Prerequisites

Ensure you have the required data files in `inputs/`:
- `play_by_play_{season}.parquet` - For advanced metrics (EPA, success rate, etc.)
- `player_stats_{season}.csv` - For player features
- `{season}_schedule.parquet` - For game data

### Import the API

```python
# Recommended: Use singleton for automatic caching
from backend.metrics.unified_metrics_api import get_metrics_api

api = get_metrics_api(season=2025)
```

### Verify Setup

```python
# Check which calculators are available
summary = api.get_summary()
print(summary['calculators'])

# Should show:
# {
#   'team_metrics': True,
#   'matchup_analyzer': True/False,
#   'defense_analyzer': True,
#   'team_features_engine': True,
#   'game_metrics_engine': True
# }
```

---

## Core Concepts

### 1. Metric Categories

The system provides metrics in 6 categories:

**Team Efficiency** (7 metrics)
- `success_rate_offense`, `success_rate_defense`
- `epa_per_play_offense`, `epa_per_play_defense`
- `completion_pct`, `yards_per_attempt`, `yards_per_carry`

**Pace** (2 metrics)
- `plays_per_game`, `time_of_possession_pct`

**Turnovers** (3 metrics)
- `turnover_margin`, `turnover_rate`, `takeaway_rate`

**Situational** (3 metrics)
- `red_zone_td_pct`, `third_down_pct`, `explosive_play_rate`

**Defense** (3 metrics per position)
- `defense_matchup_factor`, `defense_matchup_rank`, `defense_yards_allowed_vs_pos`

**Team Performance** (6 metrics)
- `points_per_game`, `yards_per_game`, `passing_yards_per_game`
- `rushing_yards_per_game`, `home_ppg`, `away_ppg`

### 2. Caching System

All metrics are automatically cached for performance:

```python
# First call: Calculates metrics (~30ms)
metrics1 = api.get_team_metrics('KC')

# Second call: Returns cached result (~0.01ms)
metrics2 = api.get_team_metrics('KC')  # 335x faster!

# Clear cache if needed
api.clear_cache()
```

### 3. Recency Weighting

Many methods support recent-week filtering:

```python
# Full season metrics
full_season = api.get_team_metrics('KC')

# Last 4 weeks only (for current form)
recent = api.get_team_metrics('KC', weeks=[9, 10, 11, 12])
```

---

## Common Use Cases

### Use Case 1: Team Research

```python
from backend.metrics.unified_metrics_api import get_metrics_api

api = get_metrics_api(season=2025)

# Get comprehensive team report
kc_metrics = api.get_team_metrics('KC')

print("Kansas City Chiefs - 2025 Season")
print("="*50)
print(f"\nOffense:")
print(f"  Points/Game: {kc_metrics['points_per_game']:.1f}")
print(f"  Success Rate: {kc_metrics['success_rate_offense']:.1%}")
print(f"  EPA/Play: {kc_metrics['epa_per_play_offense']:+.3f}")
print(f"  Pace: {kc_metrics['plays_per_game']:.1f} plays/game")

print(f"\nDefense:")
print(f"  Points Allowed: {kc_metrics['points_allowed_per_game']:.1f}")
print(f"  Success Rate Allowed: {kc_metrics['success_rate_defense']:.1%}")
print(f"  EPA Allowed: {kc_metrics['epa_per_play_defense']:+.3f}")

print(f"\nTurnovers:")
print(f"  Margin: {kc_metrics['turnover_margin']:+d}")
print(f"  Turnover Rate: {kc_metrics['turnover_rate']:.2%}")

print(f"\nRed Zone:")
print(f"  TD %: {kc_metrics['red_zone_td_pct']:.1%}")
print(f"  Score %: {kc_metrics['red_zone_score_pct']:.1%}")
```

### Use Case 2: Head-to-Head Comparison

```python
# Compare two teams for a matchup
comparison = api.compare_teams('KC', 'BUF')

print("KC vs BUF Comparison")
print("="*50)

# KC advantages
print(f"\nKC Advantages ({len(comparison['advantages_a'])} metrics):")
for metric in comparison['advantages_a'][:5]:
    data = comparison['metrics'][metric]
    print(f"  {metric}: KC {data['KC']:.2f} vs BUF {data['BUF']:.2f}")

# BUF advantages
print(f"\nBUF Advantages ({len(comparison['advantages_b'])} metrics):")
for metric in comparison['advantages_b'][:5]:
    data = comparison['metrics'][metric]
    print(f"  {metric}: BUF {data['BUF']:.2f} vs KC {data['KC']:.2f}")
```

### Use Case 3: Week-Over-Week Trends

```python
import pandas as pd

# Track team performance over time
weeks_data = []

for week in range(1, 13):
    metrics = api.get_team_metrics('KC', weeks=[week])
    weeks_data.append({
        'week': week,
        'epa_per_play': metrics.get('epa_per_play_offense', 0),
        'success_rate': metrics.get('success_rate_offense', 0),
        'plays_per_game': metrics.get('plays_per_game', 0)
    })

df = pd.DataFrame(weeks_data)
print(df)

# Find hot/cold streaks
recent_4 = df.tail(4)['epa_per_play'].mean()
season_avg = df['epa_per_play'].mean()

if recent_4 > season_avg * 1.1:
    print("Team is HOT! (Recent EPA 10%+ above season average)")
elif recent_4 < season_avg * 0.9:
    print("Team is COLD (Recent EPA 10%+ below season average)")
```

---

## Player Props Workflow

### Step 1: Load and Enrich Player Data

```python
import pandas as pd
from backend.metrics.unified_metrics_api import get_metrics_api

api = get_metrics_api(season=2025)

# Load raw player stats
player_stats = pd.read_csv('inputs/player_stats_2025.csv')
print(f"Original: {len(player_stats.columns)} columns")

# Enrich with team metrics (adds 23 columns)
enriched_stats = api.enrich_player_features(player_stats, recency_weeks=4)
print(f"Enriched: {len(enriched_stats.columns)} columns")
print(f"New metrics: {len(enriched_stats.columns) - len(player_stats.columns)}")

# New columns include:
# - team_success_rate, team_epa_per_play, team_red_zone_td_pct
# - opp_def_success_rate, opp_def_epa_allowed
# - defense_matchup_factor, defense_matchup_rank
# - pass_efficiency_edge, rush_efficiency_edge
# ... and 14 more
```

### Step 2: Analyze Individual Player Context

```python
# Get complete context for a specific player
context = api.get_player_context(
    player_id='00-0036355',  # Patrick Mahomes
    team='KC',
    opponent='BUF',
    position='QB',
    week=13
)

print(f"Player: {context['player_id']}")
print(f"Matchup: {context['team']} @ {context['opponent']}")

# Team offensive metrics
team = context['team_metrics']
print(f"\nTeam Offense:")
print(f"  Success Rate: {team['success_rate_offense']:.1%}")
print(f"  Yards/Attempt: {team['yards_per_attempt']:.2f}")
print(f"  Red Zone TD%: {team['red_zone_td_pct']:.1%}")

# Opponent defense
opp = context['opponent_metrics']
print(f"\nOpponent Defense:")
print(f"  Success Rate Allowed: {opp['success_rate_defense']:.1%}")
print(f"  EPA Allowed: {opp['epa_per_play_defense']:+.3f}")

# Defense matchup rating
matchup = context['defense_matchup']
print(f"\nDefense Matchup:")
print(f"  Factor: {matchup['matchup_factor']:.2f}")
print(f"  Rank: {matchup['league_rank']}")

# Interpret matchup factor
if matchup['matchup_factor'] < 0.9:
    print("  ⚠️ TOUGH matchup (elite defense vs position)")
elif matchup['matchup_factor'] > 1.1:
    print("  ✅ FAVORABLE matchup (weak defense vs position)")
else:
    print("  ➖ AVERAGE matchup")
```

### Step 3: Make Predictions with Context

```python
from backend.orchestration.picks_pipeline import PicksPipeline

# Initialize pipeline (automatically uses team metrics)
pipeline = PicksPipeline(
    season=2025,
    bankroll=1000.0,
    min_edge=3.0
)

# Generate picks - team metrics are automatically included
report = pipeline.generate_picks(week=13)

print(f"Generated {len(report.single_picks)} value picks")
print(f"Expected value: ${report.total_edge_dollars:.2f}")

# Each pick now includes 23 team context metrics:
best_pick = report.best_single
if best_pick:
    print(f"\nBest Pick: {best_pick.player_name}")
    print(f"  Prop: {best_pick.prop_type} {best_pick.side} {best_pick.line}")
    print(f"  Edge: {best_pick.edge:.1f}%")
    print(f"  Grade: {best_pick.grade}")
    # Behind the scenes: prediction used team_success_rate,
    # defense_matchup_factor, pass_efficiency_edge, etc.
```

---

## Game Predictions Workflow

### Step 1: Get Game Metrics

```python
from backend.metrics.unified_metrics_api import get_metrics_api

api = get_metrics_api(season=2025)

# Get comprehensive game metrics
game_metrics = api.get_game_metrics(
    home_team='KC',
    away_team='BUF',
    week=13,
    recency_weeks=4  # Use last 4 weeks
)

# Pace analysis
pace = game_metrics['summary']['pace']
print(f"Pace Analysis:")
print(f"  KC: {pace['home_plays_per_game']:.1f} plays/game")
print(f"  BUF: {pace['away_plays_per_game']:.1f} plays/game")
print(f"  Combined: {pace['combined_pace']:.1f}")
print(f"  vs League Avg: {pace['pace_vs_league_avg']:+.1f}")

# Turnover analysis
to = game_metrics['summary']['turnovers']
print(f"\nTurnover Analysis:")
print(f"  KC Margin: {to['home_margin']:+d}")
print(f"  BUF Margin: {to['away_margin']:+d}")
print(f"  Differential: {to['margin_differential']:+d}")

# Efficiency analysis
eff = game_metrics['summary']['efficiency']
print(f"\nEfficiency:")
print(f"  KC Success Rate: {eff['home_success_rate_off']:.1%}")
print(f"  BUF Success Rate: {eff['away_success_rate_off']:.1%}")
print(f"  KC EPA Edge: {eff['home_epa_edge']:+.3f}")
print(f"  BUF EPA Edge: {eff['away_epa_edge']:+.3f}")
```

### Step 2: Make Enhanced Game Predictions

```python
from backend.analysis.game_markets import GameMarketAnalyzer

# Initialize with enhanced metrics (default)
analyzer = GameMarketAnalyzer(season=2025, use_enhanced_metrics=True)

# Predict game - automatically applies pace, turnovers, efficiency
prediction = analyzer.predict_game_outcome(
    home_team='KC',
    away_team='BUF',
    week=13,
    recent_weeks=4
)

print(f"Game Prediction: KC vs BUF")
print(f"="*50)
print(f"Predicted Score: {prediction.home_team} {prediction.home_score} - {prediction.away_team} {prediction.away_score}")
print(f"Predicted Spread: {prediction.predicted_spread:+.1f} ({prediction.home_team})")
print(f"Predicted Total: {prediction.predicted_total:.1f}")
print(f"Win Probabilities: {prediction.home_win_prob:.1%} / {prediction.away_win_prob:.1%}")
print(f"Confidence: {prediction.confidence:.1%}")

# Behind the scenes:
# - Base prediction from PPG
# - + Pace adjustment to total (+/- 3-6 points typical)
# - + Turnover adjustment to spread (+/- 2.5 pts per margin)
# - + Efficiency adjustments (EPA, success rate, red zone)
```

### Step 3: Analyze Betting Markets

```python
# Analyze spread market
market_spread = -3.0  # KC favored by 3
spread_analysis = analyzer.analyze_spread_market(
    prediction=prediction,
    market_spread=market_spread,
    market_odds=-110
)

print(f"\nSpread Analysis:")
print(f"  Market: {market_spread:+.1f}")
print(f"  Model: {prediction.predicted_spread:+.1f}")
print(f"  Edge: {spread_analysis.edge:.1f} points")
print(f"  Recommendation: {spread_analysis.recommendation}")
print(f"  Reasoning: {spread_analysis.reasoning}")

# Analyze total market
total_analysis = analyzer.analyze_total_market(
    prediction=prediction,
    market_total=49.5,
    over_odds=-110,
    under_odds=-110
)

print(f"\nTotal Analysis:")
print(f"  Market: {49.5}")
print(f"  Model: {prediction.predicted_total:.1f}")
print(f"  Recommendation: {total_analysis.recommendation}")
```

---

## Migration Guide

### From Old Code to Unified API

**❌ OLD WAY:**
```python
# Multiple imports
from backend.analysis.advanced_team_metrics import AdvancedTeamMetricsCalculator
from backend.analysis.team_matchup_analyzer import TeamMatchupAnalyzer
from backend.features.team_metrics_features import TeamMetricsFeatureEngine

# Multiple initializations
pbp_file = Path(f'inputs/play_by_play_2025.parquet')
calc = AdvancedTeamMetricsCalculator(season=2025, pbp_file=pbp_file)
analyzer = TeamMatchupAnalyzer(season=2025)
engine = TeamMetricsFeatureEngine(season=2025)

# Multiple calls
metrics1 = calc.calculate_team_metrics('KC')
profile = analyzer.team_profiles['KC']
# ... manually combine
```

**✅ NEW WAY:**
```python
# Single import
from backend.metrics.unified_metrics_api import get_metrics_api

# Single initialization
api = get_metrics_api(season=2025)

# One call
metrics = api.get_team_metrics('KC')  # All 37+ metrics combined
```

### Migration Checklist

- [ ] Replace multiple calculator imports with `get_metrics_api`
- [ ] Update `get_team_metrics()` calls to use unified API
- [ ] Update player enrichment to use `api.enrich_player_features()`
- [ ] Update game predictions to use `GameMarketAnalyzer(use_enhanced_metrics=True)`
- [ ] Remove manual calculator initialization code
- [ ] Test that metrics are still accessible (same keys)
- [ ] Verify caching is working (check performance)

---

## Best Practices

### 1. Use the Singleton Pattern

```python
# ✅ GOOD: Use singleton (automatic caching across modules)
from backend.metrics.unified_metrics_api import get_metrics_api
api = get_metrics_api(season=2025)

# ❌ BAD: Create new instances (loses caching benefits)
from backend.metrics.unified_metrics_api import MetricsAPI
api = MetricsAPI(season=2025)  # Creates new cache
```

### 2. Leverage Recency for Current Form

```python
# For predictions, use recent weeks
recent_metrics = api.get_team_metrics('KC', weeks=[9, 10, 11, 12])

# For season analysis, use full season
season_metrics = api.get_team_metrics('KC')  # All weeks
```

### 3. Check Metric Availability

```python
# Always check if metric exists before using
metrics = api.get_team_metrics('KC')

if 'success_rate_offense' in metrics:
    success_rate = metrics['success_rate_offense']
else:
    # Fallback if PBP data not available
    success_rate = 0.45  # League average
```

### 4. Batch Operations

```python
# ✅ GOOD: Enrich entire DataFrame at once
enriched_df = api.enrich_player_features(player_df)

# ❌ BAD: Enrich row by row (much slower)
for idx, row in player_df.iterrows():
    context = api.get_player_context(...)  # Don't do this!
```

### 5. Monitor Cache Size

```python
# Periodically check cache usage
summary = api.get_summary()
print(f"Cached items: {summary['cached_items']}")

# Clear if needed (e.g., after processing each week)
api.clear_cache()
```

---

## Troubleshooting

### Issue: "No play-by-play data" warning

**Problem:** Missing `play_by_play_2025.parquet` file

**Solution:**
```python
# Check if file exists
from pathlib import Path
pbp_file = Path('inputs/play_by_play_2025.parquet')
if not pbp_file.exists():
    print(f"Missing: {pbp_file}")
    # Download or generate play-by-play data
```

### Issue: Metric returns 0 or default value

**Problem:** Team abbreviation mismatch or no data

**Solution:**
```python
# Check team abbreviations
metrics = api.get_team_metrics('KC')
if metrics.get('games_played', 0) == 0:
    print("No data found for 'KC'")
    # Try different abbreviation: 'KAN', 'Kansas City', etc.
```

### Issue: Slow performance

**Problem:** Cache not being used

**Solution:**
```python
# Use singleton pattern
from backend.metrics.unified_metrics_api import get_metrics_api
api = get_metrics_api()  # Returns same instance

# Check caching is enabled
summary = api.get_summary()
assert summary['cache_enabled'] == True
```

### Issue: "Could not initialize matchup analyzer"

**Problem:** Missing schedule or games data

**Solution:**
```python
# This calculator is optional, system works without it
summary = api.get_summary()
if not summary['calculators']['matchup_analyzer']:
    print("Matchup analyzer unavailable (OK)")
    # You still get 30+ other metrics
```

---

## API Reference

### MetricsAPI Class

#### `get_team_metrics(team, weeks=None, include_defense=True)`
Get comprehensive team metrics.

**Parameters:**
- `team` (str): Team abbreviation ('KC', 'BUF', etc.)
- `weeks` (List[int], optional): Specific weeks for recency
- `include_defense` (bool): Include defensive metrics

**Returns:** Dict with 37+ metrics

**Example:**
```python
metrics = api.get_team_metrics('KC', weeks=[9, 10, 11, 12])
```

#### `compare_teams(team_a, team_b, weeks=None)`
Compare two teams across all metrics.

**Parameters:**
- `team_a` (str): First team
- `team_b` (str): Second team
- `weeks` (List[int], optional): Weeks for comparison

**Returns:** Dict with comparison results

**Example:**
```python
comp = api.compare_teams('KC', 'BUF')
print(comp['advantages_a'])  # KC advantages
```

#### `enrich_player_features(player_features, recency_weeks=4)`
Enrich player DataFrame with team metrics.

**Parameters:**
- `player_features` (pd.DataFrame): Player stats DataFrame
- `recency_weeks` (int): Number of recent weeks for team metrics

**Returns:** Enhanced DataFrame with 23 additional columns

**Example:**
```python
enriched = api.enrich_player_features(player_df, recency_weeks=4)
```

#### `get_player_context(player_id, team, opponent, position, week, recency_weeks=4)`
Get team context for a specific player.

**Parameters:**
- `player_id` (str): Player ID
- `team` (str): Player's team
- `opponent` (str): Opponent team
- `position` (str): Player position ('QB', 'RB', 'WR', 'TE')
- `week` (int): Current week
- `recency_weeks` (int): Weeks for team metrics

**Returns:** Dict with team context and matchup metrics

**Example:**
```python
context = api.get_player_context(
    player_id='00-0036355',
    team='KC',
    opponent='BUF',
    position='QB',
    week=13
)
```

#### `get_game_metrics(home_team, away_team, week, recency_weeks=4)`
Get comprehensive metrics for a game.

**Parameters:**
- `home_team` (str): Home team
- `away_team` (str): Away team
- `week` (int): Week number
- `recency_weeks` (int): Weeks for recency metrics

**Returns:** Dict with pace, turnovers, efficiency summary

**Example:**
```python
game = api.get_game_metrics('KC', 'BUF', week=13)
print(game['summary']['pace']['combined_pace'])
```

#### `analyze_matchup(home_team, away_team, week)`
Get complete matchup analysis.

**Parameters:**
- `home_team` (str): Home team
- `away_team` (str): Away team
- `week` (int): Week number

**Returns:** Dict with H2H, game metrics, comparison

**Example:**
```python
matchup = api.analyze_matchup('KC', 'BUF', week=13)
```

#### `get_available_metrics()`
Get list of all available metrics by category.

**Returns:** Dict mapping category to list of metric names

**Example:**
```python
available = api.get_available_metrics()
print(f"Total: {sum(len(m) for m in available.values())}")
```

#### `get_metric_info(metric_name)`
Get documentation for a specific metric.

**Parameters:**
- `metric_name` (str): Name of the metric

**Returns:** Dict with description, calculation, range, usage

**Example:**
```python
info = api.get_metric_info('success_rate_offense')
print(info['description'])
```

#### `get_summary()`
Get API status and configuration.

**Returns:** Dict with season, cache status, available calculators

**Example:**
```python
summary = api.get_summary()
print(summary['calculators'])
```

#### `clear_cache()`
Clear all cached metrics.

**Example:**
```python
api.clear_cache()
```

---

## Next Steps

Now that you understand the metrics system:

1. **Start Simple**: Use `get_team_metrics()` to explore data
2. **Build Dashboards**: Create visualizations of team metrics
3. **Enhance Models**: Retrain models with the 23 new player features
4. **Backtest**: Validate prediction improvements vs historical lines
5. **Automate**: Build pipelines using the unified API

## Support

- **Documentation**: `METRICS_REGISTRY.md` - Complete metric catalog
- **Examples**: See "Usage Examples" section in METRICS_REGISTRY.md
- **Tests**: Run test suites to verify your setup:
  - `python test_metrics_integration.py`
  - `python test_game_metrics_integration.py`
  - `python test_unified_metrics_api.py`

---

**Built with ❤️ for data-driven NFL analysis**
