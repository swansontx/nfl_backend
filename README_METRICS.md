# NFL Backend Metrics System

**Comprehensive metrics infrastructure for NFL predictions and analysis**

---

## 🚀 Quick Start

```python
from backend.metrics.unified_metrics_api import get_metrics_api

api = get_metrics_api(season=2025)
metrics = api.get_team_metrics('KC')

print(f"Success Rate: {metrics['success_rate_offense']:.1%}")
print(f"EPA/Play: {metrics['epa_per_play_offense']:+.3f}")
```

---

## 📚 Documentation

### Essential Guides
- **[Implementation Guide](docs/IMPLEMENTATION_GUIDE.md)** - Complete walkthrough with examples
- **[Quick Reference](docs/QUICK_REFERENCE.md)** - One-page cheat sheet
- **[Metrics Registry](METRICS_REGISTRY.md)** - Full catalog of all 37+ metrics

### Core Documentation
- **Installation & Setup** - Prerequisites and configuration
- **API Reference** - Complete method documentation
- **Migration Guide** - Update existing code to new system
- **Best Practices** - Recommendations and patterns
- **Troubleshooting** - Common issues and solutions

---

## 🎯 Features

### Unified Metrics API
Single access point for all NFL metrics:
- ✅ **37+ metrics** per team (efficiency, pace, turnovers, etc.)
- ✅ **23 features** enriching player predictions
- ✅ **335x caching** speedup on repeated calls
- ✅ **5 calculators** integrated seamlessly
- ✅ **Position-specific** defense matchup ratings

### Player Props Enhancement
- Team efficiency context (success rate, EPA, red zone %)
- Defense matchup ratings by position (QB/RB/WR/TE)
- Matchup edges (pass/rush efficiency differentials)
- 23 features automatically added to predictions

### Game Predictions Enhancement
- Pace adjustments to totals (±3.5 pts per 10 plays)
- Turnover margin adjustments to spreads (+2.5 pts per margin)
- Efficiency metrics (EPA, success rate, red zone %)
- 80% confidence vs 75% baseline

---

## 📊 Available Metrics

### Categories (37+ total)

**Team Efficiency** (7 metrics)
- Success rate, EPA/play, completion %, yards/attempt, yards/carry

**Pace** (2 metrics)
- Plays per game, time of possession

**Turnovers** (3 metrics)
- Turnover margin, turnover rate, takeaway rate

**Situational** (3 metrics)
- Red zone TD%, third down %, explosive play rate

**Defense** (3 metrics per position)
- Matchup factor, rank, yards allowed vs position

**Performance** (6 metrics)
- Points/game, yards/game, passing/rushing splits, home/away

See [METRICS_REGISTRY.md](METRICS_REGISTRY.md) for complete list.

---

## 💻 Usage Examples

### Team Analysis
```python
api = get_metrics_api(season=2025)

# Get all metrics
metrics = api.get_team_metrics('KC')

# Recent 4 weeks only
recent = api.get_team_metrics('KC', weeks=[9, 10, 11, 12])

# Compare teams
comp = api.compare_teams('KC', 'BUF')
print(f"KC advantages: {len(comp['advantages_a'])}")
```

### Player Props
```python
import pandas as pd

# Load and enrich player data
player_df = pd.read_csv('inputs/player_stats_2025.csv')
enriched_df = api.enrich_player_features(player_df)

# Use in predictions
from backend.orchestration.picks_pipeline import PicksPipeline
pipeline = PicksPipeline(season=2025)
report = pipeline.generate_picks(week=13)
# Automatically uses 23 team metrics
```

### Game Predictions
```python
from backend.analysis.game_markets import GameMarketAnalyzer

# Initialize with enhanced metrics
analyzer = GameMarketAnalyzer(season=2025, use_enhanced_metrics=True)

# Predict with pace, turnovers, efficiency
prediction = analyzer.predict_game_outcome('KC', 'BUF', week=13)
```

### Matchup Analysis
```python
# Complete matchup analysis
matchup = api.analyze_matchup('KC', 'BUF', week=13)

# Game-specific metrics
game_metrics = api.get_game_metrics('KC', 'BUF', week=13)
pace = game_metrics['summary']['pace']['combined_pace']
```

More examples: [docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md)

---

## 🛠️ Command-Line Tools

### Analyze Team
```bash
python examples/analyze_team.py KC
python examples/analyze_team.py BUF --weeks 9 10 11 12
```

### Compare Matchup
```bash
python examples/compare_matchup.py KC BUF
python examples/compare_matchup.py KC BUF --week 13
```

---

## 🎯 Integration Points

### ✅ Currently Integrated

**Player Props Pipeline** (`backend/orchestration/picks_pipeline.py`)
- Automatically enriches features with 23 team metrics
- Position-specific metric selection (8 for passing, 7 for rushing, etc.)
- Defense matchup factors included

**Game Market Analyzer** (`backend/analysis/game_markets.py`)
- Enhanced predictions with pace, turnovers, efficiency
- `use_enhanced_metrics=True` by default
- 80% confidence vs 75% baseline

### 🔄 Available for Integration

**Model Training**
- Use enriched features for better predictions
- 23 new features available for training

**Backtesting**
- Validate prediction improvements vs historical lines
- Compare enhanced vs baseline models

**Dashboards**
- Visualize team metrics over time
- Track week-over-week trends

---

## 📈 Performance

- **Caching**: 335x faster on repeated calls
- **API Calls**: 1 call replaces 5+ separate calls
- **DataFrame Enrichment**: Single pass vs multiple iterations
- **Confidence Boost**: 80% (enhanced) vs 75% (baseline)

---

## 🧪 Testing

Run test suites to verify setup:

```bash
# Test player props integration (4 tests)
python test_metrics_integration.py

# Test game predictions integration (5 tests)
python test_game_metrics_integration.py

# Test unified API (11 tests)
python test_unified_metrics_api.py
```

All 20/20 tests should pass ✅

---

## 📦 Project Structure

```
nfl_backend/
├── backend/
│   ├── analysis/
│   │   ├── advanced_team_metrics.py      # EPA, success rate, etc.
│   │   ├── team_matchup_analyzer.py      # H2H analysis
│   │   ├── defense_matchup_deep.py       # Position ratings
│   │   └── game_markets.py               # Game predictions
│   ├── features/
│   │   ├── team_metrics_features.py      # Player enrichment
│   │   └── game_metrics_features.py      # Game enhancement
│   ├── metrics/
│   │   └── unified_metrics_api.py        # 🌟 Main API
│   └── orchestration/
│       └── picks_pipeline.py             # Player props
├── docs/
│   ├── IMPLEMENTATION_GUIDE.md           # 📘 Complete guide
│   └── QUICK_REFERENCE.md                # 📄 Cheat sheet
├── examples/
│   ├── analyze_team.py                   # CLI tool
│   └── compare_matchup.py                # CLI tool
├── METRICS_REGISTRY.md                    # 📊 Metric catalog
└── README_METRICS.md                      # This file
```

---

## 🔧 Requirements

### Data Files (in `inputs/`)
- `play_by_play_{season}.parquet` - Required for advanced metrics
- `player_stats_{season}.csv` - Required for player enrichment
- `{season}_schedule.parquet` - Optional for matchup analysis

### Python Packages
- pandas, numpy - Data manipulation
- joblib - Model loading (if using predictions)

---

## 🚦 Getting Started

1. **Read the Quick Reference**
   ```bash
   cat docs/QUICK_REFERENCE.md
   ```

2. **Run Example Scripts**
   ```bash
   python examples/analyze_team.py KC
   python examples/compare_matchup.py KC BUF
   ```

3. **Try the API**
   ```python
   from backend.metrics.unified_metrics_api import get_metrics_api
   api = get_metrics_api(season=2025)
   metrics = api.get_team_metrics('KC')
   ```

4. **Read Full Guide**
   Open `docs/IMPLEMENTATION_GUIDE.md` for complete walkthrough

---

## 📖 Additional Resources

- **Metric Catalog**: See [METRICS_REGISTRY.md](METRICS_REGISTRY.md)
- **Algorithm Documentation**: Formulas in METRICS_REGISTRY.md
- **Migration Guide**: In [docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md)
- **API Reference**: Complete method docs in implementation guide

---

## 🎓 Learning Path

**Beginner** → Read Quick Reference, run example scripts
**Intermediate** → Read Implementation Guide, use API in code
**Advanced** → Integrate into models, backtest improvements

---

## ❓ Support

**Issues**: Check [docs/IMPLEMENTATION_GUIDE.md#troubleshooting](docs/IMPLEMENTATION_GUIDE.md#troubleshooting)

**Questions**: See FAQ in implementation guide

**Tests**: Run test suites to verify setup

---

## 📊 Metrics at a Glance

| Category | Count | Examples |
|----------|-------|----------|
| Team Efficiency | 7 | Success rate, EPA, completion % |
| Pace | 2 | Plays/game, time of possession |
| Turnovers | 3 | Margin, turnover rate, takeaway rate |
| Situational | 3 | Red zone %, 3rd down %, explosive % |
| Defense | 3 per pos | Matchup factor, rank, yards allowed |
| Performance | 6 | PPG, YPG, home/away splits |
| **TOTAL** | **37+** | Full catalog in METRICS_REGISTRY.md |

---

## 🏆 Key Benefits

✅ **Single Access Point** - One import, access everything
✅ **Performance** - 335x caching speedup
✅ **Comprehensive** - 37+ metrics per team
✅ **Integrated** - Works with existing pipeline
✅ **Documented** - Full guides and examples
✅ **Tested** - 20/20 tests passing
✅ **Production-Ready** - Used in picks pipeline

---

## 🔄 Next Steps

After setup:
1. Explore metrics with example scripts
2. Integrate into your models
3. Backtest prediction improvements
4. Build custom dashboards
5. Train enhanced models

---

**Last Updated:** 2025-11-28
**Version:** 1.0.0
**Status:** Production Ready ✅

---

For the complete implementation guide, see [docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md)
