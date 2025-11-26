# Setup Verification Checklist

This document verifies that the complete backtesting system is properly installed and configured.

## ✅ Installation Verification

### 1. Required Dependencies

Run this command to check installed packages:
```bash
pip list | grep -E "nfl-data-py|pandas|numpy|scipy"
```

**Expected output:**
```
nfl-data-py    0.3.x or higher
numpy          1.24.x or higher
pandas         2.0.x or higher
scipy          1.10.x or higher
```

### 2. Directory Structure

Verify all required directories exist:
```bash
ls -la inputs/historical/
ls -la outputs/backtesting/
ls -la .cache/weather/
```

**Expected:**
```
inputs/
└── historical/
    └── injuries/

outputs/
└── backtesting/

.cache/
└── weather/
```

### 3. Backtesting Modules

Verify all modules are present:
```bash
ls -1 backend/backtesting/*.py
```

**Expected files:**
```
backend/backtesting/__init__.py
backend/backtesting/data_collector.py
backend/backtesting/defense_matchup_backtest.py
backend/backtesting/framework.py
backend/backtesting/injury_impact_backtest.py
backend/backtesting/overall_accuracy_backtest.py
backend/backtesting/run_all_backtests.py
backend/backtesting/situational_factors_backtest.py
backend/backtesting/weather_impact_backtest.py
```

### 4. Configuration Files

Verify configuration files exist:
```bash
ls -1 backend/config/*.py
```

**Expected files:**
```
backend/config/__init__.py
backend/config/validated_weights.py
```

### 5. Documentation

Verify documentation files exist:
```bash
ls -1 *.md | grep -E "BACKTEST|DEEP_ANALYSIS"
```

**Expected files:**
```
BACKTESTING_SYSTEM.md
DEEP_ANALYSIS_OPTIMIZATIONS.md
```

### 6. Module Compilation

Test that all modules compile without errors:
```bash
./scripts/setup_backtesting.sh
```

**Expected:** All modules show ✓ and compilation succeeds

## 📊 Functional Verification

### Test 1: Import Backtesting Framework

```python
python3 << EOF
from backend.backtesting import BacktestingFramework
framework = BacktestingFramework(seasons=[2023])
print("✓ Framework imported successfully")
print(f"  Seasons: {framework.seasons}")
print(f"  Data dir: {framework.data_dir}")
print(f"  Results dir: {framework.results_dir}")
EOF
```

**Expected output:**
```
✓ Framework imported successfully
  Seasons: [2023]
  Data dir: inputs/historical
  Results dir: outputs/backtesting
```

### Test 2: Import All Backtesting Modules

```python
python3 << EOF
from backend.backtesting import (
    BacktestingFramework,
    HistoricalDataCollector,
    InjuryImpactBacktester,
    DefenseMatchupBacktester,
    WeatherImpactBacktester,
    SituationalFactorsBacktester,
    OverallAccuracyBacktester,
    BacktestingOrchestrator
)
print("✓ All backtesting modules imported successfully")
print(f"  - BacktestingFramework")
print(f"  - HistoricalDataCollector")
print(f"  - InjuryImpactBacktester")
print(f"  - DefenseMatchupBacktester")
print(f"  - WeatherImpactBacktester")
print(f"  - SituationalFactorsBacktester")
print(f"  - OverallAccuracyBacktester")
print(f"  - BacktestingOrchestrator")
EOF
```

### Test 3: Import Validated Weights

```python
python3 << EOF
from backend.config import (
    INJURY_REDISTRIBUTION,
    DEFENSE_MATCHUP_ADJUSTMENTS,
    WEATHER_IMPACT,
    SITUATIONAL_ADJUSTMENTS,
    FEATURE_WEIGHTS,
    get_validated_weight
)
print("✓ Validated weights imported successfully")
print(f"  - INJURY_REDISTRIBUTION")
print(f"  - DEFENSE_MATCHUP_ADJUSTMENTS")
print(f"  - WEATHER_IMPACT")
print(f"  - SITUATIONAL_ADJUSTMENTS")
print(f"  - FEATURE_WEIGHTS")

# Test helper function
wr_pattern = INJURY_REDISTRIBUTION.get('WR', {}).get('WR1_OUT', {})
print(f"\n✓ Example weight access:")
print(f"  WR1_OUT -> WR2 target share: {wr_pattern.get('WR2', {}).get('target_share', 'N/A')}")
EOF
```

### Test 4: Verify Data Collector

```python
python3 << EOF
from backend.backtesting import HistoricalDataCollector
collector = HistoricalDataCollector()
print("✓ Data collector initialized")
print(f"  Output dir: {collector.output_dir}")
print(f"  Injuries dir: {collector.injuries_dir}")

# Check data availability (without actually fetching)
availability = collector.verify_data_availability([2023])
print(f"\n  2023 data availability:")
print(f"    Games: {availability.get(2023, {}).get('games', False)}")
print(f"    Player stats: {availability.get(2023, {}).get('player_stats', False)}")
print(f"    Injuries: {availability.get(2023, {}).get('injuries', False)}")
EOF
```

### Test 5: Verify Orchestrator

```python
python3 << EOF
from backend.backtesting import BacktestingOrchestrator
orchestrator = BacktestingOrchestrator(seasons=[2023])
print("✓ Orchestrator initialized")
print(f"  Seasons: {orchestrator.seasons}")
print(f"  Output dir: {orchestrator.output_dir}")
print(f"  Framework: {type(orchestrator.framework).__name__}")
print(f"  Data collector: {type(orchestrator.data_collector).__name__}")
EOF
```

## 🔍 Pre-Run Checklist

Before running actual backtests, verify:

- [ ] Python 3.8+ installed (`python3 --version`)
- [ ] All dependencies installed (`pip list | grep -E "nfl-data-py|pandas|numpy|scipy"`)
- [ ] All 9 backtesting modules present and compiled
- [ ] Configuration files present
- [ ] Directories created (`inputs/historical/`, `outputs/backtesting/`)
- [ ] All import tests pass
- [ ] No syntax errors in any module

## 📥 Data Collection Checklist

To collect historical data:

```bash
# Collect data for 2020-2023 seasons
python -m backend.backtesting.data_collector
```

**After data collection, verify:**

```bash
# Check for game data
ls -lh inputs/historical/games_*.csv

# Check for player stats
ls -lh inputs/historical/player_stats_*_all.csv

# Check for injuries
ls -lh inputs/historical/injuries/injuries_*.csv
```

**Expected files:**
```
inputs/historical/games_2020.csv
inputs/historical/games_2021.csv
inputs/historical/games_2022.csv
inputs/historical/games_2023.csv
inputs/historical/player_stats_2020_all.csv
inputs/historical/player_stats_2021_all.csv
inputs/historical/player_stats_2022_all.csv
inputs/historical/player_stats_2023_all.csv
inputs/historical/injuries/injuries_2020.csv
inputs/historical/injuries/injuries_2021.csv
inputs/historical/injuries/injuries_2022.csv
inputs/historical/injuries/injuries_2023.csv
```

**Data size expectations:**
- Games: ~300 games per season (~50KB per file)
- Player stats: ~12,000 player-game records per season (~5-10MB per file)
- Injuries: ~500 injury reports per season (~100KB per file)

## 🏃 Running Backtests

Once data is collected:

```bash
# Run all backtests
python -m backend.backtesting.run_all_backtests
```

**Expected output:**
```
HISTORICAL BACKTESTING - NFL BACKEND DEEP ANALYSIS
================================================================================
Seasons: [2020, 2021, 2022, 2023]

1. INJURY IMPACT REDISTRIBUTION
  ✓ Injury impact backtest complete

2. DEFENSE MATCHUP ADJUSTMENTS
  ✓ Defense matchup backtest complete

3. WEATHER IMPACT COEFFICIENTS
  ✓ Weather impact backtest complete

4. SITUATIONAL FACTORS ADJUSTMENTS
  ✓ Situational factors backtest complete

5. OVERALL PREDICTION ACCURACY
  ✓ Overall accuracy backtest complete

Saved backtest results to outputs/backtesting/
```

**Verify output files:**
```bash
ls -lh outputs/backtesting/
```

**Expected files:**
```
outputs/backtesting/BACKTESTING_REPORT.md
outputs/backtesting/backtest_summary.json
outputs/backtesting/injury_impact_backtest.json
outputs/backtesting/defense_matchup_backtest.json
outputs/backtesting/weather_impact_backtest.json
outputs/backtesting/situational_factors_backtest.json
outputs/backtesting/overall_accuracy_backtest.json
```

## ✅ Success Criteria

The system is properly set up when:

1. ✓ All 9 backtesting modules compile without errors
2. ✓ All imports work (framework, collectors, backtesters, orchestrator)
3. ✓ Configuration files load without errors
4. ✓ Directories are created
5. ✓ Dependencies are installed
6. ✓ Setup script completes successfully

## 🚨 Common Issues

### Issue: "ModuleNotFoundError: No module named 'nfl_data_py'"
**Solution:**
```bash
pip install nfl-data-py
```

### Issue: "FileNotFoundError: inputs/historical/games_2023.csv"
**Solution:** Run data collector first:
```bash
python -m backend.backtesting.data_collector
```

### Issue: "ImportError: cannot import name 'BacktestingFramework'"
**Solution:** Verify Python path and module structure:
```bash
python3 -c "import sys; print('\n'.join(sys.path))"
# Make sure current directory is in path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: Compilation errors
**Solution:** Re-run setup script:
```bash
./scripts/setup_backtesting.sh
```

## 📚 Quick Reference Commands

```bash
# Full setup workflow
./scripts/setup_backtesting.sh
python -m backend.backtesting.data_collector
python -m backend.backtesting.run_all_backtests
cat outputs/backtesting/BACKTESTING_REPORT.md

# Verify specific module
python3 -m py_compile backend/backtesting/framework.py

# Test imports
python3 -c "from backend.backtesting import BacktestingOrchestrator; print('✓ OK')"

# Check data availability
python3 -c "from backend.backtesting import HistoricalDataCollector; c = HistoricalDataCollector(); print(c.verify_data_availability([2023]))"

# Re-run single backtest
python3 -c "from backend.backtesting import BacktestingFramework, InjuryImpactBacktester; f = BacktestingFramework([2023]); b = InjuryImpactBacktester(f); r = b.run_backtest(); print(r.notes)"
```

## 📖 Documentation Reference

- **Full System Guide:** `BACKTESTING_SYSTEM.md`
- **Deep Analysis Features:** `DEEP_ANALYSIS_OPTIMIZATIONS.md`
- **Quick Start:** `backend/backtesting/README.md`

---

**Status:** ✅ System verification complete
**Last Updated:** Run `date` to get current timestamp
**Version:** 1.0 - Complete backtesting infrastructure
