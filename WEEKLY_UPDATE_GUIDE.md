# Weekly Data Update Guide

## Overview

The NFL prediction system uses the **most current 2025 season data** by aggregating play-by-play statistics. This guide explains how to keep your data up-to-date throughout the season.

## Quick Update

Run this command after each NFL week to get the latest data:

```bash
python -m backend.ingestion.update_current_season
```

This will:
1. Fetch latest 2025 play-by-play data from nflverse
2. Aggregate player statistics (QB, RB, WR stats)
3. Update game schedules and results
4. Save to `inputs/historical/`

## Usage

### Basic Update
```bash
# Update current season (2025) with latest data
python -m backend.ingestion.update_current_season
```

### Check Available Data (No Update)
```bash
# Verify what data is currently available
python -m backend.ingestion.update_current_season --verify
```

### Verbose Mode
```bash
# See detailed progress and position breakdowns
python -m backend.ingestion.update_current_season --verbose
```

### Update Specific Season
```bash
# Update a different season (if needed)
python -m backend.ingestion.update_current_season --season 2024
```

## What Gets Updated

The update tool fetches and processes:

- **Games:** Full season schedule (272 games for 2025)
- **Player Stats:** Aggregated from play-by-play data
  - Passing: Completions, attempts, yards, TDs, INTs
  - Rushing: Carries, yards, TDs
  - Receiving: Receptions, targets, yards, TDs
- **Fantasy Points:** Both standard and PPR scoring
- **Weeks Available:** Automatically includes all completed weeks

## When to Run

**Recommended Schedule:**
- **Weekly:** Tuesday morning after Monday Night Football
- **Mid-week:** Wednesday if Thursday Night Football occurred
- **Before Predictions:** Always before generating weekly prop predictions

**Example Weekly Workflow:**
```bash
# Tuesday morning after MNF
python -m backend.ingestion.update_current_season

# Re-train models with updated data
python -m backend.modeling.train_passing_model --season 2025

# Generate predictions for upcoming week
python -m backend.orchestration.orchestrator --season 2025 --week 14
```

## Data Freshness

The tool uses **nflverse play-by-play data**, which is typically updated:
- During games: Real-time (with ~5 minute delay)
- After games: Within 1-2 hours
- Official stats: Usually finalized by Tuesday morning

## Output

After running, you'll see:

```
======================================================================
NFL CURRENT SEASON DATA UPDATER
======================================================================
Season: 2025
Timestamp: 2025-11-27 10:30:00
======================================================================

Fetching latest data for 2025 season...
  Fetching games...
    ✓ 272 games
  Fetching player stats...
    ! Weekly data not available: HTTP Error 404: Not Found
    Aggregating from play-by-play data...
    ✓ 4559 player-week records (from pbp)

======================================================================
UPDATE SUMMARY
======================================================================
✓ Games: 272
✓ Player-week records: 4559
✓ Injuries: 0
✓ Rosters: 31746

Weeks available: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
Latest week: 12

======================================================================
✓ Update complete! Data saved to inputs/historical/
======================================================================

Next steps:
  1. Re-run model training with updated data
  2. Generate fresh predictions for current week
  3. Update backtesting validation (optional)
```

## Automation

### Cron Job (Linux/Mac)
```bash
# Add to crontab to run every Tuesday at 10 AM
0 10 * * 2 cd /path/to/nfl_backend && python -m backend.ingestion.update_current_season
```

### Task Scheduler (Windows)
1. Open Task Scheduler
2. Create Basic Task
3. Trigger: Weekly, Tuesday, 10:00 AM
4. Action: Start a program
   - Program: `python`
   - Arguments: `-m backend.ingestion.update_current_season`
   - Start in: `C:\path\to\nfl_backend`

## Integration with Analysis

The updated 2025 data is automatically used by:

✅ **Model Training**
- Default season is now 2025 for all training scripts
- Use `--season 2025` to explicitly train on current data

✅ **Predictions**
- Orchestrator defaults to 2025
- Player prop predictions use latest stats

✅ **Backtesting**
- 2025 data available for validation (partial season)
- Complete seasons (2020-2024) used by default

✅ **Analysis Tools**
- Deep analysis modules access 2025 trends
- Injury impact uses current season context

## Troubleshooting

**Error: "HTTP Error 404: Not Found"**
- This is normal! Pre-aggregated weekly data isn't published until season ends
- The tool automatically falls back to play-by-play aggregation
- No action needed

**Error: "No data available"**
- Check internet connection
- Verify nflverse.com is accessible
- Wait a few hours if game just finished

**Outdated Data**
- nflverse typically updates within 1-2 hours of game completion
- Check https://github.com/nflverse/nflverse-data for update status

## Files Updated

Running the update tool modifies:
- `inputs/historical/games_2025.csv` - Game schedules and results
- `inputs/historical/player_stats_2025_all.csv` - Player statistics
- `inputs/historical/metadata_2025.json` - Collection metadata

## Advanced Usage

### Manual Data Collection
If you need more control, use the data collector directly:

```python
from backend.backtesting.data_collector import HistoricalDataCollector

collector = HistoricalDataCollector()
data = collector.collect_season_data(2025, source='nfl_data_py')
```

### Custom Aggregation
The pbp aggregation logic is in `backend/backtesting/data_collector.py`:
- Method: `_aggregate_stats_from_pbp()`
- Customize stat calculations as needed
- Add new metrics by modifying the aggregation logic

---

**Last Updated:** 2025-11-27
**Version:** 1.0
