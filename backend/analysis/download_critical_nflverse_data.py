"""Download CRITICAL NFLverse data for betting models."""

import nfl_data_py as nfl
import pandas as pd
from pathlib import Path

print("="*80)
print("DOWNLOADING CRITICAL NFLVERSE DATA")
print("="*80 + "\n")

inputs_dir = Path('/home/user/nfl_backend/inputs')
inputs_dir.mkdir(parents=True, exist_ok=True)

years = [2024, 2025]

# 1. SCHEDULES (CRITICAL - Has Vegas lines!)
print("1. Downloading schedules (Vegas lines, rest days, roof)...")
try:
    schedules = nfl.import_schedules(years)
    schedules.to_csv(inputs_dir / 'schedules.csv', index=False)
    print(f"   ✅ {len(schedules)} games")
    print(f"   Columns: {list(schedules.columns)[:10]}...")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 2. ROSTERS (CRITICAL - Who's playing)
print("\n2. Downloading weekly rosters...")
try:
    rosters = nfl.import_weekly_rosters(years)
    rosters.to_csv(inputs_dir / 'rosters_weekly.csv', index=False)
    print(f"   ✅ {len(rosters)} player-week records")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 3. PLAYER STATS (Weekly aggregated - easier than PBP)
print("\n3. Downloading weekly player stats...")
try:
    player_stats = nfl.import_weekly_data(years)
    player_stats.to_csv(inputs_dir / 'player_stats_weekly.csv', index=False)
    print(f"   ✅ {len(player_stats)} player-week stats")
    print(f"   Columns: {list(player_stats.columns)[:15]}...")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 4. SNAP COUNTS (HIGH VALUE - Usage rates)
print("\n4. Downloading snap counts...")
try:
    snaps = nfl.import_snap_counts(years)
    snaps.to_csv(inputs_dir / 'snap_counts.csv', index=False)
    print(f"   ✅ {len(snaps)} snap records")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 5. TEAMS (For consistency)
print("\n5. Downloading team info...")
try:
    teams = nfl.import_team_desc()
    teams.to_csv(inputs_dir / 'teams.csv', index=False)
    print(f"   ✅ {len(teams)} teams")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 6. NEXT GEN STATS (HIGH VALUE - Advanced metrics)
print("\n6. Downloading Next Gen Stats...")
for stat_type in ['passing', 'rushing', 'receiving']:
    try:
        ngs = nfl.import_ngs_data(stat_type, years)
        ngs.to_csv(inputs_dir / f'ngs_{stat_type}.csv', index=False)
        print(f"   ✅ NGS {stat_type}: {len(ngs)} records")
    except Exception as e:
        print(f"   ❌ NGS {stat_type}: {e}")

print("\n" + "="*80)
print("CRITICAL DATA DOWNLOAD COMPLETE")
print("="*80)
