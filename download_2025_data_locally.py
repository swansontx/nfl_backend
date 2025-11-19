"""
Download 2025 NFLverse data - RUN THIS LOCALLY (not in Claude Code)

This script will download the 2025 versions of:
- Weekly rosters
- Snap counts
- Schedules

Run this in your local Python/R/Colab environment to avoid 403 errors.
"""

import nfl_data_py as nfl
import pandas as pd
from pathlib import Path

print("="*80)
print("DOWNLOADING 2025 NFLVERSE DATA")
print("="*80 + "\n")

# Output directory
output_dir = Path('inputs')
output_dir.mkdir(parents=True, exist_ok=True)

years = [2025]

# 1. Weekly Rosters
print("1. Downloading weekly rosters for 2025...")
try:
    rosters = nfl.import_weekly_rosters(years)
    output_file = output_dir / 'rosters_weekly_2025.csv'
    rosters.to_csv(output_file, index=False)
    print(f"   ✅ {len(rosters)} player-week records")
    print(f"   💾 Saved to: {output_file}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 2. Snap Counts
print("\n2. Downloading snap counts for 2025...")
try:
    snaps = nfl.import_snap_counts(years)
    output_file = output_dir / 'snap_counts_2025.csv'
    snaps.to_csv(output_file, index=False)
    print(f"   ✅ {len(snaps)} snap count records")
    print(f"   💾 Saved to: {output_file}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 3. Schedules (CRITICAL - has Vegas lines!)
print("\n3. Downloading schedules for 2025...")
try:
    schedules = nfl.import_schedules(years)
    output_file = output_dir / 'schedules_2025.csv'
    schedules.to_csv(output_file, index=False)
    print(f"   ✅ {len(schedules)} games")
    print(f"   💾 Saved to: {output_file}")
    print(f"\n   Key columns:")
    print(f"   - spread_line, total_line (Vegas lines)")
    print(f"   - away_rest, home_rest (rest days)")
    print(f"   - roof, surface, temp, wind (environment)")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 4. BONUS: Next Gen Stats (if available for 2025)
print("\n4. BONUS - Downloading Next Gen Stats for 2025...")
for stat_type in ['passing', 'rushing', 'receiving']:
    try:
        ngs = nfl.import_ngs_data(stat_type, years)
        output_file = output_dir / f'ngs_{stat_type}_2025.csv'
        ngs.to_csv(output_file, index=False)
        print(f"   ✅ NGS {stat_type}: {len(ngs)} records → {output_file}")
    except Exception as e:
        print(f"   ⚠️  NGS {stat_type}: {e}")

print("\n" + "="*80)
print("DOWNLOAD COMPLETE!")
print("="*80)
print("\nNext steps:")
print("1. Commit these new files to your repo")
print("2. Claude Code will integrate them into enhanced training data")
print("3. Retrain models with Vegas lines, snap counts, and rest days")
print("="*80)
