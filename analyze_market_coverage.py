"""Analyze what markets we can train with available data."""

import pandas as pd
from pathlib import Path

print("="*80)
print("NFL DRAFTKINGS MARKET COVERAGE ANALYSIS")
print("="*80 + "\n")

# Check what data we have
print("AVAILABLE DATA:")
print("-" * 80)

data_files = {
    'player_stats_2025.csv': 'Player game stats (2025)',
    'kicker_stats_2025.csv': 'Kicker stats (2025)',
    'games_2025_with_quarters.csv': 'Game scores with quarters (2025)',
    'play_by_play_2025.parquet': 'Play-by-play data (2025)',
    'rosters_weekly_2024.csv': 'Weekly rosters (2024 - player attributes)',
    'snap_counts_2024.csv': 'Snap counts (2024 - usage rates)',
    'schedules_2024.csv': 'Schedules with Vegas lines (2024)',
}

available = []
for file, desc in data_files.items():
    path = Path(f'inputs/{file}')
    if path.exists():
        size = path.stat().st_size / 1024  # KB
        print(f"  ✅ {file:35s} ({size:>8.1f} KB) - {desc}")
        available.append(file)
    else:
        print(f"  ❌ {file:35s} - {desc}")

print(f"\n{len(available)}/{len(data_files)} data files available\n")

# Market status
print("\nMARKET TRAINING STATUS:")
print("-" * 80)

markets_trained = {
    'PLAYER PROPS - Comprehensive (12 markets)': [
        '✅ Passing Yards O/U',
        '✅ Passing TDs O/U',
        '✅ Completions O/U',
        '✅ Rushing Yards O/U',
        '✅ Rushing TDs O/U',
        '✅ Receptions O/U',
        '✅ Receiving Yards O/U',
        '✅ Receiving TDs O/U',
        '✅ Pass Attempts O/U',
        '✅ Rush Attempts O/U',
        '✅ Interceptions O/U',
        '✅ Longest Reception O/U',
    ],
    'PLAYER PROPS - Combo (2 markets)': [
        '✅ Pass + Rush Yards',
        '✅ Rec + Rush Yards',
    ],
    'GAME DERIVATIVES - Quarters/Halves (10 markets)': [
        '✅ 1st Quarter Total',
        '✅ 2nd Quarter Total',
        '✅ 3rd Quarter Total',
        '✅ 4th Quarter Total',
        '✅ 1st Half Total',
        '✅ 2nd Half Total',
        '✅ 1st Quarter Winner',
        '✅ 1st Half Winner',
        '✅ Highest Scoring Quarter',
        '✅ Highest Scoring Half',
    ],
    'KICKER PROPS (3 markets)': [
        '✅ FG Made O/U',
        '✅ XP Made O/U',
        '✅ Total Points O/U',
    ],
    'TD SCORER PROPS (3 markets)': [
        '✅ First TD Scorer',
        '✅ Last TD Scorer',
        '✅ Anytime TD Scorer',
    ],
    'PBP MARKETS (5 markets)': [
        '✅ First TD Scorer (PBP)',
        '✅ Last TD Scorer (PBP)',
        '✅ Longest Rush (Team)',
        '✅ Longest Pass (Team)',
        '✅ Team To Score First',
    ],
    'GAME OUTCOME (3 markets)': [
        '✅ Winning Margin',
        '✅ Winning Quarter',
        '✅ Winning Half',
    ],
}

total_trained = sum(len(markets) for markets in markets_trained.values())

for category, markets in markets_trained.items():
    print(f"\n{category}:")
    for market in markets:
        print(f"  {market}")

print(f"\n{'='*80}")
print(f"MARKETS TRAINED: {total_trained}/80 ({total_trained/80*100:.1f}%)")
print(f"{'='*80}\n")

# Markets we CANNOT train yet (need additional data)
print("\nMARKETS REQUIRING ADDITIONAL DATA:")
print("-" * 80)

markets_blocked = {
    'NEED 2025 Snap Counts (High Value - 8 markets)': [
        '⏸️  Snap Share % O/U (offensive players)',
        '⏸️  Target Share % O/U',
        '⏸️  Route Participation % O/U',
        '⏸️  Red Zone Snap Share %',
        '⏸️  First Half Snaps O/U',
        '⏸️  Starter Props (will player start)',
        '⏸️  Playing Time O/U',
        '⏸️  Touches per Snap',
    ],
    'NEED 2025 Schedules with Vegas Lines (Critical - 6 markets)': [
        '⏸️  Game Total O/U (with line adjustment)',
        '⏸️  Spread Cover Probability',
        '⏸️  Team Total Points O/U',
        '⏸️  1H Spread',
        '⏸️  2H Spread',
        '⏸️  Live Line Movement',
    ],
    'NEED Defensive Stats (Medium Value - 5 markets)': [
        '⏸️  Tackles O/U',
        '⏸️  Sacks O/U',
        '⏸️  Interceptions (Defensive)',
        '⏸️  QB Hits O/U',
        '⏸️  Turnovers Forced',
    ],
    'NEED Next Gen Stats (High Value - 4 markets)': [
        '⏸️  Average Depth of Target (aDOT)',
        '⏸️  Cushion Distance',
        '⏸️  Time to Throw',
        '⏸️  Separation Distance',
    ],
    'NEED Injury/Status Data (Medium Value - 3 markets)': [
        '⏸️  Backup QB Performance',
        '⏸️  Injury Impact Props',
        '⏸️  Game Time Decision Players',
    ],
    'NEED Team/Formation Data (Lower Value - 6 markets)': [
        '⏸️  Personnel Package Rates',
        '⏸️  Play Type Distribution',
        '⏸️  Formation Tendencies',
        '⏸️  Situation-Specific Props',
        '⏸️  Drive Outcomes',
        '⏸️  Red Zone Efficiency',
    ],
    'CAN TRAIN NOW with 2024 Data (7 markets)': [
        '🔶 Player Physical Props (height/weight based)',
        '🔶 Experience-Based Props (years in NFL)',
        '🔶 Draft Position Impact',
        '🔶 College Stats Correlation',
        '🔶 Age vs Performance',
        '🔶 Rest Days Impact (using 2024 schedules)',
        '🔶 Weather Impact (roof/surface from 2024)',
    ],
}

for category, markets in markets_blocked.items():
    print(f"\n{category}:")
    for market in markets:
        print(f"  {market}")

remaining = sum(len(markets) for markets in markets_blocked.values())
print(f"\n{'='*80}")
print(f"REMAINING MARKETS: {remaining}")
print(f"{'='*80}\n")

# Action items
print("\nIMMEDIATE ACTION ITEMS:")
print("-" * 80)
print("""
1. HIGH PRIORITY - Download 2025 data locally:
   Run: python download_2025_data_locally.py

   This will get:
   - rosters_weekly_2025.csv
   - snap_counts_2025.csv
   - schedules_2025.csv (CRITICAL - has Vegas lines!)
   - ngs_passing/rushing/receiving_2025.csv

2. MEDIUM PRIORITY - Additional data:
   - Defensive stats (tackles, sacks, QB hits)
   - Injury reports
   - Weather data for 2025 games

3. CAN DO NOW - Enhance existing models:
   - Add player attributes from 2024 rosters (height, weight, years_exp)
   - These attributes don't change year-to-year
   - Enrich models with physical/experience features
""")

print("="*80)
