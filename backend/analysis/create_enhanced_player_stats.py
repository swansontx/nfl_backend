"""
Create enhanced player stats dataset with snap counts, Vegas lines, and NGS metrics.
This merges all available 2025 data for comprehensive model training.
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("CREATING ENHANCED PLAYER STATS DATASET")
print("="*80 + "\n")

inputs_dir = Path('/home/user/nfl_backend/inputs')

# 1. Load base player stats (using the comprehensive NFLverse data)
print("1. Loading base player stats...")
player_stats = pd.read_csv(inputs_dir / 'player_stats_2024_2025.csv', low_memory=False)
player_stats_2025 = player_stats[player_stats['season'] == 2025].copy()
print(f"   Base player stats 2025: {len(player_stats_2025)} records")

# 2. Load snap counts
print("\n2. Loading snap counts...")
snaps = pd.read_csv(inputs_dir / 'snap_counts_2024_2025.csv')
snaps_2025 = snaps[snaps['season'] == 2025].copy()
print(f"   Snap counts 2025: {len(snaps_2025)} records")

# Create game_id for merging (if not present)
if 'game_id' not in player_stats_2025.columns:
    # Create game_id from season, week, team
    player_stats_2025['game_id'] = player_stats_2025.apply(
        lambda x: f"{x['season']}_{x['week']:02d}_{x['team']}", axis=1
    )

# 3. Load schedules with Vegas lines
print("\n3. Loading schedules with Vegas lines...")
schedules = pd.read_csv(inputs_dir / 'schedules_2024_2025.csv')
schedules_2025 = schedules[schedules['season'] == 2025].copy()
print(f"   Schedules 2025: {len(schedules_2025)} games")

# 4. Load NGS data
print("\n4. Loading Next Gen Stats...")
ngs_passing = pd.read_csv(inputs_dir / 'ngs_passing_2024_2025.csv')
ngs_rushing = pd.read_csv(inputs_dir / 'ngs_rushing_2024_2025.csv')
ngs_receiving = pd.read_csv(inputs_dir / 'ngs_receiving_2024_2025.csv')

ngs_passing_2025 = ngs_passing[ngs_passing['season'] == 2025].copy()
ngs_rushing_2025 = ngs_rushing[ngs_rushing['season'] == 2025].copy()
ngs_receiving_2025 = ngs_receiving[ngs_receiving['season'] == 2025].copy()
print(f"   NGS Passing 2025: {len(ngs_passing_2025)} records")
print(f"   NGS Rushing 2025: {len(ngs_rushing_2025)} records")
print(f"   NGS Receiving 2025: {len(ngs_receiving_2025)} records")

# 5. Load rosters for player attributes
print("\n5. Loading rosters for player attributes...")
rosters = pd.read_csv(inputs_dir / 'rosters_weekly_2024_2025.csv')
rosters_2025 = rosters[rosters['season'] == 2025].copy()
print(f"   Rosters 2025: {len(rosters_2025)} records")

# ============================================================================
# MERGE DATA
# ============================================================================
print("\n" + "="*80)
print("MERGING DATA...")
print("="*80 + "\n")

# Get unique player info from rosters (latest week)
player_info = rosters_2025.sort_values('week', ascending=False).drop_duplicates('gsis_id')[
    ['gsis_id', 'height', 'weight', 'birth_date', 'years_exp', 'college', 'draft_number']
].copy()
player_info = player_info.rename(columns={'gsis_id': 'player_id'})
print(f"Unique players with attributes: {len(player_info)}")

# Merge player attributes
enhanced = player_stats_2025.merge(
    player_info,
    on='player_id',
    how='left'
)
print(f"After merging player attributes: {len(enhanced)} records")

# Merge snap counts
# Need to match on player name and game
snaps_subset = snaps_2025[['player', 'team', 'week', 'offense_snaps', 'offense_pct',
                           'defense_snaps', 'defense_pct', 'st_snaps', 'st_pct']].copy()
snaps_subset = snaps_subset.rename(columns={'player': 'player_display_name'})

enhanced = enhanced.merge(
    snaps_subset,
    on=['player_display_name', 'team', 'week'],
    how='left'
)
print(f"After merging snap counts: {len(enhanced)} records")
print(f"   Players with snap data: {enhanced['offense_pct'].notna().sum()}")

# Merge Vegas lines from schedules
# Need to match on game_id
vegas_cols = ['game_id', 'spread_line', 'total_line', 'home_rest', 'away_rest',
              'roof', 'surface', 'temp', 'wind', 'div_game']
schedules_vegas = schedules_2025[vegas_cols].copy()

enhanced = enhanced.merge(
    schedules_vegas,
    on='game_id',
    how='left'
)
print(f"After merging Vegas lines: {len(enhanced)} records")
print(f"   Games with Vegas lines: {enhanced['spread_line'].notna().sum()}")

# Merge NGS passing
ngs_pass_cols = ['player_display_name', 'week', 'avg_time_to_throw',
                 'avg_completed_air_yards', 'avg_intended_air_yards', 'aggressiveness']
if all(col in ngs_passing_2025.columns for col in ngs_pass_cols):
    ngs_pass_subset = ngs_passing_2025[ngs_pass_cols].copy()
    enhanced = enhanced.merge(
        ngs_pass_subset,
        on=['player_display_name', 'week'],
        how='left'
    )
    print(f"After merging NGS passing: {len(enhanced)} records")

# Merge NGS rushing
ngs_rush_cols = ['player_display_name', 'week', 'efficiency',
                 'avg_time_to_los', 'percent_attempts_gte_eight_defenders']
if all(col in ngs_rushing_2025.columns for col in ngs_rush_cols):
    ngs_rush_subset = ngs_rushing_2025[ngs_rush_cols].copy()
    enhanced = enhanced.merge(
        ngs_rush_subset,
        on=['player_display_name', 'week'],
        how='left'
    )
    print(f"After merging NGS rushing: {len(enhanced)} records")

# Merge NGS receiving
ngs_rec_cols = ['player_display_name', 'week', 'avg_cushion', 'avg_separation',
                'percent_share_of_intended_air_yards']
if all(col in ngs_receiving_2025.columns for col in ngs_rec_cols):
    ngs_rec_subset = ngs_receiving_2025[ngs_rec_cols].copy()
    enhanced = enhanced.merge(
        ngs_rec_subset,
        on=['player_display_name', 'week'],
        how='left'
    )
    print(f"After merging NGS receiving: {len(enhanced)} records")

# ============================================================================
# ADD ROLLING AVERAGES
# ============================================================================
print("\n" + "="*80)
print("CALCULATING ROLLING AVERAGES...")
print("="*80 + "\n")

# Sort by player and week
enhanced = enhanced.sort_values(['player_id', 'week'])

# Key stats to calculate rolling averages for
rolling_stats = [
    'completions', 'attempts', 'passing_yards', 'passing_tds',
    'rushing_yards', 'rushing_tds', 'carries',
    'receptions', 'receiving_yards', 'receiving_tds', 'targets',
    'offense_pct', 'offense_snaps'
]

# Calculate rolling averages
for stat in rolling_stats:
    if stat in enhanced.columns:
        # Season average (expanding mean)
        enhanced[f'{stat}_season_avg'] = enhanced.groupby('player_id')[stat].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        # Last 3 games average
        enhanced[f'{stat}_l3_avg'] = enhanced.groupby('player_id')[stat].transform(
            lambda x: x.rolling(3, min_periods=1).mean().shift(1)
        )

print(f"Added rolling averages for {len(rolling_stats)} stats")

# Add games played
enhanced['games_played'] = enhanced.groupby('player_id').cumcount() + 1

# ============================================================================
# SAVE ENHANCED DATASET
# ============================================================================
print("\n" + "="*80)
print("SAVING ENHANCED DATASET...")
print("="*80 + "\n")

output_file = inputs_dir / 'player_stats_enhanced_2025.csv'
enhanced.to_csv(output_file, index=False)

print(f"Output file: {output_file}")
print(f"Total records: {len(enhanced)}")
print(f"Total columns: {len(enhanced.columns)}")
print(f"\nKey features added:")
print(f"  - Snap counts: offense_pct, defense_pct, st_pct")
print(f"  - Vegas lines: spread_line, total_line")
print(f"  - Environment: roof, surface, temp, wind")
print(f"  - NGS metrics: time_to_throw, air_yards, separation, etc.")
print(f"  - Player attributes: height, weight, years_exp")
print(f"  - Rolling averages: season_avg, l3_avg for all key stats")

# Show sample of new columns
print(f"\nNew columns sample:")
new_cols = [c for c in enhanced.columns if 'pct' in c or 'line' in c or 'avg_' in c][:10]
for col in new_cols:
    non_null = enhanced[col].notna().sum()
    print(f"  {col}: {non_null} non-null values")

print("\n" + "="*80)
print("ENHANCED DATASET CREATED SUCCESSFULLY!")
print("="*80)
