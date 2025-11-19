"""Aggregate REAL player stats from the actual 2025 play-by-play data."""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("AGGREGATING REAL PLAYER STATS FROM PBP DATA")
print("="*80 + "\n")

# Load REAL PBP data
pbp_file = Path('/home/user/nfl_backend/play_by_play_2025.parquet')
print(f"Loading REAL NFL data from {pbp_file}...")
pbp = pd.read_parquet(pbp_file)
print(f"✅ Loaded {len(pbp):,} real NFL plays")
print(f"   Games: {pbp['game_id'].nunique()}")
print(f"   Weeks: {sorted(pbp['week'].unique())}\n")

# Aggregate player stats per game
player_stats = []

print("Aggregating player stats by game...")

# Group by game
for game_id in pbp['game_id'].unique():
    game_plays = pbp[pbp['game_id'] == game_id]
    week = game_plays['week'].iloc[0]
    season = 2025  # We know this is 2025 data

    # QB stats (passers)
    passer_stats = game_plays[game_plays['passer_player_id'].notna()].groupby('passer_player_id').agg({
        'passer_player_name': 'first',
        'posteam': 'first',
        'passing_yards': 'sum',
        'pass_touchdown': 'sum',
        'complete_pass': 'sum',
        'pass_attempt': 'sum',
        'interception': 'sum',
    }).reset_index()

    for _, row in passer_stats.iterrows():
        player_stats.append({
            'player_id': row['passer_player_id'],
            'player_name': row['passer_player_name'],
            'team': row['posteam'],
            'week': week,
            'season': season,
            'game_id': game_id,
            'position': 'QB',
            'passing_yards': row['passing_yards'],
            'passing_tds': row['pass_touchdown'],
            'completions': row['complete_pass'],
            'attempts': row['pass_attempt'],
            'interceptions': row['interception'],
            'rushing_yards': 0,
            'rushing_tds': 0,
            'carries': 0,
            'receiving_yards': 0,
            'receiving_tds': 0,
            'receptions': 0,
            'targets': 0,
        })

    # RB/QB rushing stats
    rusher_stats = game_plays[game_plays['rusher_player_id'].notna()].groupby('rusher_player_id').agg({
        'rusher_player_name': 'first',
        'posteam': 'first',
        'rushing_yards': 'sum',
        'rush_touchdown': 'sum',
        'rush_attempt': 'sum',
    }).reset_index()

    for _, row in rusher_stats.iterrows():
        player_stats.append({
            'player_id': row['rusher_player_id'],
            'player_name': row['rusher_player_name'],
            'team': row['posteam'],
            'week': week,
            'season': season,
            'game_id': game_id,
            'position': 'RB',  # We'll need to infer this better
            'passing_yards': 0,
            'passing_tds': 0,
            'completions': 0,
            'attempts': 0,
            'interceptions': 0,
            'rushing_yards': row['rushing_yards'],
            'rushing_tds': row['rush_touchdown'],
            'carries': row['rush_attempt'],
            'receiving_yards': 0,
            'receiving_tds': 0,
            'receptions': 0,
            'targets': 0,
        })

    # Receiver stats
    receiver_stats = game_plays[game_plays['receiver_player_id'].notna()].groupby('receiver_player_id').agg({
        'receiver_player_name': 'first',
        'posteam': 'first',
        'receiving_yards': 'sum',
        'pass_touchdown': 'sum',  # When they're the receiver
        'complete_pass': 'sum',  # Receptions
    }).reset_index()

    for _, row in receiver_stats.iterrows():
        # Count targets (all pass attempts to this receiver)
        targets = len(game_plays[
            (game_plays['receiver_player_id'] == row['receiver_player_id']) &
            (game_plays['pass_attempt'] == 1)
        ])

        player_stats.append({
            'player_id': row['receiver_player_id'],
            'player_name': row['receiver_player_name'],
            'team': row['posteam'],
            'week': week,
            'season': season,
            'game_id': game_id,
            'position': 'WR',  # We'll need to infer this better
            'passing_yards': 0,
            'passing_tds': 0,
            'completions': 0,
            'attempts': 0,
            'interceptions': 0,
            'rushing_yards': 0,
            'rushing_tds': 0,
            'carries': 0,
            'receiving_yards': row['receiving_yards'],
            'receiving_tds': row['pass_touchdown'],
            'receptions': row['complete_pass'],
            'targets': targets,
        })

print(f"✅ Aggregated {len(player_stats)} player-game records\n")

# Create DataFrame
stats_df = pd.DataFrame(player_stats)

# Add rolling averages and features
print("Computing rolling averages and features...")
stats_df = stats_df.sort_values(['player_id', 'week'])

for stat in ['passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds',
             'receiving_yards', 'receiving_tds', 'receptions', 'carries']:
    stats_df[f'season_avg_{stat}'] = stats_df.groupby('player_id')[stat].expanding().mean().reset_index(drop=True)
    stats_df[f'l3_avg_{stat}'] = stats_df.groupby('player_id')[stat].rolling(3, min_periods=1).mean().reset_index(drop=True)

stats_df['games_played'] = stats_df.groupby('player_id').cumcount() + 1

print(f"✅ Added rolling averages\n")

# Save to inputs
output_file = Path('/home/user/nfl_backend/inputs/player_stats_2025_from_pbp.csv')
output_file.parent.mkdir(parents=True, exist_ok=True)
stats_df.to_csv(output_file, index=False)
print(f"✅ Saved {len(stats_df)} records to {output_file}")
print(f"   Players: {stats_df['player_id'].nunique()}")
print(f"   Weeks: {sorted(stats_df['week'].unique())}")
print(f"   Positions: {stats_df['position'].value_counts().to_dict()}")

print("\n" + "="*80)
print("REAL PLAYER STATS AGGREGATION COMPLETE")
print("="*80)
