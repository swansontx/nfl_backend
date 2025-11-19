"""Prepare proper training data from REAL PBP - one record per player per game."""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("PREPARING TRAINING DATA FROM REAL PBP")
print("="*80 + "\n")

# Load REAL PBP
pbp = pd.read_parquet('/home/user/nfl_backend/inputs/play_by_play_2025.parquet')
print(f"✅ Loaded {len(pbp):,} real plays\n")

# Aggregate ALL stats per player per game
player_game_stats = []

for game_id in pbp['game_id'].unique():
    game = pbp[pbp['game_id'] == game_id]
    week = game['week'].iloc[0]
    home_team = game['home_team'].iloc[0]
    away_team = game['away_team'].iloc[0]

    # Collect all players who participated
    all_players = {}

    # Passing stats
    passers = game[game['passer_player_id'].notna()]
    for pid in passers['passer_player_id'].unique():
        p_plays = passers[passers['passer_player_id'] == pid]
        if pid not in all_players:
            all_players[pid] = {
                'player_id': pid,
                'player_name': p_plays['passer_player_name'].iloc[0] if 'passer_player_name' in p_plays else str(pid),
                'team': p_plays['posteam'].iloc[0],
                'position': 'QB',
            }
        all_players[pid]['passing_yards'] = p_plays['passing_yards'].sum()
        all_players[pid]['passing_tds'] = p_plays['pass_touchdown'].sum() if 'pass_touchdown' in p_plays else 0
        all_players[pid]['completions'] = p_plays['complete_pass'].sum() if 'complete_pass' in p_plays else 0
        all_players[pid]['attempts'] = p_plays['pass_attempt'].sum() if 'pass_attempt' in p_plays else 0
        all_players[pid]['interceptions'] = p_plays['interception'].sum() if 'interception' in p_plays else 0

    # Rushing stats
    rushers = game[game['rusher_player_id'].notna()]
    for pid in rushers['rusher_player_id'].unique():
        r_plays = rushers[rushers['rusher_player_id'] == pid]
        if pid not in all_players:
            all_players[pid] = {
                'player_id': pid,
                'player_name': r_plays['rusher_player_name'].iloc[0] if 'rusher_player_name' in r_plays else str(pid),
                'team': r_plays['posteam'].iloc[0],
                'position': 'RB',  # Will refine later
            }
        all_players[pid]['rushing_yards'] = r_plays['rushing_yards'].sum()
        all_players[pid]['rushing_tds'] = r_plays['rush_touchdown'].sum() if 'rush_touchdown' in r_plays else 0
        all_players[pid]['carries'] = r_plays['rush_attempt'].sum() if 'rush_attempt' in r_plays else 0

    # Receiving stats
    receivers = game[game['receiver_player_id'].notna()]
    for pid in receivers['receiver_player_id'].unique():
        rec_plays = receivers[receivers['receiver_player_id'] == pid]
        if pid not in all_players:
            all_players[pid] = {
                'player_id': pid,
                'player_name': rec_plays['receiver_player_name'].iloc[0] if 'receiver_player_name' in rec_plays else str(pid),
                'team': rec_plays['posteam'].iloc[0],
                'position': 'WR',  # Will refine later
            }
        all_players[pid]['receiving_yards'] = rec_plays['receiving_yards'].sum()
        all_players[pid]['receiving_tds'] = rec_plays['pass_touchdown'].sum() if 'pass_touchdown' in rec_plays else 0
        all_players[pid]['receptions'] = rec_plays['complete_pass'].sum() if 'complete_pass' in rec_plays else 0
        all_players[pid]['targets'] = len(rec_plays[rec_plays['pass_attempt'] == 1])

    # Add records
    for pid, stats in all_players.items():
        opponent = away_team if stats['team'] == home_team else home_team
        record = {
            'player_id': pid,
            'player_name': stats.get('player_name', pid),
            'team': stats['team'],
            'opponent': opponent,
            'week': week,
            'season': 2025,
            'game_id': game_id,
            'position': stats['position'],
            'passing_yards': stats.get('passing_yards', 0),
            'passing_tds': stats.get('passing_tds', 0),
            'completions': stats.get('completions', 0),
            'attempts': stats.get('attempts', 0),
            'interceptions': stats.get('interceptions', 0),
            'rushing_yards': stats.get('rushing_yards', 0),
            'rushing_tds': stats.get('rushing_tds', 0),
            'carries': stats.get('carries', 0),
            'receiving_yards': stats.get('receiving_yards', 0),
            'receiving_tds': stats.get('receiving_tds', 0),
            'receptions': stats.get('receptions', 0),
            'targets': stats.get('targets', 0),
        }
        player_game_stats.append(record)

df = pd.DataFrame(player_game_stats)
print(f"✅ Aggregated {len(df)} player-game records")
print(f"   Players: {df['player_id'].nunique()}")
print(f"   Weeks: {sorted(df['week'].unique())}\n")

# Refine positions based on primary stat
def refine_position(row):
    if row['attempts'] > 5:
        return 'QB'
    elif row['receptions'] > 0 or row['targets'] > 0:
        if row['carries'] > row['receptions']:
            return 'RB'
        else:
            return 'WR'
    elif row['carries'] > 0:
        return 'RB'
    return row['position']

df['position'] = df.apply(refine_position, axis=1)

# Sort and add rolling features
df = df.sort_values(['player_id', 'week'])

# Add rolling averages for each stat
stats_to_roll = ['passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds',
                 'receiving_yards', 'receiving_tds', 'receptions', 'carries',
                 'completions', 'attempts', 'targets', 'interceptions']

for stat in stats_to_roll:
    df[f'season_avg_{stat}'] = df.groupby('player_id')[stat].expanding().mean().reset_index(drop=True)
    df[f'l3_avg_{stat}'] = df.groupby('player_id')[stat].rolling(3, min_periods=1).mean().reset_index(drop=True)

df['games_played'] = df.groupby('player_id').cumcount() + 1

print("✅ Added rolling averages\n")

# Save
output_file = Path('/home/user/nfl_backend/inputs/player_stats_2025.csv')
df.to_csv(output_file, index=False)
print(f"✅ Saved to: {output_file}")
print(f"\nPositions: {df['position'].value_counts().to_dict()}")

print("\n" + "="*80)
print("TRAINING DATA PREPARATION COMPLETE - 100% REAL DATA!")
print("="*80)
