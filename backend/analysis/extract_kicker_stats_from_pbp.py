"""Extract REAL kicker stats from PBP data."""

import pandas as pd
from pathlib import Path

print("="*80)
print("EXTRACTING REAL KICKER STATS FROM PBP")
print("="*80 + "\n")

# Load PBP
pbp = pd.read_parquet('/home/user/nfl_backend/inputs/play_by_play_2025.parquet')
print(f"✅ Loaded {len(pbp):,} plays\n")

# Extract kicking plays
fg_plays = pbp[pbp['field_goal_attempt'] == 1].copy()
xp_plays = pbp[pbp['extra_point_attempt'] == 1].copy()

print(f"FG attempts: {len(fg_plays)}")
print(f"XP attempts: {len(xp_plays)}\n")

kicker_stats = []

# Group by game and kicker
for game_id in pbp['game_id'].unique():
    game_pbp = pbp[pbp['game_id'] == game_id]
    week = game_pbp['week'].iloc[0]

    # FGs by kicker
    game_fgs = fg_plays[fg_plays['game_id'] == game_id]
    if len(game_fgs) > 0:
        for kicker_id in game_fgs['kicker_player_id'].dropna().unique():
            kicker_fgs = game_fgs[game_fgs['kicker_player_id'] == kicker_id]
            kicker_name = kicker_fgs['kicker_player_name'].iloc[0] if 'kicker_player_name' in kicker_fgs else str(kicker_id)
            team = kicker_fgs['posteam'].iloc[0]

            fg_made = (kicker_fgs['field_goal_result'] == 'made').sum()
            fg_att = len(kicker_fgs)

            # Get XPs for same kicker
            game_xps = xp_plays[(xp_plays['game_id'] == game_id) & (xp_plays['kicker_player_id'] == kicker_id)]
            xp_made = (game_xps['extra_point_result'] == 'good').sum() if len(game_xps) > 0 else 0
            xp_att = len(game_xps)

            kicker_stats.append({
                'game_id': game_id,
                'week': week,
                'kicker_id': kicker_id,
                'kicker_name': kicker_name,
                'team': team,
                'fg_made': fg_made,
                'fg_att': fg_att,
                'xp_made': xp_made,
                'xp_att': xp_att,
                'total_points': (fg_made * 3) + xp_made,
            })

stats_df = pd.DataFrame(kicker_stats)

# Add rolling averages
stats_df = stats_df.sort_values(['kicker_id', 'week'])
for stat in ['fg_made', 'xp_made', 'total_points']:
    stats_df[f'season_avg_{stat}'] = stats_df.groupby('kicker_id')[stat].expanding().mean().reset_index(drop=True)
    stats_df[f'l3_avg_{stat}'] = stats_df.groupby('kicker_id')[stat].rolling(3, min_periods=1).mean().reset_index(drop=True)

stats_df['games_played'] = stats_df.groupby('kicker_id').cumcount() + 1

# Save
output_file = Path('/home/user/nfl_backend/inputs/kicker_stats_2025_from_pbp.csv')
stats_df.to_csv(output_file, index=False)

print(f"✅ Saved {len(stats_df)} kicker-game records")
print(f"   Kickers: {stats_df['kicker_id'].nunique()}")
print(f"   Weeks: {sorted(stats_df['week'].unique())}")
print(f"\n{output_file}")

print("\n" + "="*80)
print("KICKER STATS EXTRACTION COMPLETE")
print("="*80)
