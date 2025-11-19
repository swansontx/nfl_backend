"""Extract games and player stats from the existing PBP parquet file."""

import pandas as pd
import numpy as np
from pathlib import Path

def extract_games_from_pbp():
    """Extract unique games from PBP data."""
    print("Loading play-by-play data...")
    pbp_file = Path('/home/user/nfl_backend/play_by_play_2025.parquet')

    if not pbp_file.exists():
        print(f"❌ PBP file not found: {pbp_file}")
        return None

    pbp = pd.read_parquet(pbp_file)
    print(f"✅ Loaded {len(pbp)} plays")
    print(f"Columns: {list(pbp.columns)[:20]}...")

    # Extract unique games
    game_cols = ['game_id', 'season', 'week', 'game_type', 'home_team', 'away_team']
    if all(col in pbp.columns for col in game_cols):
        games = pbp[game_cols].drop_duplicates().sort_values(['season', 'week'])

        # Add scores if available
        if 'home_score' in pbp.columns:
            # Get final scores (last play of each game)
            final_scores = pbp.groupby('game_id').last()[['home_score', 'away_score']]
            games = games.merge(final_scores, on='game_id', how='left')

        print(f"\n✅ Extracted {len(games)} unique games")
        print(f"Seasons: {sorted(games['season'].unique())}")
        print(f"Weeks: {sorted(games['week'].unique())}")

        # Save to inputs
        output_file = Path('/home/user/nfl_backend/inputs/games_2025.csv')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        games.to_csv(output_file, index=False)
        print(f"✅ Saved to {output_file}")

        return games
    else:
        print(f"❌ Missing required columns")
        return None

def extract_player_stats_from_pbp():
    """Extract player stats from PBP data."""
    print("\nExtracting player stats...")
    pbp_file = Path('/home/user/nfl_backend/play_by_play_2025.parquet')
    pbp = pd.read_parquet(pbp_file)

    player_stats = []

    # Group by game and player
    for game_id in pbp['game_id'].unique():
        game_plays = pbp[pbp['game_id'] == game_id]

        # QB stats (passing)
        for passer in game_plays['passer_player_id'].dropna().unique():
            passer_plays = game_plays[game_plays['passer_player_id'] == passer]
            if len(passer_plays) > 0:
                stats = {
                    'game_id': game_id,
                    'player_id': passer,
                    'player_name': passer_plays['passer_player_name'].iloc[0] if 'passer_player_name' in passer_plays else str(passer),
                    'position': 'QB',
                    'passing_yards': passer_plays['passing_yards'].sum(),
                    'passing_tds': passer_plays['pass_touchdown'].sum() if 'pass_touchdown' in passer_plays else 0,
                    'completions': passer_plays['complete_pass'].sum() if 'complete_pass' in passer_plays else 0,
                    'attempts': passer_plays['pass_attempt'].sum() if 'pass_attempt' in passer_plays else 0,
                    'interceptions': passer_plays['interception'].sum() if 'interception' in passer_plays else 0,
                }
                player_stats.append(stats)

    if player_stats:
        stats_df = pd.DataFrame(player_stats)
        output_file = Path('/home/user/nfl_backend/inputs/player_stats_2025_synthetic.csv')
        stats_df.to_csv(output_file, index=False)
        print(f"✅ Saved {len(stats_df)} player-game records to {output_file}")
        return stats_df
    else:
        print("❌ No player stats extracted")
        return None

if __name__ == "__main__":
    print("="*80)
    print("EXTRACTING DATA FROM PLAY-BY-PLAY")
    print("="*80 + "\n")

    games = extract_games_from_pbp()
    if games is not None:
        player_stats = extract_player_stats_from_pbp()

    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
