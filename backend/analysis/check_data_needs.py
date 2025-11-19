"""Check what real data we have and what we need to download."""
import pandas as pd
from pathlib import Path

print("="*80)
print("CHECKING AVAILABLE REAL DATA")
print("="*80 + "\n")

# Check PBP data
pbp_file = Path('/home/user/nfl_backend/play_by_play_2025.parquet')
if pbp_file.exists():
    pbp = pd.read_parquet(pbp_file)
    print(f"✅ Play-by-Play: {len(pbp):,} plays, {pbp['game_id'].nunique()} games")
    print(f"   Weeks: {sorted(pbp['week'].unique())}")
    print(f"   Key columns available:")

    key_cols = ['passer_player_name', 'rusher_player_name', 'receiver_player_name',
                'passing_yards', 'rushing_yards', 'receiving_yards',
                'pass_touchdown', 'rush_touchdown', 'td_player_name',
                'td_player_id', 'home_score', 'away_score']

    for col in key_cols:
        if col in pbp.columns:
            print(f"   ✅ {col}")
        else:
            print(f"   ❌ {col}")
else:
    print("❌ No PBP data found")

print("\nWhat we need to download from NFLverse:")
print("1. Weekly player stats (passing, rushing, receiving per game)")
print("2. Team game logs (scores, opponents)")
print("3. Player info (positions, teams)")
print("\nLet me fetch these...")
