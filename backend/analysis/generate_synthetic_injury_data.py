"""Generate synthetic but realistic injury/status data for testing.

Since NFLverse API is blocked, create realistic injury patterns based on:
- ~10% of players OUT each week
- ~5% QUESTIONABLE
- Rest ACTIVE
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_injury_data_for_players(player_stats_df: pd.DataFrame) -> pd.DataFrame:
    """Generate realistic injury status for all players in player_stats.

    Args:
        player_stats_df: Player stats DataFrame with player_id, player_name, week, position

    Returns:
        DataFrame with columns: player_id, player_name, week, position, status
    """

    print(f"\n{'='*80}")
    print(f"GENERATING SYNTHETIC INJURY DATA")
    print(f"{'='*80}\n")

    # Get unique player-week combinations
    injury_df = player_stats_df[['player_id', 'player_name', 'week', 'position']].copy()

    print(f"Total player-week records: {len(injury_df)}")
    print()

    # Assign status based on realistic distributions
    np.random.seed(42)  # For reproducibility

    statuses = []

    for idx, row in injury_df.iterrows():
        rand = np.random.random()

        # 85% active, 10% out, 5% questionable
        if rand < 0.85:
            status = 'ACT'
        elif rand < 0.95:
            status = 'OUT'
        else:
            status = 'Q'

        statuses.append(status)

    injury_df['status'] = statuses

    # Count by status
    status_counts = injury_df['status'].value_counts()

    print("Status distribution:")
    for status, count in status_counts.items():
        pct = count / len(injury_df) * 100
        print(f"  {status}: {count} ({pct:.1f}%)")
    print()

    # Make sure key players are active for Week 10 (for our backtest)
    key_qbs = ['Josh Allen', 'Patrick Mahomes', 'Lamar Jackson', 'Jalen Hurts']

    for qb in key_qbs:
        injury_df.loc[
            (injury_df['player_name'] == qb) & (injury_df['week'] == 10),
            'status'
        ] = 'ACT'

    return injury_df


def main():
    """Generate synthetic injury data from our player stats."""

    # Load player stats
    player_stats_file = Path('inputs/player_stats_2025_synthetic.csv')

    if not player_stats_file.exists():
        print(f"❌ Player stats not found: {player_stats_file}")
        print("   Run: python -m backend.analysis.generate_synthetic_player_stats")
        return

    player_stats = pd.read_csv(player_stats_file)

    # Generate injury data
    injury_df = generate_injury_data_for_players(player_stats)

    # Save
    output_file = Path('inputs/weekly_rosters_2025_synthetic.csv')
    injury_df.to_csv(output_file, index=False)

    print(f"💾 Saved synthetic injury data to: {output_file}")
    print()

    # Show Week 10 examples
    print(f"\n{'='*80}")
    print(f"WEEK 10 INJURY EXAMPLES")
    print(f"{'='*80}\n")

    week10 = injury_df[injury_df['week'] == 10].copy()

    print("Sample OUT players:")
    out_players = week10[week10['status'] == 'OUT'].head(10)
    for idx, row in out_players.iterrows():
        print(f"  ❌ {row['player_name']:30s} ({row['position']:3s}) - OUT")
    print()

    print("Sample QUESTIONABLE players:")
    q_players = week10[week10['status'] == 'Q'].head(10)
    for idx, row in q_players.iterrows():
        print(f"  ⚠️  {row['player_name']:30s} ({row['position']:3s}) - Q")
    print()

    print("Key QBs (should all be ACTIVE):")
    key_qbs = week10[week10['player_name'].isin(['Josh Allen', 'Patrick Mahomes', 'Lamar Jackson', 'Jalen Hurts'])]
    for idx, row in key_qbs.iterrows():
        status_emoji = '✅' if row['status'] == 'ACT' else '❌'
        print(f"  {status_emoji} {row['player_name']:30s} - {row['status']}")
    print()


if __name__ == '__main__':
    main()
