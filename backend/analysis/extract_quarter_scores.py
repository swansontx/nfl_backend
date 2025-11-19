"""Extract quarter-level scores from play-by-play data."""

import pandas as pd
from pathlib import Path


def extract_quarter_scores():
    """Extract quarter scores from PBP data."""

    print(f"\n{'='*80}")
    print(f"EXTRACTING QUARTER SCORES FROM PBP DATA")
    print(f"{'='*80}\n")

    # Load PBP data
    pbp_file = Path('inputs/play_by_play_2025.parquet')
    pbp = pd.read_parquet(pbp_file)

    print(f"Loaded {len(pbp):,} plays")
    print()

    # Get scoring plays
    scoring_plays = pbp[
        ((pbp['touchdown'] == 1) | (pbp['field_goal_result'] == 'made') | (pbp['extra_point_result'] == 'good'))
        & (pbp['posteam'].notna())
    ].copy()

    print(f"Scoring plays: {len(scoring_plays)}")
    print()

    # Calculate points per play
    scoring_plays['points'] = 0
    scoring_plays.loc[scoring_plays['touchdown'] == 1, 'points'] = 6
    scoring_plays.loc[scoring_plays['field_goal_result'] == 'made', 'points'] = 3
    scoring_plays.loc[scoring_plays['extra_point_result'] == 'good', 'points'] += 1

    # Group by game, quarter, team
    quarter_scores = scoring_plays.groupby(['game_id', 'week', 'qtr', 'posteam', 'home_team', 'away_team'])['points'].sum().reset_index()

    print(f"Quarter scores: {len(quarter_scores)}")
    print()

    # Pivot to wide format
    games = []

    for game_id in quarter_scores['game_id'].unique():
        game_data = quarter_scores[quarter_scores['game_id'] == game_id]

        week = game_data['week'].iloc[0]
        home_team = game_data['home_team'].iloc[0]
        away_team = game_data['away_team'].iloc[0]

        game_row = {
            'game_id': game_id,
            'week': week,
            'home_team': home_team,
            'away_team': away_team,
        }

        # Get scores by quarter and team
        for qtr in range(1, 5):
            for team_type, team_name in [('home', home_team), ('away', away_team)]:
                team_qtr = game_data[(game_data['qtr'] == qtr) & (game_data['posteam'] == team_name)]
                if len(team_qtr) > 0:
                    game_row[f'{team_type}_q{qtr}'] = team_qtr['points'].sum()
                else:
                    game_row[f'{team_type}_q{qtr}'] = 0

        games.append(game_row)

    games_df = pd.DataFrame(games)

    # Calculate totals
    games_df['home_total'] = games_df[['home_q1', 'home_q2', 'home_q3', 'home_q4']].sum(axis=1)
    games_df['away_total'] = games_df[['away_q1', 'away_q2', 'away_q3', 'away_q4']].sum(axis=1)
    games_df['total_points'] = games_df['home_total'] + games_df['away_total']

    print(f"✅ Extracted {len(games_df)} games with quarter scores")
    print()

    print("Sample games:")
    print(games_df.head())
    print()

    # Save
    output_file = Path('inputs/games_with_quarters.csv')
    games_df.to_csv(output_file, index=False)

    print(f"💾 Saved to: {output_file}")
    print()

    return games_df


if __name__ == '__main__':
    extract_quarter_scores()
