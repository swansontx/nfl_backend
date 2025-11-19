"""Generate realistic synthetic player stats for 2025 model training.

This creates realistic player prop data based on actual 2025 game results.
Later we'll swap in real NFLverse data once we get the API access sorted.

For now, this lets us:
- Build the complete training pipeline
- Test models and backtesting
- Validate API integration
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_player_stats_from_games():
    """Generate realistic player stats based on 2025 game results."""

    print(f"\n{'='*80}")
    print(f"GENERATING SYNTHETIC PLAYER STATS FROM 2025 GAMES")
    print(f"{'='*80}\n")

    # Load games data
    games_file = Path('inputs/games_2025.csv')
    games = pd.read_csv(games_file)

    # Filter to 2025 regular season completed games
    games_2025 = games[
        (games['season'] == 2025) &
        (games['game_type'] == 'REG') &
        (games['home_score'].notna())
    ].copy()

    print(f"✅ Loaded {len(games_2025)} completed 2025 games")
    print()

    # Generate player stats for each game
    all_player_stats = []

    # Define realistic QB names by team (using actual 2025 QBs where known)
    qbs_by_team = {
        'BUF': 'Josh Allen',
        'KC': 'Patrick Mahomes',
        'BAL': 'Lamar Jackson',
        'DET': 'Jared Goff',
        'SF': 'Brock Purdy',
        'PHI': 'Jalen Hurts',
        'DAL': 'Dak Prescott',
        'MIA': 'Tua Tagovailoa',
        'CIN': 'Joe Burrow',
        'LAC': 'Justin Herbert',
        'MIN': 'Sam Darnold',
        'SEA': 'Geno Smith',
        'LA': 'Matthew Stafford',
        'TB': 'Baker Mayfield',
        'ATL': 'Kirk Cousins',
        'GB': 'Jordan Love',
        'HOU': 'CJ Stroud',
        'IND': 'Anthony Richardson',
        'NO': 'Derek Carr',
        'WAS': 'Jayden Daniels',
        'DEN': 'Bo Nix',
        'PIT': 'Russell Wilson',
        'ARI': 'Kyler Murray',
        'NYJ': 'Aaron Rodgers',
        'CHI': 'Caleb Williams',
        'CLE': 'Jameis Winston',
        'NYG': 'Daniel Jones',
        'CAR': 'Bryce Young',
        'TEN': 'Will Levis',
        'NE': 'Drake Maye',
        'LV': 'Gardner Minshew',
        'JAX': 'Trevor Lawrence',
    }

    # RBs by team (lead backs)
    rbs_by_team = {
        'BAL': 'Derrick Henry',
        'SF': 'Christian McCaffrey',
        'ATL': 'Bijan Robinson',
        'DET': 'Jahmyr Gibbs',
        'PHI': 'Saquon Barkley',
        'DAL': 'Tony Pollard',
        'MIA': 'De\'Von Achane',
        'CLE': 'Nick Chubb',
        'PIT': 'Najee Harris',
        'GB': 'Josh Jacobs',
        'MIN': 'Aaron Jones',
        'LAC': 'JK Dobbins',
        'KC': 'Isiah Pacheco',
        'HOU': 'Joe Mixon',
        'BUF': 'James Cook',
        'SEA': 'Kenneth Walker',
        'LA': 'Kyren Williams',
        'TB': 'Rachaad White',
        'IND': 'Jonathan Taylor',
        'NYJ': 'Breece Hall',
        'NO': 'Alvin Kamara',
        'WAS': 'Brian Robinson',
        'ARI': 'James Conner',
        'DEN': 'Javonte Williams',
        'CHI': 'D\'Andre Swift',
        'NYG': 'Devin Singletary',
        'CAR': 'Chuba Hubbard',
        'TEN': 'Tyjae Spears',
        'NE': 'Rhamondre Stevenson',
        'LV': 'Alexander Mattison',
        'JAX': 'Travis Etienne',
    }

    # WRs by team (WR1)
    wrs_by_team = {
        'DET': 'Amon-Ra St. Brown',
        'LA': 'Puka Nacua',
        'MIN': 'Justin Jefferson',
        'MIA': 'Tyreek Hill',
        'DAL': 'CeeDee Lamb',
        'SF': 'Deebo Samuel',
        'PHI': 'AJ Brown',
        'KC': 'Travis Kelce',  # TE but close enough
        'BUF': 'Stefon Diggs',
        'CIN': 'Ja\'Marr Chase',
        'LAC': 'Keenan Allen',
        'HOU': 'Nico Collins',
        'SEA': 'DK Metcalf',
        'TB': 'Mike Evans',
        'ATL': 'Drake London',
        'GB': 'Christian Watson',
        'NYJ': 'Garrett Wilson',
        'NO': 'Chris Olave',
        'IND': 'Michael Pittman',
        'WAS': 'Terry McLaurin',
        'BAL': 'Zay Flowers',
        'DEN': 'Courtland Sutton',
        'PIT': 'George Pickens',
        'ARI': 'Marvin Harrison Jr',
        'CHI': 'DJ Moore',
        'CLE': 'Amari Cooper',
        'NYG': 'Malik Nabers',
        'CAR': 'Diontae Johnson',
        'TEN': 'DeAndre Hopkins',
        'NE': 'Demario Douglas',
        'LV': 'Davante Adams',
        'JAX': 'Christian Kirk',
    }

    print(f"📊 Generating stats for {len(games_2025)} games...")
    print()

    for idx, game in games_2025.iterrows():
        game_id = game['game_id']
        week = game['week']
        away_team = game['away_team']
        home_team = game['home_team']
        away_score = game['away_score']
        home_score = game['home_score']
        total = away_score + home_score

        # Generate passing stats based on score
        for team, score, opponent_score in [(away_team, away_score, home_score), (home_team, home_score, away_score)]:
            qb_name = qbs_by_team.get(team, f"{team} QB")

            # Pass yards: roughly 10-12 yards per point scored + variance
            base_pass_yards = score * 11 + np.random.normal(0, 30)
            pass_yards = max(120, int(base_pass_yards))  # Min 120 yards

            # TDs: roughly 1 TD per 10 points + variance
            pass_tds = max(0, int(score / 10 + np.random.normal(0, 0.5)))

            # Completions: 60-70% of attempts
            attempts = int(pass_yards / 7.5 + np.random.normal(0, 3))
            completions = int(attempts * 0.65)

            # Interceptions: ~2-3% INT rate
            int_rate = 0.025  # League average
            interceptions = 0
            if np.random.random() < int_rate * attempts / 30:  # Scale by attempts
                interceptions = 1

            all_player_stats.append({
                'player_id': f"qb_{team.lower()}",
                'player_name': qb_name,
                'team': team,
                'opponent': home_team if team == away_team else away_team,
                'week': week,
                'season': 2025,
                'game_id': game_id,
                'position': 'QB',
                'passing_yards': pass_yards,
                'passing_tds': pass_tds,
                'completions': completions,
                'attempts': attempts,
                'interceptions': interceptions,
                'rushing_yards': 0,
                'rushing_tds': 0,
                'carries': 0,
                'receiving_yards': 0,
                'receiving_tds': 0,
                'receptions': 0,
                'targets': 0,
            })

            # Generate RB stats
            rb_name = rbs_by_team.get(team, f"{team} RB")

            # Rush yards: varies by game script
            # If winning big, more rushing. If losing, less rushing.
            score_diff = score - opponent_score

            if score_diff > 10:  # Winning big = more rushing
                base_rush_yards = 90 + np.random.normal(0, 20)
            elif score_diff < -10:  # Losing big = less rushing
                base_rush_yards = 50 + np.random.normal(0, 15)
            else:  # Close game
                base_rush_yards = 70 + np.random.normal(0, 20)

            rush_yards = max(20, int(base_rush_yards))
            carries = int(rush_yards / 4.5 + np.random.normal(0, 2))
            rush_tds = max(0, int(score / 15 + np.random.normal(0, 0.3)))

            # RB receiving
            rec_yards = int(np.random.normal(30, 15))
            receptions = int(rec_yards / 8)
            targets = int(receptions * 1.3)

            all_player_stats.append({
                'player_id': f"rb_{team.lower()}",
                'player_name': rb_name,
                'team': team,
                'opponent': home_team if team == away_team else away_team,
                'week': week,
                'season': 2025,
                'game_id': game_id,
                'position': 'RB',
                'passing_yards': 0,
                'passing_tds': 0,
                'completions': 0,
                'attempts': 0,
                'interceptions': 0,
                'rushing_yards': rush_yards,
                'rushing_tds': rush_tds,
                'carries': carries,
                'receiving_yards': rec_yards,
                'receiving_tds': 0,
                'receptions': receptions,
                'targets': targets,
            })

            # Generate WR stats
            wr_name = wrs_by_team.get(team, f"{team} WR")

            # WR receiving: 25-35% of team passing yards
            wr_rec_yards = int(pass_yards * 0.30 + np.random.normal(0, 20))
            wr_receptions = int(wr_rec_yards / 12)
            wr_targets = int(wr_receptions * 1.4)
            wr_rec_tds = max(0, int(pass_tds * 0.4))

            all_player_stats.append({
                'player_id': f"wr_{team.lower()}",
                'player_name': wr_name,
                'team': team,
                'opponent': home_team if team == away_team else away_team,
                'week': week,
                'season': 2025,
                'game_id': game_id,
                'position': 'WR',
                'passing_yards': 0,
                'passing_tds': 0,
                'completions': 0,
                'attempts': 0,
                'interceptions': 0,
                'rushing_yards': 0,
                'rushing_tds': 0,
                'carries': 0,
                'receiving_yards': wr_rec_yards,
                'receiving_tds': wr_rec_tds,
                'receptions': wr_receptions,
                'targets': wr_targets,
            })

    stats_df = pd.DataFrame(all_player_stats)

    print(f"✅ Generated {len(stats_df)} player-game records")
    print()

    # Save to file
    output_file = Path('inputs/player_stats_2025_synthetic.csv')
    stats_df.to_csv(output_file, index=False)

    print(f"💾 Saved to: {output_file}")
    print()

    # Show summary
    print(f"📊 Summary by position:")
    for pos, count in stats_df['position'].value_counts().items():
        print(f"   {pos}: {count} performances")

    print()

    print(f"📊 Summary by week:")
    for week, count in sorted(stats_df['week'].value_counts().items()):
        print(f"   Week {week:2d}: {count} performances")

    print()

    return stats_df


if __name__ == '__main__':
    stats_df = generate_player_stats_from_games()

    print(f"\n{'='*80}")
    print(f"✅ SYNTHETIC PLAYER STATS GENERATED")
    print(f"{'='*80}\n")

    print("Next: Run fetch_player_stats_2025.py to create training datasets")
    print()
