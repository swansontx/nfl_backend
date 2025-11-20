"""
Generate value-optimized parlay picks for a specific game.

Uses our validated strategy:
- Focus on UNDERs with buffers
- Target 70-85% hit rate
- Reasonable odds (-120 to -250)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import json


class GamePicksGenerator:
    """Generate parlay picks for a specific matchup."""

    def __init__(self, inputs_dir: str = "inputs"):
        self.inputs_dir = Path(inputs_dir)
        self.stats = pd.read_csv(self.inputs_dir / "player_stats_2024_2025.csv", low_memory=False)

        self.prop_map = {
            'passing_yards': 'passing_yards',
            'passing_tds': 'passing_tds',
            'rushing_yards': 'rushing_yards',
            'rushing_tds': 'rushing_tds',
            'receiving_yards': 'receiving_yards',
            'receptions': 'receptions',
            'receiving_tds': 'receiving_tds',
            'completions': 'completions',
            'attempts': 'attempts',
            'interceptions': 'passing_interceptions',
            'carries': 'carries',
            'targets': 'targets',
        }

        # Our validated value strategies (from backtest)
        self.value_strategies = [
            {'prop': 'receiving_tds', 'dir': 'UNDER', 'buffer': 0.5, 'hit_rate': 82.2, 'odds': -120, 'ev': 50.7},
            {'prop': 'rushing_tds', 'dir': 'UNDER', 'buffer': 0.5, 'hit_rate': 79.4, 'odds': -140, 'ev': 36.1},
            {'prop': 'rushing_yards', 'dir': 'UNDER', 'buffer': 15, 'hit_rate': 81.0, 'odds': -200, 'ev': 21.5},
            {'prop': 'rushing_yards', 'dir': 'UNDER', 'buffer': 10, 'hit_rate': 77.8, 'odds': -200, 'ev': 16.7},
            {'prop': 'rushing_yards', 'dir': 'UNDER', 'buffer': 5, 'hit_rate': 70.9, 'odds': -140, 'ev': 21.5},
            {'prop': 'carries', 'dir': 'UNDER', 'buffer': 3, 'hit_rate': 80.1, 'odds': -200, 'ev': 20.1},
            {'prop': 'carries', 'dir': 'UNDER', 'buffer': 2, 'hit_rate': 74.6, 'odds': -165, 'ev': 19.9},
            {'prop': 'interceptions', 'dir': 'UNDER', 'buffer': 0.5, 'hit_rate': 73.3, 'odds': -165, 'ev': 17.8},
            {'prop': 'receiving_yards', 'dir': 'UNDER', 'buffer': 12.5, 'hit_rate': 77.0, 'odds': -200, 'ev': 15.6},
            {'prop': 'receiving_yards', 'dir': 'UNDER', 'buffer': 7.5, 'hit_rate': 70.9, 'odds': -165, 'ev': 13.9},
            {'prop': 'receptions', 'dir': 'UNDER', 'buffer': 1.5, 'hit_rate': 81.6, 'odds': -300, 'ev': 5.7},
            {'prop': 'receptions', 'dir': 'UNDER', 'buffer': 1, 'hit_rate': 76.8, 'odds': -250, 'ev': 11.9},
        ]

    def get_projection(self, player_id: str, stat_col: str, n_weeks: int = 4) -> float:
        """Get weighted average projection for a player."""
        player_df = self.stats[
            self.stats['player_id'] == player_id
        ].sort_values('week', ascending=False)

        if len(player_df) == 0 or stat_col not in player_df.columns:
            return 0

        values = player_df[stat_col].head(n_weeks).dropna().tolist()
        if not values:
            return 0

        # Recency weighted
        weights = [0.4, 0.3, 0.2, 0.1][:len(values)]
        return sum(v * w for v, w in zip(values, weights)) / sum(weights)

    def generate_picks(self, team1: str, team2: str) -> List[Dict]:
        """Generate value picks for a matchup."""
        picks = []

        # Get players from both teams
        teams = [team1, team2]
        max_week = int(self.stats['week'].max())

        players = self.stats[
            (self.stats['week'] >= max_week - 3) &
            (self.stats['team'].isin(teams)) &
            (self.stats['position'].isin(['QB', 'RB', 'WR', 'TE']))
        ][['player_id', 'player_display_name', 'team', 'position']].drop_duplicates()

        for _, player in players.iterrows():
            player_id = player['player_id']
            player_name = player['player_display_name']
            team = player['team']
            position = player['position']

            # Determine which props to check
            if position == 'QB':
                props = ['passing_yards', 'passing_tds', 'interceptions', 'rushing_yards']
            elif position == 'RB':
                props = ['rushing_yards', 'rushing_tds', 'carries', 'receptions', 'receiving_yards']
            elif position in ['WR', 'TE']:
                props = ['receptions', 'receiving_yards', 'receiving_tds', 'targets']
            else:
                continue

            for strategy in self.value_strategies:
                if strategy['prop'] not in props:
                    continue

                stat_col = self.prop_map.get(strategy['prop'], strategy['prop'])
                projection = self.get_projection(player_id, stat_col)

                if projection <= 0:
                    continue

                # Skip very low projections
                if strategy['prop'] in ['receiving_tds', 'rushing_tds'] and projection < 0.2:
                    continue
                if strategy['prop'] == 'interceptions' and projection < 0.3:
                    continue

                # Calculate the line
                if strategy['dir'] == 'UNDER':
                    line = projection + strategy['buffer']
                else:
                    line = projection - strategy['buffer']

                # Round line to typical sportsbook increments
                if strategy['prop'] in ['passing_yards']:
                    line = round(line / 5) * 5
                elif strategy['prop'] in ['rushing_yards', 'receiving_yards']:
                    line = round(line * 2) / 2
                else:
                    line = round(line * 2) / 2

                picks.append({
                    'player': player_name,
                    'team': team,
                    'position': position,
                    'prop': strategy['prop'],
                    'direction': strategy['dir'],
                    'projection': round(projection, 1),
                    'line': line,
                    'buffer': strategy['buffer'],
                    'est_hit_rate': strategy['hit_rate'],
                    'est_odds': strategy['odds'],
                    'est_ev': strategy['ev'],
                })

        # Sort by EV
        picks.sort(key=lambda x: -x['est_ev'])

        return picks

    def print_picks(self, picks: List[Dict], game: str):
        """Print formatted picks."""
        print(f"\n{'='*70}")
        print(f"VALUE PARLAY PICKS: {game}")
        print(f"{'='*70}")

        # Group by tier
        tier1 = [p for p in picks if p['est_ev'] >= 30]
        tier2 = [p for p in picks if 15 <= p['est_ev'] < 30]
        tier3 = [p for p in picks if p['est_ev'] < 15]

        if tier1:
            print("\n🌟 TIER 1 - BEST VALUE (30%+ EV)")
            print("-" * 70)
            for p in tier1[:8]:
                print(f"{p['player']:20} {p['prop']:15} {p['direction']} {p['line']}")
                print(f"   Proj: {p['projection']} | Hit: {p['est_hit_rate']}% | Odds: {p['est_odds']} | EV: +{p['est_ev']}%")

        if tier2:
            print("\n⭐ TIER 2 - GREAT VALUE (15-30% EV)")
            print("-" * 70)
            for p in tier2[:10]:
                print(f"{p['player']:20} {p['prop']:15} {p['direction']} {p['line']}")
                print(f"   Proj: {p['projection']} | Hit: {p['est_hit_rate']}% | Odds: {p['est_odds']} | EV: +{p['est_ev']}%")

        # Suggested parlays
        print("\n\n🎰 SUGGESTED PARLAYS")
        print("-" * 70)

        # 2-leg conservative
        if len(tier1) >= 2:
            print("\n2-LEG CONSERVATIVE:")
            for p in tier1[:2]:
                print(f"  • {p['player']} {p['prop']} {p['direction']} {p['line']} ({p['est_hit_rate']}%)")
            combined = (tier1[0]['est_hit_rate']/100) * (tier1[1]['est_hit_rate']/100)
            print(f"  Combined: {combined*100:.1f}% hit rate")

        # 3-leg balanced
        all_good = tier1 + tier2
        if len(all_good) >= 3:
            print("\n3-LEG BALANCED:")
            for p in all_good[:3]:
                print(f"  • {p['player']} {p['prop']} {p['direction']} {p['line']} ({p['est_hit_rate']}%)")
            combined = 1
            for p in all_good[:3]:
                combined *= (p['est_hit_rate']/100)
            print(f"  Combined: {combined*100:.1f}% hit rate")


def main():
    generator = GamePicksGenerator()

    # Generate picks for HOU vs BUF
    picks = generator.generate_picks('HOU', 'BUF')
    generator.print_picks(picks, "Houston Texans @ Buffalo Bills")

    # Save to file
    output_file = Path("outputs/picks_HOU_BUF.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(picks, f, indent=2)

    print(f"\n📁 Full picks saved to: {output_file}")


if __name__ == "__main__":
    main()
