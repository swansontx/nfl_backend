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
        # Include ALL props - let data and narratives determine value
        self.value_strategies = [
            # TD props - high EV when context supports
            {'prop': 'receiving_tds', 'dir': 'UNDER', 'buffer': 0.5, 'hit_rate': 82.2, 'odds': -120, 'ev': 50.7, 'category': 'scoring'},
            {'prop': 'rushing_tds', 'dir': 'UNDER', 'buffer': 0.5, 'hit_rate': 79.4, 'odds': -140, 'ev': 36.1, 'category': 'scoring'},
            {'prop': 'passing_tds', 'dir': 'UNDER', 'buffer': 0.5, 'hit_rate': 75.0, 'odds': -150, 'ev': 25.0, 'category': 'qb_scoring'},
            # Ground game props
            {'prop': 'rushing_yards', 'dir': 'UNDER', 'buffer': 15, 'hit_rate': 81.0, 'odds': -200, 'ev': 21.5, 'category': 'ground'},
            {'prop': 'rushing_yards', 'dir': 'UNDER', 'buffer': 10, 'hit_rate': 77.8, 'odds': -200, 'ev': 16.7, 'category': 'ground'},
            {'prop': 'rushing_yards', 'dir': 'UNDER', 'buffer': 5, 'hit_rate': 70.9, 'odds': -140, 'ev': 21.5, 'category': 'ground'},
            {'prop': 'carries', 'dir': 'UNDER', 'buffer': 3, 'hit_rate': 80.1, 'odds': -200, 'ev': 20.1, 'category': 'volume'},
            {'prop': 'carries', 'dir': 'UNDER', 'buffer': 2, 'hit_rate': 74.6, 'odds': -165, 'ev': 19.9, 'category': 'volume'},
            # Air attack props
            {'prop': 'receiving_yards', 'dir': 'UNDER', 'buffer': 12.5, 'hit_rate': 77.0, 'odds': -200, 'ev': 15.6, 'category': 'air'},
            {'prop': 'receiving_yards', 'dir': 'UNDER', 'buffer': 7.5, 'hit_rate': 70.9, 'odds': -165, 'ev': 13.9, 'category': 'air'},
            {'prop': 'receptions', 'dir': 'UNDER', 'buffer': 1.5, 'hit_rate': 81.6, 'odds': -300, 'ev': 5.7, 'category': 'volume'},
            {'prop': 'receptions', 'dir': 'UNDER', 'buffer': 1, 'hit_rate': 76.8, 'odds': -250, 'ev': 11.9, 'category': 'volume'},
            {'prop': 'targets', 'dir': 'UNDER', 'buffer': 2, 'hit_rate': 75.0, 'odds': -200, 'ev': 12.5, 'category': 'volume'},
            # QB props
            {'prop': 'passing_yards', 'dir': 'UNDER', 'buffer': 25, 'hit_rate': 72.0, 'odds': -165, 'ev': 14.0, 'category': 'qb'},
            {'prop': 'completions', 'dir': 'UNDER', 'buffer': 3, 'hit_rate': 70.0, 'odds': -140, 'ev': 12.0, 'category': 'qb'},
            {'prop': 'attempts', 'dir': 'UNDER', 'buffer': 4, 'hit_rate': 68.0, 'odds': -130, 'ev': 10.0, 'category': 'qb'},
            {'prop': 'interceptions', 'dir': 'UNDER', 'buffer': 0.5, 'hit_rate': 73.3, 'odds': -165, 'ev': 17.8, 'category': 'qb_turnover'},
            # OVER props for specific contexts
            {'prop': 'interceptions', 'dir': 'OVER', 'buffer': -0.5, 'hit_rate': 45.0, 'odds': +110, 'ev': -2.0, 'category': 'qb_turnover'},  # risky but narrative-driven
        ]

        # Narrative themes for parlays - contextual stories across all prop types
        self.narratives = {
            'ground_game_stuffed': {
                'name': 'Ground Game Stuffed',
                'description': 'Strong run defense limits RB production - yards, carries, and TDs all under',
                'categories': ['ground', 'volume', 'scoring'],
                'positions': ['RB'],
                'props': ['rushing_yards', 'carries', 'rushing_tds'],
            },
            'secondary_lockdown': {
                'name': 'Secondary Lockdown',
                'description': 'Receivers struggle against tight coverage - yards, catches, TDs limited',
                'categories': ['air', 'volume', 'scoring'],
                'positions': ['WR', 'TE'],
                'props': ['receiving_yards', 'receptions', 'targets', 'receiving_tds'],
            },
            'qb_under_pressure': {
                'name': 'QB Under Pressure',
                'description': 'Pass rush disrupts timing - low completions, yards, possible INTs',
                'categories': ['qb', 'qb_turnover', 'qb_scoring'],
                'positions': ['QB'],
                'props': ['passing_yards', 'completions', 'attempts', 'passing_tds', 'interceptions'],
            },
            'game_script_blowout': {
                'name': 'Game Script - Trailing Team',
                'description': 'Team falls behind, abandons run, backup time - all unders',
                'categories': ['ground', 'volume', 'scoring'],
                'positions': ['RB'],
                'props': ['rushing_yards', 'carries', 'rushing_tds'],
                'role': 'starter',
            },
            'red_zone_woes': {
                'name': 'Red Zone Struggles',
                'description': 'Team moves ball but struggles to punch it in - TD unders across the board',
                'categories': ['scoring', 'qb_scoring'],
                'positions': ['QB', 'RB', 'WR', 'TE'],
                'props': ['rushing_tds', 'receiving_tds', 'passing_tds'],
            },
            'committee_backfield': {
                'name': 'Committee Backfield',
                'description': 'Carries split between multiple backs - individual unders on volume',
                'categories': ['ground', 'volume', 'scoring'],
                'positions': ['RB'],
                'props': ['rushing_yards', 'carries', 'rushing_tds'],
            },
            'depth_chart_quiet': {
                'name': 'Depth Chart Stays Quiet',
                'description': 'Backup/rotational players with low projections stay under',
                'categories': ['ground', 'air', 'volume'],
                'positions': ['RB', 'WR', 'TE'],
                'props': ['rushing_yards', 'receiving_yards', 'receptions', 'carries'],
                'role': 'backup',
            },
            'turnover_prone_qb': {
                'name': 'Turnover-Prone QB',
                'description': 'QB with INT history facing aggressive D - consider INT over, completions under',
                'categories': ['qb', 'qb_turnover'],
                'positions': ['QB'],
                'props': ['interceptions', 'completions', 'passing_yards'],
                'include_overs': True,  # This narrative can include INT overs
            },
        }

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

                # Determine if player is a backup based on projection thresholds
                is_backup = False
                if strategy['prop'] == 'rushing_yards' and projection < 40:
                    is_backup = True
                elif strategy['prop'] == 'receiving_yards' and projection < 35:
                    is_backup = True
                elif strategy['prop'] == 'carries' and projection < 8:
                    is_backup = True
                elif strategy['prop'] == 'receptions' and projection < 3:
                    is_backup = True

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
                    'category': strategy.get('category', 'other'),
                    'is_backup': is_backup,
                })

        # Sort by EV
        picks.sort(key=lambda x: -x['est_ev'])

        return picks

    def build_narrative_parlays(self, picks: List[Dict], team1: str, team2: str) -> Dict[str, List[Dict]]:
        """Build parlays organized by narrative themes."""
        parlays = {}

        for narrative_key, narrative in self.narratives.items():
            # Filter picks matching this narrative
            matching = []
            for p in picks:
                # Check if prop type matches narrative's props
                if 'props' in narrative and p['prop'] not in narrative['props']:
                    continue

                # Check position match
                if p['position'] not in narrative['positions']:
                    continue

                # Check backup role if specified
                if narrative.get('role') == 'backup' and not p.get('is_backup', False):
                    continue
                if narrative.get('role') == 'starter' and p.get('is_backup', False):
                    continue

                # Handle OVER props - only include if narrative allows
                if p['direction'] == 'OVER' and not narrative.get('include_overs', False):
                    continue

                matching.append(p)

            # Sort by hit rate (safer bets first), then by EV
            matching.sort(key=lambda x: (-x['est_hit_rate'], -x['est_ev']))

            # Remove duplicates (same player+prop, keep highest hit rate)
            seen = set()
            unique = []
            for p in matching:
                key = f"{p['player']}_{p['prop']}"
                if key not in seen:
                    seen.add(key)
                    unique.append(p)

            parlays[narrative_key] = unique

        return parlays

    def print_picks(self, picks: List[Dict], game: str, team1: str, team2: str):
        """Print formatted narrative-based parlays."""
        print(f"\n{'='*70}")
        print(f"NARRATIVE PARLAY PICKS: {game}")
        print(f"{'='*70}")

        # Build narrative parlays
        narrative_parlays = self.build_narrative_parlays(picks, team1, team2)

        # Print each narrative
        for narrative_key, narrative in self.narratives.items():
            parlay_picks = narrative_parlays.get(narrative_key, [])

            if len(parlay_picks) < 2:
                continue

            print(f"\n\n{'='*70}")
            print(f"NARRATIVE: {narrative['name']}")
            print(f"{narrative['description']}")
            print(f"{'='*70}")

            # Show individual picks
            print("\nAvailable Picks:")
            print("-" * 70)
            for i, p in enumerate(parlay_picks[:8], 1):
                backup_tag = " [BACKUP]" if p.get('is_backup') else ""
                print(f"{i}. {p['player']}{backup_tag} ({p['team']})")
                print(f"   {p['prop']} {p['direction']} {p['line']}")
                print(f"   Proj: {p['projection']} | Hit: {p['est_hit_rate']}% | Odds: {p['est_odds']}")

            # Build suggested parlays for this narrative
            if len(parlay_picks) >= 2:
                print("\nSuggested 2-Leg:")
                legs = parlay_picks[:2]
                combined = 1
                for p in legs:
                    print(f"  • {p['player']} {p['prop']} {p['direction']} {p['line']} ({p['est_hit_rate']}%)")
                    combined *= (p['est_hit_rate']/100)
                print(f"  Combined hit rate: {combined*100:.1f}%")

            if len(parlay_picks) >= 3:
                print("\nSuggested 3-Leg:")
                legs = parlay_picks[:3]
                combined = 1
                for p in legs:
                    print(f"  • {p['player']} {p['prop']} {p['direction']} {p['line']} ({p['est_hit_rate']}%)")
                    combined *= (p['est_hit_rate']/100)
                print(f"  Combined hit rate: {combined*100:.1f}%")

        # Cross-narrative "Best of" parlay
        print(f"\n\n{'='*70}")
        print("RECOMMENDED: CROSS-NARRATIVE PARLAY")
        print("Mix of ground, air, and volume for maximum diversification")
        print(f"{'='*70}")

        # Get best from each category (excluding TDs)
        best_ground = [p for p in picks if p['category'] == 'ground']
        best_air = [p for p in picks if p['category'] == 'air']
        best_volume = [p for p in picks if p['category'] == 'volume']

        # Sort each by hit rate
        best_ground.sort(key=lambda x: -x['est_hit_rate'])
        best_air.sort(key=lambda x: -x['est_hit_rate'])
        best_volume.sort(key=lambda x: -x['est_hit_rate'])

        cross_parlay = []
        if best_ground:
            cross_parlay.append(best_ground[0])
        if best_air:
            cross_parlay.append(best_air[0])
        if best_volume and len(cross_parlay) < 3:
            # Avoid duplicate players
            for p in best_volume:
                if p['player'] not in [x['player'] for x in cross_parlay]:
                    cross_parlay.append(p)
                    break

        if len(cross_parlay) >= 2:
            print("\nDiversified 3-Leg Parlay:")
            combined = 1
            for p in cross_parlay[:3]:
                print(f"  • {p['player']} {p['prop']} {p['direction']} {p['line']} ({p['est_hit_rate']}%)")
                combined *= (p['est_hit_rate']/100)
            print(f"  Combined hit rate: {combined*100:.1f}%")

        # Optional: Add one TD for "spice"
        td_picks = [p for p in picks if p['category'] == 'td']
        if td_picks and len(cross_parlay) >= 2:
            td_picks.sort(key=lambda x: -x['est_hit_rate'])
            spice_pick = td_picks[0]
            print("\n  Optional 'Spice' Add:")
            print(f"  + {spice_pick['player']} {spice_pick['prop']} {spice_pick['direction']} {spice_pick['line']} ({spice_pick['est_hit_rate']}%)")
            combined_with_spice = combined * (spice_pick['est_hit_rate']/100)
            print(f"  4-Leg with spice: {combined_with_spice*100:.1f}% hit rate")


def main():
    generator = GamePicksGenerator()

    # Generate picks for HOU vs BUF
    team1, team2 = 'HOU', 'BUF'
    picks = generator.generate_picks(team1, team2)
    generator.print_picks(picks, "Houston Texans @ Buffalo Bills", team1, team2)

    # Save to file
    output_file = Path("outputs/picks_HOU_BUF.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(picks, f, indent=2)

    print(f"\n📁 Full picks saved to: {output_file}")


if __name__ == "__main__":
    main()
