"""
Parlay-Focused Backtest - Maximize hit rates for parlay legs.

For parlays, hit rate >> ROI because:
- 4 legs at 60% = 13% parlay success
- 4 legs at 75% = 31.6% parlay success
- 4 legs at 80% = 41% parlay success

We want to find the most reliable props with conservative buffers,
even if individual bet ROI is lower.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import defaultdict


class ParlayBacktester:
    """Find highest hit rate strategies for parlay legs."""

    def __init__(self, season: int = 2024, inputs_dir: str = "inputs"):
        self.season = season
        self.inputs_dir = Path(inputs_dir)

        self.player_stats = self._load_player_stats()
        self.schedules = self._load_schedules()

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

        self.position_props = {
            'QB': ['passing_yards', 'passing_tds', 'completions', 'attempts', 'interceptions'],
            'RB': ['rushing_yards', 'rushing_tds', 'carries', 'receptions', 'receiving_yards'],
            'WR': ['receptions', 'receiving_yards', 'receiving_tds', 'targets'],
            'TE': ['receptions', 'receiving_yards', 'receiving_tds', 'targets'],
        }

        # Test absolute buffers (not percentages) for more intuitive understanding
        # e.g., "10 yards under projection"
        self.absolute_buffers = {
            'passing_yards': [10, 20, 30, 40, 50],
            'rushing_yards': [5, 10, 15, 20, 25],
            'receiving_yards': [5, 10, 15, 20, 25],
            'receptions': [0.5, 1, 1.5, 2, 2.5],
            'targets': [0.5, 1, 1.5, 2, 2.5],
            'carries': [1, 2, 3, 4, 5],
            'completions': [1, 2, 3, 4, 5],
            'attempts': [1, 2, 3, 4, 5],
            'passing_tds': [0.5, 1],
            'rushing_tds': [0.5],
            'receiving_tds': [0.5],
            'interceptions': [0.5],
        }

    def _load_player_stats(self) -> pd.DataFrame:
        stats_file = self.inputs_dir / "player_stats_2024_2025.csv"
        if stats_file.exists():
            return pd.read_csv(stats_file, low_memory=False)
        return pd.DataFrame()

    def _load_schedules(self) -> pd.DataFrame:
        sched_file = self.inputs_dir / "schedules_2024_2025.csv"
        if sched_file.exists():
            return pd.read_csv(sched_file)
        return pd.DataFrame()

    def get_projection(self, player_id: str, max_week: int, stat_col: str) -> float:
        player_df = self.player_stats[
            (self.player_stats['player_id'] == player_id) &
            (self.player_stats['week'] < max_week)
        ].sort_values('week', ascending=False)

        if len(player_df) == 0 or stat_col not in player_df.columns:
            return 0

        values = player_df[stat_col].head(4).dropna().tolist()
        return np.mean(values) if values else 0

    def run_backtest(self, weeks: List[int] = [9, 10]) -> Dict:
        """Run parlay-focused backtest."""
        print(f"\n{'='*80}")
        print(f"PARLAY-FOCUSED BACKTEST - WEEKS {weeks}")
        print(f"Finding highest hit rate strategies for parlay legs")
        print(f"{'='*80}\n")

        # Generate predictions
        all_predictions = []
        for week in weeks:
            predictions = self._generate_predictions(week)
            all_predictions.extend(predictions)
            print(f"Week {week}: {len(predictions)} predictions")

        print(f"Total: {len(all_predictions)} predictions\n")

        # Test all strategies
        results = self._test_all_strategies(all_predictions)

        # Find best strategies for parlays
        parlay_picks = self._find_parlay_picks(results)

        # Simulate parlays
        parlay_sims = self._simulate_parlays(parlay_picks)

        report = {
            'backtest_info': {
                'season': self.season,
                'weeks': weeks,
                'strategy': 'parlay_focused',
            },
            'strategy_results': results,
            'parlay_picks': parlay_picks,
            'parlay_simulations': parlay_sims,
        }

        return report

    def _generate_predictions(self, week: int) -> List[Dict]:
        predictions = []

        week_games = self.schedules[
            (self.schedules['season'] == self.season) &
            (self.schedules['week'] == week)
        ]

        if len(week_games) == 0:
            return predictions

        teams_playing = set(week_games['home_team'].tolist() + week_games['away_team'].tolist())

        recent_players = self.player_stats[
            (self.player_stats['week'] >= week - 4) &
            (self.player_stats['week'] < week) &
            (self.player_stats['team'].isin(teams_playing))
        ][['player_id', 'player_display_name', 'team', 'position']].drop_duplicates()

        for _, player in recent_players.iterrows():
            player_id = player['player_id']
            position = player.get('position', 'UNK')

            if 'QB' in str(position):
                pos_group = 'QB'
            elif 'RB' in str(position) or 'FB' in str(position):
                pos_group = 'RB'
            elif 'WR' in str(position):
                pos_group = 'WR'
            elif 'TE' in str(position):
                pos_group = 'TE'
            else:
                continue

            for prop_type in self.position_props.get(pos_group, []):
                stat_col = self.prop_map.get(prop_type, prop_type)
                projection = self.get_projection(player_id, week, stat_col)

                if projection <= 0:
                    continue

                actual_row = self.player_stats[
                    (self.player_stats['player_id'] == player_id) &
                    (self.player_stats['week'] == week)
                ]

                if len(actual_row) == 0:
                    continue

                actual = actual_row.iloc[0].get(stat_col, None)
                if actual is None:
                    continue

                predictions.append({
                    'week': week,
                    'player_id': player_id,
                    'player_name': player['player_display_name'],
                    'position': position,
                    'prop_type': prop_type,
                    'projection': projection,
                    'actual': actual,
                })

        return predictions

    def _test_all_strategies(self, predictions: List[Dict]) -> Dict:
        """Test all buffer strategies for each prop type."""
        results = {}

        for prop_type in self.prop_map.keys():
            prop_preds = [p for p in predictions if p['prop_type'] == prop_type]
            if not prop_preds:
                continue

            buffers = self.absolute_buffers.get(prop_type, [5, 10, 15])
            prop_results = []

            for buffer in buffers:
                # Test UNDER (projection + buffer)
                under_hits = 0
                under_total = 0

                # Test OVER (projection - buffer)
                over_hits = 0
                over_total = 0

                for pred in prop_preds:
                    projection = pred['projection']
                    actual = pred['actual']

                    # UNDER: actual < (projection + buffer)
                    under_line = projection + buffer
                    if actual < under_line:
                        under_hits += 1
                    under_total += 1

                    # OVER: actual > (projection - buffer)
                    over_line = projection - buffer
                    if actual > over_line:
                        over_hits += 1
                    over_total += 1

                under_rate = under_hits / under_total * 100 if under_total > 0 else 0
                over_rate = over_hits / over_total * 100 if over_total > 0 else 0

                prop_results.append({
                    'buffer': buffer,
                    'under_hit_rate': round(under_rate, 1),
                    'under_hits': under_hits,
                    'under_total': under_total,
                    'over_hit_rate': round(over_rate, 1),
                    'over_hits': over_hits,
                    'over_total': over_total,
                })

            results[prop_type] = prop_results

        return results

    def _find_parlay_picks(self, results: Dict) -> List[Dict]:
        """Find the best strategies for parlay legs (highest hit rates)."""
        all_strategies = []

        for prop_type, prop_results in results.items():
            for r in prop_results:
                # UNDER strategies
                all_strategies.append({
                    'prop_type': prop_type,
                    'direction': 'UNDER',
                    'buffer': r['buffer'],
                    'hit_rate': r['under_hit_rate'],
                    'sample_size': r['under_total'],
                    'description': f"{prop_type} UNDER (proj + {r['buffer']})"
                })

                # OVER strategies
                all_strategies.append({
                    'prop_type': prop_type,
                    'direction': 'OVER',
                    'buffer': r['buffer'],
                    'hit_rate': r['over_hit_rate'],
                    'sample_size': r['over_total'],
                    'description': f"{prop_type} OVER (proj - {r['buffer']})"
                })

        # Sort by hit rate
        all_strategies.sort(key=lambda x: -x['hit_rate'])

        return all_strategies

    def _simulate_parlays(self, parlay_picks: List[Dict]) -> Dict:
        """Simulate parlay performance with top picks."""
        # Get top strategies (70%+ hit rate)
        top_picks = [p for p in parlay_picks if p['hit_rate'] >= 70]

        simulations = {
            '2_leg': self._calc_parlay_odds(2, top_picks),
            '3_leg': self._calc_parlay_odds(3, top_picks),
            '4_leg': self._calc_parlay_odds(4, top_picks),
            '5_leg': self._calc_parlay_odds(5, top_picks),
        }

        return simulations

    def _calc_parlay_odds(self, legs: int, picks: List[Dict]) -> Dict:
        """Calculate expected parlay performance."""
        if len(picks) < legs:
            return {'available': False}

        # Use top N picks by hit rate
        top_n = picks[:legs]
        avg_hit_rate = np.mean([p['hit_rate'] for p in top_n]) / 100

        # Parlay probability
        parlay_prob = avg_hit_rate ** legs

        # Estimate parlay odds (simplified)
        # Each leg at roughly -150 due to buffers
        single_leg_decimal = 1.67  # -150 American = 1.67 decimal
        parlay_decimal = single_leg_decimal ** legs
        parlay_american = int((parlay_decimal - 1) * 100) if parlay_decimal >= 2 else int(-100 / (parlay_decimal - 1))

        # Expected value
        # EV = (prob * payout) - (1 - prob) * stake
        ev = (parlay_prob * (parlay_decimal - 1)) - (1 - parlay_prob)

        return {
            'legs': legs,
            'avg_leg_hit_rate': round(avg_hit_rate * 100, 1),
            'parlay_hit_rate': round(parlay_prob * 100, 1),
            'approx_odds': parlay_american,
            'expected_value': round(ev * 100, 1),  # As percentage
            'top_picks': [p['description'] for p in top_n],
        }

    def print_report(self, report: Dict):
        """Print formatted report."""
        print(f"\n{'='*80}")
        print("PARLAY-FOCUSED BACKTEST REPORT")
        print(f"{'='*80}")

        # Best strategies by hit rate
        print("\n🎯 TOP 20 HIGHEST HIT RATE STRATEGIES")
        print("-" * 80)
        print(f"{'Strategy':<45} {'Hit Rate':>10} {'Sample':>8}")
        print("-" * 80)

        for i, pick in enumerate(report['parlay_picks'][:20], 1):
            print(f"{i:2}. {pick['description']:<42} {pick['hit_rate']:>9.1f}% {pick['sample_size']:>7}")

        # Strategies by prop type
        print("\n\n📊 BEST STRATEGY PER PROP TYPE")
        print("-" * 80)

        seen_props = set()
        for pick in report['parlay_picks']:
            prop = pick['prop_type']
            if prop not in seen_props:
                seen_props.add(prop)
                status = "✅" if pick['hit_rate'] >= 70 else "⚠️" if pick['hit_rate'] >= 60 else "❌"
                print(f"{status} {pick['description']:<45} {pick['hit_rate']:>6.1f}%")

        # Parlay simulations
        print("\n\n🎰 PARLAY SIMULATIONS (using 70%+ hit rate picks)")
        print("-" * 80)

        for parlay_type, sim in report['parlay_simulations'].items():
            if not sim.get('available', True):
                continue

            print(f"\n{parlay_type.upper().replace('_', ' ')} PARLAY:")
            print(f"  Avg leg hit rate: {sim['avg_leg_hit_rate']}%")
            print(f"  Parlay hit rate: {sim['parlay_hit_rate']}%")
            print(f"  Approx odds: {sim['approx_odds']:+d}")
            print(f"  Expected value: {sim['expected_value']:+.1f}%")
            print(f"  Picks: {', '.join(sim['top_picks'][:3])}...")

        # Recommendations
        print("\n\n💡 PARLAY RECOMMENDATIONS")
        print("-" * 80)

        # Count 70%+ strategies
        high_hit = [p for p in report['parlay_picks'] if p['hit_rate'] >= 70]
        very_high = [p for p in report['parlay_picks'] if p['hit_rate'] >= 75]
        ultra_high = [p for p in report['parlay_picks'] if p['hit_rate'] >= 80]

        print(f"\nAvailable high-confidence picks:")
        print(f"  70%+ hit rate: {len(high_hit)} strategies")
        print(f"  75%+ hit rate: {len(very_high)} strategies")
        print(f"  80%+ hit rate: {len(ultra_high)} strategies")


def main():
    backtester = ParlayBacktester(season=2024)
    report = backtester.run_backtest(weeks=[9, 10])
    backtester.print_report(report)

    output_file = Path("outputs/backtest_parlay_focused.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n📁 Saved to: {output_file}")


if __name__ == "__main__":
    main()
