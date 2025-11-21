"""
Alt-Line Backtest - Conservative betting with higher hit rates.

Strategy: Instead of betting at the "fair" line, use alt lines with buffers
to increase win probability at the cost of worse odds.

Example:
- Projection: 250 passing yards
- Fair line: 250.5 at -110
- Alt OVER 225.5: -150 odds but ~70% hit rate
- Alt OVER 200.5: -200 odds but ~80% hit rate

The goal is to find the buffer size that maximizes ROI after accounting
for the worse odds.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict


class AltLineBacktester:
    """Backtest with alt-line strategies for higher hit rates."""

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

        # Buffer percentages to test
        self.buffer_pcts = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

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
        """Get weighted average projection."""
        player_df = self.player_stats[
            (self.player_stats['player_id'] == player_id) &
            (self.player_stats['week'] < max_week)
        ].sort_values('week', ascending=False)

        if len(player_df) == 0 or stat_col not in player_df.columns:
            return 0

        values = player_df[stat_col].head(5).dropna().tolist()
        if not values:
            return 0

        # Recency weighted
        weights = [0.35, 0.25, 0.20, 0.12, 0.08][:len(values)]
        return sum(v * w for v, w in zip(values, weights)) / sum(weights)

    def calculate_alt_odds(self, buffer_pct: float, direction: str) -> int:
        """Calculate realistic alt-line odds based on buffer size.

        Buffer represents how much we're moving the line in our favor.
        Larger buffer = worse odds but higher probability.

        Approximate odds scaling:
        - 5% buffer: -130
        - 10% buffer: -150
        - 15% buffer: -175
        - 20% buffer: -200
        - 25% buffer: -240
        - 30% buffer: -300
        """
        # Odds get progressively worse with larger buffers
        odds_map = {
            0.05: -130,
            0.10: -150,
            0.15: -180,
            0.20: -220,
            0.25: -280,
            0.30: -350,
        }
        return odds_map.get(buffer_pct, -150)

    def calculate_profit(self, odds: int, won: bool) -> float:
        """Calculate profit/loss for a bet at given odds.

        Assumes $100 base bet.
        - American odds -150 means bet $150 to win $100
        - If you win: profit = $100
        - If you lose: loss = $150
        """
        if odds < 0:
            # Negative odds: bet |odds| to win 100
            risk = abs(odds)
            win_amount = 100
        else:
            # Positive odds: bet 100 to win odds
            risk = 100
            win_amount = odds

        if won:
            return win_amount
        else:
            return -risk

    def generate_alt_line(self, projection: float, prop_type: str,
                         buffer_pct: float, direction: str) -> float:
        """Generate alt line with buffer.

        direction: 'over' or 'under'
        buffer_pct: how much to move line in our favor

        For OVER bets: lower the line (easier to hit)
        For UNDER bets: raise the line (easier to hit)
        """
        buffer = projection * buffer_pct

        if direction == 'over':
            # Lower the line for easier OVER
            alt_line = projection - buffer
        else:
            # Raise the line for easier UNDER
            alt_line = projection + buffer

        # Round to standard increments
        if prop_type in ['passing_yards']:
            return round(alt_line / 5) * 5
        elif prop_type in ['rushing_yards', 'receiving_yards']:
            return round(alt_line / 2.5) * 2.5
        else:
            return round(alt_line * 2) / 2  # Round to 0.5

    def run_backtest(self, weeks: List[int] = [9, 10]) -> Dict:
        """Run alt-line backtest with multiple buffer sizes."""
        print(f"\n{'='*80}")
        print(f"ALT-LINE BACKTEST - WEEKS {weeks}")
        print(f"Testing buffers: {[f'{b*100:.0f}%' for b in self.buffer_pcts]}")
        print(f"{'='*80}\n")

        # Collect all predictions first
        all_predictions = []

        for week in weeks:
            predictions = self._generate_week_predictions(week)
            all_predictions.extend(predictions)
            print(f"Week {week}: {len(predictions)} base predictions")

        print(f"\nTotal base predictions: {len(all_predictions)}")

        # Now test each buffer size
        results_by_buffer = {}

        for buffer_pct in self.buffer_pcts:
            buffer_results = self._evaluate_buffer(all_predictions, buffer_pct)
            results_by_buffer[f'{buffer_pct*100:.0f}%'] = buffer_results

        # Analyze by prop type
        prop_analysis = self._analyze_by_prop(all_predictions)

        report = {
            'backtest_info': {
                'season': self.season,
                'weeks': weeks,
                'timestamp': datetime.now().isoformat(),
                'strategy': 'alt_lines',
                'buffers_tested': [f'{b*100:.0f}%' for b in self.buffer_pcts],
            },
            'results_by_buffer': results_by_buffer,
            'prop_analysis': prop_analysis,
            'recommendations': self._generate_recommendations(results_by_buffer, prop_analysis),
        }

        return report

    def _generate_week_predictions(self, week: int) -> List[Dict]:
        """Generate base predictions for a week."""
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
            team = player.get('team', 'UNK')

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

                # Get actual result
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
                    'team': team,
                    'position': position,
                    'prop_type': prop_type,
                    'projection': round(projection, 1),
                    'actual': actual,
                })

        return predictions

    def _evaluate_buffer(self, predictions: List[Dict], buffer_pct: float) -> Dict:
        """Evaluate performance for a specific buffer size."""
        odds = self.calculate_alt_odds(buffer_pct, 'over')

        over_results = {'wins': 0, 'losses': 0, 'profit': 0}
        under_results = {'wins': 0, 'losses': 0, 'profit': 0}

        for pred in predictions:
            projection = pred['projection']
            actual = pred['actual']
            prop_type = pred['prop_type']

            # OVER bet with lower alt line
            over_line = self.generate_alt_line(projection, prop_type, buffer_pct, 'over')
            over_won = actual > over_line

            if over_won:
                over_results['wins'] += 1
            else:
                over_results['losses'] += 1
            over_results['profit'] += self.calculate_profit(odds, over_won)

            # UNDER bet with higher alt line
            under_line = self.generate_alt_line(projection, prop_type, buffer_pct, 'under')
            under_won = actual < under_line

            if under_won:
                under_results['wins'] += 1
            else:
                under_results['losses'] += 1
            under_results['profit'] += self.calculate_profit(odds, under_won)

        # Calculate hit rates and ROI
        over_total = over_results['wins'] + over_results['losses']
        under_total = under_results['wins'] + under_results['losses']

        over_hit_rate = over_results['wins'] / over_total * 100 if over_total > 0 else 0
        under_hit_rate = under_results['wins'] / under_total * 100 if under_total > 0 else 0

        # ROI calculation: profit / total risked
        risk_per_bet = abs(odds)
        over_roi = over_results['profit'] / (over_total * risk_per_bet) * 100 if over_total > 0 else 0
        under_roi = under_results['profit'] / (under_total * risk_per_bet) * 100 if under_total > 0 else 0

        return {
            'buffer': f'{buffer_pct*100:.0f}%',
            'odds': odds,
            'over': {
                'bets': over_total,
                'wins': over_results['wins'],
                'hit_rate': round(over_hit_rate, 1),
                'profit': round(over_results['profit'], 0),
                'roi': round(over_roi, 1),
            },
            'under': {
                'bets': under_total,
                'wins': under_results['wins'],
                'hit_rate': round(under_hit_rate, 1),
                'profit': round(under_results['profit'], 0),
                'roi': round(under_roi, 1),
            },
            'combined': {
                'bets': over_total + under_total,
                'wins': over_results['wins'] + under_results['wins'],
                'hit_rate': round((over_results['wins'] + under_results['wins']) /
                                 (over_total + under_total) * 100, 1),
                'profit': round(over_results['profit'] + under_results['profit'], 0),
                'roi': round((over_results['profit'] + under_results['profit']) /
                            ((over_total + under_total) * risk_per_bet) * 100, 1),
            }
        }

    def _analyze_by_prop(self, predictions: List[Dict]) -> Dict:
        """Analyze optimal buffer for each prop type."""
        prop_results = defaultdict(lambda: defaultdict(list))

        for pred in predictions:
            prop_type = pred['prop_type']
            projection = pred['projection']
            actual = pred['actual']

            for buffer_pct in self.buffer_pcts:
                # Test OVER
                over_line = self.generate_alt_line(projection, prop_type, buffer_pct, 'over')
                over_won = actual > over_line
                prop_results[prop_type][f'{buffer_pct*100:.0f}%_over'].append(over_won)

                # Test UNDER
                under_line = self.generate_alt_line(projection, prop_type, buffer_pct, 'under')
                under_won = actual < under_line
                prop_results[prop_type][f'{buffer_pct*100:.0f}%_under'].append(under_won)

        # Calculate hit rates and find optimal buffer
        analysis = {}

        for prop_type, buffer_data in prop_results.items():
            prop_analysis = {'buffers': {}}
            best_roi = -999
            best_buffer = None

            for buffer_key, results in buffer_data.items():
                buffer_pct = int(buffer_key.split('%')[0])
                direction = buffer_key.split('_')[1]

                hit_rate = sum(results) / len(results) * 100 if results else 0
                odds = self.calculate_alt_odds(buffer_pct / 100, direction)

                # Calculate ROI
                wins = sum(results)
                losses = len(results) - wins
                profit = (wins * 100) - (losses * abs(odds))
                roi = profit / (len(results) * abs(odds)) * 100 if results else 0

                prop_analysis['buffers'][buffer_key] = {
                    'hit_rate': round(hit_rate, 1),
                    'roi': round(roi, 1),
                    'bets': len(results),
                }

                if roi > best_roi:
                    best_roi = roi
                    best_buffer = buffer_key

            prop_analysis['best_buffer'] = best_buffer
            prop_analysis['best_roi'] = round(best_roi, 1)
            analysis[prop_type] = prop_analysis

        return analysis

    def _generate_recommendations(self, results_by_buffer: Dict, prop_analysis: Dict) -> Dict:
        """Generate betting recommendations."""
        recommendations = {
            'overall_best': None,
            'by_prop_type': {},
            'strategy': '',
        }

        # Find overall best buffer
        best_roi = -999
        best_buffer = None

        for buffer_name, results in results_by_buffer.items():
            if results['combined']['roi'] > best_roi:
                best_roi = results['combined']['roi']
                best_buffer = buffer_name

        recommendations['overall_best'] = {
            'buffer': best_buffer,
            'roi': best_roi,
            'hit_rate': results_by_buffer[best_buffer]['combined']['hit_rate'],
        }

        # Best buffer per prop type
        for prop_type, analysis in prop_analysis.items():
            best = analysis.get('best_buffer', '')
            if best:
                recommendations['by_prop_type'][prop_type] = {
                    'buffer': best,
                    'roi': analysis['best_roi'],
                }

        return recommendations

    def print_report(self, report: Dict):
        """Print formatted report."""
        print(f"\n{'='*80}")
        print("ALT-LINE BACKTEST REPORT")
        print(f"{'='*80}")

        print("\n📊 RESULTS BY BUFFER SIZE")
        print("-" * 80)
        print(f"{'Buffer':<8} {'Odds':>6} {'OVER Hit%':>10} {'OVER ROI':>10} {'UNDER Hit%':>11} {'UNDER ROI':>10}")
        print("-" * 80)

        for buffer_name, results in report['results_by_buffer'].items():
            print(f"{buffer_name:<8} {results['odds']:>6} "
                  f"{results['over']['hit_rate']:>9.1f}% {results['over']['roi']:>+9.1f}% "
                  f"{results['under']['hit_rate']:>10.1f}% {results['under']['roi']:>+9.1f}%")

        print("\n\n🎯 OPTIMAL BUFFER BY PROP TYPE")
        print("-" * 80)

        for prop_type, analysis in sorted(report['prop_analysis'].items(),
                                         key=lambda x: -x[1]['best_roi']):
            best = analysis['best_buffer']
            roi = analysis['best_roi']
            status = "✅" if roi > 0 else "❌"
            print(f"{status} {prop_type:20s}: {best:15s} ROI: {roi:+6.1f}%")

        rec = report['recommendations']
        print(f"\n\n💡 RECOMMENDATIONS")
        print("-" * 80)
        print(f"Overall best buffer: {rec['overall_best']['buffer']} "
              f"({rec['overall_best']['hit_rate']}% hit rate, "
              f"{rec['overall_best']['roi']:+.1f}% ROI)")


def main():
    backtester = AltLineBacktester(season=2024)
    report = backtester.run_backtest(weeks=[9, 10])
    backtester.print_report(report)

    output_file = Path("outputs/backtest_alt_lines.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n📁 Saved to: {output_file}")


if __name__ == "__main__":
    main()
