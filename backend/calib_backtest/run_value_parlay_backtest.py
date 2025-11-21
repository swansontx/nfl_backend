"""
Value-Optimized Parlay Backtest

Finds the sweet spot between:
- High hit rate (70-85%)
- Reasonable odds (-150 to -300, NOT past -400)
- Lines that actually exist on sportsbooks

Key insight: -400 odds with 90% hit rate = 2.5% ROI (barely worth it)
            -200 odds with 75% hit rate = 12.5% ROI (much better value!)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import defaultdict


class ValueParlayBacktester:
    """Find optimal balance between hit rate and odds value."""

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

        # Realistic alt line increments that sportsbooks offer
        self.realistic_alts = {
            'passing_yards': [10, 15, 20, 25, 30],  # e.g., 250.5, 240.5, 235.5...
            'rushing_yards': [5, 7.5, 10, 12.5, 15],
            'receiving_yards': [5, 7.5, 10, 12.5, 15],
            'receptions': [0.5, 1, 1.5, 2],
            'targets': [0.5, 1, 1.5, 2],
            'carries': [1, 2, 3, 4],
            'completions': [1, 2, 3],
            'attempts': [1, 2, 3],
            'passing_tds': [0.5],
            'rushing_tds': [0.5],
            'receiving_tds': [0.5],
            'interceptions': [0.5],
        }

        # Realistic odds for alt lines (based on actual sportsbook patterns)
        # Key: approximate hit rate increase -> odds penalty
        self.odds_model = {
            # (min_hit_increase, max_hit_increase): american_odds
            (0, 5): -120,      # Small edge: -120
            (5, 10): -140,     # Moderate: -140
            (10, 15): -165,    # Good edge: -165
            (15, 20): -200,    # Strong edge: -200
            (20, 25): -250,    # Very strong: -250
            (25, 30): -300,    # Excellent: -300
            (30, 35): -350,    # Elite: -350
            (35, 100): -400,   # Cap at -400 (diminishing returns)
        }

        # Maximum odds we'll consider (user's threshold)
        self.max_odds = -400

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

    def estimate_odds(self, base_hit_rate: float, alt_hit_rate: float) -> int:
        """Estimate realistic odds based on hit rate improvement."""
        improvement = alt_hit_rate - base_hit_rate

        for (min_imp, max_imp), odds in self.odds_model.items():
            if min_imp <= improvement < max_imp:
                return odds

        return -400  # Cap

    def calculate_ev(self, hit_rate: float, odds: int) -> float:
        """Calculate expected value as percentage.

        EV = (P(win) * win_amount) - (P(lose) * risk_amount)
        Normalized to percentage of risk.
        """
        p_win = hit_rate / 100

        if odds < 0:
            # Negative odds: risk |odds| to win 100
            risk = abs(odds)
            win = 100
        else:
            risk = 100
            win = odds

        ev = (p_win * win) - ((1 - p_win) * risk)
        ev_pct = ev / risk * 100

        return ev_pct

    def run_backtest(self, weeks: List[int] = [9, 10]) -> Dict:
        """Run value-optimized backtest."""
        print(f"\n{'='*80}")
        print(f"VALUE-OPTIMIZED PARLAY BACKTEST - WEEKS {weeks}")
        print(f"Finding sweet spot: High hit rate + Reasonable odds (-150 to -300)")
        print(f"Max odds threshold: {self.max_odds}")
        print(f"{'='*80}\n")

        # Generate predictions
        all_predictions = []
        for week in weeks:
            predictions = self._generate_predictions(week)
            all_predictions.extend(predictions)
            print(f"Week {week}: {len(predictions)} predictions")

        print(f"Total: {len(all_predictions)}\n")

        # Calculate base hit rates (at projection)
        base_rates = self._calculate_base_rates(all_predictions)

        # Test value strategies
        value_strategies = self._find_value_strategies(all_predictions, base_rates)

        # Find optimal picks
        optimal_picks = self._rank_by_value(value_strategies)

        # Simulate parlays
        parlay_sims = self._simulate_parlays(optimal_picks)

        report = {
            'backtest_info': {
                'season': self.season,
                'weeks': weeks,
                'strategy': 'value_optimized',
                'max_odds': self.max_odds,
            },
            'base_rates': base_rates,
            'value_strategies': value_strategies,
            'optimal_picks': optimal_picks,
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
                    'player_id': player_id,
                    'player_name': player['player_display_name'],
                    'prop_type': prop_type,
                    'projection': projection,
                    'actual': actual,
                })

        return predictions

    def _calculate_base_rates(self, predictions: List[Dict]) -> Dict:
        """Calculate hit rates at standard lines (projection as line)."""
        base_rates = {}

        for prop_type in self.prop_map.keys():
            prop_preds = [p for p in predictions if p['prop_type'] == prop_type]
            if not prop_preds:
                continue

            over_hits = sum(1 for p in prop_preds if p['actual'] > p['projection'])
            under_hits = sum(1 for p in prop_preds if p['actual'] < p['projection'])
            total = len(prop_preds)

            base_rates[prop_type] = {
                'over': round(over_hits / total * 100, 1) if total > 0 else 50,
                'under': round(under_hits / total * 100, 1) if total > 0 else 50,
                'sample': total,
            }

        return base_rates

    def _find_value_strategies(self, predictions: List[Dict], base_rates: Dict) -> List[Dict]:
        """Find strategies with good value (hit rate vs odds balance)."""
        strategies = []

        for prop_type in self.prop_map.keys():
            prop_preds = [p for p in predictions if p['prop_type'] == prop_type]
            if not prop_preds:
                continue

            buffers = self.realistic_alts.get(prop_type, [5, 10])
            base_over = base_rates.get(prop_type, {}).get('over', 50)
            base_under = base_rates.get(prop_type, {}).get('under', 50)

            for buffer in buffers:
                # OVER with lower line
                over_hits = sum(1 for p in prop_preds if p['actual'] > (p['projection'] - buffer))
                over_rate = over_hits / len(prop_preds) * 100

                # UNDER with higher line
                under_hits = sum(1 for p in prop_preds if p['actual'] < (p['projection'] + buffer))
                under_rate = under_hits / len(prop_preds) * 100

                # Estimate odds based on improvement
                over_odds = self.estimate_odds(base_over, over_rate)
                under_odds = self.estimate_odds(base_under, under_rate)

                # Calculate EV
                over_ev = self.calculate_ev(over_rate, over_odds)
                under_ev = self.calculate_ev(under_rate, under_odds)

                # Only include if odds are within threshold
                if over_odds >= self.max_odds:
                    strategies.append({
                        'prop_type': prop_type,
                        'direction': 'OVER',
                        'buffer': buffer,
                        'hit_rate': round(over_rate, 1),
                        'odds': over_odds,
                        'ev': round(over_ev, 1),
                        'description': f"{prop_type} OVER (proj - {buffer})",
                        'sample': len(prop_preds),
                    })

                if under_odds >= self.max_odds:
                    strategies.append({
                        'prop_type': prop_type,
                        'direction': 'UNDER',
                        'buffer': buffer,
                        'hit_rate': round(under_rate, 1),
                        'odds': under_odds,
                        'ev': round(under_ev, 1),
                        'description': f"{prop_type} UNDER (proj + {buffer})",
                        'sample': len(prop_preds),
                    })

        return strategies

    def _rank_by_value(self, strategies: List[Dict]) -> List[Dict]:
        """Rank strategies by value (EV weighted by hit rate)."""
        # Filter: only positive EV and reasonable hit rate (65%+)
        good_strategies = [
            s for s in strategies
            if s['ev'] > 0 and s['hit_rate'] >= 65 and s['odds'] >= self.max_odds
        ]

        # Sort by EV (best value first)
        good_strategies.sort(key=lambda x: -x['ev'])

        return good_strategies

    def _simulate_parlays(self, picks: List[Dict]) -> Dict:
        """Simulate parlays with value-optimized picks."""
        if len(picks) < 2:
            return {'error': 'Not enough positive EV picks'}

        simulations = {}

        for n_legs in [2, 3, 4]:
            if len(picks) < n_legs:
                continue

            top_picks = picks[:n_legs]

            # Combined probability
            combined_prob = 1
            for pick in top_picks:
                combined_prob *= (pick['hit_rate'] / 100)

            # Combined odds (multiply decimal odds)
            combined_decimal = 1
            for pick in top_picks:
                if pick['odds'] < 0:
                    decimal = 1 + (100 / abs(pick['odds']))
                else:
                    decimal = 1 + (pick['odds'] / 100)
                combined_decimal *= decimal

            # Convert back to American
            if combined_decimal >= 2:
                parlay_odds = int((combined_decimal - 1) * 100)
            else:
                parlay_odds = int(-100 / (combined_decimal - 1))

            # Expected value
            ev = self.calculate_ev(combined_prob * 100, parlay_odds)

            simulations[f'{n_legs}_leg'] = {
                'legs': n_legs,
                'picks': [p['description'] for p in top_picks],
                'avg_hit_rate': round(np.mean([p['hit_rate'] for p in top_picks]), 1),
                'avg_odds': round(np.mean([p['odds'] for p in top_picks])),
                'parlay_prob': round(combined_prob * 100, 1),
                'parlay_odds': parlay_odds,
                'ev': round(ev, 1),
            }

        return simulations

    def print_report(self, report: Dict):
        """Print formatted report."""
        print(f"\n{'='*80}")
        print("VALUE-OPTIMIZED PARLAY REPORT")
        print(f"{'='*80}")

        # Top value picks
        print("\n🎯 TOP VALUE PICKS (Positive EV, 65%+ Hit Rate, Odds ≥ -400)")
        print("-" * 80)
        print(f"{'Strategy':<40} {'Hit%':>7} {'Odds':>7} {'EV':>7}")
        print("-" * 80)

        for i, pick in enumerate(report['optimal_picks'][:15], 1):
            print(f"{i:2}. {pick['description']:<37} {pick['hit_rate']:>6.1f}% {pick['odds']:>+6} {pick['ev']:>+6.1f}%")

        # Value zones
        print("\n\n📊 VALUE ZONE ANALYSIS")
        print("-" * 80)

        zones = {
            'Sweet Spot (-150 to -200)': [],
            'Good Value (-200 to -250)': [],
            'High Confidence (-250 to -300)': [],
            'Max Confidence (-300 to -400)': [],
        }

        for pick in report['optimal_picks']:
            odds = pick['odds']
            if -200 <= odds < -150:
                zones['Sweet Spot (-150 to -200)'].append(pick)
            elif -250 <= odds < -200:
                zones['Good Value (-200 to -250)'].append(pick)
            elif -300 <= odds < -250:
                zones['High Confidence (-250 to -300)'].append(pick)
            elif -400 <= odds < -300:
                zones['Max Confidence (-300 to -400)'].append(pick)

        for zone, picks in zones.items():
            if picks:
                avg_hit = np.mean([p['hit_rate'] for p in picks])
                avg_ev = np.mean([p['ev'] for p in picks])
                print(f"\n{zone}:")
                print(f"  {len(picks)} strategies | Avg hit: {avg_hit:.1f}% | Avg EV: {avg_ev:+.1f}%")
                # Show top 3
                for p in sorted(picks, key=lambda x: -x['ev'])[:3]:
                    print(f"    • {p['description']}: {p['hit_rate']}% at {p['odds']}")

        # Parlay simulations
        print("\n\n🎰 PARLAY SIMULATIONS")
        print("-" * 80)

        for name, sim in report['parlay_simulations'].items():
            if 'error' in sim:
                continue

            print(f"\n{name.upper().replace('_', ' ')}:")
            print(f"  Avg leg: {sim['avg_hit_rate']}% at {sim['avg_odds']}")
            print(f"  Parlay: {sim['parlay_prob']}% to hit at {sim['parlay_odds']:+}")
            print(f"  EV: {sim['ev']:+.1f}%")
            print(f"  Picks: {', '.join(sim['picks'])}")


def main():
    backtester = ValueParlayBacktester(season=2024)
    report = backtester.run_backtest(weeks=[9, 10])
    backtester.print_report(report)

    output_file = Path("outputs/backtest_value_parlay.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n📁 Saved to: {output_file}")


if __name__ == "__main__":
    main()
