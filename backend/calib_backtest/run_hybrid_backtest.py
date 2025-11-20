"""
Hybrid backtest - selective adjustments based on what works best for each prop type.

Strategy:
- TD props: Minimal adjustments (already 80%+ hit rate)
- Passing yards: Apply opponent + context adjustments (improved +6%)
- Passing TDs: Apply adjustments (improved +7.8%)
- Completions: Use matchup-based model (currently 45%)
- Rushing yards: Use Vegas-implied approach (currently 49%)
- Volume props (targets, receptions): Light adjustments
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class HybridBacktester:
    """Hybrid model using best approach for each prop type."""

    def __init__(self, season: int = 2024, inputs_dir: str = "inputs"):
        self.season = season
        self.inputs_dir = Path(inputs_dir)

        # Load data
        self.player_stats = self._load_player_stats()
        self.schedules = self._load_schedules()

        # Calculate opponent rankings
        self.def_rankings = self._calculate_defensive_rankings()

        # Prop mappings
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

        # Define which props get which treatment
        self.adjustment_strategy = {
            # Props that improved with adjustments - use them
            'passing_yards': {'opponent': True, 'context': True, 'weight': 'recency'},
            'passing_tds': {'opponent': True, 'context': True, 'weight': 'recency'},
            'receiving_yards': {'opponent': True, 'context': False, 'weight': 'recency'},
            'receptions': {'opponent': True, 'context': False, 'weight': 'recency'},

            # Props that got worse - use basic approach
            'rushing_tds': {'opponent': False, 'context': False, 'weight': 'simple'},
            'receiving_tds': {'opponent': False, 'context': False, 'weight': 'simple'},
            'interceptions': {'opponent': False, 'context': False, 'weight': 'simple'},
            'attempts': {'opponent': False, 'context': False, 'weight': 'simple'},
            'carries': {'opponent': False, 'context': False, 'weight': 'simple'},
            'targets': {'opponent': False, 'context': False, 'weight': 'recency'},

            # Props that need special handling
            'completions': {'opponent': True, 'context': True, 'weight': 'matchup'},
            'rushing_yards': {'opponent': True, 'context': True, 'weight': 'vegas_implied'},
        }

    def _load_player_stats(self) -> pd.DataFrame:
        stats_file = self.inputs_dir / "player_stats_2024_2025.csv"
        if stats_file.exists():
            df = pd.read_csv(stats_file, low_memory=False)
            print(f"Loaded {len(df)} player stat records")
            return df
        return pd.DataFrame()

    def _load_schedules(self) -> pd.DataFrame:
        sched_file = self.inputs_dir / "schedules_2024_2025.csv"
        if sched_file.exists():
            return pd.read_csv(sched_file)
        return pd.DataFrame()

    def _calculate_defensive_rankings(self) -> Dict:
        """Calculate team defensive rankings."""
        rankings = {}
        if len(self.player_stats) == 0:
            return rankings

        opp_stats = self.player_stats.groupby('opponent_team').agg({
            'passing_yards': 'mean',
            'passing_tds': 'mean',
            'rushing_yards': 'mean',
            'completions': 'mean',
            'receptions': 'mean',
            'receiving_yards': 'mean',
        }).reset_index()

        league_avg = {col: opp_stats[col].mean() for col in opp_stats.columns if col != 'opponent_team'}

        for _, row in opp_stats.iterrows():
            team = row['opponent_team']
            rankings[team] = {}
            for stat, avg in league_avg.items():
                rankings[team][stat] = row[stat] / avg if avg > 0 else 1.0

        return rankings

    def get_player_features(self, player_id: str, max_week: int, weight_type: str) -> Dict:
        """Get features with specified weighting strategy."""
        player_df = self.player_stats[
            (self.player_stats['player_id'] == player_id) &
            (self.player_stats['week'] < max_week)
        ].sort_values('week', ascending=False)

        if len(player_df) == 0:
            return {}

        features = {}
        stats = list(self.prop_map.values())

        for stat in stats:
            if stat not in player_df.columns:
                continue

            values = player_df[stat].head(5).dropna().tolist()
            if not values:
                features[f'{stat}_proj'] = 0
                continue

            if weight_type == 'simple':
                # Simple average of last 3-4 games
                features[f'{stat}_proj'] = np.mean(values[:4])

            elif weight_type == 'recency':
                # Recency weighted
                weights = [0.35, 0.25, 0.20, 0.12, 0.08][:len(values)]
                weighted = sum(v * w for v, w in zip(values, weights)) / sum(weights)
                features[f'{stat}_proj'] = weighted

            elif weight_type == 'matchup':
                # Heavily weighted on recent + variance consideration
                recent_avg = np.mean(values[:3]) if len(values) >= 3 else np.mean(values)
                features[f'{stat}_proj'] = recent_avg

            elif weight_type == 'vegas_implied':
                # Use season average more heavily (Vegas tends to regress)
                season_avg = player_df[stat].mean()
                recent_avg = np.mean(values[:3]) if len(values) >= 3 else np.mean(values)
                features[f'{stat}_proj'] = 0.6 * season_avg + 0.4 * recent_avg

            else:
                features[f'{stat}_proj'] = np.mean(values)

            # Variance for confidence
            features[f'{stat}_var'] = np.std(values) if len(values) >= 3 else features[f'{stat}_proj'] * 0.3

        features['games_played'] = len(player_df)
        features['player_name'] = player_df.iloc[0].get('player_display_name', '')
        features['position'] = player_df.iloc[0].get('position', 'UNK')
        features['team'] = player_df.iloc[0].get('team', 'UNK')

        return features

    def generate_projection(self, player_id: str, prop_type: str, features: Dict,
                           opponent: str, spread: float, total: float, is_home: bool) -> Optional[float]:
        """Generate projection using prop-specific strategy."""
        strategy = self.adjustment_strategy.get(prop_type, {})
        stat_col = self.prop_map.get(prop_type, prop_type)

        base_proj = features.get(f'{stat_col}_proj', 0)
        if base_proj <= 0:
            return None

        projection = base_proj

        # Apply opponent adjustment if enabled
        if strategy.get('opponent', False):
            if opponent in self.def_rankings:
                if stat_col in self.def_rankings[opponent]:
                    mult = self.def_rankings[opponent][stat_col]
                    # Dampen the effect
                    mult = 0.5 + 0.5 * mult  # Range: 0.5 to 1.5 becomes 0.75 to 1.25
                    projection *= mult

        # Apply context adjustment if enabled
        if strategy.get('context', False):
            avg_total = 45.0
            total_factor = total / avg_total if total > 0 else 1.0

            # Different adjustments by prop type
            if prop_type in ['passing_yards', 'completions']:
                # Underdogs pass more
                if (is_home and spread > 0) or (not is_home and spread < 0):
                    projection *= 1.03  # 3% boost for underdogs
                # High total games = more volume
                projection *= (0.8 + 0.2 * total_factor)

            elif prop_type in ['passing_tds']:
                # TDs heavily correlated with total
                projection *= (0.6 + 0.4 * total_factor)

            elif prop_type in ['rushing_yards']:
                # Favorites run more
                if (is_home and spread < -3) or (not is_home and spread > 3):
                    projection *= 1.05

        return round(projection, 1)

    def generate_line(self, projection: float, prop_type: str) -> float:
        """Generate betting line."""
        noise = np.random.uniform(-projection * 0.02, projection * 0.02)

        if prop_type in ['passing_yards']:
            return round((projection + noise) / 5) * 5
        elif prop_type in ['rushing_yards', 'receiving_yards']:
            return round((projection + noise) / 2.5) * 2.5
        else:
            return round(projection + noise, 1)

    def run_backtest(self, weeks: List[int] = [9, 10]) -> Dict:
        """Run hybrid backtest."""
        print(f"\n{'='*80}")
        print(f"HYBRID BACKTEST - WEEKS {weeks}")
        print(f"Using prop-specific strategies for optimal results")
        print(f"{'='*80}\n")

        all_results = []

        for week in weeks:
            print(f"--- Week {week} ---")
            results = self._backtest_week(week)
            all_results.extend(results)
            print(f"Generated {len(results)} predictions")

        analysis = self._analyze_results(all_results)

        report = {
            'backtest_info': {
                'season': self.season,
                'weeks': weeks,
                'timestamp': datetime.now().isoformat(),
                'model_type': 'hybrid',
            },
            'summary': analysis['summary'],
            'by_prop_type': analysis['by_prop_type'],
            'by_position': analysis['by_position'],
            'betting_simulation': analysis['betting'],
            'all_predictions': all_results
        }

        return report

    def _backtest_week(self, week: int) -> List[Dict]:
        """Backtest single week."""
        results = []

        week_games = self.schedules[
            (self.schedules['season'] == self.season) &
            (self.schedules['week'] == week)
        ]

        if len(week_games) == 0:
            return results

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

            game = week_games[
                (week_games['home_team'] == team) |
                (week_games['away_team'] == team)
            ]
            if len(game) == 0:
                continue

            game = game.iloc[0]
            opponent = game['away_team'] if game['home_team'] == team else game['home_team']
            is_home = 1 if game['home_team'] == team else 0
            spread = game.get('spread_line', 0)
            total = game.get('total_line', 45)

            for prop_type in self.position_props.get(pos_group, []):
                # Get weight type for this prop
                strategy = self.adjustment_strategy.get(prop_type, {})
                weight_type = strategy.get('weight', 'simple')

                features = self.get_player_features(player_id, week, weight_type)
                if not features:
                    continue

                projection = self.generate_projection(
                    player_id, prop_type, features,
                    opponent, spread, total, is_home
                )

                if projection is None:
                    continue

                # Get actual
                stat_col = self.prop_map.get(prop_type, prop_type)
                actual_row = self.player_stats[
                    (self.player_stats['player_id'] == player_id) &
                    (self.player_stats['week'] == week)
                ]

                if len(actual_row) == 0:
                    continue

                actual = actual_row.iloc[0].get(stat_col, None)
                if actual is None:
                    continue

                line = self.generate_line(projection, prop_type)

                error = projection - actual
                edge = projection - line

                over_hit = actual > line
                under_hit = actual < line
                push = actual == line

                recommend_over = projection > line
                correct = (recommend_over and over_hit) or (not recommend_over and under_hit)

                results.append({
                    'week': week,
                    'player_id': player_id,
                    'player_name': features.get('player_name', ''),
                    'team': team,
                    'opponent': opponent,
                    'position': position,
                    'prop_type': prop_type,
                    'projection': projection,
                    'line': line,
                    'actual': actual,
                    'error': round(error, 1),
                    'abs_error': round(abs(error), 1),
                    'edge': round(edge, 1),
                    'recommend_over': recommend_over,
                    'correct': correct,
                    'push': push,
                })

        return results

    def _analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze results."""
        if not results:
            return {'summary': {}, 'by_prop_type': {}, 'by_position': {}, 'betting': {}}

        df = pd.DataFrame(results)

        total = len(df)
        correct = df['correct'].sum()
        pushes = df['push'].sum()

        summary = {
            'total_predictions': total,
            'correct': int(correct),
            'hit_rate': round(correct / (total - pushes) * 100 if (total - pushes) > 0 else 0, 1),
            'mae': round(df['abs_error'].mean(), 2),
        }

        by_prop_type = {}
        for pt in df['prop_type'].unique():
            pt_df = df[df['prop_type'] == pt]
            pt_correct = pt_df['correct'].sum()
            pt_total = len(pt_df) - pt_df['push'].sum()
            by_prop_type[pt] = {
                'predictions': len(pt_df),
                'correct': int(pt_correct),
                'hit_rate': round(pt_correct / pt_total * 100 if pt_total > 0 else 0, 1),
                'mae': round(pt_df['abs_error'].mean(), 2),
            }

        by_position = {}
        for pos in df['position'].unique():
            pos_df = df[df['position'] == pos]
            pos_correct = pos_df['correct'].sum()
            pos_total = len(pos_df) - pos_df['push'].sum()
            by_position[pos] = {
                'hit_rate': round(pos_correct / pos_total * 100 if pos_total > 0 else 0, 1),
            }

        betting = {}
        for min_edge in [0, 1, 2, 3, 5]:
            bets = df[df['edge'].abs() >= min_edge]
            if len(bets) == 0:
                continue
            wins = bets['correct'].sum()
            losses = len(bets) - wins - bets['push'].sum()
            profit = (wins * 100) - (losses * 110)
            total_risked = (wins + losses) * 110
            roi = (profit / total_risked * 100) if total_risked > 0 else 0
            betting[f'edge_{min_edge}+'] = {
                'bets': int(wins + losses),
                'wins': int(wins),
                'losses': int(losses),
                'profit': round(profit, 2),
                'roi': round(roi, 2)
            }

        return {
            'summary': summary,
            'by_prop_type': by_prop_type,
            'by_position': by_position,
            'betting': betting
        }

    def print_report(self, report: Dict):
        """Print report."""
        print(f"\n{'='*80}")
        print(f"HYBRID BACKTEST REPORT")
        print(f"{'='*80}")

        s = report['summary']
        print(f"\n📊 OVERALL: {s['correct']}/{s['total_predictions']} ({s['hit_rate']}%)")

        print(f"\n📈 BY PROP TYPE")
        for pt, m in sorted(report['by_prop_type'].items(), key=lambda x: -x[1]['hit_rate']):
            status = "✅" if m['hit_rate'] >= 55 else "⚠️" if m['hit_rate'] >= 50 else "❌"
            print(f"   {status} {pt:20s}: {m['hit_rate']:5.1f}% ({m['correct']}/{m['predictions']}) MAE:{m['mae']:.1f}")

        print(f"\n💰 BETTING ROI")
        for strat, m in report['betting_simulation'].items():
            status = "✅" if m['profit'] > 0 else "❌"
            print(f"   {status} {strat}: {m['wins']}W-{m['losses']}L | ${m['profit']:+.0f} | ROI: {m['roi']:+.1f}%")


def main():
    backtester = HybridBacktester(season=2024)
    report = backtester.run_backtest(weeks=[9, 10])
    backtester.print_report(report)

    output_file = Path("outputs/backtest_hybrid_weeks_9_10.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n📁 Saved to: {output_file}")


if __name__ == "__main__":
    main()
