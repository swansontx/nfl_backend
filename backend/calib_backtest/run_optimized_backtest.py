"""
OPTIMIZED BACKTEST - Uses best-performing approach for each prop type.

Based on comparative analysis:
- Basic model: TD props, interceptions, targets, attempts, carries
- Improved model: Passing TDs
- Hybrid model: Yards props, receptions, rushing TDs
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class OptimizedBacktester:
    """Uses the optimal strategy for each prop type based on testing."""

    def __init__(self, season: int = 2024, inputs_dir: str = "inputs"):
        self.season = season
        self.inputs_dir = Path(inputs_dir)

        self.player_stats = self._load_player_stats()
        self.schedules = self._load_schedules()
        self.def_rankings = self._calculate_defensive_rankings()

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

        # OPTIMIZED: Best model for each prop based on testing
        self.optimal_model = {
            'rushing_tds': 'hybrid',      # 81.5%
            'receiving_tds': 'basic',     # 79.2%
            'interceptions': 'basic',     # 63.5%
            'passing_yards': 'hybrid',    # 58.1%
            'attempts': 'basic',          # 57.9%
            'targets': 'basic',           # 56.6%
            'receptions': 'hybrid',       # 55.1%
            'carries': 'basic',           # 54.4%
            'rushing_yards': 'hybrid',    # 52.3%
            'receiving_yards': 'hybrid',  # 52.2%
            'passing_tds': 'improved',    # 51.8%
            'completions': 'hybrid',      # 47.9%
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

    def _calculate_defensive_rankings(self) -> Dict:
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
            rankings[team] = {stat: row[stat] / avg if avg > 0 else 1.0 for stat, avg in league_avg.items()}

        return rankings

    def get_basic_projection(self, player_id: str, max_week: int, stat_col: str) -> float:
        """Basic: Simple L3-4 average."""
        player_df = self.player_stats[
            (self.player_stats['player_id'] == player_id) &
            (self.player_stats['week'] < max_week)
        ].sort_values('week', ascending=False)

        if len(player_df) == 0 or stat_col not in player_df.columns:
            return 0

        values = player_df[stat_col].head(4).dropna().tolist()
        return np.mean(values) if values else 0

    def get_hybrid_projection(self, player_id: str, max_week: int, stat_col: str,
                              opponent: str, spread: float, total: float, is_home: bool) -> float:
        """Hybrid: Recency weighted + opponent + context adjustments."""
        player_df = self.player_stats[
            (self.player_stats['player_id'] == player_id) &
            (self.player_stats['week'] < max_week)
        ].sort_values('week', ascending=False)

        if len(player_df) == 0 or stat_col not in player_df.columns:
            return 0

        values = player_df[stat_col].head(5).dropna().tolist()
        if not values:
            return 0

        # For rushing_yards and passing_yards, use Vegas-implied (more season average)
        if stat_col in ['rushing_yards', 'passing_yards']:
            season_avg = player_df[stat_col].mean()
            recent_avg = np.mean(values[:3]) if len(values) >= 3 else np.mean(values)
            base_proj = 0.6 * season_avg + 0.4 * recent_avg
        else:
            # Recency weighted
            weights = [0.35, 0.25, 0.20, 0.12, 0.08][:len(values)]
            base_proj = sum(v * w for v, w in zip(values, weights)) / sum(weights)

        # Opponent adjustment
        if opponent in self.def_rankings and stat_col in self.def_rankings[opponent]:
            mult = self.def_rankings[opponent][stat_col]
            mult = 0.5 + 0.5 * mult  # Dampen
            base_proj *= mult

        # Context adjustment for yards/TDs
        avg_total = 45.0
        if stat_col in ['passing_yards', 'rushing_yards', 'receiving_yards']:
            total_factor = total / avg_total if total > 0 else 1.0
            base_proj *= (0.8 + 0.2 * total_factor)

            # Game script
            if stat_col == 'rushing_yards':
                if (is_home and spread < -3) or (not is_home and spread > 3):
                    base_proj *= 1.05  # Favorites run more

        return base_proj

    def get_improved_projection(self, player_id: str, max_week: int, stat_col: str,
                                opponent: str, spread: float, total: float, is_home: bool) -> float:
        """Improved: Recency weighted + full opponent + context."""
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
        base_proj = sum(v * w for v, w in zip(values, weights)) / sum(weights)

        # Opponent adjustment
        if opponent in self.def_rankings and stat_col in self.def_rankings[opponent]:
            mult = self.def_rankings[opponent][stat_col]
            mult = max(0.85, min(1.15, mult))
            base_proj *= mult

        # Context adjustment
        avg_total = 45.0
        total_factor = total / avg_total if total > 0 else 1.0

        if stat_col == 'passing_tds':
            base_proj *= (0.6 + 0.4 * total_factor)

        return base_proj

    def generate_projection(self, player_id: str, prop_type: str,
                           opponent: str, spread: float, total: float, is_home: bool) -> Optional[float]:
        """Generate projection using optimal model for this prop type."""
        stat_col = self.prop_map.get(prop_type, prop_type)
        model_type = self.optimal_model.get(prop_type, 'basic')

        # Get max_week (this is set in the backtest loop)
        max_week = getattr(self, '_current_week', 11)

        if model_type == 'basic':
            proj = self.get_basic_projection(player_id, max_week, stat_col)
        elif model_type == 'hybrid':
            proj = self.get_hybrid_projection(player_id, max_week, stat_col, opponent, spread, total, is_home)
        elif model_type == 'improved':
            proj = self.get_improved_projection(player_id, max_week, stat_col, opponent, spread, total, is_home)
        else:
            proj = self.get_basic_projection(player_id, max_week, stat_col)

        return round(proj, 1) if proj > 0 else None

    def generate_line(self, projection: float, prop_type: str) -> float:
        noise = np.random.uniform(-projection * 0.02, projection * 0.02)
        if prop_type in ['passing_yards']:
            return round((projection + noise) / 5) * 5
        elif prop_type in ['rushing_yards', 'receiving_yards']:
            return round((projection + noise) / 2.5) * 2.5
        return round(projection + noise, 1)

    def run_backtest(self, weeks: List[int] = [9, 10]) -> Dict:
        print(f"\n{'='*80}")
        print(f"OPTIMIZED BACKTEST - WEEKS {weeks}")
        print(f"Using best model for each prop type")
        print(f"{'='*80}\n")

        all_results = []

        for week in weeks:
            self._current_week = week
            print(f"--- Week {week} ---")
            results = self._backtest_week(week)
            all_results.extend(results)
            print(f"Generated {len(results)} predictions")

        analysis = self._analyze_results(all_results)

        return {
            'backtest_info': {
                'season': self.season,
                'weeks': weeks,
                'timestamp': datetime.now().isoformat(),
                'model_type': 'optimized',
            },
            'summary': analysis['summary'],
            'by_prop_type': analysis['by_prop_type'],
            'by_position': analysis['by_position'],
            'betting_simulation': analysis['betting'],
            'all_predictions': all_results
        }

    def _backtest_week(self, week: int) -> List[Dict]:
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
                projection = self.generate_projection(
                    player_id, prop_type, opponent, spread, total, is_home
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
                    'player_name': player['player_display_name'],
                    'team': team,
                    'opponent': opponent,
                    'position': position,
                    'prop_type': prop_type,
                    'model_used': self.optimal_model.get(prop_type, 'basic'),
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
        print(f"\n{'='*80}")
        print(f"OPTIMIZED BACKTEST REPORT")
        print(f"{'='*80}")

        s = report['summary']
        print(f"\n📊 OVERALL: {s['correct']}/{s['total_predictions']} ({s['hit_rate']}%) MAE: {s['mae']}")

        print(f"\n📈 BY PROP TYPE (sorted by hit rate)")
        for pt, m in sorted(report['by_prop_type'].items(), key=lambda x: -x[1]['hit_rate']):
            status = "✅" if m['hit_rate'] >= 55 else "⚠️" if m['hit_rate'] >= 50 else "❌"
            print(f"   {status} {pt:20s}: {m['hit_rate']:5.1f}% ({m['correct']}/{m['predictions']})")

        print(f"\n💰 BETTING ROI")
        for strat, m in report['betting_simulation'].items():
            status = "✅" if m['profit'] > 0 else "❌"
            print(f"   {status} {strat}: {m['wins']}W-{m['losses']}L | ${m['profit']:+.0f} | ROI: {m['roi']:+.1f}%")


def main():
    backtester = OptimizedBacktester(season=2024)
    report = backtester.run_backtest(weeks=[9, 10])
    backtester.print_report(report)

    output_file = Path("outputs/backtest_optimized_weeks_9_10.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n📁 Saved to: {output_file}")


if __name__ == "__main__":
    main()
