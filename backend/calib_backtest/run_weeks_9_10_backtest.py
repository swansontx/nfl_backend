"""
Comprehensive backtest for weeks 9 and 10 of 2024 NFL season.

This script:
1. Uses weeks 1-8 data to generate projections for weeks 9-10
2. Compares projections to actual results
3. Calculates hit rates, edge accuracy, and simulated ROI
4. Generates detailed analysis report

Usage:
    python -m backend.calib_backtest.run_weeks_9_10_backtest
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import joblib


class WeeksBacktester:
    """Backtester for evaluating prop projections against actual results."""

    def __init__(self, season: int = 2024, inputs_dir: str = "inputs", models_dir: str = "outputs/models"):
        self.season = season
        self.inputs_dir = Path(inputs_dir)
        self.models_dir = Path(models_dir)

        # Load data
        self.player_stats = self._load_player_stats()
        self.schedules = self._load_schedules()
        self.models = self._load_models()

        # Calculate defensive rankings for opponent adjustments
        self.def_rankings = self._calculate_defensive_rankings()

        # Prop type mappings
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

        # Position-specific props
        self.position_props = {
            'QB': ['passing_yards', 'passing_tds', 'completions', 'attempts', 'interceptions'],
            'RB': ['rushing_yards', 'rushing_tds', 'carries', 'receptions', 'receiving_yards'],
            'WR': ['receptions', 'receiving_yards', 'receiving_tds', 'targets'],
            'TE': ['receptions', 'receiving_yards', 'receiving_tds', 'targets'],
        }

    def _load_player_stats(self) -> pd.DataFrame:
        """Load player stats data."""
        stats_file = self.inputs_dir / "player_stats_2024_2025.csv"
        if stats_file.exists():
            df = pd.read_csv(stats_file, low_memory=False)
            print(f"Loaded {len(df)} player stat records")
            return df
        else:
            print(f"Warning: Stats file not found: {stats_file}")
            return pd.DataFrame()

    def _load_schedules(self) -> pd.DataFrame:
        """Load schedule data."""
        sched_file = self.inputs_dir / "schedules_2024_2025.csv"
        if sched_file.exists():
            return pd.read_csv(sched_file)
        return pd.DataFrame()

    def _load_models(self) -> Dict:
        """Load trained models."""
        models = {}
        if self.models_dir.exists():
            for model_file in self.models_dir.rglob("*.pkl"):
                try:
                    model_name = model_file.stem
                    models[model_name] = joblib.load(model_file)
                except Exception as e:
                    pass  # Skip models that can't be loaded
        print(f"Loaded {len(models)} models")
        return models

    def _calculate_defensive_rankings(self) -> Dict:
        """Calculate team defensive rankings based on what they ALLOW to opponents."""
        rankings = {}

        if len(self.schedules) == 0:
            return {}

        max_week = int(self.player_stats['week'].max())
        min_week = max(1, max_week - 5)

        # Track what each team ALLOWS
        team_allowed = {}

        for _, game in self.schedules[
            (self.schedules['week'] >= min_week) &
            (self.schedules['week'] <= max_week) &
            (self.schedules['game_type'] == 'REG')
        ].iterrows():
            home = game['home_team']
            away = game['away_team']
            week = game['week']

            # Away team's production = what home team allowed
            away_stats = self.player_stats[
                (self.player_stats['team'] == away) & (self.player_stats['week'] == week)
            ]
            # Home team's production = what away team allowed
            home_stats = self.player_stats[
                (self.player_stats['team'] == home) & (self.player_stats['week'] == week)
            ]

            # Home team allowed away team's stats
            if home not in team_allowed:
                team_allowed[home] = {'games': 0, 'rush': 0, 'rec': 0, 'pass': 0}
            team_allowed[home]['games'] += 1
            team_allowed[home]['rush'] += away_stats['rushing_yards'].sum()
            team_allowed[home]['rec'] += away_stats['receiving_yards'].sum()
            team_allowed[home]['pass'] += away_stats['passing_yards'].sum()

            # Away team allowed home team's stats
            if away not in team_allowed:
                team_allowed[away] = {'games': 0, 'rush': 0, 'rec': 0, 'pass': 0}
            team_allowed[away]['games'] += 1
            team_allowed[away]['rush'] += home_stats['rushing_yards'].sum()
            team_allowed[away]['rec'] += home_stats['receiving_yards'].sum()
            team_allowed[away]['pass'] += home_stats['passing_yards'].sum()

        # Calculate per-game averages
        team_avgs = []
        for team, allowed in team_allowed.items():
            if allowed['games'] > 0:
                team_avgs.append({
                    'team': team,
                    'rush_avg': allowed['rush'] / allowed['games'],
                    'rec_avg': allowed['rec'] / allowed['games'],
                    'pass_avg': allowed['pass'] / allowed['games'],
                })

        if not team_avgs:
            return {}

        df = pd.DataFrame(team_avgs)
        league_rush_avg = df['rush_avg'].mean()
        league_rec_avg = df['rec_avg'].mean()
        league_pass_avg = df['pass_avg'].mean()

        for _, row in df.iterrows():
            rankings[row['team']] = {
                'avg': {
                    'rush': row['rush_avg'],
                    'rec': row['rec_avg'],
                    'pass': row['pass_avg'],
                },
                'league_avg': {
                    'rush': league_rush_avg,
                    'rec': league_rec_avg,
                    'pass': league_pass_avg,
                }
            }

        return rankings

    def get_opponent_multiplier(self, opponent: str, stat_col: str) -> float:
        """Get opponent adjustment multiplier based on defensive quality."""
        if opponent not in self.def_rankings:
            return 1.0

        opp_data = self.def_rankings[opponent]
        if 'avg' not in opp_data or 'league_avg' not in opp_data:
            return 1.0

        # Map stat to defensive category
        if stat_col in ['rushing_yards', 'carries', 'rushing_tds']:
            cat = 'rush'
        elif stat_col in ['receiving_yards', 'receptions', 'targets', 'receiving_tds']:
            cat = 'rec'
        elif stat_col in ['passing_yards', 'completions', 'attempts', 'passing_tds']:
            cat = 'pass'
        else:
            return 1.0

        opp_avg = opp_data['avg'].get(cat, 0)
        league_avg = opp_data['league_avg'].get(cat, 1)

        if league_avg == 0:
            return 1.0

        # Calculate multiplier
        # More aggressive for rushing (0.75-1.35), less for passing (0.85-1.20)
        multiplier = opp_avg / league_avg
        if cat == 'rush':
            return max(0.75, min(1.35, multiplier))
        elif cat == 'pass':
            return max(0.85, min(1.20, multiplier))
        else:
            return max(0.80, min(1.25, multiplier))

    def get_player_features(self, player_id: str, max_week: int) -> Dict:
        """Get rolling average features for a player using only data before max_week."""
        player_df = self.player_stats[
            (self.player_stats['player_id'] == player_id) &
            (self.player_stats['week'] < max_week)
        ].sort_values('week', ascending=False)

        if len(player_df) == 0:
            return {}

        # Calculate rolling averages
        features = {}
        recent = player_df.head(4)  # Last 4 games

        # Stats to calculate averages for
        stats = [
            'passing_yards', 'passing_tds', 'completions', 'attempts', 'passing_interceptions',
            'rushing_yards', 'rushing_tds', 'carries',
            'receiving_yards', 'receiving_tds', 'receptions', 'targets'
        ]

        for stat in stats:
            if stat in player_df.columns:
                # Season average
                features[f'{stat}_season_avg'] = player_df[stat].mean()
                # Last 3 games average
                features[f'{stat}_l3_avg'] = recent.head(3)[stat].mean() if len(recent) >= 3 else recent[stat].mean()
                # Last game
                features[f'{stat}_last'] = player_df.iloc[0][stat] if len(player_df) > 0 else 0

        # Add metadata
        features['games_played'] = len(player_df)
        features['player_name'] = player_df.iloc[0].get('player_display_name', '')
        features['position'] = player_df.iloc[0].get('position', 'UNK')
        features['team'] = player_df.iloc[0].get('team', 'UNK')

        return features

    def generate_projection(self, player_id: str, prop_type: str, features: Dict, opponent: str = None) -> Optional[float]:
        """Generate a projection for a specific prop type, adjusted for opponent."""
        # Get the stat column name
        stat_col = self.prop_map.get(prop_type, prop_type)

        # Use rolling average as projection
        season_avg = features.get(f'{stat_col}_season_avg', 0)
        l3_avg = features.get(f'{stat_col}_l3_avg', 0)
        last_game = features.get(f'{stat_col}_last', 0)

        if not any([season_avg, l3_avg, last_game]):
            return None

        # Weighted average: 50% L3, 30% season, 20% last
        base_projection = 0.5 * l3_avg + 0.3 * season_avg + 0.2 * last_game

        # Apply opponent adjustment for yardage stats
        if opponent and stat_col in ['rushing_yards', 'receiving_yards', 'passing_yards']:
            multiplier = self.get_opponent_multiplier(opponent, stat_col)
            base_projection = base_projection * multiplier

        return round(base_projection, 1) if base_projection > 0 else None

    def get_actual_result(self, player_id: str, week: int, prop_type: str) -> Optional[float]:
        """Get actual result for a player in a specific week."""
        stat_col = self.prop_map.get(prop_type, prop_type)

        player_week = self.player_stats[
            (self.player_stats['player_id'] == player_id) &
            (self.player_stats['week'] == week)
        ]

        if len(player_week) == 0:
            return None

        return player_week.iloc[0].get(stat_col, None)

    def get_default_line(self, prop_type: str, projection: float) -> Optional[float]:
        """Get a default line based on prop type when no historical data exists.

        Returns None if we should skip this prop due to lack of real data.
        Only used for props where we have strong priors about typical lines.
        """
        # For backtesting, we REQUIRE real historical lines
        # Return None to indicate this prop should be skipped
        return None

    def load_historical_props(self, week: int) -> Dict:
        """Load real historical props data if available.

        Returns dict keyed by (player, prop_type) -> line
        """
        props_file = Path(f"outputs/backtest_props_nov9_2025.json")
        if not props_file.exists():
            return {}

        with open(props_file) as f:
            data = json.load(f)

        # Map short prop types to standard names
        prop_type_map = {
            'pass_yards': 'passing_yards',
            'rush_yards': 'rushing_yards',
            'rec_yards': 'receiving_yards',
            'passing_yards': 'passing_yards',
            'rushing_yards': 'rushing_yards',
            'receiving_yards': 'receiving_yards',
            'receptions': 'receptions',
            'passing_tds': 'passing_tds',
            'rushing_tds': 'rushing_tds',
            'receiving_tds': 'receiving_tds',
        }

        # Build lookup
        historical = {}
        for prop in data.get('props', []):
            player = prop['player']
            prop_type = prop_type_map.get(prop['prop_type'], prop['prop_type'])
            key = (player, prop_type)
            historical[key] = prop['line']

        return historical

    def run_backtest(self, weeks: List[int] = [9, 10]) -> Dict:
        """Run complete backtest for specified weeks.

        Args:
            weeks: List of weeks to backtest

        Returns:
            Comprehensive backtest results
        """
        print(f"\n{'='*80}")
        print(f"RUNNING BACKTEST FOR WEEKS {weeks}")
        print(f"{'='*80}\n")

        all_results = []

        for week in weeks:
            print(f"\n--- Week {week} ---")
            week_results = self._backtest_week(week)
            all_results.extend(week_results)
            print(f"Generated {len(week_results)} prop predictions")

        # Analyze results
        analysis = self._analyze_results(all_results)

        # Create detailed report
        report = {
            'backtest_info': {
                'season': self.season,
                'weeks': weeks,
                'timestamp': datetime.now().isoformat(),
                'total_predictions': len(all_results)
            },
            'summary': analysis['summary'],
            'by_prop_type': analysis['by_prop_type'],
            'by_position': analysis['by_position'],
            'by_week': analysis['by_week'],
            'betting_simulation': analysis['betting'],
            'top_hits': analysis['top_hits'],
            'worst_misses': analysis['worst_misses'],
            'all_predictions': all_results
        }

        return report

    def _backtest_week(self, week: int) -> List[Dict]:
        """Generate and evaluate projections for a single week."""
        results = []

        # Get games for this week
        week_games = self.schedules[
            (self.schedules['season'] == self.season) &
            (self.schedules['week'] == week)
        ]

        if len(week_games) == 0:
            print(f"No games found for week {week}")
            return results

        # Get unique teams playing
        teams_playing = set(week_games['home_team'].tolist() + week_games['away_team'].tolist())

        # Load real historical props for this week
        historical_props = self.load_historical_props(week)
        if historical_props:
            print(f"  Loaded {len(historical_props)} real historical lines")

        # Get players who played in weeks before this one
        recent_players = self.player_stats[
            (self.player_stats['week'] >= week - 4) &
            (self.player_stats['week'] < week) &
            (self.player_stats['team'].isin(teams_playing))
        ][['player_id', 'player_display_name', 'team', 'position']].drop_duplicates()

        for _, player in recent_players.iterrows():
            player_id = player['player_id']
            player_name = player['player_display_name']
            position = player.get('position', 'UNK')
            team = player.get('team', 'UNK')

            # Determine position group
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

            # Get player features using only pre-week data
            features = self.get_player_features(player_id, week)
            if not features:
                continue

            # Get opponent
            game = week_games[
                (week_games['home_team'] == team) |
                (week_games['away_team'] == team)
            ]
            if len(game) == 0:
                continue

            game = game.iloc[0]
            opponent = game['away_team'] if game['home_team'] == team else game['home_team']
            is_home = 1 if game['home_team'] == team else 0

            # Generate projections for each relevant prop type
            for prop_type in self.position_props.get(pos_group, []):
                # Pass opponent for defensive adjustment
                projection = self.generate_projection(player_id, prop_type, features, opponent=opponent)
                if projection is None or projection <= 0:
                    continue

                # Get actual result
                actual = self.get_actual_result(player_id, week, prop_type)
                if actual is None:
                    continue  # Player didn't play

                # Get real historical line - skip if not available
                prop_key = (player_name, prop_type)
                if prop_key in historical_props:
                    line = historical_props[prop_key]
                else:
                    # No real line available - skip this prop
                    # We don't use simulated lines for backtesting
                    continue

                # Calculate prediction accuracy
                error = projection - actual
                abs_error = abs(error)
                pct_error = (abs_error / actual * 100) if actual > 0 else 0

                # Determine if over/under hit
                over_hit = 1 if actual > line else 0
                under_hit = 1 if actual < line else 0
                push = 1 if actual == line else 0

                # Our recommendation
                recommend_over = projection > line

                # Did we win?
                correct = (recommend_over and over_hit) or (not recommend_over and under_hit)

                # Edge (our projection vs line)
                edge = projection - line
                edge_pct = (edge / line * 100) if line > 0 else 0

                results.append({
                    'week': week,
                    'player_id': player_id,
                    'player_name': player_name,
                    'team': team,
                    'opponent': opponent,
                    'position': position,
                    'prop_type': prop_type,
                    'projection': projection,
                    'line': line,
                    'actual': actual,
                    'error': round(error, 1),
                    'abs_error': round(abs_error, 1),
                    'pct_error': round(pct_error, 1),
                    'edge': round(edge, 1),
                    'edge_pct': round(edge_pct, 1),
                    'recommend_over': recommend_over,
                    'correct': correct,
                    'over_hit': over_hit,
                    'under_hit': under_hit,
                    'push': push,
                    'is_home': is_home
                })

        return results

    def _analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze backtest results and calculate metrics."""
        if not results:
            return {'summary': {}, 'by_prop_type': {}, 'by_position': {},
                   'by_week': {}, 'betting': {}, 'top_hits': [], 'worst_misses': []}

        df = pd.DataFrame(results)

        # Overall summary
        total = len(df)
        correct = df['correct'].sum()
        pushes = df['push'].sum()

        summary = {
            'total_predictions': total,
            'correct': int(correct),
            'incorrect': int(total - correct - pushes),
            'pushes': int(pushes),
            'hit_rate': round(correct / (total - pushes) * 100 if (total - pushes) > 0 else 0, 1),
            'mae': round(df['abs_error'].mean(), 2),
            'rmse': round(np.sqrt((df['error'] ** 2).mean()), 2),
            'avg_edge': round(df['edge'].mean(), 2),
            'avg_pct_error': round(df['pct_error'].mean(), 1)
        }

        # By prop type
        by_prop_type = {}
        for prop_type in df['prop_type'].unique():
            prop_df = df[df['prop_type'] == prop_type]
            prop_correct = prop_df['correct'].sum()
            prop_total = len(prop_df) - prop_df['push'].sum()

            by_prop_type[prop_type] = {
                'predictions': len(prop_df),
                'correct': int(prop_correct),
                'hit_rate': round(prop_correct / prop_total * 100 if prop_total > 0 else 0, 1),
                'mae': round(prop_df['abs_error'].mean(), 2),
                'avg_edge': round(prop_df['edge'].mean(), 2)
            }

        # By position
        by_position = {}
        for pos in df['position'].unique():
            pos_df = df[df['position'] == pos]
            pos_correct = pos_df['correct'].sum()
            pos_total = len(pos_df) - pos_df['push'].sum()

            by_position[pos] = {
                'predictions': len(pos_df),
                'correct': int(pos_correct),
                'hit_rate': round(pos_correct / pos_total * 100 if pos_total > 0 else 0, 1),
                'mae': round(pos_df['abs_error'].mean(), 2)
            }

        # By week
        by_week = {}
        for week in df['week'].unique():
            week_df = df[df['week'] == week]
            week_correct = week_df['correct'].sum()
            week_total = len(week_df) - week_df['push'].sum()

            by_week[int(week)] = {
                'predictions': len(week_df),
                'correct': int(week_correct),
                'hit_rate': round(week_correct / week_total * 100 if week_total > 0 else 0, 1),
                'mae': round(week_df['abs_error'].mean(), 2)
            }

        # Betting simulation
        betting = self._simulate_betting(df)

        # Top hits (smallest errors)
        top_hits_df = df.nsmallest(10, 'abs_error')
        top_hits = []
        for _, row in top_hits_df.iterrows():
            top_hits.append({
                'player': row['player_name'],
                'prop_type': row['prop_type'],
                'projection': row['projection'],
                'actual': row['actual'],
                'error': row['error']
            })

        # Worst misses (largest errors)
        worst_misses_df = df.nlargest(10, 'abs_error')
        worst_misses = []
        for _, row in worst_misses_df.iterrows():
            worst_misses.append({
                'player': row['player_name'],
                'prop_type': row['prop_type'],
                'projection': row['projection'],
                'actual': row['actual'],
                'error': row['error']
            })

        return {
            'summary': summary,
            'by_prop_type': by_prop_type,
            'by_position': by_position,
            'by_week': by_week,
            'betting': betting,
            'top_hits': top_hits,
            'worst_misses': worst_misses
        }

    def _simulate_betting(self, df: pd.DataFrame) -> Dict:
        """Simulate betting ROI with various strategies."""
        # Standard -110 odds (bet $110 to win $100)
        standard_odds = -110

        # Calculate ROI for different edge thresholds
        strategies = {}

        for min_edge in [0, 1, 2, 3, 5]:
            strategy_name = f'edge_{min_edge}+'

            # Filter by minimum edge
            bets = df[df['edge'].abs() >= min_edge]

            if len(bets) == 0:
                strategies[strategy_name] = {
                    'bets': 0, 'wins': 0, 'losses': 0, 'pushes': 0,
                    'profit': 0, 'roi': 0
                }
                continue

            wins = bets['correct'].sum()
            losses = len(bets) - wins - bets['push'].sum()
            pushes = bets['push'].sum()

            # Calculate profit (assuming $100 per bet)
            # Win: +100, Loss: -110, Push: 0
            profit = (wins * 100) - (losses * 110)
            total_risked = (wins + losses) * 110
            roi = (profit / total_risked * 100) if total_risked > 0 else 0

            strategies[strategy_name] = {
                'bets': int(wins + losses + pushes),
                'wins': int(wins),
                'losses': int(losses),
                'pushes': int(pushes),
                'profit': round(profit, 2),
                'roi': round(roi, 2)
            }

        return strategies

    def print_report(self, report: Dict):
        """Print formatted backtest report."""
        print(f"\n{'='*80}")
        print(f"BACKTEST REPORT - Weeks {report['backtest_info']['weeks']}")
        print(f"{'='*80}")

        summary = report['summary']
        print(f"\n📊 OVERALL SUMMARY")
        print(f"   Total Predictions: {summary['total_predictions']}")
        print(f"   Correct: {summary['correct']} ({summary['hit_rate']}%)")
        print(f"   Incorrect: {summary['incorrect']}")
        print(f"   Pushes: {summary['pushes']}")
        print(f"   MAE: {summary['mae']}")
        print(f"   RMSE: {summary['rmse']}")
        print(f"   Avg Edge: {summary['avg_edge']}")

        print(f"\n📈 BY PROP TYPE")
        for prop_type, metrics in sorted(report['by_prop_type'].items(), key=lambda x: -x[1]['hit_rate']):
            print(f"   {prop_type:20s}: {metrics['hit_rate']:5.1f}% ({metrics['correct']}/{metrics['predictions']}) MAE: {metrics['mae']:.1f}")

        print(f"\n👤 BY POSITION")
        for pos, metrics in sorted(report['by_position'].items(), key=lambda x: -x[1]['hit_rate']):
            print(f"   {pos:5s}: {metrics['hit_rate']:5.1f}% ({metrics['correct']}/{metrics['predictions']})")

        print(f"\n📅 BY WEEK")
        for week, metrics in sorted(report['by_week'].items()):
            print(f"   Week {week}: {metrics['hit_rate']:5.1f}% ({metrics['correct']}/{metrics['predictions']})")

        print(f"\n💰 BETTING SIMULATION (Standard -110 odds, $100/bet)")
        for strategy, metrics in report['betting_simulation'].items():
            if metrics['bets'] > 0:
                status = "✅" if metrics['profit'] > 0 else "❌"
                print(f"   {status} {strategy:10s}: {metrics['wins']}W-{metrics['losses']}L-{metrics['pushes']}P | "
                      f"Profit: ${metrics['profit']:+.0f} | ROI: {metrics['roi']:+.1f}%")

        print(f"\n🎯 TOP 5 BEST PREDICTIONS")
        for i, hit in enumerate(report['top_hits'][:5], 1):
            print(f"   {i}. {hit['player']:20s} {hit['prop_type']:15s}: "
                  f"Proj {hit['projection']:5.1f} vs Actual {hit['actual']:5.1f} (Error: {hit['error']:+.1f})")

        print(f"\n❌ TOP 5 WORST MISSES")
        for i, miss in enumerate(report['worst_misses'][:5], 1):
            print(f"   {i}. {miss['player']:20s} {miss['prop_type']:15s}: "
                  f"Proj {miss['projection']:5.1f} vs Actual {miss['actual']:5.1f} (Error: {miss['error']:+.1f})")

        print(f"\n{'='*80}\n")


def main():
    """Run the weeks 9-10 backtest."""
    backtester = WeeksBacktester(season=2024)

    # Run backtest for weeks 9 and 10
    report = backtester.run_backtest(weeks=[9, 10])

    # Print report
    backtester.print_report(report)

    # Save report to file
    output_file = Path("outputs/backtest_weeks_9_10.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"📁 Full report saved to: {output_file}")

    return report


if __name__ == "__main__":
    main()
