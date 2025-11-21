"""
Improved backtest with better projections.

Key improvements over basic backtest:
1. Opponent defensive adjustments
2. Game context (spread/total) adjustments
3. Recency-weighted averages
4. Volume/opportunity adjustments
5. Better variance estimation
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict
import joblib


class ImprovedBacktester:
    """Enhanced backtester with opponent and game context adjustments."""

    def __init__(self, season: int = 2024, inputs_dir: str = "inputs", models_dir: str = "outputs/models"):
        self.season = season
        self.inputs_dir = Path(inputs_dir)
        self.models_dir = Path(models_dir)

        # Load data
        self.player_stats = self._load_player_stats()
        self.schedules = self._load_schedules()
        self.team_stats = self._load_team_stats()

        # Calculate defensive rankings
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

    def _load_team_stats(self) -> pd.DataFrame:
        """Load team defensive stats."""
        team_file = self.inputs_dir / "team_stats_2024_2025.csv"
        if team_file.exists():
            return pd.read_csv(team_file, low_memory=False)
        return pd.DataFrame()

    def _calculate_defensive_rankings(self) -> Dict:
        """Calculate team defensive rankings for opponent adjustments."""
        rankings = {}

        if len(self.player_stats) == 0:
            return rankings

        # Group opponent stats to get defensive performance
        # Higher values = worse defense (allows more)
        opp_stats = self.player_stats.groupby('opponent_team').agg({
            'passing_yards': 'mean',
            'passing_tds': 'mean',
            'rushing_yards': 'mean',
            'rushing_tds': 'mean',
            'receiving_yards': 'mean',
            'receptions': 'mean',
            'targets': 'mean',
            'completions': 'mean',
        }).reset_index()

        # Calculate league averages
        league_avg = {
            'passing_yards': opp_stats['passing_yards'].mean(),
            'passing_tds': opp_stats['passing_tds'].mean(),
            'rushing_yards': opp_stats['rushing_yards'].mean(),
            'rushing_tds': opp_stats['rushing_tds'].mean(),
            'receiving_yards': opp_stats['receiving_yards'].mean(),
            'receptions': opp_stats['receptions'].mean(),
            'targets': opp_stats['targets'].mean(),
            'completions': opp_stats['completions'].mean(),
        }

        # Calculate multipliers for each team (>1 = bad defense, <1 = good defense)
        for _, row in opp_stats.iterrows():
            team = row['opponent_team']
            rankings[team] = {}

            for stat in league_avg.keys():
                if league_avg[stat] > 0:
                    # Ratio vs league average
                    rankings[team][stat] = row[stat] / league_avg[stat]
                else:
                    rankings[team][stat] = 1.0

        return rankings

    def get_opponent_multiplier(self, opponent: str, prop_type: str) -> float:
        """Get opponent adjustment multiplier for a prop type."""
        if opponent not in self.def_rankings:
            return 1.0

        stat_col = self.prop_map.get(prop_type, prop_type)

        # Map prop types to defensive stats
        if stat_col in self.def_rankings[opponent]:
            multiplier = self.def_rankings[opponent][stat_col]
            # Dampen extreme values (0.85 to 1.15 range)
            return max(0.85, min(1.15, multiplier))

        return 1.0

    def get_game_context_adjustment(self, spread: float, total: float, is_home: bool, prop_type: str) -> float:
        """Adjust projection based on game context."""
        adjustment = 1.0

        # Higher totals = more scoring = more stats
        avg_total = 45.0
        total_factor = total / avg_total if total > 0 else 1.0

        # Spread affects game script
        # Favorites may run more, underdogs may pass more
        spread_factor = 1.0

        if prop_type in ['passing_yards', 'passing_tds', 'completions', 'attempts']:
            # Underdogs pass more (chasing)
            if is_home and spread > 3:  # Home underdog
                spread_factor = 1.05
            elif not is_home and spread < -3:  # Away underdog
                spread_factor = 1.05
            # Blowout favorites may not pass as much
            if abs(spread) > 10:
                spread_factor *= 0.95

        elif prop_type in ['rushing_yards', 'rushing_tds', 'carries']:
            # Favorites run more (protecting leads)
            if is_home and spread < -3:  # Home favorite
                spread_factor = 1.05
            elif not is_home and spread > 3:  # Away favorite
                spread_factor = 1.05

        # Total affects volume
        if prop_type in ['passing_yards', 'rushing_yards', 'receiving_yards']:
            adjustment = (total_factor * 0.3 + 0.7)  # 30% weight on total
        elif prop_type in ['passing_tds', 'rushing_tds', 'receiving_tds']:
            adjustment = (total_factor * 0.4 + 0.6)  # 40% weight for TDs
        else:
            adjustment = (total_factor * 0.2 + 0.8)  # 20% for other stats

        return adjustment * spread_factor

    def get_player_features(self, player_id: str, max_week: int) -> Dict:
        """Get enhanced player features with recency weighting."""
        player_df = self.player_stats[
            (self.player_stats['player_id'] == player_id) &
            (self.player_stats['week'] < max_week)
        ].sort_values('week', ascending=False)

        if len(player_df) == 0:
            return {}

        features = {}

        # Stats to process
        stats = [
            'passing_yards', 'passing_tds', 'completions', 'attempts', 'passing_interceptions',
            'rushing_yards', 'rushing_tds', 'carries',
            'receiving_yards', 'receiving_tds', 'receptions', 'targets'
        ]

        # Recency weights: most recent games weighted higher
        weights = [0.35, 0.25, 0.20, 0.12, 0.08]  # Last 5 games

        for stat in stats:
            if stat not in player_df.columns:
                continue

            values = player_df[stat].head(5).tolist()

            if len(values) == 0:
                features[f'{stat}_proj'] = 0
                features[f'{stat}_var'] = 0
                continue

            # Weighted average
            weighted_sum = 0
            weight_sum = 0
            for i, val in enumerate(values):
                if pd.notna(val):
                    w = weights[i] if i < len(weights) else 0.05
                    weighted_sum += val * w
                    weight_sum += w

            if weight_sum > 0:
                features[f'{stat}_proj'] = weighted_sum / weight_sum
            else:
                features[f'{stat}_proj'] = 0

            # Variance (for confidence)
            if len(values) >= 3:
                features[f'{stat}_var'] = np.std(values)
            else:
                features[f'{stat}_var'] = features[f'{stat}_proj'] * 0.3  # High uncertainty

            # Season average for reference
            features[f'{stat}_season_avg'] = player_df[stat].mean()

        # Metadata
        features['games_played'] = len(player_df)
        features['player_name'] = player_df.iloc[0].get('player_display_name', '')
        features['position'] = player_df.iloc[0].get('position', 'UNK')
        features['team'] = player_df.iloc[0].get('team', 'UNK')

        return features

    def generate_projection(self, player_id: str, prop_type: str, features: Dict,
                           opponent: str, spread: float, total: float, is_home: bool) -> Optional[Dict]:
        """Generate enhanced projection with adjustments."""
        stat_col = self.prop_map.get(prop_type, prop_type)

        # Base projection from recency-weighted average
        base_proj = features.get(f'{stat_col}_proj', 0)
        variance = features.get(f'{stat_col}_var', 0)

        if base_proj <= 0:
            return None

        # Apply opponent adjustment
        opp_mult = self.get_opponent_multiplier(opponent, prop_type)

        # Apply game context adjustment
        context_adj = self.get_game_context_adjustment(spread, total, is_home, prop_type)

        # Final projection
        projection = base_proj * opp_mult * context_adj

        # Adjust variance based on confidence
        games_played = features.get('games_played', 1)
        confidence = min(1.0, games_played / 5)  # Full confidence at 5+ games

        # Higher variance for less confident projections
        adjusted_var = variance / confidence if confidence > 0 else variance * 2

        return {
            'projection': round(projection, 1),
            'variance': round(adjusted_var, 2),
            'base_proj': round(base_proj, 1),
            'opp_mult': round(opp_mult, 3),
            'context_adj': round(context_adj, 3),
            'confidence': round(confidence, 2)
        }

    def load_historical_props(self, week: int) -> Dict:
        """Load real historical props data."""
        props_file = Path("outputs/backtest_props_nov9_2025.json")
        if not props_file.exists():
            return {}

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

        try:
            with open(props_file) as f:
                data = json.load(f)

            historical = {}
            for prop in data.get('props', []):
                player = prop['player']
                prop_type = prop_type_map.get(prop['prop_type'], prop['prop_type'])
                key = (player, prop_type)
                historical[key] = prop['line']
            return historical
        except Exception:
            return {}

    def get_historical_line(self, player_name: str, prop_type: str, historical_props: Dict) -> Optional[float]:
        """Get real historical line for backtesting.

        Returns None if no real line is available - prop should be skipped.
        """
        prop_key = (player_name, prop_type)
        return historical_props.get(prop_key)

    def run_backtest(self, weeks: List[int] = [9, 10]) -> Dict:
        """Run improved backtest."""
        print(f"\n{'='*80}")
        print(f"IMPROVED BACKTEST FOR WEEKS {weeks}")
        print(f"With: Opponent Adjustments + Game Context + Recency Weighting")
        print(f"{'='*80}\n")

        all_results = []

        for week in weeks:
            print(f"\n--- Week {week} ---")
            week_results = self._backtest_week(week)
            all_results.extend(week_results)
            print(f"Generated {len(week_results)} predictions")

        # Analyze
        analysis = self._analyze_results(all_results)

        report = {
            'backtest_info': {
                'season': self.season,
                'weeks': weeks,
                'timestamp': datetime.now().isoformat(),
                'total_predictions': len(all_results),
                'improvements': ['opponent_adjustments', 'game_context', 'recency_weighting']
            },
            'summary': analysis['summary'],
            'by_prop_type': analysis['by_prop_type'],
            'by_position': analysis['by_position'],
            'betting_simulation': analysis['betting'],
            'all_predictions': all_results
        }

        return report

    def _backtest_week(self, week: int) -> List[Dict]:
        """Backtest a single week with improved projections."""
        results = []

        week_games = self.schedules[
            (self.schedules['season'] == self.season) &
            (self.schedules['week'] == week)
        ]

        if len(week_games) == 0:
            return results

        teams_playing = set(week_games['home_team'].tolist() + week_games['away_team'].tolist())

        # Load real historical props for this week
        historical_props = self.load_historical_props(week)
        if historical_props:
            print(f"  Loaded {len(historical_props)} real historical lines")
        else:
            print(f"  WARNING: No historical props data - skipping all props for week {week}")
            return results

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

            # Position group
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

            features = self.get_player_features(player_id, week)
            if not features:
                continue

            # Get game info
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
                proj_result = self.generate_projection(
                    player_id, prop_type, features,
                    opponent, spread, total, is_home
                )

                if proj_result is None:
                    continue

                projection = proj_result['projection']
                variance = proj_result['variance']

                # Get actual
                actual = self._get_actual(player_id, week, prop_type)
                if actual is None:
                    continue

                # Get real historical line - skip if not available
                line = self.get_historical_line(player_name, prop_type, historical_props)
                if line is None:
                    continue  # No real line available

                # Calculate metrics
                error = projection - actual
                abs_error = abs(error)

                over_hit = 1 if actual > line else 0
                under_hit = 1 if actual < line else 0
                push = 1 if actual == line else 0

                recommend_over = projection > line
                correct = (recommend_over and over_hit) or (not recommend_over and under_hit)

                edge = projection - line

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
                    'edge': round(edge, 1),
                    'opp_mult': proj_result['opp_mult'],
                    'context_adj': proj_result['context_adj'],
                    'confidence': proj_result['confidence'],
                    'recommend_over': recommend_over,
                    'correct': correct,
                    'push': push,
                    'is_home': is_home
                })

        return results

    def _get_actual(self, player_id: str, week: int, prop_type: str) -> Optional[float]:
        stat_col = self.prop_map.get(prop_type, prop_type)

        player_week = self.player_stats[
            (self.player_stats['player_id'] == player_id) &
            (self.player_stats['week'] == week)
        ]

        if len(player_week) == 0:
            return None

        return player_week.iloc[0].get(stat_col, None)

    def _analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze backtest results."""
        if not results:
            return {'summary': {}, 'by_prop_type': {}, 'by_position': {}, 'betting': {}}

        df = pd.DataFrame(results)

        # Overall
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
            'avg_edge': round(df['edge'].mean(), 2),
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
            }

        # By position
        by_position = {}
        for pos in df['position'].unique():
            pos_df = df[df['position'] == pos]
            pos_correct = pos_df['correct'].sum()
            pos_total = len(pos_df) - pos_df['push'].sum()

            by_position[pos] = {
                'predictions': len(pos_df),
                'hit_rate': round(pos_correct / pos_total * 100 if pos_total > 0 else 0, 1),
            }

        # Betting simulation
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
        """Print formatted report."""
        print(f"\n{'='*80}")
        print(f"IMPROVED BACKTEST REPORT")
        print(f"{'='*80}")

        s = report['summary']
        print(f"\n📊 OVERALL: {s['correct']}/{s['total_predictions']} ({s['hit_rate']}%)")
        print(f"   MAE: {s['mae']} | Avg Edge: {s['avg_edge']}")

        print(f"\n📈 BY PROP TYPE")
        for pt, m in sorted(report['by_prop_type'].items(), key=lambda x: -x[1]['hit_rate']):
            status = "✅" if m['hit_rate'] >= 55 else "⚠️" if m['hit_rate'] >= 50 else "❌"
            print(f"   {status} {pt:20s}: {m['hit_rate']:5.1f}% ({m['correct']}/{m['predictions']})")

        print(f"\n💰 BETTING SIMULATION")
        for strategy, m in report['betting_simulation'].items():
            status = "✅" if m['profit'] > 0 else "❌"
            print(f"   {status} {strategy}: {m['wins']}W-{m['losses']}L | ${m['profit']:+.0f} | ROI: {m['roi']:+.1f}%")


def main():
    backtester = ImprovedBacktester(season=2024)
    report = backtester.run_backtest(weeks=[9, 10])
    backtester.print_report(report)

    # Save
    output_file = Path("outputs/backtest_improved_weeks_9_10.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n📁 Report saved to: {output_file}")
    return report


if __name__ == "__main__":
    main()
