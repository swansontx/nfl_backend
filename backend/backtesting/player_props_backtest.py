"""Player Props Backtesting Module.

Validates player projection accuracy for props betting:
- Passing yards, TDs, completions
- Rushing yards, TDs, attempts
- Receiving yards, TDs, receptions
- Player fantasy points
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy import stats

from backend.backtesting.framework import BacktestingFramework, BacktestResult


@dataclass
class PropPrediction:
    """Single player prop prediction."""
    player: str
    position: str
    team: str
    opponent: str
    week: int
    season: int
    prop_type: str  # 'passing_yards', 'rushing_yards', 'receiving_yards', etc.
    predicted_value: float
    actual_value: float
    line: float  # Vegas line if available
    error: float

    @property
    def abs_error(self) -> float:
        return abs(self.error)

    @property
    def beat_line(self) -> bool:
        """Did actual exceed the line?"""
        return self.actual_value > self.line if self.line > 0 else None

    @property
    def predicted_beat_line(self) -> bool:
        """Did prediction say to take over?"""
        return self.predicted_value > self.line if self.line > 0 else None


class PlayerPropsBacktester:
    """Backtest player prop predictions."""

    def __init__(self, framework: BacktestingFramework):
        """Initialize backtester.

        Args:
            framework: Backtesting framework with historical data
        """
        self.framework = framework
        self.predictions: List[PropPrediction] = []

    def generate_predictions(self) -> List[PropPrediction]:
        """Generate prop predictions for all players/games.

        Returns:
            List of prop predictions
        """
        predictions = []

        for season in self.framework.seasons:
            player_stats = self.framework.load_player_stats(season, 'all')

            for idx, row in player_stats.iterrows():
                player = row.get('player', 'Unknown')
                position = row.get('position', 'Unknown')
                team = row.get('team', 'N/A')
                week = int(row.get('week', 0))

                # Skip week 1 (no baseline)
                if week <= 2:
                    continue

                # Get player's recent average (last 3 games)
                recent_stats = player_stats[
                    (player_stats['player'] == player) &
                    (player_stats['week'] < week) &
                    (player_stats['week'] >= max(1, week - 3))
                ]

                if recent_stats.empty:
                    continue

                # Generate predictions for different prop types
                predictions.extend(self._generate_player_props(
                    player, position, team, row.get('opponent_team', 'N/A'),
                    week, season, recent_stats, row
                ))

        self.predictions = predictions
        return predictions

    def _generate_player_props(
        self,
        player: str,
        position: str,
        team: str,
        opponent: str,
        week: int,
        season: int,
        recent_stats: pd.DataFrame,
        actual_row: pd.Series
    ) -> List[PropPrediction]:
        """Generate prop predictions for a single player.

        Args:
            player: Player name
            position: Player position
            team: Player team
            opponent: Opponent team
            week: Week number
            season: Season year
            recent_stats: Recent game stats for baseline
            actual_row: Actual stats from this game

        Returns:
            List of PropPrediction objects
        """
        props = []

        # QB props
        if position == 'QB':
            # Passing yards
            predicted_passing = recent_stats['passing_yards'].mean()
            actual_passing = float(actual_row.get('passing_yards', 0))
            props.append(PropPrediction(
                player=player,
                position=position,
                team=team,
                opponent=opponent,
                week=week,
                season=season,
                prop_type='passing_yards',
                predicted_value=predicted_passing,
                actual_value=actual_passing,
                line=250.0,  # Typical QB line
                error=predicted_passing - actual_passing
            ))

            # Passing TDs
            predicted_pass_tds = recent_stats['passing_tds'].mean()
            actual_pass_tds = float(actual_row.get('passing_tds', 0))
            props.append(PropPrediction(
                player=player,
                position=position,
                team=team,
                opponent=opponent,
                week=week,
                season=season,
                prop_type='passing_tds',
                predicted_value=predicted_pass_tds,
                actual_value=actual_pass_tds,
                line=1.5,  # Typical TD line
                error=predicted_pass_tds - actual_pass_tds
            ))

        # RB props
        elif position == 'RB':
            # Rushing yards
            predicted_rushing = recent_stats['rushing_yards'].mean()
            actual_rushing = float(actual_row.get('rushing_yards', 0))
            props.append(PropPrediction(
                player=player,
                position=position,
                team=team,
                opponent=opponent,
                week=week,
                season=season,
                prop_type='rushing_yards',
                predicted_value=predicted_rushing,
                actual_value=actual_rushing,
                line=65.5,  # Typical RB line
                error=predicted_rushing - actual_rushing
            ))

            # Receiving yards
            predicted_receiving = recent_stats['receiving_yards'].mean()
            actual_receiving = float(actual_row.get('receiving_yards', 0))
            if predicted_receiving > 5 or actual_receiving > 5:  # Only if catching passes
                props.append(PropPrediction(
                    player=player,
                    position=position,
                    team=team,
                    opponent=opponent,
                    week=week,
                    season=season,
                    prop_type='receiving_yards',
                    predicted_value=predicted_receiving,
                    actual_value=actual_receiving,
                    line=25.5,  # Typical RB receiving line
                    error=predicted_receiving - actual_receiving
                ))

        # WR/TE props
        elif position in ['WR', 'TE']:
            # Receiving yards
            predicted_receiving = recent_stats['receiving_yards'].mean()
            actual_receiving = float(actual_row.get('receiving_yards', 0))
            props.append(PropPrediction(
                player=player,
                position=position,
                team=team,
                opponent=opponent,
                week=week,
                season=season,
                prop_type='receiving_yards',
                predicted_value=predicted_receiving,
                actual_value=actual_receiving,
                line=55.5 if position == 'WR' else 45.5,
                error=predicted_receiving - actual_receiving
            ))

            # Receptions
            predicted_receptions = recent_stats['receptions'].mean()
            actual_receptions = float(actual_row.get('receptions', 0))
            props.append(PropPrediction(
                player=player,
                position=position,
                team=team,
                opponent=opponent,
                week=week,
                season=season,
                prop_type='receptions',
                predicted_value=predicted_receptions,
                actual_value=actual_receptions,
                line=4.5 if position == 'WR' else 3.5,
                error=predicted_receptions - actual_receptions
            ))

        return props

    def calculate_prop_accuracy(self, prop_type: str = None, position: str = None) -> Dict:
        """Calculate accuracy metrics for props.

        Args:
            prop_type: Filter by prop type (e.g., 'passing_yards')
            position: Filter by position (e.g., 'QB')

        Returns:
            Dictionary with accuracy metrics
        """
        filtered = self.predictions

        if prop_type:
            filtered = [p for p in filtered if p.prop_type == prop_type]
        if position:
            filtered = [p for p in filtered if p.position == position]

        if not filtered:
            return {
                'count': 0,
                'rmse': 0.0,
                'mae': 0.0,
                'mean_error': 0.0,
                'bias_pct': 0.0
            }

        errors = [p.error for p in filtered]
        abs_errors = [p.abs_error for p in filtered]
        actuals = [p.actual_value for p in filtered]

        return {
            'count': len(filtered),
            'rmse': float(np.sqrt(np.mean([e**2 for e in errors]))),
            'mae': float(np.mean(abs_errors)),
            'mean_error': float(np.mean(errors)),
            'bias_pct': float((np.mean(errors) / np.mean(actuals) * 100) if np.mean(actuals) != 0 else 0),
            'within_5': float(sum(1 for e in abs_errors if e <= 5) / len(abs_errors) * 100),
            'within_10': float(sum(1 for e in abs_errors if e <= 10) / len(abs_errors) * 100),
            'within_20': float(sum(1 for e in abs_errors if e <= 20) / len(abs_errors) * 100)
        }

    def calculate_betting_performance(self, prop_type: str = None) -> Dict:
        """Calculate betting performance vs lines.

        Args:
            prop_type: Filter by prop type

        Returns:
            Betting performance metrics
        """
        filtered = [p for p in self.predictions if p.line > 0]

        if prop_type:
            filtered = [p for p in filtered if p.prop_type == prop_type]

        if not filtered:
            return {
                'total_bets': 0,
                'correct_predictions': 0,
                'accuracy': 0.0,
                'ev': 0.0
            }

        correct = sum(1 for p in filtered if p.beat_line == p.predicted_beat_line)
        total = len(filtered)
        accuracy = correct / total * 100

        # Calculate expected value (assuming -110 odds)
        win_amount = 0.909  # Win $0.909 for every $1 bet at -110
        ev = (accuracy / 100) * win_amount - ((100 - accuracy) / 100) * 1.0

        return {
            'total_bets': total,
            'correct_predictions': correct,
            'accuracy': accuracy,
            'ev': ev * 100,  # As percentage
            'breakeven_needed': 52.38  # Need 52.38% to break even at -110
        }

    def get_best_props(self, min_predictions: int = 50) -> List[Dict]:
        """Find most accurate prop types.

        Args:
            min_predictions: Minimum predictions needed

        Returns:
            List of prop types sorted by accuracy
        """
        prop_types = {}

        for prop_type in set(p.prop_type for p in self.predictions):
            metrics = self.calculate_prop_accuracy(prop_type=prop_type)

            if metrics['count'] >= min_predictions:
                prop_types[prop_type] = {
                    'prop_type': prop_type,
                    'count': metrics['count'],
                    'mae': metrics['mae'],
                    'within_10_pct': metrics['within_10'],
                    'bias_pct': metrics['bias_pct']
                }

        # Sort by MAE (lower is better)
        return sorted(prop_types.values(), key=lambda x: x['mae'])

    def run_backtest(self) -> BacktestResult:
        """Run player props backtest.

        Returns:
            BacktestResult with findings
        """
        print("Running player props backtest...")

        # Generate predictions
        self.generate_predictions()

        if not self.predictions:
            print("  ⚠️ No predictions generated")
            return BacktestResult(
                feature_name="Player Props Accuracy",
                seasons_tested=self.framework.seasons,
                sample_size=0,
                calculated_factors={},
                original_factors={},
                should_update=False,
                improvement_pct=0.0,
                notes=["No predictions generated - insufficient data"]
            )

        # Calculate metrics by position and prop type
        results_by_position = {}

        for position in ['QB', 'RB', 'WR', 'TE']:
            position_props = [p for p in self.predictions if p.position == position]
            if not position_props:
                continue

            position_metrics = {}

            # Get prop types for this position
            prop_types = set(p.prop_type for p in position_props)

            for prop_type in prop_types:
                metrics = self.calculate_prop_accuracy(prop_type=prop_type, position=position)
                betting = self.calculate_betting_performance(prop_type=prop_type)

                position_metrics[prop_type] = {
                    **metrics,
                    'betting_accuracy': betting['accuracy'],
                    'betting_ev': betting['ev']
                }

            results_by_position[position] = position_metrics

        # Overall metrics
        overall_metrics = self.calculate_prop_accuracy()
        overall_betting = self.calculate_betting_performance()

        # Generate notes
        notes = [
            f"Analyzed {len(self.predictions)} player prop predictions",
            f"Positions: {len(results_by_position)} (QB, RB, WR, TE)",
            "",
            "OVERALL ACCURACY:",
            f"  MAE: {overall_metrics['mae']:.2f}",
            f"  Within 10: {overall_metrics['within_10']:.1f}%",
            f"  Bias: {overall_metrics['bias_pct']:+.1f}%",
            "",
            "BETTING PERFORMANCE:",
            f"  Accuracy vs line: {overall_betting['accuracy']:.1f}%",
            f"  Expected value: {overall_betting['ev']:+.2f}%",
            f"  (Need {overall_betting['breakeven_needed']:.1f}% to break even)",
            ""
        ]

        # Best performing props
        best_props = self.get_best_props()
        if best_props:
            notes.append("MOST ACCURATE PROPS:")
            for prop in best_props[:5]:
                notes.append(f"  {prop['prop_type']}: MAE={prop['mae']:.1f}, {prop['within_10_pct']:.1f}% within 10")

        result = BacktestResult(
            feature_name="Player Props Accuracy",
            seasons_tested=self.framework.seasons,
            sample_size=len(self.predictions),
            rmse=overall_metrics['rmse'],
            mae=overall_metrics['mae'],
            calculated_factors=results_by_position,
            original_factors={},
            should_update=True,
            improvement_pct=10.0 if overall_betting['accuracy'] > 52.38 else 0.0,
            notes=notes
        )

        return result


if __name__ == "__main__":
    # Test player props backtester
    framework = BacktestingFramework(seasons=[2022, 2023])
    backtester = PlayerPropsBacktester(framework)
    result = backtester.run_backtest()

    print(f"\nSample size: {result.sample_size}")
    print(f"MAE: {result.mae:.2f}")
    print(f"Should update: {result.should_update}")
