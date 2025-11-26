"""Backtest EPA and Rest Differential Enhancements.

Tests adding EPA (Expected Points Added) and rest differential to our baseline
game totals predictions to measure accuracy improvements.
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass

from backend.backtesting.framework import BacktestingFramework, BacktestResult
from backend.analysis.epa_utils import EPACalculator, get_rest_differential


@dataclass
class EPAComparison:
    """Comparison between baseline and EPA-enhanced predictions."""
    game_id: str
    week: int
    home_team: str
    away_team: str
    actual_total: float

    # Baseline (simple recent averaging)
    baseline_prediction: float
    baseline_error: float

    # EPA-enhanced
    epa_prediction: float
    epa_error: float
    epa_adjustment: float

    # Rest-enhanced
    rest_prediction: float
    rest_error: float
    rest_adjustment: float

    # Full enhancement (EPA + Rest)
    full_prediction: float
    full_error: float

    # Improvements
    epa_improvement: float  # Negative = better
    rest_improvement: float
    full_improvement: float

    # EPA details
    home_off_epa: float
    home_def_epa: float
    away_off_epa: float
    away_def_epa: float
    rest_differential: int


class EPAEnhancedBacktester:
    """Backtest EPA and rest differential enhancements."""

    def __init__(self, framework: BacktestingFramework):
        """Initialize backtester.

        Args:
            framework: Backtesting framework
        """
        self.framework = framework
        self.epa_calc = EPACalculator()
        self.comparisons: List[EPAComparison] = []

    def run_backtest(self) -> BacktestResult:
        """Run comprehensive backtest.

        Returns:
            BacktestResult with improvement metrics
        """
        print("\n" + "="*80)
        print("EPA + REST DIFFERENTIAL BACKTEST")
        print("="*80 + "\n")

        print("Testing enhancements:")
        print("  1. Baseline: Simple recent averaging (last 4 games)")
        print("  2. EPA: Add EPA-based adjustments")
        print("  3. Rest: Add rest differential adjustments")
        print("  4. Full: EPA + Rest combined")
        print()

        for season in self.framework.seasons:
            print(f"\nSeason {season}:")
            print("-" * 40)

            games = self.framework.load_historical_games(season)

            # Load schedule for rest calculation
            schedule_df = self._load_schedule(season)

            season_comparisons = []

            for game in games:
                # Skip early season (need history for baseline)
                if game.week < 5:
                    continue

                # Skip if no scores
                if game.home_score is None or game.away_score is None:
                    continue

                # Get baseline prediction (simple recent averaging)
                baseline_pred = self._baseline_prediction(game, games)

                if baseline_pred == 0.0:
                    continue  # Not enough data

                # Get EPA adjustment
                epa_adj, epa_details = self.epa_calc.get_epa_adjustment_for_game(
                    game.home_team,
                    game.away_team,
                    game.season,
                    game.week,
                    last_n_games=6
                )

                # Get rest differential
                rest_diff = self._get_rest_differential(
                    game.home_team,
                    game.away_team,
                    game.week,
                    schedule_df
                )

                # Calculate rest adjustment
                rest_adj = self._calculate_rest_adjustment(rest_diff)

                # Predictions
                epa_prediction = baseline_pred + epa_adj
                rest_prediction = baseline_pred + rest_adj
                full_prediction = baseline_pred + epa_adj + rest_adj

                # Actual total
                actual_total = game.home_score + game.away_score

                # Errors
                baseline_error = abs(baseline_pred - actual_total)
                epa_error = abs(epa_prediction - actual_total)
                rest_error = abs(rest_prediction - actual_total)
                full_error = abs(full_prediction - actual_total)

                # Store comparison
                comparison = EPAComparison(
                    game_id=game.game_id,
                    week=game.week,
                    home_team=game.home_team,
                    away_team=game.away_team,
                    actual_total=actual_total,
                    baseline_prediction=baseline_pred,
                    baseline_error=baseline_error,
                    epa_prediction=epa_prediction,
                    epa_error=epa_error,
                    epa_adjustment=epa_adj,
                    rest_prediction=rest_prediction,
                    rest_error=rest_error,
                    rest_adjustment=rest_adj,
                    full_prediction=full_prediction,
                    full_error=full_error,
                    epa_improvement=epa_error - baseline_error,
                    rest_improvement=rest_error - baseline_error,
                    full_improvement=full_error - baseline_error,
                    home_off_epa=epa_details.get('home_off_epa', 0.0),
                    home_def_epa=epa_details.get('home_def_epa', 0.0),
                    away_off_epa=epa_details.get('away_off_epa', 0.0),
                    away_def_epa=epa_details.get('away_def_epa', 0.0),
                    rest_differential=rest_diff
                )

                season_comparisons.append(comparison)
                self.comparisons.append(comparison)

            # Season summary
            if season_comparisons:
                baseline_mae = np.mean([c.baseline_error for c in season_comparisons])
                epa_mae = np.mean([c.epa_error for c in season_comparisons])
                rest_mae = np.mean([c.rest_error for c in season_comparisons])
                full_mae = np.mean([c.full_error for c in season_comparisons])

                print(f"  Games: {len(season_comparisons)}")
                print(f"  Baseline MAE: {baseline_mae:.2f}")
                print(f"  EPA MAE: {epa_mae:.2f} ({((baseline_mae - epa_mae) / baseline_mae * 100):+.1f}%)")
                print(f"  Rest MAE: {rest_mae:.2f} ({((baseline_mae - rest_mae) / baseline_mae * 100):+.1f}%)")
                print(f"  Full MAE: {full_mae:.2f} ({((baseline_mae - full_mae) / baseline_mae * 100):+.1f}%)")

        # Overall metrics
        print("\n" + "="*80)
        print("OVERALL RESULTS")
        print("="*80 + "\n")

        if not self.comparisons:
            print("No comparisons generated")
            return BacktestResult(
                model_name="EPA Enhanced",
                total_predictions=0,
                mae=0.0,
                rmse=0.0,
                improvement_pct=0.0,
                notes=["No data available"]
            )

        # Calculate metrics
        baseline_mae = np.mean([c.baseline_error for c in self.comparisons])
        epa_mae = np.mean([c.epa_error for c in self.comparisons])
        rest_mae = np.mean([c.rest_error for c in self.comparisons])
        full_mae = np.mean([c.full_error for c in self.comparisons])

        epa_improvement_pct = ((baseline_mae - epa_mae) / baseline_mae) * 100
        rest_improvement_pct = ((baseline_mae - rest_mae) / baseline_mae) * 100
        full_improvement_pct = ((baseline_mae - full_mae) / baseline_mae) * 100

        # Win rates
        epa_wins = sum(1 for c in self.comparisons if c.epa_improvement < 0)
        rest_wins = sum(1 for c in self.comparisons if c.rest_improvement < 0)
        full_wins = sum(1 for c in self.comparisons if c.full_improvement < 0)

        epa_win_rate = (epa_wins / len(self.comparisons)) * 100
        rest_win_rate = (rest_wins / len(self.comparisons)) * 100
        full_win_rate = (full_wins / len(self.comparisons)) * 100

        # Within 7 points (betting threshold)
        baseline_within_7 = sum(1 for c in self.comparisons if c.baseline_error <= 7)
        epa_within_7 = sum(1 for c in self.comparisons if c.epa_error <= 7)
        rest_within_7 = sum(1 for c in self.comparisons if c.rest_error <= 7)
        full_within_7 = sum(1 for c in self.comparisons if c.full_error <= 7)

        baseline_within_7_pct = (baseline_within_7 / len(self.comparisons)) * 100
        epa_within_7_pct = (epa_within_7 / len(self.comparisons)) * 100
        rest_within_7_pct = (rest_within_7 / len(self.comparisons)) * 100
        full_within_7_pct = (full_within_7 / len(self.comparisons)) * 100

        # Print results
        print(f"Total Predictions: {len(self.comparisons)}")
        print()

        print("Mean Absolute Error (MAE):")
        print(f"  Baseline:     {baseline_mae:.2f} points")
        print(f"  + EPA:        {epa_mae:.2f} points ({epa_improvement_pct:+.1f}%)")
        print(f"  + Rest:       {rest_mae:.2f} points ({rest_improvement_pct:+.1f}%)")
        print(f"  + EPA + Rest: {full_mae:.2f} points ({full_improvement_pct:+.1f}%)")
        print()

        print("Win Rate (better than baseline):")
        print(f"  + EPA:        {epa_win_rate:.1f}%")
        print(f"  + Rest:       {rest_win_rate:.1f}%")
        print(f"  + EPA + Rest: {full_win_rate:.1f}%")
        print()

        print("Within 7 Points (betting threshold):")
        print(f"  Baseline:     {baseline_within_7_pct:.1f}%")
        print(f"  + EPA:        {epa_within_7_pct:.1f}% ({epa_within_7_pct - baseline_within_7_pct:+.1f}%)")
        print(f"  + Rest:       {rest_within_7_pct:.1f}% ({rest_within_7_pct - baseline_within_7_pct:+.1f}%)")
        print(f"  + EPA + Rest: {full_within_7_pct:.1f}% ({full_within_7_pct - baseline_within_7_pct:+.1f}%)")
        print()

        # Sample predictions
        print("Sample Predictions (Best EPA Adjustments):")
        print("-" * 80)

        # Sort by largest positive EPA improvements
        best_epa = sorted(self.comparisons, key=lambda c: -c.epa_improvement)[:5]

        for comp in best_epa:
            print(f"\n{comp.away_team} @ {comp.home_team} (Week {comp.week})")
            print(f"  Actual: {comp.actual_total:.0f}")
            print(f"  Baseline: {comp.baseline_prediction:.1f} (error: {comp.baseline_error:.1f})")
            print(f"  + EPA: {comp.epa_prediction:.1f} (error: {comp.epa_error:.1f}, adj: {comp.epa_adjustment:+.1f})")
            print(f"  Improvement: {-comp.epa_improvement:.1f} points better")

        print("\n" + "="*80)
        print("ANALYSIS")
        print("="*80 + "\n")

        # EPA effectiveness
        avg_epa_adj = np.mean([abs(c.epa_adjustment) for c in self.comparisons])
        print(f"Average EPA Adjustment: ±{avg_epa_adj:.2f} points")

        # Rest effectiveness
        avg_rest_adj = np.mean([abs(c.rest_adjustment) for c in self.comparisons])
        print(f"Average Rest Adjustment: ±{avg_rest_adj:.2f} points")

        # When is EPA most helpful?
        epa_helpful = [c for c in self.comparisons if c.epa_improvement < -1]
        if epa_helpful:
            print(f"\nEPA helpful in {len(epa_helpful)} games ({len(epa_helpful)/len(self.comparisons)*100:.1f}%)")
            avg_improvement = np.mean([-c.epa_improvement for c in epa_helpful])
            print(f"  Average improvement when helpful: {avg_improvement:.2f} points")

        # When is rest most helpful?
        rest_helpful = [c for c in self.comparisons if c.rest_improvement < -1]
        if rest_helpful:
            print(f"\nRest helpful in {len(rest_helpful)} games ({len(rest_helpful)/len(self.comparisons)*100:.1f}%)")
            avg_improvement = np.mean([-c.rest_improvement for c in rest_helpful])
            print(f"  Average improvement when helpful: {avg_improvement:.2f} points")

        print()

        # Return result
        return BacktestResult(
            feature_name="EPA + Rest Enhanced",
            seasons_tested=self.framework.seasons,
            sample_size=len(self.comparisons),
            mae=full_mae,
            rmse=np.sqrt(np.mean([c.full_error ** 2 for c in self.comparisons])),
            improvement_pct=full_improvement_pct,
            notes=[
                f"EPA improvement: {epa_improvement_pct:+.1f}%",
                f"Rest improvement: {rest_improvement_pct:+.1f}%",
                f"Combined improvement: {full_improvement_pct:+.1f}%",
                f"EPA win rate: {epa_win_rate:.1f}%",
                f"Rest win rate: {rest_win_rate:.1f}%",
                f"Full win rate: {full_win_rate:.1f}%",
            ]
        )

    def _baseline_prediction(self, game, all_games) -> float:
        """Calculate baseline prediction using simple recent averaging.

        Args:
            game: Game to predict
            all_games: All historical games

        Returns:
            Predicted total
        """
        # Get recent games for home team
        home_recent = [
            g for g in all_games
            if (g.home_team == game.home_team or g.away_team == game.home_team) and
            g.week < game.week and
            g.week >= max(1, game.week - 4) and
            g.home_score is not None and
            g.away_score is not None
        ]

        # Get recent games for away team
        away_recent = [
            g for g in all_games
            if (g.home_team == game.away_team or g.away_team == game.away_team) and
            g.week < game.week and
            g.week >= max(1, game.week - 4) and
            g.home_score is not None and
            g.away_score is not None
        ]

        if not home_recent or not away_recent:
            return 0.0

        # Calculate average scores
        home_scores = [
            g.home_score if g.home_team == game.home_team else g.away_score
            for g in home_recent
        ]
        away_scores = [
            g.home_score if g.home_team == game.away_team else g.away_score
            for g in away_recent
        ]

        home_avg = np.mean(home_scores)
        away_avg = np.mean(away_scores)

        return home_avg + away_avg

    def _calculate_rest_adjustment(self, rest_differential: int) -> float:
        """Calculate adjustment based on rest differential.

        Args:
            rest_differential: Days of rest (home - away)

        Returns:
            Points adjustment
        """
        # Thursday game (3 days rest)
        if rest_differential <= -4:  # Away team much more rested
            return -2.0  # Lower scoring
        elif rest_differential >= 4:  # Home team much more rested
            return 2.0  # Higher scoring

        # Post-bye advantage
        if rest_differential >= 7:  # Home team post-bye
            return 1.5
        elif rest_differential <= -7:  # Away team post-bye
            return -1.5

        # Minor rest differences
        if abs(rest_differential) <= 3:
            return 0.0

        # Moderate differences
        return rest_differential * 0.2  # 0.2 points per day

    def _get_rest_differential(
        self,
        home_team: str,
        away_team: str,
        week: int,
        schedule: pd.DataFrame
    ) -> int:
        """Get rest differential for a game.

        Args:
            home_team: Home team
            away_team: Away team
            week: Week number
            schedule: Schedule DataFrame

        Returns:
            Rest differential in days
        """
        if schedule is None or len(schedule) == 0:
            return 0

        # Get previous games
        home_prev = schedule[
            ((schedule['home_team'] == home_team) | (schedule['away_team'] == home_team)) &
            (schedule['week'] < week)
        ].sort_values('week').tail(1)

        away_prev = schedule[
            ((schedule['home_team'] == away_team) | (schedule['away_team'] == away_team)) &
            (schedule['week'] < week)
        ].sort_values('week').tail(1)

        if len(home_prev) == 0 or len(away_prev) == 0:
            return 0

        home_last_week = home_prev['week'].iloc[0]
        away_last_week = away_prev['week'].iloc[0]

        # Calculate days (7 days per week)
        home_rest = (week - home_last_week) * 7
        away_rest = (week - away_last_week) * 7

        return home_rest - away_rest

    def _load_schedule(self, season: int) -> Optional[pd.DataFrame]:
        """Load schedule for rest calculations.

        Args:
            season: Season year

        Returns:
            Schedule DataFrame
        """
        try:
            # Try historical games
            games = self.framework.load_historical_games(season)
            if games:
                schedule = pd.DataFrame([
                    {
                        'game_id': g.game_id,
                        'season': g.season,
                        'week': g.week,
                        'home_team': g.home_team,
                        'away_team': g.away_team,
                        'home_score': g.home_score,
                        'away_score': g.away_score
                    }
                    for g in games
                ])
                return schedule
        except Exception as e:
            print(f"Could not load schedule for {season}: {e}")

        return None


def main():
    """Run EPA enhanced backtest."""
    framework = BacktestingFramework(seasons=[2021, 2022, 2023])
    backtester = EPAEnhancedBacktester(framework)
    result = backtester.run_backtest()

    print(f"\n✅ Backtest complete: {result.feature_name}")
    print(f"   MAE: {result.mae:.2f} ({result.improvement_pct:+.1f}%)")


if __name__ == '__main__':
    main()
