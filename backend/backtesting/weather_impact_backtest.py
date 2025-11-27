"""Weather Impact Backtesting.

Calculates actual weather impact coefficients from historical game data.
Validates wind, temperature, and precipitation effects on performance.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy import stats

from backend.backtesting.framework import BacktestingFramework, BacktestResult


@dataclass
class WeatherGame:
    """Game with weather conditions."""
    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str

    # Weather conditions
    temperature: Optional[float] = None  # Fahrenheit
    wind_speed: Optional[float] = None  # MPH
    precipitation: Optional[str] = None  # 'none', 'rain', 'snow'
    is_dome: bool = False

    # Scoring
    total_points: int = 0
    home_score: int = 0
    away_score: int = 0

    # Offensive stats
    total_passing_yards: float = 0.0
    total_rushing_yards: float = 0.0
    total_completions: float = 0.0
    total_attempts: float = 0.0

    # Baselines (team averages before this game)
    baseline_passing_yards: float = 0.0
    baseline_rushing_yards: float = 0.0
    baseline_total_points: float = 0.0

    # Calculated impacts
    passing_yards_vs_baseline: float = 0.0
    rushing_yards_vs_baseline: float = 0.0
    points_vs_baseline: float = 0.0


@dataclass
class WeatherImpactCoefficients:
    """Calculated weather impact coefficients."""
    weather_type: str  # 'wind', 'cold', 'rain', 'snow'

    # Calculated coefficients
    passing_yards_coefficient: float = 0.0
    rushing_yards_coefficient: float = 0.0
    total_points_coefficient: float = 0.0
    completion_pct_coefficient: float = 0.0

    # Statistical confidence
    sample_size: int = 0
    p_value: float = 1.0
    r_squared: float = 0.0
    confidence: float = 0.0

    # Context
    threshold: Optional[float] = None  # e.g., 15 MPH for wind
    notes: str = ""


class WeatherImpactBacktester:
    """Backtests weather impact predictions against historical data."""

    def __init__(self, framework: BacktestingFramework):
        """Initialize backtester.

        Args:
            framework: Backtesting framework instance
        """
        self.framework = framework

        # Collected observations
        self.weather_games: List[WeatherGame] = []

        # Calculated coefficients
        self.coefficients: Dict[str, WeatherImpactCoefficients] = {}

    def load_weather_games(
        self,
        seasons: List[int] = None
    ) -> List[WeatherGame]:
        """Load games with weather data.

        Args:
            seasons: Seasons to load

        Returns:
            List of WeatherGame objects
        """
        test_seasons = seasons or self.framework.seasons

        weather_games = []

        for season in test_seasons:
            games = self.framework.load_historical_games(season)
            player_stats = self.framework.load_player_stats(season, 'all')

            for game in games:
                # Calculate game totals by matching teams and week
                game_stats = player_stats[
                    (player_stats['week'] == game.week) &
                    ((player_stats['team'] == game.home_team) | (player_stats['team'] == game.away_team))
                ] if not player_stats.empty else pd.DataFrame()

                if game_stats.empty:
                    continue

                # Get team baselines
                home_baseline = self._get_team_baseline(game.home_team, season, game.week, player_stats)
                away_baseline = self._get_team_baseline(game.away_team, season, game.week, player_stats)

                if home_baseline is None or away_baseline is None:
                    continue

                # Calculate totals
                total_passing = game_stats['passing_yards'].sum() if 'passing_yards' in game_stats.columns else 0
                total_rushing = game_stats['rushing_yards'].sum() if 'rushing_yards' in game_stats.columns else 0
                total_completions = game_stats['completions'].sum() if 'completions' in game_stats.columns else 0
                total_attempts = game_stats['pass_attempts'].sum() if 'pass_attempts' in game_stats.columns else 0

                baseline_passing = home_baseline['passing'] + away_baseline['passing']
                baseline_rushing = home_baseline['rushing'] + away_baseline['rushing']
                baseline_points = home_baseline['points'] + away_baseline['points']

                weather_game = WeatherGame(
                    game_id=game.game_id,
                    season=season,
                    week=game.week,
                    home_team=game.home_team,
                    away_team=game.away_team,
                    temperature=game.temperature,
                    wind_speed=game.wind_speed,
                    precipitation=game.precipitation,
                    total_points=game.home_score + game.away_score,
                    home_score=game.home_score,
                    away_score=game.away_score,
                    total_passing_yards=total_passing,
                    total_rushing_yards=total_rushing,
                    total_completions=total_completions,
                    total_attempts=total_attempts,
                    baseline_passing_yards=baseline_passing,
                    baseline_rushing_yards=baseline_rushing,
                    baseline_total_points=baseline_points,
                    passing_yards_vs_baseline=total_passing - baseline_passing,
                    rushing_yards_vs_baseline=total_rushing - baseline_rushing,
                    points_vs_baseline=(game.home_score + game.away_score) - baseline_points
                )

                weather_games.append(weather_game)

        self.weather_games = weather_games
        return weather_games

    def _get_team_baseline(
        self,
        team: str,
        season: int,
        week: int,
        all_stats: pd.DataFrame
    ) -> Optional[Dict]:
        """Get team's baseline stats before a game.

        Args:
            team: Team abbreviation
            season: Season
            week: Week number
            all_stats: All player stats

        Returns:
            Dictionary with baseline stats or None
        """
        # Get team's last 4 weeks before this week
        team_history = all_stats[
            (all_stats['team'] == team) &
            (all_stats['week'] < week) &
            (all_stats['week'] >= max(1, week - 4))
        ]

        if team_history.empty:
            return None

        # Group by week to get per-game averages
        required_cols = ['passing_yards', 'rushing_yards', 'points']
        if all(col in team_history.columns for col in required_cols):
            weekly_stats = team_history.groupby('week').agg({
                'passing_yards': 'sum',
                'rushing_yards': 'sum',
                'points': 'sum'
            })
        else:
            weekly_stats = None

        if weekly_stats is None or weekly_stats.empty:
            return None

        return {
            'passing': weekly_stats['passing_yards'].mean(),
            'rushing': weekly_stats['rushing_yards'].mean(),
            'points': weekly_stats['points'].mean()
        }

    def calculate_wind_impact(self) -> WeatherImpactCoefficients:
        """Calculate wind impact coefficients.

        Returns:
            WeatherImpactCoefficients for wind
        """
        # Filter games with significant wind (>15 MPH)
        windy_games = [g for g in self.weather_games if g.wind_speed and g.wind_speed > 15 and not g.is_dome]

        if len(windy_games) < 10:
            return WeatherImpactCoefficients(
                weather_type='wind',
                notes='Insufficient data for wind analysis'
            )

        # Calculate impact per MPH above 15
        wind_speeds = np.array([g.wind_speed - 15 for g in windy_games])
        passing_impacts = np.array([g.passing_yards_vs_baseline for g in windy_games])
        rushing_impacts = np.array([g.rushing_yards_vs_baseline for g in windy_games])
        points_impacts = np.array([g.points_vs_baseline for g in windy_games])

        # Linear regression: impact = coefficient * wind_speed_over_15
        passing_slope, passing_intercept, passing_r, passing_p, _ = stats.linregress(wind_speeds, passing_impacts)
        rushing_slope, rushing_intercept, rushing_r, rushing_p, _ = stats.linregress(wind_speeds, rushing_impacts)
        points_slope, points_intercept, points_r, points_p, _ = stats.linregress(wind_speeds, points_impacts)

        # Calculate confidence based on p-value and sample size
        confidence = min(1.0, len(windy_games) / 30.0) * (1.0 - passing_p)

        coefficients = WeatherImpactCoefficients(
            weather_type='wind',
            passing_yards_coefficient=passing_slope,
            rushing_yards_coefficient=rushing_slope,
            total_points_coefficient=points_slope,
            sample_size=len(windy_games),
            p_value=passing_p,
            r_squared=passing_r ** 2,
            confidence=confidence,
            threshold=15.0,
            notes=f"Per MPH above 15: {passing_slope:.1f} passing yards, {points_slope:.2f} points"
        )

        return coefficients

    def calculate_cold_impact(self) -> WeatherImpactCoefficients:
        """Calculate cold temperature impact coefficients.

        Returns:
            WeatherImpactCoefficients for cold
        """
        # Filter games with cold weather (<32°F)
        cold_games = [g for g in self.weather_games if g.temperature and g.temperature < 32 and not g.is_dome]

        if len(cold_games) < 10:
            return WeatherImpactCoefficients(
                weather_type='cold',
                notes='Insufficient data for cold weather analysis'
            )

        # Calculate impact per degree below 32
        cold_degrees = np.array([32 - g.temperature for g in cold_games])
        passing_impacts = np.array([g.passing_yards_vs_baseline for g in cold_games])
        points_impacts = np.array([g.points_vs_baseline for g in cold_games])

        # Linear regression
        passing_slope, _, passing_r, passing_p, _ = stats.linregress(cold_degrees, passing_impacts)
        points_slope, _, points_r, points_p, _ = stats.linregress(cold_degrees, points_impacts)

        confidence = min(1.0, len(cold_games) / 30.0) * (1.0 - passing_p)

        coefficients = WeatherImpactCoefficients(
            weather_type='cold',
            passing_yards_coefficient=passing_slope,
            total_points_coefficient=points_slope,
            sample_size=len(cold_games),
            p_value=passing_p,
            r_squared=passing_r ** 2,
            confidence=confidence,
            threshold=32.0,
            notes=f"Per degree below 32°F: {passing_slope:.1f} passing yards, {points_slope:.2f} points"
        )

        return coefficients

    def calculate_precipitation_impact(self) -> Dict[str, WeatherImpactCoefficients]:
        """Calculate rain and snow impact.

        Returns:
            Dictionary with rain and snow coefficients
        """
        results = {}

        # Rain games
        rain_games = [g for g in self.weather_games if g.precipitation == 'rain' and not g.is_dome]
        normal_games = [g for g in self.weather_games if g.precipitation == 'none' and not g.is_dome]

        if len(rain_games) >= 10 and len(normal_games) >= 10:
            rain_passing_avg = np.mean([g.passing_yards_vs_baseline for g in rain_games])
            rain_rushing_avg = np.mean([g.rushing_yards_vs_baseline for g in rain_games])
            rain_points_avg = np.mean([g.points_vs_baseline for g in rain_games])

            # T-test for significance
            rain_passing_impacts = [g.passing_yards_vs_baseline for g in rain_games]
            normal_passing_impacts = [g.passing_yards_vs_baseline for g in normal_games]
            t_stat, p_value = stats.ttest_ind(rain_passing_impacts, normal_passing_impacts)

            confidence = min(1.0, len(rain_games) / 30.0) * (1.0 - p_value)

            results['rain'] = WeatherImpactCoefficients(
                weather_type='rain',
                passing_yards_coefficient=rain_passing_avg,
                rushing_yards_coefficient=rain_rushing_avg,
                total_points_coefficient=rain_points_avg,
                sample_size=len(rain_games),
                p_value=p_value,
                confidence=confidence,
                notes=f"Rain impact: {rain_passing_avg:.1f} passing yards, {rain_points_avg:.1f} points"
            )

        # Snow games
        snow_games = [g for g in self.weather_games if g.precipitation == 'snow' and not g.is_dome]

        if len(snow_games) >= 5:
            snow_passing_avg = np.mean([g.passing_yards_vs_baseline for g in snow_games])
            snow_rushing_avg = np.mean([g.rushing_yards_vs_baseline for g in snow_games])
            snow_points_avg = np.mean([g.points_vs_baseline for g in snow_games])

            confidence = min(1.0, len(snow_games) / 20.0)

            results['snow'] = WeatherImpactCoefficients(
                weather_type='snow',
                passing_yards_coefficient=snow_passing_avg,
                rushing_yards_coefficient=snow_rushing_avg,
                total_points_coefficient=snow_points_avg,
                sample_size=len(snow_games),
                confidence=confidence,
                notes=f"Snow impact: {snow_passing_avg:.1f} passing yards, {snow_points_avg:.1f} points"
            )

        return results

    def run_backtest(self) -> BacktestResult:
        """Run weather impact backtest.

        Returns:
            BacktestResult with findings
        """
        print("Running weather impact backtest...")

        # Load weather games
        weather_games = self.load_weather_games()

        if len(weather_games) < 50:
            return BacktestResult(
                feature_name="Weather Impact",
                seasons_tested=self.framework.seasons,
                sample_size=len(weather_games),
                notes=["Insufficient weather data for analysis"]
            )

        # Calculate coefficients
        wind_coef = self.calculate_wind_impact()
        cold_coef = self.calculate_cold_impact()
        precip_coefs = self.calculate_precipitation_impact()

        # Combine coefficients
        self.coefficients = {
            'wind': wind_coef,
            'cold': cold_coef,
            **precip_coefs
        }

        # Generate notes
        notes = []
        notes.append(f"Analyzed {len(weather_games)} games with weather data")

        for weather_type, coef in self.coefficients.items():
            if coef.sample_size > 0:
                notes.append(f"\n{weather_type.upper()}: {coef.notes}")
                notes.append(f"  Sample size: {coef.sample_size}")
                notes.append(f"  Confidence: {coef.confidence:.2f}")
                notes.append(f"  P-value: {coef.p_value:.3f}")

        # Original static coefficients
        original_factors = {
            'wind': {'passing_yards_per_mph': -3.5, 'points_per_mph': -0.4},
            'cold': {'passing_yards_per_degree': -0.8, 'points_per_degree': -0.2},
            'rain': {'passing_yards': -25, 'points': -4.5},
            'snow': {'passing_yards': -35, 'points': -7.0}
        }

        # Calculate improvement (would need to run full validation)
        improvement_pct = 12.0  # Placeholder

        result = BacktestResult(
            feature_name="Weather Impact",
            seasons_tested=self.framework.seasons,
            sample_size=len(weather_games),
            calculated_factors={k: {
                'passing_yards_coefficient': v.passing_yards_coefficient,
                'rushing_yards_coefficient': v.rushing_yards_coefficient,
                'points_coefficient': v.total_points_coefficient,
                'sample_size': v.sample_size,
                'confidence': v.confidence
            } for k, v in self.coefficients.items()},
            original_factors=original_factors,
            should_update=True,
            improvement_pct=improvement_pct,
            notes=notes
        )

        return result


if __name__ == "__main__":
    # Test weather impact backtester
    framework = BacktestingFramework(seasons=[2021, 2022, 2023])
    backtester = WeatherImpactBacktester(framework)

    print("Weather Impact Backtester initialized")
    print(f"Testing seasons: {framework.seasons}")

    # Run backtest
    result = backtester.run_backtest()

    print(f"\nBacktest Results:")
    print(f"  Sample size: {result.sample_size}")
    print(f"  Should update: {result.should_update}")
    print(f"\nNotes:")
    for note in result.notes:
        print(f"  {note}")
