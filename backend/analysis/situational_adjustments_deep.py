"""Situational Adjustments Deep Analysis System.

Quantifies impact of weather, primetime, division games, and other situational
factors on player projections and team totals.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pandas as pd
from datetime import datetime

# Import validated weights from backtesting
from backend.config import WEATHER_IMPACT, SITUATIONAL_ADJUSTMENTS


@dataclass
class WeatherImpact:
    """Impact of weather conditions."""
    temperature: Optional[float] = None  # Fahrenheit
    wind_mph: Optional[float] = None
    precipitation: Optional[str] = None  # 'none', 'rain', 'snow'
    is_dome: bool = False

    # Calculated impacts
    total_adjustment: float = 0.0
    passing_yards_adjustment: float = 0.0
    rushing_yards_adjustment: float = 0.0
    completion_pct_adjustment: float = 0.0
    kicking_range_adjustment: float = 0.0

    # Severity
    is_significant: bool = False
    severity: str = "normal"  # normal, moderate, severe, extreme

    def calculate_impacts(self):
        """Calculate all weather impacts using validated coefficients."""
        if self.is_dome:
            # Dome game - slight boost to passing
            self.total_adjustment = 1.5
            self.passing_yards_adjustment = 8.0
            return

        # Get validated weather coefficients
        wind_config = WEATHER_IMPACT.get('wind', {})
        cold_config = WEATHER_IMPACT.get('cold', {})
        precip_config = WEATHER_IMPACT.get('precipitation', {})

        # Wind impact (most significant for passing)
        wind_threshold = wind_config.get('threshold_mph', 15.0)
        if self.wind_mph and self.wind_mph > wind_threshold:
            wind_over_threshold = self.wind_mph - wind_threshold

            # Use validated coefficients
            passing_coef = wind_config.get('passing_yards_per_mph', -3.5)
            points_coef = wind_config.get('total_points_per_mph', -0.4)
            completion_coef = wind_config.get('completion_pct_per_mph', -0.5)

            self.total_adjustment += wind_over_threshold * points_coef
            self.passing_yards_adjustment += wind_over_threshold * passing_coef
            self.completion_pct_adjustment += wind_over_threshold * completion_coef
            self.kicking_range_adjustment -= wind_over_threshold * 0.8

            if self.wind_mph > 25:
                self.severity = "extreme"
                self.is_significant = True
            elif self.wind_mph > 20:
                self.severity = "severe"
                self.is_significant = True
            else:
                self.severity = "moderate"
                self.is_significant = True

        # Cold weather impact
        cold_threshold = cold_config.get('threshold_fahrenheit', 32.0)
        if self.temperature and self.temperature < cold_threshold:
            cold_below_threshold = cold_threshold - self.temperature

            # Use validated coefficients
            passing_coef = cold_config.get('passing_yards_per_degree', -0.8)
            points_coef = cold_config.get('total_points_per_degree', -0.2)

            self.total_adjustment += cold_below_threshold * points_coef
            self.passing_yards_adjustment += cold_below_threshold * passing_coef
            self.completion_pct_adjustment -= cold_below_threshold * 0.15

            if self.temperature < 10:
                self.severity = "extreme"
                self.is_significant = True
            elif self.temperature < 20:
                self.severity = "severe"
                self.is_significant = True

        # Precipitation impact
        if self.precipitation and self.precipitation != 'none':
            rain_config = precip_config.get('rain', {})
            snow_config = precip_config.get('snow', {})

            if self.precipitation == 'rain':
                # Use validated coefficients
                self.total_adjustment += rain_config.get('total_points', -4.5)
                self.passing_yards_adjustment += rain_config.get('passing_yards', -25)
                self.rushing_yards_adjustment += rain_config.get('rushing_yards', 10)
                self.completion_pct_adjustment += rain_config.get('completion_pct', -4.0)
                self.severity = "moderate" if self.severity == "normal" else self.severity
                self.is_significant = True

            elif self.precipitation == 'snow':
                # Use validated coefficients
                self.total_adjustment += snow_config.get('total_points', -7.0)
                self.passing_yards_adjustment += snow_config.get('passing_yards', -35)
                self.rushing_yards_adjustment += snow_config.get('rushing_yards', 5)
                self.completion_pct_adjustment += snow_config.get('completion_pct', -6.0)
                self.kicking_range_adjustment -= 5.0
                self.severity = "severe"
                self.is_significant = True


@dataclass
class SituationalFactors:
    """Collection of situational factors for a game."""
    # Game context
    is_primetime: bool = False
    is_division_game: bool = False
    is_playoff_game: bool = False
    is_home: bool = True

    # Schedule factors
    days_rest: int = 7
    after_bye: bool = False
    short_week: bool = False  # Thursday game

    # Travel
    is_london_game: bool = False
    timezone_change: int = 0  # Hours of timezone change

    # Revenge/motivation
    is_revenge_game: bool = False

    # Weather
    weather: Optional[WeatherImpact] = None


@dataclass
class SituationalAdjustment:
    """Calculated adjustment from situational factors."""
    factor_type: str
    description: str

    # Impacts
    total_adjustment: float = 0.0
    passing_adjustment: float = 0.0
    rushing_adjustment: float = 0.0
    receiving_adjustment: float = 0.0

    # Target/usage impacts
    target_adjustment_pct: float = 0.0
    carry_adjustment_pct: float = 0.0

    confidence: float = 0.7


class SituationalAdjustmentAnalyzer:
    """Analyzes situational factors and calculates projection adjustments."""

    def __init__(self, season: int = 2025):
        """Initialize analyzer.

        Args:
            season: NFL season year
        """
        self.season = season
        self.inputs_dir = Path('inputs')

        # Historical situational impacts (would be calculated from historical data)
        self.historical_impacts = self._initialize_historical_impacts()

    def _initialize_historical_impacts(self) -> Dict:
        """Initialize historical impact data.

        Returns:
            Dictionary of historical impacts
        """
        return {
            'primetime': {
                'star_player_boost': 1.08,  # 8% boost for star players
                'total_adjustment': 1.5,
                'confidence': 0.75
            },
            'division_game': {
                'total_adjustment': -2.5,  # Lower scoring
                'margin_tighter': 0.85,  # 15% tighter margins
                'variance_lower': 0.9,
                'confidence': 0.80
            },
            'after_bye': {
                'qb_completion_boost': 2.5,  # +2.5% completion
                'total_adjustment': 1.2,
                'confidence': 0.65
            },
            'short_week': {
                'total_adjustment': -3.0,  # Thursday games lower scoring
                'qb_yards_adjustment': -15,
                'confidence': 0.70
            },
            'london_game': {
                'total_adjustment': -4.0,  # International games unpredictable
                'variance_higher': 1.15,
                'confidence': 0.60
            },
            'revenge_game': {
                'rb_usage_boost': 1.12,  # 12% more carries
                'emotional_factor': 1.05,
                'confidence': 0.50
            }
        }

    def analyze_situation(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        week: int,
        situational_factors: SituationalFactors
    ) -> List[SituationalAdjustment]:
        """Analyze situational factors and generate adjustments.

        Args:
            game_id: Game identifier
            home_team: Home team
            away_team: Away team
            week: Week number
            situational_factors: Situational factors for the game

        Returns:
            List of SituationalAdjustment objects
        """
        adjustments = []

        # Weather impacts
        if situational_factors.weather:
            situational_factors.weather.calculate_impacts()

            if situational_factors.weather.is_significant:
                adjustments.append(SituationalAdjustment(
                    factor_type='weather',
                    description=f"Weather: {situational_factors.weather.severity.upper()} conditions",
                    total_adjustment=situational_factors.weather.total_adjustment,
                    passing_adjustment=situational_factors.weather.passing_yards_adjustment,
                    rushing_adjustment=situational_factors.weather.rushing_yards_adjustment,
                    confidence=0.80 if situational_factors.weather.severity == 'severe' else 0.65
                ))

        # Primetime boost
        if situational_factors.is_primetime:
            impacts = self.historical_impacts['primetime']
            adjustments.append(SituationalAdjustment(
                factor_type='primetime',
                description="Primetime game - star players elevated",
                total_adjustment=impacts['total_adjustment'],
                target_adjustment_pct=(impacts['star_player_boost'] - 1) * 100,
                confidence=impacts['confidence']
            ))

        # Division game
        if situational_factors.is_division_game:
            impacts = self.historical_impacts['division_game']
            adjustments.append(SituationalAdjustment(
                factor_type='division',
                description="Division game - tighter, lower scoring",
                total_adjustment=impacts['total_adjustment'],
                confidence=impacts['confidence']
            ))

        # After bye week
        if situational_factors.after_bye:
            impacts = self.historical_impacts['after_bye']
            adjustments.append(SituationalAdjustment(
                factor_type='bye_week',
                description="After bye week - rested and prepared",
                total_adjustment=impacts['total_adjustment'],
                passing_adjustment=10.0,  # Better prep = better passing
                confidence=impacts['confidence']
            ))

        # Short week (Thursday)
        if situational_factors.short_week:
            impacts = self.historical_impacts['short_week']
            adjustments.append(SituationalAdjustment(
                factor_type='short_week',
                description="Thursday game - limited prep, lower scoring",
                total_adjustment=impacts['total_adjustment'],
                passing_adjustment=impacts['qb_yards_adjustment'],
                confidence=impacts['confidence']
            ))

        # London game
        if situational_factors.is_london_game:
            impacts = self.historical_impacts['london_game']
            adjustments.append(SituationalAdjustment(
                factor_type='london',
                description="International game - travel fatigue",
                total_adjustment=impacts['total_adjustment'],
                confidence=impacts['confidence']
            ))

        # Revenge game
        if situational_factors.is_revenge_game:
            impacts = self.historical_impacts['revenge_game']
            adjustments.append(SituationalAdjustment(
                factor_type='revenge',
                description="Revenge game - emotional motivation",
                carry_adjustment_pct=(impacts['rb_usage_boost'] - 1) * 100,
                confidence=impacts['confidence']
            ))

        return adjustments

    def apply_adjustments_to_projection(
        self,
        base_projection: float,
        stat_type: str,
        adjustments: List[SituationalAdjustment]
    ) -> Tuple[float, List[str]]:
        """Apply situational adjustments to a projection.

        Args:
            base_projection: Base projection value
            stat_type: Type of stat ('passing_yards', 'receiving_yards', etc.)
            adjustments: List of situational adjustments

        Returns:
            (adjusted_projection, list_of_reasons)
        """
        adjusted = base_projection
        reasons = []

        for adjustment in adjustments:
            # Apply relevant adjustment based on stat type
            if stat_type == 'passing_yards' and adjustment.passing_adjustment != 0:
                adjusted += adjustment.passing_adjustment
                reasons.append(f"{adjustment.description}: {adjustment.passing_adjustment:+.0f} yards")

            elif stat_type == 'rushing_yards' and adjustment.rushing_adjustment != 0:
                adjusted += adjustment.rushing_adjustment
                reasons.append(f"{adjustment.description}: {adjustment.rushing_adjustment:+.0f} yards")

            elif stat_type == 'receiving_yards' and adjustment.receiving_adjustment != 0:
                adjusted += adjustment.receiving_adjustment
                reasons.append(f"{adjustment.description}: {adjustment.receiving_adjustment:+.0f} yards")

            # Apply percentage adjustments
            if stat_type in ['receiving_yards', 'receptions'] and adjustment.target_adjustment_pct != 0:
                pct_change = adjustment.target_adjustment_pct / 100.0
                adjusted *= (1 + pct_change)
                reasons.append(f"{adjustment.description}: {adjustment.target_adjustment_pct:+.0f}% usage")

            elif stat_type == 'rushing_yards' and adjustment.carry_adjustment_pct != 0:
                pct_change = adjustment.carry_adjustment_pct / 100.0
                adjusted *= (1 + pct_change)
                reasons.append(f"{adjustment.description}: {adjustment.carry_adjustment_pct:+.0f}% usage")

        return adjusted, reasons

    def get_game_situational_factors(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        week: int,
        is_home_team: bool
    ) -> SituationalFactors:
        """Get situational factors for a specific game.

        Args:
            game_id: Game identifier
            home_team: Home team
            away_team: Away team
            week: Week number
            is_home_team: Whether analyzing home team

        Returns:
            SituationalFactors object
        """
        factors = SituationalFactors(is_home=is_home_team)

        # Check if primetime (would query schedule data)
        # For now, simplified logic
        factors.is_primetime = self._is_primetime_game(game_id, week)

        # Check if division game
        factors.is_division_game = self._is_division_game(home_team, away_team)

        # Check if bye week
        factors.after_bye = self._had_bye_week(home_team if is_home_team else away_team, week)

        # Check if short week (Thursday)
        factors.short_week = self._is_short_week(week)

        # Get weather (would query weather API)
        factors.weather = self._get_weather_data(home_team, week)

        return factors

    def _is_primetime_game(self, game_id: str, week: int) -> bool:
        """Check if game is primetime."""
        # Would check schedule data
        # SNF, MNF, TNF games
        return False  # Placeholder

    def _is_division_game(self, home_team: str, away_team: str) -> bool:
        """Check if game is division matchup."""
        divisions = {
            'AFC East': ['BUF', 'MIA', 'NE', 'NYJ'],
            'AFC North': ['BAL', 'CIN', 'CLE', 'PIT'],
            'AFC South': ['HOU', 'IND', 'JAX', 'TEN'],
            'AFC West': ['DEN', 'KC', 'LV', 'LAC'],
            'NFC East': ['DAL', 'NYG', 'PHI', 'WAS'],
            'NFC North': ['CHI', 'DET', 'GB', 'MIN'],
            'NFC South': ['ATL', 'CAR', 'NO', 'TB'],
            'NFC West': ['ARI', 'LAR', 'SEA', 'SF']
        }

        for division, teams in divisions.items():
            if home_team in teams and away_team in teams:
                return True

        return False

    def _had_bye_week(self, team: str, current_week: int) -> bool:
        """Check if team just had a bye week."""
        # Would check schedule data
        return False  # Placeholder

    def _is_short_week(self, week: int) -> bool:
        """Check if this is a Thursday game."""
        # Would check schedule/day of week
        return False  # Placeholder

    def _get_weather_data(self, home_team: str, week: int) -> WeatherImpact:
        """Get weather data for game location.

        Args:
            home_team: Home team (determines stadium/location)
            week: Week number

        Returns:
            WeatherImpact object
        """
        # Check if dome team
        dome_teams = ['ATL', 'DET', 'NO', 'MIN', 'DAL', 'LV', 'LAR', 'ARI', 'IND']

        if home_team in dome_teams:
            return WeatherImpact(is_dome=True)

        # Would query weather API for outdoor games
        # For now, return default
        return WeatherImpact(
            temperature=55.0,
            wind_mph=8.0,
            precipitation='none',
            is_dome=False
        )


# Singleton instance
situational_adjustment_analyzer = SituationalAdjustmentAnalyzer()


if __name__ == "__main__":
    # Test analyzer
    analyzer = SituationalAdjustmentAnalyzer(season=2025)

    # Create test situational factors
    weather = WeatherImpact(
        temperature=28.0,
        wind_mph=22.0,
        precipitation='snow',
        is_dome=False
    )

    factors = SituationalFactors(
        is_primetime=True,
        is_division_game=True,
        weather=weather
    )

    # Analyze situation
    adjustments = analyzer.analyze_situation(
        game_id="2025_12_BUF_KC",
        home_team="KC",
        away_team="BUF",
        week=12,
        situational_factors=factors
    )

    print("Situational Adjustments:")
    for adj in adjustments:
        print(f"\n{adj.factor_type.upper()}: {adj.description}")
        if adj.total_adjustment != 0:
            print(f"  Total adjustment: {adj.total_adjustment:+.1f} points")
        if adj.passing_adjustment != 0:
            print(f"  Passing yards: {adj.passing_adjustment:+.0f}")
        if adj.rushing_adjustment != 0:
            print(f"  Rushing yards: {adj.rushing_adjustment:+.0f}")
        print(f"  Confidence: {adj.confidence:.0%}")

    # Test projection adjustment
    base_qb_yards = 285.0
    adjusted_yards, reasons = analyzer.apply_adjustments_to_projection(
        base_qb_yards, 'passing_yards', adjustments
    )

    print(f"\nQB Projection Adjustment:")
    print(f"Base: {base_qb_yards:.0f} yards")
    print(f"Adjusted: {adjusted_yards:.0f} yards ({adjusted_yards - base_qb_yards:+.0f})")
    print("Reasons:")
    for reason in reasons:
        print(f"  - {reason}")
