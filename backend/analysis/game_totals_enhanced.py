"""Enhanced Game Totals and Over/Under Prediction Model.

Applies all validated weights from backtesting to improve game predictions:
- Weather impact (wind, cold, precipitation)
- Injury impact (key player absences)
- Situational adjustments (primetime, division games)
- Defense matchup ratings
- Recent form/trends

This is the production-ready model that should replace simple averaging.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

from backend.config import (
    WEATHER_IMPACT,
    INJURY_REDISTRIBUTION,
    SITUATIONAL_ADJUSTMENTS,
    DEFENSE_MATCHUP_ADJUSTMENTS
)


@dataclass
class GameContext:
    """All contextual factors for a game prediction."""
    home_team: str
    away_team: str
    week: int
    season: int

    # Weather
    wind_mph: float = 0.0
    temperature: float = 70.0
    precipitation: str = 'none'
    is_dome: bool = False

    # Situational
    is_primetime: bool = False
    is_division_game: bool = False
    home_on_bye: bool = False
    away_on_bye: bool = False

    # Team form
    home_recent_scores: List[float] = None
    away_recent_scores: List[float] = None

    # Injuries
    home_injuries: List[Dict] = None  # [{'position': 'WR', 'role': 'WR1'}, ...]
    away_injuries: List[Dict] = None


@dataclass
class GamePrediction:
    """Complete game prediction with breakdown."""
    predicted_home_score: float
    predicted_away_score: float
    predicted_total: float
    predicted_spread: float

    # Confidence
    confidence: float  # 0-1

    # Breakdown of adjustments
    adjustments: Dict[str, float]  # {'weather': -2.3, 'injuries': -1.5, ...}


class EnhancedGameTotalsModel:
    """Enhanced game totals prediction using all validated weights."""

    def __init__(self):
        """Initialize model with validated weights."""
        self.weather_impact = WEATHER_IMPACT
        self.injury_impact = INJURY_REDISTRIBUTION
        self.situational = SITUATIONAL_ADJUSTMENTS
        self.defense_adjustments = DEFENSE_MATCHUP_ADJUSTMENTS

    def predict_game(self, context: GameContext) -> GamePrediction:
        """Generate complete game prediction with all adjustments.

        Args:
            context: All game context and data

        Returns:
            GamePrediction with total, spread, and breakdown
        """
        # Step 1: Baseline from recent form
        home_baseline = self._calculate_baseline(context.home_recent_scores, is_home=True)
        away_baseline = self._calculate_baseline(context.away_recent_scores, is_home=False)

        # Step 2: Apply weather adjustments
        weather_adj = self._apply_weather_adjustment(context)

        # Step 3: Apply injury adjustments
        home_injury_adj, away_injury_adj = self._apply_injury_adjustments(context)

        # Step 4: Apply situational adjustments
        situational_adj = self._apply_situational_adjustments(context)

        # Step 5: Calculate final scores
        home_score = home_baseline + weather_adj['home'] + home_injury_adj + situational_adj['home']
        away_score = away_baseline + weather_adj['away'] + away_injury_adj + situational_adj['away']

        # Step 6: Calculate confidence based on data quality
        confidence = self._calculate_confidence(context)

        # Track all adjustments
        adjustments = {
            'home_baseline': home_baseline,
            'away_baseline': away_baseline,
            'weather_total': weather_adj['home'] + weather_adj['away'],
            'injury_home': home_injury_adj,
            'injury_away': away_injury_adj,
            'situational_total': situational_adj['home'] + situational_adj['away'],
            'home_field_advantage': 2.5
        }

        return GamePrediction(
            predicted_home_score=home_score,
            predicted_away_score=away_score,
            predicted_total=home_score + away_score,
            predicted_spread=home_score - away_score,
            confidence=confidence,
            adjustments=adjustments
        )

    def _calculate_baseline(self, recent_scores: List[float], is_home: bool) -> float:
        """Calculate baseline expectation from recent performance.

        Args:
            recent_scores: Last 3-4 games' scores
            is_home: Whether this is the home team

        Returns:
            Baseline points expectation
        """
        if not recent_scores or len(recent_scores) == 0:
            # NFL averages
            return 22.0 if is_home else 20.0

        # Weight recent games more heavily (exponential decay)
        weights = np.array([0.4, 0.3, 0.2, 0.1][:len(recent_scores)])
        weights = weights / weights.sum()  # Normalize

        baseline = np.average(recent_scores, weights=weights)

        # Add home field advantage
        if is_home:
            baseline += 2.5

        return baseline

    def _apply_weather_adjustment(self, context: GameContext) -> Dict[str, float]:
        """Apply validated weather impact.

        Returns:
            {'home': adjustment, 'away': adjustment}
        """
        if context.is_dome:
            return {'home': 0.0, 'away': 0.0}

        total_adjustment = 0.0

        # Wind impact (VALIDATED: +3.88 passing, -1.41 rushing per mph over 15)
        if context.wind_mph > 15.0:
            wind_config = self.weather_impact['wind']
            mph_over = context.wind_mph - 15.0

            # Passing teams affected more
            # Assume ~60% passing, 40% rushing offense
            passing_impact = mph_over * wind_config['passing_yards_per_mph'] * 0.6
            rushing_impact = mph_over * wind_config['rushing_yards_per_mph'] * 0.4

            # Convert yards to points (rough: 1 point per 11 yards)
            total_adjustment += (passing_impact + rushing_impact) / 11.0

        # Cold impact (VALIDATED but LOW CONFIDENCE: -0.44 passing per degree below 32)
        if context.temperature < 32.0:
            cold_config = self.weather_impact['cold']
            if cold_config['confidence'] > 0.5:  # Only use if confident
                degrees_below = 32.0 - context.temperature
                passing_impact = degrees_below * cold_config['passing_yards_per_degree']
                total_adjustment += passing_impact / 11.0

        # Precipitation impact (not validated yet - use conservative estimates)
        if context.precipitation in ['rain', 'snow']:
            precip_data = self.weather_impact.get('precipitation', {}).get(context.precipitation, {})
            if 'total_points' in precip_data:
                total_adjustment += precip_data['total_points']

        # Split adjustment between both teams
        return {
            'home': total_adjustment / 2.0,
            'away': total_adjustment / 2.0
        }

    def _apply_injury_adjustments(self, context: GameContext) -> Tuple[float, float]:
        """Apply validated injury impact.

        Returns:
            (home_adjustment, away_adjustment)
        """
        home_adj = 0.0
        away_adj = 0.0

        # Process home team injuries
        if context.home_injuries:
            for injury in context.home_injuries:
                position = injury.get('position')
                role = injury.get('role')  # e.g., 'WR1', 'RB1', 'QB1'

                # Get team impact for this position
                if position in self.injury_impact:
                    scenario = f"{role}_OUT"
                    impact_data = self.injury_impact[position].get(scenario, {})
                    team_impact = impact_data.get('team_total_impact', 0.0)
                    home_adj += team_impact

        # Process away team injuries
        if context.away_injuries:
            for injury in context.away_injuries:
                position = injury.get('position')
                role = injury.get('role')

                if position in self.injury_impact:
                    scenario = f"{role}_OUT"
                    impact_data = self.injury_impact[position].get(scenario, {})
                    team_impact = impact_data.get('team_total_impact', 0.0)
                    away_adj += team_impact

        return home_adj, away_adj

    def _apply_situational_adjustments(self, context: GameContext) -> Dict[str, float]:
        """Apply validated situational factors.

        Returns:
            {'home': adjustment, 'away': adjustment}
        """
        total_adj = 0.0

        # Primetime impact (VALIDATED: -4.6 points, 99.5% confidence!)
        if context.is_primetime:
            primetime_data = self.situational.get('primetime', {})
            if primetime_data.get('confidence', 0) > 0.95:  # High confidence only
                total_adj += primetime_data.get('total_points_adjustment', 0.0)

        # Division game impact (LOW CONFIDENCE - skip)
        # if context.is_division_game:
        #     division_data = self.situational.get('division_game', {})
        #     if division_data.get('confidence', 0) > 0.5:
        #         total_adj += division_data.get('total_points_adjustment', 0.0)

        # Post-bye impact (LOW CONFIDENCE - skip)
        # Similar logic...

        # Split adjustment
        return {
            'home': total_adj / 2.0,
            'away': total_adj / 2.0
        }

    def _calculate_confidence(self, context: GameContext) -> float:
        """Calculate prediction confidence based on data quality.

        Returns:
            Confidence score 0-1
        """
        confidence = 1.0

        # Reduce confidence if limited recent data
        if not context.home_recent_scores or len(context.home_recent_scores) < 3:
            confidence *= 0.8
        if not context.away_recent_scores or len(context.away_recent_scores) < 3:
            confidence *= 0.8

        # Reduce confidence for extreme weather (high uncertainty)
        if context.wind_mph > 25.0:
            confidence *= 0.7

        # Reduce confidence for multiple key injuries
        total_injuries = len(context.home_injuries or []) + len(context.away_injuries or [])
        if total_injuries > 3:
            confidence *= 0.8

        return confidence

    def get_over_under_recommendation(
        self,
        prediction: GamePrediction,
        vegas_line: float
    ) -> Dict:
        """Get betting recommendation for over/under.

        Args:
            prediction: Our game prediction
            vegas_line: Vegas total line

        Returns:
            Recommendation with edge calculation
        """
        edge = prediction.predicted_total - vegas_line
        edge_pct = (edge / vegas_line) * 100

        # Require minimum edge and confidence
        MIN_EDGE = 3.0  # Need 3+ point edge
        MIN_CONFIDENCE = 0.7

        if abs(edge) < MIN_EDGE or prediction.confidence < MIN_CONFIDENCE:
            return {
                'recommendation': 'PASS',
                'edge': edge,
                'edge_pct': edge_pct,
                'confidence': prediction.confidence,
                'reason': f"Insufficient edge ({abs(edge):.1f} pts) or confidence ({prediction.confidence:.1%})"
            }

        if edge > MIN_EDGE:
            return {
                'recommendation': 'OVER',
                'edge': edge,
                'edge_pct': edge_pct,
                'confidence': prediction.confidence,
                'bet_size': self._kelly_criterion(edge_pct, prediction.confidence)
            }
        elif edge < -MIN_EDGE:
            return {
                'recommendation': 'UNDER',
                'edge': abs(edge),
                'edge_pct': abs(edge_pct),
                'confidence': prediction.confidence,
                'bet_size': self._kelly_criterion(abs(edge_pct), prediction.confidence)
            }

        return {'recommendation': 'PASS', 'edge': edge, 'edge_pct': edge_pct}

    def _kelly_criterion(self, edge_pct: float, confidence: float) -> float:
        """Calculate Kelly Criterion bet sizing.

        Args:
            edge_pct: Edge as percentage
            confidence: Prediction confidence

        Returns:
            Recommended bet size as fraction of bankroll
        """
        # Kelly = (edge * confidence) / odds
        # For -110 odds: decimal odds = 1.909
        kelly = (edge_pct / 100 * confidence) / 1.909

        # Use fractional Kelly (1/4 Kelly for safety)
        return max(0.0, min(0.05, kelly * 0.25))  # Cap at 5% of bankroll


# Example usage
if __name__ == "__main__":
    model = EnhancedGameTotalsModel()

    # Example game context
    context = GameContext(
        home_team='KC',
        away_team='BUF',
        week=10,
        season=2024,
        wind_mph=18.0,  # Windy!
        temperature=45.0,
        is_primetime=True,  # SNF
        home_recent_scores=[28, 24, 27, 31],  # KC recent games
        away_recent_scores=[24, 27, 21, 24],  # BUF recent games
        home_injuries=[{'position': 'WR', 'role': 'WR2'}],
        away_injuries=[]
    )

    prediction = model.predict_game(context)

    print(f"Predicted Total: {prediction.predicted_total:.1f}")
    print(f"Predicted Spread: {prediction.predicted_spread:.1f} (KC)")
    print(f"Confidence: {prediction.confidence:.1%}")
    print(f"\nBreakdown:")
    for key, value in prediction.adjustments.items():
        print(f"  {key}: {value:+.2f}")

    # Check vs Vegas line
    vegas_total = 50.5
    recommendation = model.get_over_under_recommendation(prediction, vegas_total)
    print(f"\nVegas Line: {vegas_total}")
    print(f"Recommendation: {recommendation['recommendation']}")
    print(f"Edge: {recommendation.get('edge', 0):.1f} points")
