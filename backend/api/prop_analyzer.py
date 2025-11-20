"""Prop value analyzer for finding +EV betting opportunities.

This module analyzes prop lines from sportsbooks and compares them to
model projections to identify value plays.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from math import erf, sqrt

# Import calibration module
try:
    from backend.betting.probability_calibration import calibrate_probability, probability_calibrator
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False
    def calibrate_probability(prop_type, prob):
        return prob

# Import distribution models for prop-type-specific probability calculations
try:
    from backend.modeling.distributions import (
        get_distribution,
        calculate_hit_probability,
        PROP_DISTRIBUTION_MAP,
        DEFAULT_STD_BY_PROP
    )
    DISTRIBUTIONS_AVAILABLE = True
except ImportError:
    DISTRIBUTIONS_AVAILABLE = False


class PropType(Enum):
    """Types of prop bets."""
    PASSING_YARDS = "passing_yards"
    RUSHING_YARDS = "rushing_yards"
    RECEIVING_YARDS = "receiving_yards"
    RECEPTIONS = "receptions"
    PASSING_TDS = "passing_tds"
    RUSHING_TDS = "rushing_tds"
    TOUCHDOWNS = "touchdowns_scored"


@dataclass
class PropLine:
    """Sportsbook prop line."""
    player_id: str
    player_name: str
    prop_type: str
    line: float
    over_odds: int  # American odds (e.g., -110)
    under_odds: int
    book: str
    timestamp: str


@dataclass
class PropProjection:
    """Model projection for a prop."""
    player_id: str
    player_name: str
    prop_type: str
    projection: float
    std_dev: float  # Standard deviation
    confidence_interval: Tuple[float, float]  # 80% CI
    hit_probability_over: float  # Probability of going over the line
    hit_probability_under: float
    # Quality metrics for filtering
    games_sampled: int = 0  # Number of games in training sample
    model_quality: float = 0.0  # Model R² or accuracy score
    usage_metric: float = 0.0  # Usage % for RB/WR (snap %, target share)
    position: str = ""  # Player position


@dataclass
class PropValue:
    """Value assessment for a prop."""
    prop_line: PropLine
    projection: PropProjection
    edge_over: float  # +EV percentage on over
    edge_under: float  # +EV percentage on under
    recommendation: str  # "OVER", "UNDER", or "PASS"
    confidence: float
    value_grade: str  # "A+", "A", "B+", "B", "C", "F"


class PropAnalyzer:
    """Analyze props for value."""

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        min_games_sampled: int = 5,
        min_model_quality: float = 0.3,
        min_usage_threshold: float = 0.15
    ):
        """Initialize prop analyzer with quality thresholds.

        Args:
            kelly_fraction: Fraction of Kelly criterion to use (default 0.25 for quarter Kelly)
            min_games_sampled: Minimum games in training sample
            min_model_quality: Minimum model R²/accuracy score
            min_usage_threshold: Minimum usage % for RB/WR props
        """
        self.kelly_fraction = kelly_fraction
        self.min_games_sampled = min_games_sampled
        self.min_model_quality = min_model_quality
        self.min_usage_threshold = min_usage_threshold

    def passes_quality_gates(self, projection: PropProjection) -> Tuple[bool, List[str]]:
        """Check if a projection passes quality gates.

        Args:
            projection: PropProjection to evaluate

        Returns:
            Tuple of (passes: bool, reasons: list of failed gates)
        """
        reasons = []

        # Gate 1: Minimum games sampled
        if projection.games_sampled > 0 and projection.games_sampled < self.min_games_sampled:
            reasons.append(f"Insufficient sample: {projection.games_sampled} games < {self.min_games_sampled} min")

        # Gate 2: Model quality
        if projection.model_quality > 0 and projection.model_quality < self.min_model_quality:
            reasons.append(f"Low model quality: {projection.model_quality:.2f} < {self.min_model_quality} min")

        # Gate 3: Usage threshold (for RB/WR skill position props)
        usage_props = ['rushing_yards', 'receiving_yards', 'receptions', 'targets', 'carries']
        if projection.prop_type in usage_props:
            if projection.usage_metric > 0 and projection.usage_metric < self.min_usage_threshold:
                reasons.append(f"Low usage: {projection.usage_metric:.1%} < {self.min_usage_threshold:.1%} min")

        # Gate 4: Sanity check on projection value
        if projection.projection <= 0:
            reasons.append(f"Invalid projection: {projection.projection}")

        passes = len(reasons) == 0
        return passes, reasons

    @staticmethod
    def american_to_implied_probability(odds: int) -> float:
        """Convert American odds to implied probability.

        Args:
            odds: American odds (e.g., -110, +150)

        Returns:
            Implied probability as decimal (0-1)
        """
        if odds < 0:
            return abs(odds) / (abs(odds) + 100)
        else:
            return 100 / (odds + 100)

    @staticmethod
    def calculate_edge(true_probability: float, implied_probability: float) -> float:
        """Calculate edge (expected value).

        Args:
            true_probability: Model's probability
            implied_probability: Market's probability

        Returns:
            Edge as percentage
        """
        return (true_probability - implied_probability) / implied_probability * 100

    def estimate_hit_probability(
        self,
        projection: float,
        line: float,
        std_dev: float,
        prop_type: Optional[str] = None,
        apply_calibration: bool = True
    ) -> float:
        """Estimate probability of hitting over/under using appropriate distribution.

        Uses Normal distribution for yardage props (continuous) and
        Poisson distribution for count props (receptions, TDs, etc.).

        Args:
            projection: Model projection
            line: Sportsbook line
            std_dev: Standard deviation of projection
            prop_type: Type of prop (for distribution selection and calibration)
            apply_calibration: Whether to apply probability calibration

        Returns:
            Probability of going over the line (0-1)
        """
        if std_dev == 0:
            return 1.0 if projection > line else 0.0

        # Use distribution models if available (Normal for yards, Poisson for counts)
        if DISTRIBUTIONS_AVAILABLE and prop_type:
            probability_over = calculate_hit_probability(
                prop_type=prop_type,
                projection=projection,
                line=line,
                std_dev=std_dev,
                side='over'
            )
        else:
            # Fallback to simple normal CDF
            z_score = (line - projection) / std_dev
            probability_under = 0.5 * (1 + erf(z_score / sqrt(2)))
            probability_over = 1 - probability_under

        # Apply calibration if available and requested
        if apply_calibration and CALIBRATION_AVAILABLE and prop_type:
            probability_over = calibrate_probability(prop_type, probability_over)

        return probability_over

    def analyze_prop(self, prop_line: PropLine, projection: PropProjection) -> PropValue:
        """Analyze a single prop for value.

        Args:
            prop_line: Sportsbook line
            projection: Model projection

        Returns:
            PropValue assessment
        """
        # Calculate hit probabilities with calibration
        prob_over = self.estimate_hit_probability(
            projection.projection,
            prop_line.line,
            projection.std_dev,
            prop_type=prop_line.prop_type,
            apply_calibration=True
        )
        prob_under = 1 - prob_over

        # Get implied probabilities from odds
        implied_prob_over = self.american_to_implied_probability(prop_line.over_odds)
        implied_prob_under = self.american_to_implied_probability(prop_line.under_odds)

        # Calculate edges
        edge_over = self.calculate_edge(prob_over, implied_prob_over)
        edge_under = self.calculate_edge(prob_under, implied_prob_under)

        # Determine recommendation
        min_edge_threshold = 3.0  # Minimum 3% edge to recommend
        if edge_over >= min_edge_threshold and edge_over > edge_under:
            recommendation = "OVER"
            confidence = min(prob_over, 0.95)
            primary_edge = edge_over
        elif edge_under >= min_edge_threshold and edge_under > edge_over:
            recommendation = "UNDER"
            confidence = min(prob_under, 0.95)
            primary_edge = edge_under
        else:
            recommendation = "PASS"
            confidence = 0.5
            primary_edge = max(edge_over, edge_under)

        # Grade the value
        if primary_edge >= 15:
            value_grade = "A+"
        elif primary_edge >= 10:
            value_grade = "A"
        elif primary_edge >= 7:
            value_grade = "B+"
        elif primary_edge >= 5:
            value_grade = "B"
        elif primary_edge >= 3:
            value_grade = "C"
        else:
            value_grade = "F"

        return PropValue(
            prop_line=prop_line,
            projection=projection,
            edge_over=edge_over,
            edge_under=edge_under,
            recommendation=recommendation,
            confidence=confidence,
            value_grade=value_grade
        )

    def find_best_props(
        self,
        prop_lines: List[PropLine],
        projections: List[PropProjection],
        min_edge: float = 5.0,
        min_grade: str = "B",
        apply_quality_filters: bool = True
    ) -> List[PropValue]:
        """Find best value props from a set of lines.

        Args:
            prop_lines: List of sportsbook lines
            projections: List of model projections
            min_edge: Minimum edge to include
            min_grade: Minimum value grade to include
            apply_quality_filters: Whether to apply quality gate filters

        Returns:
            List of PropValue objects, sorted by edge
        """
        # Create projection lookup
        proj_lookup = {
            (p.player_id, p.prop_type): p
            for p in projections
        }

        value_props = []
        filtered_count = 0

        for line in prop_lines:
            key = (line.player_id, line.prop_type)
            if key not in proj_lookup:
                continue

            projection = proj_lookup[key]

            # Apply quality gates if enabled
            if apply_quality_filters:
                passes, reasons = self.passes_quality_gates(projection)
                if not passes:
                    filtered_count += 1
                    continue

            value = self.analyze_prop(line, projection)

            # Filter by criteria
            grade_order = ["F", "C", "B", "B+", "A", "A+"]
            if value.value_grade in grade_order[grade_order.index(min_grade):]:
                if max(value.edge_over, value.edge_under) >= min_edge:
                    value_props.append(value)

        # Sort by primary edge (descending)
        value_props.sort(
            key=lambda v: max(v.edge_over, v.edge_under),
            reverse=True
        )

        if filtered_count > 0:
            print(f"Quality filters removed {filtered_count} props")

        return value_props

    def calculate_kelly_stake(self, edge: float, confidence: float, bankroll: float) -> float:
        """Calculate Kelly criterion stake size.

        Args:
            edge: Edge percentage
            confidence: Confidence in the bet (0-1)
            bankroll: Total bankroll

        Returns:
            Recommended stake amount
        """
        # Convert edge to decimal
        edge_decimal = edge / 100

        # Full Kelly
        kelly = edge_decimal / (1 - confidence) if confidence < 1 else edge_decimal

        # Apply Kelly fraction for conservative sizing
        fractional_kelly = kelly * self.kelly_fraction

        # Cap at 5% of bankroll for safety
        max_stake_pct = 0.05
        stake_pct = min(fractional_kelly, max_stake_pct)

        return bankroll * stake_pct


# Singleton instance
prop_analyzer = PropAnalyzer(kelly_fraction=0.25)
