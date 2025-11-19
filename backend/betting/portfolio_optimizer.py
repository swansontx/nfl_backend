"""Portfolio optimizer for correlation-aware parlay construction.

This module implements "netting" - the process of constructing optimal parlays
that account for correlation between props, game scripts, and portfolio-level risk.

Key concepts:
1. **Correlation Grouping**: Props in the same game are correlated
2. **Game Script Scenarios**: Different game flows favor different props
3. **Marginal Contribution**: Each prop's incremental value to portfolio
4. **Risk-Adjusted Sizing**: Kelly criterion with correlation adjustments

This is THE feature that separates amateur from professional parlay construction.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict


class GameScript(Enum):
    """Game script scenarios that affect prop outcomes."""
    SHOOTOUT = "shootout"          # High-scoring, lots of passes
    DEFENSIVE_SLOG = "defensive"   # Low-scoring, grinding
    RB_DOMINATION = "rb_game"      # Run-heavy, clock control
    BLOWOUT_FAVORITE = "blowout_fav"  # Favorite dominates early
    BLOWOUT_UNDERDOG = "blowout_dog"  # Underdog dominates (rare)


@dataclass
class PropValue:
    """A single prop with value metrics.

    This extends the basic PropValue from prop_analyzer with portfolio fields.
    """
    # Identity
    player_id: str
    player_name: str
    game_id: str
    team: str
    opponent: str

    # Prop details
    prop_type: str  # e.g., "player_pass_yds", "player_receptions"
    line: float
    side: str  # "over" or "under"
    odds: int   # American odds

    # Model outputs
    projection: float
    hit_probability: float  # P(prop hits)
    edge: float             # Model prob - implied prob
    ev: float              # Expected value (as %)

    # Context
    is_home: bool
    spread: float
    total: float

    # Quality metrics
    games_sampled: int = 5
    model_r2: float = 0.0
    trust_score: float = 0.0


@dataclass
class CorrelationGroup:
    """A group of correlated props."""
    game_id: str
    props: List[PropValue]
    correlation_theme: str  # e.g., "same_team_passing", "opposing_qbs"
    estimated_correlation: float  # -1 to 1


@dataclass
class ParlaySuggestion:
    """A suggested parlay with risk-adjusted metrics."""
    legs: List[PropValue]
    combined_odds: int
    combined_probability: float  # Correlation-adjusted
    raw_probability: float       # If independent
    ev: float                    # Expected value
    recommended_stake_pct: float # % of bankroll (Kelly-adjusted)
    confidence: str              # "HIGH", "MEDIUM", "LOW"
    correlation_adjustment: float  # How much correlation reduced EV
    scenarios: Dict[GameScript, float]  # Probability by scenario


def group_props_by_correlation(
    props: List[PropValue],
    max_correlation_threshold: float = 0.7
) -> Dict[str, List[CorrelationGroup]]:
    """Group props into correlation buckets.

    Args:
        props: List of prop values
        max_correlation_threshold: Max correlation to allow in same parlay

    Returns:
        Dict mapping group_type -> list of correlation groups:
        - "independent": Props from different games/scripts
        - "correlated_same_team": Same-team props (QB + WR)
        - "correlated_opposing": Opposing props (both QBs)
        - "correlated_game": Same-game but different sides
    """
    # Group by game first
    by_game = defaultdict(list)
    for prop in props:
        by_game[prop.game_id].append(prop)

    groups = {
        "independent": [],
        "correlated_same_team": [],
        "correlated_opposing": [],
        "correlated_game": []
    }

    # Props from different games are independent
    if len(by_game) > 1:
        for game_id, game_props in by_game.items():
            if len(game_props) == 1:
                groups["independent"].append(CorrelationGroup(
                    game_id=game_id,
                    props=game_props,
                    correlation_theme="different_game",
                    estimated_correlation=0.0
                ))

    # Analyze within-game correlations
    for game_id, game_props in by_game.items():
        if len(game_props) < 2:
            continue

        # Group by team
        by_team = defaultdict(list)
        for prop in game_props:
            by_team[prop.team].append(prop)

        # Same-team correlations (QB + WR, RB + team total, etc.)
        for team, team_props in by_team.items():
            if len(team_props) >= 2:
                # Check for correlated prop types
                passing_props = [p for p in team_props if 'pass' in p.prop_type]
                receiving_props = [p for p in team_props if 'rec' in p.prop_type]

                if passing_props and receiving_props:
                    # QB + WR = HIGH correlation
                    groups["correlated_same_team"].append(CorrelationGroup(
                        game_id=game_id,
                        props=passing_props + receiving_props,
                        correlation_theme="qb_wr_stack",
                        estimated_correlation=0.65
                    ))

        # Opposing team correlations (both QBs, etc.)
        teams = list(by_team.keys())
        if len(teams) == 2:
            team1_passing = [p for p in by_team[teams[0]] if 'pass' in p.prop_type]
            team2_passing = [p for p in by_team[teams[1]] if 'pass' in p.prop_type]

            if team1_passing and team2_passing:
                # Both QBs = MEDIUM correlation (game script dependent)
                groups["correlated_opposing"].append(CorrelationGroup(
                    game_id=game_id,
                    props=team1_passing + team2_passing,
                    correlation_theme="opposing_qbs",
                    estimated_correlation=0.45  # Shootout correlation
                ))

    return groups


def estimate_scenario_probabilities(
    game_id: str,
    spread: float,
    total: float
) -> Dict[GameScript, float]:
    """Estimate probability of each game script scenario.

    Uses spread and total to predict game flow.

    Args:
        game_id: Game identifier
        spread: Point spread (negative = favorite)
        total: Over/under total

    Returns:
        Dict mapping GameScript -> probability
    """
    scenarios = {}

    # High total suggests shootout
    if total >= 50:
        scenarios[GameScript.SHOOTOUT] = 0.40
        scenarios[GameScript.DEFENSIVE_SLOG] = 0.10
        scenarios[GameScript.RB_DOMINATION] = 0.15
    elif total <= 42:
        scenarios[GameScript.SHOOTOUT] = 0.10
        scenarios[GameScript.DEFENSIVE_SLOG] = 0.45
        scenarios[GameScript.RB_DOMINATION] = 0.30
    else:
        scenarios[GameScript.SHOOTOUT] = 0.25
        scenarios[GameScript.DEFENSIVE_SLOG] = 0.25
        scenarios[GameScript.RB_DOMINATION] = 0.25

    # Large spread suggests blowout
    if abs(spread) >= 10:
        if spread < 0:  # Favorite
            scenarios[GameScript.BLOWOUT_FAVORITE] = 0.20
        else:  # Underdog
            scenarios[GameScript.BLOWOUT_UNDERDOG] = 0.05
        # Reduce other scenarios proportionally
        remaining = 1.0 - scenarios.get(GameScript.BLOWOUT_FAVORITE, 0) - scenarios.get(GameScript.BLOWOUT_UNDERDOG, 0)
        for script in [GameScript.SHOOTOUT, GameScript.DEFENSIVE_SLOG, GameScript.RB_DOMINATION]:
            scenarios[script] = scenarios.get(script, 0.2) * remaining / sum(scenarios.get(s, 0.2) for s in [GameScript.SHOOTOUT, GameScript.DEFENSIVE_SLOG, GameScript.RB_DOMINATION])
    else:
        # Close game - fill remaining probability
        remaining = 1.0 - sum(scenarios.values())
        scenarios[GameScript.BLOWOUT_FAVORITE] = remaining * 0.1
        scenarios[GameScript.BLOWOUT_UNDERDOG] = remaining * 0.05
        # Distribute rest evenly
        leftover = remaining * 0.85
        for script in scenarios:
            if script not in [GameScript.BLOWOUT_FAVORITE, GameScript.BLOWOUT_UNDERDOG]:
                scenarios[script] = scenarios.get(script, 0) + leftover / 3

    # Normalize to ensure sum = 1.0
    total_prob = sum(scenarios.values())
    scenarios = {k: v / total_prob for k, v in scenarios.items()}

    return scenarios


def prop_hit_probability_by_scenario(
    prop: PropValue,
    scenario: GameScript
) -> float:
    """Adjust prop hit probability based on game script scenario.

    Args:
        prop: The prop value
        scenario: Game script scenario

    Returns:
        Adjusted hit probability
    """
    base_prob = prop.hit_probability

    # Passing props
    if 'pass' in prop.prop_type:
        if scenario == GameScript.SHOOTOUT:
            return min(0.95, base_prob * 1.15)  # +15% in shootout
        elif scenario == GameScript.DEFENSIVE_SLOG:
            return max(0.10, base_prob * 0.85)  # -15% in defensive game
        elif scenario == GameScript.RB_DOMINATION:
            return max(0.10, base_prob * 0.80)  # -20% in run-heavy game
        elif scenario == GameScript.BLOWOUT_FAVORITE and prop.is_home:
            return min(0.95, base_prob * 1.10)  # Favorite passes to build lead
        elif scenario == GameScript.BLOWOUT_UNDERDOG and not prop.is_home:
            return min(0.95, base_prob * 1.20)  # Underdog passes to catch up

    # Rushing props
    elif 'rush' in prop.prop_type:
        if scenario == GameScript.SHOOTOUT:
            return max(0.10, base_prob * 0.85)  # -15% in shootout
        elif scenario == GameScript.RB_DOMINATION:
            return min(0.95, base_prob * 1.20)  # +20% in RB game
        elif scenario == GameScript.BLOWOUT_FAVORITE and prop.is_home:
            return min(0.95, base_prob * 1.15)  # Favorite runs clock
        elif scenario == GameScript.DEFENSIVE_SLOG:
            return min(0.95, base_prob * 1.10)  # +10% more runs

    # Receiving props
    elif 'rec' in prop.prop_type:
        # Similar to passing but slightly dampened
        if scenario == GameScript.SHOOTOUT:
            return min(0.95, base_prob * 1.12)
        elif scenario == GameScript.DEFENSIVE_SLOG:
            return max(0.10, base_prob * 0.88)
        elif scenario == GameScript.RB_DOMINATION:
            return max(0.10, base_prob * 0.82)

    # Default: return base probability
    return base_prob


def calculate_parlay_probability_with_correlation(
    props: List[PropValue],
    correlation_group: Optional[CorrelationGroup] = None
) -> Tuple[float, float]:
    """Calculate parlay probability accounting for correlation.

    Args:
        props: List of props in parlay
        correlation_group: If provided, use correlation adjustment

    Returns:
        Tuple of (raw_probability, adjusted_probability)
        - raw: assuming independence
        - adjusted: accounting for correlation
    """
    # Raw probability (independence assumption)
    raw_prob = np.prod([p.hit_probability for p in props])

    if not correlation_group or len(props) <= 1:
        return raw_prob, raw_prob

    # Correlation adjustment using scenario analysis
    if len(props) == 1:
        return props[0].hit_probability, props[0].hit_probability

    # Get first prop's game context
    first_prop = props[0]
    scenarios = estimate_scenario_probabilities(
        first_prop.game_id,
        first_prop.spread,
        first_prop.total
    )

    # Calculate probability across scenarios
    adjusted_prob = 0.0
    for scenario, scenario_prob in scenarios.items():
        # Probability all props hit in this scenario
        props_hit_in_scenario = np.prod([
            prop_hit_probability_by_scenario(p, scenario)
            for p in props
        ])
        adjusted_prob += scenario_prob * props_hit_in_scenario

    return raw_prob, adjusted_prob


def build_parlay_suggestions(
    value_props: List[PropValue],
    max_legs: int = 4,
    max_risk_per_game: float = 0.05,
    kelly_fraction: float = 0.25,
    min_parlay_ev: float = 0.10
) -> List[ParlaySuggestion]:
    """Build optimal parlay suggestions with correlation awareness.

    Args:
        value_props: List of props with identified value
        max_legs: Maximum legs per parlay
        max_risk_per_game: Max % of bankroll to risk per game
        kelly_fraction: Fraction of Kelly criterion to use
        min_parlay_ev: Minimum EV to suggest parlay

    Returns:
        List of parlay suggestions, sorted by EV
    """
    # Group props by correlation
    correlation_groups = group_props_by_correlation(value_props)

    suggestions = []

    # Strategy 1: Independent props (different games)
    independent_props = []
    for group in correlation_groups["independent"]:
        independent_props.extend(group.props)

    if len(independent_props) >= 2:
        # Build parlays from independent props
        for combo_size in range(2, min(max_legs + 1, len(independent_props) + 1)):
            # For simplicity, take top N by EV
            sorted_props = sorted(independent_props, key=lambda p: p.ev, reverse=True)
            combo_props = sorted_props[:combo_size]

            suggestion = _build_parlay_from_props(
                combo_props,
                kelly_fraction=kelly_fraction,
                max_risk=max_risk_per_game
            )

            if suggestion and suggestion.ev >= min_parlay_ev:
                suggestions.append(suggestion)

    # Strategy 2: Correlated same-game parlays (with discount)
    for group in correlation_groups["correlated_same_team"]:
        if len(group.props) >= 2:
            suggestion = _build_parlay_from_props(
                group.props[:3],  # Max 3 legs in same game
                kelly_fraction=kelly_fraction * 0.75,  # Reduce sizing due to correlation
                max_risk=max_risk_per_game * 0.5,  # Half risk for correlated
                correlation_group=group
            )

            if suggestion and suggestion.ev >= min_parlay_ev * 0.8:  # Lower threshold for same-game
                suggestions.append(suggestion)

    # Sort by EV and return top suggestions
    suggestions.sort(key=lambda s: s.ev, reverse=True)

    return suggestions[:10]  # Return top 10


def _build_parlay_from_props(
    props: List[PropValue],
    kelly_fraction: float,
    max_risk: float,
    correlation_group: Optional[CorrelationGroup] = None
) -> Optional[ParlaySuggestion]:
    """Build a single parlay suggestion from props.

    Args:
        props: Props to include
        kelly_fraction: Kelly fraction
        max_risk: Max stake as % of bankroll
        correlation_group: If correlated, include group

    Returns:
        ParlaySuggestion or None if invalid
    """
    if not props:
        return None

    # Calculate combined odds (multiply American odds)
    combined_decimal_odds = 1.0
    for prop in props:
        decimal = american_to_decimal(prop.odds)
        combined_decimal_odds *= decimal

    combined_american = decimal_to_american(combined_decimal_odds)

    # Calculate probabilities
    raw_prob, adjusted_prob = calculate_parlay_probability_with_correlation(
        props, correlation_group
    )

    # Calculate EV
    implied_prob = 1.0 / combined_decimal_odds
    ev = (adjusted_prob - implied_prob) * 100  # as percentage

    # Kelly sizing
    b = combined_decimal_odds - 1  # Net odds
    f = (adjusted_prob * b - (1 - adjusted_prob)) / b
    recommended_stake = kelly_fraction * f
    recommended_stake = min(recommended_stake, max_risk)  # Cap at max risk
    recommended_stake = max(0.01, recommended_stake)  # Min 1%

    # Confidence based on EV and probability
    if ev >= 20 and adjusted_prob >= 0.40:
        confidence = "HIGH"
    elif ev >= 10 and adjusted_prob >= 0.30:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # Correlation adjustment
    correlation_adjustment = (raw_prob - adjusted_prob) / raw_prob if raw_prob > 0 else 0.0

    # Scenarios (only for same-game parlays)
    scenarios = {}
    if correlation_group and props:
        scenarios = estimate_scenario_probabilities(
            props[0].game_id,
            props[0].spread,
            props[0].total
        )

    return ParlaySuggestion(
        legs=props,
        combined_odds=combined_american,
        combined_probability=adjusted_prob,
        raw_probability=raw_prob,
        ev=ev,
        recommended_stake_pct=recommended_stake * 100,
        confidence=confidence,
        correlation_adjustment=correlation_adjustment * 100,
        scenarios={k: round(v, 3) for k, v in scenarios.items()}
    )


def american_to_decimal(american_odds: int) -> float:
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to American odds."""
    if decimal_odds >= 2.0:
        return int((decimal_odds - 1) * 100)
    else:
        return int(-100 / (decimal_odds - 1))
