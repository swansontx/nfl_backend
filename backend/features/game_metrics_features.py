"""
Game Metrics Feature Engineering

Enhances game predictions with advanced team metrics:
- Pace metrics (plays per game, time of possession)
- Turnover margin (takeaways - turnovers)
- Efficiency metrics (success rate, EPA)
- Red zone and third down conversion rates
"""

from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path
from backend.analysis.advanced_team_metrics import AdvancedTeamMetricsCalculator


@dataclass
class EnhancedTeamStrength:
    """Enhanced team strength with advanced metrics."""
    team: str

    # Basic ratings (from existing system)
    offensive_rating: float  # Points per game
    defensive_rating: float  # Points allowed per game
    net_rating: float  # Point differential

    # NEW: Pace metrics
    plays_per_game: float = 65.0  # Offensive pace
    opp_plays_per_game: float = 65.0  # Defensive pace faced
    time_of_possession_pct: float = 0.50  # % of game possession

    # NEW: Turnover metrics
    turnover_margin: int = 0  # Takeaways - Turnovers
    turnover_rate: float = 0.0  # Turnovers per 100 plays
    takeaway_rate: float = 0.0  # Takeaways per 100 plays

    # NEW: Efficiency metrics
    success_rate_offense: float = 0.45  # % successful plays
    success_rate_defense: float = 0.45  # Opponent success rate
    epa_per_play_offense: float = 0.0  # Expected points added
    epa_per_play_defense: float = 0.0  # EPA allowed

    # NEW: Situational metrics
    red_zone_td_pct: float = 0.55  # Red zone TD efficiency
    third_down_pct: float = 0.40  # 3rd down conversion rate
    explosive_play_rate: float = 0.10  # Big plays per snap

    # Existing
    recent_form_rating: float = 0.0
    home_advantage: float = 2.5


class GameMetricsEngine:
    """
    Enriches game predictions with advanced team metrics.

    Integrates pace, turnovers, and efficiency metrics into game outcome models.
    """

    def __init__(self, season: int = 2025, inputs_dir: str = "inputs"):
        self.season = season
        self.inputs_dir = Path(inputs_dir)

        # Initialize advanced metrics calculator
        pbp_file = self.inputs_dir / f"play_by_play_{season}.parquet"
        if pbp_file.exists():
            self.calculator = AdvancedTeamMetricsCalculator(season=season, pbp_file=pbp_file)
        else:
            print(f"Warning: No play-by-play data for {season}, advanced metrics unavailable")
            self.calculator = None

        # Cache metrics
        self._metrics_cache = {}

    def _blend_metrics(
        self,
        full_season: Dict,
        recent: Dict,
        full_weight: float = 0.70,
        recent_weight: float = 0.30
    ) -> Dict:
        """
        Blend full-season and recent metrics with specified weights.

        Args:
            full_season: Full season metrics
            recent: Recent weeks metrics
            full_weight: Weight for full season (default 0.70)
            recent_weight: Weight for recent (default 0.30)

        Returns:
            Blended metrics dictionary
        """
        blended = {}

        # Metrics to blend (continuous values)
        continuous_metrics = [
            'plays_per_game', 'opp_plays_per_game', 'time_of_possession_pct',
            'turnover_rate', 'takeaway_rate',
            'success_rate_offense', 'success_rate_defense',
            'epa_per_play_offense', 'epa_per_play_defense',
            'red_zone_td_pct', 'third_down_pct', 'explosive_play_rate'
        ]

        for metric in continuous_metrics:
            full_val = full_season.get(metric, 0.0)
            recent_val = recent.get(metric, full_val)  # Fallback to full season if recent missing

            # Weighted blend
            blended[metric] = (full_val * full_weight) + (recent_val * recent_weight)

        # Turnover margin needs special handling (integer, regression to mean)
        full_margin = full_season.get('turnover_margin', 0)
        recent_margin = recent.get('turnover_margin', full_margin)

        # Blend turnover margins
        blended_margin = (full_margin * full_weight) + (recent_margin * recent_weight)

        # REGRESSION TO MEAN for turnovers (high variance stat)
        # Regress 40% toward league average (0)
        blended_margin = blended_margin * 0.6 + 0 * 0.4

        blended['turnover_margin'] = int(round(blended_margin))

        return blended

    def get_enhanced_team_strength(
        self,
        team: str,
        base_offensive_rating: float,
        base_defensive_rating: float,
        recent_form: float = 0.0,
        is_home: bool = False,
        weeks: Optional[List[int]] = None,
        blend_with_full_season: bool = True
    ) -> EnhancedTeamStrength:
        """
        Get enhanced team strength with advanced metrics.

        IMPROVED: Now blends full-season metrics (70%) with recent metrics (30%)
        to reduce small-sample noise while capturing recent trends.

        Args:
            team: Team abbreviation
            base_offensive_rating: Base PPG rating
            base_defensive_rating: Base points allowed rating
            recent_form: Recent form adjustment
            is_home: Whether team is home
            weeks: Optional weeks to include for recent metrics
            blend_with_full_season: Whether to blend with full season (default True)

        Returns:
            EnhancedTeamStrength with all metrics
        """
        # Start with base ratings
        net_rating = base_offensive_rating - base_defensive_rating

        # Get advanced metrics if available
        metrics = {}
        if self.calculator:
            if blend_with_full_season and weeks:
                # Calculate both full-season and recent metrics
                full_cache_key = f"{team}_full"
                recent_cache_key = f"{team}_{weeks}"

                # Get full season metrics
                if full_cache_key not in self._metrics_cache:
                    try:
                        self._metrics_cache[full_cache_key] = self.calculator.calculate_team_metrics(team, None)
                    except Exception as e:
                        print(f"Error calculating full-season metrics for {team}: {e}")
                        self._metrics_cache[full_cache_key] = {}

                # Get recent metrics
                if recent_cache_key not in self._metrics_cache:
                    try:
                        self._metrics_cache[recent_cache_key] = self.calculator.calculate_team_metrics(team, weeks)
                    except Exception as e:
                        print(f"Error calculating recent metrics for {team}: {e}")
                        self._metrics_cache[recent_cache_key] = {}

                # Blend metrics (70% full season, 30% recent)
                full_metrics = self._metrics_cache[full_cache_key]
                recent_metrics = self._metrics_cache[recent_cache_key]
                metrics = self._blend_metrics(full_metrics, recent_metrics, 0.70, 0.30)

            else:
                # Use only the specified weeks (or full season if weeks=None)
                cache_key = f"{team}_{weeks}"
                if cache_key not in self._metrics_cache:
                    try:
                        self._metrics_cache[cache_key] = self.calculator.calculate_team_metrics(team, weeks)
                    except Exception as e:
                        print(f"Error calculating metrics for {team}: {e}")
                        self._metrics_cache[cache_key] = {}

                metrics = self._metrics_cache[cache_key]

        # Build enhanced strength object
        return EnhancedTeamStrength(
            team=team,
            offensive_rating=base_offensive_rating,
            defensive_rating=base_defensive_rating,
            net_rating=net_rating,

            # Pace metrics
            plays_per_game=metrics.get('plays_per_game', 65.0),
            opp_plays_per_game=metrics.get('opp_plays_per_game', 65.0),
            time_of_possession_pct=metrics.get('time_of_possession_pct', 0.50),

            # Turnover metrics (with regression to mean applied if blended)
            turnover_margin=metrics.get('turnover_margin', 0),
            turnover_rate=metrics.get('turnover_rate', 0.0),
            takeaway_rate=metrics.get('takeaway_rate', 0.0),

            # Efficiency metrics
            success_rate_offense=metrics.get('success_rate_offense', 0.45),
            success_rate_defense=metrics.get('success_rate_defense', 0.45),
            epa_per_play_offense=metrics.get('epa_per_play_offense', 0.0),
            epa_per_play_defense=metrics.get('epa_per_play_defense', 0.0),

            # Situational metrics
            red_zone_td_pct=metrics.get('red_zone_td_pct', 0.55),
            third_down_pct=metrics.get('third_down_pct', 0.40),
            explosive_play_rate=metrics.get('explosive_play_rate', 0.10),

            # Existing
            recent_form_rating=recent_form,
            home_advantage=2.5 if is_home else 0.0
        )

    def calculate_pace_adjusted_total(
        self,
        home_strength: EnhancedTeamStrength,
        away_strength: EnhancedTeamStrength,
        base_total: float
    ) -> Tuple[float, str]:
        """
        Adjust predicted total based on pace metrics.

        Args:
            home_strength: Home team enhanced strength
            away_strength: Away team enhanced strength
            base_total: Base predicted total (from PPG model)

        Returns:
            (adjusted_total, reasoning)
        """
        # Calculate combined pace
        avg_plays_per_game = (home_strength.plays_per_game + away_strength.plays_per_game) / 2.0
        league_avg_pace = 65.0  # Approximate NFL average

        # Pace adjustment: more plays = more scoring opportunities
        # Each additional 10 plays ≈ 3-4 points
        pace_diff = avg_plays_per_game - league_avg_pace
        pace_adjustment = (pace_diff / 10.0) * 3.5

        adjusted_total = base_total + pace_adjustment

        # Generate reasoning
        if abs(pace_adjustment) < 1.0:
            pace_desc = "average pace"
        elif pace_adjustment > 0:
            pace_desc = f"fast pace (+{avg_plays_per_game - league_avg_pace:.1f} plays/game)"
        else:
            pace_desc = f"slow pace ({avg_plays_per_game - league_avg_pace:.1f} plays/game)"

        reasoning = f"Pace adjustment: {pace_adjustment:+.1f} points ({pace_desc})"

        return round(adjusted_total, 1), reasoning

    def calculate_turnover_adjusted_spread(
        self,
        home_strength: EnhancedTeamStrength,
        away_strength: EnhancedTeamStrength,
        base_spread: float,
        sample_weeks: int = 4
    ) -> Tuple[float, str]:
        """
        Adjust predicted spread based on turnover margin.

        Args:
            home_strength: Home team enhanced strength
            away_strength: Away team enhanced strength
            base_spread: Base predicted spread (from ratings model)
            sample_weeks: Number of weeks in sample (affects multiplier)

        Returns:
            (adjusted_spread, reasoning)
        """
        # Turnover margin differential
        home_to_margin = home_strength.turnover_margin
        away_to_margin = away_strength.turnover_margin
        to_diff = home_to_margin - away_to_margin

        # CALIBRATED: Reduce multiplier based on sample size
        # Small samples (≤4 weeks): 0.8 points per margin (conservative due to variance)
        # Medium samples (5-8 weeks): 1.2 points per margin
        # Large samples (9+ weeks): 1.5 points per margin
        if sample_weeks <= 4:
            multiplier = 0.8  # Very conservative for small samples
        elif sample_weeks <= 8:
            multiplier = 1.2  # Moderate for medium samples
        else:
            multiplier = 1.5  # More confident for large samples

        turnover_adjustment = to_diff * multiplier

        adjusted_spread = base_spread + turnover_adjustment

        # Generate reasoning
        if abs(turnover_adjustment) < 1.0:
            to_desc = "neutral turnover margin"
        elif turnover_adjustment > 0:
            to_desc = f"{home_strength.team} has +{to_diff} turnover margin edge"
        else:
            to_desc = f"{away_strength.team} has +{abs(to_diff)} turnover margin edge"

        reasoning = f"Turnover adjustment: {turnover_adjustment:+.1f} points ({to_desc})"

        return round(adjusted_spread, 1), reasoning

    def calculate_efficiency_adjustments(
        self,
        home_strength: EnhancedTeamStrength,
        away_strength: EnhancedTeamStrength
    ) -> Dict[str, float]:
        """
        Calculate additional adjustments based on efficiency metrics.

        Args:
            home_strength: Home team enhanced strength
            away_strength: Away team enhanced strength

        Returns:
            Dict with spread and total adjustments
        """
        adjustments = {
            'spread_adj': 0.0,
            'total_adj': 0.0,
            'reasoning': []
        }

        # EPA differential (expected points per play advantage)
        home_epa_edge = (
            home_strength.epa_per_play_offense - away_strength.epa_per_play_defense
        )
        away_epa_edge = (
            away_strength.epa_per_play_offense - home_strength.epa_per_play_defense
        )

        epa_diff = home_epa_edge - away_epa_edge

        # CALIBRATED: Add 50% damping factor to prevent overcorrection
        # EPA difference translates to point differential, but multiplying by full
        # 65 plays creates extreme predictions. Apply 50% damping for stability.
        epa_spread_adj = epa_diff * 65.0 * 0.5  # 50% damping
        adjustments['spread_adj'] += epa_spread_adj

        if abs(epa_spread_adj) >= 1.0:
            adjustments['reasoning'].append(
                f"EPA edge: {epa_spread_adj:+.1f} points"
            )

        # Red zone efficiency (affects scoring in close games)
        rz_diff = home_strength.red_zone_td_pct - away_strength.red_zone_td_pct
        # Each 10% red zone advantage ≈ 1.5 points (conservative)
        rz_adj = (rz_diff / 0.10) * 1.5
        adjustments['spread_adj'] += rz_adj

        if abs(rz_adj) >= 1.0:
            adjustments['reasoning'].append(
                f"Red zone efficiency: {rz_adj:+.1f} points"
            )

        # Success rate differential (consistency)
        success_diff = (
            home_strength.success_rate_offense - away_strength.success_rate_defense
        ) - (
            away_strength.success_rate_offense - home_strength.success_rate_defense
        )

        # CALIBRATED: Add 30% damping to success rate for stability
        # Each 5% success rate advantage ≈ 1 point, but damped to prevent stacking
        success_adj = (success_diff / 0.05) * 1.0 * 0.7  # 30% damping
        adjustments['spread_adj'] += success_adj

        if abs(success_adj) >= 1.0:
            adjustments['reasoning'].append(
                f"Success rate edge: {success_adj:+.1f} points"
            )

        # Explosive play rate (affects total)
        combined_explosive_rate = (
            home_strength.explosive_play_rate + away_strength.explosive_play_rate
        ) / 2.0
        league_avg_explosive = 0.10

        # More explosive plays = higher variance and slightly higher totals
        explosive_diff = combined_explosive_rate - league_avg_explosive
        explosive_total_adj = (explosive_diff / 0.02) * 2.0  # Each 2% = 2 points
        adjustments['total_adj'] += explosive_total_adj

        if abs(explosive_total_adj) >= 1.5:
            adjustments['reasoning'].append(
                f"Explosive plays: {explosive_total_adj:+.1f} total adjustment"
            )

        # CALIBRATED: Cap adjustments to prevent extreme predictions
        # Spread adjustments capped at ±12 points (reasonable max for efficiency edge)
        # Total adjustments capped at ±8 points (reasonable max for pace/explosiveness)
        original_spread_adj = adjustments['spread_adj']
        original_total_adj = adjustments['total_adj']

        adjustments['spread_adj'] = max(-12.0, min(12.0, adjustments['spread_adj']))
        adjustments['total_adj'] = max(-8.0, min(8.0, adjustments['total_adj']))

        if abs(original_spread_adj - adjustments['spread_adj']) > 0.1:
            adjustments['reasoning'].append(
                f"Spread adjustment capped at {adjustments['spread_adj']:+.1f}"
            )
        if abs(original_total_adj - adjustments['total_adj']) > 0.1:
            adjustments['reasoning'].append(
                f"Total adjustment capped at {adjustments['total_adj']:+.1f}"
            )

        return adjustments

    def get_game_metrics_summary(
        self,
        home_strength: EnhancedTeamStrength,
        away_strength: EnhancedTeamStrength
    ) -> Dict:
        """
        Get comprehensive summary of game metrics for analysis.

        Args:
            home_strength: Home team metrics
            away_strength: Away team metrics

        Returns:
            Dict with all relevant matchup metrics
        """
        return {
            'pace': {
                'home_plays_per_game': home_strength.plays_per_game,
                'away_plays_per_game': away_strength.plays_per_game,
                'combined_pace': (home_strength.plays_per_game + away_strength.plays_per_game) / 2.0,
                'pace_vs_league_avg': ((home_strength.plays_per_game + away_strength.plays_per_game) / 2.0) - 65.0
            },
            'turnovers': {
                'home_margin': home_strength.turnover_margin,
                'away_margin': away_strength.turnover_margin,
                'margin_differential': home_strength.turnover_margin - away_strength.turnover_margin,
                'home_turnover_rate': home_strength.turnover_rate,
                'away_turnover_rate': away_strength.turnover_rate
            },
            'efficiency': {
                'home_success_rate_off': home_strength.success_rate_offense,
                'home_success_rate_def': home_strength.success_rate_defense,
                'away_success_rate_off': away_strength.success_rate_offense,
                'away_success_rate_def': away_strength.success_rate_defense,
                'home_epa_edge': home_strength.epa_per_play_offense - away_strength.epa_per_play_defense,
                'away_epa_edge': away_strength.epa_per_play_offense - home_strength.epa_per_play_defense
            },
            'red_zone': {
                'home_rz_td_pct': home_strength.red_zone_td_pct,
                'away_rz_td_pct': away_strength.red_zone_td_pct,
                'rz_advantage': 'home' if home_strength.red_zone_td_pct > away_strength.red_zone_td_pct else 'away'
            },
            'situational': {
                'home_third_down_pct': home_strength.third_down_pct,
                'away_third_down_pct': away_strength.third_down_pct,
                'home_explosive_rate': home_strength.explosive_play_rate,
                'away_explosive_rate': away_strength.explosive_play_rate
            }
        }


# Utility function for easy integration
def enhance_game_prediction(
    home_team: str,
    away_team: str,
    base_home_score: float,
    base_away_score: float,
    home_offensive_rating: float,
    home_defensive_rating: float,
    away_offensive_rating: float,
    away_defensive_rating: float,
    season: int = 2025,
    recent_weeks: int = 4
) -> Dict:
    """
    Enhance a game prediction with advanced metrics.

    Args:
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        base_home_score: Base predicted home score
        base_away_score: Base predicted away score
        home_offensive_rating: Home PPG
        home_defensive_rating: Home points allowed per game
        away_offensive_rating: Away PPG
        away_defensive_rating: Away points allowed per game
        season: NFL season
        recent_weeks: Number of recent weeks for metrics

    Returns:
        Dict with enhanced predictions and reasoning
    """
    engine = GameMetricsEngine(season=season)

    # Get enhanced team strengths
    home_strength = engine.get_enhanced_team_strength(
        home_team,
        home_offensive_rating,
        home_defensive_rating,
        is_home=True
    )

    away_strength = engine.get_enhanced_team_strength(
        away_team,
        away_offensive_rating,
        away_defensive_rating,
        is_home=False
    )

    # Calculate adjustments
    base_spread = base_home_score - base_away_score
    base_total = base_home_score + base_away_score

    # Pace adjustment to total
    adjusted_total, pace_reasoning = engine.calculate_pace_adjusted_total(
        home_strength, away_strength, base_total
    )

    # Turnover adjustment to spread (pass sample size for calibration)
    adjusted_spread, to_reasoning = engine.calculate_turnover_adjusted_spread(
        home_strength, away_strength, base_spread, sample_weeks=recent_weeks
    )

    # Efficiency adjustments
    efficiency_adjs = engine.calculate_efficiency_adjustments(home_strength, away_strength)

    # Apply all adjustments
    final_spread = adjusted_spread + efficiency_adjs['spread_adj']
    final_total = adjusted_total + efficiency_adjs['total_adj']

    # Recalculate scores
    final_home_score = (final_total + final_spread) / 2.0
    final_away_score = (final_total - final_spread) / 2.0

    return {
        'base_prediction': {
            'home_score': base_home_score,
            'away_score': base_away_score,
            'spread': base_spread,
            'total': base_total
        },
        'enhanced_prediction': {
            'home_score': round(final_home_score, 1),
            'away_score': round(final_away_score, 1),
            'spread': round(final_spread, 1),
            'total': round(final_total, 1)
        },
        'adjustments': {
            'pace_total_adj': adjusted_total - base_total,
            'turnover_spread_adj': adjusted_spread - base_spread,
            'efficiency_spread_adj': efficiency_adjs['spread_adj'],
            'efficiency_total_adj': efficiency_adjs['total_adj']
        },
        'reasoning': {
            'pace': pace_reasoning,
            'turnovers': to_reasoning,
            'efficiency': '; '.join(efficiency_adjs['reasoning'])
        },
        'metrics_summary': engine.get_game_metrics_summary(home_strength, away_strength)
    }
