"""Advanced Matchup Analysis Engine using EPA, CPOE, Success Rate, and Pressure Metrics.

Compares offensive and defensive advanced metrics to identify favorable/unfavorable matchups:

Matchup Types:
1. QB EPA vs Defense EPA Allowed (passing)
2. RB Rushing EPA vs Defense Rushing EPA Allowed
3. WR/TE Receiving EPA vs Defense Receiving EPA Allowed
4. QB CPOE vs Secondary CPOE Allowed
5. Success Rate Offense vs Success Rate Defense
6. OL Pressure Rate Allowed vs DL Pressure Rate Generated
7. Air Yards per Attempt vs Air Yards Allowed

These matchups identify exploitable advantages for prop betting.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import statistics
import json
from pathlib import Path


@dataclass
class OffensiveProfile:
    """Player/team offensive advanced metrics."""
    player_id: str
    player_name: str
    position: str
    team: str

    # EPA metrics
    avg_epa: float
    qb_epa: float = 0.0
    rushing_epa: float = 0.0
    receiving_epa: float = 0.0

    # Efficiency metrics
    cpoe: float = 0.0
    success_rate: float = 0.0

    # Volume metrics
    attempts: int = 0
    air_yards_per_attempt: float = 0.0

    # Pressure metrics (for QBs)
    pressure_rate_faced: float = 0.0

    # Sample size
    games_played: int = 0


@dataclass
class DefensiveProfile:
    """Team defensive advanced metrics."""
    team: str

    # EPA allowed (lower is better for defense)
    passing_epa_allowed: float
    rushing_epa_allowed: float
    receiving_epa_allowed: float
    total_epa_allowed: float

    # Efficiency allowed
    cpoe_allowed: float = 0.0
    success_rate_allowed: float = 0.0

    # Pressure metrics (for pass rush)
    pressure_rate_generated: float = 0.0

    # Volume allowed
    yards_per_attempt_allowed: float = 0.0
    air_yards_allowed: float = 0.0

    # Sample size
    games_played: int = 0


@dataclass
class MatchupEdge:
    """Quantified matchup advantage/disadvantage."""
    matchup_type: str
    offensive_value: float
    defensive_value: float
    edge: float  # Positive = offense advantage, negative = defense advantage
    edge_percentile: float  # 0-100, higher = more favorable for offense
    confidence: float
    recommendation: str
    explanation: str


class MatchupAnalyzer:
    """Analyzes offensive vs defensive matchups using advanced metrics."""

    def __init__(self):
        self.league_averages = {
            'epa': 0.0,  # League average EPA ~0
            'qb_epa': 0.0,
            'rushing_epa': 0.0,
            'receiving_epa': 0.0,
            'cpoe': 0.0,  # League average CPOE ~0%
            'success_rate': 48.0,  # League average ~48%
            'pressure_rate': 27.0,  # League average ~27%
            'air_yards_per_attempt': 7.5,
        }

    def analyze_qb_vs_defense(
        self,
        qb_profile: OffensiveProfile,
        defense_profile: DefensiveProfile
    ) -> List[MatchupEdge]:
        """Analyze QB matchup vs defense using advanced metrics.

        Args:
            qb_profile: QB offensive profile
            defense_profile: Opposing defense profile

        Returns:
            List of MatchupEdge objects for different metrics
        """
        edges = []

        # 1. QB EPA vs Passing EPA Allowed
        epa_edge = self._calculate_epa_edge(
            qb_profile.qb_epa,
            defense_profile.passing_epa_allowed,
            'passing'
        )
        edges.append(epa_edge)

        # 2. CPOE vs CPOE Allowed (secondary quality)
        if qb_profile.cpoe != 0 and defense_profile.cpoe_allowed != 0:
            cpoe_edge = self._calculate_cpoe_edge(
                qb_profile.cpoe,
                defense_profile.cpoe_allowed
            )
            edges.append(cpoe_edge)

        # 3. Success Rate vs Success Rate Allowed
        if qb_profile.success_rate > 0 and defense_profile.success_rate_allowed > 0:
            success_edge = self._calculate_success_rate_edge(
                qb_profile.success_rate,
                defense_profile.success_rate_allowed
            )
            edges.append(success_edge)

        # 4. Pressure Rate Matchup (OL protection vs DL pressure)
        if qb_profile.pressure_rate_faced > 0 and defense_profile.pressure_rate_generated > 0:
            pressure_edge = self._calculate_pressure_edge(
                qb_profile.pressure_rate_faced,
                defense_profile.pressure_rate_generated
            )
            edges.append(pressure_edge)

        return edges

    def analyze_rusher_vs_defense(
        self,
        rusher_profile: OffensiveProfile,
        defense_profile: DefensiveProfile
    ) -> List[MatchupEdge]:
        """Analyze RB/QB rusher matchup vs run defense.

        Args:
            rusher_profile: Rusher offensive profile
            defense_profile: Opposing defense profile

        Returns:
            List of MatchupEdge objects
        """
        edges = []

        # Rushing EPA vs Rushing EPA Allowed
        epa_edge = self._calculate_epa_edge(
            rusher_profile.rushing_epa,
            defense_profile.rushing_epa_allowed,
            'rushing'
        )
        edges.append(epa_edge)

        # Success Rate (if available)
        if rusher_profile.success_rate > 0 and defense_profile.success_rate_allowed > 0:
            success_edge = self._calculate_success_rate_edge(
                rusher_profile.success_rate,
                defense_profile.success_rate_allowed
            )
            edges.append(success_edge)

        return edges

    def analyze_receiver_vs_defense(
        self,
        receiver_profile: OffensiveProfile,
        defense_profile: DefensiveProfile
    ) -> List[MatchupEdge]:
        """Analyze WR/TE matchup vs pass defense.

        Args:
            receiver_profile: Receiver offensive profile
            defense_profile: Opposing defense profile

        Returns:
            List of MatchupEdge objects
        """
        edges = []

        # Receiving EPA vs Receiving EPA Allowed
        epa_edge = self._calculate_epa_edge(
            receiver_profile.receiving_epa,
            defense_profile.receiving_epa_allowed,
            'receiving'
        )
        edges.append(epa_edge)

        # Air Yards Matchup (if available)
        if receiver_profile.air_yards_per_attempt > 0 and defense_profile.air_yards_allowed > 0:
            air_yards_edge = self._calculate_air_yards_edge(
                receiver_profile.air_yards_per_attempt,
                defense_profile.air_yards_allowed
            )
            edges.append(air_yards_edge)

        return edges

    def _calculate_epa_edge(
        self,
        offensive_epa: float,
        defensive_epa_allowed: float,
        play_type: str
    ) -> MatchupEdge:
        """Calculate EPA matchup edge.

        Args:
            offensive_epa: Player's EPA per play
            defensive_epa_allowed: Defense's EPA allowed per play
            play_type: 'passing', 'rushing', or 'receiving'

        Returns:
            MatchupEdge with EPA analysis
        """
        # Edge = offensive EPA + (league avg - defensive EPA allowed)
        # If defense allows high EPA (poor defense), that's good for offense
        league_avg = self.league_averages['epa']

        # Normalize defensive EPA: positive means defense is good (allows less EPA)
        defensive_quality = league_avg - defensive_epa_allowed

        # Total edge: offensive EPA minus defensive quality
        edge = offensive_epa - defensive_quality

        # Convert to percentile (assume std dev of 0.1 EPA)
        # Edge of +0.2 EPA = very favorable, -0.2 EPA = very unfavorable
        std_dev = 0.1
        z_score = edge / std_dev

        # Convert z-score to percentile (rough approximation)
        if z_score >= 2.0:
            percentile = 95
        elif z_score >= 1.5:
            percentile = 85
        elif z_score >= 1.0:
            percentile = 75
        elif z_score >= 0.5:
            percentile = 65
        elif z_score >= 0:
            percentile = 55
        elif z_score >= -0.5:
            percentile = 45
        elif z_score >= -1.0:
            percentile = 35
        elif z_score >= -1.5:
            percentile = 25
        else:
            percentile = 15

        # Confidence based on magnitude of edge
        confidence = min(0.6 + abs(edge) * 2, 0.9)

        # Recommendation
        if edge > 0.15:
            recommendation = f"STRONG VALUE: {play_type} OVER props"
        elif edge > 0.05:
            recommendation = f"VALUE: {play_type} OVER props"
        elif edge > -0.05:
            recommendation = f"NEUTRAL: {play_type} matchup"
        elif edge > -0.15:
            recommendation = f"FADE: {play_type} props"
        else:
            recommendation = f"STRONG FADE: {play_type} UNDER props"

        explanation = (
            f"Offensive EPA: {offensive_epa:+.3f} | "
            f"Defensive EPA allowed: {defensive_epa_allowed:+.3f} | "
            f"Edge: {edge:+.3f} EPA per play ({percentile}th percentile)"
        )

        return MatchupEdge(
            matchup_type=f'{play_type}_epa',
            offensive_value=offensive_epa,
            defensive_value=defensive_epa_allowed,
            edge=edge,
            edge_percentile=percentile,
            confidence=confidence,
            recommendation=recommendation,
            explanation=explanation
        )

    def _calculate_cpoe_edge(
        self,
        offensive_cpoe: float,
        defensive_cpoe_allowed: float
    ) -> MatchupEdge:
        """Calculate CPOE matchup edge.

        Args:
            offensive_cpoe: QB's CPOE percentage
            defensive_cpoe_allowed: Defense's CPOE allowed

        Returns:
            MatchupEdge with CPOE analysis
        """
        # Edge = QB CPOE - Defense CPOE allowed
        # If defense allows high CPOE (poor secondary), that's good for QB
        edge = offensive_cpoe - defensive_cpoe_allowed

        # CPOE edges in percentage points
        # +3% CPOE edge = very favorable
        percentile = 50 + (edge * 5)  # Rough conversion
        percentile = max(10, min(90, percentile))

        confidence = min(0.65 + abs(edge) * 0.03, 0.85)

        if edge > 3.0:
            recommendation = "STRONG VALUE: Completion props, passing yards OVER"
        elif edge > 1.0:
            recommendation = "VALUE: QB accuracy advantage"
        elif edge > -1.0:
            recommendation = "NEUTRAL: Even accuracy matchup"
        elif edge > -3.0:
            recommendation = "FADE: Secondary has edge"
        else:
            recommendation = "STRONG FADE: Elite secondary vs accuracy concerns"

        explanation = (
            f"QB CPOE: {offensive_cpoe:+.1f}% | "
            f"Defense CPOE allowed: {defensive_cpoe_allowed:+.1f}% | "
            f"Edge: {edge:+.1f}% CPOE"
        )

        return MatchupEdge(
            matchup_type='cpoe_accuracy',
            offensive_value=offensive_cpoe,
            defensive_value=defensive_cpoe_allowed,
            edge=edge,
            edge_percentile=percentile,
            confidence=confidence,
            recommendation=recommendation,
            explanation=explanation
        )

    def _calculate_success_rate_edge(
        self,
        offensive_success_rate: float,
        defensive_success_rate_allowed: float
    ) -> MatchupEdge:
        """Calculate success rate matchup edge.

        Args:
            offensive_success_rate: Offense success rate %
            defensive_success_rate_allowed: Defense success rate allowed %

        Returns:
            MatchupEdge with success rate analysis
        """
        # Edge = offensive success rate - defensive success rate allowed
        edge = offensive_success_rate - defensive_success_rate_allowed

        # Success rate edges in percentage points
        percentile = 50 + (edge * 2)
        percentile = max(10, min(90, percentile))

        confidence = min(0.60 + abs(edge) * 0.02, 0.80)

        if edge > 8.0:
            recommendation = "STRONG VALUE: Efficiency edge - high success rate expected"
        elif edge > 3.0:
            recommendation = "VALUE: Positive efficiency matchup"
        elif edge > -3.0:
            recommendation = "NEUTRAL: Even efficiency matchup"
        elif edge > -8.0:
            recommendation = "FADE: Defense has efficiency edge"
        else:
            recommendation = "STRONG FADE: Defense dominates efficiency"

        explanation = (
            f"Offensive Success Rate: {offensive_success_rate:.1f}% | "
            f"Defensive Success Rate Allowed: {defensive_success_rate_allowed:.1f}% | "
            f"Edge: {edge:+.1f}%"
        )

        return MatchupEdge(
            matchup_type='success_rate',
            offensive_value=offensive_success_rate,
            defensive_value=defensive_success_rate_allowed,
            edge=edge,
            edge_percentile=percentile,
            confidence=confidence,
            recommendation=recommendation,
            explanation=explanation
        )

    def _calculate_pressure_edge(
        self,
        ol_pressure_rate_allowed: float,
        dl_pressure_rate_generated: float
    ) -> MatchupEdge:
        """Calculate pressure matchup edge (OL vs DL).

        Args:
            ol_pressure_rate_allowed: QB's pressure rate faced
            dl_pressure_rate_generated: Defense's pressure rate generated

        Returns:
            MatchupEdge with pressure analysis
        """
        # Edge = DL pressure rate - OL pressure allowed
        # Positive = DL advantage (bad for QB), Negative = OL advantage (good for QB)
        edge = dl_pressure_rate_generated - ol_pressure_rate_allowed

        percentile = 50 - (edge * 2)  # Invert: negative edge = good for offense
        percentile = max(10, min(90, percentile))

        confidence = min(0.65 + abs(edge) * 0.02, 0.85)

        if edge < -8.0:
            recommendation = "STRONG VALUE: Elite OL protection vs weak pass rush"
        elif edge < -3.0:
            recommendation = "VALUE: OL has advantage, clean pocket expected"
        elif edge < 3.0:
            recommendation = "NEUTRAL: Even pressure matchup"
        elif edge < 8.0:
            recommendation = "FADE: DL has edge, pressure expected"
        else:
            recommendation = "STRONG FADE: Elite pass rush vs weak OL"

        explanation = (
            f"OL Pressure Rate Allowed: {ol_pressure_rate_allowed:.1f}% | "
            f"DL Pressure Rate Generated: {dl_pressure_rate_generated:.1f}% | "
            f"Edge: {edge:+.1f}% (negative = OL advantage)"
        )

        return MatchupEdge(
            matchup_type='pressure_rate',
            offensive_value=ol_pressure_rate_allowed,
            defensive_value=dl_pressure_rate_generated,
            edge=edge,
            edge_percentile=percentile,
            confidence=confidence,
            recommendation=recommendation,
            explanation=explanation
        )

    def _calculate_air_yards_edge(
        self,
        offensive_air_yards: float,
        defensive_air_yards_allowed: float
    ) -> MatchupEdge:
        """Calculate air yards matchup edge.

        Args:
            offensive_air_yards: Player's air yards per attempt
            defensive_air_yards_allowed: Defense's air yards allowed per attempt

        Returns:
            MatchupEdge with air yards analysis
        """
        # Edge = offensive air yards - defensive air yards allowed
        # Positive = offense throws deeper than defense typically allows
        edge = offensive_air_yards - defensive_air_yards_allowed

        percentile = 50 + (edge * 5)
        percentile = max(10, min(90, percentile))

        confidence = min(0.60 + abs(edge) * 0.05, 0.80)

        if edge > 2.0:
            recommendation = "VALUE: Deep passing game advantage"
        elif edge > 0.5:
            recommendation = "SLIGHT VALUE: Air yards advantage"
        elif edge > -0.5:
            recommendation = "NEUTRAL: Even air yards matchup"
        elif edge > -2.0:
            recommendation = "FADE: Defense limits deep passing"
        else:
            recommendation = "STRONG FADE: Defense excels vs deep passing"

        explanation = (
            f"Offensive Air Yards/Attempt: {offensive_air_yards:.1f} | "
            f"Defensive Air Yards Allowed: {defensive_air_yards_allowed:.1f} | "
            f"Edge: {edge:+.1f} yards"
        )

        return MatchupEdge(
            matchup_type='air_yards',
            offensive_value=offensive_air_yards,
            defensive_value=defensive_air_yards_allowed,
            edge=edge,
            edge_percentile=percentile,
            confidence=confidence,
            recommendation=recommendation,
            explanation=explanation
        )

    def generate_matchup_report(
        self,
        player_profile: OffensiveProfile,
        defense_profile: DefensiveProfile
    ) -> Dict:
        """Generate comprehensive matchup report.

        Args:
            player_profile: Player offensive profile
            defense_profile: Opposing defense profile

        Returns:
            Dict with all matchup edges and overall assessment
        """
        # Analyze based on position
        if player_profile.position == 'QB':
            edges = self.analyze_qb_vs_defense(player_profile, defense_profile)
        elif player_profile.position in ['RB', 'FB']:
            edges = self.analyze_rusher_vs_defense(player_profile, defense_profile)
        elif player_profile.position in ['WR', 'TE']:
            edges = self.analyze_receiver_vs_defense(player_profile, defense_profile)
        else:
            edges = []

        # Calculate overall matchup grade
        if edges:
            avg_percentile = statistics.mean([e.edge_percentile for e in edges])
            avg_confidence = statistics.mean([e.confidence for e in edges])

            if avg_percentile >= 75:
                overall_grade = "A"
                overall_recommendation = "STRONG PLAY"
            elif avg_percentile >= 65:
                overall_grade = "B"
                overall_recommendation = "GOOD PLAY"
            elif avg_percentile >= 55:
                overall_grade = "C"
                overall_recommendation = "NEUTRAL"
            elif avg_percentile >= 45:
                overall_grade = "D"
                overall_recommendation = "FADE"
            else:
                overall_grade = "F"
                overall_recommendation = "STRONG FADE"
        else:
            avg_percentile = 50
            avg_confidence = 0.5
            overall_grade = "N/A"
            overall_recommendation = "INSUFFICIENT DATA"

        return {
            'player': {
                'id': player_profile.player_id,
                'name': player_profile.player_name,
                'position': player_profile.position,
                'team': player_profile.team
            },
            'opponent_defense': defense_profile.team,
            'overall_matchup_grade': overall_grade,
            'overall_percentile': round(avg_percentile, 1),
            'overall_confidence': round(avg_confidence, 2),
            'overall_recommendation': overall_recommendation,
            'edges': [
                {
                    'type': edge.matchup_type,
                    'offensive_value': round(edge.offensive_value, 3),
                    'defensive_value': round(edge.defensive_value, 3),
                    'edge': round(edge.edge, 3),
                    'percentile': edge.edge_percentile,
                    'confidence': round(edge.confidence, 2),
                    'recommendation': edge.recommendation,
                    'explanation': edge.explanation
                }
                for edge in edges
            ]
        }


# Singleton instance
matchup_analyzer = MatchupAnalyzer()
