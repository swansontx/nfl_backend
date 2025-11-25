"""Injury Impact Deep Analysis System.

Quantifies injury impacts and automatically adjusts projections when key players are out.
Includes usage redistribution, beneficiary identification, and team total adjustments.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Import validated weights from backtesting
from backend.config import INJURY_REDISTRIBUTION


@dataclass
class InjuryImpact:
    """Quantified impact of a player injury."""
    injured_player: str
    player_id: str
    position: str
    team: str
    injury_status: str  # OUT, DOUBTFUL, QUESTIONABLE

    # Player's typical usage
    avg_targets_per_game: float = 0.0
    avg_carries_per_game: float = 0.0
    avg_snap_percentage: float = 0.0

    # Impact on team
    team_total_impact: float = 0.0  # Points impact on team total
    opponent_total_impact: float = 0.0  # Impact on opponent (if defensive player)

    # Beneficiaries (who gains from injury)
    beneficiaries: List['PlayerBenefit'] = None

    # Confidence in impact estimation
    confidence: float = 0.5  # 0-1 based on data quality

    def __post_init__(self):
        if self.beneficiaries is None:
            self.beneficiaries = []


@dataclass
class PlayerBenefit:
    """How much a player benefits from teammate's injury."""
    player: str
    player_id: str
    position: str

    # Usage increases
    target_increase: float = 0.0
    carry_increase: float = 0.0
    snap_increase: float = 0.0

    # Projected stat increases
    receiving_yards_increase: float = 0.0
    rushing_yards_increase: float = 0.0
    receptions_increase: float = 0.0

    # Scoring increases
    td_probability_increase: float = 0.0

    # Confidence
    confidence: float = 0.5


@dataclass
class ProjectionAdjustment:
    """Adjustment to apply to a player's projection."""
    player: str
    player_id: str
    stat_type: str  # 'receiving_yards', 'rushing_yards', 'receptions', etc.

    base_projection: float
    adjustment: float
    adjusted_projection: float

    reason: str
    confidence: float


class InjuryImpactAnalyzer:
    """Deep analysis of injury impacts with automatic projection adjustments."""

    def __init__(self, season: int = 2025):
        """Initialize analyzer.

        Args:
            season: NFL season year
        """
        self.season = season
        self.inputs_dir = Path('inputs')
        self.data_dir = Path('data/injury_impacts')
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Historical impact database
        self.historical_impacts = self._load_historical_impacts()

        # Position-based redistribution patterns
        self.redistribution_patterns = self._initialize_redistribution_patterns()

    def _load_historical_impacts(self) -> pd.DataFrame:
        """Load historical injury impact data.

        Returns:
            DataFrame with historical injury impacts
        """
        impact_file = self.data_dir / 'historical_injury_impacts.csv'

        if impact_file.exists():
            return pd.read_csv(impact_file)
        else:
            # Create empty DataFrame with schema
            return pd.DataFrame(columns=[
                'season', 'week', 'injured_player', 'position', 'team',
                'backup_player', 'backup_position',
                'target_shift', 'carry_shift', 'yards_shift',
                'team_total_impact', 'games_missed'
            ])

    def _initialize_redistribution_patterns(self) -> Dict:
        """Initialize usage redistribution patterns based on position.

        Uses validated weights from historical backtesting when available.
        Falls back to default patterns if backtesting hasn't been run yet.

        Returns:
            Dictionary of redistribution patterns by position
        """
        # Use validated weights from backtesting
        validated_patterns = {}

        # Convert validated weights to expected format
        for position, scenarios in INJURY_REDISTRIBUTION.items():
            if position == 'metadata':
                continue  # Skip metadata

            validated_patterns[position] = {}

            for scenario, beneficiaries in scenarios.items():
                if isinstance(beneficiaries, dict):
                    validated_patterns[position][scenario] = {}

                    for beneficiary, stats in beneficiaries.items():
                        if isinstance(stats, dict) and 'target_share' in stats:
                            # Convert from validated weights format
                            validated_patterns[position][scenario][beneficiary] = {
                                'targets': stats.get('target_share', 0.0),
                                'carries': stats.get('carry_share', 0.0),
                                'yards': stats.get('yards_per_target', 0.0) * stats.get('target_share', 0.0),
                                'confidence': stats.get('confidence', 0.5)
                            }
                        elif beneficiary == 'team_total_impact':
                            validated_patterns[position][scenario]['team_total_impact'] = stats

        return validated_patterns if validated_patterns else {
            # Fallback to default patterns if config is empty
            'WR': {
                # When WR1 out
                'WR1_OUT': {
                    'WR2': {'targets': 0.25, 'yards': 0.20, 'confidence': 0.8},
                    'WR3': {'targets': 0.15, 'yards': 0.12, 'confidence': 0.7},
                    'TE': {'targets': 0.10, 'yards': 0.08, 'confidence': 0.6},
                    'team_total_impact': -2.5
                },
                # When WR2 out
                'WR2_OUT': {
                    'WR1': {'targets': 0.10, 'yards': 0.08, 'confidence': 0.7},
                    'WR3': {'targets': 0.20, 'yards': 0.15, 'confidence': 0.8},
                    'TE': {'targets': 0.08, 'yards': 0.06, 'confidence': 0.6},
                    'team_total_impact': -1.5
                }
            },
            'RB': {
                # When RB1 out
                'RB1_OUT': {
                    'RB2': {'carries': 0.60, 'targets': 0.40, 'yards': 0.55, 'confidence': 0.9},
                    'RB3': {'carries': 0.20, 'targets': 0.10, 'yards': 0.15, 'confidence': 0.6},
                    'team_total_impact': -3.5
                },
                # When RB2 out
                'RB2_OUT': {
                    'RB1': {'carries': 0.15, 'targets': 0.10, 'yards': 0.12, 'confidence': 0.8},
                    'RB3': {'carries': 0.40, 'targets': 0.30, 'yards': 0.35, 'confidence': 0.7},
                    'team_total_impact': -1.0
                }
            },
            'TE': {
                # When TE1 out
                'TE1_OUT': {
                    'TE2': {'targets': 0.50, 'yards': 0.45, 'confidence': 0.8},
                    'WR2': {'targets': 0.15, 'yards': 0.12, 'confidence': 0.6},
                    'WR3': {'targets': 0.10, 'yards': 0.08, 'confidence': 0.5},
                    'team_total_impact': -2.0
                }
            },
            'QB': {
                # When QB1 out
                'QB1_OUT': {
                    'team_total_impact': -7.0,  # Backup QB = major impact
                    'confidence': 0.9
                }
            }
        }

    def analyze_injury(
        self,
        injured_player: str,
        player_id: str,
        position: str,
        team: str,
        injury_status: str,
        week: int
    ) -> InjuryImpact:
        """Analyze impact of a player injury.

        Args:
            injured_player: Player name
            player_id: Player ID
            position: Position (WR, RB, TE, QB)
            team: Team abbreviation
            injury_status: OUT, DOUBTFUL, QUESTIONABLE
            week: Current week

        Returns:
            InjuryImpact with quantified impacts
        """
        # Get player's typical usage
        usage = self._get_player_usage(player_id, team, week)

        # Create impact object
        impact = InjuryImpact(
            injured_player=injured_player,
            player_id=player_id,
            position=position,
            team=team,
            injury_status=injury_status,
            avg_targets_per_game=usage.get('targets', 0),
            avg_carries_per_game=usage.get('carries', 0),
            avg_snap_percentage=usage.get('snap_pct', 0)
        )

        # Only calculate impacts for OUT status
        if injury_status == 'OUT':
            # Calculate team total impact
            impact.team_total_impact = self._calculate_team_total_impact(
                position, usage
            )

            # Identify beneficiaries
            impact.beneficiaries = self._identify_beneficiaries(
                team, position, usage, week
            )

            # Calculate confidence based on data quality
            impact.confidence = self._calculate_confidence(usage, impact.beneficiaries)

        elif injury_status == 'DOUBTFUL':
            # 75% chance of being out - scale impacts
            impact.team_total_impact = self._calculate_team_total_impact(position, usage) * 0.75
            impact.beneficiaries = self._identify_beneficiaries(team, position, usage, week)
            for beneficiary in impact.beneficiaries:
                beneficiary.target_increase *= 0.75
                beneficiary.carry_increase *= 0.75
                beneficiary.receiving_yards_increase *= 0.75
                beneficiary.rushing_yards_increase *= 0.75
            impact.confidence = self._calculate_confidence(usage, impact.beneficiaries) * 0.8

        elif injury_status == 'QUESTIONABLE':
            # 50% chance of being out - scale impacts
            impact.team_total_impact = self._calculate_team_total_impact(position, usage) * 0.5
            impact.beneficiaries = self._identify_beneficiaries(team, position, usage, week)
            for beneficiary in impact.beneficiaries:
                beneficiary.target_increase *= 0.5
                beneficiary.carry_increase *= 0.5
                beneficiary.receiving_yards_increase *= 0.5
                beneficiary.rushing_yards_increase *= 0.5
            impact.confidence = self._calculate_confidence(usage, impact.beneficiaries) * 0.6

        return impact

    def _get_player_usage(
        self,
        player_id: str,
        team: str,
        week: int
    ) -> Dict[str, float]:
        """Get player's typical usage stats.

        Args:
            player_id: Player ID
            team: Team abbreviation
            week: Current week

        Returns:
            Dictionary with usage stats
        """
        try:
            # Load player stats
            stats_file = self.inputs_dir / f'{self.season}_player_stats.parquet'
            if not stats_file.exists():
                return {}

            stats = pd.read_parquet(stats_file)

            # Filter for this player, before this week
            player_stats = stats[
                (stats['player_id'] == player_id) &
                (stats['week'] < week)
            ]

            if len(player_stats) == 0:
                return {}

            # Calculate averages
            usage = {
                'targets': player_stats['targets'].mean() if 'targets' in player_stats else 0,
                'carries': player_stats['carries'].mean() if 'carries' in player_stats else 0,
                'receptions': player_stats['receptions'].mean() if 'receptions' in player_stats else 0,
                'receiving_yards': player_stats['receiving_yards'].mean() if 'receiving_yards' in player_stats else 0,
                'rushing_yards': player_stats['rushing_yards'].mean() if 'rushing_yards' in player_stats else 0,
                'snap_pct': player_stats['snap_pct'].mean() if 'snap_pct' in player_stats else 0
            }

            return usage

        except Exception as e:
            print(f"Error getting player usage: {e}")
            return {}

    def _calculate_team_total_impact(
        self,
        position: str,
        usage: Dict[str, float]
    ) -> float:
        """Calculate impact on team total points.

        Args:
            position: Player position
            usage: Player's typical usage

        Returns:
            Points impact on team total
        """
        # Get base impact from redistribution patterns
        base_impact = 0.0

        if position in ['WR', 'RB', 'TE']:
            # Determine if WR1, WR2, RB1, RB2, etc. based on usage
            role = self._determine_player_role(position, usage)

            if role and role in self.redistribution_patterns.get(position, {}):
                base_impact = self.redistribution_patterns[position][role].get('team_total_impact', 0)

        elif position == 'QB':
            base_impact = self.redistribution_patterns['QB']['QB1_OUT']['team_total_impact']

        return base_impact

    def _determine_player_role(
        self,
        position: str,
        usage: Dict[str, float]
    ) -> Optional[str]:
        """Determine player's role (WR1, WR2, RB1, etc.) based on usage.

        Args:
            position: Player position
            usage: Usage stats

        Returns:
            Role string (e.g., 'WR1_OUT')
        """
        if position in ['WR', 'TE']:
            targets = usage.get('targets', 0)
            if targets >= 7.0:
                return f'{position}1_OUT'
            elif targets >= 4.0:
                return f'{position}2_OUT'

        elif position == 'RB':
            carries = usage.get('carries', 0)
            if carries >= 12.0:
                return 'RB1_OUT'
            elif carries >= 6.0:
                return 'RB2_OUT'

        return None

    def _identify_beneficiaries(
        self,
        team: str,
        injured_position: str,
        injured_usage: Dict[str, float],
        week: int
    ) -> List[PlayerBenefit]:
        """Identify players who benefit from teammate's injury.

        Args:
            team: Team abbreviation
            injured_position: Injured player's position
            injured_usage: Injured player's usage stats
            week: Current week

        Returns:
            List of PlayerBenefit objects
        """
        beneficiaries = []

        # Get redistribution pattern
        role = self._determine_player_role(injured_position, injured_usage)
        if not role or injured_position not in self.redistribution_patterns:
            return beneficiaries

        pattern = self.redistribution_patterns[injured_position].get(role, {})

        # Get team roster
        roster = self._get_team_roster(team, week)

        # Apply redistribution to eligible players
        for beneficiary_pos, redistribution in pattern.items():
            if beneficiary_pos == 'team_total_impact':
                continue

            # Find players at this position
            eligible = [p for p in roster if p['position'] == beneficiary_pos]

            for player in eligible:
                benefit = PlayerBenefit(
                    player=player['name'],
                    player_id=player['player_id'],
                    position=player['position']
                )

                # Calculate increases
                if 'targets' in redistribution:
                    benefit.target_increase = injured_usage.get('targets', 0) * redistribution['targets']
                    # Estimate yards increase (7 yards per target)
                    benefit.receiving_yards_increase = benefit.target_increase * 7.0
                    # Estimate reception increase (65% catch rate)
                    benefit.receptions_increase = benefit.target_increase * 0.65

                if 'carries' in redistribution:
                    benefit.carry_increase = injured_usage.get('carries', 0) * redistribution['carries']
                    # Estimate yards increase (4.5 yards per carry)
                    benefit.rushing_yards_increase = benefit.carry_increase * 4.5

                if 'yards' in redistribution:
                    # Direct yards redistribution
                    if injured_position in ['WR', 'TE']:
                        benefit.receiving_yards_increase = injured_usage.get('receiving_yards', 0) * redistribution['yards']
                    else:
                        benefit.rushing_yards_increase = injured_usage.get('rushing_yards', 0) * redistribution['yards']

                benefit.confidence = redistribution.get('confidence', 0.5)

                # Only add if meaningful increase
                if benefit.target_increase > 0.5 or benefit.carry_increase > 0.5:
                    beneficiaries.append(benefit)

        return beneficiaries

    def _get_team_roster(self, team: str, week: int) -> List[Dict]:
        """Get team roster for a week.

        Args:
            team: Team abbreviation
            week: Week number

        Returns:
            List of player dictionaries
        """
        try:
            # Load roster data
            roster_file = self.inputs_dir / f'{self.season}_rosters.parquet'
            if not roster_file.exists():
                return []

            rosters = pd.read_parquet(roster_file)
            team_roster = rosters[rosters['team'] == team]

            players = []
            for _, player in team_roster.iterrows():
                players.append({
                    'name': player.get('player_name', ''),
                    'player_id': player.get('player_id', ''),
                    'position': player.get('position', '')
                })

            return players

        except Exception as e:
            print(f"Error loading roster: {e}")
            return []

    def _calculate_confidence(
        self,
        usage: Dict[str, float],
        beneficiaries: List[PlayerBenefit]
    ) -> float:
        """Calculate confidence in impact estimation.

        Args:
            usage: Injured player's usage stats
            beneficiaries: List of beneficiaries

        Returns:
            Confidence score (0-1)
        """
        confidence = 0.5  # Base confidence

        # Higher confidence if we have good usage data
        if usage.get('targets', 0) > 5 or usage.get('carries', 0) > 10:
            confidence += 0.2

        # Higher confidence if we identified clear beneficiaries
        if len(beneficiaries) > 0:
            confidence += 0.2

        # Lower confidence if usage data is sparse
        if usage.get('targets', 0) < 2 and usage.get('carries', 0) < 3:
            confidence -= 0.2

        return max(0.1, min(1.0, confidence))

    def generate_projection_adjustments(
        self,
        injury_impact: InjuryImpact
    ) -> List[ProjectionAdjustment]:
        """Generate projection adjustments for beneficiaries.

        Args:
            injury_impact: InjuryImpact object

        Returns:
            List of ProjectionAdjustment objects
        """
        adjustments = []

        for beneficiary in injury_impact.beneficiaries:
            # Receiving yards adjustment
            if beneficiary.receiving_yards_increase > 5:
                # Would need to fetch base projection here
                # For now, just create the adjustment structure
                adjustments.append(ProjectionAdjustment(
                    player=beneficiary.player,
                    player_id=beneficiary.player_id,
                    stat_type='receiving_yards',
                    base_projection=0.0,  # Would fetch actual projection
                    adjustment=beneficiary.receiving_yards_increase,
                    adjusted_projection=0.0,  # Would calculate
                    reason=f"{injury_impact.injured_player} OUT → +{beneficiary.target_increase:.1f} targets",
                    confidence=beneficiary.confidence
                ))

            # Rushing yards adjustment
            if beneficiary.rushing_yards_increase > 5:
                adjustments.append(ProjectionAdjustment(
                    player=beneficiary.player,
                    player_id=beneficiary.player_id,
                    stat_type='rushing_yards',
                    base_projection=0.0,
                    adjustment=beneficiary.rushing_yards_increase,
                    adjusted_projection=0.0,
                    reason=f"{injury_impact.injured_player} OUT → +{beneficiary.carry_increase:.1f} carries",
                    confidence=beneficiary.confidence
                ))

        return adjustments


# Singleton instance
injury_impact_analyzer = InjuryImpactAnalyzer()


if __name__ == "__main__":
    # Test analyzer
    analyzer = InjuryImpactAnalyzer(season=2025)

    # Test injury analysis
    impact = analyzer.analyze_injury(
        injured_player="Travis Kelce",
        player_id="kelce_travis",
        position="TE",
        team="KC",
        injury_status="OUT",
        week=12
    )

    print(f"Injury Impact Analysis: {impact.injured_player}")
    print(f"Team Total Impact: {impact.team_total_impact:.1f} points")
    print(f"Confidence: {impact.confidence:.2f}")
    print(f"\nBeneficiaries:")
    for beneficiary in impact.beneficiaries:
        print(f"  {beneficiary.player}:")
        if beneficiary.target_increase > 0:
            print(f"    +{beneficiary.target_increase:.1f} targets")
            print(f"    +{beneficiary.receiving_yards_increase:.1f} receiving yards")
        if beneficiary.carry_increase > 0:
            print(f"    +{beneficiary.carry_increase:.1f} carries")
            print(f"    +{beneficiary.rushing_yards_increase:.1f} rushing yards")
