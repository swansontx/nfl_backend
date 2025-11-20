"""Injury Impact Analyzer for betting implications.

This module analyzes injuries and their cascading effects on:
- Team performance and game outcomes
- Player prop redistribution (targets, carries, etc.)
- Backup player opportunities
- Betting recommendations

Key Features:
- Depth chart lookup for replacements
- Target/touch redistribution modeling
- Position importance scoring
- Prop betting implications
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class PlayerDepthInfo:
    """Player depth chart information."""
    name: str
    position: str
    depth_position: str  # e.g., 'QB', 'RB', 'WR'
    depth_rank: int  # 1=starter, 2=backup, 3=third string
    team: str
    gsis_id: str = ""


@dataclass
class InjuryImpact:
    """Impact assessment for an injury."""
    injured_player: str
    position: str
    team: str
    status: str  # OUT, DOUBTFUL, QUESTIONABLE
    replacement: Optional[str]
    replacement_depth: int
    team_impact_score: float  # 0-100 scale
    prop_implications: List[Dict]
    narrative: str
    betting_recommendations: List[str]


@dataclass
class TeamInjuryReport:
    """Comprehensive injury report for a team."""
    team: str
    total_impact_score: float
    key_injuries: List[InjuryImpact]
    prop_redistributions: Dict[str, List[Dict]]  # position -> list of affected players
    summary: str
    betting_angle: str


class InjuryImpactAnalyzer:
    """Analyzes injury impacts for betting purposes."""

    # Position importance multipliers for team impact
    POSITION_IMPORTANCE = {
        'QB': 10.0,
        'RB': 6.0,
        'WR': 5.5,
        'TE': 4.0,
        'LT': 5.0,
        'RT': 4.5,
        'LG': 3.5,
        'RG': 3.5,
        'C': 4.0,
        'OL': 4.0,
        'T': 4.5,
        'G': 3.5,
        'DE': 4.0,
        'DT': 3.5,
        'NT': 3.0,
        'OLB': 3.5,
        'ILB': 3.5,
        'LB': 3.5,
        'CB': 4.0,
        'FS': 3.5,
        'SS': 3.5,
        'S': 3.5,
        'K': 2.0,
        'P': 1.5
    }

    # Typical target/touch shares by position and depth
    TARGET_SHARES = {
        'WR': {1: 0.25, 2: 0.18, 3: 0.10, 4: 0.05},  # % of team targets
        'TE': {1: 0.15, 2: 0.05, 3: 0.02},
        'RB': {1: 0.12, 2: 0.06, 3: 0.02},
    }

    CARRY_SHARES = {
        'RB': {1: 0.65, 2: 0.25, 3: 0.08},  # % of team carries
    }

    # Drop-off multipliers when going from starter to backup
    DROPOFF_MULTIPLIERS = {
        'QB': {1: 1.0, 2: 0.70, 3: 0.50},  # Production relative to starter
        'RB': {1: 1.0, 2: 0.80, 3: 0.60},
        'WR': {1: 1.0, 2: 0.75, 3: 0.55},
        'TE': {1: 1.0, 2: 0.65, 3: 0.45},
        'OL': {1: 1.0, 2: 0.75, 3: 0.55},
    }

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize analyzer with data directory."""
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent / 'inputs'
        else:
            self.data_dir = Path(data_dir)

        self.depth_charts: Dict[str, List[PlayerDepthInfo]] = {}
        self.rosters: Dict[str, Dict] = {}
        self._load_data()

    def _load_data(self):
        """Load depth chart and roster data."""
        # Load depth charts
        depth_chart_file = self.data_dir / 'depth_charts_2024_2025.csv'
        if depth_chart_file.exists():
            self._load_depth_charts(depth_chart_file)

        # Load rosters for additional context
        roster_file = self.data_dir / 'rosters_weekly_2024_2025.csv'
        if roster_file.exists():
            self._load_rosters(roster_file)

    def _load_depth_charts(self, filepath: Path):
        """Load depth charts from CSV."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    team = row.get('club_code', '')
                    if not team:
                        continue

                    if team not in self.depth_charts:
                        self.depth_charts[team] = []

                    # Parse depth rank from depth_team
                    depth_team_val = row.get('depth_team', '1') or '1'
                    try:
                        depth_rank = int(depth_team_val)
                    except (ValueError, TypeError):
                        depth_rank = 1  # Default to starter if NA or invalid

                    player = PlayerDepthInfo(
                        name=row.get('full_name', row.get('player_name', '')),
                        position=row.get('position', ''),
                        depth_position=row.get('depth_position', ''),
                        depth_rank=depth_rank,
                        team=team,
                        gsis_id=row.get('gsis_id', '')
                    )

                    self.depth_charts[team].append(player)

        except Exception as e:
            print(f"Error loading depth charts: {e}")

    def _load_rosters(self, filepath: Path):
        """Load roster data from CSV."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    team = row.get('team', '')
                    player_name = row.get('full_name', '')

                    if not team or not player_name:
                        continue

                    if team not in self.rosters:
                        self.rosters[team] = {}

                    # Store latest roster info for each player
                    self.rosters[team][player_name] = {
                        'position': row.get('position', ''),
                        'depth_chart_position': row.get('depth_chart_position', ''),
                        'status': row.get('status', ''),
                        'gsis_id': row.get('gsis_id', '')
                    }

        except Exception as e:
            print(f"Error loading rosters: {e}")

    def get_replacement(self, team: str, position: str, injured_player: str) -> Tuple[Optional[str], int]:
        """Find replacement player from depth chart.

        Args:
            team: Team abbreviation
            position: Position (QB, RB, WR, etc.)
            injured_player: Name of injured player

        Returns:
            Tuple of (replacement_name, depth_rank) or (None, 0)
        """
        if team not in self.depth_charts:
            return None, 0

        # Normalize position for matching
        pos_variants = self._get_position_variants(position)

        # Find all players at this position, sorted by depth
        position_players = []
        for player in self.depth_charts[team]:
            if player.position in pos_variants or player.depth_position in pos_variants:
                position_players.append(player)

        # Sort by depth rank
        position_players.sort(key=lambda p: p.depth_rank)

        # Find the injured player's rank and get next in line
        injured_rank = None
        for player in position_players:
            if injured_player.lower() in player.name.lower():
                injured_rank = player.depth_rank
                break

        # Get next player after injured one
        for player in position_players:
            if injured_rank is not None and player.depth_rank > injured_rank:
                return player.name, player.depth_rank
            elif injured_rank is None and player.depth_rank == 2:
                # If we can't find injured player, assume they're starter
                return player.name, player.depth_rank

        return None, 0

    def _get_position_variants(self, position: str) -> List[str]:
        """Get position variants for matching."""
        position = position.upper()

        variants = {
            'QB': ['QB'],
            'RB': ['RB', 'HB', 'FB'],
            'WR': ['WR', 'FL', 'SE', 'SWR'],
            'TE': ['TE'],
            'OL': ['OL', 'T', 'G', 'C', 'LT', 'RT', 'LG', 'RG'],
            'T': ['T', 'LT', 'RT', 'OL'],
            'G': ['G', 'LG', 'RG', 'OL'],
            'C': ['C', 'OL'],
            'DL': ['DL', 'DE', 'DT', 'NT'],
            'DE': ['DE', 'DL'],
            'DT': ['DT', 'DL', 'NT'],
            'LB': ['LB', 'OLB', 'ILB', 'MLB'],
            'OLB': ['OLB', 'LB'],
            'ILB': ['ILB', 'LB', 'MLB'],
            'CB': ['CB', 'DB', 'NB'],
            'S': ['S', 'FS', 'SS', 'DB'],
            'FS': ['FS', 'S', 'DB'],
            'SS': ['SS', 'S', 'DB'],
        }

        return variants.get(position, [position])

    def get_depth_at_position(self, team: str, position: str) -> List[PlayerDepthInfo]:
        """Get all players at a position sorted by depth."""
        if team not in self.depth_charts:
            return []

        pos_variants = self._get_position_variants(position)

        players = []
        seen = set()
        for player in self.depth_charts[team]:
            if player.position in pos_variants or player.depth_position in pos_variants:
                # Avoid duplicates
                key = f"{player.name}_{player.depth_rank}"
                if key not in seen:
                    players.append(player)
                    seen.add(key)

        players.sort(key=lambda p: p.depth_rank)
        return players

    def calculate_prop_redistribution(self, team: str, position: str,
                                      injured_player: str, status: str) -> List[Dict]:
        """Calculate how targets/touches redistribute when player is out.

        Args:
            team: Team abbreviation
            position: Position of injured player
            injured_player: Name of injured player
            status: Injury status (OUT, DOUBTFUL, QUESTIONABLE)

        Returns:
            List of dicts with player name, expected change, and prop implications
        """
        redistributions = []

        # Determine probability of missing game
        miss_probability = {
            'OUT': 1.0,
            'DOUBTFUL': 0.75,
            'QUESTIONABLE': 0.50,
            'IR': 1.0,
            'INJURED RESERVE': 1.0,
            'PUP': 1.0
        }.get(status.upper(), 0.25)

        position = position.upper()

        if position == 'WR':
            redistributions = self._redistribute_wr_targets(team, injured_player, miss_probability)
        elif position == 'RB':
            redistributions = self._redistribute_rb_touches(team, injured_player, miss_probability)
        elif position == 'TE':
            redistributions = self._redistribute_te_targets(team, injured_player, miss_probability)
        elif position == 'QB':
            redistributions = self._qb_injury_impact(team, injured_player, miss_probability)

        return redistributions

    def _redistribute_wr_targets(self, team: str, injured_wr: str, miss_prob: float) -> List[Dict]:
        """Redistribute WR targets when a receiver is out."""
        results = []

        # Get all WRs on team
        wrs = self.get_depth_at_position(team, 'WR')
        tes = self.get_depth_at_position(team, 'TE')
        rbs = self.get_depth_at_position(team, 'RB')

        # Find injured WR's depth
        injured_depth = 1
        for wr in wrs:
            if injured_wr.lower() in wr.name.lower():
                injured_depth = wr.depth_rank
                break

        # Injured WR's lost target share
        lost_share = self.TARGET_SHARES['WR'].get(injured_depth, 0.15) * miss_prob

        # Redistribute primarily to other WRs
        for wr in wrs:
            if injured_wr.lower() not in wr.name.lower():
                if wr.depth_rank <= injured_depth + 1:
                    # Players at same depth or one below get boost
                    boost_share = lost_share * 0.35  # 35% of lost targets
                    if boost_share > 0.02:  # Only report significant changes
                        results.append({
                            'player': wr.name,
                            'position': 'WR',
                            'change_type': 'targets_increase',
                            'expected_boost_pct': round(boost_share * 100, 1),
                            'miss_probability': miss_prob,
                            'prop_recommendation': f"OVER on {wr.name} receptions/receiving yards",
                            'confidence': 0.7 if miss_prob > 0.7 else 0.5
                        })
                        break  # Main beneficiary found

        # TE may see increase
        if tes:
            te_boost = lost_share * 0.25
            if te_boost > 0.02:
                results.append({
                    'player': tes[0].name,
                    'position': 'TE',
                    'change_type': 'targets_increase',
                    'expected_boost_pct': round(te_boost * 100, 1),
                    'miss_probability': miss_prob,
                    'prop_recommendation': f"OVER on {tes[0].name} receptions",
                    'confidence': 0.6 if miss_prob > 0.7 else 0.4
                })

        # RB may see more checkdowns
        if rbs:
            rb_boost = lost_share * 0.15
            if rb_boost > 0.01:
                results.append({
                    'player': rbs[0].name,
                    'position': 'RB',
                    'change_type': 'targets_increase',
                    'expected_boost_pct': round(rb_boost * 100, 1),
                    'miss_probability': miss_prob,
                    'prop_recommendation': f"OVER on {rbs[0].name} receptions",
                    'confidence': 0.5 if miss_prob > 0.7 else 0.35
                })

        return results

    def _redistribute_rb_touches(self, team: str, injured_rb: str, miss_prob: float) -> List[Dict]:
        """Redistribute RB touches when a running back is out."""
        results = []

        rbs = self.get_depth_at_position(team, 'RB')
        wrs = self.get_depth_at_position(team, 'WR')

        # Find injured RB's depth
        injured_depth = 1
        for rb in rbs:
            if injured_rb.lower() in rb.name.lower():
                injured_depth = rb.depth_rank
                break

        # Lost carry share
        lost_carries = self.CARRY_SHARES['RB'].get(injured_depth, 0.50) * miss_prob
        lost_targets = self.TARGET_SHARES['RB'].get(injured_depth, 0.10) * miss_prob

        # Primary backup gets most carries
        for rb in rbs:
            if injured_rb.lower() not in rb.name.lower():
                carry_boost = lost_carries * 0.75
                target_boost = lost_targets * 0.50

                if carry_boost > 0.05:
                    results.append({
                        'player': rb.name,
                        'position': 'RB',
                        'change_type': 'workload_increase',
                        'expected_carry_boost_pct': round(carry_boost * 100, 1),
                        'expected_target_boost_pct': round(target_boost * 100, 1),
                        'miss_probability': miss_prob,
                        'prop_recommendation': f"OVER on {rb.name} rushing yards/attempts AND receptions",
                        'confidence': 0.8 if miss_prob > 0.7 else 0.6
                    })
                    break

        # Team may pass more
        if wrs and lost_carries > 0.30:
            results.append({
                'player': wrs[0].name if wrs else 'WR1',
                'position': 'WR',
                'change_type': 'scheme_shift',
                'note': 'Team may shift to pass-heavy approach',
                'expected_boost_pct': round(lost_carries * 15, 1),  # ~15% of lost runs become passes
                'miss_probability': miss_prob,
                'prop_recommendation': f"Consider OVER on team passing attempts, {wrs[0].name if wrs else 'WR1'} targets",
                'confidence': 0.55
            })

        return results

    def _redistribute_te_targets(self, team: str, injured_te: str, miss_prob: float) -> List[Dict]:
        """Redistribute TE targets when tight end is out."""
        results = []

        tes = self.get_depth_at_position(team, 'TE')
        wrs = self.get_depth_at_position(team, 'WR')
        rbs = self.get_depth_at_position(team, 'RB')

        # Lost target share
        lost_share = self.TARGET_SHARES['TE'].get(1, 0.15) * miss_prob

        # Backup TE
        for te in tes:
            if injured_te.lower() not in te.name.lower():
                te_boost = lost_share * 0.40
                if te_boost > 0.02:
                    results.append({
                        'player': te.name,
                        'position': 'TE',
                        'change_type': 'targets_increase',
                        'expected_boost_pct': round(te_boost * 100, 1),
                        'miss_probability': miss_prob,
                        'prop_recommendation': f"OVER on {te.name} receptions",
                        'confidence': 0.6 if miss_prob > 0.7 else 0.4
                    })
                    break

        # Slot WR likely benefits most
        if len(wrs) >= 3:
            slot_wr = wrs[2] if len(wrs) > 2 else wrs[1]
            wr_boost = lost_share * 0.35
            results.append({
                'player': slot_wr.name,
                'position': 'WR',
                'change_type': 'targets_increase',
                'note': 'Slot/middle-field routes absorb TE targets',
                'expected_boost_pct': round(wr_boost * 100, 1),
                'miss_probability': miss_prob,
                'prop_recommendation': f"OVER on {slot_wr.name} receptions",
                'confidence': 0.55
            })

        return results

    def _qb_injury_impact(self, team: str, injured_qb: str, miss_prob: float) -> List[Dict]:
        """Calculate cascading effects of QB injury."""
        results = []

        qbs = self.get_depth_at_position(team, 'QB')
        rbs = self.get_depth_at_position(team, 'RB')
        wrs = self.get_depth_at_position(team, 'WR')

        # Find backup QB
        backup_qb = None
        for qb in qbs:
            if injured_qb.lower() not in qb.name.lower():
                backup_qb = qb.name
                break

        if backup_qb:
            # Backup QB typically has lower efficiency
            dropoff = self.DROPOFF_MULTIPLIERS['QB'].get(2, 0.70)

            results.append({
                'player': backup_qb,
                'position': 'QB',
                'change_type': 'starter_replacement',
                'expected_efficiency': round(dropoff * 100, 1),
                'miss_probability': miss_prob,
                'prop_recommendation': f"UNDER on {backup_qb} passing yards/TDs (but OVER on conservative metrics)",
                'confidence': 0.75 if miss_prob > 0.7 else 0.55
            })

        # RB likely sees increased role
        if rbs:
            results.append({
                'player': rbs[0].name,
                'position': 'RB',
                'change_type': 'scheme_adjustment',
                'note': 'Team will likely run more with backup QB',
                'expected_boost_pct': 15.0,
                'miss_probability': miss_prob,
                'prop_recommendation': f"OVER on {rbs[0].name} rushing attempts/yards",
                'confidence': 0.70 if miss_prob > 0.7 else 0.5
            })

        # Deep threat WRs suffer most
        if wrs:
            results.append({
                'player': wrs[0].name,
                'position': 'WR',
                'change_type': 'scheme_adjustment',
                'note': 'WRs see reduction with backup QB, especially deep threats',
                'expected_reduction_pct': 20.0,
                'miss_probability': miss_prob,
                'prop_recommendation': f"UNDER on {wrs[0].name} receiving yards (but not receptions)",
                'confidence': 0.65 if miss_prob > 0.7 else 0.45
            })

        return results

    def analyze_injury(self, team: str, player_name: str, position: str,
                       status: str, injury_detail: str = "") -> InjuryImpact:
        """Analyze the full impact of a single injury.

        Args:
            team: Team abbreviation
            player_name: Name of injured player
            position: Player position
            status: Injury status
            injury_detail: Optional injury description

        Returns:
            InjuryImpact object with full analysis
        """
        # Get replacement
        replacement, replacement_depth = self.get_replacement(team, position, player_name)

        # Calculate team impact score
        base_importance = self.POSITION_IMPORTANCE.get(position.upper(), 3.0)

        # Adjust for status severity
        status_multiplier = {
            'OUT': 1.0,
            'DOUBTFUL': 0.75,
            'QUESTIONABLE': 0.40,
            'IR': 1.0,
            'PUP': 1.0
        }.get(status.upper(), 0.3)

        # Adjust for drop-off quality
        if replacement:
            dropoff = self.DROPOFF_MULTIPLIERS.get(position.upper(), {}).get(replacement_depth, 0.70)
            quality_penalty = (1.0 - dropoff) * 10
        else:
            quality_penalty = 5.0  # Unknown backup

        team_impact = min(100, base_importance * status_multiplier * 10 + quality_penalty)

        # Get prop redistributions
        prop_implications = self.calculate_prop_redistribution(team, position, player_name, status)

        # Generate narrative
        narrative = self._generate_injury_narrative(
            team, player_name, position, status, replacement,
            replacement_depth, team_impact, injury_detail
        )

        # Generate betting recommendations
        recommendations = self._generate_betting_recommendations(
            team, player_name, position, status, replacement, prop_implications
        )

        return InjuryImpact(
            injured_player=player_name,
            position=position,
            team=team,
            status=status,
            replacement=replacement,
            replacement_depth=replacement_depth,
            team_impact_score=round(team_impact, 1),
            prop_implications=prop_implications,
            narrative=narrative,
            betting_recommendations=recommendations
        )

    def _generate_injury_narrative(self, team: str, player: str, position: str,
                                   status: str, replacement: str, depth: int,
                                   impact: float, detail: str) -> str:
        """Generate human-readable injury impact narrative."""

        severity = "critical" if impact >= 70 else "significant" if impact >= 40 else "moderate"

        narrative = f"{player} ({position}) is listed as {status}"
        if detail:
            narrative += f" with a {detail} injury"
        narrative += f", representing a {severity} impact for {team}."

        if replacement:
            depth_desc = "primary backup" if depth == 2 else f"depth-{depth} option"
            narrative += f" {replacement} steps in as the {depth_desc}."

            if position == 'QB':
                narrative += " Expect a more conservative offensive approach with increased emphasis on the ground game."
            elif position == 'RB':
                narrative += f" {replacement} should see a significant workload increase."
            elif position == 'WR':
                narrative += f" Targets should redistribute to remaining receivers and tight ends."
            elif position == 'TE':
                narrative += " Middle-field targets likely shift to slot receivers and backs."
        else:
            narrative += " Replacement player unknown - monitor updates."

        return narrative

    def _generate_betting_recommendations(self, team: str, player: str, position: str,
                                          status: str, replacement: str,
                                          prop_implications: List[Dict]) -> List[str]:
        """Generate specific betting recommendations."""
        recommendations = []

        # Team-level recommendations
        if position == 'QB' and status.upper() in ['OUT', 'DOUBTFUL', 'IR']:
            recommendations.append(f"UNDER team total for {team}")
            recommendations.append(f"Fade {team} in spread bets")

        # Position-specific
        if position == 'RB' and replacement:
            recommendations.append(f"Target {replacement} rushing props OVER")

        if position == 'WR' and status.upper() in ['OUT', 'DOUBTFUL']:
            recommendations.append(f"Look for value on secondary {team} receivers")

        # Add recommendations from prop implications
        for impl in prop_implications:
            if impl.get('confidence', 0) >= 0.6:
                rec = impl.get('prop_recommendation', '')
                if rec and rec not in recommendations:
                    recommendations.append(rec)

        return recommendations

    def analyze_team_injuries(self, team: str, injuries: List[Dict]) -> TeamInjuryReport:
        """Analyze all injuries for a team and generate comprehensive report.

        Args:
            team: Team abbreviation
            injuries: List of injury dicts with player_name, position, status

        Returns:
            TeamInjuryReport with full analysis
        """
        key_impacts = []
        all_redistributions = defaultdict(list)
        total_impact = 0.0

        # Analyze each injury
        for injury in injuries:
            status = injury.get('status', injury.get('injury_status', ''))

            # Only analyze significant injuries
            if status.upper() not in ['OUT', 'DOUBTFUL', 'QUESTIONABLE', 'IR', 'PUP', 'INJURED RESERVE']:
                continue

            impact = self.analyze_injury(
                team=team,
                player_name=injury.get('player_name', injury.get('name', '')),
                position=injury.get('position', ''),
                status=status,
                injury_detail=injury.get('injury_type', injury.get('body_part', ''))
            )

            # Track high-impact injuries
            if impact.team_impact_score >= 30:
                key_impacts.append(impact)
                total_impact += impact.team_impact_score

            # Aggregate redistributions by position
            for impl in impact.prop_implications:
                pos = impl.get('position', 'UNKNOWN')
                all_redistributions[pos].append(impl)

        # Cap total impact at 100
        total_impact = min(100, total_impact)

        # Generate summary
        summary = self._generate_team_summary(team, key_impacts, total_impact)

        # Generate betting angle
        betting_angle = self._generate_team_betting_angle(team, key_impacts, total_impact)

        return TeamInjuryReport(
            team=team,
            total_impact_score=round(total_impact, 1),
            key_injuries=key_impacts,
            prop_redistributions=dict(all_redistributions),
            summary=summary,
            betting_angle=betting_angle
        )

    def _generate_team_summary(self, team: str, impacts: List[InjuryImpact],
                               total_impact: float) -> str:
        """Generate team injury summary."""
        if not impacts:
            return f"{team} has no significant injuries to report."

        if total_impact >= 70:
            severity = "severely impacted"
        elif total_impact >= 40:
            severity = "significantly affected"
        else:
            severity = "moderately affected"

        summary = f"{team} is {severity} by injuries "
        summary += f"(total impact score: {total_impact:.0f}/100). "

        # List key absences
        out_players = [i for i in impacts if i.status.upper() in ['OUT', 'IR', 'PUP']]
        if out_players:
            names = [f"{i.injured_player} ({i.position})" for i in out_players[:3]]
            summary += f"Key absences: {', '.join(names)}."

        return summary

    def _generate_team_betting_angle(self, team: str, impacts: List[InjuryImpact],
                                     total_impact: float) -> str:
        """Generate overall betting angle for team."""
        if total_impact >= 70:
            return f"STRONG FADE {team}. Multiple key injuries create significant value on opponent spreads and totals."
        elif total_impact >= 40:
            positions = [i.position for i in impacts]
            if 'QB' in positions:
                return f"FADE {team} passing game. Target opponent D/ST and unders on {team} passing props."
            elif 'RB' in positions:
                return f"Target backup RB props for {team}. Team may shift to pass-heavy approach."
            else:
                return f"Moderate concerns for {team}. Look for value on replacement player props."
        else:
            return f"Minor injury concerns for {team}. Standard betting approach applies."

    def analyze_game_injuries(self, home_team: str, away_team: str,
                              home_injuries: List[Dict],
                              away_injuries: List[Dict]) -> Dict:
        """Analyze injuries for both teams in a game.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            home_injuries: List of home team injuries
            away_injuries: List of away team injuries

        Returns:
            Dict with both team reports and overall game implications
        """
        home_report = self.analyze_team_injuries(home_team, home_injuries)
        away_report = self.analyze_team_injuries(away_team, away_injuries)

        # Determine which team is more affected
        impact_diff = home_report.total_impact_score - away_report.total_impact_score

        if abs(impact_diff) >= 30:
            if impact_diff > 0:
                edge = f"{away_team} has significant injury edge over {home_team}"
                lean = away_team
            else:
                edge = f"{home_team} has significant injury edge over {away_team}"
                lean = home_team
        elif abs(impact_diff) >= 15:
            if impact_diff > 0:
                edge = f"{away_team} has moderate injury advantage"
                lean = away_team
            else:
                edge = f"{home_team} has moderate injury advantage"
                lean = home_team
        else:
            edge = "Injury situations relatively balanced"
            lean = None

        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_report': home_report,
            'away_report': away_report,
            'injury_edge': edge,
            'lean': lean,
            'impact_differential': round(impact_diff, 1),
            'all_prop_plays': self._consolidate_prop_plays(home_report, away_report)
        }

    def _consolidate_prop_plays(self, home_report: TeamInjuryReport,
                                away_report: TeamInjuryReport) -> List[Dict]:
        """Consolidate all high-confidence prop plays from both teams."""
        all_plays = []

        for report in [home_report, away_report]:
            for impact in report.key_injuries:
                for impl in impact.prop_implications:
                    if impl.get('confidence', 0) >= 0.6:
                        all_plays.append({
                            'team': report.team,
                            'player': impl.get('player'),
                            'position': impl.get('position'),
                            'recommendation': impl.get('prop_recommendation'),
                            'confidence': impl.get('confidence'),
                            'due_to': impact.injured_player
                        })

        # Sort by confidence
        all_plays.sort(key=lambda x: x.get('confidence', 0), reverse=True)

        return all_plays


# Singleton instance
injury_analyzer = InjuryImpactAnalyzer()


def get_injury_impact_for_game(game_id: str, home_injuries: List[Dict],
                               away_injuries: List[Dict]) -> Dict:
    """Convenience function to get injury impact analysis for a game.

    Args:
        game_id: Game ID (e.g., "2025_12_KC_BUF")
        home_injuries: Home team injury list
        away_injuries: Away team injury list

    Returns:
        Full game injury impact analysis
    """
    # Parse teams from game_id
    parts = game_id.split('_')
    if len(parts) >= 4:
        away_team = parts[2]
        home_team = parts[3]
    else:
        # Fallback
        away_team = "AWAY"
        home_team = "HOME"

    return injury_analyzer.analyze_game_injuries(
        home_team=home_team,
        away_team=away_team,
        home_injuries=home_injuries,
        away_injuries=away_injuries
    )
