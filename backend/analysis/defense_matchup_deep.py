"""Defense Matchup Deep Analysis System.

Provides positional defense breakdowns and automatically adjusts projections
based on opponent defensive matchups.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np


@dataclass
class PositionalDefenseStats:
    """Defense stats against a specific position."""
    team: str
    position: str  # 'WR1', 'WR2', 'Slot', 'RB_rush', 'RB_recv', 'TE'

    # Performance metrics
    yards_per_game_allowed: float
    yards_per_target_allowed: float = 0.0
    yards_per_carry_allowed: float = 0.0

    # Rate stats
    completion_pct_allowed: float = 0.0
    target_rate_vs_avg: float = 1.0  # 1.0 = average, <1.0 = stingy, >1.0 = generous

    # Success rates
    big_play_rate: float = 0.0  # % of plays 20+ yards
    td_rate_allowed: float = 0.0

    # Ranking (1 = best defense vs position)
    league_rank: int = 16

    # Sample size
    games_analyzed: int = 0
    confidence: float = 0.5

    def get_matchup_factor(self) -> float:
        """Calculate matchup adjustment factor.

        Returns:
            Multiplier for projections (0.8 = tough, 1.0 = average, 1.2 = soft)
        """
        # Base factor on yards allowed vs league average
        if self.yards_per_game_allowed == 0:
            return 1.0

        # League average baselines (approximate)
        league_avg = {
            'WR1': 65.0,
            'WR2': 45.0,
            'Slot': 40.0,
            'RB_rush': 55.0,
            'RB_recv': 25.0,
            'TE': 45.0
        }

        avg = league_avg.get(self.position, 50.0)

        # Calculate factor (inverse - more yards allowed = easier matchup)
        factor = self.yards_per_game_allowed / avg

        # Clamp to reasonable range
        return max(0.7, min(1.3, factor))


@dataclass
class MatchupAnalysis:
    """Complete matchup analysis for a player."""
    player: str
    player_id: str
    position: str
    team: str

    opponent: str
    opponent_rank: int  # Overall defensive rank

    # Positional matchup
    positional_matchup: Optional[PositionalDefenseStats] = None

    # Projection adjustment
    base_projection: float = 0.0
    matchup_factor: float = 1.0
    adjusted_projection: float = 0.0

    # Matchup quality
    matchup_quality: str = "Average"  # Smash, Great, Good, Average, Tough, Avoid
    confidence: float = 0.5

    # Reasoning
    reasoning: str = ""


class DefenseMatchupAnalyzer:
    """Deep defense matchup analysis with automatic projection adjustments."""

    def __init__(self, season: int = 2025):
        """Initialize analyzer.

        Args:
            season: NFL season year
        """
        self.season = season
        self.inputs_dir = Path('inputs')
        self.data_dir = Path('data/defense_matchups')
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Positional defense stats (calculated from historical data)
        self.positional_stats = self._calculate_positional_stats()

    def _calculate_positional_stats(self) -> Dict[str, Dict[str, PositionalDefenseStats]]:
        """Calculate positional defense stats for all teams.

        Returns:
            Dictionary mapping team -> position -> stats
        """
        stats_by_team = {}

        try:
            # Load player stats to analyze defense
            stats_file = self.inputs_dir / f'{self.season}_player_stats.parquet'
            if not stats_file.exists():
                return self._get_default_positional_stats()

            player_stats = pd.read_parquet(stats_file)

            # Get all teams
            teams = player_stats['opponent_team'].unique() if 'opponent_team' in player_stats.columns else []

            for team in teams:
                if pd.isna(team):
                    continue

                stats_by_team[team] = self._calculate_team_positional_stats(
                    team, player_stats
                )

        except Exception as e:
            print(f"Error calculating positional stats: {e}")
            return self._get_default_positional_stats()

        return stats_by_team if stats_by_team else self._get_default_positional_stats()

    def _calculate_team_positional_stats(
        self,
        team: str,
        player_stats: pd.DataFrame
    ) -> Dict[str, PositionalDefenseStats]:
        """Calculate positional stats for a specific team's defense.

        Args:
            team: Team abbreviation
            player_stats: DataFrame of player stats

        Returns:
            Dictionary mapping position to stats
        """
        position_stats = {}

        # Filter for games against this team
        vs_team = player_stats[player_stats['opponent_team'] == team]

        if len(vs_team) == 0:
            return self._get_default_team_stats(team)

        # WR positions
        for wr_pos in ['WR1', 'WR2', 'Slot']:
            position_stats[wr_pos] = self._calculate_wr_position_stats(
                vs_team, wr_pos, team
            )

        # RB positions
        position_stats['RB_rush'] = self._calculate_rb_rush_stats(vs_team, team)
        position_stats['RB_recv'] = self._calculate_rb_recv_stats(vs_team, team)

        # TE position
        position_stats['TE'] = self._calculate_te_stats(vs_team, team)

        return position_stats

    def _calculate_wr_position_stats(
        self,
        vs_team: pd.DataFrame,
        wr_position: str,
        team: str
    ) -> PositionalDefenseStats:
        """Calculate stats for WRs against this defense.

        Args:
            vs_team: Stats against this team
            wr_position: 'WR1', 'WR2', or 'Slot'
            team: Team abbreviation

        Returns:
            PositionalDefenseStats
        """
        # Filter for WRs
        wrs = vs_team[vs_team['position'] == 'WR']

        if len(wrs) == 0:
            return self._get_default_position_stats(team, wr_position)

        # Classify WRs by role (simplified - would need actual depth chart data)
        # For now, use targets as proxy: WR1 = 7+ targets, WR2 = 4-7, Slot = varies

        if wr_position == 'WR1':
            position_wrs = wrs[wrs['targets'] >= 7]
        elif wr_position == 'WR2':
            position_wrs = wrs[(wrs['targets'] >= 4) & (wrs['targets'] < 7)]
        else:  # Slot
            position_wrs = wrs[wrs['targets'] < 7]  # Simplified

        if len(position_wrs) == 0:
            return self._get_default_position_stats(team, wr_position)

        # Calculate stats
        yards_allowed = position_wrs['receiving_yards'].mean()
        targets = position_wrs['targets'].mean()
        receptions = position_wrs['receptions'].mean()

        yards_per_target = yards_allowed / targets if targets > 0 else 0

        return PositionalDefenseStats(
            team=team,
            position=wr_position,
            yards_per_game_allowed=yards_allowed,
            yards_per_target_allowed=yards_per_target,
            completion_pct_allowed=receptions / targets if targets > 0 else 0,
            games_analyzed=len(position_wrs),
            confidence=min(1.0, len(position_wrs) / 10.0)  # Higher confidence with more games
        )

    def _calculate_rb_rush_stats(
        self,
        vs_team: pd.DataFrame,
        team: str
    ) -> PositionalDefenseStats:
        """Calculate rushing stats for RBs against this defense."""
        rbs = vs_team[vs_team['position'] == 'RB']

        if len(rbs) == 0:
            return self._get_default_position_stats(team, 'RB_rush')

        # Filter for primary rushers (5+ carries)
        rushers = rbs[rbs['carries'] >= 5]

        if len(rushers) == 0:
            rushers = rbs

        rush_yards = rushers['rushing_yards'].mean()
        carries = rushers['carries'].mean()
        yards_per_carry = rush_yards / carries if carries > 0 else 0

        return PositionalDefenseStats(
            team=team,
            position='RB_rush',
            yards_per_game_allowed=rush_yards,
            yards_per_carry_allowed=yards_per_carry,
            games_analyzed=len(rushers),
            confidence=min(1.0, len(rushers) / 10.0)
        )

    def _calculate_rb_recv_stats(
        self,
        vs_team: pd.DataFrame,
        team: str
    ) -> PositionalDefenseStats:
        """Calculate receiving stats for RBs against this defense."""
        rbs = vs_team[vs_team['position'] == 'RB']

        if len(rbs) == 0:
            return self._get_default_position_stats(team, 'RB_recv')

        # Filter for receiving RBs (2+ targets)
        receivers = rbs[rbs['targets'] >= 2]

        if len(receivers) == 0:
            receivers = rbs

        rec_yards = receivers['receiving_yards'].mean()
        targets = receivers['targets'].mean()

        return PositionalDefenseStats(
            team=team,
            position='RB_recv',
            yards_per_game_allowed=rec_yards,
            yards_per_target_allowed=rec_yards / targets if targets > 0 else 0,
            games_analyzed=len(receivers),
            confidence=min(1.0, len(receivers) / 10.0)
        )

    def _calculate_te_stats(
        self,
        vs_team: pd.DataFrame,
        team: str
    ) -> PositionalDefenseStats:
        """Calculate stats for TEs against this defense."""
        tes = vs_team[vs_team['position'] == 'TE']

        if len(tes) == 0:
            return self._get_default_position_stats(team, 'TE')

        yards = tes['receiving_yards'].mean()
        targets = tes['targets'].mean()

        return PositionalDefenseStats(
            team=team,
            position='TE',
            yards_per_game_allowed=yards,
            yards_per_target_allowed=yards / targets if targets > 0 else 0,
            games_analyzed=len(tes),
            confidence=min(1.0, len(tes) / 10.0)
        )

    def _get_default_positional_stats(self) -> Dict[str, Dict[str, PositionalDefenseStats]]:
        """Get default positional stats (league average) for all teams."""
        teams = [
            'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
            'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
            'LAC', 'LAR', 'LV', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
            'NYJ', 'PHI', 'PIT', 'SF', 'SEA', 'TB', 'TEN', 'WAS'
        ]

        return {team: self._get_default_team_stats(team) for team in teams}

    def _get_default_team_stats(self, team: str) -> Dict[str, PositionalDefenseStats]:
        """Get default stats for a team (league average)."""
        return {
            'WR1': self._get_default_position_stats(team, 'WR1'),
            'WR2': self._get_default_position_stats(team, 'WR2'),
            'Slot': self._get_default_position_stats(team, 'Slot'),
            'RB_rush': self._get_default_position_stats(team, 'RB_rush'),
            'RB_recv': self._get_default_position_stats(team, 'RB_recv'),
            'TE': self._get_default_position_stats(team, 'TE')
        }

    def _get_default_position_stats(self, team: str, position: str) -> PositionalDefenseStats:
        """Get default stats for a position (league average)."""
        defaults = {
            'WR1': 65.0,
            'WR2': 45.0,
            'Slot': 40.0,
            'RB_rush': 55.0,
            'RB_recv': 25.0,
            'TE': 45.0
        }

        return PositionalDefenseStats(
            team=team,
            position=position,
            yards_per_game_allowed=defaults.get(position, 50.0),
            yards_per_target_allowed=8.0,
            yards_per_carry_allowed=4.5,
            games_analyzed=0,
            confidence=0.3  # Low confidence for defaults
        )

    def analyze_matchup(
        self,
        player: str,
        player_id: str,
        position: str,
        team: str,
        opponent: str,
        base_projection: float
    ) -> MatchupAnalysis:
        """Analyze defensive matchup for a player.

        Args:
            player: Player name
            player_id: Player ID
            position: Position (WR, RB, TE)
            team: Player's team
            opponent: Opponent team
            base_projection: Base stat projection

        Returns:
            MatchupAnalysis with adjustment
        """
        # Determine positional matchup
        positional_role = self._determine_positional_role(player_id, position, team)

        # Get opponent's positional defense stats
        opponent_stats = self.positional_stats.get(opponent, {})
        positional_matchup = opponent_stats.get(positional_role)

        if not positional_matchup:
            positional_matchup = self._get_default_position_stats(opponent, positional_role)

        # Calculate matchup factor
        matchup_factor = positional_matchup.get_matchup_factor()

        # Adjusted projection
        adjusted_projection = base_projection * matchup_factor

        # Determine matchup quality
        matchup_quality, reasoning = self._determine_matchup_quality(
            matchup_factor, positional_matchup
        )

        return MatchupAnalysis(
            player=player,
            player_id=player_id,
            position=position,
            team=team,
            opponent=opponent,
            opponent_rank=16,  # Would calculate from overall defensive stats
            positional_matchup=positional_matchup,
            base_projection=base_projection,
            matchup_factor=matchup_factor,
            adjusted_projection=adjusted_projection,
            matchup_quality=matchup_quality,
            confidence=positional_matchup.confidence,
            reasoning=reasoning
        )

    def _determine_positional_role(
        self,
        player_id: str,
        position: str,
        team: str
    ) -> str:
        """Determine player's specific positional role.

        Args:
            player_id: Player ID
            position: Position (WR, RB, TE)
            team: Team abbreviation

        Returns:
            Specific role (e.g., 'WR1', 'RB_rush', 'Slot')
        """
        # Would need depth chart data or usage patterns
        # For now, use simplified logic

        if position == 'WR':
            # Default to WR1 for simplicity
            # In reality, would check targets, snap counts, alignment data
            return 'WR1'
        elif position == 'RB':
            # Default to RB_rush
            return 'RB_rush'
        elif position == 'TE':
            return 'TE'

        return position

    def _determine_matchup_quality(
        self,
        matchup_factor: float,
        positional_matchup: PositionalDefenseStats
    ) -> Tuple[str, str]:
        """Determine matchup quality rating.

        Args:
            matchup_factor: Adjustment factor
            positional_matchup: Positional defense stats

        Returns:
            (quality_rating, reasoning)
        """
        if matchup_factor >= 1.20:
            return "Smash", f"Elite matchup vs {positional_matchup.team} (allows {positional_matchup.yards_per_game_allowed:.1f} ypg to {positional_matchup.position})"
        elif matchup_factor >= 1.10:
            return "Great", f"Favorable matchup vs {positional_matchup.team} (+{(matchup_factor-1)*100:.0f}% boost)"
        elif matchup_factor >= 1.05:
            return "Good", f"Slight edge vs {positional_matchup.team}"
        elif matchup_factor >= 0.95:
            return "Average", f"Neutral matchup vs {positional_matchup.team}"
        elif matchup_factor >= 0.85:
            return "Tough", f"Challenging matchup vs {positional_matchup.team} ({(1-matchup_factor)*100:.0f}% reduction)"
        else:
            return "Avoid", f"Brutal matchup vs {positional_matchup.team} (allows only {positional_matchup.yards_per_game_allowed:.1f} ypg)"


# Singleton instance
defense_matchup_analyzer = DefenseMatchupAnalyzer()


if __name__ == "__main__":
    # Test analyzer
    analyzer = DefenseMatchupAnalyzer(season=2025)

    # Test matchup analysis
    matchup = analyzer.analyze_matchup(
        player="Tyreek Hill",
        player_id="hill_tyreek",
        position="WR",
        team="MIA",
        opponent="NYJ",
        base_projection=85.5
    )

    print(f"Matchup Analysis: {matchup.player}")
    print(f"Opponent: {matchup.opponent}")
    print(f"Base Projection: {matchup.base_projection:.1f} yards")
    print(f"Matchup Factor: {matchup.matchup_factor:.2f}x")
    print(f"Adjusted Projection: {matchup.adjusted_projection:.1f} yards")
    print(f"Matchup Quality: {matchup.matchup_quality}")
    print(f"Confidence: {matchup.confidence:.2f}")
    print(f"Reasoning: {matchup.reasoning}")
