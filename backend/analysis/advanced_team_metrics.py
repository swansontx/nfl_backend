"""Advanced Team Metrics Calculator from Play-by-Play Data.

Calculates professional-grade team metrics for matchup analysis:
- Red zone efficiency
- Third down conversion rate
- Success rate (situational efficiency)
- EPA per play
- Yards per carry/attempt
- Turnover rates
- Time of possession
- QB efficiency metrics

Usage:
    from backend.analysis.advanced_team_metrics import AdvancedTeamMetricsCalculator

    calc = AdvancedTeamMetricsCalculator(season=2025)
    metrics = calc.calculate_team_metrics('KC')

    print(f"Red Zone TD%: {metrics['red_zone_td_pct']:.1%}")
    print(f"3rd Down%: {metrics['third_down_pct']:.1%}")
    print(f"Success Rate: {metrics['success_rate']:.1%}")
"""

from dataclasses import dataclass
from typing import Dict, Optional
from pathlib import Path
import pandas as pd
import numpy as np


@dataclass
class AdvancedTeamMetrics:
    """Advanced team performance metrics."""
    team: str
    season: int
    games_played: int = 0

    # Efficiency Metrics
    success_rate_offense: float = 0.0    # % of plays that are "successful"
    success_rate_defense: float = 0.0    # % opponent successful plays
    epa_per_play_offense: float = 0.0    # Expected points added per play
    epa_per_play_defense: float = 0.0    # EPA allowed per play

    # Red Zone Performance
    red_zone_attempts: int = 0           # Trips inside 20
    red_zone_tds: int = 0                # TDs from red zone
    red_zone_td_pct: float = 0.0         # TD% in red zone
    red_zone_fgs: int = 0                # FGs from red zone
    red_zone_score_pct: float = 0.0      # Any score in red zone

    # Third Down Efficiency
    third_down_attempts: int = 0
    third_down_conversions: int = 0
    third_down_pct: float = 0.0          # Conversion rate

    # Passing Efficiency
    pass_attempts: int = 0
    completions: int = 0
    completion_pct: float = 0.0
    yards_per_attempt: float = 0.0
    air_yards_per_attempt: float = 0.0
    yards_after_catch_per_comp: float = 0.0

    # Rushing Efficiency
    rush_attempts: int = 0
    yards_per_carry: float = 0.0
    rush_success_rate: float = 0.0       # % successful rushes
    explosive_rush_rate: float = 0.0     # % rushes 10+ yards

    # Turnover Metrics
    turnovers: int = 0                   # Total given
    turnovers_forced: int = 0            # Total taken
    turnover_rate: float = 0.0           # Per 100 plays
    takeaway_rate: float = 0.0           # Per 100 defensive plays
    turnover_margin: int = 0             # +/- differential

    # Explosive Plays
    explosive_play_rate: float = 0.0     # % plays 20+ yards (pass) or 10+ (rush)
    explosive_plays_allowed_rate: float = 0.0

    # Time Control
    avg_time_of_possession: float = 0.0  # Minutes per game
    plays_per_game: float = 0.0          # Pace
    seconds_per_play: float = 0.0        # Pace metric

    # Situational
    scoring_drives_pct: float = 0.0      # % of drives that score
    three_and_out_pct: float = 0.0       # % drives end in 3-and-out


class AdvancedTeamMetricsCalculator:
    """Calculates advanced team metrics from play-by-play data."""

    def __init__(self, season: int = 2025, pbp_file: Optional[Path] = None):
        """Initialize calculator.

        Args:
            season: Season year
            pbp_file: Optional path to pbp parquet file
        """
        self.season = season
        self.pbp_file = pbp_file or Path(f'inputs/play_by_play_{season}.parquet')

        # Load pbp data
        self.pbp = self._load_pbp_data()

    def _load_pbp_data(self) -> pd.DataFrame:
        """Load play-by-play data."""
        if self.pbp_file.exists():
            return pd.read_parquet(self.pbp_file)
        return pd.DataFrame()

    def calculate_team_metrics(
        self,
        team: str,
        weeks: Optional[list] = None
    ) -> Dict:
        """Calculate all advanced metrics for a team.

        Args:
            team: Team abbreviation
            weeks: Optional list of weeks to include (for recent form)

        Returns:
            Dictionary of metrics
        """
        if self.pbp.empty:
            return {}

        # Filter to team's plays
        if weeks:
            team_plays = self.pbp[
                (self.pbp['posteam'] == team) &
                (self.pbp['week'].isin(weeks))
            ]
            opponent_plays = self.pbp[
                (self.pbp['defteam'] == team) &
                (self.pbp['week'].isin(weeks))
            ]
        else:
            team_plays = self.pbp[self.pbp['posteam'] == team]
            opponent_plays = self.pbp[self.pbp['defteam'] == team]

        # Calculate all metrics
        metrics = AdvancedTeamMetrics(team=team, season=self.season)

        # Games played
        metrics.games_played = team_plays['game_id'].nunique()

        if len(team_plays) == 0:
            return metrics.__dict__

        # Success Rate
        success_plays = team_plays[team_plays['success'] == 1]
        metrics.success_rate_offense = len(success_plays) / len(team_plays) if len(team_plays) > 0 else 0.0

        opp_success = opponent_plays[opponent_plays['success'] == 1]
        metrics.success_rate_defense = len(opp_success) / len(opponent_plays) if len(opponent_plays) > 0 else 0.0

        # EPA
        metrics.epa_per_play_offense = team_plays['epa'].mean()
        metrics.epa_per_play_defense = opponent_plays['epa'].mean()

        # Red Zone
        red_zone = team_plays[team_plays['yardline_100'] <= 20]
        metrics.red_zone_attempts = len(red_zone)
        metrics.red_zone_tds = len(red_zone[red_zone['touchdown'] == 1])
        metrics.red_zone_td_pct = metrics.red_zone_tds / metrics.red_zone_attempts if metrics.red_zone_attempts > 0 else 0.0

        rz_fgs = red_zone[red_zone['field_goal_result'] == 'made']
        metrics.red_zone_fgs = len(rz_fgs)
        metrics.red_zone_score_pct = (metrics.red_zone_tds + metrics.red_zone_fgs) / metrics.red_zone_attempts if metrics.red_zone_attempts > 0 else 0.0

        # Third Down
        third_downs = team_plays[team_plays['down'] == 3]
        metrics.third_down_attempts = len(third_downs)
        metrics.third_down_conversions = len(third_downs[third_downs['third_down_converted'] == 1])
        metrics.third_down_pct = metrics.third_down_conversions / metrics.third_down_attempts if metrics.third_down_attempts > 0 else 0.0

        # Passing
        pass_plays = team_plays[team_plays['pass'] == 1]
        metrics.pass_attempts = len(pass_plays)
        metrics.completions = len(pass_plays[pass_plays['complete_pass'] == 1])
        metrics.completion_pct = metrics.completions / metrics.pass_attempts if metrics.pass_attempts > 0 else 0.0
        metrics.yards_per_attempt = pass_plays['yards_gained'].mean()
        metrics.air_yards_per_attempt = pass_plays['air_yards'].mean()

        completed_passes = pass_plays[pass_plays['complete_pass'] == 1]
        metrics.yards_after_catch_per_comp = completed_passes['yards_after_catch'].mean()

        # Rushing
        rush_plays = team_plays[team_plays['rush'] == 1]
        metrics.rush_attempts = len(rush_plays)
        metrics.yards_per_carry = rush_plays['yards_gained'].mean()

        rush_success = rush_plays[rush_plays['success'] == 1]
        metrics.rush_success_rate = len(rush_success) / len(rush_plays) if len(rush_plays) > 0 else 0.0

        explosive_rushes = rush_plays[rush_plays['yards_gained'] >= 10]
        metrics.explosive_rush_rate = len(explosive_rushes) / len(rush_plays) if len(rush_plays) > 0 else 0.0

        # Turnovers
        tos = team_plays[(team_plays['interception'] == 1) | (team_plays['fumble_lost'] == 1)]
        metrics.turnovers = len(tos)
        metrics.turnover_rate = (metrics.turnovers / len(team_plays)) * 100 if len(team_plays) > 0 else 0.0

        takeaways = opponent_plays[(opponent_plays['interception'] == 1) | (opponent_plays['fumble_lost'] == 1)]
        metrics.turnovers_forced = len(takeaways)
        metrics.takeaway_rate = (metrics.turnovers_forced / len(opponent_plays)) * 100 if len(opponent_plays) > 0 else 0.0
        metrics.turnover_margin = metrics.turnovers_forced - metrics.turnovers

        # Explosive Plays (20+ pass, 10+ rush)
        explosive_pass = pass_plays[pass_plays['yards_gained'] >= 20]
        explosive_rush_20 = rush_plays[rush_plays['yards_gained'] >= 10]
        total_explosive = len(explosive_pass) + len(explosive_rush_20)
        metrics.explosive_play_rate = total_explosive / len(team_plays) if len(team_plays) > 0 else 0.0

        opp_pass = opponent_plays[opponent_plays['pass'] == 1]
        opp_rush = opponent_plays[opponent_plays['rush'] == 1]
        opp_explosive = len(opp_pass[opp_pass['yards_gained'] >= 20]) + len(opp_rush[opp_rush['yards_gained'] >= 10])
        metrics.explosive_plays_allowed_rate = opp_explosive / len(opponent_plays) if len(opponent_plays) > 0 else 0.0

        # Pace
        metrics.plays_per_game = len(team_plays) / metrics.games_played if metrics.games_played > 0 else 0.0

        # Note: Time of possession would require drive-level analysis
        # Placeholder for now
        metrics.avg_time_of_possession = 30.0  # Default ~30 min per game

        return metrics.__dict__

    def compare_teams(
        self,
        team_a: str,
        team_b: str,
        weeks: Optional[list] = None
    ) -> Dict:
        """Compare two teams across all metrics.

        Args:
            team_a: First team
            team_b: Second team
            weeks: Optional weeks filter

        Returns:
            Comparison dictionary with advantages
        """
        metrics_a = self.calculate_team_metrics(team_a, weeks)
        metrics_b = self.calculate_team_metrics(team_b, weeks)

        comparison = {
            'team_a': team_a,
            'team_b': team_b,
            'advantages_a': [],
            'advantages_b': [],
            'metrics_a': metrics_a,
            'metrics_b': metrics_b
        }

        # Compare key metrics
        if metrics_a.get('success_rate_offense', 0) > metrics_b.get('success_rate_offense', 0) * 1.05:
            comparison['advantages_a'].append(
                f"Success rate: {metrics_a['success_rate_offense']:.1%} vs {metrics_b['success_rate_offense']:.1%}"
            )
        elif metrics_b.get('success_rate_offense', 0) > metrics_a.get('success_rate_offense', 0) * 1.05:
            comparison['advantages_b'].append(
                f"Success rate: {metrics_b['success_rate_offense']:.1%} vs {metrics_a['success_rate_offense']:.1%}"
            )

        if metrics_a.get('red_zone_td_pct', 0) > metrics_b.get('red_zone_td_pct', 0) * 1.1:
            comparison['advantages_a'].append(
                f"Red zone TD%: {metrics_a['red_zone_td_pct']:.1%} vs {metrics_b['red_zone_td_pct']:.1%}"
            )
        elif metrics_b.get('red_zone_td_pct', 0) > metrics_a.get('red_zone_td_pct', 0) * 1.1:
            comparison['advantages_b'].append(
                f"Red zone TD%: {metrics_b['red_zone_td_pct']:.1%} vs {metrics_a['red_zone_td_pct']:.1%}"
            )

        if metrics_a.get('third_down_pct', 0) > metrics_b.get('third_down_pct', 0) * 1.1:
            comparison['advantages_a'].append(
                f"3rd down%: {metrics_a['third_down_pct']:.1%} vs {metrics_b['third_down_pct']:.1%}"
            )
        elif metrics_b.get('third_down_pct', 0) > metrics_a.get('third_down_pct', 0) * 1.1:
            comparison['advantages_b'].append(
                f"3rd down%: {metrics_b['third_down_pct']:.1%} vs {metrics_a['third_down_pct']:.1%}"
            )

        if metrics_a.get('turnover_margin', 0) > metrics_b.get('turnover_margin', 0) + 3:
            comparison['advantages_a'].append(
                f"Turnover margin: +{metrics_a['turnover_margin']} vs {metrics_b['turnover_margin']:+d}"
            )
        elif metrics_b.get('turnover_margin', 0) > metrics_a.get('turnover_margin', 0) + 3:
            comparison['advantages_b'].append(
                f"Turnover margin: +{metrics_b['turnover_margin']} vs {metrics_a['turnover_margin']:+d}"
            )

        return comparison


if __name__ == "__main__":
    # Test the calculator
    calc = AdvancedTeamMetricsCalculator(season=2025)

    # Calculate metrics for KC
    print("="*70)
    print("KANSAS CITY CHIEFS - ADVANCED METRICS (2025)")
    print("="*70)

    kc_metrics = calc.calculate_team_metrics('KC')

    print(f"\n📊 Efficiency Metrics:")
    print(f"  Success Rate: {kc_metrics.get('success_rate_offense', 0):.1%}")
    print(f"  EPA/play: {kc_metrics.get('epa_per_play_offense', 0):+.3f}")

    print(f"\n🎯 Red Zone Performance:")
    print(f"  Attempts: {kc_metrics.get('red_zone_attempts', 0)}")
    print(f"  TDs: {kc_metrics.get('red_zone_tds', 0)} ({kc_metrics.get('red_zone_td_pct', 0):.1%})")
    print(f"  Any Score: {kc_metrics.get('red_zone_score_pct', 0):.1%}")

    print(f"\n⬇️ Third Down:")
    print(f"  Conversions: {kc_metrics.get('third_down_conversions', 0)}/{kc_metrics.get('third_down_attempts', 0)} ({kc_metrics.get('third_down_pct', 0):.1%})")

    print(f"\n🎯 Passing:")
    print(f"  Completion%: {kc_metrics.get('completion_pct', 0):.1%}")
    print(f"  Yards/Attempt: {kc_metrics.get('yards_per_attempt', 0):.1f}")
    print(f"  Air Yards/Att: {kc_metrics.get('air_yards_per_attempt', 0):.1f}")
    print(f"  YAC/Comp: {kc_metrics.get('yards_after_catch_per_comp', 0):.1f}")

    print(f"\n🏃 Rushing:")
    print(f"  Yards/Carry: {kc_metrics.get('yards_per_carry', 0):.1f}")
    print(f"  Success Rate: {kc_metrics.get('rush_success_rate', 0):.1%}")
    print(f"  Explosive%: {kc_metrics.get('explosive_rush_rate', 0):.1%}")

    print(f"\n🔄 Turnovers:")
    print(f"  Turnovers: {kc_metrics.get('turnovers', 0)} ({kc_metrics.get('turnover_rate', 0):.1f} per 100)")
    print(f"  Takeaways: {kc_metrics.get('turnovers_forced', 0)}")
    print(f"  Margin: {kc_metrics.get('turnover_margin', 0):+d}")

    print(f"\n💥 Explosiveness:")
    print(f"  Explosive Plays: {kc_metrics.get('explosive_play_rate', 0):.1%}")
    print(f"  Allowed: {kc_metrics.get('explosive_plays_allowed_rate', 0):.1%}")

    print(f"\n⏱️ Pace:")
    print(f"  Plays/Game: {kc_metrics.get('plays_per_game', 0):.1f}")

    # Compare teams
    print(f"\n\n{'='*70}")
    print("TEAM COMPARISON: KC vs BUF")
    print("="*70)

    comparison = calc.compare_teams('KC', 'BUF')

    if comparison['advantages_a']:
        print(f"\n✓ {comparison['team_a']} Advantages:")
        for adv in comparison['advantages_a']:
            print(f"  - {adv}")

    if comparison['advantages_b']:
        print(f"\n✓ {comparison['team_b']} Advantages:")
        for adv in comparison['advantages_b']:
            print(f"  - {adv}")
