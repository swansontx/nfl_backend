"""Defense Bucketing - Categorize defenses and analyze player performance vs each category.

Example Use Case:
"How does Josh Allen perform against Top 10 pass defenses?"
- Bucket defenses by EPA allowed: Elite (top 10), Average (11-22), Weak (23-32)
- Find all games where Allen faced Elite defenses
- Calculate his avg yards/TDs/EPA in those games
- Use that as contextual adjustment for upcoming game vs elite defense

Bucketing Categories:
1. By EPA Allowed (overall defensive quality)
2. By CPOE Allowed (secondary quality)
3. By Pressure Rate Generated (pass rush quality)
4. By Success Rate Allowed (defensive efficiency)
5. By Rushing EPA Allowed (run defense quality)
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
import statistics


@dataclass
class DefensiveBucket:
    """A category of defenses based on specific metric."""
    bucket_name: str
    metric_name: str
    threshold_desc: str  # e.g., "Top 10", "Bottom 10"
    teams: List[str]
    avg_metric_value: float


class DefenseBucketer:
    """Categorizes defenses into buckets for performance analysis."""

    def __init__(self):
        self.bucket_definitions = {
            'elite': (1, 10),      # Rank 1-10
            'above_avg': (11, 16), # Rank 11-16
            'below_avg': (17, 22), # Rank 17-22
            'weak': (23, 32)       # Rank 23-32
        }

    def bucket_defenses_by_epa(
        self,
        defensive_stats: Dict[str, Dict]
    ) -> Dict[str, DefensiveBucket]:
        """Bucket defenses by EPA allowed (lower EPA = better defense).

        Args:
            defensive_stats: Dict mapping team -> {'epa_allowed': float, ...}

        Returns:
            Dict with buckets: 'elite', 'above_avg', 'below_avg', 'weak'
        """
        # Rank teams by EPA allowed (ascending = best defenses)
        ranked_teams = sorted(
            defensive_stats.items(),
            key=lambda x: x[1].get('passing_epa_allowed', 0)
        )

        buckets = {}

        for bucket_name, (start_rank, end_rank) in self.bucket_definitions.items():
            bucket_teams = []
            bucket_values = []

            for idx, (team, stats) in enumerate(ranked_teams, start=1):
                if start_rank <= idx <= end_rank:
                    bucket_teams.append(team)
                    bucket_values.append(stats.get('passing_epa_allowed', 0))

            if bucket_teams:
                buckets[bucket_name] = DefensiveBucket(
                    bucket_name=bucket_name,
                    metric_name='passing_epa_allowed',
                    threshold_desc=f"Rank {start_rank}-{end_rank}",
                    teams=bucket_teams,
                    avg_metric_value=statistics.mean(bucket_values)
                )

        return buckets

    def bucket_defenses_by_pressure_rate(
        self,
        defensive_stats: Dict[str, Dict]
    ) -> Dict[str, DefensiveBucket]:
        """Bucket defenses by pressure rate generated (higher = better pass rush).

        Args:
            defensive_stats: Dict mapping team -> {'pressure_rate_generated': float, ...}

        Returns:
            Dict with buckets
        """
        # Rank teams by pressure rate (descending = best pass rushes)
        ranked_teams = sorted(
            defensive_stats.items(),
            key=lambda x: x[1].get('pressure_rate_generated', 0),
            reverse=True
        )

        buckets = {}

        for bucket_name, (start_rank, end_rank) in self.bucket_definitions.items():
            bucket_teams = []
            bucket_values = []

            for idx, (team, stats) in enumerate(ranked_teams, start=1):
                if start_rank <= idx <= end_rank:
                    bucket_teams.append(team)
                    bucket_values.append(stats.get('pressure_rate_generated', 0))

            if bucket_teams:
                buckets[bucket_name] = DefensiveBucket(
                    bucket_name=bucket_name,
                    metric_name='pressure_rate_generated',
                    threshold_desc=f"Rank {start_rank}-{end_rank}",
                    teams=bucket_teams,
                    avg_metric_value=statistics.mean(bucket_values)
                )

        return buckets

    def bucket_defenses_by_cpoe_allowed(
        self,
        defensive_stats: Dict[str, Dict]
    ) -> Dict[str, DefensiveBucket]:
        """Bucket defenses by CPOE allowed (lower = better secondary).

        Args:
            defensive_stats: Dict mapping team -> {'cpoe_allowed': float, ...}

        Returns:
            Dict with buckets
        """
        # Rank teams by CPOE allowed (ascending = best secondaries)
        ranked_teams = sorted(
            defensive_stats.items(),
            key=lambda x: x[1].get('cpoe_allowed', 0)
        )

        buckets = {}

        for bucket_name, (start_rank, end_rank) in self.bucket_definitions.items():
            bucket_teams = []
            bucket_values = []

            for idx, (team, stats) in enumerate(ranked_teams, start=1):
                if start_rank <= idx <= end_rank:
                    bucket_teams.append(team)
                    bucket_values.append(stats.get('cpoe_allowed', 0))

            if bucket_teams:
                buckets[bucket_name] = DefensiveBucket(
                    bucket_name=bucket_name,
                    metric_name='cpoe_allowed',
                    threshold_desc=f"Rank {start_rank}-{end_rank}",
                    teams=bucket_teams,
                    avg_metric_value=statistics.mean(bucket_values)
                )

        return buckets

    def analyze_player_vs_bucket(
        self,
        player_games: List[Dict],
        bucket: DefensiveBucket,
        stat_name: str = 'passing_yards'
    ) -> Dict:
        """Analyze how player performs against teams in a specific bucket.

        Args:
            player_games: List of player's game dictionaries
            bucket: DefensiveBucket to analyze against
            stat_name: Stat to measure (e.g., 'passing_yards', 'rushing_yards')

        Returns:
            Dict with performance vs this bucket
        """
        # Filter to games against teams in this bucket
        bucket_games = [
            game for game in player_games
            if game.get('opponent') in bucket.teams
        ]

        if not bucket_games:
            return {
                'bucket': bucket.bucket_name,
                'games': 0,
                'avg_yards': 0,
                'avg_epa': 0,
                'sample_size': 'insufficient'
            }

        # Calculate averages
        avg_yards = statistics.mean([g.get(stat_name, 0) for g in bucket_games])

        # Determine EPA stat name based on position
        if 'passing' in stat_name:
            epa_stat = 'qb_epa'
        elif 'rushing' in stat_name:
            epa_stat = 'rushing_epa'
        else:
            epa_stat = 'receiving_epa'

        avg_epa = statistics.mean([g.get(epa_stat, 0) for g in bucket_games])

        # Success rate
        success_count = sum(g.get('success_plays', 0) for g in bucket_games)
        total_plays = sum(g.get('total_plays', 1) for g in bucket_games)
        success_rate = (success_count / total_plays * 100) if total_plays > 0 else 0

        return {
            'bucket': bucket.bucket_name,
            'bucket_desc': bucket.threshold_desc,
            'games': len(bucket_games),
            'avg_yards': round(avg_yards, 1),
            'avg_epa': round(avg_epa, 3),
            'success_rate': round(success_rate, 1),
            'sample_size': 'good' if len(bucket_games) >= 3 else 'limited',
            'teams_faced': list(set(g.get('opponent') for g in bucket_games))
        }

    def get_full_bucket_analysis(
        self,
        player_games: List[Dict],
        defensive_buckets: Dict[str, DefensiveBucket],
        stat_name: str = 'passing_yards'
    ) -> Dict[str, Dict]:
        """Analyze player performance across all defensive buckets.

        Args:
            player_games: List of player's game dictionaries
            defensive_buckets: Dict of DefensiveBucket objects
            stat_name: Stat to measure

        Returns:
            Dict mapping bucket_name -> performance analysis
        """
        analysis = {}

        for bucket_name, bucket in defensive_buckets.items():
            bucket_perf = self.analyze_player_vs_bucket(
                player_games,
                bucket,
                stat_name
            )
            analysis[bucket_name] = bucket_perf

        return analysis


# Example usage
if __name__ == '__main__':
    # Mock defensive stats (in reality, calculate from play-by-play data)
    defensive_stats = {
        'BAL': {'passing_epa_allowed': -0.05, 'pressure_rate_generated': 32, 'cpoe_allowed': -1.5},  # Elite
        'SF': {'passing_epa_allowed': -0.03, 'pressure_rate_generated': 30, 'cpoe_allowed': -1.2},   # Elite
        'BUF': {'passing_epa_allowed': -0.02, 'pressure_rate_generated': 28, 'cpoe_allowed': -0.8},  # Elite
        'DAL': {'passing_epa_allowed': 0.00, 'pressure_rate_generated': 26, 'cpoe_allowed': 0.0},    # Above Avg
        'KC': {'passing_epa_allowed': 0.01, 'pressure_rate_generated': 25, 'cpoe_allowed': 0.2},     # Above Avg
        'MIA': {'passing_epa_allowed': 0.02, 'pressure_rate_generated': 23, 'cpoe_allowed': 0.5},    # Below Avg
        'LAC': {'passing_epa_allowed': 0.03, 'pressure_rate_generated': 22, 'cpoe_allowed': 1.0},    # Below Avg
        'ARI': {'passing_epa_allowed': 0.06, 'pressure_rate_generated': 20, 'cpoe_allowed': 2.0},    # Weak
        'CAR': {'passing_epa_allowed': 0.07, 'pressure_rate_generated': 19, 'cpoe_allowed': 2.5},    # Weak
        'NYG': {'passing_epa_allowed': 0.08, 'pressure_rate_generated': 18, 'cpoe_allowed': 3.0},    # Weak
    }

    bucketer = DefenseBucketer()

    # Bucket defenses by EPA
    epa_buckets = bucketer.bucket_defenses_by_epa(defensive_stats)

    print("Defense Buckets by Passing EPA Allowed:")
    print("=" * 60)
    for bucket_name, bucket in epa_buckets.items():
        print(f"\n{bucket_name.upper()} ({bucket.threshold_desc}):")
        print(f"  Avg EPA Allowed: {bucket.avg_metric_value:.3f}")
        print(f"  Teams: {', '.join(bucket.teams)}")

    # Mock Josh Allen's games
    josh_allen_games = [
        {'opponent': 'BAL', 'passing_yards': 240, 'qb_epa': 0.05, 'success_plays': 16, 'total_plays': 35},
        {'opponent': 'SF', 'passing_yards': 255, 'qb_epa': 0.08, 'success_plays': 17, 'total_plays': 34},
        {'opponent': 'BUF', 'passing_yards': 265, 'qb_epa': 0.10, 'success_plays': 18, 'total_plays': 33},
        {'opponent': 'MIA', 'passing_yards': 295, 'qb_epa': 0.18, 'success_plays': 21, 'total_plays': 36},
        {'opponent': 'ARI', 'passing_yards': 320, 'qb_epa': 0.22, 'success_plays': 23, 'total_plays': 38},
        {'opponent': 'CAR', 'passing_yards': 310, 'qb_epa': 0.20, 'success_plays': 22, 'total_plays': 37},
    ]

    # Analyze Josh Allen vs each bucket
    print("\n\n" + "=" * 60)
    print("Josh Allen Performance vs Defense Buckets:")
    print("=" * 60)

    bucket_analysis = bucketer.get_full_bucket_analysis(
        josh_allen_games,
        epa_buckets,
        stat_name='passing_yards'
    )

    for bucket_name, perf in bucket_analysis.items():
        if perf['games'] > 0:
            print(f"\nVs {bucket_name.upper()} Defenses ({perf['bucket_desc']}):")
            print(f"  Games: {perf['games']}")
            print(f"  Avg Yards: {perf['avg_yards']:.1f}")
            print(f"  Avg EPA: {perf['avg_epa']:.3f}")
            print(f"  Success Rate: {perf['success_rate']:.1f}%")
            print(f"  Sample Size: {perf['sample_size']}")
            print(f"  Teams Faced: {', '.join(perf['teams_faced'])}")

    # Calculate contextual adjustment for upcoming game vs elite defense
    print("\n\n" + "=" * 60)
    print("Contextual Adjustment for Upcoming Game:")
    print("=" * 60)

    # Overall average
    overall_avg = statistics.mean([g['passing_yards'] for g in josh_allen_games])
    print(f"\nJosh Allen Season Average: {overall_avg:.1f} yards")

    # Vs elite defenses
    if 'elite' in bucket_analysis and bucket_analysis['elite']['games'] > 0:
        elite_avg = bucket_analysis['elite']['avg_yards']
        adjustment = elite_avg - overall_avg

        print(f"Vs Elite Defenses: {elite_avg:.1f} yards")
        print(f"Contextual Adjustment: {adjustment:+.1f} yards")
        print(f"\n→ If Allen faces BAL (elite defense) next week:")
        print(f"   Projection = {overall_avg:.1f} + {adjustment:+.1f} = {overall_avg + adjustment:.1f} yards")
