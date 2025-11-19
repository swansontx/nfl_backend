"""Contextual Performance Splits - Analyze player performance under specific conditions.

This module helps answer questions like:
- "How does Josh Allen perform when facing 30%+ pressure?"
- "How do QBs perform against the Ravens when they generate high pressure?"
- "How does Tyreek Hill do vs elite secondaries (CPOE allowed < -2%)?"

These splits provide conditional expectations that improve prediction accuracy.
"""

from typing import Dict, List, Optional, Tuple
import statistics
from dataclasses import dataclass


@dataclass
class PerformanceSplit:
    """Performance statistics under a specific condition."""
    condition_name: str
    condition_value: str
    games_count: int

    # Basic stats
    passing_yards_avg: float = 0.0
    rushing_yards_avg: float = 0.0
    receiving_yards_avg: float = 0.0

    # Advanced metrics
    epa_avg: float = 0.0
    success_rate_avg: float = 0.0
    cpoe_avg: float = 0.0

    # Comparison to baseline
    yards_delta: float = 0.0  # vs player's overall average
    epa_delta: float = 0.0


class ContextualSplitter:
    """Analyzes player performance under different conditions."""

    def __init__(self):
        self.min_games_threshold = 3  # Minimum games for reliable split

    def analyze_qb_vs_pressure(
        self,
        player_games: List[Dict],
        pressure_threshold: float = 30.0
    ) -> Dict[str, PerformanceSplit]:
        """Analyze QB performance split by pressure faced.

        Args:
            player_games: List of game dictionaries with metrics
            pressure_threshold: Percentage to split high/low pressure (default 30%)

        Returns:
            Dict with 'low_pressure' and 'high_pressure' splits
        """
        low_pressure_games = []
        high_pressure_games = []

        for game in player_games:
            attempts = game.get('attempts', 0)
            pressures = game.get('qb_pressures', 0)

            if attempts < 15:  # Skip games with low attempts
                continue

            pressure_rate = (pressures / attempts * 100) if attempts > 0 else 0

            if pressure_rate < pressure_threshold:
                low_pressure_games.append(game)
            else:
                high_pressure_games.append(game)

        # Calculate overall baseline
        all_games = low_pressure_games + high_pressure_games
        baseline_yards = self._avg_stat(all_games, 'passing_yards')
        baseline_epa = self._avg_stat(all_games, 'qb_epa')

        # Build splits
        splits = {}

        if len(low_pressure_games) >= self.min_games_threshold:
            low_yards = self._avg_stat(low_pressure_games, 'passing_yards')
            low_epa = self._avg_stat(low_pressure_games, 'qb_epa')

            splits['low_pressure'] = PerformanceSplit(
                condition_name='pressure_rate',
                condition_value=f'<{pressure_threshold}%',
                games_count=len(low_pressure_games),
                passing_yards_avg=low_yards,
                epa_avg=low_epa,
                success_rate_avg=self._avg_stat(low_pressure_games, 'success_rate'),
                cpoe_avg=self._avg_cpoe(low_pressure_games),
                yards_delta=low_yards - baseline_yards,
                epa_delta=low_epa - baseline_epa
            )

        if len(high_pressure_games) >= self.min_games_threshold:
            high_yards = self._avg_stat(high_pressure_games, 'passing_yards')
            high_epa = self._avg_stat(high_pressure_games, 'qb_epa')

            splits['high_pressure'] = PerformanceSplit(
                condition_name='pressure_rate',
                condition_value=f'≥{pressure_threshold}%',
                games_count=len(high_pressure_games),
                passing_yards_avg=high_yards,
                epa_avg=high_epa,
                success_rate_avg=self._avg_stat(high_pressure_games, 'success_rate'),
                cpoe_avg=self._avg_cpoe(high_pressure_games),
                yards_delta=high_yards - baseline_yards,
                epa_delta=high_epa - baseline_epa
            )

        return splits

    def analyze_vs_defense_quality(
        self,
        player_games: List[Dict],
        defense_epa_rankings: Dict[str, float]  # team -> EPA allowed rank
    ) -> Dict[str, PerformanceSplit]:
        """Analyze player performance vs top-10 vs bottom-22 defenses.

        Args:
            player_games: List of game dictionaries
            defense_epa_rankings: Dict mapping opponent -> defensive EPA rank (1-32)

        Returns:
            Dict with 'vs_elite_defense' and 'vs_weak_defense' splits
        """
        elite_defense_games = []  # Top 10 defenses
        weak_defense_games = []   # Bottom 22 defenses

        for game in player_games:
            opponent = game.get('opponent', '')
            if not opponent or opponent not in defense_epa_rankings:
                continue

            rank = defense_epa_rankings[opponent]

            if rank <= 10:
                elite_defense_games.append(game)
            elif rank >= 23:
                weak_defense_games.append(game)

        # Determine stat based on position
        position = player_games[0].get('position', '') if player_games else ''

        if position == 'QB':
            stat_name = 'passing_yards'
            epa_name = 'qb_epa'
        elif position in ['RB', 'FB']:
            stat_name = 'rushing_yards'
            epa_name = 'rushing_epa'
        else:
            stat_name = 'receiving_yards'
            epa_name = 'receiving_epa'

        # Calculate baseline
        all_games = elite_defense_games + weak_defense_games
        baseline_yards = self._avg_stat(all_games, stat_name)
        baseline_epa = self._avg_stat(all_games, epa_name)

        splits = {}

        if len(elite_defense_games) >= self.min_games_threshold:
            elite_yards = self._avg_stat(elite_defense_games, stat_name)
            elite_epa = self._avg_stat(elite_defense_games, epa_name)

            splits['vs_elite_defense'] = PerformanceSplit(
                condition_name='opponent_defense_rank',
                condition_value='Top 10',
                games_count=len(elite_defense_games),
                passing_yards_avg=elite_yards if position == 'QB' else 0,
                rushing_yards_avg=elite_yards if position in ['RB', 'FB'] else 0,
                receiving_yards_avg=elite_yards if position in ['WR', 'TE'] else 0,
                epa_avg=elite_epa,
                success_rate_avg=self._avg_stat(elite_defense_games, 'success_rate'),
                yards_delta=elite_yards - baseline_yards,
                epa_delta=elite_epa - baseline_epa
            )

        if len(weak_defense_games) >= self.min_games_threshold:
            weak_yards = self._avg_stat(weak_defense_games, stat_name)
            weak_epa = self._avg_stat(weak_defense_games, epa_name)

            splits['vs_weak_defense'] = PerformanceSplit(
                condition_name='opponent_defense_rank',
                condition_value='Bottom 10',
                games_count=len(weak_defense_games),
                passing_yards_avg=weak_yards if position == 'QB' else 0,
                rushing_yards_avg=weak_yards if position in ['RB', 'FB'] else 0,
                receiving_yards_avg=weak_yards if position in ['WR', 'TE'] else 0,
                epa_avg=weak_epa,
                success_rate_avg=self._avg_stat(weak_defense_games, 'success_rate'),
                yards_delta=weak_yards - baseline_yards,
                epa_delta=weak_epa - baseline_epa
            )

        return splits

    def analyze_home_away(
        self,
        player_games: List[Dict]
    ) -> Dict[str, PerformanceSplit]:
        """Analyze home vs away performance.

        Args:
            player_games: List of game dictionaries

        Returns:
            Dict with 'home' and 'away' splits
        """
        home_games = [g for g in player_games if g.get('location') == 'Home']
        away_games = [g for g in player_games if g.get('location') == 'Away']

        position = player_games[0].get('position', '') if player_games else ''

        if position == 'QB':
            stat_name = 'passing_yards'
            epa_name = 'qb_epa'
        elif position in ['RB', 'FB']:
            stat_name = 'rushing_yards'
            epa_name = 'rushing_epa'
        else:
            stat_name = 'receiving_yards'
            epa_name = 'receiving_epa'

        baseline_yards = self._avg_stat(player_games, stat_name)
        baseline_epa = self._avg_stat(player_games, epa_name)

        splits = {}

        if len(home_games) >= self.min_games_threshold:
            home_yards = self._avg_stat(home_games, stat_name)
            home_epa = self._avg_stat(home_games, epa_name)

            splits['home'] = PerformanceSplit(
                condition_name='location',
                condition_value='Home',
                games_count=len(home_games),
                passing_yards_avg=home_yards if position == 'QB' else 0,
                rushing_yards_avg=home_yards if position in ['RB', 'FB'] else 0,
                receiving_yards_avg=home_yards if position in ['WR', 'TE'] else 0,
                epa_avg=home_epa,
                yards_delta=home_yards - baseline_yards,
                epa_delta=home_epa - baseline_epa
            )

        if len(away_games) >= self.min_games_threshold:
            away_yards = self._avg_stat(away_games, stat_name)
            away_epa = self._avg_stat(away_games, epa_name)

            splits['away'] = PerformanceSplit(
                condition_name='location',
                condition_value='Away',
                games_count=len(away_games),
                passing_yards_avg=away_yards if position == 'QB' else 0,
                rushing_yards_avg=away_yards if position in ['RB', 'FB'] else 0,
                receiving_yards_avg=away_yards if position in ['WR', 'TE'] else 0,
                epa_avg=away_epa,
                yards_delta=away_yards - baseline_yards,
                epa_delta=away_epa - baseline_epa
            )

        return splits

    def get_contextual_adjustment(
        self,
        splits: Dict[str, PerformanceSplit],
        upcoming_condition: str
    ) -> float:
        """Get yards adjustment for upcoming game based on splits.

        Args:
            splits: Performance splits dict
            upcoming_condition: Which condition applies to upcoming game

        Returns:
            Yards adjustment (positive or negative)
        """
        if upcoming_condition not in splits:
            return 0.0

        split = splits[upcoming_condition]
        return split.yards_delta

    def _avg_stat(self, games: List[Dict], stat_name: str) -> float:
        """Calculate average of a stat across games."""
        if not games:
            return 0.0

        values = [g.get(stat_name, 0) for g in games]
        return statistics.mean(values) if values else 0.0

    def _avg_cpoe(self, games: List[Dict]) -> float:
        """Calculate average CPOE across games."""
        if not games:
            return 0.0

        cpoe_values = []
        for game in games:
            cpoe_sum = game.get('cpoe_sum', 0)
            cpoe_count = game.get('cpoe_count', 0)
            if cpoe_count > 0:
                cpoe_values.append(cpoe_sum / cpoe_count * 100)

        return statistics.mean(cpoe_values) if cpoe_values else 0.0


# Example usage and testing
if __name__ == '__main__':
    # Example: Josh Allen's games
    josh_allen_games = [
        # Low pressure games
        {'game_id': '1', 'attempts': 35, 'qb_pressures': 7, 'passing_yards': 295,
         'qb_epa': 0.18, 'success_rate': 54, 'cpoe_sum': 3.2, 'cpoe_count': 1},
        {'game_id': '2', 'attempts': 32, 'qb_pressures': 6, 'passing_yards': 280,
         'qb_epa': 0.15, 'success_rate': 52, 'cpoe_sum': 2.8, 'cpoe_count': 1},
        {'game_id': '3', 'attempts': 38, 'qb_pressures': 8, 'passing_yards': 310,
         'qb_epa': 0.20, 'success_rate': 55, 'cpoe_sum': 4.1, 'cpoe_count': 1},

        # High pressure games
        {'game_id': '4', 'attempts': 40, 'qb_pressures': 14, 'passing_yards': 245,
         'qb_epa': 0.08, 'success_rate': 47, 'cpoe_sum': 1.2, 'cpoe_count': 1},
        {'game_id': '5', 'attempts': 36, 'qb_pressures': 12, 'passing_yards': 238,
         'qb_epa': 0.06, 'success_rate': 45, 'cpoe_sum': 0.8, 'cpoe_count': 1},
        {'game_id': '6', 'attempts': 42, 'qb_pressures': 15, 'passing_yards': 252,
         'qb_epa': 0.05, 'success_rate': 46, 'cpoe_sum': -0.5, 'cpoe_count': 1},
        {'game_id': '7', 'attempts': 38, 'qb_pressures': 13, 'passing_yards': 240,
         'qb_epa': 0.07, 'success_rate': 48, 'cpoe_sum': 1.5, 'cpoe_count': 1},
    ]

    splitter = ContextualSplitter()

    # Analyze pressure splits
    pressure_splits = splitter.analyze_qb_vs_pressure(josh_allen_games, pressure_threshold=30.0)

    print("Josh Allen Pressure Splits:")
    print("="*60)

    if 'low_pressure' in pressure_splits:
        split = pressure_splits['low_pressure']
        print(f"\nLow Pressure ({split.condition_value}):")
        print(f"  Games: {split.games_count}")
        print(f"  Avg Yards: {split.passing_yards_avg:.1f}")
        print(f"  Avg EPA: {split.epa_avg:.3f}")
        print(f"  Success Rate: {split.success_rate_avg:.1f}%")
        print(f"  CPOE: {split.cpoe_avg:+.1f}%")
        print(f"  Delta from baseline: {split.yards_delta:+.1f} yards")

    if 'high_pressure' in pressure_splits:
        split = pressure_splits['high_pressure']
        print(f"\nHigh Pressure ({split.condition_value}):")
        print(f"  Games: {split.games_count}")
        print(f"  Avg Yards: {split.passing_yards_avg:.1f}")
        print(f"  Avg EPA: {split.epa_avg:.3f}")
        print(f"  Success Rate: {split.success_rate_avg:.1f}%")
        print(f"  CPOE: {split.cpoe_avg:+.1f}%")
        print(f"  Delta from baseline: {split.yards_delta:+.1f} yards")

    # Calculate adjustment for upcoming game vs Ravens (35% pressure expected)
    if 'high_pressure' in pressure_splits:
        adjustment = splitter.get_contextual_adjustment(pressure_splits, 'high_pressure')
        print(f"\n{'='*60}")
        print(f"Upcoming Game vs Ravens (35% pressure expected):")
        print(f"  Contextual Adjustment: {adjustment:+.1f} yards")
        print(f"  If baseline projection is 270 yards → {270 + adjustment:.1f} yards")
