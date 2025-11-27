"""Defense Matchup Backtesting.

Calculates actual positional defense performance from historical data.
Validates matchup adjustment factors against actual game results.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

from backend.backtesting.framework import BacktestingFramework, BacktestResult


@dataclass
class PositionalPerformance:
    """Actual positional performance against a defense."""
    player: str
    position_role: str  # WR1, WR2, Slot, RB_rush, RB_recv, TE
    opponent_defense: str
    season: int
    week: int

    # Player's averages before this game
    avg_yards: float = 0.0
    avg_targets: float = 0.0
    avg_receptions: float = 0.0

    # Actual performance in this game
    actual_yards: float = 0.0
    actual_targets: float = 0.0
    actual_receptions: float = 0.0

    # Calculated impact
    yards_vs_expectation: float = 0.0
    performance_multiplier: float = 1.0


@dataclass
class DefensePositionalStats:
    """Defense performance against specific position."""
    defense_team: str
    position_role: str
    season: int

    games_played: int = 0
    total_yards_allowed: float = 0.0
    total_targets_allowed: float = 0.0
    total_receptions_allowed: float = 0.0

    # Calculated averages
    yards_per_game: float = 0.0
    yards_per_target: float = 0.0
    league_rank: int = 16

    # Performance factor vs league average
    adjustment_factor: float = 1.0
    confidence: float = 0.5


class DefenseMatchupBacktester:
    """Backtests defense matchup predictions against historical data."""

    def __init__(self, framework: BacktestingFramework):
        """Initialize backtester.

        Args:
            framework: Backtesting framework instance
        """
        self.framework = framework

        # Collected observations
        self.observations: List[PositionalPerformance] = []

        # Calculated defense stats
        self.defense_stats: Dict[str, Dict] = {}

    def classify_position_role(
        self,
        player: str,
        position: str,
        team_stats: pd.DataFrame
    ) -> str:
        """Classify player's role (WR1, WR2, Slot, etc.).

        Args:
            player: Player name
            position: Position (WR, RB, TE)
            team_stats: Team stats for context

        Returns:
            Position role string
        """
        if position == 'WR':
            # Sort WRs by targets
            team_wrs = team_stats[team_stats['position'] == 'WR'].sort_values('targets', ascending=False)
            wr_list = team_wrs['player'].tolist()

            if player not in wr_list:
                return 'WR3'

            idx = wr_list.index(player)
            if idx == 0:
                return 'WR1'
            elif idx == 1:
                return 'WR2'
            else:
                # Check if slot receiver (would need alignment data)
                return 'Slot'

        elif position == 'RB':
            # Determine if rush or receiving specialist
            player_stats = team_stats[team_stats['player'] == player].iloc[0]
            carries = player_stats.get('carries', 0)
            targets = player_stats.get('targets', 0)

            if carries > targets * 2:
                return 'RB_rush'
            else:
                return 'RB_recv'

        elif position == 'TE':
            team_tes = team_stats[team_stats['position'] == 'TE'].sort_values('targets', ascending=False)
            te_list = team_tes['player'].tolist()

            if player not in te_list or te_list.index(player) == 0:
                return 'TE'
            else:
                return 'TE2'

        return position

    def calculate_defense_stats(
        self,
        seasons: List[int] = None
    ) -> Dict[str, Dict]:
        """Calculate actual defense stats by position from historical data.

        Args:
            seasons: Seasons to analyze

        Returns:
            Dictionary of defense stats by team and position
        """
        test_seasons = seasons or self.framework.seasons

        # Collect all defensive performances
        defense_performances = defaultdict(lambda: defaultdict(list))

        for season in test_seasons:
            player_stats = self.framework.load_player_stats(season, 'all')

            if player_stats.empty:
                print(f"    No player stats for {season}")
                continue

            print(f"    Season {season}: {len(player_stats)} player records")
            print(f"    Columns: {list(player_stats.columns)}")
            print(f"    Sample row:")
            if not player_stats.empty:
                print(f"      {player_stats.iloc[0].to_dict()}")

            # Group by team and week
            total_weeks = len(player_stats['week'].unique())
            processed_games = 0
            observations_added = 0

            for week_idx, week in enumerate(player_stats['week'].unique()):
                week_stats = player_stats[player_stats['week'] == week]

                # DEBUG: Check first week
                if week_idx == 0:
                    print(f"\n    DEBUG week {week}:")
                    print(f"      Week stats: {len(week_stats)} records")
                    print(f"      Unique game_ids: {len(week_stats['game_id'].unique())}")
                    if not week_stats.empty:
                        print(f"      First game_id: {week_stats.iloc[0]['game_id']}")

                # Process each game (each team's stats are separate rows)
                for game_id in week_stats['game_id'].unique():
                    game_stats = week_stats[week_stats['game_id'] == game_id]
                    processed_games += 1

                    # DEBUG: Check first game
                    if processed_games == 1:
                        print(f"\n    DEBUG first game ({game_id}):")
                        print(f"      Game stats: {len(game_stats)} records")
                        print(f"      Teams in game: {list(game_stats['team'].unique())}")
                        if 'opponent_team' in game_stats.columns:
                            print(f"      Opponent teams: {list(game_stats['opponent_team'].unique())}")

                    # In player stats, each game_id has rows for ONE team playing
                    # The opponent is in opponent_team column
                    teams = game_stats['team'].unique()

                    # Process each team's offensive performance
                    for team in teams:
                        team_players = game_stats[game_stats['team'] == team]

                        if team_players.empty:
                            continue

                        # Get opponent from the opponent_team column
                        opponent = team_players.iloc[0]['opponent_team'] if 'opponent_team' in team_players.columns else None

                        if not opponent:
                            continue

                        # DEBUG: Check first team
                        if processed_games == 1 and team == teams[0]:
                            print(f"      Team {team} vs {opponent}, players: {len(team_players)}")

                        # DEBUG: Check first time only
                        if not hasattr(self, '_debug_loop_shown'):
                            self._debug_loop_shown = True
                            print(f"\n    DEBUG player loop:")
                            print(f"      Team: {team}, Opponent: {opponent}")
                            print(f"      Team players: {len(team_players)}")
                            if not team_players.empty:
                                print(f"      First player: {team_players.iloc[0]['player']}, Position: {team_players.iloc[0]['position']}")

                        for _, player_row in team_players.iterrows():
                            player = player_row['player']
                            position = player_row['position']

                            # DEBUG: Track iterations
                            if not hasattr(self, '_player_iterations'):
                                self._player_iterations = 0
                            self._player_iterations += 1

                            # Classify role
                            try:
                                role = self.classify_position_role(player, position, team_players)
                            except Exception as e:
                                if not hasattr(self, '_debug_classify_error_shown'):
                                    self._debug_classify_error_shown = True
                                    print(f"\n    ERROR in classify_position_role: {e}")
                                    print(f"      Player: {player}, Position: {position}")
                                continue

                            # Get player's baseline
                            baseline = self._get_player_baseline(
                                player, season, week, player_stats
                            )

                            if baseline is None:
                                continue
                            else:
                                observations_added += 1

                            # Record performance vs this defense
                            performance = PositionalPerformance(
                                player=player,
                                position_role=role,
                                opponent_defense=opponent,
                                season=season,
                                week=week,
                                avg_yards=baseline['yards'],
                                avg_targets=baseline['targets'],
                                actual_yards=player_row.get('receiving_yards', 0) + player_row.get('rushing_yards', 0),
                                actual_targets=player_row.get('targets', 0) + player_row.get('carries', 0)
                            )

                            performance.yards_vs_expectation = performance.actual_yards - performance.avg_yards
                            if performance.avg_yards > 0:
                                performance.performance_multiplier = performance.actual_yards / performance.avg_yards

                            # Store observation
                            defense_performances[opponent][role].append(performance)
                            self.observations.append(performance)

            total_iters = getattr(self, '_player_iterations', 0)
            print(f"    Processed {processed_games} games, {total_iters} player iterations, added {observations_added} observations")

        # Calculate defense stats from observations
        defense_stats = {}

        for defense_team, position_data in defense_performances.items():
            defense_stats[defense_team] = {}

            for position_role, performances in position_data.items():
                if not performances:
                    continue

                total_yards = sum(p.actual_yards for p in performances)
                games = len(performances)
                yards_per_game = total_yards / games if games > 0 else 0

                # Calculate adjustment factor vs league average
                league_avg = self._get_league_average(position_role, defense_performances)
                adjustment_factor = yards_per_game / league_avg if league_avg > 0 else 1.0

                # Calculate confidence based on sample size
                confidence = min(1.0, games / 16.0)  # Full confidence at 16+ games

                stats = DefensePositionalStats(
                    defense_team=defense_team,
                    position_role=position_role,
                    season=seasons[0] if seasons else 2023,  # Would track per season
                    games_played=games,
                    total_yards_allowed=total_yards,
                    yards_per_game=yards_per_game,
                    adjustment_factor=adjustment_factor,
                    confidence=confidence
                )

                defense_stats[defense_team][position_role] = stats

        self.defense_stats = defense_stats
        return defense_stats

    def _get_player_baseline(
        self,
        player: str,
        season: int,
        week: int,
        all_stats: pd.DataFrame
    ) -> Optional[Dict]:
        """Get player's baseline stats before a game.

        Args:
            player: Player name
            season: Season
            week: Week number
            all_stats: All player stats

        Returns:
            Dictionary with baseline stats or None
        """
        # Get player's history from the entire season before this week
        # Changed from last 4 weeks to entire season for better sample size
        player_history = all_stats[
            (all_stats['player'] == player) &
            (all_stats['week'] < week)
        ]

        # DEBUG: Check first time only
        if not hasattr(self, '_debug_baseline_shown'):
            self._debug_baseline_shown = True
            print(f"\n    DEBUG _get_player_baseline:")
            print(f"      Player: {player}, Week: {week}")
            print(f"      Player history length: {len(player_history)}")
            if not player_history.empty:
                print(f"      Sample: {player_history.iloc[0][['week', 'player']].to_dict()}")

        # Require at least 1 game of history (lowered from 2)
        if player_history.empty or len(player_history) < 1:
            return None

        # Calculate total yards (receiving + rushing) per game, then average
        receiving_yards = player_history['receiving_yards'].fillna(0) if 'receiving_yards' in player_history.columns else pd.Series([0] * len(player_history))
        rushing_yards = player_history['rushing_yards'].fillna(0) if 'rushing_yards' in player_history.columns else pd.Series([0] * len(player_history))
        total_yards = (receiving_yards + rushing_yards).mean()

        targets = player_history['targets'].fillna(0).mean() if 'targets' in player_history.columns else 0
        receptions = player_history['receptions'].fillna(0).mean() if 'receptions' in player_history.columns else 0

        return {
            'yards': total_yards,
            'targets': targets,
            'receptions': receptions
        }

    def _get_league_average(
        self,
        position_role: str,
        all_defense_performances: Dict
    ) -> float:
        """Calculate league average for position role.

        Args:
            position_role: Position role
            all_defense_performances: All defense performances

        Returns:
            League average yards per game
        """
        all_yards = []

        for defense_team, position_data in all_defense_performances.items():
            if position_role in position_data:
                performances = position_data[position_role]
                total_yards = sum(p.actual_yards for p in performances)
                games = len(performances)
                if games > 0:
                    all_yards.append(total_yards / games)

        if not all_yards:
            # Default league averages
            defaults = {
                'WR1': 65.0,
                'WR2': 45.0,
                'Slot': 40.0,
                'RB_rush': 55.0,
                'RB_recv': 25.0,
                'TE': 45.0
            }
            return defaults.get(position_role, 50.0)

        return np.mean(all_yards)

    def validate_matchup_factors(self) -> Dict:
        """Validate matchup adjustment factors.

        Returns:
            Validation results
        """
        # Test predictions with calculated factors vs original factors
        predicted_original = []
        predicted_optimized = []
        actuals = []

        original_factors = {
            'WR1': 1.0,
            'WR2': 1.0,
            'Slot': 1.0,
            'RB_rush': 1.0,
            'RB_recv': 1.0,
            'TE': 1.0
        }

        for obs in self.observations:
            if obs.avg_yards == 0:
                continue

            # Original prediction (no adjustment)
            pred_original = obs.avg_yards * original_factors.get(obs.position_role, 1.0)

            # Optimized prediction (with calculated factor)
            defense_stat = self.defense_stats.get(obs.opponent_defense, {}).get(obs.position_role)
            if defense_stat:
                pred_optimized = obs.avg_yards * defense_stat.adjustment_factor
            else:
                pred_optimized = obs.avg_yards

            predicted_original.append(pred_original)
            predicted_optimized.append(pred_optimized)
            actuals.append(obs.actual_yards)

        # Calculate metrics
        metrics_original = self.framework.calculate_metrics(predicted_original, actuals)
        metrics_optimized = self.framework.calculate_metrics(predicted_optimized, actuals)

        # Calculate improvement with division by zero protection
        improvement = {
            'rmse_improvement': ((metrics_original['rmse'] - metrics_optimized['rmse']) / metrics_original['rmse']) * 100 if metrics_original['rmse'] > 0 else 0.0,
            'mae_improvement': ((metrics_original['mae'] - metrics_optimized['mae']) / metrics_original['mae']) * 100 if metrics_original['mae'] > 0 else 0.0,
            'correlation_improvement': metrics_optimized['correlation'] - metrics_original['correlation']
        }

        return {
            'original': metrics_original,
            'optimized': metrics_optimized,
            'improvement': improvement
        }

    def run_backtest(self) -> BacktestResult:
        """Run defense matchup backtest.

        Returns:
            BacktestResult with findings
        """
        print("Running defense matchup backtest...")

        # Calculate defense stats from historical data
        print("  Calculating defense stats...")
        defense_stats = self.calculate_defense_stats()
        print(f"  Found {len(self.observations)} observations")
        print(f"  Calculated stats for {len(defense_stats)} defenses")

        # Validate matchup factors
        print("  Validating matchup factors...")
        validation = self.validate_matchup_factors()

        # Generate notes
        notes = []
        notes.append(f"Analyzed {len(self.observations)} positional matchups")
        notes.append(f"Calculated stats for {len(defense_stats)} defenses")

        # Sample findings
        for team, positions in list(defense_stats.items())[:3]:
            notes.append(f"\n{team} Defense:")
            for role, stats in positions.items():
                notes.append(f"  vs {role}: {stats.yards_per_game:.1f} YPG allowed (factor: {stats.adjustment_factor:.2f}, n={stats.games_played})")

        notes.append(f"\nImprovement vs baseline:")
        notes.append(f"  RMSE: {validation['improvement']['rmse_improvement']:+.1f}%")
        notes.append(f"  MAE: {validation['improvement']['mae_improvement']:+.1f}%")
        notes.append(f"  Correlation: {validation['improvement']['correlation_improvement']:+.3f}")

        # Calculate overall improvement
        improvement_pct = (
            validation['improvement']['rmse_improvement'] +
            validation['improvement']['mae_improvement']
        ) / 2

        result = BacktestResult(
            feature_name="Defense Matchup Adjustments",
            seasons_tested=self.framework.seasons,
            sample_size=len(self.observations),
            rmse=validation['optimized']['rmse'],
            mae=validation['optimized']['mae'],
            correlation=validation['optimized']['correlation'],
            r_squared=validation['optimized']['r_squared'],
            calculated_factors={team: {role: {
                'adjustment_factor': stats.adjustment_factor,
                'yards_per_game': stats.yards_per_game,
                'confidence': stats.confidence,
                'games': stats.games_played
            } for role, stats in positions.items()} for team, positions in defense_stats.items()},
            should_update=improvement_pct > 5.0,
            improvement_pct=improvement_pct,
            notes=notes
        )

        return result


if __name__ == "__main__":
    # Test defense matchup backtester
    framework = BacktestingFramework(seasons=[2022, 2023])
    backtester = DefenseMatchupBacktester(framework)

    print("Defense Matchup Backtester initialized")
    print(f"Testing seasons: {framework.seasons}")

    # Run backtest
    result = backtester.run_backtest()

    print(f"\nBacktest Results:")
    print(f"  Sample size: {result.sample_size}")
    print(f"  RMSE: {result.rmse:.2f}")
    print(f"  MAE: {result.mae:.2f}")
    print(f"  Should update: {result.should_update}")
    print(f"  Improvement: {result.improvement_pct:+.1f}%")
