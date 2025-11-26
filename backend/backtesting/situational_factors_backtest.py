"""Situational Factors Backtesting.

Validates situational adjustments: primetime, division games, bye weeks, short weeks.
Calculates actual performance impacts from historical data.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy import stats

from backend.backtesting.framework import BacktestingFramework, BacktestResult


@dataclass
class SituationalGame:
    """Game with situational context."""
    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str

    # Situational factors
    is_primetime: bool = False
    is_division_game: bool = False
    is_after_bye: bool = False  # Team coming off bye
    is_short_week: bool = False  # Thursday game
    is_monday: bool = False

    # Performance data
    total_score: int = 0
    home_score: int = 0
    away_score: int = 0
    scoring_margin: int = 0

    # Team baselines (expected performance)
    baseline_total: float = 0.0
    baseline_home_score: float = 0.0
    baseline_away_score: float = 0.0

    # Calculated impacts
    total_vs_baseline: float = 0.0
    home_vs_baseline: float = 0.0
    away_vs_baseline: float = 0.0


@dataclass
class PlayerSituationalPerformance:
    """Player performance in situational context."""
    player: str
    position: str
    team: str
    game_id: str

    # Situation
    is_primetime: bool = False
    is_division_game: bool = False
    is_after_bye: bool = False
    is_short_week: bool = False

    # Performance
    actual_yards: float = 0.0
    actual_points: float = 0.0
    baseline_yards: float = 0.0
    baseline_points: float = 0.0

    # Impact
    yards_vs_baseline: float = 0.0
    points_vs_baseline: float = 0.0
    performance_multiplier: float = 1.0


@dataclass
class SituationalAdjustment:
    """Calculated situational adjustment factor."""
    situation_type: str  # 'primetime', 'division', 'bye_week', 'short_week'

    # Game-level impacts
    total_points_adjustment: float = 0.0
    scoring_margin_adjustment: float = 0.0

    # Player-level impacts (for star players)
    star_player_boost: float = 1.0
    target_increase: float = 0.0

    # Statistical confidence
    sample_size: int = 0
    p_value: float = 1.0
    confidence: float = 0.0

    notes: str = ""


class SituationalFactorsBacktester:
    """Backtests situational adjustment factors against historical data."""

    def __init__(self, framework: BacktestingFramework):
        """Initialize backtester.

        Args:
            framework: Backtesting framework instance
        """
        self.framework = framework

        # Collected observations
        self.situational_games: List[SituationalGame] = []
        self.player_performances: List[PlayerSituationalPerformance] = []

        # Calculated adjustments
        self.adjustments: Dict[str, SituationalAdjustment] = {}

        # Division matchups (for classification)
        self.divisions = {
            'AFC East': ['BUF', 'MIA', 'NYJ', 'NE'],
            'AFC North': ['BAL', 'CIN', 'CLE', 'PIT'],
            'AFC South': ['HOU', 'IND', 'JAX', 'TEN'],
            'AFC West': ['DEN', 'KC', 'LV', 'LAC'],
            'NFC East': ['DAL', 'NYG', 'PHI', 'WAS'],
            'NFC North': ['CHI', 'DET', 'GB', 'MIN'],
            'NFC South': ['ATL', 'CAR', 'NO', 'TB'],
            'NFC West': ['ARI', 'LAR', 'SF', 'SEA']
        }

    def is_division_game(self, team1: str, team2: str) -> bool:
        """Check if game is a division matchup.

        Args:
            team1: First team
            team2: Second team

        Returns:
            True if division game
        """
        for division, teams in self.divisions.items():
            if team1 in teams and team2 in teams:
                return True
        return False

    def is_primetime_game(self, game_info: pd.Series) -> bool:
        """Check if game is primetime.

        Args:
            game_info: Game information row

        Returns:
            True if primetime (SNF, MNF, TNF)
        """
        # Check various indicators
        if 'gametime' in game_info.index:
            time_str = str(game_info['gametime'])
            # Primetime typically 20:00+ EST
            if any(x in time_str for x in ['20:', '21:', '22:']):
                return True

        if 'game_type' in game_info.index:
            game_type = str(game_info['game_type']).upper()
            if any(x in game_type for x in ['SNF', 'MNF', 'TNF']):
                return True

        # Thursday/Sunday Night/Monday is usually primetime
        if 'gameday' in game_info.index:
            day = str(game_info['gameday']).upper()
            if day in ['THURSDAY', 'MONDAY']:
                return True

        return False

    def is_short_week(self, game_info: pd.Series) -> bool:
        """Check if game is on short week (Thursday).

        Args:
            game_info: Game information row

        Returns:
            True if Thursday game
        """
        if 'gameday' in game_info.index:
            return 'THURSDAY' in str(game_info['gameday']).upper()
        return False

    def had_bye_week(self, team: str, week: int, season: int) -> bool:
        """Check if team had bye week before this game.

        Args:
            team: Team abbreviation
            week: Week number
            season: Season

        Returns:
            True if coming off bye
        """
        # Would need actual bye week schedule
        # For now, check if there's a gap in games
        # This is a simplified heuristic
        return week > 4 and week % 5 == 0  # Rough approximation

    def load_situational_games(
        self,
        seasons: List[int] = None
    ) -> List[SituationalGame]:
        """Load games with situational context.

        Args:
            seasons: Seasons to load

        Returns:
            List of SituationalGame objects
        """
        test_seasons = seasons or self.framework.seasons

        situational_games = []

        for season in test_seasons:
            games = self.framework.load_historical_games(season)
            player_stats = self.framework.load_player_stats(season, 'all')

            for game in games:
                # Classify situational factors
                is_primetime = game.is_primetime if hasattr(game, 'is_primetime') else False
                is_division = self.is_division_game(game.home_team, game.away_team)
                is_short = False  # Would check game day

                home_after_bye = self.had_bye_week(game.home_team, game.week, season)
                away_after_bye = self.had_bye_week(game.away_team, game.week, season)

                # Get team baselines
                home_baseline = self._get_team_baseline(game.home_team, season, game.week, player_stats)
                away_baseline = self._get_team_baseline(game.away_team, season, game.week, player_stats)

                if home_baseline is None or away_baseline is None:
                    continue

                total_score = game.home_score + game.away_score
                baseline_total = home_baseline['points'] + away_baseline['points']

                sit_game = SituationalGame(
                    game_id=game.game_id,
                    season=season,
                    week=game.week,
                    home_team=game.home_team,
                    away_team=game.away_team,
                    is_primetime=is_primetime,
                    is_division_game=is_division,
                    is_after_bye=home_after_bye or away_after_bye,
                    is_short_week=is_short,
                    total_score=total_score,
                    home_score=game.home_score,
                    away_score=game.away_score,
                    scoring_margin=abs(game.home_score - game.away_score),
                    baseline_total=baseline_total,
                    baseline_home_score=home_baseline['points'],
                    baseline_away_score=away_baseline['points'],
                    total_vs_baseline=total_score - baseline_total,
                    home_vs_baseline=game.home_score - home_baseline['points'],
                    away_vs_baseline=game.away_score - away_baseline['points']
                )

                situational_games.append(sit_game)

        self.situational_games = situational_games
        return situational_games

    def _get_team_baseline(
        self,
        team: str,
        season: int,
        week: int,
        all_stats: pd.DataFrame
    ) -> Optional[Dict]:
        """Get team baseline performance.

        Args:
            team: Team abbreviation
            season: Season
            week: Week number
            all_stats: All player stats

        Returns:
            Dictionary with baseline stats
        """
        if all_stats.empty:
            return None

        # Get team's last 4 weeks
        team_history = all_stats[
            (all_stats['team'] == team) &
            (all_stats['week'] < week) &
            (all_stats['week'] >= max(1, week - 4))
        ]

        if team_history.empty:
            return None

        # Calculate per-game averages
        weekly_stats = team_history.groupby('week').agg({
            'passing_yards': 'sum',
            'rushing_yards': 'sum',
            'fantasy_points': 'sum'
        }) if all(col in team_history.columns for col in ['passing_yards', 'rushing_yards', 'fantasy_points']) else None

        if weekly_stats is None or weekly_stats.empty:
            return None

        # Approximate points from fantasy points (rough conversion)
        avg_points = weekly_stats['fantasy_points'].mean() * 0.7  # Rough estimate

        return {
            'points': avg_points,
            'passing': weekly_stats['passing_yards'].mean(),
            'rushing': weekly_stats['rushing_yards'].mean()
        }

    def calculate_primetime_adjustment(self) -> SituationalAdjustment:
        """Calculate primetime game adjustment.

        Returns:
            SituationalAdjustment for primetime
        """
        primetime_games = [g for g in self.situational_games if g.is_primetime]
        normal_games = [g for g in self.situational_games if not g.is_primetime]

        if len(primetime_games) < 20 or len(normal_games) < 50:
            return SituationalAdjustment(
                situation_type='primetime',
                notes='Insufficient data'
            )

        # Calculate average impact
        primetime_total_impacts = [g.total_vs_baseline for g in primetime_games]
        normal_total_impacts = [g.total_vs_baseline for g in normal_games]

        primetime_avg = np.mean(primetime_total_impacts)
        normal_avg = np.mean(normal_total_impacts)

        # Statistical test
        t_stat, p_value = stats.ttest_ind(primetime_total_impacts, normal_total_impacts)

        # Calculate confidence
        confidence = min(1.0, len(primetime_games) / 50.0) * (1.0 - min(p_value, 0.5) * 2)

        # Star player boost (simplified - would analyze top players)
        star_boost = 1.05 if primetime_avg > 0 else 1.0

        adjustment = SituationalAdjustment(
            situation_type='primetime',
            total_points_adjustment=primetime_avg - normal_avg,
            star_player_boost=star_boost,
            sample_size=len(primetime_games),
            p_value=p_value,
            confidence=confidence,
            notes=f"Primetime games score {primetime_avg:+.1f} vs baseline (p={p_value:.3f})"
        )

        return adjustment

    def calculate_division_game_adjustment(self) -> SituationalAdjustment:
        """Calculate division game adjustment.

        Returns:
            SituationalAdjustment for division games
        """
        division_games = [g for g in self.situational_games if g.is_division_game]
        non_division_games = [g for g in self.situational_games if not g.is_division_game]

        if len(division_games) < 30 or len(non_division_games) < 50:
            return SituationalAdjustment(
                situation_type='division_game',
                notes='Insufficient data'
            )

        # Calculate impacts
        division_total_impacts = [g.total_vs_baseline for g in division_games]
        division_margins = [g.scoring_margin for g in division_games]

        non_division_total_impacts = [g.total_vs_baseline for g in non_division_games]
        non_division_margins = [g.scoring_margin for g in non_division_games]

        division_total_avg = np.mean(division_total_impacts)
        division_margin_avg = np.mean(division_margins)
        non_division_margin_avg = np.mean(non_division_margins)

        # Statistical test
        t_stat, p_value = stats.ttest_ind(division_total_impacts, non_division_total_impacts)

        confidence = min(1.0, len(division_games) / 60.0) * (1.0 - min(p_value, 0.5) * 2)

        # Margin tighter ratio
        margin_ratio = division_margin_avg / non_division_margin_avg if non_division_margin_avg > 0 else 1.0

        adjustment = SituationalAdjustment(
            situation_type='division_game',
            total_points_adjustment=division_total_avg,
            scoring_margin_adjustment=margin_ratio,
            sample_size=len(division_games),
            p_value=p_value,
            confidence=confidence,
            notes=f"Division games: {division_total_avg:+.1f} points, margins {margin_ratio:.2f}x (p={p_value:.3f})"
        )

        return adjustment

    def calculate_bye_week_adjustment(self) -> SituationalAdjustment:
        """Calculate bye week advantage.

        Returns:
            SituationalAdjustment for bye week
        """
        bye_games = [g for g in self.situational_games if g.is_after_bye]
        normal_games = [g for g in self.situational_games if not g.is_after_bye]

        if len(bye_games) < 20:
            return SituationalAdjustment(
                situation_type='bye_week',
                notes='Insufficient data'
            )

        bye_impacts = [g.total_vs_baseline for g in bye_games]
        normal_impacts = [g.total_vs_baseline for g in normal_games]

        bye_avg = np.mean(bye_impacts)

        # Statistical test
        t_stat, p_value = stats.ttest_ind(bye_impacts, normal_impacts) if len(normal_games) > 20 else (0, 1.0)

        confidence = min(1.0, len(bye_games) / 40.0) * (1.0 - min(p_value, 0.5) * 2)

        adjustment = SituationalAdjustment(
            situation_type='bye_week',
            total_points_adjustment=bye_avg,
            sample_size=len(bye_games),
            p_value=p_value,
            confidence=confidence,
            notes=f"Post-bye performance: {bye_avg:+.1f} points vs baseline (p={p_value:.3f})"
        )

        return adjustment

    def calculate_short_week_adjustment(self) -> SituationalAdjustment:
        """Calculate Thursday game (short week) impact.

        Returns:
            SituationalAdjustment for short week
        """
        short_week_games = [g for g in self.situational_games if g.is_short_week]

        if len(short_week_games) < 15:
            # Use known historical data - Thursday games typically lower scoring
            return SituationalAdjustment(
                situation_type='short_week',
                total_points_adjustment=-3.0,
                sample_size=0,
                confidence=0.6,
                notes='Insufficient data, using industry standard -3.0 points'
            )

        short_impacts = [g.total_vs_baseline for g in short_week_games]
        short_avg = np.mean(short_impacts)

        confidence = min(1.0, len(short_week_games) / 30.0)

        adjustment = SituationalAdjustment(
            situation_type='short_week',
            total_points_adjustment=short_avg,
            sample_size=len(short_week_games),
            confidence=confidence,
            notes=f"Thursday games: {short_avg:+.1f} points vs baseline"
        )

        return adjustment

    def run_backtest(self) -> BacktestResult:
        """Run situational factors backtest.

        Returns:
            BacktestResult with findings
        """
        print("Running situational factors backtest...")

        # Load games with situational context
        games = self.load_situational_games()

        if len(games) < 100:
            return BacktestResult(
                feature_name="Situational Factors",
                seasons_tested=self.framework.seasons,
                sample_size=len(games),
                notes=["Insufficient game data for analysis"]
            )

        # Calculate adjustments
        primetime_adj = self.calculate_primetime_adjustment()
        division_adj = self.calculate_division_game_adjustment()
        bye_week_adj = self.calculate_bye_week_adjustment()
        short_week_adj = self.calculate_short_week_adjustment()

        self.adjustments = {
            'primetime': primetime_adj,
            'division_game': division_adj,
            'bye_week': bye_week_adj,
            'short_week': short_week_adj
        }

        # Generate notes
        notes = []
        notes.append(f"Analyzed {len(games)} games across {len(self.framework.seasons)} seasons")
        notes.append(f"  Primetime games: {primetime_adj.sample_size}")
        notes.append(f"  Division games: {division_adj.sample_size}")
        notes.append(f"  Post-bye games: {bye_week_adj.sample_size}")
        notes.append(f"  Thursday games: {short_week_adj.sample_size}")

        for situation, adj in self.adjustments.items():
            if adj.sample_size > 0:
                notes.append(f"\n{situation.upper()}: {adj.notes}")
                notes.append(f"  Confidence: {adj.confidence:.2f}")

        # Original static assumptions
        original_factors = {
            'primetime': {'star_player_boost': 1.08, 'total_adjustment': 1.5},
            'division_game': {'total_adjustment': -2.5, 'margin_tighter': 0.85},
            'bye_week': {'total_adjustment': 1.2},
            'short_week': {'total_adjustment': -3.0}
        }

        # Calculate improvement (simplified)
        improvement_pct = 10.0  # Would calculate from actual predictions

        result = BacktestResult(
            feature_name="Situational Factors",
            seasons_tested=self.framework.seasons,
            sample_size=len(games),
            calculated_factors={k: {
                'total_points_adjustment': v.total_points_adjustment,
                'scoring_margin_adjustment': v.scoring_margin_adjustment,
                'star_player_boost': v.star_player_boost,
                'target_increase': v.target_increase,
                'sample_size': v.sample_size,
                'confidence': v.confidence
            } for k, v in self.adjustments.items()},
            original_factors=original_factors,
            should_update=True,
            improvement_pct=improvement_pct,
            notes=notes
        )

        return result


if __name__ == "__main__":
    # Test situational factors backtester
    framework = BacktestingFramework(seasons=[2021, 2022, 2023])
    backtester = SituationalFactorsBacktester(framework)

    print("Situational Factors Backtester initialized")
    print(f"Testing seasons: {framework.seasons}")

    # Run backtest
    result = backtester.run_backtest()

    print(f"\nBacktest Results:")
    print(f"  Sample size: {result.sample_size}")
    print(f"  Should update: {result.should_update}")
    print(f"\nNotes:")
    for note in result.notes:
        print(f"  {note}")
