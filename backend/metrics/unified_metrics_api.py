"""
Unified Metrics API

Single point of access for all NFL metrics across the system.
Provides:
- Team efficiency metrics (EPA, success rate, red zone, etc.)
- Player feature enrichment (team context, defense matchups)
- Game prediction metrics (pace, turnovers, efficiency)
- Matchup analysis (team vs team comparisons)

Benefits:
- Centralized access to all metrics
- Automatic caching to avoid redundant calculations
- Consistent interface across the codebase
- Easy discovery of available metrics

Usage:
    from backend.metrics.unified_metrics_api import MetricsAPI

    api = MetricsAPI(season=2025)

    # Get team metrics
    kc_metrics = api.get_team_metrics('KC', weeks=[9, 10, 11, 12])

    # Enrich player features
    enriched = api.enrich_player_features(player_df)

    # Get game prediction metrics
    game_metrics = api.get_game_metrics('KC', 'BUF', week=13)

    # Get matchup analysis
    matchup = api.analyze_matchup('KC', 'BUF', week=13)
"""

from typing import Dict, Optional, List, Any
from pathlib import Path
from dataclasses import asdict
import pandas as pd

from backend.analysis.advanced_team_metrics import AdvancedTeamMetricsCalculator
from backend.analysis.team_matchup_analyzer import TeamMatchupAnalyzer
from backend.analysis.defense_matchup_deep import DefenseMatchupAnalyzer
from backend.features.team_metrics_features import TeamMetricsFeatureEngine
from backend.features.game_metrics_features import GameMetricsEngine


class MetricsAPI:
    """
    Unified API for accessing all NFL metrics.

    Provides a single, consistent interface to all metric calculators
    in the system. Handles initialization, caching, and error handling.
    """

    def __init__(
        self,
        season: int = 2025,
        inputs_dir: str = "inputs",
        cache_enabled: bool = True
    ):
        """
        Initialize the unified metrics API.

        Args:
            season: NFL season year
            inputs_dir: Directory containing input data files
            cache_enabled: Whether to cache metric calculations
        """
        self.season = season
        self.inputs_dir = Path(inputs_dir)
        self.cache_enabled = cache_enabled

        # Initialize all metric calculators
        self._init_calculators()

        # Cache for metric results
        self._cache = {} if cache_enabled else None

    def _init_calculators(self):
        """Initialize all metric calculation engines."""
        pbp_file = self.inputs_dir / f"play_by_play_{self.season}.parquet"

        # Advanced team metrics (EPA, success rate, red zone, etc.)
        if pbp_file.exists():
            self.team_metrics_calc = AdvancedTeamMetricsCalculator(
                season=self.season,
                pbp_file=pbp_file
            )
        else:
            print(f"Warning: No PBP data for {self.season}, some metrics unavailable")
            self.team_metrics_calc = None

        # Team matchup analyzer (H2H, statistical edges)
        try:
            self.matchup_analyzer = TeamMatchupAnalyzer(season=self.season)
        except Exception as e:
            print(f"Warning: Could not initialize matchup analyzer: {e}")
            self.matchup_analyzer = None

        # Defense matchup analyzer (position-specific ratings)
        try:
            self.defense_analyzer = DefenseMatchupAnalyzer(season=self.season)
        except Exception as e:
            print(f"Warning: Could not initialize defense analyzer: {e}")
            self.defense_analyzer = None

        # Team metrics feature engine (for player enrichment)
        try:
            self.team_features_engine = TeamMetricsFeatureEngine(
                season=self.season,
                inputs_dir=str(self.inputs_dir)
            )
        except Exception as e:
            print(f"Warning: Could not initialize team features engine: {e}")
            self.team_features_engine = None

        # Game metrics engine (for game predictions)
        try:
            self.game_metrics_engine = GameMetricsEngine(
                season=self.season,
                inputs_dir=str(self.inputs_dir)
            )
        except Exception as e:
            print(f"Warning: Could not initialize game metrics engine: {e}")
            self.game_metrics_engine = None

    def _get_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_parts = [prefix] + [str(arg) for arg in args]
        if kwargs:
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return ":".join(key_parts)

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get value from cache if available."""
        if self.cache_enabled and key in self._cache:
            return self._cache[key]
        return None

    def _set_cached(self, key: str, value: Any):
        """Store value in cache."""
        if self.cache_enabled:
            self._cache[key] = value

    def clear_cache(self):
        """Clear all cached metrics."""
        if self.cache_enabled:
            self._cache.clear()

    # ========================================================================
    # TEAM METRICS
    # ========================================================================

    def get_team_metrics(
        self,
        team: str,
        weeks: Optional[List[int]] = None,
        include_defense: bool = True
    ) -> Dict:
        """
        Get comprehensive team metrics.

        Args:
            team: Team abbreviation (e.g., 'KC', 'BUF')
            weeks: Optional list of weeks to include (for recency)
            include_defense: Whether to include defensive metrics

        Returns:
            Dict with all available team metrics
        """
        cache_key = self._get_cache_key('team_metrics', team, weeks, include_defense)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        metrics = {}

        # Get advanced metrics (EPA, success rate, etc.)
        if self.team_metrics_calc:
            try:
                adv_metrics = self.team_metrics_calc.calculate_team_metrics(team, weeks)
                metrics.update(adv_metrics)
            except Exception as e:
                print(f"Error getting team metrics for {team}: {e}")

        # Add team profile (PPG, YPG, etc.)
        if self.matchup_analyzer and hasattr(self.matchup_analyzer, 'team_profiles'):
            profile = self.matchup_analyzer.team_profiles.get(team)
            if profile:
                metrics['points_per_game'] = profile.points_per_game
                metrics['yards_per_game'] = profile.yards_per_game
                metrics['passing_yards_per_game'] = profile.passing_yards_per_game
                metrics['rushing_yards_per_game'] = profile.rushing_yards_per_game
                metrics['home_ppg'] = profile.home_ppg
                metrics['away_ppg'] = profile.away_ppg

        self._set_cached(cache_key, metrics)
        return metrics

    def compare_teams(
        self,
        team_a: str,
        team_b: str,
        weeks: Optional[List[int]] = None
    ) -> Dict:
        """
        Compare two teams across all metrics.

        Args:
            team_a: First team
            team_b: Second team
            weeks: Optional weeks for recency

        Returns:
            Dict with comparison results
        """
        cache_key = self._get_cache_key('compare_teams', team_a, team_b, weeks)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Get metrics for both teams
        metrics_a = self.get_team_metrics(team_a, weeks)
        metrics_b = self.get_team_metrics(team_b, weeks)

        # Compare key metrics
        comparison = {
            'team_a': team_a,
            'team_b': team_b,
            'metrics': {},
            'advantages_a': [],
            'advantages_b': []
        }

        # Compare each metric
        for metric, value_a in metrics_a.items():
            value_b = metrics_b.get(metric, 0)

            if isinstance(value_a, (int, float)) and isinstance(value_b, (int, float)):
                comparison['metrics'][metric] = {
                    team_a: value_a,
                    team_b: value_b,
                    'difference': value_a - value_b,
                    'advantage': team_a if value_a > value_b else team_b
                }

                # Track significant advantages (>10% difference)
                if value_b != 0 and abs((value_a - value_b) / value_b) > 0.10:
                    if value_a > value_b:
                        comparison['advantages_a'].append(metric)
                    else:
                        comparison['advantages_b'].append(metric)

        self._set_cached(cache_key, comparison)
        return comparison

    # ========================================================================
    # PLAYER METRICS
    # ========================================================================

    def enrich_player_features(
        self,
        player_features: pd.DataFrame,
        recency_weeks: int = 4
    ) -> pd.DataFrame:
        """
        Enrich player features with team metrics.

        Args:
            player_features: DataFrame with player stats
            recency_weeks: Number of recent weeks for team metrics

        Returns:
            Enhanced DataFrame with 23 additional team metric columns
        """
        if self.team_features_engine is None:
            print("Warning: Team features engine not available")
            return player_features

        try:
            return self.team_features_engine.enrich_player_dataframe(
                player_features,
                recency_weeks=recency_weeks
            )
        except Exception as e:
            print(f"Error enriching player features: {e}")
            return player_features

    def get_player_context(
        self,
        player_id: str,
        team: str,
        opponent: str,
        position: str,
        week: int,
        recency_weeks: int = 4
    ) -> Dict:
        """
        Get team context metrics for a specific player.

        Args:
            player_id: Player ID
            team: Player's team
            opponent: Opponent team
            position: Player position
            week: Current week
            recency_weeks: Weeks for team metrics

        Returns:
            Dict with team context and matchup metrics
        """
        cache_key = self._get_cache_key(
            'player_context', player_id, team, opponent, position, week
        )
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        context = {
            'player_id': player_id,
            'team': team,
            'opponent': opponent,
            'position': position,
            'week': week
        }

        # Get team metrics
        context['team_metrics'] = self.get_team_metrics(team, weeks=None)
        context['opponent_metrics'] = self.get_team_metrics(opponent, weeks=None)

        # Get defense matchup rating
        if self.team_features_engine:
            defense_rating = self.team_features_engine._get_defense_matchup_rating(
                opponent, position
            )
            context['defense_matchup'] = defense_rating

        self._set_cached(cache_key, context)
        return context

    # ========================================================================
    # GAME METRICS
    # ========================================================================

    def get_game_metrics(
        self,
        home_team: str,
        away_team: str,
        week: int,
        recency_weeks: int = 4
    ) -> Dict:
        """
        Get comprehensive metrics for a game.

        Args:
            home_team: Home team
            away_team: Away team
            week: Week number
            recency_weeks: Weeks for recency metrics

        Returns:
            Dict with game metrics including pace, turnovers, efficiency
        """
        cache_key = self._get_cache_key(
            'game_metrics', home_team, away_team, week, recency_weeks
        )
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        game_metrics = {
            'home_team': home_team,
            'away_team': away_team,
            'week': week
        }

        # Get team metrics for both teams
        game_metrics['home_metrics'] = self.get_team_metrics(home_team)
        game_metrics['away_metrics'] = self.get_team_metrics(away_team)

        # Get enhanced team strengths for game predictions
        if self.game_metrics_engine:
            try:
                weeks_list = list(range(max(1, week - recency_weeks), week))

                home_strength = self.game_metrics_engine.get_enhanced_team_strength(
                    home_team,
                    game_metrics['home_metrics'].get('points_per_game', 21.5),
                    game_metrics['home_metrics'].get('points_allowed_per_game', 21.5),
                    is_home=True,
                    weeks=weeks_list
                )

                away_strength = self.game_metrics_engine.get_enhanced_team_strength(
                    away_team,
                    game_metrics['away_metrics'].get('points_per_game', 21.5),
                    game_metrics['away_metrics'].get('points_allowed_per_game', 21.5),
                    is_home=False,
                    weeks=weeks_list
                )

                # Get metrics summary
                summary = self.game_metrics_engine.get_game_metrics_summary(
                    home_strength, away_strength
                )
                game_metrics['summary'] = summary

            except Exception as e:
                print(f"Error getting game prediction metrics: {e}")

        self._set_cached(cache_key, game_metrics)
        return game_metrics

    def analyze_matchup(
        self,
        home_team: str,
        away_team: str,
        week: int
    ) -> Dict:
        """
        Get complete matchup analysis between two teams.

        Args:
            home_team: Home team
            away_team: Away team
            week: Week number

        Returns:
            Dict with matchup analysis including H2H, edges, predictions
        """
        cache_key = self._get_cache_key('matchup', home_team, away_team, week)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        matchup = {}

        # Get matchup analysis from analyzer
        if self.matchup_analyzer:
            try:
                analysis = self.matchup_analyzer.analyze_matchup(
                    home_team, away_team, week
                )
                matchup['analysis'] = asdict(analysis)
            except Exception as e:
                print(f"Error analyzing matchup: {e}")

        # Get game metrics
        matchup['game_metrics'] = self.get_game_metrics(
            home_team, away_team, week
        )

        # Get team comparison
        matchup['comparison'] = self.compare_teams(home_team, away_team)

        self._set_cached(cache_key, matchup)
        return matchup

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_available_metrics(self) -> Dict[str, List[str]]:
        """
        Get list of all available metrics organized by category.

        Returns:
            Dict mapping category to list of metric names
        """
        available = {
            'team_efficiency': [],
            'team_performance': [],
            'pace': [],
            'turnovers': [],
            'situational': [],
            'defense': []
        }

        if self.team_metrics_calc:
            available['team_efficiency'] = [
                'success_rate_offense', 'success_rate_defense',
                'epa_per_play_offense', 'epa_per_play_defense',
                'completion_pct', 'yards_per_attempt', 'yards_per_carry'
            ]
            available['situational'] = [
                'red_zone_td_pct', 'third_down_pct', 'explosive_play_rate'
            ]
            available['pace'] = [
                'plays_per_game', 'time_of_possession_pct'
            ]
            available['turnovers'] = [
                'turnover_margin', 'turnover_rate', 'takeaway_rate'
            ]

        if self.matchup_analyzer:
            available['team_performance'] = [
                'points_per_game', 'yards_per_game',
                'passing_yards_per_game', 'rushing_yards_per_game',
                'home_ppg', 'away_ppg'
            ]

        return available

    def get_metric_info(self, metric_name: str) -> Dict:
        """
        Get information about a specific metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Dict with metric description, source, and usage
        """
        metric_info_map = {
            'success_rate_offense': {
                'name': 'Success Rate (Offense)',
                'description': 'Percentage of plays that achieve situational goals',
                'calculation': '1st down: 45% of yards, 2nd: 60%, 3rd/4th: conversion',
                'source': 'play_by_play data',
                'typical_range': '40-50%',
                'used_in': ['player_props', 'game_predictions', 'matchup_analysis']
            },
            'epa_per_play_offense': {
                'name': 'EPA per Play (Offense)',
                'description': 'Expected points added per offensive play',
                'calculation': 'From PBP epa column',
                'source': 'play_by_play data',
                'typical_range': '-0.1 to +0.2',
                'used_in': ['player_props', 'game_predictions']
            },
            'plays_per_game': {
                'name': 'Plays Per Game',
                'description': 'Average offensive plays per game (pace metric)',
                'calculation': 'Total plays / games',
                'source': 'play_by_play data',
                'typical_range': '60-70 plays',
                'used_in': ['game_totals', 'player_props']
            },
            'turnover_margin': {
                'name': 'Turnover Margin',
                'description': 'Season turnover differential (takeaways - turnovers)',
                'calculation': 'Interceptions + fumbles recovered - turnovers lost',
                'source': 'play_by_play data',
                'typical_range': '-10 to +10',
                'used_in': ['game_spreads']
            },
            # Add more as needed...
        }

        return metric_info_map.get(metric_name, {
            'name': metric_name,
            'description': 'No description available',
            'source': 'Unknown',
            'used_in': []
        })

    def get_summary(self) -> Dict:
        """
        Get summary of API status and available features.

        Returns:
            Dict with API status information
        """
        return {
            'season': self.season,
            'inputs_dir': str(self.inputs_dir),
            'cache_enabled': self.cache_enabled,
            'cached_items': len(self._cache) if self.cache_enabled else 0,
            'calculators': {
                'team_metrics': self.team_metrics_calc is not None,
                'matchup_analyzer': self.matchup_analyzer is not None,
                'defense_analyzer': self.defense_analyzer is not None,
                'team_features_engine': self.team_features_engine is not None,
                'game_metrics_engine': self.game_metrics_engine is not None
            },
            'available_metrics': self.get_available_metrics()
        }


# Singleton instance for convenience
_default_api = None

def get_metrics_api(season: int = 2025, inputs_dir: str = "inputs") -> MetricsAPI:
    """
    Get or create the default MetricsAPI instance.

    Args:
        season: NFL season
        inputs_dir: Inputs directory

    Returns:
        MetricsAPI instance
    """
    global _default_api
    if _default_api is None or _default_api.season != season:
        _default_api = MetricsAPI(season=season, inputs_dir=inputs_dir)
    return _default_api
