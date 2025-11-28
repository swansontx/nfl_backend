"""
Team Metrics Feature Engineering

Enhances player prop predictions with team-level efficiency metrics.
Integrates AdvancedTeamMetricsCalculator data into the prediction pipeline.
"""

from typing import Dict, Optional, List
import pandas as pd
from pathlib import Path
from backend.analysis.advanced_team_metrics import AdvancedTeamMetricsCalculator
from backend.analysis.defense_matchup_deep import DefenseMatchupAnalyzer


class TeamMetricsFeatureEngine:
    """
    Adds team-level efficiency metrics to player features.

    Features added:
    - Team offensive efficiency (success rate, EPA, red zone %)
    - Team defensive metrics (opponent's defensive strength)
    - Matchup-specific features (explosive play rates, 3rd down %)
    """

    def __init__(self, season: int = 2025, inputs_dir: str = "inputs"):
        self.season = season
        self.inputs_dir = Path(inputs_dir)

        # Initialize metrics calculator
        pbp_file = self.inputs_dir / f"play_by_play_{season}.parquet"
        if not pbp_file.exists():
            print(f"Warning: No play-by-play data for {season}, team metrics unavailable")
            self.calculator = None
        else:
            self.calculator = AdvancedTeamMetricsCalculator(season=season, pbp_file=pbp_file)

        # Initialize defense matchup analyzer
        self.defense_analyzer = DefenseMatchupAnalyzer(season=season)

        # Cache team metrics
        self._team_metrics_cache = {}
        self._opponent_metrics_cache = {}

    def get_team_metrics(self, team: str, weeks: Optional[List[int]] = None) -> Dict:
        """
        Get cached or calculated team metrics.

        Args:
            team: Team abbreviation (e.g., 'KC', 'BUF')
            weeks: Optional list of weeks to include (for recency)

        Returns:
            Dict of team metrics
        """
        if not self.calculator:
            return {}

        cache_key = f"{team}_{weeks}"
        if cache_key not in self._team_metrics_cache:
            try:
                self._team_metrics_cache[cache_key] = self.calculator.calculate_team_metrics(team, weeks)
            except Exception as e:
                print(f"Error calculating metrics for {team}: {e}")
                return {}

        return self._team_metrics_cache[cache_key]

    def enrich_player_features(
        self,
        player_features: Dict,
        team: str,
        opponent: str,
        week: int,
        recency_weeks: int = 4
    ) -> Dict:
        """
        Enrich player features with team-level efficiency metrics.

        Args:
            player_features: Base player feature dict
            team: Player's team
            opponent: Opponent team
            week: Current week
            recency_weeks: Number of recent weeks for metrics (default 4)

        Returns:
            Enhanced feature dict with team metrics
        """
        enhanced = player_features.copy()

        if not self.calculator:
            return enhanced

        # Calculate recent weeks to include
        recent_weeks = list(range(max(1, week - recency_weeks), week))

        # Get team offensive metrics (recent form)
        team_metrics = self.get_team_metrics(team, recent_weeks)

        # Get opponent defensive metrics (recent form)
        opp_metrics = self.get_team_metrics(opponent, recent_weeks)

        # Add offensive efficiency features
        enhanced['team_success_rate'] = team_metrics.get('success_rate_offense', 0.0)
        enhanced['team_epa_per_play'] = team_metrics.get('epa_per_play_offense', 0.0)
        enhanced['team_red_zone_td_pct'] = team_metrics.get('red_zone_td_pct', 0.0)
        enhanced['team_third_down_pct'] = team_metrics.get('third_down_pct', 0.0)
        enhanced['team_explosive_play_rate'] = team_metrics.get('explosive_play_rate', 0.0)

        # Add position-specific offensive metrics
        enhanced['team_completion_pct'] = team_metrics.get('completion_pct', 0.0)
        enhanced['team_yards_per_attempt'] = team_metrics.get('yards_per_attempt', 0.0)
        enhanced['team_yards_per_carry'] = team_metrics.get('yards_per_carry', 0.0)
        enhanced['team_yards_after_catch_per_comp'] = team_metrics.get('yards_after_catch_per_comp', 0.0)

        # Add defensive matchup features (opponent defense)
        enhanced['opp_def_success_rate'] = opp_metrics.get('success_rate_defense', 0.0)
        enhanced['opp_def_epa_allowed'] = opp_metrics.get('epa_per_play_defense', 0.0)
        enhanced['opp_def_explosive_allowed_rate'] = opp_metrics.get('explosive_plays_allowed_rate', 0.0)
        enhanced['opp_def_third_down_allowed'] = opp_metrics.get('third_down_pct_defense', 0.0)

        # Add turnover context
        enhanced['team_turnover_rate'] = team_metrics.get('turnover_rate', 0.0)
        enhanced['opp_def_takeaway_rate'] = opp_metrics.get('takeaway_rate', 0.0)

        # Pace metrics (important for volume props)
        enhanced['team_plays_per_game'] = team_metrics.get('plays_per_game', 0.0)
        enhanced['opp_plays_per_game'] = opp_metrics.get('plays_per_game', 0.0)

        # Calculate matchup advantages
        enhanced['pass_efficiency_edge'] = (
            team_metrics.get('yards_per_attempt', 0.0) -
            opp_metrics.get('yards_per_attempt_defense', 0.0)
        )
        enhanced['rush_efficiency_edge'] = (
            team_metrics.get('yards_per_carry', 0.0) -
            opp_metrics.get('yards_per_carry_defense', 0.0)
        )
        enhanced['red_zone_matchup'] = (
            team_metrics.get('red_zone_td_pct', 0.0) -
            opp_metrics.get('red_zone_td_pct_defense', 0.0)
        )

        # Add positional defense matchup ratings
        position = player_features.get('position', 'UNK')
        defense_matchup_rating = self._get_defense_matchup_rating(opponent, position)
        enhanced['defense_matchup_factor'] = defense_matchup_rating.get('matchup_factor', 1.0)
        enhanced['defense_matchup_rank'] = defense_matchup_rating.get('league_rank', 16)
        enhanced['defense_yards_allowed_vs_pos'] = defense_matchup_rating.get('yards_allowed', 0.0)

        return enhanced

    def _get_defense_matchup_rating(self, opponent: str, position: str) -> Dict:
        """
        Get defense matchup rating for a specific position.

        Args:
            opponent: Opponent team
            position: Player position (QB, RB, WR, TE)

        Returns:
            Dict with matchup_factor, league_rank, yards_allowed
        """
        # Map player positions to defensive position categories
        position_map = {
            'QB': 'QB',
            'RB': 'RB_rush',  # Default to rush, can be adjusted for receiving backs
            'WR': 'WR1',  # Default to WR1, can be refined based on depth chart
            'TE': 'TE',
        }

        def_position = position_map.get(position, 'WR1')

        # Get positional stats from analyzer
        team_stats = self.defense_analyzer.positional_stats.get(opponent, {})
        pos_stats = team_stats.get(def_position)

        if not pos_stats:
            return {
                'matchup_factor': 1.0,
                'league_rank': 16,
                'yards_allowed': 0.0
            }

        return {
            'matchup_factor': pos_stats.get_matchup_factor(),
            'league_rank': pos_stats.league_rank,
            'yards_allowed': pos_stats.yards_per_game_allowed
        }

    def enrich_player_dataframe(
        self,
        player_df: pd.DataFrame,
        recency_weeks: int = 4
    ) -> pd.DataFrame:
        """
        Enrich an entire DataFrame of player features.

        Args:
            player_df: DataFrame with player features (must have 'team', 'opponent_team', 'week')
            recency_weeks: Number of recent weeks for team metrics

        Returns:
            Enhanced DataFrame with team metric columns added
        """
        if not self.calculator:
            return player_df

        enhanced_df = player_df.copy()

        # Initialize new columns
        team_metric_cols = [
            'team_success_rate', 'team_epa_per_play', 'team_red_zone_td_pct',
            'team_third_down_pct', 'team_explosive_play_rate', 'team_completion_pct',
            'team_yards_per_attempt', 'team_yards_per_carry', 'team_yards_after_catch_per_comp',
            'opp_def_success_rate', 'opp_def_epa_allowed', 'opp_def_explosive_allowed_rate',
            'opp_def_third_down_allowed', 'team_turnover_rate', 'opp_def_takeaway_rate',
            'team_plays_per_game', 'opp_plays_per_game', 'pass_efficiency_edge',
            'rush_efficiency_edge', 'red_zone_matchup', 'defense_matchup_factor',
            'defense_matchup_rank', 'defense_yards_allowed_vs_pos'
        ]

        for col in team_metric_cols:
            enhanced_df[col] = 0.0

        # Enrich each row
        for idx, row in enhanced_df.iterrows():
            if 'team' not in row.index or 'opponent_team' not in row.index or 'week' not in row.index:
                continue

            team = row['team']
            opponent = row['opponent_team']
            week = row['week']

            if pd.isna(team) or pd.isna(opponent) or pd.isna(week):
                continue

            # Get enriched features
            enriched = self.enrich_player_features(
                row.to_dict(),
                team,
                opponent,
                int(week),
                recency_weeks
            )

            # Update DataFrame
            for col in team_metric_cols:
                if col in enriched:
                    enhanced_df.at[idx, col] = enriched[col]

        return enhanced_df

    def get_feature_importance_context(self, position: str, prop_type: str) -> List[str]:
        """
        Get most relevant team metrics for a given position/prop type.

        Args:
            position: Player position (QB, RB, WR, TE)
            prop_type: Prop type (pass_yds, rush_yds, rec_yds, etc.)

        Returns:
            List of most relevant team metric feature names
        """
        # QB passing props
        if position == 'QB' and 'pass' in prop_type:
            return [
                'team_success_rate',
                'team_epa_per_play',
                'team_completion_pct',
                'team_yards_per_attempt',
                'opp_def_success_rate',
                'opp_def_epa_allowed',
                'pass_efficiency_edge',
                'team_plays_per_game',
                'opp_def_explosive_allowed_rate',
            ]

        # RB rushing props
        elif position == 'RB' and 'rush' in prop_type:
            return [
                'team_yards_per_carry',
                'team_success_rate',
                'rush_efficiency_edge',
                'team_plays_per_game',
                'opp_def_success_rate',
                'team_explosive_play_rate',
            ]

        # WR/TE receiving props
        elif position in ['WR', 'TE'] and 'rec' in prop_type:
            return [
                'team_completion_pct',
                'team_yards_per_attempt',
                'team_yards_after_catch_per_comp',
                'pass_efficiency_edge',
                'team_explosive_play_rate',
                'opp_def_explosive_allowed_rate',
                'team_plays_per_game',
            ]

        # TD props (any position)
        elif 'td' in prop_type:
            return [
                'team_red_zone_td_pct',
                'red_zone_matchup',
                'team_success_rate',
                'opp_def_success_rate',
            ]

        # Default: return all
        return [
            'team_success_rate',
            'team_epa_per_play',
            'pass_efficiency_edge',
            'rush_efficiency_edge',
            'team_plays_per_game',
        ]


# Utility function for easy integration
def add_team_metrics_to_features(
    player_features: pd.DataFrame,
    season: int = 2025,
    inputs_dir: str = "inputs",
    recency_weeks: int = 4
) -> pd.DataFrame:
    """
    Convenience function to add team metrics to player features DataFrame.

    Args:
        player_features: DataFrame with player features
        season: NFL season year
        inputs_dir: Directory with play-by-play data
        recency_weeks: Recent weeks for team metric calculation

    Returns:
        Enhanced DataFrame with team metrics
    """
    engine = TeamMetricsFeatureEngine(season=season, inputs_dir=inputs_dir)
    return engine.enrich_player_dataframe(player_features, recency_weeks=recency_weeks)
