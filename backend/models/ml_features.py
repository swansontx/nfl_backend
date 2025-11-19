"""
Feature matrix builder for ML models

Combines player stats, matchup context, and game features into
comprehensive feature matrices for XGBoost/LightGBM training.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass

from backend.config import settings
from backend.config.logging_config import get_logger
from backend.features import PlayerFeatures, MatchupFeatures
from backend.features.smoothing import SmoothedFeatures
from backend.database.session import get_db
from backend.database.models import Player, Game, PlayerGameFeature

logger = get_logger(__name__)


@dataclass
class MLFeatureSet:
    """Complete feature set for ML models"""
    features: pd.DataFrame  # Feature matrix
    feature_names: List[str]  # Feature column names
    targets: Optional[pd.Series] = None  # Target variable (for training)
    player_ids: Optional[List[str]] = None  # Player IDs for each row
    game_ids: Optional[List[str]] = None  # Game IDs for each row


class MLFeatureBuilder:
    """
    Builds comprehensive feature matrices for ML models

    Combines:
    - Player historical stats (smoothed)
    - Matchup context (opponent, weather, etc.)
    - Game environment (spread, total, home/away)
    - Position-specific features
    - Usage patterns (snap %, target share, etc.)
    """

    def __init__(self):
        """Initialize feature builder"""
        pass

    def build_td_features(
        self,
        player_id: str,
        game_id: str,
        smoothed_features: SmoothedFeatures,
        matchup_features: MatchupFeatures,
        player: Player
    ) -> Dict[str, float]:
        """
        Build feature dict for anytime TD prediction

        TD scoring is highly non-linear - factors that matter most:
        - Redzone usage (carries/targets inside 20)
        - Goal line role (GL carries)
        - Target/carry volume
        - Team implied total (game script)
        - Opponent defense strength
        - Position and role
        """
        features = {}

        # === Player Usage Features ===
        features['targets_per_game'] = smoothed_features.targets_per_game
        features['target_share'] = smoothed_features.target_share or 0
        features['rush_attempts_per_game'] = smoothed_features.rush_attempts_per_game
        features['rush_share'] = smoothed_features.rush_share or 0

        # Total touches
        features['total_touches_per_game'] = (
            smoothed_features.targets_per_game +
            smoothed_features.rush_attempts_per_game
        )

        # === TD-Specific Features ===
        features['receiving_td_rate'] = smoothed_features.receiving_td_rate
        features['rushing_td_rate'] = smoothed_features.rushing_td_rate

        # Expected TDs (baseline)
        features['expected_rec_tds'] = (
            smoothed_features.receiving_td_rate * smoothed_features.targets_per_game
        )
        features['expected_rush_tds'] = (
            smoothed_features.rushing_td_rate * smoothed_features.rush_attempts_per_game
        )
        features['expected_total_tds'] = (
            features['expected_rec_tds'] + features['expected_rush_tds']
        )

        # === Production Features ===
        features['receiving_yards_per_game'] = smoothed_features.receiving_yards_per_game
        features['rushing_yards_per_game'] = smoothed_features.rushing_yards_per_game
        features['yards_per_touch'] = (
            (smoothed_features.receiving_yards_per_game + smoothed_features.rushing_yards_per_game) /
            max(features['total_touches_per_game'], 1)
        )

        # === Matchup Features ===
        features['is_home'] = 1.0 if matchup_features.is_home else 0.0
        features['game_total'] = matchup_features.game_total or 45.0
        features['spread'] = matchup_features.spread or 0.0
        features['team_implied_total'] = matchup_features.team_implied_total or 22.5

        # Opponent defense
        features['opp_points_allowed_per_game'] = (
            matchup_features.opp_points_allowed_per_game or 22.0
        )
        features['opp_pass_def_rank'] = matchup_features.opp_pass_def_rank or 16
        features['opp_rush_def_rank'] = matchup_features.opp_rush_def_rank or 16

        # === Position Features ===
        # One-hot encode position
        for pos in ['QB', 'RB', 'WR', 'TE']:
            features[f'position_{pos}'] = 1.0 if player.position == pos else 0.0

        # === Confidence ===
        features['confidence'] = smoothed_features.confidence

        return features

    def build_receptions_features(
        self,
        player_id: str,
        game_id: str,
        smoothed_features: SmoothedFeatures,
        matchup_features: MatchupFeatures,
        player: Player
    ) -> Dict[str, float]:
        """
        Build features for receptions prediction

        Receptions are game-script dependent:
        - Target volume and share
        - Catch rate
        - Team pass rate
        - Game script (spread, total)
        - Opponent pass defense
        """
        features = {}

        # === Target Features ===
        features['targets_per_game'] = smoothed_features.targets_per_game
        features['receptions_per_game'] = smoothed_features.receptions_per_game
        features['target_share'] = smoothed_features.target_share or 0

        # Catch rate
        features['catch_rate'] = (
            smoothed_features.receptions_per_game /
            max(smoothed_features.targets_per_game, 1)
        )

        # === Production ===
        features['receiving_yards_per_game'] = smoothed_features.receiving_yards_per_game
        features['yards_per_target'] = (
            smoothed_features.receiving_yards_per_game /
            max(smoothed_features.targets_per_game, 1)
        )
        features['yards_per_reception'] = (
            smoothed_features.receiving_yards_per_game /
            max(smoothed_features.receptions_per_game, 1)
        )

        # === Game Script Features ===
        features['game_total'] = matchup_features.game_total or 45.0
        features['spread'] = matchup_features.spread or 0.0
        features['team_implied_total'] = matchup_features.team_implied_total or 22.5

        # Pass volume indicators
        features['team_pass_rate'] = matchup_features.team_pass_rate or 0.60
        features['team_pace'] = matchup_features.team_pace or 65.0

        # Expected pass volume
        features['expected_team_pass_plays'] = (
            features['team_pace'] * features['team_pass_rate']
        )

        # === Matchup ===
        features['is_home'] = 1.0 if matchup_features.is_home else 0.0
        features['opp_pass_def_rank'] = matchup_features.opp_pass_def_rank or 16

        # Weather (affects passing volume)
        features['wind_speed'] = matchup_features.wind_speed or 0.0
        features['is_dome'] = 1.0 if matchup_features.is_dome else 0.0
        features['has_precipitation'] = 1.0 if matchup_features.precipitation not in [None, 'none'] else 0.0

        # === Position ===
        for pos in ['WR', 'TE', 'RB']:
            features[f'position_{pos}'] = 1.0 if player.position == pos else 0.0

        features['confidence'] = smoothed_features.confidence

        return features

    def build_yards_features(
        self,
        player_id: str,
        game_id: str,
        smoothed_features: SmoothedFeatures,
        matchup_features: MatchupFeatures,
        player: Player,
        stat_type: str  # 'passing', 'rushing', or 'receiving'
    ) -> Dict[str, float]:
        """Build features for yards prediction (passing/rushing/receiving)"""
        features = {}

        # === Base Production ===
        if stat_type == 'passing':
            features['yards_per_game'] = smoothed_features.passing_yards_per_game
            features['volume'] = smoothed_features.passing_yards_per_game / 7.5  # Attempts
            features['yards_per_attempt'] = 7.5  # Simplified
            features['opp_def_rank'] = matchup_features.opp_pass_def_rank or 16
        elif stat_type == 'rushing':
            features['yards_per_game'] = smoothed_features.rushing_yards_per_game
            features['volume'] = smoothed_features.rush_attempts_per_game
            features['yards_per_attempt'] = (
                smoothed_features.rushing_yards_per_game /
                max(smoothed_features.rush_attempts_per_game, 1)
            )
            features['opp_def_rank'] = matchup_features.opp_rush_def_rank or 16
        else:  # receiving
            features['yards_per_game'] = smoothed_features.receiving_yards_per_game
            features['volume'] = smoothed_features.targets_per_game
            features['yards_per_attempt'] = (
                smoothed_features.receiving_yards_per_game /
                max(smoothed_features.targets_per_game, 1)
            )
            features['opp_def_rank'] = matchup_features.opp_pass_def_rank or 16

        # === Game Context ===
        features['game_total'] = matchup_features.game_total or 45.0
        features['spread'] = matchup_features.spread or 0.0
        features['team_implied_total'] = matchup_features.team_implied_total or 22.5
        features['is_home'] = 1.0 if matchup_features.is_home else 0.0

        # === Pace ===
        features['team_pace'] = matchup_features.team_pace or 65.0

        # === Weather (for passing/receiving) ===
        if stat_type in ['passing', 'receiving']:
            features['wind_speed'] = matchup_features.wind_speed or 0.0
            features['is_dome'] = 1.0 if matchup_features.is_dome else 0.0

        # === Position ===
        features['position_primary'] = 1.0  # Simplified

        features['confidence'] = smoothed_features.confidence

        return features

    def build_feature_matrix(
        self,
        feature_dicts: List[Dict[str, float]]
    ) -> pd.DataFrame:
        """Convert list of feature dicts to DataFrame"""
        if not feature_dicts:
            return pd.DataFrame()

        df = pd.DataFrame(feature_dicts)

        # Fill NaN with 0
        df = df.fillna(0)

        return df

    def build_training_set(
        self,
        market: str,
        season: int,
        min_samples: int = 100
    ) -> MLFeatureSet:
        """
        Build training set for a specific market

        Args:
            market: Market name (e.g., 'player_anytime_td')
            season: Season to build from
            min_samples: Minimum samples required

        Returns:
            MLFeatureSet with features and targets
        """
        logger.info("building_training_set", market=market, season=season)

        feature_dicts = []
        targets = []
        player_ids = []
        game_ids = []

        with get_db() as session:
            # Get all games for season
            games = (
                session.query(Game)
                .filter(Game.season == season)
                .all()
            )

            for game in games:
                # Get all players in game
                features_in_game = (
                    session.query(PlayerGameFeature)
                    .filter(PlayerGameFeature.game_id == game.game_id)
                    .all()
                )

                for pgf in features_in_game:
                    # Get player
                    player = session.query(Player).filter(
                        Player.player_id == pgf.player_id
                    ).first()

                    if not player:
                        continue

                    # Build features based on market
                    # Note: In production, this would use historical smoothed features
                    # For now, simplified

                    # Extract target (actual outcome)
                    if market == 'player_anytime_td':
                        target = 1.0 if (pgf.rushing_tds or 0) + (pgf.receiving_tds or 0) > 0 else 0.0
                    elif market == 'player_receptions':
                        target = float(pgf.receptions or 0)
                    elif market == 'player_rec_yds':
                        target = float(pgf.receiving_yards or 0)
                    else:
                        continue

                    # Skip if no meaningful data
                    if market == 'player_anytime_td' and (pgf.targets or 0) + (pgf.rush_attempts or 0) < 3:
                        continue

                    # Build feature dict (simplified for now)
                    feature_dict = {
                        'targets': pgf.targets or 0,
                        'rush_attempts': pgf.rush_attempts or 0,
                        'total_touches': (pgf.targets or 0) + (pgf.rush_attempts or 0),
                    }

                    feature_dicts.append(feature_dict)
                    targets.append(target)
                    player_ids.append(pgf.player_id)
                    game_ids.append(game.game_id)

        if len(feature_dicts) < min_samples:
            logger.warning(
                "insufficient_training_data",
                market=market,
                samples=len(feature_dicts),
                min_required=min_samples
            )
            return MLFeatureSet(
                features=pd.DataFrame(),
                feature_names=[],
                targets=None
            )

        # Build feature matrix
        features_df = self.build_feature_matrix(feature_dicts)
        targets_series = pd.Series(targets)

        logger.info(
            "training_set_built",
            market=market,
            samples=len(features_df),
            features=len(features_df.columns)
        )

        return MLFeatureSet(
            features=features_df,
            feature_names=list(features_df.columns),
            targets=targets_series,
            player_ids=player_ids,
            game_ids=game_ids
        )
