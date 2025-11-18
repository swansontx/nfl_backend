"""Feature extraction from play-by-play data"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from backend.config import settings
from backend.config.logging_config import get_logger
from backend.database.session import get_db
from backend.database.models import PlayerGameFeature, Player, Game

logger = get_logger(__name__)


@dataclass
class PlayerFeatures:
    """Extracted features for a player-game"""
    player_id: str
    game_id: str

    # Participation
    snaps: int = 0
    snap_share: float = 0.0
    routes_run: int = 0

    # Usage
    targets: int = 0
    receptions: int = 0
    rush_attempts: int = 0
    pass_attempts: int = 0

    # Production
    receiving_yards: float = 0.0
    rushing_yards: float = 0.0
    passing_yards: float = 0.0
    receiving_tds: int = 0
    rushing_tds: int = 0
    passing_tds: int = 0

    # Advanced
    redzone_targets: int = 0
    redzone_carries: int = 0
    goalline_carries: int = 0
    air_yards_share: Optional[float] = None
    target_share: Optional[float] = None
    rush_share: Optional[float] = None

    # Rates (per opportunity)
    yards_per_target: Optional[float] = None
    yards_per_route: Optional[float] = None
    yards_per_carry: Optional[float] = None
    yards_per_snap: Optional[float] = None
    td_per_target: Optional[float] = None
    td_per_carry: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class FeatureExtractor:
    """
    Extract player features from play-by-play data

    Handles:
    - Participation (snaps, routes)
    - Usage (targets, carries, passes)
    - Production (yards, TDs)
    - Advanced (redzone, shares, rates)
    """

    def __init__(self):
        """Initialize feature extractor"""
        pass

    def extract_from_pbp(
        self,
        pbp_df: pd.DataFrame,
        game_id: str,
        save_to_db: bool = True
    ) -> Dict[str, PlayerFeatures]:
        """
        Extract features from play-by-play DataFrame

        Args:
            pbp_df: Play-by-play data (nflverse format expected)
            game_id: Game ID
            save_to_db: Whether to save to database

        Returns:
            Dict mapping player_id to PlayerFeatures
        """
        logger.info("extracting_features", game_id=game_id, plays=len(pbp_df))

        # Initialize storage
        player_features: Dict[str, PlayerFeatures] = {}

        # Get team snap counts
        team_snaps = self._compute_team_snaps(pbp_df)

        # Extract for each player type
        self._extract_passing_features(pbp_df, game_id, player_features)
        self._extract_rushing_features(pbp_df, game_id, player_features)
        self._extract_receiving_features(pbp_df, game_id, player_features)

        # Compute shares and rates
        self._compute_shares_and_rates(player_features, pbp_df, team_snaps)

        # Save to database
        if save_to_db:
            self._save_to_db(player_features, game_id)

        logger.info("features_extracted", game_id=game_id, players=len(player_features))

        return player_features

    def _extract_passing_features(
        self,
        pbp_df: pd.DataFrame,
        game_id: str,
        player_features: Dict[str, PlayerFeatures]
    ) -> None:
        """Extract passing features"""
        passing_plays = pbp_df[pbp_df['play_type'].isin(['pass', 'qb_kneel', 'qb_spike'])]

        for passer_id in passing_plays['passer_player_id'].dropna().unique():
            if passer_id not in player_features:
                player_features[passer_id] = PlayerFeatures(
                    player_id=passer_id,
                    game_id=game_id
                )

            pf = player_features[passer_id]
            player_plays = passing_plays[passing_plays['passer_player_id'] == passer_id]

            # Count attempts (all pass plays)
            pf.pass_attempts = len(player_plays)

            # Passing yards (completions only)
            pf.passing_yards = player_plays['passing_yards'].fillna(0).sum()

            # Passing TDs
            pf.passing_tds = player_plays['pass_touchdown'].sum()

    def _extract_rushing_features(
        self,
        pbp_df: pd.DataFrame,
        game_id: str,
        player_features: Dict[str, PlayerFeatures]
    ) -> None:
        """Extract rushing features"""
        rushing_plays = pbp_df[pbp_df['play_type'] == 'run']

        for rusher_id in rushing_plays['rusher_player_id'].dropna().unique():
            if rusher_id not in player_features:
                player_features[rusher_id] = PlayerFeatures(
                    player_id=rusher_id,
                    game_id=game_id
                )

            pf = player_features[rusher_id]
            player_plays = rushing_plays[rushing_plays['rusher_player_id'] == rusher_id]

            # Count attempts
            pf.rush_attempts = len(player_plays)

            # Rushing yards
            pf.rushing_yards = player_plays['rushing_yards'].fillna(0).sum()

            # Rushing TDs
            pf.rushing_tds = player_plays['rush_touchdown'].sum()

            # Redzone and goalline carries
            redzone_plays = player_plays[player_plays['yardline_100'] <= 20]
            pf.redzone_carries = len(redzone_plays)

            goalline_plays = player_plays[player_plays['yardline_100'] <= 5]
            pf.goalline_carries = len(goalline_plays)

    def _extract_receiving_features(
        self,
        pbp_df: pd.DataFrame,
        game_id: str,
        player_features: Dict[str, PlayerFeatures]
    ) -> None:
        """Extract receiving features"""
        passing_plays = pbp_df[pbp_df['play_type'] == 'pass']

        for receiver_id in passing_plays['receiver_player_id'].dropna().unique():
            if receiver_id not in player_features:
                player_features[receiver_id] = PlayerFeatures(
                    player_id=receiver_id,
                    game_id=game_id
                )

            pf = player_features[receiver_id]
            player_plays = passing_plays[passing_plays['receiver_player_id'] == receiver_id]

            # Targets
            pf.targets = len(player_plays)

            # Receptions (complete passes only)
            completions = player_plays[player_plays['complete_pass'] == 1]
            pf.receptions = len(completions)

            # Receiving yards
            pf.receiving_yards = player_plays['receiving_yards'].fillna(0).sum()

            # Receiving TDs
            pf.receiving_tds = player_plays['pass_touchdown'].sum()

            # Redzone targets
            redzone_plays = player_plays[player_plays['yardline_100'] <= 20]
            pf.redzone_targets = len(redzone_plays)

    def _compute_shares_and_rates(
        self,
        player_features: Dict[str, PlayerFeatures],
        pbp_df: pd.DataFrame,
        team_snaps: Dict[str, int]
    ) -> None:
        """Compute shares and rate stats"""
        # Team totals
        team_targets = {}
        team_carries = {}
        team_air_yards = {}

        passing_plays = pbp_df[pbp_df['play_type'] == 'pass']
        rushing_plays = pbp_df[pbp_df['play_type'] == 'run']

        for team in pbp_df['posteam'].dropna().unique():
            team_passing = passing_plays[passing_plays['posteam'] == team]
            team_rushing = rushing_plays[rushing_plays['posteam'] == team]

            team_targets[team] = len(team_passing)
            team_carries[team] = len(team_rushing)
            team_air_yards[team] = team_passing['air_yards'].fillna(0).sum()

        # Compute shares for each player
        for player_id, pf in player_features.items():
            # Get player's team
            player_team = self._get_player_team(player_id, pbp_df)

            if not player_team:
                continue

            # Target share
            if player_team in team_targets and team_targets[player_team] > 0:
                pf.target_share = pf.targets / team_targets[player_team]

            # Rush share
            if player_team in team_carries and team_carries[player_team] > 0:
                pf.rush_share = pf.rush_attempts / team_carries[player_team]

            # Air yards share
            if player_team in team_air_yards and team_air_yards[player_team] > 0:
                player_air_yards = passing_plays[
                    passing_plays['receiver_player_id'] == player_id
                ]['air_yards'].fillna(0).sum()
                pf.air_yards_share = player_air_yards / team_air_yards[player_team]

            # Snap share (if snap data available)
            if player_team in team_snaps and team_snaps[player_team] > 0:
                # This would come from external snap count data
                # For now, estimate from involvement
                pf.snap_share = self._estimate_snap_share(pf)

            # Rate stats
            if pf.targets > 0:
                pf.yards_per_target = pf.receiving_yards / pf.targets
                pf.td_per_target = pf.receiving_tds / pf.targets

            if pf.rush_attempts > 0:
                pf.yards_per_carry = pf.rushing_yards / pf.rush_attempts
                pf.td_per_carry = pf.rushing_tds / pf.rush_attempts

            # Estimate routes run (roughly targets + incompletions nearby)
            if pf.targets > 0:
                pf.routes_run = int(pf.targets * 1.5)  # Rough estimate

            if pf.routes_run > 0:
                pf.yards_per_route = pf.receiving_yards / pf.routes_run

    def _compute_team_snaps(self, pbp_df: pd.DataFrame) -> Dict[str, int]:
        """Compute total team snaps"""
        team_snaps = {}

        for team in pbp_df['posteam'].dropna().unique():
            team_plays = pbp_df[pbp_df['posteam'] == team]
            # Count offensive plays
            team_snaps[team] = len(team_plays[
                team_plays['play_type'].isin(['pass', 'run', 'qb_kneel', 'qb_spike'])
            ])

        return team_snaps

    def _get_player_team(self, player_id: str, pbp_df: pd.DataFrame) -> Optional[str]:
        """Get player's team from PBP data"""
        # Try receiver
        receiver_plays = pbp_df[pbp_df['receiver_player_id'] == player_id]
        if len(receiver_plays) > 0:
            return receiver_plays.iloc[0]['posteam']

        # Try rusher
        rusher_plays = pbp_df[pbp_df['rusher_player_id'] == player_id]
        if len(rusher_plays) > 0:
            return rusher_plays.iloc[0]['posteam']

        # Try passer
        passer_plays = pbp_df[pbp_df['passer_player_id'] == player_id]
        if len(passer_plays) > 0:
            return passer_plays.iloc[0]['posteam']

        return None

    def _estimate_snap_share(self, pf: PlayerFeatures) -> float:
        """Estimate snap share from involvement"""
        # Simple heuristic: high involvement = high snap share
        total_touches = pf.targets + pf.rush_attempts + pf.pass_attempts

        if total_touches >= 20:
            return 0.9
        elif total_touches >= 10:
            return 0.7
        elif total_touches >= 5:
            return 0.5
        elif total_touches >= 1:
            return 0.3
        else:
            return 0.1

    def _save_to_db(self, player_features: Dict[str, PlayerFeatures], game_id: str) -> None:
        """Save features to database"""
        with get_db() as session:
            for player_id, pf in player_features.items():
                # Check if record exists
                existing = (
                    session.query(PlayerGameFeature)
                    .filter(
                        PlayerGameFeature.player_id == player_id,
                        PlayerGameFeature.game_id == game_id
                    )
                    .first()
                )

                if existing:
                    # Update existing
                    for key, value in pf.to_dict().items():
                        if key not in ['player_id', 'game_id']:
                            setattr(existing, key, value)
                else:
                    # Create new
                    record = PlayerGameFeature(
                        player_id=player_id,
                        game_id=game_id,
                        **{k: v for k, v in pf.to_dict().items()
                           if k not in ['player_id', 'game_id']}
                    )
                    session.add(record)

        logger.info("features_saved", game_id=game_id, count=len(player_features))

    def get_historical_features(
        self,
        player_id: str,
        before_game_id: str,
        n_games: int = 8
    ) -> List[PlayerFeatures]:
        """
        Get historical features for a player

        Args:
            player_id: Player ID
            before_game_id: Get games before this one
            n_games: Number of games to retrieve

        Returns:
            List of PlayerFeatures for past N games
        """
        with get_db() as session:
            # Get game date
            game = session.query(Game).filter(Game.game_id == before_game_id).first()
            if not game:
                return []

            # Get historical games
            historical_games = (
                session.query(Game)
                .filter(
                    Game.season <= game.season,
                    Game.game_date < game.game_date
                )
                .order_by(Game.game_date.desc())
                .limit(n_games)
                .all()
            )

            game_ids = [g.game_id for g in historical_games]

            # Get features
            feature_records = (
                session.query(PlayerGameFeature)
                .filter(
                    PlayerGameFeature.player_id == player_id,
                    PlayerGameFeature.game_id.in_(game_ids)
                )
                .all()
            )

            # Convert to PlayerFeatures objects
            features = []
            for record in feature_records:
                pf = PlayerFeatures(
                    player_id=record.player_id,
                    game_id=record.game_id,
                    snaps=record.snaps or 0,
                    snap_share=record.snap_share or 0.0,
                    routes_run=record.routes_run or 0,
                    targets=record.targets or 0,
                    receptions=record.receptions or 0,
                    rush_attempts=record.rush_attempts or 0,
                    pass_attempts=record.pass_attempts or 0,
                    receiving_yards=record.receiving_yards or 0.0,
                    rushing_yards=record.rushing_yards or 0.0,
                    passing_yards=record.passing_yards or 0.0,
                    receiving_tds=record.receiving_tds or 0,
                    rushing_tds=record.rushing_tds or 0,
                    passing_tds=record.passing_tds or 0,
                    redzone_targets=record.redzone_targets or 0,
                    redzone_carries=record.redzone_carries or 0,
                    goalline_carries=record.goalline_carries or 0,
                    air_yards_share=record.air_yards_share,
                    target_share=record.target_share,
                    rush_share=record.rush_share,
                )
                features.append(pf)

            return features
