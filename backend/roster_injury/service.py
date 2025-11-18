"""Roster and injury status service with caching"""
from dataclasses import dataclass
from typing import Dict, Optional
from redis import Redis
import json

from backend.config import settings
from backend.config.logging_config import get_logger
from backend.database.session import get_db
from backend.database.models import RosterStatus, PlayerStatus, Player, Game

logger = get_logger(__name__)


@dataclass
class PlayerGameStatus:
    """Player status for a specific game"""
    player_id: str
    game_id: str
    status: PlayerStatus
    injury_type: Optional[str]
    confidence: float  # 0.0 to 1.0 multiplier
    is_active: bool
    depth_chart_position: Optional[int]


class RosterInjuryService:
    """
    Service for querying player roster and injury status

    Handles:
    - Player active/inactive status
    - Injury designations (Q, D, P, OUT, IR)
    - Confidence multipliers for questionable players
    - Redis caching for performance
    """

    def __init__(self, use_cache: bool = True):
        """
        Initialize roster/injury service

        Args:
            use_cache: Whether to use Redis caching
        """
        self.use_cache = use_cache

        if use_cache:
            try:
                self.redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("redis_connected")
            except Exception as e:
                logger.warning("redis_connection_failed", error=str(e))
                self.redis_client = None
                self.use_cache = False
        else:
            self.redis_client = None

        # Status to confidence multiplier mapping
        self.status_multipliers = {
            PlayerStatus.ACTIVE: 1.0,
            PlayerStatus.PROBABLE: settings.probable_multiplier,
            PlayerStatus.QUESTIONABLE: settings.questionable_multiplier,
            PlayerStatus.DOUBTFUL: settings.doubtful_multiplier,
            PlayerStatus.OUT: 0.0,
            PlayerStatus.INJURED_RESERVE: 0.0,
            PlayerStatus.CUT: 0.0,
            PlayerStatus.RETIRED: 0.0,
            PlayerStatus.PRACTICE_SQUAD: 0.0,  # DEV players excluded by default
        }

    def get_player_status(
        self,
        player_id: str,
        game_id: str,
        use_cache: bool = True
    ) -> PlayerGameStatus:
        """
        Get player status for a specific game

        Args:
            player_id: Canonical player ID
            game_id: Game ID
            use_cache: Whether to check cache first

        Returns:
            PlayerGameStatus with all relevant information
        """
        # Check cache first
        if use_cache and self.use_cache and self.redis_client:
            cached = self._get_from_cache(player_id, game_id)
            if cached:
                return cached

        # Query database
        with get_db() as session:
            roster_record = (
                session.query(RosterStatus)
                .filter(
                    RosterStatus.player_id == player_id,
                    RosterStatus.game_id == game_id
                )
                .first()
            )

            if not roster_record:
                # No record means active by default (or unknown)
                logger.warning("no_roster_record", player_id=player_id, game_id=game_id)
                status = PlayerGameStatus(
                    player_id=player_id,
                    game_id=game_id,
                    status=PlayerStatus.ACTIVE,
                    injury_type=None,
                    confidence=1.0,
                    is_active=True,
                    depth_chart_position=None
                )
            else:
                status = self._build_status(roster_record)

        # Cache the result
        if self.use_cache and self.redis_client:
            self._set_cache(status)

        return status

    def get_batch_status(
        self,
        player_game_pairs: list[tuple[str, str]]
    ) -> Dict[tuple[str, str], PlayerGameStatus]:
        """
        Get status for multiple player-game pairs

        Args:
            player_game_pairs: List of (player_id, game_id) tuples

        Returns:
            Dict mapping (player_id, game_id) to PlayerGameStatus
        """
        results = {}

        # Check cache first
        uncached_pairs = []
        for player_id, game_id in player_game_pairs:
            if self.use_cache and self.redis_client:
                cached = self._get_from_cache(player_id, game_id)
                if cached:
                    results[(player_id, game_id)] = cached
                    continue
            uncached_pairs.append((player_id, game_id))

        # Query database for uncached
        if uncached_pairs:
            with get_db() as session:
                roster_records = (
                    session.query(RosterStatus)
                    .filter(
                        RosterStatus.player_id.in_([p[0] for p in uncached_pairs]),
                        RosterStatus.game_id.in_([p[1] for p in uncached_pairs])
                    )
                    .all()
                )

                # Build lookup
                record_lookup = {
                    (r.player_id, r.game_id): r
                    for r in roster_records
                }

                # Process each pair
                for player_id, game_id in uncached_pairs:
                    record = record_lookup.get((player_id, game_id))

                    if record:
                        status = self._build_status(record)
                    else:
                        # Default to active
                        status = PlayerGameStatus(
                            player_id=player_id,
                            game_id=game_id,
                            status=PlayerStatus.ACTIVE,
                            injury_type=None,
                            confidence=1.0,
                            is_active=True,
                            depth_chart_position=None
                        )

                    results[(player_id, game_id)] = status

                    # Cache
                    if self.use_cache and self.redis_client:
                        self._set_cache(status)

        return results

    def get_active_players_for_game(
        self,
        game_id: str,
        min_confidence: float = 0.0
    ) -> list[PlayerGameStatus]:
        """
        Get all active players for a game

        Args:
            game_id: Game ID
            min_confidence: Minimum confidence threshold

        Returns:
            List of PlayerGameStatus for active players
        """
        with get_db() as session:
            roster_records = (
                session.query(RosterStatus)
                .filter(
                    RosterStatus.game_id == game_id,
                    RosterStatus.confidence >= min_confidence
                )
                .all()
            )

            results = []
            for record in roster_records:
                status = self._build_status(record)
                if status.is_active:
                    results.append(status)

        return results

    def get_inactive_players_for_game(
        self,
        game_id: str,
        team: Optional[str] = None
    ) -> list[PlayerGameStatus]:
        """
        Get all inactive players for a game

        Args:
            game_id: Game ID
            team: Optional team filter

        Returns:
            List of PlayerGameStatus for inactive players
        """
        with get_db() as session:
            query = (
                session.query(RosterStatus)
                .filter(RosterStatus.game_id == game_id)
            )

            if team:
                query = query.join(Player).filter(Player.team == team)

            roster_records = query.all()

            results = []
            for record in roster_records:
                status = self._build_status(record)
                if not status.is_active:
                    results.append(status)

        return results

    def update_player_status(
        self,
        player_id: str,
        game_id: str,
        status: PlayerStatus,
        injury_type: Optional[str] = None,
        depth_chart_position: Optional[int] = None
    ) -> None:
        """
        Update player status for a game

        Args:
            player_id: Player ID
            game_id: Game ID
            status: New PlayerStatus
            injury_type: Injury description
            depth_chart_position: Position on depth chart
        """
        confidence = self.status_multipliers.get(status, 1.0)

        with get_db() as session:
            # Check if record exists
            record = (
                session.query(RosterStatus)
                .filter(
                    RosterStatus.player_id == player_id,
                    RosterStatus.game_id == game_id
                )
                .first()
            )

            if record:
                # Update existing
                record.status = status
                record.injury_type = injury_type
                record.confidence = confidence
                record.depth_chart_position = depth_chart_position
            else:
                # Create new
                record = RosterStatus(
                    player_id=player_id,
                    game_id=game_id,
                    status=status,
                    injury_type=injury_type,
                    confidence=confidence,
                    depth_chart_position=depth_chart_position
                )
                session.add(record)

        # Invalidate cache
        if self.use_cache and self.redis_client:
            cache_key = f"roster:{player_id}:{game_id}"
            self.redis_client.delete(cache_key)

        logger.info(
            "status_updated",
            player_id=player_id,
            game_id=game_id,
            status=status.value,
            confidence=confidence
        )

    def _build_status(self, record: RosterStatus) -> PlayerGameStatus:
        """Build PlayerGameStatus from database record"""
        is_active = record.status in [
            PlayerStatus.ACTIVE,
            PlayerStatus.PROBABLE,
            PlayerStatus.QUESTIONABLE,
            PlayerStatus.DOUBTFUL,
        ]

        return PlayerGameStatus(
            player_id=record.player_id,
            game_id=record.game_id,
            status=record.status,
            injury_type=record.injury_type,
            confidence=record.confidence,
            is_active=is_active,
            depth_chart_position=record.depth_chart_position
        )

    def _get_from_cache(self, player_id: str, game_id: str) -> Optional[PlayerGameStatus]:
        """Get status from Redis cache"""
        if not self.redis_client:
            return None

        cache_key = f"roster:{player_id}:{game_id}"
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                return PlayerGameStatus(
                    player_id=data["player_id"],
                    game_id=data["game_id"],
                    status=PlayerStatus(data["status"]),
                    injury_type=data.get("injury_type"),
                    confidence=data["confidence"],
                    is_active=data["is_active"],
                    depth_chart_position=data.get("depth_chart_position")
                )
        except Exception as e:
            logger.warning("cache_read_error", error=str(e), key=cache_key)

        return None

    def _set_cache(self, status: PlayerGameStatus, ttl: int = 3600) -> None:
        """Set status in Redis cache"""
        if not self.redis_client:
            return

        cache_key = f"roster:{status.player_id}:{status.game_id}"
        data = {
            "player_id": status.player_id,
            "game_id": status.game_id,
            "status": status.status.value,
            "injury_type": status.injury_type,
            "confidence": status.confidence,
            "is_active": status.is_active,
            "depth_chart_position": status.depth_chart_position
        }

        try:
            self.redis_client.setex(cache_key, ttl, json.dumps(data))
        except Exception as e:
            logger.warning("cache_write_error", error=str(e), key=cache_key)

    def clear_cache_for_game(self, game_id: str) -> None:
        """Clear all cached statuses for a game"""
        if not self.redis_client:
            return

        # Get all roster cache keys for this game
        pattern = f"roster:*:{game_id}"
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
                logger.info("cache_cleared", game_id=game_id, keys_deleted=len(keys))
        except Exception as e:
            logger.warning("cache_clear_error", error=str(e), game_id=game_id)
