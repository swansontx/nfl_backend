"""
Outcome extraction from historical games

Extracts actual results for all markets to enable calibration
and model validation.
"""

from typing import Dict, Optional, List
from datetime import datetime

from backend.config import settings
from backend.config.logging_config import get_logger
from backend.database.session import get_db
from backend.database.models import (
    Game, Player, PlayerGameFeature, Outcome, Projection
)

logger = get_logger(__name__)


class OutcomeExtractor:
    """
    Extracts actual outcomes from completed games

    Maps player-game features to binary outcomes (over/under)
    or exact values for all markets.
    """

    def __init__(self):
        """Initialize outcome extractor"""
        pass

    def extract_game_outcomes(
        self,
        game_id: str,
        save_to_db: bool = True
    ) -> Dict[str, List[Outcome]]:
        """
        Extract all outcomes for a completed game

        Args:
            game_id: Game ID
            save_to_db: Whether to save to database

        Returns:
            Dict mapping market to list of Outcomes
        """
        outcomes_by_market: Dict[str, List[Outcome]] = {
            market: [] for market in settings.supported_markets
        }

        with get_db() as session:
            # Verify game exists and is complete
            game = session.query(Game).filter(Game.game_id == game_id).first()
            if not game:
                logger.warning("game_not_found", game_id=game_id)
                return outcomes_by_market

            # Get all player-game features
            features = (
                session.query(PlayerGameFeature)
                .filter(PlayerGameFeature.game_id == game_id)
                .all()
            )

            for feature in features:
                # Get player for position info
                player = session.query(Player).filter(
                    Player.player_id == feature.player_id
                ).first()

                if not player:
                    continue

                # Extract outcomes for all relevant markets
                market_outcomes = self._extract_player_outcomes(
                    feature, player, game
                )

                # Store by market
                for market, outcome in market_outcomes.items():
                    if outcome is not None:
                        outcomes_by_market[market].append(outcome)

                        # Save to database
                        if save_to_db:
                            session.add(outcome)

            if save_to_db:
                session.commit()

        # Log summary
        total_outcomes = sum(len(outcomes) for outcomes in outcomes_by_market.values())
        logger.info(
            "outcomes_extracted",
            game_id=game_id,
            total_outcomes=total_outcomes,
            markets=len([m for m in outcomes_by_market if outcomes_by_market[m]])
        )

        return outcomes_by_market

    def _extract_player_outcomes(
        self,
        feature: PlayerGameFeature,
        player: Player,
        game: Game
    ) -> Dict[str, Optional[Outcome]]:
        """
        Extract all market outcomes for a single player-game

        Args:
            feature: PlayerGameFeature record
            player: Player record
            game: Game record

        Returns:
            Dict mapping market to Outcome
        """
        outcomes = {}

        # === Passing Markets ===
        if player.position == 'QB' and feature.pass_attempts and feature.pass_attempts > 0:
            # Passing yards
            outcomes['player_pass_yds'] = self._create_outcome(
                feature.player_id, game.game_id, 'player_pass_yds',
                actual_value=feature.passing_yards or 0
            )

            # Passing TDs
            outcomes['player_pass_tds'] = self._create_outcome(
                feature.player_id, game.game_id, 'player_pass_tds',
                actual_value=feature.passing_tds or 0
            )

            # Completions
            outcomes['player_pass_completions'] = self._create_outcome(
                feature.player_id, game.game_id, 'player_pass_completions',
                actual_value=feature.completions or 0
            )

            # Attempts
            outcomes['player_pass_attempts'] = self._create_outcome(
                feature.player_id, game.game_id, 'player_pass_attempts',
                actual_value=feature.pass_attempts or 0
            )

            # Interceptions
            outcomes['player_interceptions'] = self._create_outcome(
                feature.player_id, game.game_id, 'player_interceptions',
                actual_value=feature.interceptions or 0
            )

            # 300+ yard milestone
            outcomes['player_300_pass_yds'] = self._create_outcome(
                feature.player_id, game.game_id, 'player_300_pass_yds',
                actual_value=1 if (feature.passing_yards or 0) >= 300 else 0
            )

        # === Rushing Markets ===
        if feature.rush_attempts and feature.rush_attempts > 0:
            # Rushing yards
            outcomes['player_rush_yds'] = self._create_outcome(
                feature.player_id, game.game_id, 'player_rush_yds',
                actual_value=feature.rushing_yards or 0
            )

            # Rush attempts
            outcomes['player_rush_attempts'] = self._create_outcome(
                feature.player_id, game.game_id, 'player_rush_attempts',
                actual_value=feature.rush_attempts or 0
            )

            # Rushing TDs
            outcomes['player_rush_tds'] = self._create_outcome(
                feature.player_id, game.game_id, 'player_rush_tds',
                actual_value=feature.rushing_tds or 0
            )

            # 100+ yard milestone
            outcomes['player_100_rush_yds'] = self._create_outcome(
                feature.player_id, game.game_id, 'player_100_rush_yds',
                actual_value=1 if (feature.rushing_yards or 0) >= 100 else 0
            )

        # === Receiving Markets ===
        if feature.targets and feature.targets > 0:
            # Receiving yards
            outcomes['player_rec_yds'] = self._create_outcome(
                feature.player_id, game.game_id, 'player_rec_yds',
                actual_value=feature.receiving_yards or 0
            )

            # Receptions
            outcomes['player_receptions'] = self._create_outcome(
                feature.player_id, game.game_id, 'player_receptions',
                actual_value=feature.receptions or 0
            )

            # Receiving TDs
            outcomes['player_rec_tds'] = self._create_outcome(
                feature.player_id, game.game_id, 'player_rec_tds',
                actual_value=feature.receiving_tds or 0
            )

            # 100+ yard milestone
            outcomes['player_100_rec_yds'] = self._create_outcome(
                feature.player_id, game.game_id, 'player_100_rec_yds',
                actual_value=1 if (feature.receiving_yards or 0) >= 100 else 0
            )

        # === TD Markets ===
        total_tds = (feature.rushing_tds or 0) + (feature.receiving_tds or 0) + (feature.passing_tds or 0)
        non_passing_tds = (feature.rushing_tds or 0) + (feature.receiving_tds or 0)

        if (feature.targets or 0) + (feature.rush_attempts or 0) > 0:
            # Anytime TD (rushing + receiving)
            outcomes['player_anytime_td'] = self._create_outcome(
                feature.player_id, game.game_id, 'player_anytime_td',
                actual_value=1 if non_passing_tds > 0 else 0
            )

            # Total TDs (all types)
            outcomes['player_total_tds'] = self._create_outcome(
                feature.player_id, game.game_id, 'player_total_tds',
                actual_value=total_tds
            )

            # 2+ TDs
            outcomes['player_2plus_tds'] = self._create_outcome(
                feature.player_id, game.game_id, 'player_2plus_tds',
                actual_value=1 if non_passing_tds >= 2 else 0
            )

        return outcomes

    def _create_outcome(
        self,
        player_id: str,
        game_id: str,
        market: str,
        actual_value: float
    ) -> Outcome:
        """
        Create Outcome record

        Args:
            player_id: Player ID
            game_id: Game ID
            market: Market name
            actual_value: Actual outcome value

        Returns:
            Outcome object
        """
        return Outcome(
            player_id=player_id,
            game_id=game_id,
            market=market,
            actual_value=actual_value,
            recorded_at=datetime.utcnow()
        )

    def extract_season_outcomes(
        self,
        season: int,
        save_to_db: bool = True
    ) -> int:
        """
        Extract outcomes for all games in a season

        Args:
            season: NFL season year
            save_to_db: Whether to save to database

        Returns:
            Total number of outcomes extracted
        """
        logger.info("extracting_season_outcomes", season=season)

        total_outcomes = 0

        with get_db() as session:
            # Get all games for season
            games = (
                session.query(Game)
                .filter(Game.season == season)
                .all()
            )

            for game in games:
                outcomes_by_market = self.extract_game_outcomes(
                    game.game_id,
                    save_to_db=save_to_db
                )

                game_total = sum(len(outcomes) for outcomes in outcomes_by_market.values())
                total_outcomes += game_total

        logger.info(
            "season_outcomes_extracted",
            season=season,
            games=len(games),
            total_outcomes=total_outcomes
        )

        return total_outcomes

    def link_projections_to_outcomes(
        self,
        game_id: str
    ) -> int:
        """
        Link existing projections to their actual outcomes

        Useful for calculating projection accuracy

        Args:
            game_id: Game ID

        Returns:
            Number of projections linked
        """
        linked = 0

        with get_db() as session:
            # Get all projections for game
            projections = (
                session.query(Projection)
                .filter(Projection.game_id == game_id)
                .all()
            )

            for proj in projections:
                # Find matching outcome
                outcome = (
                    session.query(Outcome)
                    .filter(
                        Outcome.player_id == proj.player_id,
                        Outcome.game_id == proj.game_id,
                        Outcome.market == proj.market
                    )
                    .first()
                )

                if outcome:
                    # Could store outcome_id in projection for easy lookup
                    # For now, just log
                    linked += 1

            session.commit()

        logger.info("projections_linked", game_id=game_id, linked=linked)

        return linked
