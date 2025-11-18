"""Prop model runner - ties together features, smoothing, and modeling"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import numpy as np

from backend.config import settings
from backend.config.logging_config import get_logger
from backend.features import FeatureExtractor, FeatureSmoother, PlayerFeatures
from backend.features.smoothing import SmoothedFeatures
from backend.roster_injury import RosterInjuryService
from backend.database.session import get_db
from backend.database.models import Projection, Player, Game
from .distributions import (
    PoissonModel,
    LognormalModel,
    BernoulliModel,
    DistributionParams
)

logger = get_logger(__name__)


@dataclass
class PropProjection:
    """A single prop projection"""
    player_id: str
    player_name: str
    game_id: str
    market: str
    team: str
    position: str

    # Model outputs
    mu: float  # Expected value
    sigma: Optional[float]  # Standard deviation
    dist_params: Dict  # Full distribution parameters

    # Probabilities (for common thresholds)
    prob_over_line: Optional[float] = None  # If line is provided

    # Metadata
    confidence: float = 1.0  # From smoothing
    usage_norm: Optional[float] = None
    model_version: str = "v1"

    def to_dict(self) -> Dict:
        return asdict(self)


class PropModelRunner:
    """
    Main prop model runner

    Orchestrates:
    1. Feature extraction
    2. Feature smoothing
    3. Injury adjustments
    4. Distribution modeling
    5. Projection generation
    """

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.feature_smoother = FeatureSmoother()
        self.roster_service = RosterInjuryService(use_cache=True)

        # Distribution models
        self.poisson_model = PoissonModel()
        self.lognormal_model = LognormalModel()
        self.bernoulli_model = BernoulliModel()

    def generate_projections(
        self,
        game_id: str,
        markets: Optional[List[str]] = None,
        save_to_db: bool = True
    ) -> List[PropProjection]:
        """
        Generate projections for all players in a game

        Args:
            game_id: Game ID to project for
            markets: List of markets to project (None = all supported)
            save_to_db: Whether to save projections to database

        Returns:
            List of PropProjections
        """
        if markets is None:
            markets = settings.supported_markets

        logger.info("generating_projections", game_id=game_id, markets=markets)

        # Get game and players
        with get_db() as session:
            game = session.query(Game).filter(Game.game_id == game_id).first()
            if not game:
                raise ValueError(f"Game not found: {game_id}")

            # Get players for both teams
            players = (
                session.query(Player)
                .filter(Player.team.in_([game.home_team, game.away_team]))
                .all()
            )

        # Get roster status for all players
        player_statuses = self.roster_service.get_batch_status(
            [(p.player_id, game_id) for p in players]
        )

        projections = []

        for player in players:
            status = player_statuses.get((player.player_id, game_id))

            # Skip if player is not active
            if not status or not status.is_active:
                continue

            # Get historical features
            historical = self.feature_extractor.get_historical_features(
                player_id=player.player_id,
                before_game_id=game_id,
                n_games=settings.lookback_games
            )

            if not historical:
                logger.debug("no_historical_data", player_id=player.player_id)
                continue

            # Smooth features
            smoothed = self.feature_smoother.smooth_features(
                player_id=player.player_id,
                position=player.position,
                historical_features=historical,
                target_game_id=game_id
            )

            # Apply injury adjustment
            smoothed = self._apply_injury_adjustment(smoothed, status)

            # Generate projections for each market
            for market in markets:
                proj = self._generate_market_projection(
                    player=player,
                    game_id=game_id,
                    market=market,
                    smoothed_features=smoothed
                )

                if proj:
                    projections.append(proj)

        # Save to database
        if save_to_db:
            self._save_projections(projections)

        logger.info("projections_generated", game_id=game_id, count=len(projections))

        return projections

    def _generate_market_projection(
        self,
        player: Player,
        game_id: str,
        market: str,
        smoothed_features: SmoothedFeatures
    ) -> Optional[PropProjection]:
        """Generate projection for a specific market"""

        # Route to appropriate model based on market
        if market == 'player_receptions':
            return self._project_receptions(player, game_id, smoothed_features)
        elif market == 'player_rec_yds':
            return self._project_receiving_yards(player, game_id, smoothed_features)
        elif market == 'player_rush_yds':
            return self._project_rushing_yards(player, game_id, smoothed_features)
        elif market == 'player_rush_attempts':
            return self._project_rush_attempts(player, game_id, smoothed_features)
        elif market == 'player_anytime_td':
            return self._project_anytime_td(player, game_id, smoothed_features)
        elif market == 'player_pass_yds':
            return self._project_passing_yards(player, game_id, smoothed_features)
        elif market == 'player_pass_tds':
            return self._project_passing_tds(player, game_id, smoothed_features)
        else:
            logger.warning("unsupported_market", market=market)
            return None

    def _project_receptions(
        self,
        player: Player,
        game_id: str,
        features: SmoothedFeatures
    ) -> PropProjection:
        """Project receptions using Poisson"""
        # Expected receptions from smoothed features
        lambda_param = features.receptions_per_game

        # Create simple Poisson params
        params = {'dist_type': 'poisson', 'lambda': lambda_param}
        dist_params = self.poisson_model.predict(params)

        return PropProjection(
            player_id=player.player_id,
            player_name=player.display_name,
            game_id=game_id,
            market='player_receptions',
            team=player.team,
            position=player.position,
            mu=dist_params.mu,
            sigma=dist_params.sigma,
            dist_params=dist_params.params,
            confidence=features.confidence,
            usage_norm=features.target_share
        )

    def _project_receiving_yards(
        self,
        player: Player,
        game_id: str,
        features: SmoothedFeatures
    ) -> PropProjection:
        """Project receiving yards using lognormal"""
        mu_yards = features.receiving_yards_per_game

        # Estimate sigma from historical variance (simplified)
        # In production, fit actual lognormal from historical data
        sigma_estimate = np.log(1 + (0.5 ** 2))  # CV of 0.5

        params = {
            'dist_type': 'lognormal',
            'mu': np.log(max(mu_yards, 1)),  # Log of expected
            'sigma': sigma_estimate,
            'zero_prob': 0.1 if mu_yards < 10 else 0.0
        }

        dist_params = self.lognormal_model.predict(params)

        return PropProjection(
            player_id=player.player_id,
            player_name=player.display_name,
            game_id=game_id,
            market='player_rec_yds',
            team=player.team,
            position=player.position,
            mu=dist_params.mu,
            sigma=dist_params.sigma,
            dist_params=dist_params.params,
            confidence=features.confidence,
            usage_norm=features.target_share
        )

    def _project_rushing_yards(
        self,
        player: Player,
        game_id: str,
        features: SmoothedFeatures
    ) -> PropProjection:
        """Project rushing yards using lognormal"""
        mu_yards = features.rushing_yards_per_game

        sigma_estimate = np.log(1 + (0.6 ** 2))  # Higher variance for rushing

        params = {
            'dist_type': 'lognormal',
            'mu': np.log(max(mu_yards, 1)),
            'sigma': sigma_estimate,
            'zero_prob': 0.1 if mu_yards < 10 else 0.0
        }

        dist_params = self.lognormal_model.predict(params)

        return PropProjection(
            player_id=player.player_id,
            player_name=player.display_name,
            game_id=game_id,
            market='player_rush_yds',
            team=player.team,
            position=player.position,
            mu=dist_params.mu,
            sigma=dist_params.sigma,
            dist_params=dist_params.params,
            confidence=features.confidence,
            usage_norm=features.rush_share
        )

    def _project_rush_attempts(
        self,
        player: Player,
        game_id: str,
        features: SmoothedFeatures
    ) -> PropProjection:
        """Project rush attempts using Poisson"""
        lambda_param = features.rush_attempts_per_game

        params = {'dist_type': 'poisson', 'lambda': lambda_param}
        dist_params = self.poisson_model.predict(params)

        return PropProjection(
            player_id=player.player_id,
            player_name=player.display_name,
            game_id=game_id,
            market='player_rush_attempts',
            team=player.team,
            position=player.position,
            mu=dist_params.mu,
            sigma=dist_params.sigma,
            dist_params=dist_params.params,
            confidence=features.confidence,
            usage_norm=features.rush_share
        )

    def _project_anytime_td(
        self,
        player: Player,
        game_id: str,
        features: SmoothedFeatures
    ) -> PropProjection:
        """Project anytime TD probability"""
        # Combine receiving and rushing TD rates
        rec_td_contribution = features.receiving_td_rate * features.targets_per_game
        rush_td_contribution = features.rushing_td_rate * features.rush_attempts_per_game

        # Approximate probability of at least 1 TD
        # P(TD >= 1) ≈ 1 - P(TD = 0) = 1 - exp(-lambda)
        # where lambda = expected TDs
        expected_tds = rec_td_contribution + rush_td_contribution
        prob_anytime_td = 1 - np.exp(-expected_tds)

        dist_params = {
            'p': prob_anytime_td,
            'expected_tds': expected_tds
        }

        return PropProjection(
            player_id=player.player_id,
            player_name=player.display_name,
            game_id=game_id,
            market='player_anytime_td',
            team=player.team,
            position=player.position,
            mu=prob_anytime_td,
            sigma=np.sqrt(prob_anytime_td * (1 - prob_anytime_td)),
            dist_params=dist_params,
            confidence=features.confidence,
            usage_norm=(features.target_share or 0) + (features.rush_share or 0)
        )

    def _project_passing_yards(
        self,
        player: Player,
        game_id: str,
        features: SmoothedFeatures
    ) -> PropProjection:
        """Project passing yards"""
        mu_yards = features.passing_yards_per_game

        sigma_estimate = np.log(1 + (0.3 ** 2))  # Lower variance for passing

        params = {
            'dist_type': 'lognormal',
            'mu': np.log(max(mu_yards, 1)),
            'sigma': sigma_estimate,
            'zero_prob': 0.0
        }

        dist_params = self.lognormal_model.predict(params)

        return PropProjection(
            player_id=player.player_id,
            player_name=player.display_name,
            game_id=game_id,
            market='player_pass_yds',
            team=player.team,
            position=player.position,
            mu=dist_params.mu,
            sigma=dist_params.sigma,
            dist_params=dist_params.params,
            confidence=features.confidence,
            usage_norm=1.0 if player.position == 'QB' else 0.0
        )

    def _project_passing_tds(
        self,
        player: Player,
        game_id: str,
        features: SmoothedFeatures
    ) -> PropProjection:
        """Project passing TDs"""
        expected_tds = features.passing_td_rate * features.passing_yards_per_game / 100  # Rough approx

        params = {'dist_type': 'poisson', 'lambda': expected_tds}
        dist_params = self.poisson_model.predict(params)

        return PropProjection(
            player_id=player.player_id,
            player_name=player.display_name,
            game_id=game_id,
            market='player_pass_tds',
            team=player.team,
            position=player.position,
            mu=dist_params.mu,
            sigma=dist_params.sigma,
            dist_params=dist_params.params,
            confidence=features.confidence,
            usage_norm=1.0 if player.position == 'QB' else 0.0
        )

    def _apply_injury_adjustment(
        self,
        features: SmoothedFeatures,
        status
    ) -> SmoothedFeatures:
        """Apply injury confidence multiplier to features"""
        multiplier = status.confidence

        # Scale down usage and production features
        features.targets_per_game *= multiplier
        features.receptions_per_game *= multiplier
        features.rush_attempts_per_game *= multiplier
        features.receiving_yards_per_game *= multiplier
        features.rushing_yards_per_game *= multiplier
        features.passing_yards_per_game *= multiplier

        # Also reduce overall confidence
        features.confidence *= multiplier

        return features

    def _save_projections(self, projections: List[PropProjection]) -> None:
        """Save projections to database"""
        with get_db() as session:
            for proj in projections:
                # Check if exists
                existing = (
                    session.query(Projection)
                    .filter(
                        Projection.player_id == proj.player_id,
                        Projection.game_id == proj.game_id,
                        Projection.market == proj.market
                    )
                    .first()
                )

                if existing:
                    # Update
                    existing.mu = proj.mu
                    existing.sigma = proj.sigma
                    existing.dist_params = proj.dist_params
                    existing.usage_norm = proj.usage_norm
                    existing.model_version = proj.model_version
                else:
                    # Create
                    record = Projection(
                        player_id=proj.player_id,
                        game_id=proj.game_id,
                        market=proj.market,
                        mu=proj.mu,
                        sigma=proj.sigma,
                        dist_params=proj.dist_params,
                        usage_norm=proj.usage_norm,
                        model_version=proj.model_version
                    )
                    session.add(record)

        logger.info("projections_saved", count=len(projections))
