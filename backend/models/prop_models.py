"""Prop model runner - ties together features, smoothing, and modeling"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import numpy as np

from backend.config import settings
from backend.config.logging_config import get_logger
from backend.features import FeatureExtractor, FeatureSmoother, PlayerFeatures
from backend.features.smoothing import SmoothedFeatures
from backend.roster_injury import RosterInjuryService
from backend.redistribution import VolumeRedistributor
from backend.scoring import ProjectionScorer, TierAssigner
from backend.database.session import get_db
from backend.database.models import Projection, Player, Game
from .distributions import (
    PoissonModel,
    LognormalModel,
    BernoulliModel,
    DistributionParams
)
from .distribution_fitter import DistributionFitter

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

    # Scoring and ranking (set after initial generation)
    score: Optional[float] = None
    tier: Optional[str] = None  # core, mid, lotto

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
        self.redistributor = VolumeRedistributor()
        self.scorer = ProjectionScorer()
        self.tier_assigner = TierAssigner()

        # Distribution models
        self.poisson_model = PoissonModel()
        self.lognormal_model = LognormalModel()
        self.bernoulli_model = BernoulliModel()

        # Distribution fitter for player-specific parameters
        self.dist_fitter = DistributionFitter()

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

            # Generate projections even for inactive (for redistribution)
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

            # Apply injury adjustment (sets confidence multiplier)
            if status:
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

        logger.info("initial_projections_generated", count=len(projections))

        # Redistribute volume from inactive to active players
        projections, redistribution_results = self.redistributor.redistribute_game_projections(
            game_id=game_id,
            projections=projections,
            save_updates=False
        )

        # Log redistribution summary
        if redistribution_results:
            summary = self.redistributor.get_redistribution_summary(redistribution_results)
            logger.info("redistribution_summary", **summary)

        # Score and tier projections
        scored_projections = self.scorer.score_batch(projections)
        tier_assignment = self.tier_assigner.assign_tiers(scored_projections, strategy="hybrid")

        # Apply tiers and scores back to projections
        for sp in scored_projections:
            sp.projection.score = sp.score

        self.tier_assigner.apply_tiers_to_projections(tier_assignment, update_db=False)

        # Log tier summary
        tier_summary = self.tier_assigner.get_tier_summary(tier_assignment)
        logger.info("tier_summary", **tier_summary)

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

        # Passing markets
        if market == 'player_pass_yds':
            return self._project_passing_yards(player, game_id, smoothed_features)
        elif market == 'player_pass_tds':
            return self._project_passing_tds(player, game_id, smoothed_features)
        elif market == 'player_pass_completions':
            return self._project_pass_completions(player, game_id, smoothed_features)
        elif market == 'player_pass_attempts':
            return self._project_pass_attempts(player, game_id, smoothed_features)
        elif market == 'player_interceptions':
            return self._project_interceptions(player, game_id, smoothed_features)
        elif market == 'player_pass_longest':
            return self._project_pass_longest(player, game_id, smoothed_features)
        elif market == 'player_300_pass_yds':
            return self._project_milestone(player, game_id, smoothed_features, 'passing', 300)

        # Rushing markets
        elif market == 'player_rush_yds':
            return self._project_rushing_yards(player, game_id, smoothed_features)
        elif market == 'player_rush_attempts':
            return self._project_rush_attempts(player, game_id, smoothed_features)
        elif market == 'player_rush_tds':
            return self._project_rushing_tds(player, game_id, smoothed_features)
        elif market == 'player_rush_longest':
            return self._project_rush_longest(player, game_id, smoothed_features)
        elif market == 'player_100_rush_yds':
            return self._project_milestone(player, game_id, smoothed_features, 'rushing', 100)

        # Receiving markets
        elif market == 'player_rec_yds':
            return self._project_receiving_yards(player, game_id, smoothed_features)
        elif market == 'player_receptions':
            return self._project_receptions(player, game_id, smoothed_features)
        elif market == 'player_rec_tds':
            return self._project_receiving_tds(player, game_id, smoothed_features)
        elif market == 'player_rec_longest':
            return self._project_rec_longest(player, game_id, smoothed_features)
        elif market == 'player_100_rec_yds':
            return self._project_milestone(player, game_id, smoothed_features, 'receiving', 100)

        # TD markets
        elif market == 'player_anytime_td':
            return self._project_anytime_td(player, game_id, smoothed_features)
        elif market == 'player_first_td':
            return self._project_first_td(player, game_id, smoothed_features)
        elif market == 'player_last_td':
            return self._project_last_td(player, game_id, smoothed_features)
        elif market == 'player_total_tds':
            return self._project_total_tds(player, game_id, smoothed_features)
        elif market == 'player_2plus_tds':
            return self._project_2plus_tds(player, game_id, smoothed_features)

        # Combo markets
        elif market == 'player_pass_rush_yds':
            return self._project_pass_rush_yds(player, game_id, smoothed_features)
        elif market == 'player_rec_rush_yds':
            return self._project_rec_rush_yds(player, game_id, smoothed_features)

        # Kicking markets
        elif market == 'player_fg_made':
            return self._project_fg_made(player, game_id, smoothed_features)
        elif market == 'player_fg_longest':
            return self._project_fg_longest(player, game_id, smoothed_features)
        elif market == 'player_extra_points':
            return self._project_extra_points(player, game_id, smoothed_features)

        # Defense markets
        elif market == 'player_tackles_assists':
            return self._project_tackles_assists(player, game_id, smoothed_features)
        elif market == 'player_sacks':
            return self._project_sacks(player, game_id, smoothed_features)
        elif market == 'player_def_interceptions':
            return self._project_def_interceptions(player, game_id, smoothed_features)

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
        """Project receiving yards with fitted player-specific distribution"""
        mu_yards = features.receiving_yards_per_game

        # Try to fit player-specific distribution
        fitted = self.dist_fitter.fit_receiving_yards(
            player_id=player.player_id,
            n_games=settings.lookback_games
        )

        if fitted and fitted.confidence > 0.5:
            # Use fitted parameters
            params = fitted.params.copy()
            confidence = features.confidence * fitted.confidence
        else:
            # Fallback to assumed CV
            sigma_estimate = np.log(1 + (0.5 ** 2))
            params = {
                'dist_type': 'lognormal',
                'mu': np.log(max(mu_yards, 1)),
                'sigma': sigma_estimate,
                'zero_prob': 0.1 if mu_yards < 10 else 0.0
            }
            confidence = features.confidence * 0.7

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
            confidence=confidence,
            usage_norm=features.target_share
        )

    def _project_rushing_yards(
        self,
        player: Player,
        game_id: str,
        features: SmoothedFeatures
    ) -> PropProjection:
        """Project rushing yards with fitted player-specific distribution"""
        mu_yards = features.rushing_yards_per_game

        # Try to fit player-specific distribution
        fitted = self.dist_fitter.fit_rushing_yards(
            player_id=player.player_id,
            n_games=settings.lookback_games
        )

        if fitted and fitted.confidence > 0.5:
            # Use fitted parameters
            params = fitted.params.copy()
            confidence = features.confidence * fitted.confidence
        else:
            # Fallback to assumed CV
            sigma_estimate = np.log(1 + (0.6 ** 2))
            params = {
                'dist_type': 'lognormal',
                'mu': np.log(max(mu_yards, 1)),
                'sigma': sigma_estimate,
                'zero_prob': 0.1 if mu_yards < 10 else 0.0
            }
            confidence = features.confidence * 0.7

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
            confidence=confidence,
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
        """Project passing yards with fitted player-specific distribution"""
        mu_yards = features.passing_yards_per_game

        # Try to fit player-specific distribution
        fitted = self.dist_fitter.fit_passing_yards(
            player_id=player.player_id,
            n_games=settings.lookback_games
        )

        if fitted and fitted.confidence > 0.5:
            # Use fitted parameters
            params = fitted.params.copy()
            confidence = features.confidence * fitted.confidence
        else:
            # Fallback to assumed CV
            sigma_estimate = np.log(1 + (0.3 ** 2))
            params = {
                'dist_type': 'lognormal',
                'mu': np.log(max(mu_yards, 1)),
                'sigma': sigma_estimate,
                'zero_prob': 0.0
            }
            confidence = features.confidence * 0.7

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
            confidence=confidence,
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

    # Additional Passing Projections
    def _project_pass_completions(
        self,
        player: Player,
        game_id: str,
        features: SmoothedFeatures
    ) -> PropProjection:
        """Project pass completions using Poisson with fitted player-specific rate"""
        # Try to fit player-specific completion distribution
        fitted = self.dist_fitter.fit_passing_completions(
            player_id=player.player_id,
            n_games=settings.lookback_games
        )

        if fitted and fitted.confidence > 0.5:
            # Use fitted distribution
            lambda_completions = fitted.mean
            completion_rate = fitted.params.get('completion_rate', 0.65)
            confidence = features.confidence * fitted.confidence
        else:
            # Fallback to estimate from yards and average rate
            ypa = self.dist_fitter.fit_yards_per_attempt(player.player_id) or 7.5
            pass_attempts = features.passing_yards_per_game / ypa
            completion_rate = 0.65  # Fallback
            lambda_completions = pass_attempts * completion_rate
            confidence = features.confidence * 0.7  # Lower confidence

        params = {'dist_type': 'poisson', 'lambda': lambda_completions}
        dist_params = self.poisson_model.predict(params)

        return PropProjection(
            player_id=player.player_id,
            player_name=player.display_name,
            game_id=game_id,
            market='player_pass_completions',
            team=player.team,
            position=player.position,
            mu=dist_params.mu,
            sigma=dist_params.sigma,
            dist_params=dist_params.params,
            confidence=confidence,
            usage_norm=1.0 if player.position == 'QB' else 0.0
        )

    def _project_pass_attempts(
        self,
        player: Player,
        game_id: str,
        features: SmoothedFeatures
    ) -> PropProjection:
        """Project pass attempts using Poisson with fitted YPA"""
        # Use player-specific yards per attempt
        ypa = self.dist_fitter.fit_yards_per_attempt(player.player_id) or 7.5
        lambda_attempts = features.passing_yards_per_game / ypa

        params = {'dist_type': 'poisson', 'lambda': lambda_attempts}
        dist_params = self.poisson_model.predict(params)

        return PropProjection(
            player_id=player.player_id,
            player_name=player.display_name,
            game_id=game_id,
            market='player_pass_attempts',
            team=player.team,
            position=player.position,
            mu=dist_params.mu,
            sigma=dist_params.sigma,
            dist_params=dist_params.params,
            confidence=features.confidence,
            usage_norm=1.0 if player.position == 'QB' else 0.0
        )

    def _project_interceptions(
        self,
        player: Player,
        game_id: str,
        features: SmoothedFeatures
    ) -> PropProjection:
        """Project interceptions using Poisson with fitted player-specific INT rate"""
        # Use player-specific INT rate
        int_rate = self.dist_fitter.fit_interception_rate(player.player_id) or 0.02

        # Use player-specific YPA for attempts estimate
        ypa = self.dist_fitter.fit_yards_per_attempt(player.player_id) or 7.5
        pass_attempts = features.passing_yards_per_game / ypa

        lambda_ints = pass_attempts * int_rate

        params = {'dist_type': 'poisson', 'lambda': lambda_ints}
        dist_params = self.poisson_model.predict(params)

        return PropProjection(
            player_id=player.player_id,
            player_name=player.display_name,
            game_id=game_id,
            market='player_interceptions',
            team=player.team,
            position=player.position,
            mu=dist_params.mu,
            sigma=dist_params.sigma,
            dist_params=dist_params.params,
            confidence=features.confidence,
            usage_norm=1.0 if player.position == 'QB' else 0.0
        )

    def _project_pass_longest(
        self,
        player: Player,
        game_id: str,
        features: SmoothedFeatures
    ) -> PropProjection:
        """Project longest pass completion"""
        # Model longest as extreme value distribution
        # Simplified: use empirical rule based on pass yards
        avg_longest = 20 + (features.passing_yards_per_game / 15)  # Rough heuristic

        return PropProjection(
            player_id=player.player_id,
            player_name=player.display_name,
            game_id=game_id,
            market='player_pass_longest',
            team=player.team,
            position=player.position,
            mu=avg_longest,
            sigma=avg_longest * 0.3,  # 30% CV
            dist_params={'expected_longest': avg_longest},
            confidence=features.confidence * 0.7,  # Lower confidence for max stats
            usage_norm=1.0 if player.position == 'QB' else 0.0
        )

    # Rushing TD Projections
    def _project_rushing_tds(
        self,
        player: Player,
        game_id: str,
        features: SmoothedFeatures
    ) -> PropProjection:
        """Project rushing TDs using Poisson"""
        expected_tds = features.rushing_td_rate * features.rush_attempts_per_game

        params = {'dist_type': 'poisson', 'lambda': expected_tds}
        dist_params = self.poisson_model.predict(params)

        return PropProjection(
            player_id=player.player_id,
            player_name=player.display_name,
            game_id=game_id,
            market='player_rush_tds',
            team=player.team,
            position=player.position,
            mu=dist_params.mu,
            sigma=dist_params.sigma,
            dist_params=dist_params.params,
            confidence=features.confidence,
            usage_norm=features.rush_share
        )

    def _project_rush_longest(
        self,
        player: Player,
        game_id: str,
        features: SmoothedFeatures
    ) -> PropProjection:
        """Project longest rush"""
        avg_longest = 10 + (features.rushing_yards_per_game / 5)  # Rough heuristic

        return PropProjection(
            player_id=player.player_id,
            player_name=player.display_name,
            game_id=game_id,
            market='player_rush_longest',
            team=player.team,
            position=player.position,
            mu=avg_longest,
            sigma=avg_longest * 0.4,  # Higher variance for rushing
            dist_params={'expected_longest': avg_longest},
            confidence=features.confidence * 0.7,
            usage_norm=features.rush_share
        )

    # Receiving TD Projections
    def _project_receiving_tds(
        self,
        player: Player,
        game_id: str,
        features: SmoothedFeatures
    ) -> PropProjection:
        """Project receiving TDs using Poisson"""
        expected_tds = features.receiving_td_rate * features.targets_per_game

        params = {'dist_type': 'poisson', 'lambda': expected_tds}
        dist_params = self.poisson_model.predict(params)

        return PropProjection(
            player_id=player.player_id,
            player_name=player.display_name,
            game_id=game_id,
            market='player_rec_tds',
            team=player.team,
            position=player.position,
            mu=dist_params.mu,
            sigma=dist_params.sigma,
            dist_params=dist_params.params,
            confidence=features.confidence,
            usage_norm=features.target_share
        )

    def _project_rec_longest(
        self,
        player: Player,
        game_id: str,
        features: SmoothedFeatures
    ) -> PropProjection:
        """Project longest reception"""
        avg_longest = 15 + (features.receiving_yards_per_game / 6)

        return PropProjection(
            player_id=player.player_id,
            player_name=player.display_name,
            game_id=game_id,
            market='player_rec_longest',
            team=player.team,
            position=player.position,
            mu=avg_longest,
            sigma=avg_longest * 0.35,
            dist_params={'expected_longest': avg_longest},
            confidence=features.confidence * 0.7,
            usage_norm=features.target_share
        )

    # Milestone Projections (300 pass yds, 100 rush/rec yds)
    def _project_milestone(
        self,
        player: Player,
        game_id: str,
        features: SmoothedFeatures,
        stat_type: str,
        threshold: int
    ) -> PropProjection:
        """Project probability of hitting milestone yardage"""
        # Get base yards projection
        if stat_type == 'passing':
            mu_yards = features.passing_yards_per_game
            sigma_estimate = np.log(1 + (0.3 ** 2))
            market = f'player_{threshold}_pass_yds'
            usage = 1.0 if player.position == 'QB' else 0.0
        elif stat_type == 'rushing':
            mu_yards = features.rushing_yards_per_game
            sigma_estimate = np.log(1 + (0.6 ** 2))
            market = f'player_{threshold}_rush_yds'
            usage = features.rush_share
        elif stat_type == 'receiving':
            mu_yards = features.receiving_yards_per_game
            sigma_estimate = np.log(1 + (0.5 ** 2))
            market = f'player_{threshold}_rec_yds'
            usage = features.target_share
        else:
            raise ValueError(f"Unknown stat_type: {stat_type}")

        # Calculate P(X > threshold) using lognormal CDF
        from scipy.stats import lognorm
        mu_log = np.log(max(mu_yards, 1))
        prob_milestone = 1 - lognorm.cdf(threshold, s=sigma_estimate, scale=np.exp(mu_log))

        return PropProjection(
            player_id=player.player_id,
            player_name=player.display_name,
            game_id=game_id,
            market=market,
            team=player.team,
            position=player.position,
            mu=prob_milestone,
            sigma=np.sqrt(prob_milestone * (1 - prob_milestone)),
            dist_params={'threshold': threshold, 'prob': prob_milestone},
            confidence=features.confidence,
            usage_norm=usage
        )

    # TD-specific markets
    def _project_first_td(
        self,
        player: Player,
        game_id: str,
        features: SmoothedFeatures
    ) -> PropProjection:
        """Project first TD scorer probability"""
        # First TD is harder - need to account for game script
        # Approximate as anytime TD probability discounted
        rec_td_contribution = features.receiving_td_rate * features.targets_per_game
        rush_td_contribution = features.rushing_td_rate * features.rush_attempts_per_game
        expected_tds = rec_td_contribution + rush_td_contribution

        # Discount by ~70% for first TD (very rough heuristic)
        prob_first_td = (1 - np.exp(-expected_tds)) * 0.15

        return PropProjection(
            player_id=player.player_id,
            player_name=player.display_name,
            game_id=game_id,
            market='player_first_td',
            team=player.team,
            position=player.position,
            mu=prob_first_td,
            sigma=np.sqrt(prob_first_td * (1 - prob_first_td)),
            dist_params={'p': prob_first_td},
            confidence=features.confidence * 0.6,  # Lower confidence
            usage_norm=(features.target_share or 0) + (features.rush_share or 0)
        )

    def _project_last_td(
        self,
        player: Player,
        game_id: str,
        features: SmoothedFeatures
    ) -> PropProjection:
        """Project last TD scorer probability"""
        # Similar to first TD
        rec_td_contribution = features.receiving_td_rate * features.targets_per_game
        rush_td_contribution = features.rushing_td_rate * features.rush_attempts_per_game
        expected_tds = rec_td_contribution + rush_td_contribution

        prob_last_td = (1 - np.exp(-expected_tds)) * 0.15

        return PropProjection(
            player_id=player.player_id,
            player_name=player.display_name,
            game_id=game_id,
            market='player_last_td',
            team=player.team,
            position=player.position,
            mu=prob_last_td,
            sigma=np.sqrt(prob_last_td * (1 - prob_last_td)),
            dist_params={'p': prob_last_td},
            confidence=features.confidence * 0.6,
            usage_norm=(features.target_share or 0) + (features.rush_share or 0)
        )

    def _project_total_tds(
        self,
        player: Player,
        game_id: str,
        features: SmoothedFeatures
    ) -> PropProjection:
        """Project total TDs (passing + rushing + receiving)"""
        # Combine all TD sources
        expected_tds = (
            features.passing_td_rate * features.passing_yards_per_game / 100 +
            features.rushing_td_rate * features.rush_attempts_per_game +
            features.receiving_td_rate * features.targets_per_game
        )

        params = {'dist_type': 'poisson', 'lambda': expected_tds}
        dist_params = self.poisson_model.predict(params)

        return PropProjection(
            player_id=player.player_id,
            player_name=player.display_name,
            game_id=game_id,
            market='player_total_tds',
            team=player.team,
            position=player.position,
            mu=dist_params.mu,
            sigma=dist_params.sigma,
            dist_params=dist_params.params,
            confidence=features.confidence,
            usage_norm=1.0  # Everyone can score TDs
        )

    def _project_2plus_tds(
        self,
        player: Player,
        game_id: str,
        features: SmoothedFeatures
    ) -> PropProjection:
        """Project probability of 2+ TDs"""
        expected_tds = (
            features.rushing_td_rate * features.rush_attempts_per_game +
            features.receiving_td_rate * features.targets_per_game
        )

        # P(X >= 2) = 1 - P(X = 0) - P(X = 1)
        # For Poisson: P(X >= 2) = 1 - exp(-λ) - λ*exp(-λ)
        prob_2plus = 1 - np.exp(-expected_tds) * (1 + expected_tds)

        return PropProjection(
            player_id=player.player_id,
            player_name=player.display_name,
            game_id=game_id,
            market='player_2plus_tds',
            team=player.team,
            position=player.position,
            mu=prob_2plus,
            sigma=np.sqrt(prob_2plus * (1 - prob_2plus)),
            dist_params={'p': prob_2plus, 'lambda': expected_tds},
            confidence=features.confidence,
            usage_norm=(features.target_share or 0) + (features.rush_share or 0)
        )

    # Combo markets
    def _project_pass_rush_yds(
        self,
        player: Player,
        game_id: str,
        features: SmoothedFeatures
    ) -> PropProjection:
        """Project combined passing + rushing yards"""
        # Sum of two lognormals is approximately lognormal
        combined_mu = features.passing_yards_per_game + features.rushing_yards_per_game

        # Simplified: treat as single lognormal
        sigma_estimate = np.log(1 + (0.35 ** 2))

        params = {
            'dist_type': 'lognormal',
            'mu': np.log(max(combined_mu, 1)),
            'sigma': sigma_estimate,
            'zero_prob': 0.0
        }

        dist_params = self.lognormal_model.predict(params)

        return PropProjection(
            player_id=player.player_id,
            player_name=player.display_name,
            game_id=game_id,
            market='player_pass_rush_yds',
            team=player.team,
            position=player.position,
            mu=dist_params.mu,
            sigma=dist_params.sigma,
            dist_params=dist_params.params,
            confidence=features.confidence,
            usage_norm=1.0 if player.position == 'QB' else features.rush_share
        )

    def _project_rec_rush_yds(
        self,
        player: Player,
        game_id: str,
        features: SmoothedFeatures
    ) -> PropProjection:
        """Project combined receiving + rushing yards"""
        combined_mu = features.receiving_yards_per_game + features.rushing_yards_per_game

        sigma_estimate = np.log(1 + (0.45 ** 2))

        params = {
            'dist_type': 'lognormal',
            'mu': np.log(max(combined_mu, 1)),
            'sigma': sigma_estimate,
            'zero_prob': 0.05 if combined_mu < 20 else 0.0
        }

        dist_params = self.lognormal_model.predict(params)

        return PropProjection(
            player_id=player.player_id,
            player_name=player.display_name,
            game_id=game_id,
            market='player_rec_rush_yds',
            team=player.team,
            position=player.position,
            mu=dist_params.mu,
            sigma=dist_params.sigma,
            dist_params=dist_params.params,
            confidence=features.confidence,
            usage_norm=(features.target_share or 0) + (features.rush_share or 0)
        )

    # Kicking markets
    def _project_fg_made(
        self,
        player: Player,
        game_id: str,
        features: SmoothedFeatures
    ) -> PropProjection:
        """Project field goals made"""
        # Kickers average ~2 FG attempts per game
        lambda_fg = 1.8

        params = {'dist_type': 'poisson', 'lambda': lambda_fg}
        dist_params = self.poisson_model.predict(params)

        return PropProjection(
            player_id=player.player_id,
            player_name=player.display_name,
            game_id=game_id,
            market='player_fg_made',
            team=player.team,
            position=player.position,
            mu=dist_params.mu,
            sigma=dist_params.sigma,
            dist_params=dist_params.params,
            confidence=0.8,  # Medium confidence for kickers
            usage_norm=1.0 if player.position == 'K' else 0.0
        )

    def _project_fg_longest(
        self,
        player: Player,
        game_id: str,
        features: SmoothedFeatures
    ) -> PropProjection:
        """Project longest field goal"""
        # Most kickers average 40-45 yard longest FG
        avg_longest = 42.0

        return PropProjection(
            player_id=player.player_id,
            player_name=player.display_name,
            game_id=game_id,
            market='player_fg_longest',
            team=player.team,
            position=player.position,
            mu=avg_longest,
            sigma=avg_longest * 0.15,
            dist_params={'expected_longest': avg_longest},
            confidence=0.6,  # Lower confidence for max stats
            usage_norm=1.0 if player.position == 'K' else 0.0
        )

    def _project_extra_points(
        self,
        player: Player,
        game_id: str,
        features: SmoothedFeatures
    ) -> PropProjection:
        """Project extra points made"""
        # Average ~2.5 XPs per game
        lambda_xp = 2.5

        params = {'dist_type': 'poisson', 'lambda': lambda_xp}
        dist_params = self.poisson_model.predict(params)

        return PropProjection(
            player_id=player.player_id,
            player_name=player.display_name,
            game_id=game_id,
            market='player_extra_points',
            team=player.team,
            position=player.position,
            mu=dist_params.mu,
            sigma=dist_params.sigma,
            dist_params=dist_params.params,
            confidence=0.8,
            usage_norm=1.0 if player.position == 'K' else 0.0
        )

    # Defense markets
    def _project_tackles_assists(
        self,
        player: Player,
        game_id: str,
        features: SmoothedFeatures
    ) -> PropProjection:
        """Project tackles + assists"""
        # Defensive players average 5-8 combined tackles
        lambda_tackles = 6.5 if player.position in ['LB', 'SS', 'FS'] else 4.0

        params = {'dist_type': 'poisson', 'lambda': lambda_tackles}
        dist_params = self.poisson_model.predict(params)

        return PropProjection(
            player_id=player.player_id,
            player_name=player.display_name,
            game_id=game_id,
            market='player_tackles_assists',
            team=player.team,
            position=player.position,
            mu=dist_params.mu,
            sigma=dist_params.sigma,
            dist_params=dist_params.params,
            confidence=0.7,
            usage_norm=1.0 if player.position in ['LB', 'SS', 'FS', 'CB', 'DE', 'DT'] else 0.0
        )

    def _project_sacks(
        self,
        player: Player,
        game_id: str,
        features: SmoothedFeatures
    ) -> PropProjection:
        """Project sacks"""
        # Pass rushers average 0.5-1.0 sacks per game
        lambda_sacks = 0.7 if player.position in ['DE', 'LB'] else 0.3

        params = {'dist_type': 'poisson', 'lambda': lambda_sacks}
        dist_params = self.poisson_model.predict(params)

        return PropProjection(
            player_id=player.player_id,
            player_name=player.display_name,
            game_id=game_id,
            market='player_sacks',
            team=player.team,
            position=player.position,
            mu=dist_params.mu,
            sigma=dist_params.sigma,
            dist_params=dist_params.params,
            confidence=0.7,
            usage_norm=1.0 if player.position in ['DE', 'LB', 'DT'] else 0.0
        )

    def _project_def_interceptions(
        self,
        player: Player,
        game_id: str,
        features: SmoothedFeatures
    ) -> PropProjection:
        """Project defensive interceptions"""
        # DBs average ~0.15 INTs per game
        lambda_ints = 0.15 if player.position in ['CB', 'SS', 'FS'] else 0.05

        params = {'dist_type': 'poisson', 'lambda': lambda_ints}
        dist_params = self.poisson_model.predict(params)

        return PropProjection(
            player_id=player.player_id,
            player_name=player.display_name,
            game_id=game_id,
            market='player_def_interceptions',
            team=player.team,
            position=player.position,
            mu=dist_params.mu,
            sigma=dist_params.sigma,
            dist_params=dist_params.params,
            confidence=0.6,  # Lower confidence for rare events
            usage_norm=1.0 if player.position in ['CB', 'SS', 'FS'] else 0.0
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
