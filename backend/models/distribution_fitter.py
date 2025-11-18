"""
Distribution fitting from historical player data

Replaces hardcoded distribution parameters with player-specific fits:
- Completion rates from actual QB data
- INT rates from actual history
- Yards variance from real distributions
- TD rates by player and position

Includes robust player/team matching validation.
"""

from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import poisson, nbinom, lognorm

from backend.config import settings
from backend.config.logging_config import get_logger
from backend.database.session import get_db
from backend.database.models import Player, PlayerGameFeature, Game
from backend.canonical import TeamMapper

logger = get_logger(__name__)


@dataclass
class FittedDistribution:
    """Result of fitting a distribution to player data"""
    dist_type: str  # "poisson", "nbinom", "lognormal", "normal"
    params: Dict  # Distribution parameters
    sample_size: int  # Number of games used
    confidence: float  # Confidence in fit (0-1)

    # Summary stats
    mean: float
    std: float
    cv: Optional[float] = None  # Coefficient of variation


class DistributionFitter:
    """
    Fit statistical distributions from historical player data

    Key improvements over hardcoded parameters:
    1. Player-specific rates (completion %, INT %, TD rates)
    2. Player-specific variance (some players more consistent)
    3. Matchup-aware adjustments possible
    4. Validation of player/team matching
    """

    def __init__(self):
        self.team_mapper = TeamMapper()
        self._cache: Dict[Tuple[str, str], FittedDistribution] = {}

    def fit_passing_completions(
        self,
        player_id: str,
        n_games: int = 16,
        min_attempts_per_game: int = 10
    ) -> Optional[FittedDistribution]:
        """
        Fit completion distribution for a QB

        Returns player-specific completion rate and distribution,
        not hardcoded 65% average
        """
        cache_key = (player_id, f"completions_{n_games}")
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Get historical completions
        historical = self._get_historical_completions(player_id, n_games, min_attempts_per_game)

        if len(historical) < 3:
            logger.warning(
                "insufficient_completion_data",
                player_id=player_id,
                games=len(historical)
            )
            return None

        # Extract completions and attempts
        completions = np.array([g['completions'] for g in historical])
        attempts = np.array([g['attempts'] for g in historical])

        # Fit binomial/poisson
        # For completions given attempts, use Poisson as approximation
        mean_completions = completions.mean()
        var_completions = completions.var()

        # Check for overdispersion
        if var_completions > mean_completions * 1.5:
            # Use negative binomial for overdispersed data
            # Fit NB parameters
            p = mean_completions / var_completions
            r = (mean_completions ** 2) / (var_completions - mean_completions)

            params = {
                'dist_type': 'nbinom',
                'r': r,
                'p': p,
                'mean_attempts': attempts.mean()
            }
            dist_type = 'nbinom'
        else:
            # Use Poisson
            params = {
                'dist_type': 'poisson',
                'lambda': mean_completions,
                'mean_attempts': attempts.mean()
            }
            dist_type = 'poisson'

        # Calculate actual completion rate
        total_completions = completions.sum()
        total_attempts = attempts.sum()
        completion_rate = total_completions / total_attempts if total_attempts > 0 else 0.65

        params['completion_rate'] = completion_rate

        fitted = FittedDistribution(
            dist_type=dist_type,
            params=params,
            sample_size=len(historical),
            confidence=min(len(historical) / 16, 1.0),
            mean=mean_completions,
            std=np.sqrt(var_completions),
            cv=np.sqrt(var_completions) / mean_completions if mean_completions > 0 else None
        )

        self._cache[cache_key] = fitted

        logger.info(
            "completions_fitted",
            player_id=player_id,
            completion_rate=completion_rate,
            mean=mean_completions,
            games=len(historical)
        )

        return fitted

    def fit_passing_yards(
        self,
        player_id: str,
        n_games: int = 16
    ) -> Optional[FittedDistribution]:
        """
        Fit passing yards distribution (lognormal)

        Returns player-specific mean and variance,
        not assumed CV of 0.3
        """
        cache_key = (player_id, f"pass_yds_{n_games}")
        if cache_key in self._cache:
            return self._cache[cache_key]

        yards_data = self._get_historical_stat(
            player_id=player_id,
            stat='passing_yards',
            n_games=n_games,
            min_value=50  # Filter low-attempt games
        )

        if len(yards_data) < 3:
            return None

        # Fit lognormal
        yards_positive = np.array([y for y in yards_data if y > 0])

        if len(yards_positive) < 3:
            return None

        # Fit lognormal parameters
        shape, loc, scale = lognorm.fit(yards_positive, floc=0)

        # Calculate stats
        mean_yards = yards_positive.mean()
        std_yards = yards_positive.std()
        cv = std_yards / mean_yards if mean_yards > 0 else 0.3

        fitted = FittedDistribution(
            dist_type='lognormal',
            params={
                'dist_type': 'lognormal',
                'mu': np.log(scale),  # Log-space mean
                'sigma': shape,  # Log-space std
                'zero_prob': 0.0
            },
            sample_size=len(yards_positive),
            confidence=min(len(yards_positive) / 16, 1.0),
            mean=mean_yards,
            std=std_yards,
            cv=cv
        )

        self._cache[cache_key] = fitted

        logger.info(
            "pass_yards_fitted",
            player_id=player_id,
            mean=mean_yards,
            cv=cv,
            games=len(yards_positive)
        )

        return fitted

    def fit_interception_rate(
        self,
        player_id: str,
        n_games: int = 16
    ) -> Optional[float]:
        """
        Calculate player-specific INT rate

        Returns actual rate, not hardcoded 2%
        """
        historical = self._get_historical_completions(player_id, n_games, min_attempts_per_game=10)

        if len(historical) < 3:
            return None

        total_attempts = sum(g['attempts'] for g in historical)
        total_ints = sum(g.get('interceptions', 0) for g in historical)

        int_rate = total_ints / total_attempts if total_attempts > 0 else 0.02

        logger.info(
            "int_rate_calculated",
            player_id=player_id,
            int_rate=int_rate,
            total_ints=total_ints,
            total_attempts=total_attempts,
            games=len(historical)
        )

        return int_rate

    def fit_yards_per_attempt(
        self,
        player_id: str,
        n_games: int = 16
    ) -> Optional[float]:
        """
        Calculate player-specific YPA

        Returns actual YPA, not hardcoded 7.5
        """
        historical = self._get_historical_completions(player_id, n_games, min_attempts_per_game=10)

        if len(historical) < 3:
            return None

        total_yards = sum(g.get('yards', 0) for g in historical)
        total_attempts = sum(g['attempts'] for g in historical)

        ypa = total_yards / total_attempts if total_attempts > 0 else 7.5

        logger.info(
            "ypa_calculated",
            player_id=player_id,
            ypa=ypa,
            games=len(historical)
        )

        return ypa

    def fit_receiving_yards(
        self,
        player_id: str,
        n_games: int = 16
    ) -> Optional[FittedDistribution]:
        """Fit receiving yards distribution"""
        cache_key = (player_id, f"rec_yds_{n_games}")
        if cache_key in self._cache:
            return self._cache[cache_key]

        yards_data = self._get_historical_stat(
            player_id=player_id,
            stat='receiving_yards',
            n_games=n_games,
            min_value=0  # Include zero games for WRs
        )

        if len(yards_data) < 3:
            return None

        # Calculate zero probability
        zero_count = sum(1 for y in yards_data if y == 0)
        zero_prob = zero_count / len(yards_data)

        # Fit lognormal to non-zero values
        yards_positive = np.array([y for y in yards_data if y > 0])

        if len(yards_positive) < 2:
            return None

        shape, loc, scale = lognorm.fit(yards_positive, floc=0)

        mean_yards = np.mean(yards_data)  # Include zeros
        std_yards = np.std(yards_data)

        fitted = FittedDistribution(
            dist_type='lognormal',
            params={
                'dist_type': 'lognormal',
                'mu': np.log(scale),
                'sigma': shape,
                'zero_prob': zero_prob
            },
            sample_size=len(yards_data),
            confidence=min(len(yards_data) / 16, 1.0),
            mean=mean_yards,
            std=std_yards,
            cv=std_yards / mean_yards if mean_yards > 0 else 0.5
        )

        self._cache[cache_key] = fitted

        return fitted

    def fit_rushing_yards(
        self,
        player_id: str,
        n_games: int = 16
    ) -> Optional[FittedDistribution]:
        """Fit rushing yards distribution"""
        cache_key = (player_id, f"rush_yds_{n_games}")
        if cache_key in self._cache:
            return self._cache[cache_key]

        yards_data = self._get_historical_stat(
            player_id=player_id,
            stat='rushing_yards',
            n_games=n_games,
            min_value=0
        )

        if len(yards_data) < 3:
            return None

        zero_count = sum(1 for y in yards_data if y == 0)
        zero_prob = zero_count / len(yards_data)

        yards_positive = np.array([y for y in yards_data if y > 0])

        if len(yards_positive) < 2:
            return None

        shape, loc, scale = lognorm.fit(yards_positive, floc=0)

        mean_yards = np.mean(yards_data)
        std_yards = np.std(yards_data)

        fitted = FittedDistribution(
            dist_type='lognormal',
            params={
                'dist_type': 'lognormal',
                'mu': np.log(scale),
                'sigma': shape,
                'zero_prob': zero_prob
            },
            sample_size=len(yards_data),
            confidence=min(len(yards_data) / 16, 1.0),
            mean=mean_yards,
            std=std_yards,
            cv=std_yards / mean_yards if mean_yards > 0 else 0.6
        )

        self._cache[cache_key] = fitted

        return fitted

    def _get_historical_completions(
        self,
        player_id: str,
        n_games: int,
        min_attempts_per_game: int = 5
    ) -> List[Dict]:
        """
        Get historical completion data for a QB

        With careful player ID and team validation
        """
        with get_db() as session:
            # Validate player exists
            player = session.query(Player).filter(Player.player_id == player_id).first()
            if not player:
                logger.warning("player_not_found", player_id=player_id)
                return []

            # Validate position
            if player.position not in ['QB']:
                logger.warning(
                    "invalid_position_for_passing",
                    player_id=player_id,
                    position=player.position
                )
                return []

            # Get recent games
            features = (
                session.query(PlayerGameFeature, Game)
                .join(Game, PlayerGameFeature.game_id == Game.game_id)
                .filter(
                    PlayerGameFeature.player_id == player_id,
                    PlayerGameFeature.pass_attempts >= min_attempts_per_game
                )
                .order_by(Game.game_date.desc())
                .limit(n_games)
                .all()
            )

            result = []
            for feature, game in features:
                # Validate team match
                if not self._validate_player_team(player, feature, game):
                    continue

                result.append({
                    'completions': feature.completions or 0,
                    'attempts': feature.pass_attempts or 0,
                    'yards': feature.passing_yards or 0,
                    'tds': feature.passing_tds or 0,
                    'interceptions': feature.interceptions or 0,
                    'game_id': feature.game_id
                })

            return result

    def _get_historical_stat(
        self,
        player_id: str,
        stat: str,
        n_games: int,
        min_value: float = 0
    ) -> List[float]:
        """
        Get historical stat values with validation

        Args:
            player_id: Player ID
            stat: Stat name (e.g., 'receiving_yards')
            n_games: Number of games to fetch
            min_value: Minimum value to include (filters noise)

        Returns:
            List of stat values
        """
        with get_db() as session:
            player = session.query(Player).filter(Player.player_id == player_id).first()
            if not player:
                return []

            features = (
                session.query(PlayerGameFeature, Game)
                .join(Game, PlayerGameFeature.game_id == Game.game_id)
                .filter(PlayerGameFeature.player_id == player_id)
                .order_by(Game.game_date.desc())
                .limit(n_games)
                .all()
            )

            values = []
            for feature, game in features:
                if not self._validate_player_team(player, feature, game):
                    continue

                value = getattr(feature, stat, 0) or 0

                # Include if meets minimum or is exactly 0 (legit zero game)
                if value >= min_value or value == 0:
                    values.append(float(value))

            return values

    def _validate_player_team(
        self,
        player: Player,
        feature: PlayerGameFeature,
        game: Game
    ) -> bool:
        """
        Validate that player's team matches game teams

        Critical for accurate historical data
        """
        player_team = player.team

        # Normalize all team names
        player_team_norm = self.team_mapper.normalize_team(player_team, game.season)
        home_team_norm = self.team_mapper.normalize_team(game.home_team, game.season)
        away_team_norm = self.team_mapper.normalize_team(game.away_team, game.season)

        # Player must be on one of the teams
        if player_team_norm not in [home_team_norm, away_team_norm]:
            logger.warning(
                "team_mismatch",
                player_id=player.player_id,
                player_team=player_team,
                player_team_norm=player_team_norm,
                game_home=game.home_team,
                game_away=game.away_team,
                game_id=game.game_id
            )
            return False

        return True

    def clear_cache(self):
        """Clear fitted distribution cache"""
        self._cache.clear()
        logger.info("distribution_cache_cleared")
