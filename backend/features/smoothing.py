"""Feature smoothing with Empirical Bayes and rolling averages"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from scipy import stats

from backend.config import settings
from backend.config.logging_config import get_logger
from .extractor import PlayerFeatures

logger = get_logger(__name__)


@dataclass
class SmoothedFeatures:
    """Smoothed features for modeling"""
    player_id: str
    game_id: str

    # Smoothed rates
    targets_per_game: float = 0.0
    receptions_per_game: float = 0.0
    rush_attempts_per_game: float = 0.0
    receiving_yards_per_game: float = 0.0
    rushing_yards_per_game: float = 0.0
    passing_yards_per_game: float = 0.0

    # Smoothed TD rates (Empirical Bayes)
    receiving_td_rate: float = 0.0
    rushing_td_rate: float = 0.0
    passing_td_rate: float = 0.0

    # Smoothed shares
    target_share: float = 0.0
    rush_share: float = 0.0
    snap_share: float = 0.0
    redzone_share: float = 0.0

    # Efficiency metrics
    yards_per_target: float = 0.0
    yards_per_carry: float = 0.0
    yards_per_route: float = 0.0

    # Confidence (based on sample size)
    confidence: float = 0.0


class FeatureSmoother:
    """
    Smooth features using Empirical Bayes and rolling averages

    Methods:
    - Empirical Bayes: For low-sample rates (TDs, rare events)
    - Rolling average with exponential decay: For counts and yards
    - Position-aware priors: Different priors for QB, RB, WR, TE
    """

    def __init__(self):
        """Initialize smoother with position-specific priors"""
        # Position-specific TD rate priors (mean, std)
        self.td_priors = {
            'QB': {'passing_td_rate': (0.04, 0.02)},  # 4% per attempt
            'RB': {
                'rushing_td_rate': (0.03, 0.02),  # 3% per carry
                'receiving_td_rate': (0.08, 0.04),  # 8% per target
            },
            'WR': {'receiving_td_rate': (0.07, 0.03)},  # 7% per target
            'TE': {'receiving_td_rate': (0.06, 0.03)},  # 6% per target
        }

        self.alpha = settings.eb_alpha  # Empirical Bayes weight
        self.rolling_alpha = settings.rolling_alpha  # Exponential decay

    def smooth_features(
        self,
        player_id: str,
        position: str,
        historical_features: List[PlayerFeatures],
        target_game_id: str
    ) -> SmoothedFeatures:
        """
        Smooth historical features for a player

        Args:
            player_id: Player ID
            position: Player position (QB, RB, WR, TE)
            historical_features: List of historical PlayerFeatures
            target_game_id: Game to predict for

        Returns:
            SmoothedFeatures
        """
        if not historical_features:
            return self._get_default_features(player_id, target_game_id, position)

        # Sort by game_id (assuming chronological)
        features_sorted = sorted(historical_features, key=lambda x: x.game_id)

        # Compute rolling averages
        smoothed = SmoothedFeatures(player_id=player_id, game_id=target_game_id)

        # Count stats with exponential weighting
        smoothed.targets_per_game = self._rolling_average(
            [f.targets for f in features_sorted]
        )
        smoothed.receptions_per_game = self._rolling_average(
            [f.receptions for f in features_sorted]
        )
        smoothed.rush_attempts_per_game = self._rolling_average(
            [f.rush_attempts for f in features_sorted]
        )

        # Yards with exponential weighting
        smoothed.receiving_yards_per_game = self._rolling_average(
            [f.receiving_yards for f in features_sorted]
        )
        smoothed.rushing_yards_per_game = self._rolling_average(
            [f.rushing_yards for f in features_sorted]
        )
        smoothed.passing_yards_per_game = self._rolling_average(
            [f.passing_yards for f in features_sorted]
        )

        # TD rates with Empirical Bayes
        smoothed.receiving_td_rate = self._empirical_bayes_rate(
            successes=sum(f.receiving_tds for f in features_sorted),
            trials=sum(f.targets for f in features_sorted),
            position=position,
            stat='receiving_td_rate'
        )

        smoothed.rushing_td_rate = self._empirical_bayes_rate(
            successes=sum(f.rushing_tds for f in features_sorted),
            trials=sum(f.rush_attempts for f in features_sorted),
            position=position,
            stat='rushing_td_rate'
        )

        smoothed.passing_td_rate = self._empirical_bayes_rate(
            successes=sum(f.passing_tds for f in features_sorted),
            trials=sum(f.pass_attempts for f in features_sorted),
            position=position,
            stat='passing_td_rate'
        )

        # Shares (rolling average)
        target_shares = [f.target_share for f in features_sorted if f.target_share is not None]
        if target_shares:
            smoothed.target_share = self._rolling_average(target_shares)

        rush_shares = [f.rush_share for f in features_sorted if f.rush_share is not None]
        if rush_shares:
            smoothed.rush_share = self._rolling_average(rush_shares)

        snap_shares = [f.snap_share for f in features_sorted if f.snap_share is not None]
        if snap_shares:
            smoothed.snap_share = self._rolling_average(snap_shares)

        # Redzone share (targets + carries in redzone / total redzone opportunities)
        redzone_opps = sum(f.redzone_targets + f.redzone_carries for f in features_sorted)
        if redzone_opps > 0:
            # Estimate team redzone plays (rough)
            smoothed.redzone_share = redzone_opps / (len(features_sorted) * 10)  # ~10 RZ plays/game
            smoothed.redzone_share = min(smoothed.redzone_share, 1.0)

        # Efficiency metrics
        ypts = [f.yards_per_target for f in features_sorted if f.yards_per_target is not None]
        if ypts:
            smoothed.yards_per_target = self._rolling_average(ypts)

        ypcs = [f.yards_per_carry for f in features_sorted if f.yards_per_carry is not None]
        if ypcs:
            smoothed.yards_per_carry = self._rolling_average(ypcs)

        yprs = [f.yards_per_route for f in features_sorted if f.yards_per_route is not None]
        if yprs:
            smoothed.yards_per_route = self._rolling_average(yprs)

        # Confidence based on sample size
        smoothed.confidence = self._compute_confidence(features_sorted)

        return smoothed

    def _rolling_average(self, values: List[float]) -> float:
        """
        Compute exponentially-weighted rolling average

        More recent values get higher weight
        """
        if not values:
            return 0.0

        # Exponential weights (more recent = higher weight)
        weights = [self.rolling_alpha ** i for i in range(len(values) - 1, -1, -1)]
        weights = np.array(weights) / sum(weights)  # Normalize

        return float(np.average(values, weights=weights))

    def _empirical_bayes_rate(
        self,
        successes: int,
        trials: int,
        position: str,
        stat: str
    ) -> float:
        """
        Empirical Bayes estimate of a rate

        Shrinks observed rate toward position-specific prior

        Args:
            successes: Number of successes (e.g., TDs)
            trials: Number of trials (e.g., targets or carries)
            position: Player position
            stat: Stat name (e.g., 'receiving_td_rate')

        Returns:
            Smoothed rate estimate
        """
        if trials == 0:
            return 0.0

        # Get prior for this position and stat
        prior = self._get_prior(position, stat)
        if prior is None:
            # No prior, use observed rate
            return successes / trials

        prior_mean, prior_std = prior

        # Observed rate
        observed_rate = successes / trials

        # Empirical Bayes shrinkage
        # Weight based on sample size (more trials = less shrinkage)
        weight = trials / (trials + 100)  # Arbitrary scale, adjust as needed

        # Weighted average of prior and observed
        smoothed_rate = weight * observed_rate + (1 - weight) * prior_mean

        return smoothed_rate

    def _get_prior(self, position: str, stat: str) -> Optional[tuple[float, float]]:
        """Get prior (mean, std) for a position and stat"""
        if position in self.td_priors and stat in self.td_priors[position]:
            return self.td_priors[position][stat]
        return None

    def _compute_confidence(self, features: List[PlayerFeatures]) -> float:
        """
        Compute confidence based on sample size and consistency

        Args:
            features: List of historical features

        Returns:
            Confidence score 0.0 to 1.0
        """
        n_games = len(features)

        # Base confidence on number of games
        if n_games >= settings.lookback_games:
            base_confidence = 1.0
        elif n_games >= settings.min_games_for_model:
            base_confidence = n_games / settings.lookback_games
        else:
            base_confidence = 0.5

        # Penalize if recent games have very low usage
        recent_features = features[-3:] if len(features) >= 3 else features
        recent_snaps = [f.snaps for f in recent_features if f.snaps > 0]

        if recent_snaps:
            avg_snaps = np.mean(recent_snaps)
            if avg_snaps < settings.min_snap_count:
                base_confidence *= 0.5

        return min(base_confidence, 1.0)

    def _get_default_features(
        self,
        player_id: str,
        game_id: str,
        position: str
    ) -> SmoothedFeatures:
        """Return default features when no historical data"""
        logger.warning("no_historical_features", player_id=player_id, game_id=game_id)

        return SmoothedFeatures(
            player_id=player_id,
            game_id=game_id,
            confidence=0.1  # Low confidence for no data
        )

    def batch_smooth(
        self,
        player_positions: Dict[str, str],
        player_historical: Dict[str, List[PlayerFeatures]],
        target_game_id: str
    ) -> Dict[str, SmoothedFeatures]:
        """
        Batch smooth features for multiple players

        Args:
            player_positions: Dict mapping player_id to position
            player_historical: Dict mapping player_id to historical features
            target_game_id: Target game ID

        Returns:
            Dict mapping player_id to SmoothedFeatures
        """
        results = {}

        for player_id, position in player_positions.items():
            historical = player_historical.get(player_id, [])
            smoothed = self.smooth_features(
                player_id=player_id,
                position=position,
                historical_features=historical,
                target_game_id=target_game_id
            )
            results[player_id] = smoothed

        logger.info("batch_smoothed", count=len(results), game_id=target_game_id)

        return results
