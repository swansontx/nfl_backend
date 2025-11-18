"""Projection scoring and opportunity ranking"""
from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np

from backend.config import settings
from backend.config.logging_config import get_logger
from backend.models.prop_models import PropProjection

logger = get_logger(__name__)


@dataclass
class ScoredProjection:
    """Projection with calculated score"""
    projection: PropProjection

    # Scoring components
    edge: float  # Model edge over market
    model_prob: float  # Model probability
    implied_prob: Optional[float]  # Market implied probability
    volatility: float  # Historical volatility
    score: float  # Final opportunity score

    # Metadata
    market_line: Optional[float] = None  # If available
    odds: Optional[float] = None  # If available


class ProjectionScorer:
    """
    Score projections to identify best betting opportunities

    Formula:
    score = (edge * model_p * confidence) / (1 + k * volatility) + lambda * usage_norm

    Where:
    - edge: model_prob - implied_prob (from market odds)
    - model_p: model's probability estimate
    - confidence: model confidence (0-1)
    - volatility: historical variance of outcomes
    - usage_norm: normalized usage share (0-1)
    - k: volatility penalty coefficient
    - lambda: usage weight coefficient
    """

    def __init__(self):
        self.min_edge_threshold = settings.min_edge_threshold
        self.volatility_penalty_k = settings.volatility_penalty_k
        self.usage_weight_lambda = settings.usage_weight_lambda

    def score_projection(
        self,
        projection: PropProjection,
        market_line: Optional[float] = None,
        market_odds: Optional[float] = None,
        historical_volatility: Optional[float] = None
    ) -> ScoredProjection:
        """
        Score a single projection

        Args:
            projection: PropProjection to score
            market_line: Market betting line (e.g., 75.5 yards)
            market_odds: Market odds (e.g., -110 American odds)
            historical_volatility: Historical variance for this player/market

        Returns:
            ScoredProjection with calculated score
        """
        # Calculate model probability
        model_prob = self._calculate_model_prob(projection, market_line)

        # Calculate implied probability from odds
        implied_prob = self._odds_to_probability(market_odds) if market_odds else None

        # Calculate edge
        if implied_prob:
            edge = model_prob - implied_prob
        else:
            # No market odds available, assume no edge
            edge = 0.0

        # Get or estimate volatility
        volatility = historical_volatility if historical_volatility is not None else self._estimate_volatility(projection)

        # Calculate score
        confidence = projection.confidence
        usage_norm = projection.usage_norm or 0.0

        # Core scoring formula
        edge_component = (edge * model_prob * confidence) / (1 + self.volatility_penalty_k * volatility)
        usage_component = self.usage_weight_lambda * usage_norm

        score = edge_component + usage_component

        logger.debug(
            "projection_scored",
            player_id=projection.player_id,
            market=projection.market,
            edge=edge,
            model_prob=model_prob,
            score=score
        )

        return ScoredProjection(
            projection=projection,
            edge=edge,
            model_prob=model_prob,
            implied_prob=implied_prob,
            volatility=volatility,
            score=score,
            market_line=market_line,
            odds=market_odds
        )

    def score_batch(
        self,
        projections: List[PropProjection],
        market_data: Optional[Dict[str, Dict]] = None
    ) -> List[ScoredProjection]:
        """
        Score multiple projections

        Args:
            projections: List of projections to score
            market_data: Dict mapping (player_id, market) -> {line, odds, volatility}

        Returns:
            List of ScoredProjections
        """
        market_data = market_data or {}
        scored = []

        for proj in projections:
            key = (proj.player_id, proj.market)
            data = market_data.get(key, {})

            scored_proj = self.score_projection(
                projection=proj,
                market_line=data.get('line'),
                market_odds=data.get('odds'),
                historical_volatility=data.get('volatility')
            )
            scored.append(scored_proj)

        logger.info("batch_scored", count=len(scored))
        return scored

    def filter_positive_ev(self, scored_projections: List[ScoredProjection]) -> List[ScoredProjection]:
        """Filter to only positive expected value opportunities"""
        return [sp for sp in scored_projections if sp.edge >= self.min_edge_threshold]

    def rank_by_score(self, scored_projections: List[ScoredProjection]) -> List[ScoredProjection]:
        """Sort projections by score descending"""
        return sorted(scored_projections, key=lambda x: x.score, reverse=True)

    def _calculate_model_prob(self, projection: PropProjection, market_line: Optional[float]) -> float:
        """
        Calculate model's probability for the market

        For O/U markets, this is P(X > line)
        For binary markets (TD), this is the projection mu
        """
        if not market_line:
            # For binary markets like anytime TD
            if 'td' in projection.market.lower() or 'touchdown' in projection.market.lower():
                return projection.mu
            # Without a line, can't calculate probability
            return 0.5

        # Calculate P(X > line) from distribution
        if projection.dist_params:
            dist_type = projection.dist_params.get('dist_type') or projection.market

            if 'poisson' in dist_type or 'nbinom' in dist_type:
                # For count distributions
                from scipy import stats
                if 'lambda' in projection.dist_params:
                    lambda_param = projection.dist_params['lambda']
                    prob = 1 - stats.poisson.cdf(market_line, lambda_param)
                elif 'n' in projection.dist_params and 'p' in projection.dist_params:
                    n = projection.dist_params['n']
                    p = projection.dist_params['p']
                    prob = 1 - stats.nbinom.cdf(market_line, n, p)
                else:
                    prob = 0.5

            elif 'lognormal' in dist_type:
                # For yards distributions
                from scipy import stats
                mu = projection.dist_params.get('mu', np.log(projection.mu))
                sigma = projection.dist_params.get('sigma', 0.5)
                zero_prob = projection.dist_params.get('zero_prob', 0.0)

                if market_line <= 0:
                    prob = 1.0 - zero_prob
                else:
                    prob = 1 - stats.lognorm.cdf(market_line, s=sigma, scale=np.exp(mu))
                    prob *= (1 - zero_prob)

            elif 'normal' in dist_type:
                from scipy import stats
                mu = projection.dist_params.get('mu', projection.mu)
                sigma = projection.dist_params.get('sigma', projection.sigma or projection.mu * 0.3)
                prob = 1 - stats.norm.cdf(market_line, loc=mu, scale=sigma)

            else:
                # Unknown distribution, use simple heuristic
                prob = 1 / (1 + np.exp(-(projection.mu - market_line)))
        else:
            # No distribution params, use simple sigmoid
            prob = 1 / (1 + np.exp(-(projection.mu - market_line)))

        # Clip to reasonable range
        prob = np.clip(prob, 0.01, 0.99)

        return float(prob)

    def _odds_to_probability(self, odds: float) -> float:
        """
        Convert American odds to implied probability

        Args:
            odds: American odds (e.g., -110, +150)

        Returns:
            Implied probability (0-1)
        """
        if odds < 0:
            # Favorite (negative odds)
            prob = -odds / (-odds + 100)
        else:
            # Underdog (positive odds)
            prob = 100 / (odds + 100)

        # Include vig (typically ~4-5% per side, so true prob is lower)
        # For now, return raw implied prob
        return prob

    def _estimate_volatility(self, projection: PropProjection) -> float:
        """
        Estimate volatility from distribution parameters

        For now, use coefficient of variation (sigma / mu)
        """
        if projection.sigma and projection.mu > 0:
            cv = projection.sigma / projection.mu
            return min(cv, 2.0)  # Cap at 2.0

        # Default volatility by position/market type
        if projection.position == 'QB':
            return 0.3  # QBs are more consistent
        elif projection.position == 'RB':
            return 0.6  # RBs have moderate variance
        elif projection.position in ['WR', 'TE']:
            return 0.8  # Pass catchers are more volatile
        else:
            return 0.5

    def get_scoring_summary(self, scored_projections: List[ScoredProjection]) -> Dict:
        """Get summary statistics of scoring"""
        if not scored_projections:
            return {
                'total': 0,
                'positive_ev': 0,
                'avg_edge': 0,
                'avg_score': 0,
                'max_score': 0
            }

        edges = [sp.edge for sp in scored_projections]
        scores = [sp.score for sp in scored_projections]
        positive_ev = [sp for sp in scored_projections if sp.edge >= self.min_edge_threshold]

        return {
            'total': len(scored_projections),
            'positive_ev': len(positive_ev),
            'avg_edge': float(np.mean(edges)),
            'max_edge': float(np.max(edges)),
            'min_edge': float(np.min(edges)),
            'avg_score': float(np.mean(scores)),
            'max_score': float(np.max(scores)),
            'std_score': float(np.std(scores))
        }
