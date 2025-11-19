"""
Correlation modeling for same-game parlays

Estimates correlations between props to properly price parlays.
Naive parlay pricing assumes independence, which is wrong.

Examples:
- QB pass yards ↔ WR rec yards (POSITIVE correlation)
- Team total pass attempts ↔ Team total rush attempts (NEGATIVE)
- RB touches when team is winning ↔ Game script (POSITIVE)
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd

from backend.config import settings
from backend.config.logging_config import get_logger
from backend.database.session import get_db
from backend.database.models import PlayerGameFeature, Game, Player

logger = get_logger(__name__)


@dataclass
class CorrelationPair:
    """Correlation between two props"""
    player1_id: str
    player2_id: str
    market1: str
    market2: str

    correlation: float  # -1 to +1
    confidence: float  # 0 to 1
    sample_size: int

    # Metadata
    same_team: bool = False
    same_position: bool = False


@dataclass
class ParlayAdjustment:
    """Adjustment for parlay pricing based on correlations"""
    raw_probability: float  # P(A) * P(B) * P(C) assuming independence
    adjusted_probability: float  # Actual P(A ∩ B ∩ C) accounting for correlations

    adjustment_factor: float  # adjusted / raw
    correlation_impact: str  # "positive", "negative", "neutral"

    # Supporting data
    prop_count: int
    avg_correlation: float


class CorrelationEngine:
    """
    Models correlations between player props

    Uses historical data to estimate how props move together
    """

    def __init__(self):
        """Initialize correlation engine"""
        self._correlation_cache: Dict[Tuple, float] = {}

    def calculate_correlation(
        self,
        player1_id: str,
        player2_id: str,
        market1: str,
        market2: str,
        n_games: int = 50
    ) -> CorrelationPair:
        """
        Calculate correlation between two player-market pairs

        Args:
            player1_id: First player
            player2_id: Second player
            market1: First market
            market2: Second market
            n_games: Number of games to analyze

        Returns:
            CorrelationPair with correlation estimate
        """
        cache_key = (player1_id, player2_id, market1, market2)

        if cache_key in self._correlation_cache:
            correlation = self._correlation_cache[cache_key]
        else:
            # Get historical data
            data1, data2 = self._get_paired_data(
                player1_id, player2_id, market1, market2, n_games
            )

            if len(data1) < 10:
                logger.warning(
                    "insufficient_correlation_data",
                    player1=player1_id,
                    player2=player2_id,
                    samples=len(data1)
                )
                correlation = 0.0
            else:
                # Calculate Pearson correlation
                correlation = float(np.corrcoef(data1, data2)[0, 1])

                # Cache it
                self._correlation_cache[cache_key] = correlation

        # Get player info
        with get_db() as session:
            p1 = session.query(Player).filter(Player.player_id == player1_id).first()
            p2 = session.query(Player).filter(Player.player_id == player2_id).first()

            same_team = p1.team == p2.team if (p1 and p2) else False
            same_position = p1.position == p2.position if (p1 and p2) else False

        # Confidence based on sample size
        confidence = min(len(data1) / 30, 1.0) if data1 else 0.0

        pair = CorrelationPair(
            player1_id=player1_id,
            player2_id=player2_id,
            market1=market1,
            market2=market2,
            correlation=correlation,
            confidence=confidence,
            sample_size=len(data1) if data1 else 0,
            same_team=same_team,
            same_position=same_position
        )

        logger.debug(
            "correlation_calculated",
            correlation=correlation,
            samples=len(data1) if data1 else 0
        )

        return pair

    def estimate_parlay_probability(
        self,
        props: List[Dict],  # List of {player_id, market, model_prob}
        use_correlation: bool = True
    ) -> ParlayAdjustment:
        """
        Estimate true parlay probability accounting for correlations

        Args:
            props: List of props in parlay
            use_correlation: Whether to adjust for correlations

        Returns:
            ParlayAdjustment with true probability
        """
        # Raw probability (assuming independence)
        raw_prob = 1.0
        for prop in props:
            raw_prob *= prop['model_prob']

        if not use_correlation or len(props) < 2:
            return ParlayAdjustment(
                raw_probability=raw_prob,
                adjusted_probability=raw_prob,
                adjustment_factor=1.0,
                correlation_impact="neutral",
                prop_count=len(props),
                avg_correlation=0.0
            )

        # Calculate pairwise correlations
        correlations = []
        for i in range(len(props)):
            for j in range(i + 1, len(props)):
                pair = self.calculate_correlation(
                    props[i]['player_id'],
                    props[j]['player_id'],
                    props[i]['market'],
                    props[j]['market']
                )
                correlations.append(pair.correlation)

        avg_correlation = np.mean(correlations) if correlations else 0.0

        # Adjust probability based on average correlation
        # Positive correlation → parlay more likely
        # Negative correlation → parlay less likely
        adjustment_factor = self._correlation_adjustment_factor(
            avg_correlation, len(props)
        )

        adjusted_prob = raw_prob * adjustment_factor

        # Determine impact
        if adjustment_factor > 1.05:
            impact = "positive"
        elif adjustment_factor < 0.95:
            impact = "negative"
        else:
            impact = "neutral"

        adjustment = ParlayAdjustment(
            raw_probability=raw_prob,
            adjusted_probability=adjusted_prob,
            adjustment_factor=adjustment_factor,
            correlation_impact=impact,
            prop_count=len(props),
            avg_correlation=avg_correlation
        )

        logger.info(
            "parlay_adjusted",
            raw_prob=raw_prob,
            adjusted_prob=adjusted_prob,
            avg_correlation=avg_correlation
        )

        return adjustment

    def find_correlated_props(
        self,
        base_prop: Dict,  # {player_id, market}
        min_correlation: float = 0.3,
        limit: int = 5
    ) -> List[CorrelationPair]:
        """
        Find props that are highly correlated with base prop

        Useful for building correlated parlays (stacks)

        Args:
            base_prop: Base prop to find correlations for
            min_correlation: Minimum correlation threshold
            limit: Max results to return

        Returns:
            List of correlated props
        """
        correlated = []

        with get_db() as session:
            # Get all active players
            players = session.query(Player).filter(
                Player.position.in_(['QB', 'RB', 'WR', 'TE'])
            ).limit(100).all()  # Limit for performance

            for player in players:
                if player.player_id == base_prop['player_id']:
                    continue

                # Check common markets
                for market in ['player_rec_yds', 'player_rush_yds', 'player_pass_yds']:
                    pair = self.calculate_correlation(
                        base_prop['player_id'],
                        player.player_id,
                        base_prop['market'],
                        market
                    )

                    if pair.correlation >= min_correlation and pair.confidence > 0.5:
                        correlated.append(pair)

        # Sort by correlation strength
        correlated.sort(key=lambda x: x.correlation, reverse=True)

        return correlated[:limit]

    def _get_paired_data(
        self,
        player1_id: str,
        player2_id: str,
        market1: str,
        market2: str,
        n_games: int
    ) -> Tuple[List[float], List[float]]:
        """
        Get paired historical data for correlation calculation

        Only includes games where both players played
        """
        with get_db() as session:
            # Get common games
            features1 = (
                session.query(PlayerGameFeature, Game)
                .join(Game, PlayerGameFeature.game_id == Game.game_id)
                .filter(PlayerGameFeature.player_id == player1_id)
                .order_by(Game.game_date.desc())
                .limit(n_games * 2)  # Get extra to find overlaps
                .all()
            )

            features2 = (
                session.query(PlayerGameFeature, Game)
                .join(Game, PlayerGameFeature.game_id == Game.game_id)
                .filter(PlayerGameFeature.player_id == player2_id)
                .order_by(Game.game_date.desc())
                .limit(n_games * 2)
                .all()
            )

            # Create dicts for lookup
            f1_dict = {f.game_id: f for f, g in features1}
            f2_dict = {f.game_id: f for f, g in features2}

            # Find common games
            common_games = set(f1_dict.keys()) & set(f2_dict.keys())

            data1 = []
            data2 = []

            for game_id in list(common_games)[:n_games]:
                val1 = self._extract_market_value(f1_dict[game_id], market1)
                val2 = self._extract_market_value(f2_dict[game_id], market2)

                if val1 is not None and val2 is not None:
                    data1.append(val1)
                    data2.append(val2)

            return data1, data2

    def _extract_market_value(
        self,
        feature: PlayerGameFeature,
        market: str
    ) -> Optional[float]:
        """Extract stat value for market"""
        market_stat_map = {
            'player_rec_yds': 'receiving_yards',
            'player_receptions': 'receptions',
            'player_rush_yds': 'rushing_yards',
            'player_rush_attempts': 'rush_attempts',
            'player_pass_yds': 'passing_yards',
            'player_pass_tds': 'passing_tds',
        }

        stat_name = market_stat_map.get(market)
        if not stat_name:
            return None

        value = getattr(feature, stat_name, None)
        return float(value) if value is not None else None

    def _correlation_adjustment_factor(
        self,
        avg_correlation: float,
        num_props: int
    ) -> float:
        """
        Calculate adjustment factor for parlay probability

        Positive correlation increases joint probability
        Negative correlation decreases it
        """
        # For positive correlation:
        # If props are perfectly correlated (+1), hitting one makes others more likely
        # Adjustment factor > 1

        # For negative correlation:
        # If props are inversely correlated (-1), hitting one makes others less likely
        # Adjustment factor < 1

        # Simple linear adjustment (can be refined)
        # At correlation = 0, factor = 1.0 (no adjustment)
        # At correlation = +0.5, factor ≈ 1.2 (20% boost)
        # At correlation = -0.5, factor ≈ 0.8 (20% penalty)

        # Scale by number of props (more props = stronger effect)
        scale = 1 + (num_props - 2) * 0.1  # More props = more correlation impact

        adjustment = 1.0 + (avg_correlation * 0.4 * scale)

        # Clip to reasonable range
        return float(np.clip(adjustment, 0.5, 2.0))

    def get_known_correlations(self) -> Dict[str, float]:
        """
        Return known/expected correlations for common prop combinations

        Based on football logic, before empirical data
        """
        return {
            # Same team, same game
            'qb_pass_yds_wr_rec_yds_same_team': 0.6,  # Strong positive
            'qb_pass_tds_wr_rec_tds_same_team': 0.5,
            'rb_rush_att_team_pass_att': -0.4,  # Negative (game script)

            # Opposing teams
            'team1_score_team2_score': 0.3,  # Moderate positive (shootouts)

            # Same player
            'player_rec_yds_player_receptions': 0.8,  # Very strong
            'player_rush_yds_player_rush_att': 0.7,
        }

    def clear_cache(self):
        """Clear correlation cache"""
        self._correlation_cache.clear()
        logger.info("correlation_cache_cleared")
