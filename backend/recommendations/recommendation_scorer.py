"""
Unified recommendation system for props and parlays

Combines all signals:
- Statistical projections (distributions, XGBoost)
- Matchup features (opponent, game script, weather)
- Injury/roster updates (confidence adjustments)
- Trend analysis (recent form, streaks, hot/cold)
- News sentiment (impact estimation)
- Calibrated probabilities (real-world accuracy)
- Correlation modeling (parlay pricing)

Output: Ranked recommendations with confidence scores and reasoning
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np

from backend.config import settings
from backend.config.logging_config import get_logger
from backend.database.session import get_db
from backend.database.models import Player, Game, Projection

# Import all signal generators
from backend.models.prop_model_runner import PropModelRunner
from backend.trends.trend_analyzer import TrendAnalyzer, TrendSignal
from backend.news.news_fetcher import NewsAnalyzer, PlayerNewsDigest
from backend.correlation.correlation_engine import CorrelationEngine, ParlayAdjustment
from backend.calibration.calibrator import ProbabilityCalibrator
from backend.services.roster_injury_service import RosterInjuryService

logger = get_logger(__name__)


class RecommendationStrength(Enum):
    """Recommendation confidence level"""
    ELITE = "elite"  # 90%+ confidence, all signals aligned
    STRONG = "strong"  # 75-90% confidence, most signals positive
    MODERATE = "moderate"  # 60-75% confidence, mixed signals
    WEAK = "weak"  # 50-60% confidence, conflicting signals
    AVOID = "avoid"  # <50% confidence or negative signals


@dataclass
class SignalWeights:
    """Configurable weights for each signal type"""
    base_projection: float = 0.35  # Core statistical model
    matchup: float = 0.15  # Opponent strength, game script
    trend: float = 0.15  # Recent form, streaks
    news: float = 0.10  # Injury reports, news sentiment
    roster_confidence: float = 0.10  # Injury/availability confidence
    calibration: float = 0.10  # Historical accuracy adjustment
    value: float = 0.05  # Market odds vs model (if available)

    def normalize(self) -> 'SignalWeights':
        """Ensure weights sum to 1.0"""
        total = (
            self.base_projection + self.matchup + self.trend +
            self.news + self.roster_confidence + self.calibration + self.value
        )
        return SignalWeights(
            base_projection=self.base_projection / total,
            matchup=self.matchup / total,
            trend=self.trend / total,
            news=self.news / total,
            roster_confidence=self.roster_confidence / total,
            calibration=self.calibration / total,
            value=self.value / total,
        )


@dataclass
class PropRecommendation:
    """Single prop recommendation with full signal breakdown"""
    # Identifiers
    player_id: str
    player_name: str
    position: str
    team: str
    game_id: str
    market: str

    # Core prediction
    line: float  # Projected line
    model_prob: float  # Probability of going over
    calibrated_prob: float  # Calibrated probability

    # Signal breakdown (0-1 scores)
    base_signal: float  # From statistical model
    matchup_signal: float  # From matchup features
    trend_signal: float  # From recent performance
    news_signal: float  # From news/injuries
    roster_signal: float  # From roster confidence

    # Overall score
    overall_score: float  # Weighted combination (0-1)
    recommendation_strength: RecommendationStrength
    confidence: float  # 0-1

    # Metadata
    reasoning: List[str] = field(default_factory=list)  # Human-readable factors
    flags: List[str] = field(default_factory=list)  # Warnings/notes

    # Market info (if available)
    market_line: Optional[float] = None
    market_odds: Optional[float] = None
    edge: Optional[float] = None  # model_prob - implied_prob from odds


@dataclass
class ParlayRecommendation:
    """Parlay recommendation with correlation adjustments"""
    props: List[PropRecommendation]

    # Parlay pricing
    raw_probability: float  # Assuming independence
    adjusted_probability: float  # With correlation adjustments
    adjustment_factor: float

    # Parlay score
    overall_score: float  # Average of prop scores, adjusted for correlation
    correlation_impact: str  # "positive", "negative", "neutral"

    # Metadata
    confidence: float
    reasoning: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    # Market info
    market_odds: Optional[float] = None
    edge: Optional[float] = None


class RecommendationScorer:
    """
    Unified recommendation engine

    Combines all available signals to rank prop bets and parlay combinations
    """

    def __init__(
        self,
        signal_weights: Optional[SignalWeights] = None,
        min_confidence: float = 0.5
    ):
        """
        Initialize recommendation scorer

        Args:
            signal_weights: Custom weights for signal types
            min_confidence: Minimum confidence for recommendations
        """
        self.weights = (signal_weights or SignalWeights()).normalize()
        self.min_confidence = min_confidence

        # Initialize all signal generators
        self.prop_runner = PropModelRunner()
        self.trend_analyzer = TrendAnalyzer()
        self.news_analyzer = NewsAnalyzer()
        self.correlation_engine = CorrelationEngine()
        self.calibrator = ProbabilityCalibrator()
        self.roster_service = RosterInjuryService()

    def recommend_props(
        self,
        game_id: str,
        markets: Optional[List[str]] = None,
        limit: int = 20,
        include_reasoning: bool = True
    ) -> List[PropRecommendation]:
        """
        Generate prop recommendations for a game

        Args:
            game_id: Game ID
            markets: Markets to analyze (None = all supported)
            limit: Max recommendations to return
            include_reasoning: Include human-readable reasoning

        Returns:
            Ranked list of prop recommendations
        """
        logger.info("generating_prop_recommendations", game_id=game_id)

        markets = markets or settings.supported_markets
        recommendations = []

        with get_db() as session:
            # Get game info
            game = session.query(Game).filter(Game.game_id == game_id).first()
            if not game:
                logger.warning("game_not_found", game_id=game_id)
                return []

            # Get all players in game (both teams)
            players = (
                session.query(Player)
                .filter(Player.team.in_([game.home_team, game.away_team]))
                .filter(Player.position.in_(['QB', 'RB', 'WR', 'TE']))
                .all()
            )

            # Score each player-market combination
            for player in players:
                for market in markets:
                    try:
                        rec = self._score_prop(
                            player_id=player.player_id,
                            game_id=game_id,
                            market=market,
                            player=player,
                            game=game,
                            include_reasoning=include_reasoning
                        )

                        if rec and rec.confidence >= self.min_confidence:
                            recommendations.append(rec)

                    except Exception as e:
                        logger.error(
                            "prop_scoring_failed",
                            player_id=player.player_id,
                            market=market,
                            error=str(e)
                        )

        # Sort by overall score
        recommendations.sort(key=lambda x: x.overall_score, reverse=True)

        logger.info(
            "recommendations_generated",
            game_id=game_id,
            total_recs=len(recommendations),
            top_score=recommendations[0].overall_score if recommendations else 0
        )

        return recommendations[:limit]

    def recommend_parlays(
        self,
        game_id: str,
        parlay_size: int = 3,
        min_correlation: float = 0.0,
        max_correlation: float = 0.8,
        limit: int = 10
    ) -> List[ParlayRecommendation]:
        """
        Generate correlated parlay recommendations

        Args:
            game_id: Game ID
            parlay_size: Number of props in parlay (2-6)
            min_correlation: Minimum correlation for stacking
            max_correlation: Maximum correlation (avoid over-correlated)
            limit: Max parlays to return

        Returns:
            Ranked list of parlay recommendations
        """
        logger.info(
            "generating_parlay_recommendations",
            game_id=game_id,
            parlay_size=parlay_size
        )

        # First get high-quality single props
        prop_recs = self.recommend_props(
            game_id=game_id,
            limit=50,  # Get more candidates for combinations
            include_reasoning=False  # Skip for performance
        )

        if len(prop_recs) < parlay_size:
            logger.warning("insufficient_props_for_parlays", available=len(prop_recs))
            return []

        parlays = []

        # Generate parlay combinations
        from itertools import combinations

        for combo in combinations(prop_recs, parlay_size):
            try:
                parlay_rec = self._score_parlay(
                    props=list(combo),
                    min_correlation=min_correlation,
                    max_correlation=max_correlation
                )

                if parlay_rec and parlay_rec.confidence >= self.min_confidence:
                    parlays.append(parlay_rec)

            except Exception as e:
                logger.error("parlay_scoring_failed", error=str(e))

        # Sort by overall score
        parlays.sort(key=lambda x: x.overall_score, reverse=True)

        logger.info(
            "parlays_generated",
            game_id=game_id,
            total_parlays=len(parlays)
        )

        return parlays[:limit]

    def _score_prop(
        self,
        player_id: str,
        game_id: str,
        market: str,
        player: Player,
        game: Game,
        include_reasoning: bool
    ) -> Optional[PropRecommendation]:
        """
        Score a single prop using all signals

        Returns:
            PropRecommendation or None if insufficient data
        """
        # 1. Base projection (statistical model)
        try:
            projection = self.prop_runner.generate_projection(
                player_id=player_id,
                game_id=game_id,
                market=market
            )

            if not projection:
                return None

            base_signal = projection.model_prob or 0.5
            line = projection.mu or 0.0

        except Exception as e:
            logger.debug("base_projection_failed", player_id=player_id, error=str(e))
            return None

        # 2. Calibrated probability
        calibrated_prob = self.calibrator.apply_calibration(
            model_prob=base_signal,
            market=market,
            season=game.season
        )

        # 3. Trend signal
        try:
            trend_signal_obj = self.trend_analyzer.analyze_player_trends(
                player_id=player_id,
                market=market,
                n_games=10
            )

            # Convert trend signal strength (-1 to +1) to 0-1 scale
            trend_signal = (trend_signal_obj.signal_strength + 1) / 2
            trend_signal *= trend_signal_obj.confidence  # Weight by confidence

            trend_reasoning = []
            if trend_signal_obj.recent_form_score > 0.3:
                trend_reasoning.append(f"Hot streak: {trend_signal_obj.streak_count} game streak")
            elif trend_signal_obj.recent_form_score < -0.3:
                trend_reasoning.append(f"Cold streak: {abs(trend_signal_obj.streak_count)} games under")

        except Exception as e:
            trend_signal = 0.5  # Neutral
            trend_reasoning = []

        # 4. News signal
        try:
            news_digest = self.news_analyzer.fetch_player_news(
                player_id=player_id,
                player_name=player.full_name,
                max_age_hours=48
            )

            # Convert news impact to signal (0-1)
            # MAJOR_BOOST = 1.0, MAJOR_DECREASE = 0.0
            impact_map = {
                'MAJOR_BOOST': 0.9,
                'MODERATE_BOOST': 0.7,
                'MINOR_BOOST': 0.6,
                'NEUTRAL': 0.5,
                'MINOR_DECREASE': 0.4,
                'MODERATE_DECREASE': 0.3,
                'MAJOR_DECREASE': 0.1,
            }

            news_signal = impact_map.get(
                news_digest.aggregate_impact.name,
                0.5
            ) if news_digest else 0.5

            news_reasoning = []
            if news_digest and news_digest.key_items:
                top_news = news_digest.key_items[0]
                news_reasoning.append(f"News: {top_news.headline[:50]}...")

        except Exception as e:
            news_signal = 0.5
            news_reasoning = []

        # 5. Roster/injury confidence signal
        try:
            roster_status = self.roster_service.get_player_status(player_id)

            # Convert status to signal
            status_map = {
                'ACTIVE': 0.9,
                'PROBABLE': 0.7,
                'QUESTIONABLE': 0.5,
                'DOUBTFUL': 0.3,
                'OUT': 0.0,
            }

            roster_signal = status_map.get(
                roster_status.status.name if roster_status else 'ACTIVE',
                0.5
            )

            roster_reasoning = []
            if roster_status and roster_status.status.name != 'ACTIVE':
                roster_reasoning.append(f"Status: {roster_status.status.name}")

        except Exception as e:
            roster_signal = 0.8  # Default to mostly healthy
            roster_reasoning = []

        # 6. Matchup signal (already in projection, extract from features)
        # For now, use a simple heuristic based on opponent rank
        matchup_signal = 0.5  # Neutral default
        matchup_reasoning = []

        # 7. Combine all signals using weights
        overall_score = (
            self.weights.base_projection * base_signal +
            self.weights.matchup * matchup_signal +
            self.weights.trend * trend_signal +
            self.weights.news * news_signal +
            self.weights.roster_confidence * roster_signal +
            self.weights.calibration * calibrated_prob
        )

        # 8. Determine recommendation strength
        if overall_score >= 0.85:
            strength = RecommendationStrength.ELITE
        elif overall_score >= 0.75:
            strength = RecommendationStrength.STRONG
        elif overall_score >= 0.65:
            strength = RecommendationStrength.MODERATE
        elif overall_score >= 0.55:
            strength = RecommendationStrength.WEAK
        else:
            strength = RecommendationStrength.AVOID

        # 9. Calculate confidence (variance of signals)
        signal_values = [
            base_signal, matchup_signal, trend_signal,
            news_signal, roster_signal, calibrated_prob
        ]
        signal_variance = np.var(signal_values)
        confidence = 1.0 - min(signal_variance * 2, 0.5)  # High variance = low confidence

        # 10. Build reasoning
        reasoning = []
        if include_reasoning:
            reasoning.append(f"Model probability: {calibrated_prob:.1%}")
            reasoning.extend(trend_reasoning)
            reasoning.extend(news_reasoning)
            reasoning.extend(matchup_reasoning)
            reasoning.extend(roster_reasoning)

        # 11. Flags
        flags = []
        if roster_signal < 0.7:
            flags.append("INJURY_CONCERN")
        if signal_variance > 0.3:
            flags.append("CONFLICTING_SIGNALS")
        if trend_signal < 0.4:
            flags.append("COLD_STREAK")

        return PropRecommendation(
            player_id=player_id,
            player_name=player.full_name,
            position=player.position,
            team=player.team,
            game_id=game_id,
            market=market,
            line=line,
            model_prob=base_signal,
            calibrated_prob=calibrated_prob,
            base_signal=base_signal,
            matchup_signal=matchup_signal,
            trend_signal=trend_signal,
            news_signal=news_signal,
            roster_signal=roster_signal,
            overall_score=overall_score,
            recommendation_strength=strength,
            confidence=confidence,
            reasoning=reasoning,
            flags=flags
        )

    def _score_parlay(
        self,
        props: List[PropRecommendation],
        min_correlation: float,
        max_correlation: float
    ) -> Optional[ParlayRecommendation]:
        """
        Score a parlay combination

        Args:
            props: List of prop recommendations
            min_correlation: Minimum average correlation
            max_correlation: Maximum average correlation

        Returns:
            ParlayRecommendation or None if invalid
        """
        # Calculate correlation adjustment
        prop_dicts = [
            {
                'player_id': p.player_id,
                'market': p.market,
                'model_prob': p.calibrated_prob
            }
            for p in props
        ]

        adjustment = self.correlation_engine.estimate_parlay_probability(
            props=prop_dicts,
            use_correlation=True
        )

        # Check correlation constraints
        if adjustment.avg_correlation < min_correlation:
            return None  # Not correlated enough for stacking
        if adjustment.avg_correlation > max_correlation:
            return None  # Too correlated (risky)

        # Calculate parlay score
        # Average prop scores, adjusted by correlation impact
        avg_prop_score = np.mean([p.overall_score for p in props])

        # Boost for positive correlation, penalize for negative
        correlation_boost = 1.0 + (adjustment.avg_correlation * 0.2)
        parlay_score = avg_prop_score * correlation_boost
        parlay_score = min(parlay_score, 1.0)  # Cap at 1.0

        # Confidence (average of prop confidences, adjusted for correlation certainty)
        avg_confidence = np.mean([p.confidence for p in props])

        # Build reasoning
        reasoning = [
            f"Parlay of {len(props)} props",
            f"Raw probability: {adjustment.raw_probability:.2%}",
            f"Adjusted probability: {adjustment.adjusted_probability:.2%}",
            f"Correlation: {adjustment.avg_correlation:+.2f} ({adjustment.correlation_impact})",
        ]

        # Flags
        flags = []
        if adjustment.correlation_impact == "negative":
            flags.append("NEGATIVE_CORRELATION")
        if any("INJURY_CONCERN" in p.flags for p in props):
            flags.append("CONTAINS_INJURY_CONCERNS")

        return ParlayRecommendation(
            props=props,
            raw_probability=adjustment.raw_probability,
            adjusted_probability=adjustment.adjusted_probability,
            adjustment_factor=adjustment.adjustment_factor,
            overall_score=parlay_score,
            correlation_impact=adjustment.correlation_impact,
            confidence=avg_confidence,
            reasoning=reasoning,
            flags=flags
        )

    def get_best_single_props(
        self,
        season: int,
        week: Optional[int] = None,
        limit: int = 10
    ) -> List[PropRecommendation]:
        """
        Get best single prop recommendations across all games

        Args:
            season: NFL season
            week: Optional week filter
            limit: Max recommendations

        Returns:
            Top recommendations
        """
        all_recs = []

        with get_db() as session:
            # Get all games
            query = session.query(Game).filter(Game.season == season)

            if week is not None:
                query = query.filter(Game.week == week)

            games = query.all()

            for game in games:
                game_recs = self.recommend_props(
                    game_id=game.game_id,
                    limit=100,
                    include_reasoning=True
                )
                all_recs.extend(game_recs)

        # Sort by score
        all_recs.sort(key=lambda x: x.overall_score, reverse=True)

        return all_recs[:limit]

    def find_stacks(
        self,
        game_id: str,
        base_player_id: str,
        min_correlation: float = 0.3,
        limit: int = 5
    ) -> List[PropRecommendation]:
        """
        Find props that correlate well with a base prop (for stacking)

        Args:
            game_id: Game ID
            base_player_id: Base player to stack with
            min_correlation: Minimum correlation threshold
            limit: Max correlated props

        Returns:
            List of correlated prop recommendations
        """
        # Get all props for game
        all_props = self.recommend_props(
            game_id=game_id,
            limit=100,
            include_reasoning=False
        )

        # Find base prop
        base_props = [p for p in all_props if p.player_id == base_player_id]
        if not base_props:
            return []

        base_prop = base_props[0]

        # Calculate correlations with other props
        correlated = []
        for prop in all_props:
            if prop.player_id == base_player_id:
                continue

            pair = self.correlation_engine.calculate_correlation(
                player1_id=base_prop.player_id,
                player2_id=prop.player_id,
                market1=base_prop.market,
                market2=prop.market
            )

            if pair.correlation >= min_correlation and pair.confidence > 0.5:
                # Add correlation info to reasoning
                prop.reasoning.insert(
                    0,
                    f"Correlation with {base_prop.player_name}: {pair.correlation:+.2f}"
                )
                correlated.append((pair.correlation, prop))

        # Sort by correlation strength
        correlated.sort(key=lambda x: x[0], reverse=True)

        return [prop for _, prop in correlated[:limit]]
