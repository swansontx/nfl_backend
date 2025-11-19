"""Recommendations API routes"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional

from backend.recommendations import RecommendationScorer
from backend.api.models.recommendations import (
    PropRecommendationResponse,
    ParlayRecommendationResponse,
    RecommendationsListResponse,
    ParlaysListResponse,
    PropInParlayResponse,
)
from backend.database.session import get_db
from backend.database.models import Game
from backend.config.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get("/{game_id}", response_model=RecommendationsListResponse)
def get_recommendations(
    game_id: str,
    limit: int = Query(10, ge=1, le=100, description="Max recommendations to return"),
    min_confidence: float = Query(0.6, ge=0, le=1, description="Minimum confidence threshold"),
    markets: Optional[List[str]] = Query(None, description="Specific markets to analyze"),
):
    """
    Get prop recommendations for a game

    Returns ranked prop recommendations with full signal breakdown.
    """
    logger.info("get_recommendations", game_id=game_id, limit=limit)

    # Get game info
    with get_db() as session:
        game = session.query(Game).filter(Game.game_id == game_id).first()

        if not game:
            raise HTTPException(status_code=404, detail=f"Game {game_id} not found")

        # Extract game data while session is active
        game_date = game.game_date
        home_team = game.home_team
        away_team = game.away_team

    # Generate recommendations
    try:
        scorer = RecommendationScorer(min_confidence=min_confidence)

        recs = scorer.recommend_props(
            game_id=game_id,
            markets=markets,
            limit=limit,
            include_reasoning=True
        )

        # Convert to response models
        recommendations = [
            PropRecommendationResponse(
                player_id=r.player_id,
                player_name=r.player_name,
                position=r.position,
                team=r.team,
                game_id=r.game_id,
                market=r.market,
                line=r.line,
                model_prob=r.model_prob,
                calibrated_prob=r.calibrated_prob,
                base_signal=r.base_signal,
                matchup_signal=r.matchup_signal,
                trend_signal=r.trend_signal,
                news_signal=r.news_signal,
                roster_signal=r.roster_signal,
                overall_score=r.overall_score,
                recommendation_strength=r.recommendation_strength.value,
                confidence=r.confidence,
                reasoning=r.reasoning,
                flags=r.flags,
                market_line=r.market_line,
                market_odds=r.market_odds,
                edge=r.edge,
            )
            for r in recs
        ]

        return RecommendationsListResponse(
            game_id=game_id,
            game_time=game_date,
            home_team=home_team,
            away_team=away_team,
            recommendations=recommendations,
            total_count=len(recommendations),
            markets_analyzed=markets or ["all"],
            min_confidence=min_confidence,
        )

    except Exception as e:
        logger.error("recommendations_failed", game_id=game_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")


@router.get("/{game_id}/parlays", response_model=ParlaysListResponse)
def get_parlays(
    game_id: str,
    parlay_size: int = Query(3, ge=2, le=6, description="Number of props in parlay"),
    min_correlation: float = Query(0.0, ge=-1, le=1, description="Minimum correlation for stacking"),
    max_correlation: float = Query(0.8, ge=-1, le=1, description="Maximum correlation (avoid over-correlated)"),
    limit: int = Query(5, ge=1, le=20, description="Max parlays to return"),
):
    """
    Get parlay recommendations for a game

    Returns correlation-adjusted parlay combinations.
    """
    logger.info("get_parlays", game_id=game_id, size=parlay_size)

    # Get game info
    with get_db() as session:
        game = session.query(Game).filter(Game.game_id == game_id).first()

        if not game:
            raise HTTPException(status_code=404, detail=f"Game {game_id} not found")

        # Game exists, we just need to verify it for now
        # (no need to extract data since we don't use it in response)

    # Generate parlays
    try:
        scorer = RecommendationScorer()

        parlays = scorer.recommend_parlays(
            game_id=game_id,
            parlay_size=parlay_size,
            min_correlation=min_correlation,
            max_correlation=max_correlation,
            limit=limit
        )

        # Convert to response models
        parlay_responses = [
            ParlayRecommendationResponse(
                props=[
                    PropInParlayResponse(
                        player_name=p.player_name,
                        market=p.market,
                        line=p.line,
                        probability=p.calibrated_prob,
                        score=p.overall_score,
                    )
                    for p in parlay.props
                ],
                raw_probability=parlay.raw_probability,
                adjusted_probability=parlay.adjusted_probability,
                adjustment_factor=parlay.adjustment_factor,
                overall_score=parlay.overall_score,
                correlation_impact=parlay.correlation_impact,
                confidence=parlay.confidence,
                reasoning=parlay.reasoning,
                flags=parlay.flags,
                market_odds=parlay.market_odds,
                edge=parlay.edge,
            )
            for parlay in parlays
        ]

        return ParlaysListResponse(
            game_id=game_id,
            parlays=parlay_responses,
            total_count=len(parlay_responses),
            parlay_size=parlay_size,
            min_correlation=min_correlation,
            max_correlation=max_correlation,
        )

    except Exception as e:
        logger.error("parlays_failed", game_id=game_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to generate parlays: {str(e)}")


@router.get("/player/{player_id}", response_model=List[PropRecommendationResponse])
def get_player_recommendations(
    player_id: str,
    season: int = Query(..., description="Season year"),
    week: Optional[int] = Query(None, description="Specific week"),
):
    """
    Get all recommendations for a specific player

    Useful for tracking a player's props across games.
    """
    logger.info("get_player_recommendations", player_id=player_id, season=season)

    # Get player's games
    with get_db() as session:
        query = session.query(Game).filter(Game.season == season)

        if week:
            query = query.filter(Game.week == week)

        games = query.all()

        # Extract game IDs while session is active
        game_ids = [g.game_id for g in games]

    # Generate recommendations for each game
    all_recs = []
    scorer = RecommendationScorer()

    for game_id in game_ids:
        try:
            recs = scorer.recommend_props(game_id=game_id, limit=100)

            # Filter to this player
            player_recs = [r for r in recs if r.player_id == player_id]
            all_recs.extend(player_recs)

        except Exception as e:
            logger.warning("player_rec_failed", game_id=game_id, error=str(e))
            continue

    # Convert to response
    return [
        PropRecommendationResponse(
            player_id=r.player_id,
            player_name=r.player_name,
            position=r.position,
            team=r.team,
            game_id=r.game_id,
            market=r.market,
            line=r.line,
            model_prob=r.model_prob,
            calibrated_prob=r.calibrated_prob,
            base_signal=r.base_signal,
            matchup_signal=r.matchup_signal,
            trend_signal=r.trend_signal,
            news_signal=r.news_signal,
            roster_signal=r.roster_signal,
            overall_score=r.overall_score,
            recommendation_strength=r.recommendation_strength.value,
            confidence=r.confidence,
            reasoning=r.reasoning,
            flags=r.flags,
        )
        for r in all_recs
    ]
