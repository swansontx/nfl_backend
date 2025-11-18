"""Projection endpoints"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session

from backend.api.schemas import (
    GameProjectionsResponse,
    PlayerProjectionsResponse,
    ProjectionResponse
)
from backend.api.dependencies import get_db
from backend.database.models import Projection, Game, Player
from backend.config.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/projections", tags=["projections"])


@router.get("/games/{game_id}", response_model=GameProjectionsResponse)
async def get_game_projections(
    game_id: str,
    market: Optional[str] = Query(None, description="Filter by market"),
    tier: Optional[str] = Query(None, description="Filter by tier (core/mid/lotto)"),
    position: Optional[str] = Query(None, description="Filter by position"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Minimum confidence"),
    limit: int = Query(100, ge=1, le=500, description="Max results"),
    db: Session = Depends(get_db)
):
    """
    Get projections for a specific game

    Args:
        game_id: Game identifier
        market: Filter by market (e.g., player_rec_yds)
        tier: Filter by tier (core, mid, lotto)
        position: Filter by position (QB, RB, WR, TE)
        min_confidence: Minimum model confidence (0.0 - 1.0)
        limit: Maximum number of results

    Returns:
        GameProjectionsResponse with filtered projections
    """
    logger.info("get_game_projections", game_id=game_id, market=market, tier=tier)

    # Get game info
    game = db.query(Game).filter(Game.game_id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail=f"Game not found: {game_id}")

    # Build query
    query = (
        db.query(Projection)
        .join(Player, Projection.player_id == Player.player_id)
        .filter(Projection.game_id == game_id)
    )

    # Apply filters
    if market:
        query = query.filter(Projection.market == market)

    if tier:
        query = query.filter(Projection.tier == tier.lower())

    if position:
        query = query.filter(Player.position == position.upper())

    # Order by score descending (best opportunities first)
    query = query.order_by(Projection.score.desc().nulls_last())

    # Get total count
    total_count = query.count()

    # Apply limit
    projections = query.limit(limit).all()

    # Get player info for response
    player_ids = [p.player_id for p in projections]
    players = {
        p.player_id: p
        for p in db.query(Player).filter(Player.player_id.in_(player_ids)).all()
    }

    # Build response
    projection_responses = []
    for proj in projections:
        player = players.get(proj.player_id)
        if not player:
            continue

        projection_responses.append(
            ProjectionResponse(
                player_id=proj.player_id,
                player_name=player.display_name,
                team=player.team,
                position=player.position,
                market=proj.market,
                mu=proj.mu,
                sigma=proj.sigma,
                prob_over_line=proj.calibrated_prob,
                confidence=1.0,  # Could get from features
                usage_norm=proj.usage_norm,
                tier=proj.tier,
                score=proj.score,
                model_version=proj.model_version or "v1"
            )
        )

    return GameProjectionsResponse(
        game_id=game_id,
        home_team=game.home_team,
        away_team=game.away_team,
        game_date=game.game_date,
        total_projections=total_count,
        projections=projection_responses
    )


@router.get("/players/{player_id}", response_model=PlayerProjectionsResponse)
async def get_player_projections(
    player_id: str,
    season: Optional[int] = Query(None, description="Filter by season"),
    market: Optional[str] = Query(None, description="Filter by market"),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Get projections for a specific player

    Args:
        player_id: Player identifier
        season: Filter by season year
        market: Filter by market type
        limit: Maximum number of results

    Returns:
        PlayerProjectionsResponse with player's projections
    """
    logger.info("get_player_projections", player_id=player_id, season=season)

    # Get player info
    player = db.query(Player).filter(Player.player_id == player_id).first()
    if not player:
        raise HTTPException(status_code=404, detail=f"Player not found: {player_id}")

    # Build query
    query = (
        db.query(Projection)
        .join(Game, Projection.game_id == Game.game_id)
        .filter(Projection.player_id == player_id)
    )

    # Apply filters
    if season:
        query = query.filter(Game.season == season)

    if market:
        query = query.filter(Projection.market == market)

    # Order by game date descending (most recent first)
    query = query.order_by(Game.game_date.desc())

    # Get total and projections
    total_count = query.count()
    projections = query.limit(limit).all()

    # Build response
    projection_responses = []
    for proj in projections:
        projection_responses.append(
            ProjectionResponse(
                player_id=proj.player_id,
                player_name=player.display_name,
                team=player.team,
                position=player.position,
                market=proj.market,
                mu=proj.mu,
                sigma=proj.sigma,
                prob_over_line=proj.calibrated_prob,
                confidence=1.0,
                usage_norm=proj.usage_norm,
                tier=proj.tier,
                score=proj.score,
                model_version=proj.model_version or "v1"
            )
        )

    return PlayerProjectionsResponse(
        player_id=player_id,
        player_name=player.display_name,
        team=player.team,
        position=player.position,
        season=season or 2025,
        total_projections=total_count,
        projections=projection_responses
    )
