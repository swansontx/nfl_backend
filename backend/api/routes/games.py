"""Games API routes"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import datetime, timedelta

from backend.api.models.games import GameResponse, GamesListResponse
from backend.database.session import get_db
from backend.database.models import Game
from backend.config.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/games", tags=["games"])


@router.get("/", response_model=GamesListResponse)
def get_games(
    season: Optional[int] = Query(None, description="Filter by season"),
    week: Optional[int] = Query(None, ge=1, le=18, description="Filter by week"),
    team: Optional[str] = Query(None, description="Filter by team (home or away)"),
    date: Optional[str] = Query(None, description="Filter by date (YYYY-MM-DD)"),
    upcoming: bool = Query(False, description="Only upcoming games"),
):
    """
    Get list of games with optional filters

    Filters:
    - season: NFL season year
    - week: Week number
    - team: Team abbreviation (e.g., KC, BUF)
    - date: Specific date
    - upcoming: Only games that haven't been played yet
    """
    logger.info("get_games", season=season, week=week, team=team)

    with get_db() as session:
        query = session.query(Game)

        # Apply filters
        if season:
            query = query.filter(Game.season == season)

        if week:
            query = query.filter(Game.week == week)

        if team:
            query = query.filter(
                (Game.home_team == team.upper()) | (Game.away_team == team.upper())
            )

        if date:
            try:
                target_date = datetime.fromisoformat(date).date()
                query = query.filter(Game.game_date >= target_date)
                query = query.filter(Game.game_date < target_date + timedelta(days=1))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

        if upcoming:
            query = query.filter(Game.game_date >= datetime.utcnow())

        # Order by date
        query = query.order_by(Game.game_date)

        games = query.all()

        # Convert to response
        game_responses = [
            GameResponse(
                game_id=g.game_id,
                season=g.season,
                week=g.week,
                game_type=g.game_type,
                home_team=g.home_team,
                away_team=g.away_team,
                game_date=g.game_date,
                stadium=g.stadium,
                home_score=g.home_score,
                away_score=g.away_score,
                completed=g.game_date < datetime.utcnow() and g.home_score is not None,
            )
            for g in games
        ]

        return GamesListResponse(
            games=game_responses,
            total_count=len(game_responses),
            season=season,
            week=week,
            team=team,
            date=date,
        )


@router.get("/today", response_model=GamesListResponse)
def get_todays_games():
    """
    Get today's games

    Convenience endpoint for getting games scheduled for today.
    """
    today = datetime.utcnow().date()

    with get_db() as session:
        games = (
            session.query(Game)
            .filter(Game.game_date >= today)
            .filter(Game.game_date < today + timedelta(days=1))
            .order_by(Game.game_date)
            .all()
        )

        game_responses = [
            GameResponse(
                game_id=g.game_id,
                season=g.season,
                week=g.week,
                game_type=g.game_type,
                home_team=g.home_team,
                away_team=g.away_team,
                game_date=g.game_date,
                stadium=g.stadium,
                home_score=g.home_score,
                away_score=g.away_score,
                completed=False,
            )
            for g in games
        ]

        return GamesListResponse(
            games=game_responses,
            total_count=len(game_responses),
            date=str(today),
        )


@router.get("/{game_id}", response_model=GameResponse)
def get_game(game_id: str):
    """
    Get single game by ID

    Returns detailed information for a specific game.
    """
    logger.info("get_game", game_id=game_id)

    with get_db() as session:
        game = session.query(Game).filter(Game.game_id == game_id).first()

        if not game:
            raise HTTPException(status_code=404, detail=f"Game {game_id} not found")

        return GameResponse(
            game_id=game.game_id,
            season=game.season,
            week=game.week,
            game_type=game.game_type,
            home_team=game.home_team,
            away_team=game.away_team,
            game_date=game.game_date,
            stadium=game.stadium,
            home_score=game.home_score,
            away_score=game.away_score,
            completed=game.game_date < datetime.utcnow() and game.home_score is not None,
        )
