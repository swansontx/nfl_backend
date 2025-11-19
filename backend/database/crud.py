"""CRUD operations for database models.

Provides high-level database operations for:
- User management
- Bet tracking
- Prediction storage
- Line caching
- Bankroll management
"""

from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import uuid

from backend.database.models import (
    User, BetHistory, ModelPrediction,
    CachedLine, UserBankroll, UserWatchlist
)


# =============================================================================
# User Operations
# =============================================================================

def create_user(db: Session, email: str, username: str, hashed_password: str,
                subscription_tier: str = 'free') -> User:
    """Create a new user."""
    user = User(
        email=email,
        username=username,
        hashed_password=hashed_password,
        subscription_tier=subscription_tier
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Initialize bankroll
    create_user_bankroll(db, user.user_id)

    return user


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email."""
    return db.query(User).filter(User.email == email).first()


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Get user by username."""
    return db.query(User).filter(User.username == username).first()


def get_user_by_id(db: Session, user_id: uuid.UUID) -> Optional[User]:
    """Get user by ID."""
    return db.query(User).filter(User.user_id == user_id).first()


def update_user_login(db: Session, user_id: uuid.UUID):
    """Update user's last login timestamp."""
    user = get_user_by_id(db, user_id)
    if user:
        user.last_login = datetime.now()
        db.commit()


# =============================================================================
# Bet History Operations
# =============================================================================

def create_bet(db: Session, user_id: uuid.UUID, game_id: str,
               bet_type: str, line_value: float, odds: float,
               stake: float = 1.0, player_id: Optional[str] = None,
               prop_type: Optional[str] = None, notes: Optional[str] = None) -> BetHistory:
    """Create a new bet."""
    bet = BetHistory(
        user_id=user_id,
        game_id=game_id,
        player_id=player_id,
        bet_type=bet_type,
        prop_type=prop_type,
        line_value=line_value,
        odds=odds,
        stake=stake,
        result='pending',
        notes=notes
    )
    db.add(bet)
    db.commit()
    db.refresh(bet)

    # Update bankroll
    update_bankroll_total_bets(db, user_id)

    return bet


def settle_bet(db: Session, bet_id: uuid.UUID, result: str,
               actual_value: Optional[float] = None) -> BetHistory:
    """Settle a bet with result (win, loss, push).

    Args:
        db: Database session
        bet_id: Bet ID
        result: 'win', 'loss', or 'push'
        actual_value: Actual stat value (optional)
    """
    bet = db.query(BetHistory).filter(BetHistory.bet_id == bet_id).first()
    if not bet:
        raise ValueError(f"Bet {bet_id} not found")

    bet.result = result
    bet.actual_value = actual_value
    bet.settled_at = datetime.now()

    # Calculate payout
    if result == 'win':
        if bet.odds >= 0:
            bet.payout = bet.stake * (bet.odds / 100)
        else:
            bet.payout = bet.stake * (100 / abs(bet.odds))
    elif result == 'loss':
        bet.payout = -bet.stake
    else:  # push
        bet.payout = 0.0

    db.commit()
    db.refresh(bet)

    # Update bankroll
    update_bankroll_after_bet(db, bet.user_id, bet)

    return bet


def get_user_bets(db: Session, user_id: uuid.UUID, limit: int = 50,
                  result_filter: Optional[str] = None) -> List[BetHistory]:
    """Get user's bet history.

    Args:
        db: Database session
        user_id: User ID
        limit: Max number of bets to return
        result_filter: Filter by result ('win', 'loss', 'push', 'pending')
    """
    query = db.query(BetHistory).filter(BetHistory.user_id == user_id)

    if result_filter:
        query = query.filter(BetHistory.result == result_filter)

    return query.order_by(BetHistory.placed_at.desc()).limit(limit).all()


def get_pending_bets(db: Session, user_id: uuid.UUID) -> List[BetHistory]:
    """Get user's pending bets."""
    return get_user_bets(db, user_id, result_filter='pending')


# =============================================================================
# Bankroll Operations
# =============================================================================

def create_user_bankroll(db: Session, user_id: uuid.UUID,
                        starting_units: float = 100.0) -> UserBankroll:
    """Create bankroll for new user."""
    bankroll = UserBankroll(
        user_id=user_id,
        total_units=starting_units,
        starting_units=starting_units
    )
    db.add(bankroll)
    db.commit()
    db.refresh(bankroll)
    return bankroll


def get_user_bankroll(db: Session, user_id: uuid.UUID) -> Optional[UserBankroll]:
    """Get user's bankroll."""
    return db.query(UserBankroll).filter(UserBankroll.user_id == user_id).first()


def update_bankroll_total_bets(db: Session, user_id: uuid.UUID):
    """Update total bets count."""
    bankroll = get_user_bankroll(db, user_id)
    if bankroll:
        bankroll.total_bets += 1
        bankroll.last_bet_at = datetime.now()
        db.commit()


def update_bankroll_after_bet(db: Session, user_id: uuid.UUID, bet: BetHistory):
    """Update bankroll after bet settlement."""
    bankroll = get_user_bankroll(db, user_id)
    if not bankroll:
        return

    # Update counts
    if bet.result == 'win':
        bankroll.winning_bets += 1
    elif bet.result == 'loss':
        bankroll.losing_bets += 1
    elif bet.result == 'push':
        bankroll.push_bets += 1

    # Update units and profit
    bankroll.total_units += bet.payout
    bankroll.total_profit += bet.payout

    # Calculate metrics
    total_decided = bankroll.winning_bets + bankroll.losing_bets
    if total_decided > 0:
        bankroll.win_rate = bankroll.winning_bets / total_decided

    if bankroll.starting_units > 0:
        bankroll.roi = (bankroll.total_profit / bankroll.starting_units) * 100

    db.commit()


# =============================================================================
# Model Prediction Operations
# =============================================================================

def create_prediction(db: Session, game_id: str, player_id: str,
                     prediction_type: str, predicted_value: float,
                     confidence: Optional[float] = None,
                     vegas_line: Optional[float] = None,
                     season: Optional[int] = None,
                     week: Optional[int] = None,
                     model_version: str = 'v1',
                     features: Optional[Dict] = None) -> ModelPrediction:
    """Create a model prediction."""
    model_edge = None
    if vegas_line is not None:
        model_edge = predicted_value - vegas_line

    prediction = ModelPrediction(
        game_id=game_id,
        player_id=player_id,
        prediction_type=prediction_type,
        predicted_value=predicted_value,
        confidence=confidence,
        vegas_line=vegas_line,
        model_edge=model_edge,
        season=season,
        week=week,
        model_version=model_version,
        features=features or {}
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction


def update_prediction_actual(db: Session, prediction_id: uuid.UUID,
                            actual_value: float) -> ModelPrediction:
    """Update prediction with actual value after game."""
    prediction = db.query(ModelPrediction).filter(
        ModelPrediction.prediction_id == prediction_id
    ).first()

    if prediction:
        prediction.actual_value = actual_value
        prediction.prediction_error = abs(prediction.predicted_value - actual_value)
        db.commit()
        db.refresh(prediction)

    return prediction


def get_predictions_for_game(db: Session, game_id: str) -> List[ModelPrediction]:
    """Get all predictions for a game."""
    return db.query(ModelPrediction).filter(
        ModelPrediction.game_id == game_id
    ).all()


def get_prediction_accuracy(db: Session, prediction_type: str,
                           season: Optional[int] = None) -> Dict:
    """Calculate prediction accuracy metrics.

    Returns:
        Dict with RMSE, MAE, and sample count
    """
    query = db.query(ModelPrediction).filter(
        ModelPrediction.prediction_type == prediction_type,
        ModelPrediction.actual_value.isnot(None)
    )

    if season:
        query = query.filter(ModelPrediction.season == season)

    predictions = query.all()

    if not predictions:
        return {'rmse': None, 'mae': None, 'samples': 0}

    errors = [p.prediction_error for p in predictions if p.prediction_error is not None]

    mae = sum(errors) / len(errors) if errors else None
    rmse = (sum(e**2 for e in errors) / len(errors)) ** 0.5 if errors else None

    return {
        'rmse': rmse,
        'mae': mae,
        'samples': len(predictions)
    }


# =============================================================================
# Cached Line Operations
# =============================================================================

def cache_line(db: Session, game_id: str, sportsbook: str, market_type: str,
               line_value: float, over_odds: float, under_odds: float,
               player_id: Optional[str] = None, prop_type: Optional[str] = None,
               ttl_minutes: int = 60) -> CachedLine:
    """Cache a sportsbook line."""
    expires_at = datetime.now() + timedelta(minutes=ttl_minutes)

    line = CachedLine(
        game_id=game_id,
        player_id=player_id,
        sportsbook=sportsbook,
        market_type=market_type,
        prop_type=prop_type,
        line_value=line_value,
        over_odds=over_odds,
        under_odds=under_odds,
        expires_at=expires_at
    )
    db.add(line)
    db.commit()
    db.refresh(line)
    return line


def get_cached_line(db: Session, game_id: str, sportsbook: str,
                   market_type: str, player_id: Optional[str] = None) -> Optional[CachedLine]:
    """Get cached line if not expired."""
    query = db.query(CachedLine).filter(
        CachedLine.game_id == game_id,
        CachedLine.sportsbook == sportsbook,
        CachedLine.market_type == market_type,
        CachedLine.expires_at > datetime.now()
    )

    if player_id:
        query = query.filter(CachedLine.player_id == player_id)

    return query.order_by(CachedLine.fetched_at.desc()).first()


def cleanup_expired_lines(db: Session):
    """Remove expired cached lines."""
    db.query(CachedLine).filter(
        CachedLine.expires_at < datetime.now()
    ).delete()
    db.commit()


# =============================================================================
# Watchlist Operations
# =============================================================================

def add_to_watchlist(db: Session, user_id: uuid.UUID, game_id: str,
                    player_id: Optional[str] = None, prop_type: Optional[str] = None,
                    target_line: Optional[float] = None,
                    target_odds: Optional[float] = None,
                    notes: Optional[str] = None) -> UserWatchlist:
    """Add item to user's watchlist."""
    watchlist_item = UserWatchlist(
        user_id=user_id,
        game_id=game_id,
        player_id=player_id,
        prop_type=prop_type,
        target_line=target_line,
        target_odds=target_odds,
        notes=notes
    )
    db.add(watchlist_item)
    db.commit()
    db.refresh(watchlist_item)
    return watchlist_item


def get_user_watchlist(db: Session, user_id: uuid.UUID) -> List[UserWatchlist]:
    """Get user's watchlist."""
    return db.query(UserWatchlist).filter(
        UserWatchlist.user_id == user_id,
        UserWatchlist.alert_triggered == False
    ).all()


def remove_from_watchlist(db: Session, watchlist_id: uuid.UUID):
    """Remove item from watchlist."""
    db.query(UserWatchlist).filter(
        UserWatchlist.watchlist_id == watchlist_id
    ).delete()
    db.commit()
