"""SQLAlchemy ORM models for NFL Props database.

Defines database tables for:
- Users (authentication, profiles)
- Bet History (user bets and outcomes)
- Model Predictions (prediction tracking)
- Cached Lines (sportsbook odds caching)
- User Bankroll (performance tracking)
- User Watchlist (bet alerts)
"""

from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey, Date
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from backend.database import Base
import uuid


class User(Base):
    """User account model."""
    __tablename__ = 'users'

    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    subscription_tier = Column(String(20), default='free')  # free, basic, premium
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True))
    is_active = Column(Boolean, default=True)
    preferences = Column(JSONB, default={})

    # Relationships
    bets = relationship('BetHistory', back_populates='user', cascade='all, delete-orphan')
    bankroll = relationship('UserBankroll', back_populates='user', uselist=False)
    watchlist = relationship('UserWatchlist', back_populates='user', cascade='all, delete-orphan')

    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"


class BetHistory(Base):
    """Bet history model."""
    __tablename__ = 'bet_history'

    bet_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False, index=True)
    game_id = Column(String(50), nullable=False, index=True)
    player_id = Column(String(50), index=True)
    bet_type = Column(String(50), nullable=False)  # over, under, spread, moneyline
    prop_type = Column(String(50))  # passing_yards, rushing_yards, etc.
    line_value = Column(Float)
    odds = Column(Float)  # American odds
    stake = Column(Float, default=1.0)  # Units bet
    result = Column(String(20), index=True)  # win, loss, push, pending
    actual_value = Column(Float)  # Actual stat value
    payout = Column(Float)  # Profit/loss
    placed_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    settled_at = Column(DateTime(timezone=True))
    notes = Column(Text)
    metadata = Column(JSONB, default={})

    # Relationships
    user = relationship('User', back_populates='bets')

    def __repr__(self):
        return f"<BetHistory(game='{self.game_id}', type='{self.bet_type}', result='{self.result}')>"


class ModelPrediction(Base):
    """Model prediction model."""
    __tablename__ = 'model_predictions'

    prediction_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    game_id = Column(String(50), nullable=False, index=True)
    player_id = Column(String(50), nullable=False, index=True)
    prediction_type = Column(String(50), nullable=False, index=True)  # passing_yards, etc.
    predicted_value = Column(Float, nullable=False)
    confidence = Column(Float)  # 0.0 to 1.0
    vegas_line = Column(Float)
    model_edge = Column(Float)  # predicted_value - vegas_line
    actual_value = Column(Float)  # Filled after game
    prediction_error = Column(Float)  # abs(predicted - actual)
    model_version = Column(String(50))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    game_date = Column(Date)
    season = Column(Integer, index=True)
    week = Column(Integer, index=True)
    features = Column(JSONB, default={})

    def __repr__(self):
        return f"<ModelPrediction(game='{self.game_id}', player='{self.player_id}', type='{self.prediction_type}')>"


class CachedLine(Base):
    """Cached sportsbook line model."""
    __tablename__ = 'cached_lines'

    line_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    game_id = Column(String(50), nullable=False, index=True)
    player_id = Column(String(50), index=True)
    sportsbook = Column(String(50), nullable=False)  # draftkings, fanduel, etc.
    market_type = Column(String(50), nullable=False)  # player_prop, game_total, spread
    prop_type = Column(String(50))  # passing_yards, rushing_yards, etc.
    line_value = Column(Float)
    over_odds = Column(Float)
    under_odds = Column(Float)
    fetched_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    expires_at = Column(DateTime(timezone=True), index=True)
    metadata = Column(JSONB, default={})

    def __repr__(self):
        return f"<CachedLine(game='{self.game_id}', sportsbook='{self.sportsbook}')>"


class UserBankroll(Base):
    """User bankroll tracking model."""
    __tablename__ = 'user_bankroll'

    bankroll_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False, index=True)
    total_units = Column(Float, default=100.0)
    starting_units = Column(Float, default=100.0)
    total_bets = Column(Integer, default=0)
    winning_bets = Column(Integer, default=0)
    losing_bets = Column(Integer, default=0)
    push_bets = Column(Integer, default=0)
    total_profit = Column(Float, default=0.0)
    roi = Column(Float)  # Return on investment %
    win_rate = Column(Float)  # Win percentage
    last_bet_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship('User', back_populates='bankroll')

    def __repr__(self):
        return f"<UserBankroll(user_id='{self.user_id}', units={self.total_units}, roi={self.roi})>"


class UserWatchlist(Base):
    """User watchlist for bet alerts."""
    __tablename__ = 'user_watchlist'

    watchlist_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False, index=True)
    game_id = Column(String(50), index=True)
    player_id = Column(String(50))
    prop_type = Column(String(50))
    target_line = Column(Float)  # Alert when line reaches this
    target_odds = Column(Float)
    alert_triggered = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    notes = Column(Text)

    # Relationships
    user = relationship('User', back_populates='watchlist')

    def __repr__(self):
        return f"<UserWatchlist(user_id='{self.user_id}', game='{self.game_id}')>"
