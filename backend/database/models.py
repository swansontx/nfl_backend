"""SQLAlchemy database models"""
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Boolean,
    JSON,
    ForeignKey,
    Index,
    Enum as SQLEnum,
)
from sqlalchemy.orm import declarative_base, relationship
import enum

Base = declarative_base()


class PlayerStatus(str, enum.Enum):
    """Player roster status"""
    ACTIVE = "ACT"
    OUT = "OUT"
    INJURED_RESERVE = "IR"
    QUESTIONABLE = "Q"
    DOUBTFUL = "D"
    PROBABLE = "P"
    PRACTICE_SQUAD = "DEV"
    CUT = "CUT"
    RETIRED = "RET"


class Player(Base):
    """Canonical player table"""
    __tablename__ = "players"

    player_id = Column(String, primary_key=True)  # nflverse canonical ID
    gsis_id = Column(String, index=True, nullable=True)
    display_name = Column(String, nullable=False)
    first_name = Column(String)
    last_name = Column(String)
    position = Column(String, index=True)
    team = Column(String, index=True)
    college = Column(String, nullable=True)
    height = Column(Integer, nullable=True)
    weight = Column(Integer, nullable=True)
    birth_date = Column(DateTime, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    features = relationship("PlayerGameFeature", back_populates="player")
    projections = relationship("Projection", back_populates="player")
    outcomes = relationship("Outcome", back_populates="player")
    roster_status = relationship("RosterStatus", back_populates="player")

    __table_args__ = (
        Index("idx_player_name", "display_name"),
        Index("idx_player_team_position", "team", "position"),
    )


class Game(Base):
    """Game metadata table"""
    __tablename__ = "games"

    game_id = Column(String, primary_key=True)
    season = Column(Integer, nullable=False, index=True)
    week = Column(Integer, nullable=False, index=True)
    game_type = Column(String, nullable=False)  # REG, POST, PRE
    game_date = Column(DateTime, nullable=False)

    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)

    stadium = Column(String, nullable=True)
    roof = Column(String, nullable=True)  # dome, outdoors, open, closed
    surface = Column(String, nullable=True)
    temp = Column(Float, nullable=True)
    wind = Column(Float, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    features = relationship("PlayerGameFeature", back_populates="game")
    projections = relationship("Projection", back_populates="game")
    outcomes = relationship("Outcome", back_populates="game")
    roster_status = relationship("RosterStatus", back_populates="game")

    __table_args__ = (
        Index("idx_game_season_week", "season", "week"),
        Index("idx_game_date", "game_date"),
    )


class RosterStatus(Base):
    """Player roster and injury status per game"""
    __tablename__ = "roster_status"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(String, ForeignKey("players.player_id"), nullable=False)
    game_id = Column(String, ForeignKey("games.game_id"), nullable=False)

    status = Column(SQLEnum(PlayerStatus), nullable=False, default=PlayerStatus.ACTIVE)
    injury_type = Column(String, nullable=True)
    depth_chart_position = Column(Integer, nullable=True)

    # Confidence multiplier (0.0 - 1.0)
    confidence = Column(Float, default=1.0)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    player = relationship("Player", back_populates="roster_status")
    game = relationship("Game", back_populates="roster_status")

    __table_args__ = (
        Index("idx_roster_player_game", "player_id", "game_id"),
        Index("idx_roster_game_status", "game_id", "status"),
    )


class PlayerGameFeature(Base):
    """Extracted features per player per game"""
    __tablename__ = "player_game_features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(String, ForeignKey("players.player_id"), nullable=False)
    game_id = Column(String, ForeignKey("games.game_id"), nullable=False)

    # Snap counts and participation
    snaps = Column(Integer, nullable=True)
    snap_share = Column(Float, nullable=True)
    routes_run = Column(Integer, nullable=True)

    # Usage features
    targets = Column(Integer, default=0)
    receptions = Column(Integer, default=0)
    rush_attempts = Column(Integer, default=0)
    pass_attempts = Column(Integer, default=0)

    # Production features
    receiving_yards = Column(Float, default=0.0)
    rushing_yards = Column(Float, default=0.0)
    passing_yards = Column(Float, default=0.0)
    receiving_tds = Column(Integer, default=0)
    rushing_tds = Column(Integer, default=0)
    passing_tds = Column(Integer, default=0)

    # Advanced features
    redzone_targets = Column(Integer, default=0)
    redzone_carries = Column(Integer, default=0)
    goalline_carries = Column(Integer, default=0)
    air_yards_share = Column(Float, nullable=True)
    target_share = Column(Float, nullable=True)
    rush_share = Column(Float, nullable=True)

    # Rolling/smoothed features (JSON for flexibility)
    smoothed_features = Column(JSON, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    player = relationship("Player", back_populates="features")
    game = relationship("Game", back_populates="features")

    __table_args__ = (
        Index("idx_features_player_game", "player_id", "game_id"),
    )


class Projection(Base):
    """Model projections for player props"""
    __tablename__ = "projections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(String, ForeignKey("players.player_id"), nullable=False)
    game_id = Column(String, ForeignKey("games.game_id"), nullable=False)

    market = Column(String, nullable=False, index=True)

    # Model outputs
    mu = Column(Float, nullable=False)  # Expected value
    sigma = Column(Float, nullable=True)  # Standard deviation (if applicable)
    model_prob = Column(Float, nullable=True)  # Raw model probability
    calibrated_prob = Column(Float, nullable=True)  # Calibrated probability

    # Distribution parameters (JSON for flexibility)
    dist_params = Column(JSON, nullable=True)

    # Scoring and metadata
    usage_norm = Column(Float, nullable=True)
    volatility = Column(Float, nullable=True)
    score = Column(Float, nullable=True)
    tier = Column(String, nullable=True)  # Core, Mid, Lotto

    # Model metadata
    model_version = Column(String, nullable=True)
    features_version = Column(String, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    player = relationship("Player", back_populates="projections")
    game = relationship("Game", back_populates="projections")

    __table_args__ = (
        Index("idx_proj_player_game_market", "player_id", "game_id", "market"),
        Index("idx_proj_game_market_tier", "game_id", "market", "tier"),
    )


class Outcome(Base):
    """Actual outcomes for backtesting"""
    __tablename__ = "outcomes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(String, ForeignKey("players.player_id"), nullable=False)
    game_id = Column(String, ForeignKey("games.game_id"), nullable=False)

    market = Column(String, nullable=False, index=True)
    actual_value = Column(Float, nullable=False)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    player = relationship("Player", back_populates="outcomes")
    game = relationship("Game", back_populates="outcomes")

    __table_args__ = (
        Index("idx_outcome_player_game_market", "player_id", "game_id", "market"),
    )


class CalibrationParameter(Base):
    """Calibration parameters per market"""
    __tablename__ = "calibration_parameters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market = Column(String, nullable=False, index=True)
    season = Column(Integer, nullable=False)

    method = Column(String, nullable=False)  # platt, isotonic
    parameters = Column(JSON, nullable=False)

    # Performance metrics
    brier_score = Column(Float, nullable=True)
    log_loss = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)

    # Metadata
    trained_on_games = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_calib_market_season", "market", "season"),
    )
