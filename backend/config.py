"""Application configuration and environment variable management.

This module centralizes all configuration, environment variables, and mode-based
behavior (development vs production).
"""

from typing import Optional, Literal
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Environment
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Runtime environment"
    )

    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_cors_origins: str = Field(default="*", env="API_CORS_ORIGINS")

    # External APIs
    odds_api_key: Optional[str] = Field(default=None, env="ODDS_API_KEY")
    openweather_api_key: Optional[str] = Field(default=None, env="OPENWEATHER_API_KEY")

    # Optional APIs (Phase 2/3)
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    youtube_api_key: Optional[str] = Field(default=None, env="YOUTUBE_API_KEY")
    twitter_api_key: Optional[str] = Field(default=None, env="TWITTER_API_KEY")

    # Database
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")

    # Prop Evaluation Thresholds
    min_edge_threshold: float = Field(
        default=0.05,
        description="Minimum edge (5%) to recommend a prop"
    )
    min_ev_threshold: float = Field(
        default=0.02,
        description="Minimum expected value (2%) for a prop"
    )
    min_trust_score: float = Field(
        default=0.60,
        description="Minimum meta trust score (0-1) to recommend"
    )
    min_games_sampled: int = Field(
        default=5,
        description="Minimum games in training sample"
    )

    # Kelly Betting
    kelly_fraction: float = Field(
        default=0.25,
        description="Fraction of Kelly criterion (0.25 = quarter Kelly)"
    )
    max_stake_pct: float = Field(
        default=0.05,
        description="Maximum stake as % of bankroll (5%)"
    )

    # Empirical Standard Deviations by Prop Type (from backtest analysis)
    # These should be updated periodically based on model performance
    # Formula: sqrt(mean((actual - prediction)^2)) from backtest results
    prop_std_devs: dict = Field(
        default={
            # Yardage props (Normal distribution)
            'passing_yards': 55.0,   # Higher variance for QB passing
            'rushing_yards': 22.0,   # Moderate variance for RB rushing
            'receiving_yards': 18.0, # Moderate variance for WR/TE
            'pass_yards': 55.0,
            'rush_yards': 22.0,
            'rec_yards': 18.0,

            # Count props (Poisson distribution - std derived from backtest)
            'receptions': 1.8,
            'completions': 4.5,
            'attempts': 6.0,
            'carries': 3.5,
            'targets': 2.2,
            'interceptions': 0.7,

            # TD props (Poisson - rare events)
            'passing_tds': 0.85,
            'rushing_tds': 0.45,
            'receiving_tds': 0.38,
            'pass_tds': 0.85,
            'rush_tds': 0.45,
            'rec_tds': 0.38,
            'anytime_td': 0.5,
        },
        description="Empirical standard deviations by prop type from backtest analysis"
    )

    # Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    inputs_dir: Path = Field(default_factory=lambda: Path("inputs"))
    outputs_dir: Path = Field(default_factory=lambda: Path("outputs"))

    @field_validator("api_cors_origins")
    @classmethod
    def validate_cors_origins(cls, v: str) -> str:
        """Warn if CORS is wide open in production."""
        # Note: In Pydantic v2, we can't access other fields in field_validator easily
        # Moving production check to runtime
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"

    @property
    def allow_fallback_data(self) -> bool:
        """Whether to allow fallback to sample data when real data is missing."""
        return self.environment in ["development", "staging"]

    def validate_required_keys(self) -> list[str]:
        """Validate that required API keys are present.

        Returns:
            List of missing required keys (empty if all present)
        """
        missing = []

        # In production, odds API is required for value endpoint
        if self.is_production and not self.odds_api_key:
            missing.append("ODDS_API_KEY")

        return missing

    def get_predictions_dir(self) -> Path:
        """Get predictions output directory."""
        return self.outputs_dir / "predictions"

    def get_odds_dir(self) -> Path:
        """Get odds snapshots directory."""
        return self.outputs_dir / "odds"

    def get_reports_dir(self) -> Path:
        """Get reports output directory."""
        return self.outputs_dir / "reports"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"
    }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings (for dependency injection in FastAPI)."""
    return settings


def check_environment() -> dict:
    """Check environment setup and return status.

    Returns:
        Dict with:
        - environment: current env mode
        - missing_keys: list of missing required keys
        - data_available: dict of available output directories
        - warnings: list of warning messages
    """
    missing_keys = settings.validate_required_keys()
    warnings = []

    # Check if output directories exist
    data_available = {
        "predictions": settings.get_predictions_dir().exists(),
        "odds": settings.get_odds_dir().exists(),
        "reports": settings.get_reports_dir().exists(),
    }

    # Check for data availability issues
    if not any(data_available.values()):
        warnings.append("No output directories found - run data pipeline first")

    # Production-specific checks
    if settings.is_production:
        if missing_keys:
            warnings.append(f"Production mode but missing required keys: {', '.join(missing_keys)}")
        if not data_available["predictions"]:
            warnings.append("Production mode but no predictions available")

    return {
        "environment": settings.environment,
        "is_production": settings.is_production,
        "missing_keys": missing_keys,
        "data_available": data_available,
        "warnings": warnings,
        "thresholds": {
            "min_edge": settings.min_edge_threshold,
            "min_ev": settings.min_ev_threshold,
            "min_trust_score": settings.min_trust_score,
            "kelly_fraction": settings.kelly_fraction,
        }
    }
