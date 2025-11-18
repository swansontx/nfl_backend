"""Application settings using pydantic-settings"""
from pathlib import Path
from typing import List, Optional
from pydantic import SecretStr, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # App metadata
    app_name: str = "NFL Props Backend"
    app_version: str = "0.1.0"
    debug: bool = False

    # Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    inputs_dir: Path = Field(default_factory=lambda: Path("inputs"))
    outputs_dir: Path = Field(default_factory=lambda: Path("outputs"))
    cache_dir: Path = Field(default_factory=lambda: Path("cache"))
    data_cache_dir: Path = Field(default_factory=lambda: Path("data/cache"))

    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "nfl_props"
    postgres_password: SecretStr = SecretStr("changeme")
    postgres_db: str = "nfl_props"

    @property
    def database_url(self) -> str:
        """Construct database URL"""
        return (
            f"postgresql://{self.postgres_user}:"
            f"{self.postgres_password.get_secret_value()}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[SecretStr] = None

    @property
    def redis_url(self) -> str:
        """Construct Redis URL"""
        if self.redis_password:
            return (
                f"redis://:{self.redis_password.get_secret_value()}@"
                f"{self.redis_host}:{self.redis_port}/{self.redis_db}"
            )
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # API Keys
    odds_api_key: Optional[SecretStr] = None

    # Feature extraction parameters
    lookback_games: int = 8
    short_lookback_games: int = 3
    long_lookback_games: int = 16
    min_snap_count: int = 10
    min_games_for_model: int = 3

    # Smoothing parameters
    eb_alpha: float = 0.3  # Empirical Bayes weight
    rolling_alpha: float = 0.4  # Exponential decay for rolling averages

    # Injury adjustments
    questionable_multiplier: float = 0.6
    doubtful_multiplier: float = 0.3
    probable_multiplier: float = 0.9

    # Modeling parameters
    poisson_overdispersion_threshold: float = 1.5
    yards_distribution: str = "lognormal"  # or "normal"

    # Markets to support
    supported_markets: List[str] = Field(
        default_factory=lambda: [
            # Passing (QB)
            "player_pass_yds",
            "player_pass_tds",
            "player_pass_completions",
            "player_pass_attempts",
            "player_interceptions",
            "player_pass_longest",
            "player_300_pass_yds",

            # Rushing
            "player_rush_yds",
            "player_rush_attempts",
            "player_rush_tds",
            "player_rush_longest",
            "player_100_rush_yds",

            # Receiving
            "player_rec_yds",
            "player_receptions",
            "player_rec_tds",
            "player_rec_longest",
            "player_100_rec_yds",

            # Touchdowns
            "player_anytime_td",
            "player_first_td",
            "player_last_td",
            "player_total_tds",
            "player_2plus_tds",

            # Combo Markets
            "player_pass_rush_yds",
            "player_rec_rush_yds",

            # Kicking
            "player_fg_made",
            "player_fg_longest",
            "player_extra_points",

            # Defense
            "player_tackles_assists",
            "player_sacks",
            "player_def_interceptions",
        ]
    )

    # Scoring parameters
    core_picks_count: int = 8
    mid_picks_count: int = 12
    lotto_picks_count: int = 5
    min_edge_threshold: float = 0.03
    volatility_penalty_k: float = 0.5
    usage_weight_lambda: float = 0.2

    # Calibration
    calibration_method: str = "platt"  # or "isotonic"
    calibration_min_prob: float = 0.01
    calibration_max_prob: float = 0.99

    # Backtest
    backtest_seasons: List[int] = Field(default_factory=lambda: [2022, 2023, 2024])
    kelly_fraction: float = 0.25

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    admin_token: Optional[SecretStr] = None

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # or "console"

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        for directory in [
            self.data_dir,
            self.inputs_dir,
            self.outputs_dir,
            self.cache_dir,
            self.data_cache_dir,
        ]:
            Path(directory).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
