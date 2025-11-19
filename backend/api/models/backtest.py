"""Pydantic models for backtest responses"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime


class BacktestMetricsResponse(BaseModel):
    """Backtest performance metrics"""
    # Calibration
    brier_score: float = Field(..., description="Mean squared error (lower = better)")
    log_loss: float = Field(..., description="Cross-entropy loss (lower = better)")
    roc_auc: Optional[float] = Field(None, description="Discrimination ability (higher = better)")

    # Betting performance
    total_bets: int
    winning_bets: int
    win_rate: float = Field(..., ge=0, le=1)
    roi_percent: float = Field(..., description="Return on investment %")
    sharpe_ratio: float = Field(..., description="Risk-adjusted returns")
    max_drawdown_percent: float = Field(..., description="Maximum drawdown %")

    # Bankroll
    initial_bankroll: float
    final_bankroll: float
    total_profit: float


class BacktestResponse(BaseModel):
    """Full backtest results"""
    # Period
    start_date: datetime
    end_date: datetime
    total_games: int
    total_projections: int

    # Markets tested
    markets_tested: List[str]

    # Performance
    metrics: BacktestMetricsResponse

    # Metadata
    kelly_fraction: float = 0.25
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        schema_extra = {
            "example": {
                "start_date": "2024-09-01T00:00:00Z",
                "end_date": "2024-12-31T00:00:00Z",
                "total_games": 267,
                "total_projections": 8534,
                "markets_tested": ["player_rec_yds", "player_rush_yds", "player_pass_yds"],
                "metrics": {
                    "brier_score": 0.218,
                    "log_loss": 0.642,
                    "roc_auc": 0.627,
                    "total_bets": 247,
                    "winning_bets": 134,
                    "win_rate": 0.5425,
                    "roi_percent": 7.84,
                    "sharpe_ratio": 1.23,
                    "max_drawdown_percent": 18.35,
                    "initial_bankroll": 1000.0,
                    "final_bankroll": 1078.40,
                    "total_profit": 78.40
                },
                "kelly_fraction": 0.25
            }
        }


class SignalContributionResponse(BaseModel):
    """Signal effectiveness metrics"""
    signal_name: str
    standalone_auc: float
    standalone_brier: float
    correlation: float
    optimal_weight: float
    mean_value: float
    std_value: float


class SignalAnalysisResponse(BaseModel):
    """Signal effectiveness analysis"""
    signal_contributions: List[SignalContributionResponse]

    # Combined performance
    combined_auc: float
    combined_brier: float

    # Best signal
    best_signal_name: str
    best_signal_auc: float

    # Optimal weights
    optimal_weights: Dict[str, float]

    # Sample size
    n_samples: int

    class Config:
        schema_extra = {
            "example": {
                "signal_contributions": [
                    {
                        "signal_name": "base_signal",
                        "standalone_auc": 0.623,
                        "standalone_brier": 0.215,
                        "correlation": 0.342,
                        "optimal_weight": 0.350,
                        "mean_value": 0.625,
                        "std_value": 0.185
                    },
                    {
                        "signal_name": "trend_signal",
                        "standalone_auc": 0.582,
                        "standalone_brier": 0.229,
                        "correlation": 0.213,
                        "optimal_weight": 0.180,
                        "mean_value": 0.548,
                        "std_value": 0.223
                    }
                ],
                "combined_auc": 0.658,
                "combined_brier": 0.201,
                "best_signal_name": "base_signal",
                "best_signal_auc": 0.623,
                "optimal_weights": {
                    "base_signal": 0.350,
                    "trend_signal": 0.180,
                    "matchup_signal": 0.120,
                    "news_signal": 0.080,
                    "roster_signal": 0.060
                },
                "n_samples": 8534
            }
        }
