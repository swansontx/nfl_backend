"""Configuration Package.

Validated weights and coefficients from historical backtesting.
"""

from backend.config.validated_weights import (
    INJURY_REDISTRIBUTION,
    DEFENSE_MATCHUP_ADJUSTMENTS,
    WEATHER_IMPACT,
    SITUATIONAL_ADJUSTMENTS,
    TREND_WEIGHTS,
    CONFIDENCE_ADJUSTMENTS,
    FEATURE_WEIGHTS,
    VALIDATION_STATUS,
    get_validated_weight,
    update_from_backtest_results
)

__all__ = [
    'INJURY_REDISTRIBUTION',
    'DEFENSE_MATCHUP_ADJUSTMENTS',
    'WEATHER_IMPACT',
    'SITUATIONAL_ADJUSTMENTS',
    'TREND_WEIGHTS',
    'CONFIDENCE_ADJUSTMENTS',
    'FEATURE_WEIGHTS',
    'VALIDATION_STATUS',
    'get_validated_weight',
    'update_from_backtest_results',
]
