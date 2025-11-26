"""Historical Backtesting System.

Validates deep analysis features against historical NFL data.
"""

from backend.backtesting.framework import BacktestingFramework, BacktestResult
from backend.backtesting.data_collector import HistoricalDataCollector
from backend.backtesting.injury_impact_backtest import InjuryImpactBacktester
from backend.backtesting.defense_matchup_backtest import DefenseMatchupBacktester
from backend.backtesting.weather_impact_backtest import WeatherImpactBacktester
from backend.backtesting.situational_factors_backtest import SituationalFactorsBacktester
from backend.backtesting.overall_accuracy_backtest import OverallAccuracyBacktester
from backend.backtesting.player_props_backtest import PlayerPropsBacktester
from backend.backtesting.run_all_backtests import BacktestingOrchestrator

__all__ = [
    'BacktestingFramework',
    'BacktestResult',
    'HistoricalDataCollector',
    'InjuryImpactBacktester',
    'DefenseMatchupBacktester',
    'WeatherImpactBacktester',
    'SituationalFactorsBacktester',
    'OverallAccuracyBacktester',
    'PlayerPropsBacktester',
    'BacktestingOrchestrator',
]
