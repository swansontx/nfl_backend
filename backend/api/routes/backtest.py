"""Backtest API routes"""
from fastapi import APIRouter, HTTPException, Query
from datetime import datetime

from backend.api.models.backtest import (
    BacktestResponse,
    BacktestMetricsResponse,
    SignalAnalysisResponse,
    SignalContributionResponse,
)
from backend.backtest import BacktestEngine, SignalEffectivenessAnalyzer
from backend.database.session import get_db
from backend.database.models import Game
from backend.config.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/backtest", tags=["backtest"])


@router.get("/", response_model=BacktestResponse)
def run_backtest(
    season: int = Query(..., description="Season to backtest"),
    kelly_fraction: float = Query(0.25, ge=0.1, le=1.0, description="Kelly fraction for betting"),
):
    """
    Run historical backtest on a season

    Tests the system on past data to measure performance.
    Returns ROI, Sharpe ratio, calibration metrics, etc.
    """
    logger.info("run_backtest", season=season)

    # Get season date range
    with get_db() as session:
        games = (
            session.query(Game)
            .filter(Game.season == season)
            .order_by(Game.game_date)
            .all()
        )

        if not games:
            raise HTTPException(status_code=404, detail=f"No games found for season {season}")

        start_date = games[0].game_date
        end_date = games[-1].game_date

    # Run backtest
    try:
        engine = BacktestEngine()
        engine.kelly_fraction = kelly_fraction

        result = engine.run_backtest(
            start_date=start_date,
            end_date=end_date,
            simulate_betting=True
        )

        # Build response
        if result.strategy_results:
            strat = result.strategy_results

            metrics = BacktestMetricsResponse(
                brier_score=result.metrics.get("overall", {}).get("brier_score", 0),
                log_loss=result.metrics.get("overall", {}).get("log_loss", 0),
                roc_auc=result.metrics.get("overall", {}).get("roc_auc"),
                total_bets=strat["total_bets"],
                winning_bets=strat["winning_bets"],
                win_rate=strat["hit_rate"],
                roi_percent=strat["roi_percent"],
                sharpe_ratio=strat["sharpe_ratio"],
                max_drawdown_percent=strat["max_drawdown_percent"],
                initial_bankroll=strat["initial_bankroll"],
                final_bankroll=strat["final_bankroll"],
                total_profit=strat["total_profit"],
            )
        else:
            # No betting simulation
            metrics = BacktestMetricsResponse(
                brier_score=result.metrics.get("overall", {}).get("brier_score", 0),
                log_loss=result.metrics.get("overall", {}).get("log_loss", 0),
                roc_auc=result.metrics.get("overall", {}).get("roc_auc"),
                total_bets=0,
                winning_bets=0,
                win_rate=0,
                roi_percent=0,
                sharpe_ratio=0,
                max_drawdown_percent=0,
                initial_bankroll=0,
                final_bankroll=0,
                total_profit=0,
            )

        return BacktestResponse(
            start_date=result.start_date,
            end_date=result.end_date,
            total_games=result.total_games,
            total_projections=result.total_projections,
            markets_tested=result.markets_tested,
            metrics=metrics,
            kelly_fraction=kelly_fraction,
        )

    except Exception as e:
        logger.error("backtest_failed", season=season, error=str(e))
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")


@router.get("/signals", response_model=SignalAnalysisResponse)
def analyze_signals(
    season: int = Query(..., description="Season to analyze"),
):
    """
    Analyze signal effectiveness

    Shows which signals (trend, news, matchup, etc.) contribute most
    to prediction accuracy. Returns optimal weights.
    """
    logger.info("analyze_signals", season=season)

    # Run backtest to get results
    with get_db() as session:
        games = (
            session.query(Game)
            .filter(Game.season == season)
            .order_by(Game.game_date)
            .all()
        )

        if not games:
            raise HTTPException(status_code=404, detail=f"No games found for season {season}")

        start_date = games[0].game_date
        end_date = games[-1].game_date

    try:
        # Run backtest
        engine = BacktestEngine()
        result = engine.run_backtest(
            start_date=start_date,
            end_date=end_date,
            simulate_betting=False  # Don't need betting for signal analysis
        )

        if not result.projection_results:
            raise HTTPException(status_code=400, detail="No projection results to analyze")

        # Analyze signals
        analyzer = SignalEffectivenessAnalyzer()
        signal_result = analyzer.analyze_signals(
            backtest_results=result.projection_results
        )

        # Build response
        contributions = [
            SignalContributionResponse(
                signal_name=c.signal_name,
                standalone_auc=c.standalone_auc,
                standalone_brier=c.standalone_brier,
                correlation=c.correlation,
                optimal_weight=c.optimal_weight,
                mean_value=c.mean_value,
                std_value=c.std_value,
            )
            for c in signal_result.signal_contributions
        ]

        return SignalAnalysisResponse(
            signal_contributions=contributions,
            combined_auc=signal_result.combined_auc,
            combined_brier=signal_result.combined_brier,
            best_signal_name=signal_result.best_signal_name,
            best_signal_auc=signal_result.best_signal_auc,
            optimal_weights=signal_result.optimal_weights,
            n_samples=signal_result.n_samples,
        )

    except Exception as e:
        logger.error("signal_analysis_failed", season=season, error=str(e))
        raise HTTPException(status_code=500, detail=f"Signal analysis failed: {str(e)}")
