"""Backtest engine for evaluating model performance"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np

from backend.config import settings
from backend.config.logging_config import get_logger
from backend.database.session import get_db
from backend.database.models import Projection, Outcome, Game
from backend.models.prop_models import PropModelRunner
from .metrics import BacktestMetrics

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """Result of a backtest run"""
    start_date: datetime
    end_date: datetime
    total_games: int
    total_projections: int
    markets_tested: List[str]

    # Performance metrics
    metrics: Dict[str, any]

    # Strategy performance (if betting simulation included)
    strategy_results: Optional[Dict] = None

    # Detailed results
    projection_results: List[Dict] = field(default_factory=list)


class BacktestEngine:
    """
    Backtest engine for model validation

    Runs model on historical games and evaluates:
    1. Probability calibration (Brier, log loss, ROC AUC)
    2. Expected value accuracy
    3. Betting strategy performance (Kelly criterion)
    4. Closing line value (CLV)
    """

    def __init__(self):
        self.model_runner = PropModelRunner()
        self.metrics_calculator = BacktestMetrics()
        self.kelly_fraction = settings.kelly_fraction

    def run_backtest(
        self,
        start_date: datetime,
        end_date: datetime,
        markets: Optional[List[str]] = None,
        simulate_betting: bool = True
    ) -> BacktestResult:
        """
        Run backtest over a date range

        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            markets: Markets to test (None = all)
            simulate_betting: Whether to simulate betting with Kelly criterion

        Returns:
            BacktestResult with comprehensive metrics
        """
        logger.info("starting_backtest", start=start_date, end=end_date)

        markets = markets or settings.supported_markets

        # Get historical games in range
        with get_db() as session:
            games = (
                session.query(Game)
                .filter(
                    Game.game_date >= start_date,
                    Game.game_date <= end_date,
                    Game.game_type == 'REG'  # Regular season only
                )
                .order_by(Game.game_date)
                .all()
            )

        logger.info("games_loaded", count=len(games))

        # Run model on each game and collect results
        all_projections = []
        all_outcomes = []

        for game in games:
            try:
                # Get projections (should already exist in DB from historical runs)
                # In production, you might re-run the model or use stored projections
                proj_results = self._get_game_projections(game.game_id, markets)
                outcome_results = self._get_game_outcomes(game.game_id, markets)

                all_projections.extend(proj_results)
                all_outcomes.extend(outcome_results)

            except Exception as e:
                logger.error("backtest_game_failed", game_id=game.game_id, error=str(e))

        logger.info(
            "projections_collected",
            projections=len(all_projections),
            outcomes=len(all_outcomes)
        )

        # Match projections with outcomes
        matched_results = self._match_projections_outcomes(all_projections, all_outcomes)

        # Calculate metrics
        metrics = self.metrics_calculator.calculate_all_metrics(matched_results)

        # Simulate betting if requested
        strategy_results = None
        if simulate_betting:
            strategy_results = self._simulate_betting(matched_results)

        result = BacktestResult(
            start_date=start_date,
            end_date=end_date,
            total_games=len(games),
            total_projections=len(all_projections),
            markets_tested=markets,
            metrics=metrics,
            strategy_results=strategy_results,
            projection_results=matched_results
        )

        logger.info("backtest_complete", games=len(games), projections=len(matched_results))

        return result

    def run_rolling_backtest(
        self,
        seasons: List[int],
        window_weeks: int = 4
    ) -> List[BacktestResult]:
        """
        Run rolling window backtest across seasons

        Useful for detecting model drift over time

        Args:
            seasons: List of seasons to test
            window_weeks: Size of rolling window in weeks

        Returns:
            List of BacktestResults, one per window
        """
        results = []

        with get_db() as session:
            for season in seasons:
                # Get all weeks in season
                games = (
                    session.query(Game)
                    .filter(
                        Game.season == season,
                        Game.game_type == 'REG'
                    )
                    .order_by(Game.week)
                    .all()
                )

                max_week = max(g.week for g in games) if games else 0

                # Run backtest for each window
                for start_week in range(1, max_week - window_weeks + 2):
                    end_week = start_week + window_weeks - 1

                    window_games = [g for g in games if start_week <= g.week <= end_week]

                    if not window_games:
                        continue

                    result = self.run_backtest(
                        start_date=min(g.game_date for g in window_games),
                        end_date=max(g.game_date for g in window_games),
                        simulate_betting=True
                    )

                    results.append(result)

                    logger.info(
                        "rolling_window_complete",
                        season=season,
                        weeks=f"{start_week}-{end_week}"
                    )

        return results

    def _get_game_projections(self, game_id: str, markets: List[str]) -> List[Dict]:
        """Get projections for a game from database"""
        with get_db() as session:
            projections = (
                session.query(Projection)
                .filter(
                    Projection.game_id == game_id,
                    Projection.market.in_(markets),
                    Projection.model_prob.isnot(None)
                )
                .all()
            )

            return [
                {
                    'player_id': p.player_id,
                    'game_id': p.game_id,
                    'market': p.market,
                    'mu': p.mu,
                    'model_prob': p.model_prob,
                    'calibrated_prob': p.calibrated_prob,
                    'score': p.score,
                    'tier': p.tier
                }
                for p in projections
            ]

    def _get_game_outcomes(self, game_id: str, markets: List[str]) -> List[Dict]:
        """Get actual outcomes for a game from database"""
        with get_db() as session:
            outcomes = (
                session.query(Outcome)
                .filter(
                    Outcome.game_id == game_id,
                    Outcome.market.in_(markets)
                )
                .all()
            )

            return [
                {
                    'player_id': o.player_id,
                    'game_id': o.game_id,
                    'market': o.market,
                    'actual_value': o.actual_value
                }
                for o in outcomes
            ]

    def _match_projections_outcomes(
        self,
        projections: List[Dict],
        outcomes: List[Dict]
    ) -> List[Dict]:
        """Match projections with their actual outcomes"""
        # Build outcome lookup
        outcome_lookup = {
            (o['player_id'], o['game_id'], o['market']): o
            for o in outcomes
        }

        matched = []

        for proj in projections:
            key = (proj['player_id'], proj['game_id'], proj['market'])
            outcome = outcome_lookup.get(key)

            if outcome:
                matched.append({
                    **proj,
                    'actual_value': outcome['actual_value'],
                    # Binary outcome (did actual exceed projection?)
                    'outcome_binary': 1 if outcome['actual_value'] > proj['mu'] else 0
                })

        return matched

    def _simulate_betting(self, matched_results: List[Dict]) -> Dict:
        """
        Simulate betting strategy using Kelly criterion

        Args:
            matched_results: List of matched projection/outcome pairs

        Returns:
            Dict with strategy performance metrics
        """
        # Initial bankroll
        bankroll = 1000.0
        bankroll_history = [bankroll]

        # Track bets
        total_bets = 0
        winning_bets = 0
        total_wagered = 0
        total_profit = 0

        # Market odds (simplified - assume -110 for all)
        market_prob = 0.52  # Implied prob of -110

        for result in matched_results:
            model_prob = result.get('calibrated_prob') or result.get('model_prob')
            if not model_prob:
                continue

            # Calculate edge
            edge = model_prob - market_prob

            # Only bet if positive edge
            if edge <= 0:
                continue

            # Kelly criterion bet sizing
            # Kelly = (edge * p) / (1 - p)
            # Use fractional Kelly to reduce variance
            kelly_bet = (edge * model_prob / (1 - model_prob))
            bet_size = bankroll * kelly_bet * self.kelly_fraction

            # Cap bet at 5% of bankroll
            bet_size = min(bet_size, bankroll * 0.05)

            # Skip if bet too small
            if bet_size < 1.0:
                continue

            total_bets += 1
            total_wagered += bet_size

            # Determine win/loss (simplified - assume -110 odds)
            if result['outcome_binary'] == 1:
                # Win
                profit = bet_size * (100 / 110)  # -110 odds payout
                winning_bets += 1
            else:
                # Loss
                profit = -bet_size

            total_profit += profit
            bankroll += profit
            bankroll_history.append(bankroll)

        # Calculate performance metrics
        if total_bets > 0:
            hit_rate = winning_bets / total_bets
            roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0

            # Sharpe ratio (simplified - daily returns)
            returns = np.diff(bankroll_history) / bankroll_history[:-1]
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if len(returns) > 1 else 0

            # Max drawdown
            peak = np.maximum.accumulate(bankroll_history)
            drawdown = (peak - bankroll_history) / peak
            max_drawdown = np.max(drawdown) * 100 if len(drawdown) > 0 else 0

        else:
            hit_rate = 0
            roi = 0
            sharpe = 0
            max_drawdown = 0

        return {
            'initial_bankroll': 1000.0,
            'final_bankroll': bankroll,
            'total_profit': total_profit,
            'total_bets': total_bets,
            'winning_bets': winning_bets,
            'hit_rate': hit_rate,
            'roi_percent': roi,
            'sharpe_ratio': sharpe,
            'max_drawdown_percent': max_drawdown,
            'bankroll_history': bankroll_history
        }

    def generate_backtest_report(self, result: BacktestResult) -> str:
        """Generate a formatted backtest report"""
        lines = []
        lines.append("="* 80)
        lines.append("BACKTEST REPORT")
        lines.append("=" * 80)
        lines.append(f"Period: {result.start_date.date()} to {result.end_date.date()}")
        lines.append(f"Games: {result.total_games}")
        lines.append(f"Projections: {result.total_projections}")
        lines.append(f"Markets: {', '.join(result.markets_tested)}")
        lines.append("")

        lines.append("CALIBRATION METRICS:")
        lines.append("-" * 80)
        metrics = result.metrics
        if 'overall' in metrics:
            overall = metrics['overall']
            lines.append(f"  Brier Score:  {overall.get('brier_score', 0):.4f}")
            lines.append(f"  Log Loss:     {overall.get('log_loss', 0):.4f}")
            lines.append(f"  ROC AUC:      {overall.get('roc_auc', 0):.4f}")
        lines.append("")

        if result.strategy_results:
            lines.append("BETTING SIMULATION (Kelly Criterion):")
            lines.append("-" * 80)
            strat = result.strategy_results
            lines.append(f"  Total Bets:    {strat['total_bets']}")
            lines.append(f"  Win Rate:      {strat['hit_rate']*100:.2f}%")
            lines.append(f"  ROI:           {strat['roi_percent']:.2f}%")
            lines.append(f"  Sharpe Ratio:  {strat['sharpe_ratio']:.2f}")
            lines.append(f"  Max Drawdown:  {strat['max_drawdown_percent']:.2f}%")
            lines.append(f"  Final P/L:     ${strat['total_profit']:.2f}")
            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)
