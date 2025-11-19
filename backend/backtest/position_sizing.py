"""
Position sizing strategies for bankroll management

Strategies:
- Kelly Criterion: Optimal growth rate
- Fractional Kelly: Reduced variance
- Fixed percentage: Simple risk management
- Confidence-scaled: Size based on signal confidence
- Kelly with ceiling: Capped maximum bet size
"""

from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

from backend.config.logging_config import get_logger

logger = get_logger(__name__)


class SizingStrategy(Enum):
    """Position sizing strategy type"""
    KELLY = "kelly"
    FRACTIONAL_KELLY = "fractional_kelly"
    FIXED_PERCENTAGE = "fixed_percentage"
    CONFIDENCE_SCALED = "confidence_scaled"
    KELLY_WITH_CEIL = "kelly_with_ceil"


@dataclass
class PositionSize:
    """Calculated position size"""
    bet_amount: float  # Dollar amount to bet
    bet_fraction: float  # Fraction of bankroll
    edge: float  # Estimated edge
    confidence: float  # Confidence in bet (0-1)
    strategy: str  # Strategy used

    # Constraints applied
    capped: bool = False
    skip_reason: Optional[str] = None


class KellyCriterion:
    """
    Kelly Criterion position sizing

    Optimal bet size for maximum long-term growth:
    Kelly = (edge * p) / (1 - p)

    Where:
    - edge = model_prob - market_prob
    - p = model probability of winning
    """

    def calculate(
        self,
        model_prob: float,
        market_prob: float,
        bankroll: float,
        fraction: float = 1.0,
        max_bet_fraction: float = 0.05
    ) -> PositionSize:
        """
        Calculate Kelly bet size

        Args:
            model_prob: Model's probability estimate
            market_prob: Market's implied probability
            bankroll: Current bankroll
            fraction: Kelly fraction (0-1, lower = more conservative)
            max_bet_fraction: Maximum bet as fraction of bankroll

        Returns:
            PositionSize with bet recommendation
        """
        # Calculate edge
        edge = model_prob - market_prob

        # No bet if no edge
        if edge <= 0:
            return PositionSize(
                bet_amount=0,
                bet_fraction=0,
                edge=edge,
                confidence=0,
                strategy="kelly",
                skip_reason="NO_EDGE"
            )

        # Kelly formula
        kelly_fraction_raw = (edge * model_prob) / (1 - model_prob)

        # Apply fractional Kelly
        kelly_fraction_adj = kelly_fraction_raw * fraction

        # Cap at max bet size
        capped = False
        if kelly_fraction_adj > max_bet_fraction:
            kelly_fraction_adj = max_bet_fraction
            capped = True

        # Calculate dollar amount
        bet_amount = bankroll * kelly_fraction_adj

        # Skip if bet too small
        if bet_amount < 1.0:
            return PositionSize(
                bet_amount=0,
                bet_fraction=0,
                edge=edge,
                confidence=model_prob,
                strategy="kelly",
                skip_reason="BET_TOO_SMALL"
            )

        return PositionSize(
            bet_amount=bet_amount,
            bet_fraction=kelly_fraction_adj,
            edge=edge,
            confidence=model_prob,
            strategy=f"kelly_{fraction:.2f}",
            capped=capped
        )


class PositionSizer:
    """
    Flexible position sizing with multiple strategies

    Supports:
    - Kelly criterion (various fractions)
    - Fixed percentage
    - Confidence-scaled sizing
    - Multi-level bet sizing (small/medium/large)
    """

    def __init__(
        self,
        strategy: SizingStrategy = SizingStrategy.FRACTIONAL_KELLY,
        kelly_fraction: float = 0.25,
        fixed_percentage: float = 0.02,
        max_bet_fraction: float = 0.05,
        min_edge: float = 0.02,
        min_confidence: float = 0.55
    ):
        """
        Initialize position sizer

        Args:
            strategy: Sizing strategy to use
            kelly_fraction: Fraction for Kelly (0.25 = quarter Kelly)
            fixed_percentage: Fixed bet percentage
            max_bet_fraction: Maximum bet size
            min_edge: Minimum edge to place bet
            min_confidence: Minimum confidence to place bet
        """
        self.strategy = strategy
        self.kelly_fraction = kelly_fraction
        self.fixed_percentage = fixed_percentage
        self.max_bet_fraction = max_bet_fraction
        self.min_edge = min_edge
        self.min_confidence = min_confidence

        self.kelly_calculator = KellyCriterion()

    def calculate_position(
        self,
        model_prob: float,
        market_prob: float,
        bankroll: float,
        signal_confidence: Optional[float] = None
    ) -> PositionSize:
        """
        Calculate position size using configured strategy

        Args:
            model_prob: Model's probability
            market_prob: Market's implied probability
            bankroll: Current bankroll
            signal_confidence: Optional signal confidence (0-1)

        Returns:
            PositionSize recommendation
        """
        # Calculate edge
        edge = model_prob - market_prob

        # Skip if below thresholds
        if edge < self.min_edge:
            return PositionSize(
                bet_amount=0,
                bet_fraction=0,
                edge=edge,
                confidence=model_prob,
                strategy=self.strategy.value,
                skip_reason="INSUFFICIENT_EDGE"
            )

        if model_prob < self.min_confidence:
            return PositionSize(
                bet_amount=0,
                bet_fraction=0,
                edge=edge,
                confidence=model_prob,
                strategy=self.strategy.value,
                skip_reason="LOW_CONFIDENCE"
            )

        # Route to strategy
        if self.strategy == SizingStrategy.KELLY:
            return self.kelly_calculator.calculate(
                model_prob, market_prob, bankroll,
                fraction=1.0,
                max_bet_fraction=self.max_bet_fraction
            )

        elif self.strategy == SizingStrategy.FRACTIONAL_KELLY:
            return self.kelly_calculator.calculate(
                model_prob, market_prob, bankroll,
                fraction=self.kelly_fraction,
                max_bet_fraction=self.max_bet_fraction
            )

        elif self.strategy == SizingStrategy.FIXED_PERCENTAGE:
            return self._fixed_percentage_sizing(
                edge, model_prob, bankroll
            )

        elif self.strategy == SizingStrategy.CONFIDENCE_SCALED:
            return self._confidence_scaled_sizing(
                edge, model_prob, bankroll, signal_confidence
            )

        elif self.strategy == SizingStrategy.KELLY_WITH_CEIL:
            return self._kelly_with_ceiling(
                model_prob, market_prob, bankroll
            )

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _fixed_percentage_sizing(
        self,
        edge: float,
        model_prob: float,
        bankroll: float
    ) -> PositionSize:
        """Fixed percentage of bankroll"""
        bet_fraction = self.fixed_percentage
        bet_amount = bankroll * bet_fraction

        return PositionSize(
            bet_amount=bet_amount,
            bet_fraction=bet_fraction,
            edge=edge,
            confidence=model_prob,
            strategy="fixed_percentage"
        )

    def _confidence_scaled_sizing(
        self,
        edge: float,
        model_prob: float,
        bankroll: float,
        signal_confidence: Optional[float]
    ) -> PositionSize:
        """
        Scale bet size by signal confidence

        Higher confidence = larger bet
        """
        # Base size using fractional Kelly
        kelly_position = self.kelly_calculator.calculate(
            model_prob,
            1 - edge,  # Implied market prob
            bankroll,
            fraction=self.kelly_fraction,
            max_bet_fraction=self.max_bet_fraction
        )

        # Scale by confidence
        if signal_confidence:
            # Confidence boost (0.5 = no change, 1.0 = 2x, 0.0 = 0x)
            confidence_multiplier = signal_confidence * 2
            bet_amount = kelly_position.bet_amount * confidence_multiplier

            # Re-cap
            bet_fraction = bet_amount / bankroll
            if bet_fraction > self.max_bet_fraction:
                bet_fraction = self.max_bet_fraction
                bet_amount = bankroll * bet_fraction

            return PositionSize(
                bet_amount=bet_amount,
                bet_fraction=bet_fraction,
                edge=edge,
                confidence=signal_confidence,
                strategy="confidence_scaled"
            )
        else:
            return kelly_position

    def _kelly_with_ceiling(
        self,
        model_prob: float,
        market_prob: float,
        bankroll: float
    ) -> PositionSize:
        """
        Kelly with hard ceiling at max_bet_fraction

        More conservative than fractional Kelly
        """
        return self.kelly_calculator.calculate(
            model_prob,
            market_prob,
            bankroll,
            fraction=1.0,  # Full Kelly
            max_bet_fraction=self.max_bet_fraction  # But capped
        )

    def tiered_sizing(
        self,
        edge: float,
        model_prob: float,
        bankroll: float
    ) -> PositionSize:
        """
        Tiered bet sizing based on edge strength

        Strong edge = large bet
        Moderate edge = medium bet
        Weak edge = small bet
        """
        # Determine tier
        if edge >= 0.10:
            # Strong edge
            size_fraction = 0.05
            tier = "LARGE"
        elif edge >= 0.05:
            # Moderate edge
            size_fraction = 0.03
            tier = "MEDIUM"
        elif edge >= 0.02:
            # Weak edge
            size_fraction = 0.01
            tier = "SMALL"
        else:
            # No edge
            return PositionSize(
                bet_amount=0,
                bet_fraction=0,
                edge=edge,
                confidence=model_prob,
                strategy="tiered",
                skip_reason="NO_EDGE"
            )

        bet_amount = bankroll * size_fraction

        return PositionSize(
            bet_amount=bet_amount,
            bet_fraction=size_fraction,
            edge=edge,
            confidence=model_prob,
            strategy=f"tiered_{tier}"
        )

    def simulate_strategy(
        self,
        backtest_results: list,
        initial_bankroll: float = 1000.0
    ) -> Dict:
        """
        Simulate position sizing strategy on backtest results

        Args:
            backtest_results: List of matched projection/outcome pairs
            initial_bankroll: Starting bankroll

        Returns:
            Dict with simulation results
        """
        bankroll = initial_bankroll
        bankroll_history = [bankroll]

        total_bets = 0
        winning_bets = 0
        total_wagered = 0
        total_profit = 0

        bet_sizes = []

        for result in backtest_results:
            model_prob = result.get('calibrated_prob') or result.get('model_prob')
            if not model_prob:
                continue

            # Simplified market prob (assume -110)
            market_prob = 0.52

            # Calculate position
            position = self.calculate_position(
                model_prob=model_prob,
                market_prob=market_prob,
                bankroll=bankroll,
                signal_confidence=result.get('confidence')
            )

            # Skip if no bet
            if position.bet_amount == 0:
                continue

            total_bets += 1
            total_wagered += position.bet_amount
            bet_sizes.append(position.bet_amount)

            # Determine outcome (simplified -110 odds)
            if result.get('outcome_binary') == 1:
                # Win
                profit = position.bet_amount * (100 / 110)
                winning_bets += 1
            else:
                # Loss
                profit = -position.bet_amount

            total_profit += profit
            bankroll += profit
            bankroll_history.append(bankroll)

        # Calculate metrics
        if total_bets > 0:
            hit_rate = winning_bets / total_bets
            roi = (total_profit / total_wagered) * 100

            # Sharpe ratio
            returns = np.diff(bankroll_history) / np.array(bankroll_history[:-1])
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0

            # Max drawdown
            peak = np.maximum.accumulate(bankroll_history)
            drawdown = (peak - bankroll_history) / peak
            max_drawdown = np.max(drawdown) * 100

            avg_bet_size = np.mean(bet_sizes)
            max_bet_size = np.max(bet_sizes)

        else:
            hit_rate = 0
            roi = 0
            sharpe = 0
            max_drawdown = 0
            avg_bet_size = 0
            max_bet_size = 0

        return {
            'strategy': self.strategy.value,
            'initial_bankroll': initial_bankroll,
            'final_bankroll': bankroll,
            'total_profit': total_profit,
            'total_bets': total_bets,
            'winning_bets': winning_bets,
            'hit_rate': hit_rate,
            'roi_percent': roi,
            'sharpe_ratio': sharpe,
            'max_drawdown_percent': max_drawdown,
            'avg_bet_size': avg_bet_size,
            'max_bet_size': max_bet_size,
            'bankroll_history': bankroll_history
        }
