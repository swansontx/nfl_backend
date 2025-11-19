"""
Signal effectiveness analysis for backtesting

Evaluates which signals contribute most to prediction accuracy:
- Trend signals (recent form, streaks)
- News sentiment
- Matchup features
- Injury/roster confidence

Helps optimize signal weights in recommendation system.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

from backend.config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SignalContribution:
    """Contribution of a single signal to overall performance"""
    signal_name: str

    # Standalone performance
    standalone_auc: float
    standalone_brier: float

    # Incremental value (how much it improves combined model)
    incremental_auc: float
    incremental_brier: float

    # Correlation with outcome
    correlation: float

    # Optimal weight (from regression)
    optimal_weight: float

    # Sample statistics
    mean_value: float
    std_value: float
    missing_rate: float


@dataclass
class SignalAnalysisResult:
    """Complete signal analysis"""
    signal_contributions: List[SignalContribution]

    # Combined model performance
    combined_auc: float
    combined_brier: float

    # Best signal
    best_signal_name: str
    best_signal_auc: float

    # Optimal weights
    optimal_weights: Dict[str, float]

    # Sample size
    n_samples: int


class SignalEffectivenessAnalyzer:
    """
    Analyzes effectiveness of different signals in prediction accuracy

    Useful for optimizing signal weights in recommendation system
    """

    def analyze_signals(
        self,
        backtest_results: List[Dict],
        signal_columns: Optional[List[str]] = None
    ) -> SignalAnalysisResult:
        """
        Analyze signal effectiveness from backtest results

        Args:
            backtest_results: List of dicts with predictions, outcomes, and signals
            signal_columns: List of signal column names to analyze

        Returns:
            SignalAnalysisResult with contribution analysis
        """
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(backtest_results)

        if 'outcome_binary' not in df.columns:
            logger.error("no_outcome_binary_in_results")
            raise ValueError("backtest_results must contain 'outcome_binary' column")

        # Auto-detect signal columns if not provided
        if signal_columns is None:
            signal_columns = [
                col for col in df.columns
                if col.endswith('_signal') or col in [
                    'trend_signal', 'news_signal', 'matchup_signal',
                    'roster_signal', 'base_signal'
                ]
            ]

        if not signal_columns:
            logger.warning("no_signal_columns_found")
            return None

        logger.info(
            "analyzing_signals",
            signals=signal_columns,
            samples=len(df)
        )

        # Get outcomes
        outcomes = df['outcome_binary'].values

        # Analyze each signal
        contributions = []

        for signal_name in signal_columns:
            if signal_name not in df.columns:
                continue

            signal_values = df[signal_name].values

            # Skip if all missing
            valid_mask = ~np.isnan(signal_values)
            if not valid_mask.any():
                continue

            contrib = self._analyze_single_signal(
                signal_name=signal_name,
                signal_values=signal_values,
                outcomes=outcomes,
                valid_mask=valid_mask
            )

            contributions.append(contrib)

        # Analyze combined performance
        combined_auc, combined_brier = self._analyze_combined_signals(
            df, signal_columns, outcomes
        )

        # Find best signal
        contributions.sort(key=lambda x: x.standalone_auc, reverse=True)
        best_signal = contributions[0] if contributions else None

        # Calculate optimal weights using logistic regression
        optimal_weights = self._calculate_optimal_weights(
            df, signal_columns, outcomes
        )

        result = SignalAnalysisResult(
            signal_contributions=contributions,
            combined_auc=combined_auc,
            combined_brier=combined_brier,
            best_signal_name=best_signal.signal_name if best_signal else None,
            best_signal_auc=best_signal.standalone_auc if best_signal else 0,
            optimal_weights=optimal_weights,
            n_samples=len(df)
        )

        logger.info(
            "signal_analysis_complete",
            best_signal=result.best_signal_name,
            combined_auc=combined_auc
        )

        return result

    def _analyze_single_signal(
        self,
        signal_name: str,
        signal_values: np.ndarray,
        outcomes: np.ndarray,
        valid_mask: np.ndarray
    ) -> SignalContribution:
        """Analyze effectiveness of a single signal"""
        # Filter to valid values
        valid_signals = signal_values[valid_mask]
        valid_outcomes = outcomes[valid_mask]

        # Standalone performance
        try:
            standalone_auc = roc_auc_score(valid_outcomes, valid_signals)
        except ValueError:
            standalone_auc = 0.5  # No discrimination

        try:
            standalone_brier = brier_score_loss(valid_outcomes, valid_signals)
        except ValueError:
            standalone_brier = 0.25

        # Correlation with outcome
        correlation = np.corrcoef(valid_signals, valid_outcomes)[0, 1]

        # Statistics
        mean_value = float(np.mean(valid_signals))
        std_value = float(np.std(valid_signals))
        missing_rate = float(1 - valid_mask.mean())

        return SignalContribution(
            signal_name=signal_name,
            standalone_auc=standalone_auc,
            standalone_brier=standalone_brier,
            incremental_auc=0.0,  # Will be calculated later
            incremental_brier=0.0,
            correlation=correlation,
            optimal_weight=0.0,  # Will be calculated later
            mean_value=mean_value,
            std_value=std_value,
            missing_rate=missing_rate
        )

    def _analyze_combined_signals(
        self,
        df: pd.DataFrame,
        signal_columns: List[str],
        outcomes: np.ndarray
    ) -> Tuple[float, float]:
        """Analyze performance when all signals are combined"""
        # Build feature matrix
        X = df[signal_columns].fillna(0.5).values  # Fill missing with neutral

        if X.shape[0] < 10:
            return 0.5, 0.25

        try:
            # Train simple logistic regression
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X, outcomes)

            # Predict probabilities
            probs = lr.predict_proba(X)[:, 1]

            # Calculate metrics
            combined_auc = roc_auc_score(outcomes, probs)
            combined_brier = brier_score_loss(outcomes, probs)

            return combined_auc, combined_brier

        except Exception as e:
            logger.error("combined_analysis_failed", error=str(e))
            return 0.5, 0.25

    def _calculate_optimal_weights(
        self,
        df: pd.DataFrame,
        signal_columns: List[str],
        outcomes: np.ndarray
    ) -> Dict[str, float]:
        """Calculate optimal weights for signals using logistic regression"""
        # Build feature matrix
        X = df[signal_columns].fillna(0.5).values

        if X.shape[0] < 10:
            return {col: 1.0 / len(signal_columns) for col in signal_columns}

        try:
            # Train logistic regression
            lr = LogisticRegression(max_iter=1000, random_state=42, penalty='l2')
            lr.fit(X, outcomes)

            # Get coefficients
            coefs = lr.coef_[0]

            # Normalize to sum to 1.0
            coef_sum = np.abs(coefs).sum()
            if coef_sum > 0:
                normalized_weights = np.abs(coefs) / coef_sum
            else:
                normalized_weights = np.ones(len(coefs)) / len(coefs)

            # Build dict
            weights = {
                col: float(weight)
                for col, weight in zip(signal_columns, normalized_weights)
            }

            return weights

        except Exception as e:
            logger.error("optimal_weights_failed", error=str(e))
            return {col: 1.0 / len(signal_columns) for col in signal_columns}

    def compare_signal_sets(
        self,
        backtest_results: List[Dict],
        signal_set_a: List[str],
        signal_set_b: List[str]
    ) -> Dict[str, float]:
        """
        Compare two different signal sets

        Useful for A/B testing new signals

        Args:
            backtest_results: Backtest results with signals
            signal_set_a: First set of signal columns
            signal_set_b: Second set of signal columns

        Returns:
            Dict with comparison metrics
        """
        df = pd.DataFrame(backtest_results)
        outcomes = df['outcome_binary'].values

        # Performance of set A
        X_a = df[signal_set_a].fillna(0.5).values
        lr_a = LogisticRegression(max_iter=1000)
        lr_a.fit(X_a, outcomes)
        probs_a = lr_a.predict_proba(X_a)[:, 1]
        auc_a = roc_auc_score(outcomes, probs_a)
        brier_a = brier_score_loss(outcomes, probs_a)

        # Performance of set B
        X_b = df[signal_set_b].fillna(0.5).values
        lr_b = LogisticRegression(max_iter=1000)
        lr_b.fit(X_b, outcomes)
        probs_b = lr_b.predict_proba(X_b)[:, 1]
        auc_b = roc_auc_score(outcomes, probs_b)
        brier_b = brier_score_loss(outcomes, probs_b)

        return {
            'signal_set_a_auc': auc_a,
            'signal_set_a_brier': brier_a,
            'signal_set_b_auc': auc_b,
            'signal_set_b_brier': brier_b,
            'auc_improvement': auc_b - auc_a,
            'brier_improvement': brier_a - brier_b,  # Lower is better
        }

    def generate_signal_report(self, result: SignalAnalysisResult) -> str:
        """Generate formatted report of signal analysis"""
        lines = []
        lines.append("=" * 80)
        lines.append("SIGNAL EFFECTIVENESS ANALYSIS")
        lines.append("=" * 80)
        lines.append(f"Samples: {result.n_samples}")
        lines.append(f"Combined AUC: {result.combined_auc:.4f}")
        lines.append(f"Combined Brier: {result.combined_brier:.4f}")
        lines.append("")

        lines.append("SIGNAL RANKINGS (by standalone AUC):")
        lines.append("-" * 80)
        lines.append(f"{'Signal':<25} {'AUC':<10} {'Brier':<10} {'Corr':<10} {'Weight':<10}")
        lines.append("-" * 80)

        for contrib in result.signal_contributions:
            lines.append(
                f"{contrib.signal_name:<25} "
                f"{contrib.standalone_auc:<10.4f} "
                f"{contrib.standalone_brier:<10.4f} "
                f"{contrib.correlation:<10.4f} "
                f"{contrib.optimal_weight:<10.4f}"
            )

        lines.append("")
        lines.append("OPTIMAL WEIGHTS:")
        lines.append("-" * 80)

        for signal_name, weight in result.optimal_weights.items():
            lines.append(f"  {signal_name:<30} {weight:.4f}")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)
