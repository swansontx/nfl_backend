"""
Calibration validation metrics and diagnostics

Evaluates calibration quality through:
- Reliability curves (calibration plots)
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Brier score decomposition
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass

from backend.config.logging_config import get_logger
from backend.database.session import get_db
from backend.database.models import Projection, Outcome, Game

logger = get_logger(__name__)


@dataclass
class CalibrationMetrics:
    """Comprehensive calibration metrics for a market"""
    market: str
    season: int

    # Core metrics
    brier_score: float
    log_loss: float
    expected_calibration_error: float
    maximum_calibration_error: float

    # Brier decomposition
    reliability: float  # Calibration error component
    resolution: float  # Discrimination component
    uncertainty: float  # Base rate component

    # Reliability curve data
    bin_edges: List[float]
    bin_counts: List[int]
    bin_accuracies: List[float]
    bin_confidences: List[float]

    n_samples: int

    def to_dict(self) -> Dict:
        return {
            'market': self.market,
            'season': self.season,
            'brier_score': self.brier_score,
            'log_loss': self.log_loss,
            'ece': self.expected_calibration_error,
            'mce': self.maximum_calibration_error,
            'reliability': self.reliability,
            'resolution': self.resolution,
            'uncertainty': self.uncertainty,
            'n_samples': self.n_samples
        }


class CalibrationValidator:
    """Validates and diagnoses calibration quality"""

    def __init__(self, n_bins: int = 10):
        """
        Initialize validator

        Args:
            n_bins: Number of bins for reliability curves
        """
        self.n_bins = n_bins

    def evaluate_market(
        self,
        market: str,
        season: int,
        use_calibrated: bool = True
    ) -> CalibrationMetrics:
        """
        Evaluate calibration for a market

        Args:
            market: Market name
            season: Season year
            use_calibrated: Whether to use calibrated_prob or model_prob

        Returns:
            CalibrationMetrics with all diagnostic info
        """
        logger.info("evaluating_calibration", market=market, season=season)

        # Load predictions and outcomes
        predictions, actuals = self._load_data(market, season, use_calibrated)

        if len(predictions) < 10:
            raise ValueError(f"Insufficient data for evaluation: {len(predictions)} samples")

        # Calculate reliability curve
        bin_edges, bin_counts, bin_accuracies, bin_confidences = self._compute_reliability_curve(
            predictions, actuals
        )

        # Calculate ECE and MCE
        ece = self._compute_expected_calibration_error(
            bin_counts, bin_accuracies, bin_confidences
        )
        mce = self._compute_maximum_calibration_error(
            bin_accuracies, bin_confidences
        )

        # Brier score and decomposition
        brier = self._compute_brier_score(predictions, actuals)
        reliability, resolution, uncertainty = self._brier_decomposition(
            predictions, actuals, bin_edges, bin_counts, bin_accuracies
        )

        # Log loss
        logloss = self._compute_log_loss(predictions, actuals)

        metrics = CalibrationMetrics(
            market=market,
            season=season,
            brier_score=brier,
            log_loss=logloss,
            expected_calibration_error=ece,
            maximum_calibration_error=mce,
            reliability=reliability,
            resolution=resolution,
            uncertainty=uncertainty,
            bin_edges=bin_edges,
            bin_counts=bin_counts,
            bin_accuracies=bin_accuracies,
            bin_confidences=bin_confidences,
            n_samples=len(predictions)
        )

        logger.info(
            "calibration_evaluated",
            market=market,
            ece=ece,
            mce=mce,
            brier=brier
        )

        return metrics

    def evaluate_all_markets(
        self,
        season: int,
        markets: Optional[List[str]] = None
    ) -> Dict[str, CalibrationMetrics]:
        """Evaluate all markets and return metrics dict"""
        from backend.config import settings

        markets = markets or settings.supported_markets
        results = {}

        for market in markets:
            try:
                metrics = self.evaluate_market(market, season)
                results[market] = metrics
            except Exception as e:
                logger.warning("market_evaluation_failed", market=market, error=str(e))

        return results

    def _load_data(
        self,
        market: str,
        season: int,
        use_calibrated: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load predictions and actual outcomes"""
        with get_db() as session:
            query = (
                session.query(Projection, Outcome)
                .join(
                    Outcome,
                    (Projection.player_id == Outcome.player_id) &
                    (Projection.game_id == Outcome.game_id) &
                    (Projection.market == Outcome.market)
                )
                .join(Game, Projection.game_id == Game.game_id)
                .filter(
                    Projection.market == market,
                    Game.season == season
                )
            )

            results = query.all()

            predictions = []
            actuals = []

            for proj, outcome in results:
                # Get probability (calibrated or raw)
                if use_calibrated and proj.calibrated_prob is not None:
                    prob = proj.calibrated_prob
                elif proj.model_prob is not None:
                    prob = proj.model_prob
                else:
                    continue  # Skip if no probability

                predictions.append(prob)

                # For binary outcomes, compare to threshold
                # Simplified: outcome > projection mean
                actual = 1 if outcome.actual_value > proj.mu else 0
                actuals.append(actual)

            return np.array(predictions), np.array(actuals)

    def _compute_reliability_curve(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> Tuple[List[float], List[int], List[float], List[float]]:
        """
        Compute reliability curve (calibration plot)

        Bins predictions and compares predicted probability to actual frequency

        Returns:
            (bin_edges, bin_counts, bin_accuracies, bin_confidences)
        """
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_counts = []
        bin_accuracies = []
        bin_confidences = []

        for i in range(self.n_bins):
            # Find predictions in this bin
            mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])

            # Handle last bin inclusive
            if i == self.n_bins - 1:
                mask = (predictions >= bin_edges[i]) & (predictions <= bin_edges[i + 1])

            bin_preds = predictions[mask]
            bin_actuals = actuals[mask]

            if len(bin_preds) > 0:
                bin_counts.append(len(bin_preds))
                bin_accuracies.append(bin_actuals.mean())  # Actual frequency
                bin_confidences.append(bin_preds.mean())  # Predicted probability
            else:
                bin_counts.append(0)
                bin_accuracies.append(0.0)
                bin_confidences.append((bin_edges[i] + bin_edges[i + 1]) / 2)

        return (
            bin_edges.tolist(),
            bin_counts,
            bin_accuracies,
            bin_confidences
        )

    def _compute_expected_calibration_error(
        self,
        bin_counts: List[int],
        bin_accuracies: List[float],
        bin_confidences: List[float]
    ) -> float:
        """
        Compute Expected Calibration Error (ECE)

        ECE = sum over bins: (count_bin / total) * |accuracy_bin - confidence_bin|

        Lower is better. ECE < 0.05 is well-calibrated.
        """
        total_count = sum(bin_counts)
        if total_count == 0:
            return 0.0

        ece = 0.0
        for count, acc, conf in zip(bin_counts, bin_accuracies, bin_confidences):
            if count > 0:
                ece += (count / total_count) * abs(acc - conf)

        return float(ece)

    def _compute_maximum_calibration_error(
        self,
        bin_accuracies: List[float],
        bin_confidences: List[float]
    ) -> float:
        """
        Compute Maximum Calibration Error (MCE)

        MCE = max over bins: |accuracy_bin - confidence_bin|

        Measures worst-case calibration error
        """
        errors = [abs(acc - conf) for acc, conf in zip(bin_accuracies, bin_confidences)]
        return float(max(errors)) if errors else 0.0

    def _compute_brier_score(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> float:
        """
        Compute Brier score

        BS = mean((prediction - actual)^2)

        Lower is better. BS in [0, 1]
        """
        from sklearn.metrics import brier_score_loss
        return float(brier_score_loss(actuals, predictions))

    def _brier_decomposition(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        bin_edges: List[float],
        bin_counts: List[int],
        bin_accuracies: List[float]
    ) -> Tuple[float, float, float]:
        """
        Decompose Brier score into:
        - Reliability: Calibration error (lower is better)
        - Resolution: Ability to separate outcomes (higher is better)
        - Uncertainty: Inherent difficulty (depends on base rate)

        Brier = Reliability - Resolution + Uncertainty
        """
        n = len(predictions)
        overall_mean = actuals.mean()

        # Reliability: weighted squared difference between bin accuracy and confidence
        reliability = 0.0
        resolution = 0.0

        for i in range(len(bin_counts)):
            if bin_counts[i] == 0:
                continue

            # Get predictions in this bin
            mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
            if i == len(bin_counts) - 1:
                mask = (predictions >= bin_edges[i]) & (predictions <= bin_edges[i + 1])

            bin_preds = predictions[mask]
            bin_acts = actuals[mask]

            weight = len(bin_preds) / n
            bin_conf = bin_preds.mean()
            bin_acc = bin_acts.mean()

            # Reliability term
            reliability += weight * (bin_conf - bin_acc) ** 2

            # Resolution term
            resolution += weight * (bin_acc - overall_mean) ** 2

        # Uncertainty term
        uncertainty = overall_mean * (1 - overall_mean)

        return float(reliability), float(resolution), float(uncertainty)

    def _compute_log_loss(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> float:
        """Compute log loss (cross-entropy)"""
        from sklearn.metrics import log_loss
        # Clip to avoid log(0)
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        return float(log_loss(actuals, predictions))

    def generate_report(
        self,
        season: int,
        output_format: str = 'text'
    ) -> str:
        """
        Generate calibration report for all markets

        Args:
            season: Season year
            output_format: 'text' or 'markdown'

        Returns:
            Formatted report string
        """
        metrics_dict = self.evaluate_all_markets(season)

        if output_format == 'markdown':
            return self._format_markdown_report(metrics_dict, season)
        else:
            return self._format_text_report(metrics_dict, season)

    def _format_text_report(
        self,
        metrics_dict: Dict[str, CalibrationMetrics],
        season: int
    ) -> str:
        """Format as plain text report"""
        lines = [
            f"Calibration Report - {season} Season",
            "=" * 60,
            ""
        ]

        # Summary table
        lines.append(f"{'Market':<30} {'ECE':>8} {'MCE':>8} {'Brier':>8} {'Samples':>8}")
        lines.append("-" * 60)

        for market, metrics in sorted(metrics_dict.items()):
            lines.append(
                f"{market:<30} "
                f"{metrics.expected_calibration_error:>8.4f} "
                f"{metrics.maximum_calibration_error:>8.4f} "
                f"{metrics.brier_score:>8.4f} "
                f"{metrics.n_samples:>8}"
            )

        # Overall statistics
        lines.append("")
        lines.append("Overall Statistics:")
        avg_ece = np.mean([m.expected_calibration_error for m in metrics_dict.values()])
        avg_brier = np.mean([m.brier_score for m in metrics_dict.values()])
        lines.append(f"  Average ECE: {avg_ece:.4f}")
        lines.append(f"  Average Brier: {avg_brier:.4f}")

        return "\n".join(lines)

    def _format_markdown_report(
        self,
        metrics_dict: Dict[str, CalibrationMetrics],
        season: int
    ) -> str:
        """Format as markdown report"""
        lines = [
            f"# Calibration Report - {season} Season",
            "",
            "## Summary",
            "",
            "| Market | ECE | MCE | Brier | Samples |",
            "|--------|-----|-----|-------|---------|"
        ]

        for market, metrics in sorted(metrics_dict.items()):
            lines.append(
                f"| {market} | {metrics.expected_calibration_error:.4f} | "
                f"{metrics.maximum_calibration_error:.4f} | {metrics.brier_score:.4f} | "
                f"{metrics.n_samples} |"
            )

        lines.append("")
        lines.append("## Metrics Explanation")
        lines.append("")
        lines.append("- **ECE**: Expected Calibration Error - measures overall calibration (lower is better)")
        lines.append("- **MCE**: Maximum Calibration Error - worst-case bin error (lower is better)")
        lines.append("- **Brier**: Brier Score - accuracy of probabilistic predictions (lower is better)")

        return "\n".join(lines)
