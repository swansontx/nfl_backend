"""Backtest metrics calculation"""
from typing import List, Dict
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve

from backend.config.logging_config import get_logger

logger = get_logger(__name__)


class BacktestMetrics:
    """
    Calculate comprehensive backtest metrics

    Metrics:
    - Brier score: Mean squared error of probabilities
    - Log loss: Cross-entropy loss
    - ROC AUC: Area under ROC curve
    - Calibration curves: Expected vs observed probabilities
    - Hit rate: Percentage of correct predictions
    - Expected value accuracy: How well mu predicts actual
    """

    def calculate_all_metrics(self, matched_results: List[Dict]) -> Dict:
        """
        Calculate all metrics from matched results

        Args:
            matched_results: List of dicts with predictions and outcomes

        Returns:
            Dict with all calculated metrics
        """
        if not matched_results:
            logger.warning("no_results_for_metrics")
            return {}

        # Extract arrays
        model_probs = np.array([r.get('model_prob', 0) for r in matched_results])
        calibrated_probs = np.array([
            r.get('calibrated_prob') or r.get('model_prob', 0)
            for r in matched_results
        ])
        outcomes = np.array([r['outcome_binary'] for r in matched_results])
        predicted_values = np.array([r['mu'] for r in matched_results])
        actual_values = np.array([r['actual_value'] for r in matched_results])

        # Overall metrics
        overall = self._calculate_probability_metrics(
            calibrated_probs, outcomes, "overall"
        )

        # Model (uncalibrated) metrics
        model_metrics = self._calculate_probability_metrics(
            model_probs, outcomes, "model_uncalibrated"
        )

        # Value prediction metrics
        value_metrics = self._calculate_value_metrics(predicted_values, actual_values)

        # Per-market metrics
        per_market = {}
        markets = set(r['market'] for r in matched_results)

        for market in markets:
            market_results = [r for r in matched_results if r['market'] == market]
            if len(market_results) < 10:  # Skip if too few samples
                continue

            market_probs = np.array([
                r.get('calibrated_prob') or r.get('model_prob', 0)
                for r in market_results
            ])
            market_outcomes = np.array([r['outcome_binary'] for r in market_results])

            per_market[market] = self._calculate_probability_metrics(
                market_probs, market_outcomes, market
            )

        # Per-tier metrics (if available)
        per_tier = {}
        tiers = set(r.get('tier') for r in matched_results if r.get('tier'))

        for tier in tiers:
            tier_results = [r for r in matched_results if r.get('tier') == tier]
            if len(tier_results) < 10:
                continue

            tier_probs = np.array([
                r.get('calibrated_prob') or r.get('model_prob', 0)
                for r in tier_results
            ])
            tier_outcomes = np.array([r['outcome_binary'] for r in tier_results])

            per_tier[tier] = self._calculate_probability_metrics(
                tier_probs, tier_outcomes, tier
            )

        return {
            'overall': overall,
            'model_uncalibrated': model_metrics,
            'value_prediction': value_metrics,
            'per_market': per_market,
            'per_tier': per_tier,
            'sample_size': len(matched_results)
        }

    def _calculate_probability_metrics(
        self,
        predicted_probs: np.ndarray,
        outcomes: np.ndarray,
        label: str
    ) -> Dict:
        """Calculate probability-based metrics"""
        try:
            # Brier score
            brier = brier_score_loss(outcomes, predicted_probs)

            # Log loss
            logloss = log_loss(outcomes, predicted_probs)

            # ROC AUC (only if both classes present)
            if len(np.unique(outcomes)) > 1:
                roc_auc = roc_auc_score(outcomes, predicted_probs)
            else:
                roc_auc = None

            # Hit rate (predictions > 0.5 considered "over")
            predictions_binary = (predicted_probs > 0.5).astype(int)
            hit_rate = np.mean(predictions_binary == outcomes)

            # Calibration curve
            try:
                prob_true, prob_pred = calibration_curve(
                    outcomes,
                    predicted_probs,
                    n_bins=10,
                    strategy='uniform'
                )
                calibration_data = {
                    'prob_true': prob_true.tolist(),
                    'prob_pred': prob_pred.tolist()
                }
            except Exception:
                calibration_data = None

            logger.debug(
                "metrics_calculated",
                label=label,
                brier=brier,
                log_loss=logloss,
                roc_auc=roc_auc
            )

            return {
                'brier_score': float(brier),
                'log_loss': float(logloss),
                'roc_auc': float(roc_auc) if roc_auc is not None else None,
                'hit_rate': float(hit_rate),
                'calibration_curve': calibration_data,
                'sample_size': len(outcomes)
            }

        except Exception as e:
            logger.error("metrics_calculation_failed", label=label, error=str(e))
            return {}

    def _calculate_value_metrics(
        self,
        predicted_values: np.ndarray,
        actual_values: np.ndarray
    ) -> Dict:
        """Calculate metrics for value prediction accuracy"""
        try:
            # Mean absolute error
            mae = np.mean(np.abs(predicted_values - actual_values))

            # Root mean squared error
            rmse = np.sqrt(np.mean((predicted_values - actual_values) ** 2))

            # Mean error (bias)
            bias = np.mean(predicted_values - actual_values)

            # Coefficient of determination (R²)
            ss_res = np.sum((actual_values - predicted_values) ** 2)
            ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Hit rate for over/under (does actual match direction of prediction?)
            mean_actual = np.mean(actual_values)
            predicted_over = (predicted_values > mean_actual).astype(int)
            actual_over = (actual_values > mean_actual).astype(int)
            direction_accuracy = np.mean(predicted_over == actual_over)

            return {
                'mae': float(mae),
                'rmse': float(rmse),
                'bias': float(bias),
                'r_squared': float(r_squared),
                'direction_accuracy': float(direction_accuracy),
                'mean_predicted': float(np.mean(predicted_values)),
                'mean_actual': float(np.mean(actual_values)),
                'std_predicted': float(np.std(predicted_values)),
                'std_actual': float(np.std(actual_values))
            }

        except Exception as e:
            logger.error("value_metrics_failed", error=str(e))
            return {}

    def calculate_closing_line_value(
        self,
        model_probs: np.ndarray,
        opening_odds: np.ndarray,
        closing_odds: np.ndarray
    ) -> Dict:
        """
        Calculate Closing Line Value (CLV)

        CLV measures if model beats market closing odds

        Args:
            model_probs: Model probabilities
            opening_odds: Opening odds (American format)
            closing_odds: Closing odds (American format)

        Returns:
            Dict with CLV metrics
        """
        # Convert odds to probabilities
        opening_probs = self._odds_to_prob(opening_odds)
        closing_probs = self._odds_to_prob(closing_odds)

        # CLV = model_prob - closing_prob
        clv = model_probs - closing_probs

        # Positive CLV rate
        positive_clv_rate = np.mean(clv > 0)

        # Average CLV
        avg_clv = np.mean(clv)

        # CLV when model says bet (model_prob > opening_prob)
        bet_signals = model_probs > opening_probs
        if np.any(bet_signals):
            avg_clv_when_betting = np.mean(clv[bet_signals])
        else:
            avg_clv_when_betting = 0

        return {
            'avg_clv': float(avg_clv),
            'positive_clv_rate': float(positive_clv_rate),
            'avg_clv_when_betting': float(avg_clv_when_betting),
            'total_samples': len(model_probs)
        }

    def _odds_to_prob(self, odds: np.ndarray) -> np.ndarray:
        """Convert American odds to probabilities"""
        probs = np.zeros_like(odds, dtype=float)

        # Negative odds (favorites)
        neg_mask = odds < 0
        probs[neg_mask] = -odds[neg_mask] / (-odds[neg_mask] + 100)

        # Positive odds (underdogs)
        pos_mask = odds > 0
        probs[pos_mask] = 100 / (odds[pos_mask] + 100)

        return probs

    def plot_calibration_curve(self, calibration_data: Dict, save_path: str = None):
        """
        Plot calibration curve

        Args:
            calibration_data: Dict with 'prob_true' and 'prob_pred'
            save_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt

            prob_true = calibration_data['prob_true']
            prob_pred = calibration_data['prob_pred']

            plt.figure(figsize=(10, 10))
            plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
            plt.plot(prob_pred, prob_true, 's-', label='Model')

            plt.xlabel('Predicted probability')
            plt.ylabel('Observed frequency')
            plt.title('Calibration Curve')
            plt.legend()
            plt.grid(True)

            if save_path:
                plt.savefig(save_path)
                logger.info("calibration_plot_saved", path=save_path)
            else:
                plt.show()

            plt.close()

        except ImportError:
            logger.warning("matplotlib_not_available")
        except Exception as e:
            logger.error("plot_failed", error=str(e))
