"""Probability calibration for prop predictions.

This module implements calibration techniques to convert raw model outputs
into well-calibrated probability estimates.

Key concepts:
1. **Isotonic Regression**: Non-parametric method that ensures calibrated probs
2. **Platt Scaling**: Logistic regression on model outputs
3. **Quantile Calibration**: Using historical quantiles to calibrate
4. **Per-Prop-Type Calibration**: Different props need different calibration

Well-calibrated probabilities mean:
- When model says 70% chance, it should hit ~70% of the time
- This is CRITICAL for Kelly betting and EV calculations
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import numpy as np
from datetime import datetime

# These will be imported conditionally for actual calibration
try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class CalibrationData:
    """Historical data for calibration."""
    predicted_probs: List[float] = field(default_factory=list)
    actual_outcomes: List[int] = field(default_factory=list)  # 1 = hit, 0 = miss
    timestamps: List[str] = field(default_factory=list)


@dataclass
class CalibrationParams:
    """Calibration parameters for a prop type."""
    prop_type: str
    method: str  # "isotonic", "platt", "quantile"
    params: Dict  # Method-specific parameters
    n_samples: int
    calibration_error: float  # Expected calibration error
    last_updated: str


class ProbabilityCalibrator:
    """Calibrate probability estimates using historical data."""

    def __init__(self, calibration_dir: str = None):
        if calibration_dir is None:
            # Use absolute path based on project root
            project_root = Path(__file__).parent.parent.parent
            calibration_dir = project_root / "outputs" / "calibration"
        self.calibration_dir = Path(calibration_dir)
        self.calibration_dir.mkdir(parents=True, exist_ok=True)

        # Store calibration data and models per prop type
        self.calibration_data: Dict[str, CalibrationData] = {}
        self.calibration_params: Dict[str, CalibrationParams] = {}
        self.isotonic_models: Dict[str, object] = {}  # Fitted isotonic models

        # Load existing calibration data
        self._load_calibration_data()
        self._load_calibration_params()

    def _load_calibration_data(self):
        """Load historical calibration data from disk."""
        data_file = self.calibration_dir / "calibration_history.json"

        if data_file.exists():
            try:
                with open(data_file, 'r') as f:
                    data = json.load(f)

                for prop_type, values in data.items():
                    self.calibration_data[prop_type] = CalibrationData(
                        predicted_probs=values.get('predicted_probs', []),
                        actual_outcomes=values.get('actual_outcomes', []),
                        timestamps=values.get('timestamps', [])
                    )

                print(f"Loaded calibration data for {len(self.calibration_data)} prop types")
            except Exception as e:
                print(f"Error loading calibration data: {e}")

    def _save_calibration_data(self):
        """Save calibration data to disk."""
        data_file = self.calibration_dir / "calibration_history.json"

        data = {}
        for prop_type, cal_data in self.calibration_data.items():
            data[prop_type] = {
                'predicted_probs': cal_data.predicted_probs,
                'actual_outcomes': cal_data.actual_outcomes,
                'timestamps': cal_data.timestamps
            }

        with open(data_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_calibration_params(self):
        """Load calibration parameters from disk."""
        params_file = self.calibration_dir / "calibration_params.json"

        if params_file.exists():
            try:
                with open(params_file, 'r') as f:
                    data = json.load(f)

                for prop_type, params in data.items():
                    self.calibration_params[prop_type] = CalibrationParams(
                        prop_type=prop_type,
                        method=params.get('method', 'isotonic'),
                        params=params.get('params', {}),
                        n_samples=params.get('n_samples', 0),
                        calibration_error=params.get('calibration_error', 0.0),
                        last_updated=params.get('last_updated', '')
                    )

                    # Rebuild isotonic model if params exist
                    if params.get('method') == 'isotonic' and 'x' in params.get('params', {}):
                        self._rebuild_isotonic_model(prop_type, params['params'])

            except Exception as e:
                print(f"Error loading calibration params: {e}")

    def _save_calibration_params(self):
        """Save calibration parameters to disk."""
        params_file = self.calibration_dir / "calibration_params.json"

        data = {}
        for prop_type, params in self.calibration_params.items():
            data[prop_type] = {
                'method': params.method,
                'params': params.params,
                'n_samples': params.n_samples,
                'calibration_error': params.calibration_error,
                'last_updated': params.last_updated
            }

        with open(params_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _rebuild_isotonic_model(self, prop_type: str, params: Dict):
        """Rebuild isotonic model from saved parameters."""
        if not SKLEARN_AVAILABLE:
            return

        if 'x' in params and 'y' in params:
            model = IsotonicRegression(out_of_bounds='clip')
            # Fit on the stored calibration curve
            model.fit(params['x'], params['y'])
            self.isotonic_models[prop_type] = model

    def add_outcome(
        self,
        prop_type: str,
        predicted_prob: float,
        actual_outcome: int,  # 1 = hit, 0 = miss
        timestamp: Optional[str] = None
    ):
        """Add a historical outcome for calibration.

        Args:
            prop_type: Type of prop (e.g., "passing_yards")
            predicted_prob: Model's predicted probability (0-1)
            actual_outcome: Whether prop hit (1) or missed (0)
            timestamp: When prediction was made
        """
        if prop_type not in self.calibration_data:
            self.calibration_data[prop_type] = CalibrationData()

        cal_data = self.calibration_data[prop_type]
        cal_data.predicted_probs.append(predicted_prob)
        cal_data.actual_outcomes.append(actual_outcome)
        cal_data.timestamps.append(timestamp or datetime.now().isoformat())

        # Save periodically (every 10 additions)
        if len(cal_data.predicted_probs) % 10 == 0:
            self._save_calibration_data()

    def fit_calibration(
        self,
        prop_type: str,
        method: str = "isotonic",
        min_samples: int = 30
    ) -> Optional[CalibrationParams]:
        """Fit calibration model for a prop type.

        Args:
            prop_type: Type of prop
            method: Calibration method ("isotonic", "platt", "quantile")
            min_samples: Minimum samples required

        Returns:
            CalibrationParams if successful, None otherwise
        """
        if prop_type not in self.calibration_data:
            print(f"No calibration data for {prop_type}")
            return None

        cal_data = self.calibration_data[prop_type]
        n_samples = len(cal_data.predicted_probs)

        if n_samples < min_samples:
            print(f"Insufficient samples for {prop_type}: {n_samples}/{min_samples}")
            return None

        probs = np.array(cal_data.predicted_probs)
        outcomes = np.array(cal_data.actual_outcomes)

        if method == "isotonic":
            params = self._fit_isotonic(prop_type, probs, outcomes)
        elif method == "platt":
            params = self._fit_platt(prop_type, probs, outcomes)
        elif method == "quantile":
            params = self._fit_quantile(prop_type, probs, outcomes)
        else:
            print(f"Unknown calibration method: {method}")
            return None

        # Calculate expected calibration error
        calibration_error = self._calculate_ece(probs, outcomes)

        cal_params = CalibrationParams(
            prop_type=prop_type,
            method=method,
            params=params,
            n_samples=n_samples,
            calibration_error=calibration_error,
            last_updated=datetime.now().isoformat()
        )

        self.calibration_params[prop_type] = cal_params
        self._save_calibration_params()

        return cal_params

    def _fit_isotonic(self, prop_type: str, probs: np.ndarray, outcomes: np.ndarray) -> Dict:
        """Fit isotonic regression calibration."""
        if not SKLEARN_AVAILABLE:
            # Fallback to simple binning
            return self._fit_quantile(prop_type, probs, outcomes)

        model = IsotonicRegression(out_of_bounds='clip')
        model.fit(probs, outcomes)

        self.isotonic_models[prop_type] = model

        # Store the calibration curve for serialization
        # Get unique x values and corresponding y values
        unique_x = np.unique(probs)
        calibrated_y = model.predict(unique_x)

        return {
            'x': unique_x.tolist(),
            'y': calibrated_y.tolist()
        }

    def _fit_platt(self, prop_type: str, probs: np.ndarray, outcomes: np.ndarray) -> Dict:
        """Fit Platt scaling (logistic regression on log-odds)."""
        if not SKLEARN_AVAILABLE:
            return self._fit_quantile(prop_type, probs, outcomes)

        # Convert to log-odds
        epsilon = 1e-7
        probs_clipped = np.clip(probs, epsilon, 1 - epsilon)
        log_odds = np.log(probs_clipped / (1 - probs_clipped))

        model = LogisticRegression()
        model.fit(log_odds.reshape(-1, 1), outcomes)

        return {
            'coef': model.coef_[0][0],
            'intercept': model.intercept_[0]
        }

    def _fit_quantile(self, prop_type: str, probs: np.ndarray, outcomes: np.ndarray) -> Dict:
        """Fit quantile-based calibration (binning)."""
        n_bins = min(10, len(probs) // 5)

        # Create bins
        bin_edges = np.linspace(0, 1, n_bins + 1)
        calibration_map = {}

        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
                actual_rate = np.mean(outcomes[mask])
                calibration_map[str(round(bin_center, 2))] = actual_rate

        return {
            'bin_edges': bin_edges.tolist(),
            'calibration_map': calibration_map
        }

    def _calculate_ece(self, probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_accuracy = np.mean(outcomes[mask])
                bin_confidence = np.mean(probs[mask])
                bin_size = np.sum(mask)
                ece += (bin_size / len(probs)) * abs(bin_accuracy - bin_confidence)

        return round(ece, 4)

    def calibrate(self, prop_type: str, raw_probability: float) -> float:
        """Calibrate a raw probability estimate.

        Args:
            prop_type: Type of prop
            raw_probability: Uncalibrated probability (0-1)

        Returns:
            Calibrated probability
        """
        # Clamp to valid range
        raw_probability = max(0.01, min(0.99, raw_probability))

        # If no calibration for this prop type, apply default adjustment
        if prop_type not in self.calibration_params:
            return self._apply_default_calibration(raw_probability)

        params = self.calibration_params[prop_type]

        if params.method == "isotonic":
            return self._apply_isotonic(prop_type, raw_probability)
        elif params.method == "platt":
            return self._apply_platt(params.params, raw_probability)
        elif params.method == "quantile":
            return self._apply_quantile(params.params, raw_probability)
        else:
            return raw_probability

    def _apply_default_calibration(self, prob: float) -> float:
        """Apply default calibration when no specific calibration exists.

        Default strategy: Shrink extreme probabilities toward 0.5
        This is conservative and reduces overconfidence.
        """
        # Shrinkage factor (higher = more conservative)
        shrinkage = 0.15

        # Shrink toward 0.5
        calibrated = prob * (1 - shrinkage) + 0.5 * shrinkage

        return calibrated

    def _apply_isotonic(self, prop_type: str, prob: float) -> float:
        """Apply isotonic regression calibration."""
        if prop_type in self.isotonic_models:
            return float(self.isotonic_models[prop_type].predict([[prob]])[0])

        # Fallback to interpolation from stored curve
        params = self.calibration_params[prop_type].params
        if 'x' in params and 'y' in params:
            x = np.array(params['x'])
            y = np.array(params['y'])
            return float(np.interp(prob, x, y))

        return prob

    def _apply_platt(self, params: Dict, prob: float) -> float:
        """Apply Platt scaling calibration."""
        epsilon = 1e-7
        prob_clipped = max(epsilon, min(1 - epsilon, prob))

        # Convert to log-odds
        log_odds = np.log(prob_clipped / (1 - prob_clipped))

        # Apply logistic transformation
        coef = params.get('coef', 1.0)
        intercept = params.get('intercept', 0.0)

        calibrated_log_odds = coef * log_odds + intercept
        calibrated_prob = 1 / (1 + np.exp(-calibrated_log_odds))

        return float(calibrated_prob)

    def _apply_quantile(self, params: Dict, prob: float) -> float:
        """Apply quantile-based calibration."""
        calibration_map = params.get('calibration_map', {})

        if not calibration_map:
            return prob

        # Find nearest bin center
        bin_centers = [float(k) for k in calibration_map.keys()]
        nearest = min(bin_centers, key=lambda x: abs(x - prob))

        return calibration_map[str(round(nearest, 2))]

    def get_calibration_report(self) -> Dict:
        """Get report on calibration status for all prop types."""
        report = {
            'total_prop_types': len(self.calibration_data),
            'calibrated_prop_types': len(self.calibration_params),
            'prop_types': {}
        }

        for prop_type, cal_data in self.calibration_data.items():
            n_samples = len(cal_data.predicted_probs)

            prop_report = {
                'n_samples': n_samples,
                'calibrated': prop_type in self.calibration_params
            }

            if prop_type in self.calibration_params:
                params = self.calibration_params[prop_type]
                prop_report.update({
                    'method': params.method,
                    'calibration_error': params.calibration_error,
                    'last_updated': params.last_updated
                })

            # Calculate raw accuracy if enough samples
            if n_samples >= 10:
                outcomes = np.array(cal_data.actual_outcomes)
                probs = np.array(cal_data.predicted_probs)

                # Accuracy for bets > 50% probability
                high_conf_mask = probs > 0.5
                if np.sum(high_conf_mask) > 0:
                    raw_accuracy = np.mean(outcomes[high_conf_mask])
                    prop_report['raw_accuracy'] = round(raw_accuracy, 3)

            report['prop_types'][prop_type] = prop_report

        return report

    def fit_all(self, min_samples: int = 30, method: str = "isotonic"):
        """Fit calibration for all prop types with sufficient data.

        Args:
            min_samples: Minimum samples required per prop type
            method: Calibration method to use
        """
        fitted = 0
        for prop_type in self.calibration_data.keys():
            result = self.fit_calibration(prop_type, method, min_samples)
            if result:
                fitted += 1
                print(f"  Fitted {prop_type}: ECE={result.calibration_error:.4f}, n={result.n_samples}")

        print(f"\nFitted calibration for {fitted} prop types")


# Singleton instance
probability_calibrator = ProbabilityCalibrator()


def calibrate_probability(prop_type: str, raw_probability: float) -> float:
    """Convenience function to calibrate a probability.

    Args:
        prop_type: Type of prop (e.g., "passing_yards")
        raw_probability: Uncalibrated probability

    Returns:
        Calibrated probability
    """
    return probability_calibrator.calibrate(prop_type, raw_probability)
