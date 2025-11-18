"""Probability calibration using Platt scaling and isotonic regression"""
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from backend.config import settings
from backend.config.logging_config import get_logger
from backend.database.session import get_db
from backend.database.models import CalibrationParameter, Outcome, Projection, Game

logger = get_logger(__name__)


@dataclass
class CalibrationResult:
    """Result of calibration process"""
    method: str  # platt or isotonic
    market: str
    season: int

    # Calibration parameters
    parameters: Dict

    # Performance metrics
    brier_score: float
    log_loss: float
    roc_auc: Optional[float]

    # Metadata
    n_samples: int
    calibrator: any  # Fitted calibrator object


class ProbabilityCalibrator:
    """
    Calibrate model probabilities using historical outcomes

    Methods:
    - Platt Scaling: Logistic regression on model outputs
    - Isotonic Regression: Non-parametric monotonic transformation
    """

    def __init__(self):
        self.min_prob = settings.calibration_min_prob
        self.max_prob = settings.calibration_max_prob
        self.method = settings.calibration_method

    def calibrate_market(
        self,
        market: str,
        season: int,
        method: Optional[str] = None,
        save_to_db: bool = True
    ) -> CalibrationResult:
        """
        Calibrate probabilities for a specific market

        Args:
            market: Market name (e.g., player_rec_yds)
            season: Season year
            method: Calibration method ('platt' or 'isotonic')
            save_to_db: Whether to save parameters to database

        Returns:
            CalibrationResult with fitted calibrator
        """
        method = method or self.method

        logger.info("starting_calibration", market=market, season=season, method=method)

        # Load historical data
        model_probs, actual_outcomes = self._load_historical_data(market, season)

        if len(model_probs) < 50:
            raise ValueError(f"Insufficient data for calibration: {len(model_probs)} samples")

        # Fit calibrator
        if method == 'platt':
            result = self._fit_platt(market, season, model_probs, actual_outcomes)
        elif method == 'isotonic':
            result = self._fit_isotonic(market, season, model_probs, actual_outcomes)
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        # Save to database
        if save_to_db:
            self._save_calibration(result)

        logger.info(
            "calibration_complete",
            market=market,
            method=method,
            brier=result.brier_score,
            log_loss=result.log_loss
        )

        return result

    def apply_calibration(
        self,
        model_prob: float,
        market: str,
        season: int
    ) -> float:
        """
        Apply calibration to a model probability

        Args:
            model_prob: Raw model probability
            market: Market name
            season: Season year

        Returns:
            Calibrated probability
        """
        # Load calibration parameters
        calibration = self._load_calibration(market, season)

        if not calibration:
            logger.warning("no_calibration_found", market=market, season=season)
            return self._clip_probability(model_prob)

        # Apply calibration
        if calibration.method == 'platt':
            calibrated = self._apply_platt(model_prob, calibration.calibrator)
        elif calibration.method == 'isotonic':
            calibrated = self._apply_isotonic(model_prob, calibration.calibrator)
        else:
            calibrated = model_prob

        # Clip to valid range
        return self._clip_probability(calibrated)

    def batch_calibrate(
        self,
        model_probs: np.ndarray,
        market: str,
        season: int
    ) -> np.ndarray:
        """Calibrate multiple probabilities"""
        calibration = self._load_calibration(market, season)

        if not calibration:
            return np.clip(model_probs, self.min_prob, self.max_prob)

        if calibration.method == 'platt':
            calibrated = calibration.calibrator.predict_proba(
                model_probs.reshape(-1, 1)
            )[:, 1]
        elif calibration.method == 'isotonic':
            calibrated = calibration.calibrator.predict(model_probs)
        else:
            calibrated = model_probs

        return np.clip(calibrated, self.min_prob, self.max_prob)

    def _fit_platt(
        self,
        market: str,
        season: int,
        model_probs: np.ndarray,
        actual_outcomes: np.ndarray
    ) -> CalibrationResult:
        """
        Fit Platt scaling (logistic regression on model outputs)

        Trains: P_calibrated = sigmoid(a * logit(P_model) + b)
        """
        # Transform to logit space
        # Avoid log(0) and log(1)
        model_probs_clipped = np.clip(model_probs, 1e-6, 1 - 1e-6)
        logits = np.log(model_probs_clipped / (1 - model_probs_clipped))

        # Fit logistic regression
        calibrator = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='lbfgs',
            max_iter=1000
        )

        # Use cross-validation to get calibrated predictions
        calibrated_probs = cross_val_predict(
            calibrator,
            logits.reshape(-1, 1),
            actual_outcomes,
            cv=5,
            method='predict_proba'
        )[:, 1]

        # Fit on full data
        calibrator.fit(logits.reshape(-1, 1), actual_outcomes)

        # Calculate metrics
        brier = brier_score_loss(actual_outcomes, calibrated_probs)
        logloss = log_loss(actual_outcomes, calibrated_probs)

        try:
            roc_auc = roc_auc_score(actual_outcomes, calibrated_probs)
        except ValueError:
            roc_auc = None

        # Store parameters
        parameters = {
            'coef': float(calibrator.coef_[0][0]),
            'intercept': float(calibrator.intercept_[0])
        }

        return CalibrationResult(
            method='platt',
            market=market,
            season=season,
            parameters=parameters,
            brier_score=brier,
            log_loss=logloss,
            roc_auc=roc_auc,
            n_samples=len(model_probs),
            calibrator=calibrator
        )

    def _fit_isotonic(
        self,
        market: str,
        season: int,
        model_probs: np.ndarray,
        actual_outcomes: np.ndarray
    ) -> CalibrationResult:
        """
        Fit isotonic regression (non-parametric monotonic calibration)

        More flexible than Platt but requires more data
        """
        calibrator = IsotonicRegression(
            y_min=self.min_prob,
            y_max=self.max_prob,
            out_of_bounds='clip'
        )

        # Use cross-validation
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        calibrated_probs = np.zeros_like(model_probs)

        for train_idx, test_idx in kf.split(model_probs):
            cal_temp = IsotonicRegression(
                y_min=self.min_prob,
                y_max=self.max_prob,
                out_of_bounds='clip'
            )
            cal_temp.fit(model_probs[train_idx], actual_outcomes[train_idx])
            calibrated_probs[test_idx] = cal_temp.predict(model_probs[test_idx])

        # Fit on full data
        calibrator.fit(model_probs, actual_outcomes)

        # Calculate metrics
        brier = brier_score_loss(actual_outcomes, calibrated_probs)
        logloss = log_loss(actual_outcomes, calibrated_probs)

        try:
            roc_auc = roc_auc_score(actual_outcomes, calibrated_probs)
        except ValueError:
            roc_auc = None

        # Store interpolation points (simplified)
        parameters = {
            'n_points': len(calibrator.X_thresholds_),
            'method': 'isotonic'
        }

        return CalibrationResult(
            method='isotonic',
            market=market,
            season=season,
            parameters=parameters,
            brier_score=brier,
            log_loss=logloss,
            roc_auc=roc_auc,
            n_samples=len(model_probs),
            calibrator=calibrator
        )

    def _apply_platt(self, model_prob: float, calibrator) -> float:
        """Apply Platt calibration to a single probability"""
        # Transform to logit
        model_prob_clipped = np.clip(model_prob, 1e-6, 1 - 1e-6)
        logit = np.log(model_prob_clipped / (1 - model_prob_clipped))

        # Apply logistic regression
        calibrated = calibrator.predict_proba([[logit]])[0, 1]

        return float(calibrated)

    def _apply_isotonic(self, model_prob: float, calibrator) -> float:
        """Apply isotonic calibration to a single probability"""
        calibrated = calibrator.predict([model_prob])[0]
        return float(calibrated)

    def _load_historical_data(
        self,
        market: str,
        season: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load historical projections and outcomes for calibration

        Returns:
            (model_probs, actual_outcomes) as numpy arrays
        """
        with get_db() as session:
            # Get projections with outcomes
            query = (
                session.query(Projection, Outcome)
                .join(
                    Outcome,
                    (Projection.player_id == Outcome.player_id) &
                    (Projection.game_id == Outcome.game_id) &
                    (Projection.market == Outcome.market)
                )
                .join(
                    Game,
                    Projection.game_id == Game.game_id
                )
                .filter(
                    Projection.market == market,
                    Game.season == season,
                    Projection.model_prob.isnot(None)
                )
            )

            results = query.all()

            if not results:
                raise ValueError(f"No historical data found for {market} in {season}")

            model_probs = []
            actual_outcomes = []

            for proj, outcome in results:
                # For binary outcomes (over/under a line)
                # We need to determine if actual exceeded the threshold
                # This is simplified - in production, store the line used
                model_probs.append(proj.model_prob)

                # Simplified: use actual vs projection mu as binary outcome
                # In production, you'd store the actual line and compare
                actual_binary = 1 if outcome.actual_value > proj.mu else 0
                actual_outcomes.append(actual_binary)

            return np.array(model_probs), np.array(actual_outcomes)

    def _save_calibration(self, result: CalibrationResult) -> None:
        """Save calibration parameters to database"""
        with get_db() as session:
            # Check if exists
            existing = (
                session.query(CalibrationParameter)
                .filter(
                    CalibrationParameter.market == result.market,
                    CalibrationParameter.season == result.season
                )
                .first()
            )

            if existing:
                # Update
                existing.method = result.method
                existing.parameters = result.parameters
                existing.brier_score = result.brier_score
                existing.log_loss = result.log_loss
                existing.roc_auc = result.roc_auc
                existing.trained_on_games = result.n_samples
            else:
                # Create
                calib = CalibrationParameter(
                    market=result.market,
                    season=result.season,
                    method=result.method,
                    parameters=result.parameters,
                    brier_score=result.brier_score,
                    log_loss=result.log_loss,
                    roc_auc=result.roc_auc,
                    trained_on_games=result.n_samples
                )
                session.add(calib)

        logger.info("calibration_saved", market=result.market, season=result.season)

    def _load_calibration(self, market: str, season: int) -> Optional[CalibrationResult]:
        """Load calibration from database and reconstruct calibrator"""
        with get_db() as session:
            calib_param = (
                session.query(CalibrationParameter)
                .filter(
                    CalibrationParameter.market == market,
                    CalibrationParameter.season == season
                )
                .first()
            )

            if not calib_param:
                return None

            # Reconstruct calibrator from parameters
            if calib_param.method == 'platt':
                calibrator = LogisticRegression()
                calibrator.coef_ = np.array([[calib_param.parameters['coef']]])
                calibrator.intercept_ = np.array([calib_param.parameters['intercept']])
                calibrator.classes_ = np.array([0, 1])
            else:
                # For isotonic, we can't easily reconstruct without storing full calibration curve
                # In production, you'd serialize the calibrator object
                calibrator = None

            return CalibrationResult(
                method=calib_param.method,
                market=calib_param.market,
                season=calib_param.season,
                parameters=calib_param.parameters,
                brier_score=calib_param.brier_score,
                log_loss=calib_param.log_loss,
                roc_auc=calib_param.roc_auc,
                n_samples=calib_param.trained_on_games or 0,
                calibrator=calibrator
            )

    def _clip_probability(self, prob: float) -> float:
        """Clip probability to valid range"""
        return float(np.clip(prob, self.min_prob, self.max_prob))

    def calibrate_all_markets(
        self,
        season: int,
        markets: Optional[List[str]] = None
    ) -> List[CalibrationResult]:
        """
        Calibrate all markets for a season

        Args:
            season: Season year
            markets: List of markets (None = all supported)

        Returns:
            List of CalibrationResults
        """
        markets = markets or settings.supported_markets
        results = []

        for market in markets:
            try:
                result = self.calibrate_market(
                    market=market,
                    season=season,
                    save_to_db=True
                )
                results.append(result)
            except Exception as e:
                logger.error("calibration_failed", market=market, error=str(e))

        logger.info("calibration_batch_complete", count=len(results), season=season)

        return results
