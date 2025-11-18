"""Statistical distribution models for props"""
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from backend.config import settings
from backend.config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DistributionParams:
    """Parameters for a probability distribution"""
    dist_type: str
    params: Dict
    mu: float  # Expected value
    sigma: Optional[float] = None  # Standard deviation (if applicable)


class PoissonModel:
    """
    Poisson model for count data (receptions, attempts, etc.)

    Falls back to Negative Binomial if overdispersion detected
    """

    def __init__(self):
        self.overdispersion_threshold = settings.poisson_overdispersion_threshold

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Fit Poisson or Negative Binomial model

        Args:
            X: Feature matrix
            y: Target counts

        Returns:
            Model parameters
        """
        # Check for overdispersion
        mean_y = np.mean(y)
        var_y = np.var(y)

        if var_y > mean_y * self.overdispersion_threshold:
            logger.info("overdispersion_detected", mean=mean_y, var=var_y)
            return self._fit_negative_binomial(X, y)
        else:
            return self._fit_poisson(X, y)

    def _fit_poisson(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Fit Poisson GLM"""
        # Simple Poisson: lambda = exp(X @ beta)
        # For now, use mean as baseline
        lambda_param = np.mean(y)

        return {
            'dist_type': 'poisson',
            'lambda': lambda_param
        }

    def _fit_negative_binomial(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Fit Negative Binomial"""
        # Fit NB parameters
        mean_y = np.mean(y)
        var_y = np.var(y)

        # Method of moments estimators
        if var_y > mean_y:
            r = mean_y ** 2 / (var_y - mean_y)
            p = mean_y / var_y
        else:
            r = mean_y
            p = 0.5

        return {
            'dist_type': 'nbinom',
            'n': r,
            'p': p,
            'mu': mean_y
        }

    def predict(self, params: Dict, features: Optional[np.ndarray] = None) -> DistributionParams:
        """
        Predict distribution for new observation

        Args:
            params: Model parameters
            features: Feature vector (optional, for future GLM)

        Returns:
            DistributionParams
        """
        dist_type = params['dist_type']

        if dist_type == 'poisson':
            lambda_param = params['lambda']
            return DistributionParams(
                dist_type='poisson',
                params={'lambda': lambda_param},
                mu=lambda_param,
                sigma=np.sqrt(lambda_param)
            )
        elif dist_type == 'nbinom':
            n = params['n']
            p = params['p']
            mu = n * (1 - p) / p
            var = mu / p

            return DistributionParams(
                dist_type='nbinom',
                params={'n': n, 'p': p},
                mu=mu,
                sigma=np.sqrt(var)
            )
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

    def probability_over(self, dist_params: DistributionParams, threshold: float) -> float:
        """Calculate P(X > threshold)"""
        if dist_params.dist_type == 'poisson':
            lambda_param = dist_params.params['lambda']
            return 1 - stats.poisson.cdf(threshold, lambda_param)
        elif dist_params.dist_type == 'nbinom':
            n = dist_params.params['n']
            p = dist_params.params['p']
            return 1 - stats.nbinom.cdf(threshold, n, p)
        else:
            raise ValueError(f"Unknown distribution: {dist_params.dist_type}")


class LognormalModel:
    """
    Lognormal model for yardage props

    Yards are typically right-skewed with a long tail
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Fit lognormal distribution

        Args:
            X: Feature matrix
            y: Target yards

        Returns:
            Model parameters
        """
        # Filter out zeros (can't take log of 0)
        y_positive = y[y > 0]

        if len(y_positive) == 0:
            return {'mu': 0, 'sigma': 1}

        # Fit lognormal
        log_y = np.log(y_positive)
        mu = np.mean(log_y)
        sigma = np.std(log_y)

        # Adjust for zeros (mixture model concept)
        zero_prob = (len(y) - len(y_positive)) / len(y)

        return {
            'dist_type': 'lognormal',
            'mu': mu,
            'sigma': sigma,
            'zero_prob': zero_prob
        }

    def predict(self, params: Dict, features: Optional[np.ndarray] = None) -> DistributionParams:
        """Predict lognormal distribution"""
        mu = params['mu']
        sigma = params['sigma']

        # Expected value of lognormal
        expected_value = np.exp(mu + sigma ** 2 / 2)

        # Account for zero probability
        zero_prob = params.get('zero_prob', 0)
        expected_value *= (1 - zero_prob)

        return DistributionParams(
            dist_type='lognormal',
            params={'mu': mu, 'sigma': sigma, 'zero_prob': zero_prob},
            mu=expected_value,
            sigma=sigma
        )

    def probability_over(self, dist_params: DistributionParams, threshold: float) -> float:
        """Calculate P(X > threshold)"""
        mu = dist_params.params['mu']
        sigma = dist_params.params['sigma']
        zero_prob = dist_params.params.get('zero_prob', 0)

        if threshold <= 0:
            return 1.0 - zero_prob

        # P(X > threshold) for lognormal
        prob_over = 1 - stats.lognorm.cdf(threshold, s=sigma, scale=np.exp(mu))

        # Adjust for zero probability
        prob_over *= (1 - zero_prob)

        return prob_over


class BernoulliModel:
    """
    Bernoulli/Logistic model for TD props

    Models probability of scoring a TD
    """

    def __init__(self):
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, method: str = 'logistic') -> Dict:
        """
        Fit TD probability model

        Args:
            X: Feature matrix
            y: Binary outcomes (1 if TD scored, 0 otherwise)
            method: 'logistic' or 'xgboost'

        Returns:
            Model parameters
        """
        if method == 'logistic':
            self.model = LogisticRegression(
                penalty='l2',
                C=1.0,
                max_iter=1000,
                random_state=42
            )
            self.model.fit(X, y)

            return {
                'dist_type': 'bernoulli',
                'method': 'logistic',
                'model': self.model
            }

        elif method == 'xgboost':
            self.model = xgb.XGBClassifier(
                objective='binary:logistic',
                max_depth=3,
                learning_rate=0.1,
                n_estimators=100,
                random_state=42
            )
            self.model.fit(X, y)

            return {
                'dist_type': 'bernoulli',
                'method': 'xgboost',
                'model': self.model
            }

        else:
            raise ValueError(f"Unknown method: {method}")

    def predict(self, params: Dict, features: np.ndarray) -> DistributionParams:
        """Predict TD probability"""
        model = params['model']

        # Predict probability
        prob = model.predict_proba(features.reshape(1, -1))[0, 1]

        return DistributionParams(
            dist_type='bernoulli',
            params={'p': prob},
            mu=prob,  # Expected value of Bernoulli is p
            sigma=np.sqrt(prob * (1 - prob))
        )

    def probability_over(self, dist_params: DistributionParams, threshold: float) -> float:
        """
        Calculate P(X > threshold)

        For TDs, this is typically P(TD >= 1) or P(TD >= 2)
        """
        p = dist_params.params['p']

        # For anytime TD (threshold = 0.5, meaning >= 1)
        if threshold < 1:
            return p

        # For multiple TDs (approximate with binomial)
        # This assumes independent opportunities, which is approximate
        n_opportunities = 20  # Rough estimate of red zone opportunities
        return 1 - stats.binom.cdf(threshold - 0.5, n=n_opportunities, p=p / n_opportunities)


class NormalModel:
    """
    Normal model for yards (alternative to lognormal)

    Simpler but doesn't capture right-skew as well
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Fit normal distribution"""
        mu = np.mean(y)
        sigma = np.std(y)

        return {
            'dist_type': 'normal',
            'mu': mu,
            'sigma': sigma
        }

    def predict(self, params: Dict, features: Optional[np.ndarray] = None) -> DistributionParams:
        """Predict normal distribution"""
        return DistributionParams(
            dist_type='normal',
            params={'mu': params['mu'], 'sigma': params['sigma']},
            mu=params['mu'],
            sigma=params['sigma']
        )

    def probability_over(self, dist_params: DistributionParams, threshold: float) -> float:
        """Calculate P(X > threshold)"""
        mu = dist_params.params['mu']
        sigma = dist_params.params['sigma']

        return 1 - stats.norm.cdf(threshold, loc=mu, scale=sigma)
