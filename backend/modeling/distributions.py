"""Distribution models for prop betting probabilities.

Different prop types have different statistical distributions:
- Yardage props (passing/rushing/receiving): Normal/Gaussian
- Count props (receptions, attempts, carries): Poisson or Negative Binomial
- TD props: Poisson (rare events)

Using the right distribution model improves probability estimates.
"""

from dataclasses import dataclass
from typing import Protocol
import math

# Import settings for empirical std dev config
try:
    from backend.config import settings
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    settings = None


class PropDistribution(Protocol):
    """Protocol for prop distributions."""

    def cdf(self, x: float) -> float:
        """Cumulative distribution function - P(X <= x)."""
        ...

    def prob_over(self, line: float) -> float:
        """Probability of going over the line."""
        ...

    def prob_under(self, line: float) -> float:
        """Probability of going under the line."""
        ...


@dataclass
class NormalDistribution:
    """Normal/Gaussian distribution for continuous stats like yardage.

    Good for: passing_yards, rushing_yards, receiving_yards
    """
    mean: float
    std: float

    def cdf(self, x: float) -> float:
        """Normal CDF using error function."""
        if self.std <= 0:
            return 1.0 if x >= self.mean else 0.0
        z = (x - self.mean) / self.std
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    def prob_over(self, line: float) -> float:
        """P(X > line) - uses 0.5 adjustment for discrete lines."""
        # Subtract 0.5 for half-point lines (e.g., 74.5 yards)
        adjusted_line = line - 0.5 if line == int(line) + 0.5 else line
        return 1 - self.cdf(adjusted_line)

    def prob_under(self, line: float) -> float:
        """P(X < line)."""
        adjusted_line = line - 0.5 if line == int(line) + 0.5 else line
        return self.cdf(adjusted_line)


@dataclass
class PoissonDistribution:
    """Poisson distribution for count data.

    Good for: receptions, attempts, carries, targets, TDs
    """
    lam: float  # lambda (expected value)

    def __post_init__(self):
        # Ensure lambda is positive
        self.lam = max(self.lam, 0.01)

    def pmf(self, k: int) -> float:
        """Probability mass function P(X = k)."""
        if k < 0:
            return 0.0
        try:
            return (self.lam ** k * math.exp(-self.lam)) / math.factorial(k)
        except (OverflowError, ValueError):
            # For large k, use log-space computation
            log_prob = k * math.log(self.lam) - self.lam - sum(math.log(i) for i in range(1, k + 1))
            return math.exp(log_prob)

    def cdf(self, x: float) -> float:
        """Cumulative distribution P(X <= x)."""
        if x < 0:
            return 0.0
        k = int(x)
        total = 0.0
        for i in range(k + 1):
            total += self.pmf(i)
        return min(total, 1.0)

    def prob_over(self, line: float) -> float:
        """P(X > line) for discrete distribution.

        For line 4.5, this is P(X >= 5) = 1 - P(X <= 4)
        """
        # Handle half-point lines (most common)
        if line == int(line) + 0.5:
            k = int(line)  # 4.5 -> need X >= 5, so P(X > 4)
            return 1 - self.cdf(k)
        else:
            # Whole number line - need strictly greater
            return 1 - self.cdf(line)

    def prob_under(self, line: float) -> float:
        """P(X < line) for discrete distribution.

        For line 4.5, this is P(X <= 4)
        """
        if line == int(line) + 0.5:
            k = int(line)
            return self.cdf(k)
        else:
            return self.cdf(line - 1) if line > 0 else 0.0


@dataclass
class NegativeBinomialDistribution:
    """Negative Binomial for overdispersed count data.

    Good for: high-variance count stats where Poisson variance is too low
    Parameterized by mean (mu) and dispersion (alpha).
    """
    mu: float      # mean
    alpha: float   # dispersion parameter (higher = more variance)

    def __post_init__(self):
        self.mu = max(self.mu, 0.01)
        self.alpha = max(self.alpha, 0.01)

    @property
    def r(self) -> float:
        """Number of successes parameter."""
        return 1 / self.alpha

    @property
    def p(self) -> float:
        """Success probability parameter."""
        return self.r / (self.r + self.mu)

    def pmf(self, k: int) -> float:
        """Probability mass function."""
        if k < 0:
            return 0.0
        try:
            # Using gamma function formulation
            from math import gamma, lgamma
            log_coef = lgamma(k + self.r) - lgamma(k + 1) - lgamma(self.r)
            log_prob = log_coef + self.r * math.log(self.p) + k * math.log(1 - self.p)
            return math.exp(log_prob)
        except (OverflowError, ValueError):
            return 0.0

    def cdf(self, x: float) -> float:
        """Cumulative distribution P(X <= x)."""
        if x < 0:
            return 0.0
        k = int(x)
        total = 0.0
        for i in range(k + 1):
            total += self.pmf(i)
        return min(total, 1.0)

    def prob_over(self, line: float) -> float:
        """P(X > line)."""
        if line == int(line) + 0.5:
            k = int(line)
            return 1 - self.cdf(k)
        else:
            return 1 - self.cdf(line)

    def prob_under(self, line: float) -> float:
        """P(X < line)."""
        if line == int(line) + 0.5:
            k = int(line)
            return self.cdf(k)
        else:
            return self.cdf(line - 1) if line > 0 else 0.0


# Mapping of prop types to their appropriate distribution
PROP_DISTRIBUTION_MAP = {
    # Yardage props -> Normal (continuous)
    'passing_yards': 'normal',
    'rushing_yards': 'normal',
    'receiving_yards': 'normal',
    'pass_yards': 'normal',
    'rush_yards': 'normal',
    'rec_yards': 'normal',

    # Count props -> Poisson (discrete)
    'receptions': 'poisson',
    'completions': 'poisson',
    'attempts': 'poisson',
    'carries': 'poisson',
    'targets': 'poisson',
    'interceptions': 'poisson',

    # TD props -> Poisson (rare events)
    'passing_tds': 'poisson',
    'rushing_tds': 'poisson',
    'receiving_tds': 'poisson',
    'pass_tds': 'poisson',
    'rush_tds': 'poisson',
    'rec_tds': 'poisson',
    'touchdowns': 'poisson',
    'anytime_td': 'poisson',
}

# Default standard deviations by prop type (from backtest analysis)
# These should be updated based on actual model performance
DEFAULT_STD_BY_PROP = {
    'passing_yards': 60.0,
    'rushing_yards': 25.0,
    'receiving_yards': 20.0,
    'pass_yards': 60.0,
    'rush_yards': 25.0,
    'rec_yards': 20.0,
    'receptions': 2.0,
    'completions': 5.0,
    'attempts': 7.0,
    'carries': 4.0,
    'targets': 2.5,
    'interceptions': 0.8,
    'passing_tds': 0.9,
    'rushing_tds': 0.5,
    'receiving_tds': 0.4,
}


def get_std_dev_for_prop(prop_type: str) -> float:
    """Get empirical standard deviation for a prop type.

    Uses config values if available, falls back to defaults.

    Args:
        prop_type: Type of prop (e.g., 'passing_yards', 'receptions')

    Returns:
        Standard deviation value
    """
    prop_lower = prop_type.lower()

    # Try config values first (from backtest analysis)
    if CONFIG_AVAILABLE and settings and hasattr(settings, 'prop_std_devs'):
        if prop_lower in settings.prop_std_devs:
            return settings.prop_std_devs[prop_lower]

    # Fall back to module defaults
    return DEFAULT_STD_BY_PROP.get(prop_lower, 15.0)


def get_distribution(
    prop_type: str,
    projection: float,
    std_dev: float = None
) -> PropDistribution:
    """Get the appropriate distribution for a prop type.

    Args:
        prop_type: Type of prop (e.g., 'passing_yards', 'receptions')
        projection: Projected value (mean/lambda)
        std_dev: Standard deviation (for normal) or None to use empirical default

    Returns:
        Distribution object with cdf, prob_over, prob_under methods
    """
    dist_type = PROP_DISTRIBUTION_MAP.get(prop_type.lower(), 'normal')

    if dist_type == 'poisson':
        return PoissonDistribution(lam=projection)
    else:
        # Normal distribution
        if std_dev is None or std_dev <= 0:
            std_dev = get_std_dev_for_prop(prop_type)
        return NormalDistribution(mean=projection, std=std_dev)


def calculate_hit_probability(
    prop_type: str,
    projection: float,
    line: float,
    std_dev: float = None,
    side: str = 'over'
) -> float:
    """Calculate probability of hitting a prop line.

    Args:
        prop_type: Type of prop
        projection: Model projection
        line: Sportsbook line
        std_dev: Standard deviation (optional)
        side: 'over' or 'under'

    Returns:
        Probability (0 to 1)
    """
    dist = get_distribution(prop_type, projection, std_dev)

    if side.lower() == 'over':
        return dist.prob_over(line)
    else:
        return dist.prob_under(line)
