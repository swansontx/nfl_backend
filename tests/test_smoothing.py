"""Tests for smoothing and rolling calculations."""

import pytest
from backend.features.smoothing_and_rolling import (
    calculate_ema,
    calculate_rolling_mean,
    calculate_weighted_recent
)


class TestCalculateEma:
    """Tests for exponential moving average."""

    def test_ema_empty_list(self):
        """Test EMA with empty list."""
        result = calculate_ema([])
        assert result == []

    def test_ema_single_value(self):
        """Test EMA with single value."""
        result = calculate_ema([100.0])
        assert result == [100.0]

    def test_ema_basic_calculation(self):
        """Test basic EMA calculation."""
        values = [100.0, 110.0, 105.0]
        alpha = 0.5
        result = calculate_ema(values, alpha=alpha)

        # First value should be unchanged
        assert result[0] == 100.0

        # Second: 0.5 * 110 + 0.5 * 100 = 105
        assert result[1] == 105.0

        # Third: 0.5 * 105 + 0.5 * 105 = 105
        assert result[2] == 105.0

    def test_ema_high_alpha(self):
        """Test EMA with high alpha (more weight on recent)."""
        values = [100.0, 200.0]
        result = calculate_ema(values, alpha=0.9)

        # Second value: 0.9 * 200 + 0.1 * 100 = 190
        assert result[1] == 190.0


class TestCalculateRollingMean:
    """Tests for rolling mean calculation."""

    def test_rolling_mean_insufficient_data(self):
        """Test rolling mean with insufficient data."""
        result = calculate_rolling_mean([100.0, 110.0], window=4)
        assert all(str(x) == 'nan' for x in result)

    def test_rolling_mean_basic(self):
        """Test basic rolling mean."""
        values = [100.0, 110.0, 120.0, 130.0]
        result = calculate_rolling_mean(values, window=2)

        # First value: insufficient data
        assert str(result[0]) == 'nan'

        # Second: mean of [100, 110] = 105
        assert result[1] == 105.0

        # Third: mean of [110, 120] = 115
        assert result[2] == 115.0

        # Fourth: mean of [120, 130] = 125
        assert result[3] == 125.0

    def test_rolling_mean_window_4(self):
        """Test rolling mean with window of 4."""
        values = [100.0, 110.0, 120.0, 130.0, 140.0]
        result = calculate_rolling_mean(values, window=4)

        # First three: insufficient data
        for i in range(3):
            assert str(result[i]) == 'nan'

        # Fourth: mean of [100, 110, 120, 130] = 115
        assert result[3] == 115.0

        # Fifth: mean of [110, 120, 130, 140] = 125
        assert result[4] == 125.0


class TestCalculateWeightedRecent:
    """Tests for weighted recent average."""

    def test_weighted_empty_list(self):
        """Test weighted average with empty list."""
        result = calculate_weighted_recent([])
        assert result == 0.0

    def test_weighted_single_value(self):
        """Test weighted average with single value."""
        result = calculate_weighted_recent([100.0])
        # Single value with weight 2^0 = 1
        assert result == 100.0

    def test_weighted_basic(self):
        """Test basic weighted average (most recent = highest weight)."""
        values = [100.0, 200.0]  # oldest, newest
        # Weights: [2^0, 2^1] = [1, 2]
        # Result: (100*1 + 200*2) / 3 = 500/3 = 166.67
        result = calculate_weighted_recent(values)
        assert abs(result - 166.67) < 0.01

    def test_weighted_custom_weights(self):
        """Test weighted average with custom weights."""
        values = [100.0, 200.0]
        weights = [1.0, 1.0]  # Equal weights
        result = calculate_weighted_recent(values, weights=weights)
        assert result == 150.0  # Simple average

    def test_weighted_mismatched_lengths(self):
        """Test error with mismatched lengths."""
        with pytest.raises(ValueError, match="Values and weights must have same length"):
            calculate_weighted_recent([100.0, 200.0], weights=[1.0])
