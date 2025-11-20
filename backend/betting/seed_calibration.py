"""Seed calibration with initial data based on historical prop accuracy patterns.

This script initializes the calibration system with reasonable priors based on:
1. General tendencies in NFL prop betting (favorites overvalued, etc.)
2. Known biases in raw probability estimates
3. Prop-type specific accuracy patterns

Run this once to initialize calibration, then it will be updated with real outcomes.
"""

import numpy as np
from datetime import datetime, timedelta
from backend.betting.probability_calibration import probability_calibrator


def seed_initial_calibration():
    """Seed calibration data with synthetic but realistic patterns.

    Key insights from prop betting research:
    1. High-probability events (>70%) tend to be slightly overestimated
    2. Low-probability events (<30%) tend to be underestimated
    3. Passing props have higher variance than rushing props
    4. TD props are often mispriced due to small samples
    """

    print("Seeding calibration data with initial patterns...")

    # Define prop types and their typical calibration adjustments
    prop_patterns = {
        # Yardage props - generally well-calibrated but high probs slightly over
        'passing_yards': {
            'bias': 'slight_overconfidence',
            'accuracy_rate': 0.52  # Books have slight edge
        },
        'rushing_yards': {
            'bias': 'slight_overconfidence',
            'accuracy_rate': 0.51
        },
        'receiving_yards': {
            'bias': 'slight_overconfidence',
            'accuracy_rate': 0.50
        },

        # Volume props - more predictable
        'receptions': {
            'bias': 'balanced',
            'accuracy_rate': 0.53
        },
        'completions': {
            'bias': 'balanced',
            'accuracy_rate': 0.54
        },
        'attempts': {
            'bias': 'balanced',
            'accuracy_rate': 0.52
        },
        'carries': {
            'bias': 'balanced',
            'accuracy_rate': 0.51
        },
        'targets': {
            'bias': 'balanced',
            'accuracy_rate': 0.50
        },

        # TD props - high variance, often mispriced
        'passing_tds': {
            'bias': 'underconfidence_low',  # Low prob TDs hit more than expected
            'accuracy_rate': 0.48
        },
        'rushing_tds': {
            'bias': 'underconfidence_low',
            'accuracy_rate': 0.47
        },
        'receiving_tds': {
            'bias': 'underconfidence_low',
            'accuracy_rate': 0.46
        },

        # Other props
        'interceptions': {
            'bias': 'underconfidence_low',
            'accuracy_rate': 0.49
        }
    }

    # Generate synthetic calibration data for each prop type
    for prop_type, pattern in prop_patterns.items():
        n_samples = 100  # Minimum for reasonable calibration

        # Generate predicted probabilities across the range
        predicted_probs = np.random.beta(2, 2, n_samples)  # Centered around 0.5

        # Generate outcomes based on pattern
        outcomes = []

        for pred_prob in predicted_probs:
            if pattern['bias'] == 'slight_overconfidence':
                # High probs slightly less accurate
                actual_prob = pred_prob * 0.95 if pred_prob > 0.6 else pred_prob * 1.02
            elif pattern['bias'] == 'underconfidence_low':
                # Low probs more accurate than predicted
                actual_prob = pred_prob * 1.1 if pred_prob < 0.4 else pred_prob * 0.98
            else:  # balanced
                actual_prob = pred_prob

            # Clamp probability
            actual_prob = max(0.01, min(0.99, actual_prob))

            # Generate outcome
            outcome = 1 if np.random.random() < actual_prob else 0
            outcomes.append(outcome)

        # Add to calibration data
        base_time = datetime.now() - timedelta(days=90)

        for i, (prob, outcome) in enumerate(zip(predicted_probs, outcomes)):
            timestamp = (base_time + timedelta(hours=i * 2)).isoformat()
            probability_calibrator.add_outcome(
                prop_type=prop_type,
                predicted_prob=float(prob),
                actual_outcome=int(outcome),
                timestamp=timestamp
            )

        print(f"  {prop_type}: {n_samples} samples seeded")

    # Save all calibration data
    probability_calibrator._save_calibration_data()

    # Fit calibration models for all prop types
    print("\nFitting calibration models...")
    probability_calibrator.fit_all(min_samples=50, method='isotonic')

    # Print calibration report
    print("\nCalibration Report:")
    report = probability_calibrator.get_calibration_report()

    for prop_type, info in report['prop_types'].items():
        if info['calibrated']:
            print(f"  {prop_type}: ECE={info['calibration_error']:.4f}, n={info['n_samples']}")

    print(f"\nTotal: {report['calibrated_prop_types']}/{report['total_prop_types']} prop types calibrated")


if __name__ == "__main__":
    seed_initial_calibration()
