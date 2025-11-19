"""Feature smoothing and rolling window calculations

Applies smoothing techniques and rolling aggregations to raw play-by-play
features to create more stable, predictive features.

Techniques:
- Exponential moving averages (EMA)
- Rolling means/medians
- Weighted recent games
- Trend calculations

TODOs:
- Implement EMA calculation for various stats
- Add configurable window sizes
- Support multiple stat types (yards, TDs, attempts, etc.)
- Add opponent-adjusted metrics
- Generate trend indicators
"""

from pathlib import Path
import argparse
import json
from typing import Dict, List, Optional


def calculate_ema(values: List[float], alpha: float = 0.3) -> List[float]:
    """Calculate exponential moving average.

    Args:
        values: List of values in chronological order
        alpha: Smoothing factor (0-1), higher = more weight on recent

    Returns:
        List of EMA values

    Formula: EMA_t = alpha * value_t + (1-alpha) * EMA_{t-1}
    """
    if not values:
        return []

    ema = [values[0]]  # Initialize with first value

    for value in values[1:]:
        ema_value = alpha * value + (1 - alpha) * ema[-1]
        ema.append(ema_value)

    return ema


def calculate_rolling_mean(values: List[float], window: int = 4) -> List[float]:
    """Calculate rolling mean over a window.

    Args:
        values: List of values in chronological order
        window: Window size for rolling calculation

    Returns:
        List of rolling mean values (NaN for insufficient data)
    """
    if window > len(values):
        return [float('nan')] * len(values)

    rolling_means = []

    for i in range(len(values)):
        if i < window - 1:
            # Not enough data yet
            rolling_means.append(float('nan'))
        else:
            window_values = values[i - window + 1:i + 1]
            rolling_means.append(sum(window_values) / window)

    return rolling_means


def calculate_weighted_recent(values: List[float],
                              weights: Optional[List[float]] = None) -> float:
    """Calculate weighted average with more weight on recent games.

    Args:
        values: List of values in chronological order (oldest first)
        weights: Optional custom weights (defaults to exponential decay)

    Returns:
        Weighted average
    """
    if not values:
        return 0.0

    if weights is None:
        # Default: exponential decay weights (most recent = highest weight)
        n = len(values)
        weights = [2 ** i for i in range(n)]

    if len(values) != len(weights):
        raise ValueError("Values and weights must have same length")

    weighted_sum = sum(v * w for v, w in zip(values, weights))
    total_weight = sum(weights)

    return weighted_sum / total_weight if total_weight > 0 else 0.0


def smooth_player_features(raw_features: Dict[str, List[float]],
                           config: Dict[str, any]) -> Dict[str, any]:
    """Apply smoothing to player features.

    Args:
        raw_features: Dictionary of stat_name -> list of weekly values
                     Example: {'pass_yards': [250, 300, 280, ...], ...}
        config: Smoothing configuration
               Example: {'ema_alpha': 0.3, 'rolling_window': 4}

    Returns:
        Dictionary with smoothed features

    Output includes:
        - EMA values for each stat
        - Rolling means
        - Weighted recent averages
        - Trend indicators (improving/declining)
    """
    smoothed = {}

    ema_alpha = config.get('ema_alpha', 0.3)
    rolling_window = config.get('rolling_window', 4)

    for stat_name, values in raw_features.items():
        if not values:
            continue

        # Calculate various smoothed versions
        smoothed[f'{stat_name}_ema'] = calculate_ema(values, alpha=ema_alpha)
        smoothed[f'{stat_name}_rolling_mean'] = calculate_rolling_mean(values, window=rolling_window)
        smoothed[f'{stat_name}_weighted_recent'] = calculate_weighted_recent(values)

        # Calculate trend (compare recent vs earlier average)
        if len(values) >= 4:
            recent_avg = sum(values[-2:]) / 2
            earlier_avg = sum(values[-4:-2]) / 2
            trend = (recent_avg - earlier_avg) / earlier_avg if earlier_avg > 0 else 0
            smoothed[f'{stat_name}_trend'] = trend

    return smoothed


def process_all_players(input_features_path: Path,
                       output_path: Path,
                       config: Optional[Dict] = None) -> None:
    """Process all player features with smoothing.

    Args:
        input_features_path: Path to raw player_pbp_features_by_id.json
        output_path: Output path for smoothed features
        config: Smoothing configuration

    Input format:
        {
            "player_id_1": {
                "pass_yards": [250, 300, 280, ...],
                "pass_tds": [2, 1, 3, ...],
                ...
            },
            ...
        }

    Output format: Same structure but with smoothed features added
    """
    if config is None:
        config = {
            'ema_alpha': 0.3,
            'rolling_window': 4
        }

    # Load actual features from extract_player_pbp_features.py output
    if not input_features_path.exists():
        print(f"⚠ Input features file not found: {input_features_path}")
        print(f"⚠ Creating empty smoothed features file")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('{}')
        return {}

    print(f"Loading raw features from {input_features_path}")

    with open(input_features_path, 'r') as f:
        player_games = json.load(f)

    # Convert from list of game dicts to dict of feature arrays
    # Input format: player_id -> [game1, game2, ...] where each game has multiple features
    # Output format: player_id -> {feature_name: [val1, val2, ...]}
    raw_features = {}

    for player_id, games in player_games.items():
        if not games:
            continue

        # Initialize feature arrays
        player_features = {}

        # Extract each stat across all games
        for game in games:
            for stat_name, stat_value in game.items():
                # Skip metadata fields
                if stat_name in ['game_id', 'season', 'week']:
                    continue

                if stat_name not in player_features:
                    player_features[stat_name] = []

                player_features[stat_name].append(stat_value)

        if player_features:
            raw_features[player_id] = player_features

    print(f"✓ Loaded features for {len(raw_features)} players")

    # Process each player
    all_smoothed = {}

    for player_id, player_features in raw_features.items():
        smoothed = smooth_player_features(player_features, config)
        # Combine raw and smoothed
        all_smoothed[player_id] = {**player_features, **smoothed}

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_smoothed, f, indent=2)

    print(f"Processed {len(all_smoothed)} players")
    print(f"Smoothed features written to {output_path}")

    return all_smoothed


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Apply smoothing to player features')
    p.add_argument('--input', type=Path,
                   default=Path('outputs/player_pbp_features_by_id.json'),
                   help='Path to raw features JSON')
    p.add_argument('--output', type=Path,
                   default=Path('outputs/player_features_smoothed.json'),
                   help='Output path for smoothed features')
    p.add_argument('--ema-alpha', type=float, default=0.3,
                   help='EMA smoothing factor (0-1)')
    p.add_argument('--rolling-window', type=int, default=4,
                   help='Rolling window size')
    args = p.parse_args()

    config = {
        'ema_alpha': args.ema_alpha,
        'rolling_window': args.rolling_window
    }

    process_all_players(args.input, args.output, config)
