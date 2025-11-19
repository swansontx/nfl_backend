"""Train kicker prop models (FG Made, XP Made, Total Kicker Points).

Kicker props are popular markets with predictable patterns:
- FG Made: Poisson regression (count data)
- XP Made: Poisson regression (count data)
- Total Kicker Points: Poisson or Quantile regression

Features:
- Season average kicker stats
- L3 game average
- Team implied total (more points = more opportunities)
- Is dome (kickers perform better indoors)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import poisson
import joblib


def load_kicker_data():
    """Load kicker stats."""

    print(f"\n{'='*80}")
    print(f"LOADING KICKER STATS")
    print(f"{'='*80}\n")

    kicker_file = Path('inputs/kicker_stats_2025_synthetic.csv')

    if not kicker_file.exists():
        print(f"❌ Kicker stats not found: {kicker_file}")
        print("   Run: python -m backend.analysis.generate_kicker_stats")
        return pd.DataFrame()

    kicker_df = pd.read_csv(kicker_file)

    print(f"✅ Loaded {len(kicker_df)} kicker-game records")
    print()

    return kicker_df


def create_kicker_features(kicker_df, games_df):
    """Create training features for kicker props.

    Args:
        kicker_df: Kicker stats DataFrame
        games_df: Games DataFrame for context features

    Returns:
        DataFrame with features
    """

    print(f"\n{'='*80}")
    print(f"CREATING KICKER FEATURES")
    print(f"{'='*80}\n")

    # Merge with games data to get implied totals, dome, etc
    kicker_with_context = kicker_df.copy()

    # Merge game info
    game_info = games_df[[
        'game_id', 'total_line', 'spread_line', 'roof', 'home_implied_total', 'away_implied_total'
    ]].copy()

    kicker_with_context = kicker_with_context.merge(game_info, on='game_id', how='left')

    # Add implied total for this team
    kicker_with_context['team_implied_total'] = kicker_with_context.apply(
        lambda row: row['home_implied_total'] if row['is_home'] else row['away_implied_total'],
        axis=1
    )

    # Is dome
    kicker_with_context['is_dome'] = (kicker_with_context['roof'] == 'dome').astype(int)

    # Sort by kicker and week
    kicker_with_context = kicker_with_context.sort_values(['kicker_id', 'week'])

    # Calculate season averages (expanding mean)
    for stat in ['fg_made', 'xp_made', 'total_points']:
        kicker_with_context[f'season_avg_{stat}'] = kicker_with_context.groupby('kicker_id')[stat].expanding().mean().reset_index(level=0, drop=True)

        # L3 average
        kicker_with_context[f'l3_avg_{stat}'] = kicker_with_context.groupby('kicker_id')[stat].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

    # Games played
    kicker_with_context['games_played'] = kicker_with_context.groupby('kicker_id').cumcount() + 1

    # Filter to 3+ games
    kicker_with_features = kicker_with_context[kicker_with_context['games_played'] >= 3].copy()

    print(f"Kickers with 3+ games: {len(kicker_with_features)}")
    print()

    return kicker_with_features


def train_kicker_model(train_df, prop_type, target_col):
    """Train a kicker prop model.

    Args:
        train_df: Training DataFrame
        prop_type: Type of prop (e.g., 'fg_made')
        target_col: Target column name

    Returns:
        Trained model, features, metrics
    """

    print(f"\n{'='*80}")
    print(f"TRAINING: {prop_type.upper()}")
    print(f"{'='*80}\n")

    # Features
    feature_cols = [
        f'season_avg_{prop_type}',
        f'l3_avg_{prop_type}',
        'games_played',
        'team_implied_total',
        'is_dome',
    ]

    X = train_df[feature_cols].fillna(0)
    y = train_df[target_col]

    print(f"Training set: {len(X)} kicker-games")
    print(f"Features: {feature_cols}")
    print()

    # Train Poisson model
    model = PoissonRegressor(
        alpha=1.0,
        max_iter=300
    )

    model.fit(X, y)

    # Training MAE
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)

    print(f"Training MAE: {mae:.2f}")
    print(f"Actual distribution: {y.value_counts().sort_index().to_dict()}")
    print()

    # Feature importance (coefficients)
    print("Feature coefficients:")
    for feat, coef in zip(feature_cols, model.coef_):
        print(f"  {feat:30s}: {coef:.3f}")
    print()

    return model, feature_cols, {'mae': mae}


def predict_kicker_prob(model, X, line):
    """Predict P(stat > line) from Poisson model.

    Args:
        model: Trained Poisson model
        X: Features
        line: Prop line (e.g., 1.5 for "will they make 2+ FGs?")

    Returns:
        Probability of going OVER
    """

    # Predict lambda (expected rate)
    lambda_pred = model.predict([X])[0] if len(np.array(X).shape) == 1 else model.predict(X)

    # Calculate P(X > line) using Poisson CDF
    if hasattr(lambda_pred, '__iter__'):
        probs = []
        for lam in lambda_pred:
            # P(X > line) = 1 - P(X <= line)
            prob = 1 - poisson.cdf(line, lam)
            probs.append(prob)
        return np.array(probs)

    return 1 - poisson.cdf(line, lambda_pred)


def backtest_kicker_prop(model, features, kicker_df, target_col, week, typical_line, prop_name):
    """Backtest a kicker prop on a specific week.

    Args:
        model: Trained model
        features: Feature columns
        kicker_df: Full kicker DataFrame
        target_col: Target column
        week: Week to backtest
        typical_line: Typical DK line
        prop_name: Name of prop for display

    Returns:
        Dict with results
    """

    print(f"\n{'='*80}")
    print(f"BACKTESTING: {prop_name.upper()} - WEEK {week}")
    print(f"{'='*80}\n")

    # Get week data
    week_df = kicker_df[
        (kicker_df['week'] == week) &
        (kicker_df['games_played'] >= 3)
    ].copy()

    print(f"Week {week} kickers: {len(week_df)}")

    if len(week_df) == 0:
        return None

    # Prepare features
    X = week_df[features].fillna(0)

    # Predict probabilities
    prob_over = predict_kicker_prob(model, X.values, typical_line)
    week_df['prob_over'] = prob_over

    # Actual results
    week_df['actual'] = week_df[target_col]
    week_df['hit_over'] = week_df['actual'] > typical_line

    # Calculate edge (assuming -110 odds = 52.4% implied)
    week_df['edge'] = week_df['prob_over'] - 0.524

    # Bets with >5% edge
    bets = week_df[week_df['edge'] > 0.05].copy()

    print(f"Bets with >5% edge: {len(bets)}")

    if len(bets) > 0:
        hit_rate = bets['hit_over'].mean()
        roi = ((bets['hit_over'].sum() * 90) - ((~bets['hit_over']).sum() * 100)) / (len(bets) * 100)

        print(f"Hit rate: {hit_rate:.1%}")
        print(f"ROI: {roi:+.1%}")

        # Show top bets
        print(f"\nTop {min(5, len(bets))} bets:")
        top_bets = bets.nlargest(5, 'edge')[['kicker_name', 'team', 'prob_over', 'edge', 'actual', 'hit_over']]
        for idx, row in top_bets.iterrows():
            status = '✅' if row['hit_over'] else '❌'
            print(f"  {status} {row['kicker_name']:25s} ({row['team']}): Prob={row['prob_over']:.1%}, Edge={row['edge']:+.1%}, Actual={row['actual']:.0f}")

    else:
        print("No bets with sufficient edge")
        hit_rate = 0
        roi = 0

    print()

    return {
        'prop_type': prop_name,
        'week': week,
        'bets': len(bets),
        'hit_rate': hit_rate,
        'roi': roi
    }


def main():
    """Train all kicker prop models."""

    print(f"\n{'='*80}")
    print(f"TRAINING KICKER PROP MODELS")
    print(f"{'='*80}\n")

    # Load data
    kicker_df = load_kicker_data()

    if len(kicker_df) == 0:
        return

    # Load games for context features
    games_file = Path('inputs/games_2025_with_quarters.csv')
    games_df = pd.read_csv(games_file)

    # Add implied totals to games
    games_df['home_implied_total'] = (games_df['total_line'] - games_df['spread_line']) / 2
    games_df['away_implied_total'] = (games_df['total_line'] + games_df['spread_line']) / 2

    # Create features
    kicker_with_features = create_kicker_features(kicker_df, games_df)

    # Training weeks
    training_weeks = list(range(1, 10))
    train_df = kicker_with_features[kicker_with_features['week'].isin(training_weeks)].copy()

    print(f"Training weeks: {training_weeks}")
    print(f"Training records: {len(train_df)}")
    print()

    # Kicker props configuration
    kicker_props = {
        'fg_made': {
            'target_col': 'fg_made',
            'typical_line': 1.5,  # Will they make 2+ FGs?
            'name': 'FG Made',
        },
        'xp_made': {
            'target_col': 'xp_made',
            'typical_line': 2.5,  # Will they make 3+ XPs?
            'name': 'XP Made',
        },
        'total_points': {
            'target_col': 'total_points',
            'typical_line': 7.5,  # Will they score 8+ points?
            'name': 'Total Kicker Points',
        },
    }

    # Train each prop
    all_models = {}
    backtest_results = []

    for prop_type, config in kicker_props.items():
        print(f"\n{'#'*80}")
        print(f"# {config['name'].upper()}")
        print(f"{'#'*80}")

        # Train model
        model, features, metrics = train_kicker_model(
            train_df,
            prop_type,
            config['target_col']
        )

        # Save model
        models_dir = Path('outputs/models/kicker')
        models_dir.mkdir(parents=True, exist_ok=True)

        model_file = models_dir / f'{prop_type}_model.pkl'
        joblib.dump({
            'model': model,
            'features': features,
            'config': config
        }, model_file)

        print(f"💾 Saved to: {model_file}")

        all_models[prop_type] = {
            'model': model,
            'features': features,
            'config': config
        }

        # Backtest on Weeks 10 & 11
        for week in [10, 11]:
            result = backtest_kicker_prop(
                model,
                features,
                kicker_with_features,
                config['target_col'],
                week,
                config['typical_line'],
                config['name']
            )

            if result:
                backtest_results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print(f"KICKER PROP TRAINING COMPLETE")
    print(f"{'='*80}\n")

    print(f"Props trained: {len(all_models)}")
    print()

    # Backtest summary
    if backtest_results:
        results_df = pd.DataFrame(backtest_results)

        print("BACKTEST SUMMARY:")
        print()

        summary = results_df.groupby('prop_type').agg({
            'bets': 'sum',
            'hit_rate': 'mean',
            'roi': 'mean'
        }).sort_values('roi', ascending=False)

        for prop_type, row in summary.iterrows():
            print(f"{prop_type:25s}: {int(row['bets']):3d} bets, {row['hit_rate']:.1%} hit rate, {row['roi']:+.1%} ROI")

        print()

    print("✅ All kicker models trained and backtested")
    print()

    print("New markets available:")
    print("  ✅ FG Made (1.5 line)")
    print("  ✅ XP Made (2.5 line)")
    print("  ✅ Total Kicker Points (7.5 line)")
    print()
    print("Total new markets: 3")
    print()


if __name__ == '__main__':
    main()
