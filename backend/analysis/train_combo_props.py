"""Train combo prop models (Pass+Rush Yards, Rec+Rush Yards).

Combo props are popular for:
- Mobile QBs: Pass + Rush Yards (Lamar, Hurts, Allen, Daniels)
- Pass-catching RBs: Rec + Rush Yards (CMC, Bijan, Achane)

These use the same quantile regression approach as single props.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib


def load_player_stats():
    """Load player stats."""

    print(f"\n{'='*80}")
    print(f"LOADING PLAYER STATS")
    print(f"{'='*80}\n")

    player_file = Path('/home/user/nfl_backend/inputs/player_stats_2025.csv')

    if not player_file.exists():
        print(f"❌ Player stats not found: {player_file}")
        return pd.DataFrame()

    player_df = pd.read_csv(player_file)

    print(f"✅ Loaded {len(player_df)} player-game records")
    print()

    return player_df


def create_combo_features(player_df, combo_type):
    """Create features for combo props.

    Args:
        player_df: Player stats DataFrame
        combo_type: 'pass_rush' or 'rec_rush'

    Returns:
        DataFrame with combo features
    """

    print(f"\n{'='*80}")
    print(f"CREATING COMBO FEATURES: {combo_type.upper()}")
    print(f"{'='*80}\n")

    df = player_df.copy()

    if combo_type == 'pass_rush':
        # For QBs: Pass + Rush Yards
        df = df[df['position'] == 'QB'].copy()

        # Calculate combo total
        df['pass_rush_yards'] = df['passing_yards'] + df['rushing_yards']

        target_col = 'pass_rush_yards'
        positions = ['QB']

    else:  # rec_rush
        # For RBs: Rec + Rush Yards
        df = df[df['position'] == 'RB'].copy()

        # Calculate combo total
        df['rec_rush_yards'] = df['receiving_yards'] + df['rushing_yards']

        target_col = 'rec_rush_yards'
        positions = ['RB']

    print(f"Position filter: {positions}")
    print(f"Records: {len(df)}")

    # Sort by player and week
    df = df.sort_values(['player_id', 'week'])

    # Calculate season average (expanding mean)
    df[f'season_avg_{combo_type}'] = df.groupby('player_id')[target_col].expanding().mean().reset_index(level=0, drop=True)

    # Calculate L3 average
    df[f'l3_avg_{combo_type}'] = df.groupby('player_id')[target_col].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

    # Games played
    df['games_played'] = df.groupby('player_id').cumcount() + 1

    # Filter to 3+ games
    df = df[df['games_played'] >= 3].copy()

    print(f"Records with 3+ games: {len(df)}")
    print()

    # Show distribution
    print(f"{combo_type.upper()} distribution:")
    print(f"  Mean: {df[target_col].mean():.1f}")
    print(f"  Median: {df[target_col].median():.1f}")
    print(f"  Std: {df[target_col].std():.1f}")
    print(f"  Max: {df[target_col].max():.0f}")
    print()

    return df


def train_combo_model(train_df, combo_type, target_col):
    """Train combo prop model.

    Args:
        train_df: Training DataFrame
        combo_type: Type of combo ('pass_rush' or 'rec_rush')
        target_col: Target column name

    Returns:
        Dict of quantile models, features
    """

    print(f"\n{'='*80}")
    print(f"TRAINING: {combo_type.upper()}")
    print(f"{'='*80}\n")

    # Features
    feature_cols = [
        f'season_avg_{combo_type}',
        f'l3_avg_{combo_type}',
        'games_played'
    ]

    X = train_df[feature_cols].fillna(0)
    y = train_df[target_col]

    print(f"Training set: {len(X)} records")
    print(f"Features: {feature_cols}")
    print()

    # Train 5 quantile models
    quantile_models = {}

    for quantile in [0.10, 0.25, 0.50, 0.75, 0.90]:
        print(f"Training Q{int(quantile*100)} model...")

        model = GradientBoostingRegressor(
            loss='quantile',
            alpha=quantile,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )

        model.fit(X, y)

        quantile_models[f'q{int(quantile*100)}'] = model

    print()

    # Feature importance
    print("Feature Importance (Q50 model):")
    for feat, imp in zip(feature_cols, quantile_models['q50'].feature_importances_):
        print(f"  {feat:30s}: {imp:.3f}")
    print()

    return quantile_models, feature_cols


def predict_combo_prob(models, X, line):
    """Predict P(X > line) from quantile models."""

    # Get quantile predictions
    q10 = models['q10'].predict([X])[0] if len(np.array(X).shape) == 1 else models['q10'].predict(X)
    q25 = models['q25'].predict([X])[0] if len(np.array(X).shape) == 1 else models['q25'].predict(X)
    q50 = models['q50'].predict([X])[0] if len(np.array(X).shape) == 1 else models['q50'].predict(X)
    q75 = models['q75'].predict([X])[0] if len(np.array(X).shape) == 1 else models['q75'].predict(X)
    q90 = models['q90'].predict([X])[0] if len(np.array(X).shape) == 1 else models['q90'].predict(X)

    # Handle array vs scalar
    if hasattr(q10, '__iter__'):
        probs = []
        for i in range(len(q10)):
            quantiles = [
                (0.10, q10[i]),
                (0.25, q25[i]),
                (0.50, q50[i]),
                (0.75, q75[i]),
                (0.90, q90[i])
            ]
            prob = calculate_prob_from_quantiles(quantiles, line)
            probs.append(prob)
        return np.array(probs)

    # Single prediction
    quantiles = [(0.10, q10), (0.25, q25), (0.50, q50), (0.75, q75), (0.90, q90)]
    return calculate_prob_from_quantiles(quantiles, line)


def calculate_prob_from_quantiles(quantiles, line):
    """Linear interpolation to calculate P(X > line)."""

    # Extreme cases
    if line < quantiles[0][1]:
        return 0.95
    if line > quantiles[-1][1]:
        return 0.05

    # Linear interpolation
    for i in range(len(quantiles) - 1):
        q_low, val_low = quantiles[i]
        q_high, val_high = quantiles[i + 1]

        if val_low <= line <= val_high:
            prob_under = q_low + (q_high - q_low) * (line - val_low) / (val_high - val_low)
            return 1 - prob_under

    return 0.50


def backtest_combo_prop(models, features, player_df, combo_type, target_col, week, typical_line, prop_name):
    """Backtest combo prop on a specific week."""

    print(f"\n{'='*80}")
    print(f"BACKTESTING: {prop_name.upper()} - WEEK {week}")
    print(f"{'='*80}\n")

    # Get week data
    if combo_type == 'pass_rush':
        position = 'QB'
    else:
        position = 'RB'

    week_df = player_df[
        (player_df['week'] == week) &
        (player_df['position'] == position)
    ].copy()

    print(f"Week {week} players: {len(week_df)}")

    # Calculate features using ONLY data up to week-1
    historical_df = player_df[player_df['week'] < week].copy()

    # Calculate combo total for historical data
    if combo_type == 'pass_rush':
        historical_df['pass_rush_yards'] = historical_df['passing_yards'] + historical_df['rushing_yards']
        week_df['pass_rush_yards'] = week_df['passing_yards'] + week_df['rushing_yards']
    else:
        historical_df['rec_rush_yards'] = historical_df['receiving_yards'] + historical_df['rushing_yards']
        week_df['rec_rush_yards'] = week_df['receiving_yards'] + week_df['rushing_yards']

    # Season averages
    player_averages = historical_df.groupby('player_id')[target_col].agg(['mean', 'count']).reset_index()
    player_averages.columns = ['player_id', f'season_avg_{combo_type}', 'games_played']

    # L3 averages
    historical_df = historical_df.sort_values(['player_id', 'week'])
    l3_avg = historical_df.groupby('player_id')[target_col].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    historical_df[f'l3_avg_{combo_type}'] = l3_avg
    player_l3 = historical_df.groupby('player_id')[f'l3_avg_{combo_type}'].last().reset_index()

    # Merge features
    week_df = week_df.merge(player_averages, on='player_id', how='left')
    week_df = week_df.merge(player_l3, on='player_id', how='left')

    # Filter to 3+ games
    week_df = week_df[week_df['games_played'] >= 3].copy()

    print(f"Players with 3+ games: {len(week_df)}")

    if len(week_df) == 0:
        return None

    # Prepare features
    X = week_df[features].fillna(0)

    # Predict
    prob_over = predict_combo_prob(models, X.values, typical_line)
    week_df['prob_over'] = prob_over

    # Actual results
    week_df['actual'] = week_df[target_col]
    week_df['hit_over'] = week_df['actual'] > typical_line

    # Calculate edge
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
        top_bets = bets.nlargest(5, 'edge')[['player_name', 'prob_over', 'edge', 'actual', 'hit_over']]
        for idx, row in top_bets.iterrows():
            status = '✅' if row['hit_over'] else '❌'
            print(f"  {status} {row['player_name']:25s}: Prob={row['prob_over']:.1%}, Edge={row['edge']:+.1%}, Actual={row['actual']:.0f}")

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
    """Train combo prop models."""

    print(f"\n{'='*80}")
    print(f"TRAINING COMBO PROP MODELS")
    print(f"{'='*80}\n")

    # Load player stats
    player_df = load_player_stats()

    if len(player_df) == 0:
        return

    # Training weeks
    training_weeks = list(range(1, 10))
    train_df = player_df[player_df['week'].isin(training_weeks)].copy()

    print(f"Training weeks: {training_weeks}")
    print(f"Training records: {len(train_df)}")
    print()

    # Combo props configuration
    combo_props = {
        'pass_rush': {
            'target_col': 'pass_rush_yards',
            'typical_line': 280.5,  # For mobile QBs
            'name': 'Pass + Rush Yards',
        },
        'rec_rush': {
            'target_col': 'rec_rush_yards',
            'typical_line': 100.5,  # For pass-catching RBs
            'name': 'Rec + Rush Yards',
        },
    }

    # Train each combo prop
    all_models = {}
    backtest_results = []

    for combo_type, config in combo_props.items():
        print(f"\n{'#'*80}")
        print(f"# {config['name'].upper()}")
        print(f"{'#'*80}")

        # Create features
        combo_train_df = create_combo_features(train_df, combo_type)

        if len(combo_train_df) == 0:
            print(f"❌ No training data for {combo_type}")
            continue

        # Train model
        models, features = train_combo_model(
            combo_train_df,
            combo_type,
            config['target_col']
        )

        # Save models
        models_dir = Path('outputs/models/combo')
        models_dir.mkdir(parents=True, exist_ok=True)

        model_file = models_dir / f'{combo_type}_models.pkl'
        joblib.dump({
            'models': models,
            'features': features,
            'config': config
        }, model_file)

        print(f"💾 Saved to: {model_file}")

        all_models[combo_type] = {
            'models': models,
            'features': features,
            'config': config
        }

        # Backtest on Weeks 10 & 11
        for week in [10, 11]:
            result = backtest_combo_prop(
                models,
                features,
                player_df,
                combo_type,
                config['target_col'],
                week,
                config['typical_line'],
                config['name']
            )

            if result:
                backtest_results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print(f"COMBO PROP TRAINING COMPLETE")
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

    print("✅ All combo models trained and backtested")
    print()

    print("New markets available:")
    print("  ✅ Pass + Rush Yards (for mobile QBs)")
    print("  ✅ Rec + Rush Yards (for pass-catching RBs)")
    print()
    print("Total new markets: 2")
    print()


if __name__ == '__main__':
    main()
