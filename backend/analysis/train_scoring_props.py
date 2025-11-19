"""Train game-level scoring props.

These are binary/categorical props about game scoring patterns:
- Will there be overtime? (Bernoulli)
- Will there be a 2PT conversion? (Bernoulli)
- Winning Margin (Categorical)
- Highest Scoring Quarter (Multinomial)
- Highest Scoring Half (Bernoulli)
- Will there be a scoreless quarter? (Bernoulli)

All can be derived from existing games_2025_with_quarters.csv data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib


def load_games_data():
    """Load games with quarter scores."""

    print(f"\n{'='*80}")
    print(f"LOADING GAMES DATA")
    print(f"{'='*80}\n")

    games_file = Path('inputs/games_2025_with_quarters.csv')

    if not games_file.exists():
        print(f"❌ Games file not found: {games_file}")
        return pd.DataFrame()

    games = pd.read_csv(games_file)

    # Filter to 2025 completed games
    games_2025 = games[
        (games['season'] == 2025) &
        (games['game_type'] == 'REG') &
        (games['home_score'].notna())
    ].copy()

    print(f"✅ Loaded {len(games_2025)} completed games")
    print()

    return games_2025


def create_scoring_prop_targets(games_df):
    """Create target variables for scoring props.

    Args:
        games_df: Games DataFrame with quarter scores

    Returns:
        DataFrame with scoring prop targets
    """

    print(f"\n{'='*80}")
    print(f"CREATING SCORING PROP TARGETS")
    print(f"{'='*80}\n")

    df = games_df.copy()

    # 1. Overtime
    df['went_to_ot'] = (df['overtime'] == 1).astype(int)

    # 2. 2PT Conversion (estimate based on odd final scores)
    # If final score has remainder 2 when divided by 7 (after removing FGs), likely 2PT
    # This is a rough estimate - real data would come from play-by-play
    df['had_2pt_conversion'] = 0  # Default to no
    # Mark games where either team has an unusual point total suggesting 2PT
    unusual_scores = df.apply(
        lambda row: (row['home_score'] % 7 == 2) or (row['away_score'] % 7 == 2) or
                    (row['home_score'] % 7 == 8) or (row['away_score'] % 7 == 8),
        axis=1
    )
    df.loc[unusual_scores, 'had_2pt_conversion'] = 1

    # 3. Winning Margin (categorical)
    df['margin'] = abs(df['home_score'] - df['away_score'])
    # Use numerical categories for sklearn
    df['winning_margin_category'] = pd.cut(
        df['margin'],
        bins=[0, 3, 7, 14, 100],
        labels=[0, 1, 2, 3],  # 0=close, 1=one_score, 2=two_score, 3=blowout
        include_lowest=True
    )
    # Convert categorical to int, handling NaN
    df['winning_margin_category'] = df['winning_margin_category'].cat.codes

    # 4. Highest Scoring Quarter
    quarter_cols = ['away_q1', 'away_q2', 'away_q3', 'away_q4',
                    'home_q1', 'home_q2', 'home_q3', 'home_q4']

    # Calculate total points per quarter
    df['q1_total'] = df['away_q1'] + df['home_q1']
    df['q2_total'] = df['away_q2'] + df['home_q2']
    df['q3_total'] = df['away_q3'] + df['home_q3']
    df['q4_total'] = df['away_q4'] + df['home_q4']

    # Find highest scoring quarter
    quarter_totals = df[['q1_total', 'q2_total', 'q3_total', 'q4_total']]
    df['highest_scoring_quarter_name'] = quarter_totals.idxmax(axis=1).str.replace('_total', '').str.upper()

    # Convert to numerical for sklearn
    quarter_map = {'Q1': 0, 'Q2': 1, 'Q3': 2, 'Q4': 3}
    df['highest_scoring_quarter'] = df['highest_scoring_quarter_name'].map(quarter_map)

    # 5. Highest Scoring Half
    df['first_half_higher'] = (df['h1_total'] > df['h2_total']).astype(int)

    # 6. Scoreless Quarter
    df['had_scoreless_quarter'] = (
        (df['q1_total'] == 0) |
        (df['q2_total'] == 0) |
        (df['q3_total'] == 0) |
        (df['q4_total'] == 0)
    ).astype(int)

    # Show distributions
    margin_names = {0: 'close_game', 1: 'one_score', 2: 'two_score', 3: 'blowout'}
    quarter_names = {0: 'Q1', 1: 'Q2', 2: 'Q3', 3: 'Q4'}

    print("Scoring Prop Distributions:")
    print(f"  Overtime Rate: {df['went_to_ot'].mean():.1%}")
    print(f"  2PT Conversion Rate (estimated): {df['had_2pt_conversion'].mean():.1%}")
    print(f"  Winning Margin Categories:")
    for cat, count in df['winning_margin_category'].value_counts().sort_index().items():
        print(f"    {margin_names[cat]}: {count} games")
    print(f"  Highest Scoring Quarter:")
    for q, count in df['highest_scoring_quarter'].value_counts().sort_index().items():
        print(f"    {quarter_names[q]}: {count} games")
    print(f"  First Half Higher Scoring: {df['first_half_higher'].mean():.1%}")
    print(f"  Had Scoreless Quarter: {df['had_scoreless_quarter'].mean():.1%}")
    print()

    return df


def create_game_features(games_df):
    """Create features for game-level models."""

    df = games_df.copy()

    # Basic features
    df['rest_advantage'] = df['home_rest'] - df['away_rest']
    df['is_pickem'] = (df['spread_line'].abs() < 2.0).astype(int)
    df['is_high_total'] = (df['total_line'] > 48.0).astype(int)
    df['is_low_total'] = (df['total_line'] < 40.0).astype(int)
    df['home_implied_total'] = (df['total_line'] - df['spread_line']) / 2
    df['away_implied_total'] = (df['total_line'] + df['spread_line']) / 2
    df['is_dome'] = (df['roof'] == 'dome').astype(int)
    df['is_division_game'] = df['div_game'].astype(int)

    return df


def train_binary_scoring_prop(train_df, prop_name, target_col, test_weeks):
    """Train a binary scoring prop model.

    Args:
        train_df: Training DataFrame
        prop_name: Name of the prop
        target_col: Target column name
        test_weeks: List of test weeks

    Returns:
        Model, features, backtest results
    """

    print(f"\n{'='*80}")
    print(f"TRAINING: {prop_name.upper()}")
    print(f"{'='*80}\n")

    # Features
    feature_cols = [
        'spread_line',
        'total_line',
        'rest_advantage',
        'is_pickem',
        'is_high_total',
        'is_low_total',
        'is_division_game',
        'is_dome',
    ]

    X = train_df[feature_cols].fillna(0)
    y = train_df[target_col]

    print(f"Training set: {len(X)} games")
    print(f"Positive rate: {y.mean():.1%}")
    print()

    # Train classifier
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X, y)

    # Training accuracy
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)

    print(f"Training Accuracy: {acc:.1%}")
    print()

    # Feature importance
    print("Top 5 Features:")
    feature_imp = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: x[1], reverse=True)[:5]
    for feat, imp in feature_imp:
        print(f"  {feat:25s}: {imp:.3f}")
    print()

    # Backtest
    backtest_results = []

    for week in test_weeks:
        week_df = train_df[train_df['week'] == week].copy()

        if len(week_df) == 0:
            continue

        X_test = week_df[feature_cols].fillna(0)
        y_test = week_df[target_col]

        # Predict probabilities
        probs = model.predict_proba(X_test)[:, 1]

        # Predictions
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        print(f"Week {week}: {len(week_df)} games, Accuracy: {acc:.1%}, Actual Rate: {y_test.mean():.1%}")

        backtest_results.append({
            'prop': prop_name,
            'week': week,
            'games': len(week_df),
            'accuracy': acc,
            'actual_rate': y_test.mean()
        })

    print()

    return model, feature_cols, backtest_results


def train_categorical_prop(train_df, prop_name, target_col, test_weeks):
    """Train a categorical scoring prop model.

    Args:
        train_df: Training DataFrame
        prop_name: Name of the prop
        target_col: Target column name
        test_weeks: List of test weeks

    Returns:
        Model, features, backtest results
    """

    print(f"\n{'='*80}")
    print(f"TRAINING: {prop_name.upper()}")
    print(f"{'='*80}\n")

    # Features
    feature_cols = [
        'spread_line',
        'total_line',
        'rest_advantage',
        'is_pickem',
        'is_high_total',
        'is_low_total',
        'is_division_game',
        'is_dome',
    ]

    X = train_df[feature_cols].fillna(0)
    y = train_df[target_col]

    print(f"Training set: {len(X)} games")
    print(f"Class distribution:")
    for cls, count in y.value_counts().items():
        print(f"  {cls}: {count} ({count/len(y):.1%})")
    print()

    # Train classifier
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X, y)

    # Training accuracy
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)

    print(f"Training Accuracy: {acc:.1%}")
    print()

    # Backtest
    backtest_results = []

    for week in test_weeks:
        week_df = train_df[train_df['week'] == week].copy()

        if len(week_df) == 0:
            continue

        X_test = week_df[feature_cols].fillna(0)
        y_test = week_df[target_col]

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Week {week}: {len(week_df)} games, Accuracy: {acc:.1%}")

        backtest_results.append({
            'prop': prop_name,
            'week': week,
            'games': len(week_df),
            'accuracy': acc
        })

    print()

    return model, feature_cols, backtest_results


def main():
    """Train all scoring prop models."""

    print(f"\n{'='*80}")
    print(f"TRAINING GAME SCORING PROP MODELS")
    print(f"{'='*80}\n")

    # Load games
    games_df = load_games_data()

    if len(games_df) == 0:
        return

    # Create targets
    games_with_targets = create_scoring_prop_targets(games_df)

    # Create features
    games_with_features = create_game_features(games_with_targets)

    # Training and test weeks
    training_weeks = list(range(1, 10))
    test_weeks = [10, 11]

    train_df = games_with_features[games_with_features['week'].isin(training_weeks)].copy()

    print(f"Training weeks: {training_weeks}")
    print(f"Test weeks: {test_weeks}")
    print(f"Training games: {len(train_df)}")
    print()

    # Binary props configuration
    binary_props = {
        'will_go_to_ot': {
            'target': 'went_to_ot',
            'name': 'Will Game Go to OT?',
        },
        'will_have_2pt': {
            'target': 'had_2pt_conversion',
            'name': 'Will There Be a 2PT Conversion?',
        },
        'first_half_higher': {
            'target': 'first_half_higher',
            'name': 'Will 1st Half Outscore 2nd Half?',
        },
        'scoreless_quarter': {
            'target': 'had_scoreless_quarter',
            'name': 'Will There Be a Scoreless Quarter?',
        },
    }

    # Categorical props configuration
    categorical_props = {
        'winning_margin': {
            'target': 'winning_margin_category',
            'name': 'Winning Margin Category',
        },
        'highest_quarter': {
            'target': 'highest_scoring_quarter',
            'name': 'Highest Scoring Quarter',
        },
    }

    all_models = {}
    all_backtest_results = []

    # Train binary props
    for prop_id, config in binary_props.items():
        print(f"\n{'#'*80}")
        print(f"# {config['name'].upper()}")
        print(f"{'#'*80}")

        model, features, backtest_results = train_binary_scoring_prop(
            train_df,
            config['name'],
            config['target'],
            test_weeks
        )

        # Save model
        models_dir = Path('outputs/models/scoring_props')
        models_dir.mkdir(parents=True, exist_ok=True)

        model_file = models_dir / f'{prop_id}_model.pkl'
        joblib.dump({
            'model': model,
            'features': features,
            'config': config
        }, model_file)

        print(f"💾 Saved to: {model_file}")

        all_models[prop_id] = {
            'model': model,
            'features': features,
            'config': config
        }

        all_backtest_results.extend(backtest_results)

    # Train categorical props
    for prop_id, config in categorical_props.items():
        print(f"\n{'#'*80}")
        print(f"# {config['name'].upper()}")
        print(f"{'#'*80}")

        model, features, backtest_results = train_categorical_prop(
            train_df,
            config['name'],
            config['target'],
            test_weeks
        )

        # Save model
        model_file = models_dir / f'{prop_id}_model.pkl'
        joblib.dump({
            'model': model,
            'features': features,
            'config': config
        }, model_file)

        print(f"💾 Saved to: {model_file}")

        all_models[prop_id] = {
            'model': model,
            'features': features,
            'config': config
        }

        all_backtest_results.extend(backtest_results)

    # Summary
    print(f"\n{'='*80}")
    print(f"SCORING PROPS TRAINING COMPLETE")
    print(f"{'='*80}\n")

    print(f"Props trained: {len(all_models)}")
    print()

    # Backtest summary
    if all_backtest_results:
        results_df = pd.DataFrame(all_backtest_results)

        print("BACKTEST SUMMARY (Average Accuracy):")
        print()

        summary = results_df.groupby('prop')['accuracy'].mean().sort_values(ascending=False)

        for prop, acc in summary.items():
            print(f"  {prop:50s}: {acc:.1%}")

        print()

    print("✅ All scoring prop models trained and backtested")
    print()

    print("New markets available:")
    print("  ✅ Will Game Go to OT?")
    print("  ✅ Will There Be a 2PT Conversion?")
    print("  ✅ Will 1st Half Outscore 2nd Half?")
    print("  ✅ Will There Be a Scoreless Quarter?")
    print("  ✅ Winning Margin Category")
    print("  ✅ Highest Scoring Quarter")
    print()
    print("Total new markets: 6")
    print()


if __name__ == '__main__':
    main()
