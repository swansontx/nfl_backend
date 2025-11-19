"""Train TD scorer props (Anytime TD, 2+ TDs, 3+ TDs).

These are binary classification models predicting whether a player will:
- Score ANY TD (rushing or receiving)
- Score 2+ TDs
- Score 3+ TDs

Very popular DraftKings markets with long odds (+150 to +800).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
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


def create_td_features(player_df):
    """Create features for TD scorer props.

    Args:
        player_df: Player stats DataFrame

    Returns:
        DataFrame with TD features
    """

    print(f"\n{'='*80}")
    print(f"CREATING TD SCORER FEATURES")
    print(f"{'='*80}\n")

    df = player_df.copy()

    # Only for skill positions (QB, RB, WR)
    df = df[df['position'].isin(['QB', 'RB', 'WR'])].copy()

    # Calculate total TDs
    df['total_tds'] = df['rushing_tds'] + df['receiving_tds'] + df.get('passing_tds', 0)

    # Binary targets
    df['scored_any_td'] = (df['total_tds'] > 0).astype(int)
    df['scored_2plus_tds'] = (df['total_tds'] >= 2).astype(int)
    df['scored_3plus_tds'] = (df['total_tds'] >= 3).astype(int)

    print(f"Skill position players: {len(df)}")
    print()

    # Show distribution
    print("TD Scoring Rates:")
    print(f"  Anytime TD: {df['scored_any_td'].mean():.1%}")
    print(f"  2+ TDs: {df['scored_2plus_tds'].mean():.1%}")
    print(f"  3+ TDs: {df['scored_3plus_tds'].mean():.1%}")
    print()

    # Sort by player and week
    df = df.sort_values(['player_id', 'week'])

    # Calculate TD rate features
    # Season TD rate (expanding mean)
    df['season_td_rate'] = df.groupby('player_id')['scored_any_td'].expanding().mean().reset_index(level=0, drop=True)

    # L3 TD rate
    df['l3_td_rate'] = df.groupby('player_id')['scored_any_td'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

    # Season total TDs per game
    df['season_tds_per_game'] = df.groupby('player_id')['total_tds'].expanding().mean().reset_index(level=0, drop=True)

    # L3 TDs per game
    df['l3_tds_per_game'] = df.groupby('player_id')['total_tds'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

    # Games played
    df['games_played'] = df.groupby('player_id').cumcount() + 1

    # Position dummies
    df['is_rb'] = (df['position'] == 'RB').astype(int)
    df['is_wr'] = (df['position'] == 'WR').astype(int)
    df['is_qb'] = (df['position'] == 'QB').astype(int)

    # Filter to 3+ games
    df = df[df['games_played'] >= 3].copy()

    print(f"Records with 3+ games: {len(df)}")
    print()

    return df


def train_td_scorer_model(train_df, prop_type, target_col):
    """Train TD scorer model.

    Args:
        train_df: Training DataFrame
        prop_type: Type of prop
        target_col: Target column name

    Returns:
        Trained model, features, metrics
    """

    print(f"\n{'='*80}")
    print(f"TRAINING: {prop_type.upper()}")
    print(f"{'='*80}\n")

    # Features
    feature_cols = [
        'season_td_rate',
        'l3_td_rate',
        'season_tds_per_game',
        'l3_tds_per_game',
        'games_played',
        'is_rb',
        'is_wr',
        'is_qb',
    ]

    X = train_df[feature_cols].fillna(0)
    y = train_df[target_col]

    print(f"Training set: {len(X)} records")
    print(f"Features: {feature_cols}")
    print(f"Positive rate: {y.mean():.1%}")
    print()

    # Train gradient boosting classifier
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
    print("Feature Importance:")
    feature_imp = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: x[1], reverse=True)
    for feat, imp in feature_imp:
        print(f"  {feat:25s}: {imp:.3f}")
    print()

    return model, feature_cols, {'accuracy': acc}


def backtest_td_scorer(model, features, player_df, prop_type, target_col, week, prop_name):
    """Backtest TD scorer prop."""

    print(f"\n{'='*80}")
    print(f"BACKTESTING: {prop_name.upper()} - WEEK {week}")
    print(f"{'='*80}\n")

    # Get week data
    week_df = player_df[
        (player_df['week'] == week) &
        (player_df['position'].isin(['QB', 'RB', 'WR']))
    ].copy()

    print(f"Week {week} players: {len(week_df)}")

    # Calculate features using ONLY data up to week-1
    historical_df = player_df[player_df['week'] < week].copy()

    if len(historical_df) == 0:
        print("❌ No historical data available for this week")
        return None

    # Calculate total TDs for historical
    historical_df['total_tds'] = historical_df['rushing_tds'] + historical_df['receiving_tds'] + historical_df.get('passing_tds', 0)
    historical_df['scored_any_td'] = (historical_df['total_tds'] > 0).astype(int)

    # For current week
    week_df['total_tds'] = week_df['rushing_tds'] + week_df['receiving_tds'] + week_df.get('passing_tds', 0)
    week_df['scored_any_td'] = (week_df['total_tds'] > 0).astype(int)
    week_df['scored_2plus_tds'] = (week_df['total_tds'] >= 2).astype(int)
    week_df['scored_3plus_tds'] = (week_df['total_tds'] >= 3).astype(int)

    # Calculate season stats
    historical_df = historical_df.sort_values(['player_id', 'week'])

    player_td_rate = historical_df.groupby('player_id')['scored_any_td'].mean().reset_index()
    player_td_rate.columns = ['player_id', 'season_td_rate']

    player_tds_pg = historical_df.groupby('player_id')['total_tds'].agg(['mean', 'count']).reset_index()
    player_tds_pg.columns = ['player_id', 'season_tds_per_game', 'games_played']

    # L3 stats
    l3_td_rate = historical_df.groupby('player_id')['scored_any_td'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    historical_df['l3_td_rate'] = l3_td_rate
    player_l3_rate = historical_df.groupby('player_id')['l3_td_rate'].last().reset_index()

    l3_tds_pg = historical_df.groupby('player_id')['total_tds'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    historical_df['l3_tds_per_game'] = l3_tds_pg
    player_l3_tds = historical_df.groupby('player_id')['l3_tds_per_game'].last().reset_index()

    # Merge features
    week_df = week_df.merge(player_td_rate, on='player_id', how='left')
    week_df = week_df.merge(player_tds_pg, on='player_id', how='left')
    week_df = week_df.merge(player_l3_rate, on='player_id', how='left')
    week_df = week_df.merge(player_l3_tds, on='player_id', how='left')

    # Fill NaN values for players with insufficient history
    # Use .get() to safely access columns that might not exist
    if 'games_played' in week_df.columns:
        week_df['games_played'] = week_df['games_played'].fillna(0)
    else:
        week_df['games_played'] = 0

    if 'season_td_rate' in week_df.columns:
        week_df['season_td_rate'] = week_df['season_td_rate'].fillna(0)
    else:
        week_df['season_td_rate'] = 0

    if 'season_tds_per_game' in week_df.columns:
        week_df['season_tds_per_game'] = week_df['season_tds_per_game'].fillna(0)
    else:
        week_df['season_tds_per_game'] = 0

    if 'l3_td_rate' in week_df.columns:
        week_df['l3_td_rate'] = week_df['l3_td_rate'].fillna(0)
    else:
        week_df['l3_td_rate'] = 0

    if 'l3_tds_per_game' in week_df.columns:
        week_df['l3_tds_per_game'] = week_df['l3_tds_per_game'].fillna(0)
    else:
        week_df['l3_tds_per_game'] = 0

    # Position dummies
    week_df['is_rb'] = (week_df['position'] == 'RB').astype(int)
    week_df['is_wr'] = (week_df['position'] == 'WR').astype(int)
    week_df['is_qb'] = (week_df['position'] == 'QB').astype(int)

    # Filter to 3+ games
    week_df = week_df[week_df['games_played'] >= 3].copy()

    print(f"Players with 3+ games: {len(week_df)}")

    if len(week_df) == 0:
        return None

    # Prepare features
    X = week_df[features].fillna(0)

    # Predict probabilities
    prob_scores = model.predict_proba(X)[:, 1]
    week_df['prob_scores_td'] = prob_scores

    # Actual results
    week_df['actual'] = week_df[target_col]

    # Calculate edge (typical odds are +150 = 40% implied, but varies by player)
    # Using a conservative 45% implied prob threshold
    week_df['edge'] = week_df['prob_scores_td'] - 0.45

    # Bets with >10% edge (higher threshold for binary props with variance)
    bets = week_df[week_df['edge'] > 0.10].copy()

    print(f"Bets with >10% edge: {len(bets)}")

    if len(bets) > 0:
        hit_rate = bets['actual'].mean()

        # Calculate ROI assuming +150 odds on average
        # Win: +150 = +1.5x stake
        # Loss: -1.0x stake
        roi = ((bets['actual'].sum() * 150) - ((~bets['actual'].astype(bool)).sum() * 100)) / (len(bets) * 100)

        print(f"Hit rate: {hit_rate:.1%}")
        print(f"ROI (assuming +150 odds): {roi:+.1%}")

        # Show top bets
        print(f"\nTop {min(10, len(bets))} bets:")
        top_bets = bets.nlargest(10, 'edge')[['player_name', 'position', 'prob_scores_td', 'edge', 'total_tds', 'actual']]
        for idx, row in top_bets.iterrows():
            status = '✅' if row['actual'] else '❌'
            print(f"  {status} {row['player_name']:25s} ({row['position']}): Prob={row['prob_scores_td']:.1%}, Edge={row['edge']:+.1%}, TDs={row['total_tds']:.0f}")

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
    """Train all TD scorer prop models."""

    print(f"\n{'='*80}")
    print(f"TRAINING TD SCORER PROP MODELS")
    print(f"{'='*80}\n")

    # Load player stats
    player_df = load_player_stats()

    if len(player_df) == 0:
        return

    # Create TD features
    player_with_features = create_td_features(player_df)

    # Training weeks
    training_weeks = list(range(1, 10))
    train_df = player_with_features[player_with_features['week'].isin(training_weeks)].copy()

    print(f"Training weeks: {training_weeks}")
    print(f"Training records: {len(train_df)}")
    print()

    # TD scorer props configuration
    td_props = {
        'anytime_td': {
            'target_col': 'scored_any_td',
            'name': 'Anytime TD Scorer',
        },
        '2plus_tds': {
            'target_col': 'scored_2plus_tds',
            'name': '2+ TDs',
        },
        '3plus_tds': {
            'target_col': 'scored_3plus_tds',
            'name': '3+ TDs',
        },
    }

    # Train each prop
    all_models = {}
    backtest_results = []

    for prop_type, config in td_props.items():
        print(f"\n{'#'*80}")
        print(f"# {config['name'].upper()}")
        print(f"{'#'*80}")

        # Train model
        model, features, metrics = train_td_scorer_model(
            train_df,
            prop_type,
            config['target_col']
        )

        # Save model
        models_dir = Path('outputs/models/td_scorer')
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
            result = backtest_td_scorer(
                model,
                features,
                player_df,  # Use original player_df, not filtered version
                prop_type,
                config['target_col'],
                week,
                config['name']
            )

            if result:
                backtest_results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print(f"TD SCORER PROP TRAINING COMPLETE")
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

    print("✅ All TD scorer models trained and backtested")
    print()

    print("New markets available:")
    print("  ✅ Anytime TD Scorer")
    print("  ✅ 2+ TDs")
    print("  ✅ 3+ TDs")
    print()
    print("Total new markets: 3")
    print()


if __name__ == '__main__':
    main()
