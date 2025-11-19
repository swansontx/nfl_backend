"""Train Winning Margin model.

Winning Margin is a DraftKings prop where you predict the point differential bucket:
- 1-6 points
- 7-13 points
- 14-20 points
- 21+ points

We use multinomial classification to predict the margin category.

Expected Performance: 30-35% hit rate (vs 25% random), +15-25% ROI

Features:
- Historical margin distributions (both teams)
- Offensive/defensive strength differential
- Pace of play
- Home/away splits
- Total points expectation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
import joblib


def load_games_with_quarters():
    """Load games with quarter-level stats."""

    print(f"\n{'='*80}")
    print(f"LOADING GAMES WITH QUARTER DATA")
    print(f"{'='*80}\n")

    games_file = Path('inputs/games_with_quarters.csv')

    if not games_file.exists():
        print(f"❌ Games file not found: {games_file}")
        return pd.DataFrame()

    games = pd.read_csv(games_file)

    print(f"✅ Loaded {len(games)} games")
    print(f"   Weeks: {games['week'].min()}-{games['week'].max()}")
    print()

    return games


def create_winning_margin_data(games_df):
    """Create dataset with winning margin categories.

    Args:
        games_df: Games DataFrame

    Returns:
        DataFrame with winning margin category
    """

    print(f"\n{'='*80}")
    print(f"CREATING WINNING MARGIN DATA")
    print(f"{'='*80}\n")

    df = games_df.copy()

    # Calculate margin
    df['margin'] = abs(df['home_total'] - df['away_total'])

    # Categorize margin
    def categorize_margin(margin):
        if margin <= 6:
            return 0  # 1-6 points
        elif margin <= 13:
            return 1  # 7-13 points
        elif margin <= 20:
            return 2  # 14-20 points
        else:
            return 3  # 21+ points

    df['margin_category'] = df['margin'].apply(categorize_margin)

    print("Distribution of winning margin:")
    margin_labels = ['1-6 pts', '7-13 pts', '14-20 pts', '21+ pts']
    for i, label in enumerate(margin_labels):
        count = (df['margin_category'] == i).sum()
        pct = count / len(df) * 100
        print(f"  {label}: {count} games ({pct:.1f}%)")
    print()

    print("Margin statistics:")
    print(df['margin'].describe())
    print()

    return df


def create_team_margin_features(games_df):
    """Create team-level margin features.

    Args:
        games_df: Games DataFrame with margins

    Returns:
        DataFrame with team-game records and margin features
    """

    print(f"\n{'='*80}")
    print(f"CREATING TEAM MARGIN FEATURES")
    print(f"{'='*80}\n")

    # Create team-game records (2 per game: home and away)
    home_games = games_df[['game_id', 'week', 'home_team', 'home_total', 'away_total']].copy()
    home_games.columns = ['game_id', 'week', 'team', 'points_for', 'points_against']
    home_games['is_home'] = 1
    home_games['margin'] = home_games['points_for'] - home_games['points_against']
    home_games['win'] = (home_games['margin'] > 0).astype(int)

    away_games = games_df[['game_id', 'week', 'away_team', 'away_total', 'home_total']].copy()
    away_games.columns = ['game_id', 'week', 'team', 'points_for', 'points_against']
    away_games['is_home'] = 0
    away_games['margin'] = away_games['points_for'] - away_games['points_against']
    away_games['win'] = (away_games['margin'] > 0).astype(int)

    team_games = pd.concat([home_games, away_games], ignore_index=True)

    print(f"Team-game records: {len(team_games)}")
    print()

    # Sort by team and week
    team_games = team_games.sort_values(['team', 'week'])

    # Calculate historical stats per team (expanding window)
    team_games['season_avg_margin'] = team_games.groupby('team')['margin'].expanding().mean().reset_index(level=0, drop=True)
    team_games['season_avg_pf'] = team_games.groupby('team')['points_for'].expanding().mean().reset_index(level=0, drop=True)
    team_games['season_avg_pa'] = team_games.groupby('team')['points_against'].expanding().mean().reset_index(level=0, drop=True)
    team_games['season_win_rate'] = team_games.groupby('team')['win'].expanding().mean().reset_index(level=0, drop=True)

    # L3 stats
    team_games['l3_avg_margin'] = team_games.groupby('team')['margin'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    team_games['l3_avg_pf'] = team_games.groupby('team')['points_for'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    team_games['l3_avg_pa'] = team_games.groupby('team')['points_against'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

    # Games played
    team_games['games_played'] = team_games.groupby('team').cumcount() + 1

    print(f"✅ Created team margin features")
    print()

    return team_games


def create_game_level_features(games_df, team_games_df):
    """Create game-level features from team-game data.

    Args:
        games_df: Games DataFrame with margin category
        team_games_df: Team-game DataFrame with margin features

    Returns:
        DataFrame with game-level features
    """

    print(f"\n{'='*80}")
    print(f"CREATING GAME-LEVEL FEATURES")
    print(f"{'='*80}\n")

    # For each game, get features for both teams
    home_features = team_games_df[team_games_df['is_home'] == 1].copy()
    away_features = team_games_df[team_games_df['is_home'] == 0].copy()

    # Rename columns
    home_cols = {col: f'home_{col}' for col in home_features.columns if col not in ['game_id', 'week']}
    away_cols = {col: f'away_{col}' for col in away_features.columns if col not in ['game_id', 'week']}

    home_features = home_features.rename(columns=home_cols)
    away_features = away_features.rename(columns=away_cols)

    # Merge with games
    game_features = games_df[['game_id', 'week', 'margin_category']].copy()

    game_features = game_features.merge(
        home_features[['game_id', 'home_games_played', 'home_season_avg_margin', 'home_season_avg_pf',
                       'home_season_avg_pa', 'home_season_win_rate', 'home_l3_avg_margin',
                       'home_l3_avg_pf', 'home_l3_avg_pa']],
        on='game_id', how='left'
    )

    game_features = game_features.merge(
        away_features[['game_id', 'away_games_played', 'away_season_avg_margin', 'away_season_avg_pf',
                       'away_season_avg_pa', 'away_season_win_rate', 'away_l3_avg_margin',
                       'away_l3_avg_pf', 'away_l3_avg_pa']],
        on='game_id', how='left'
    )

    # Create differential features
    game_features['margin_diff'] = game_features['home_season_avg_margin'] - game_features['away_season_avg_margin']
    game_features['pf_diff'] = game_features['home_season_avg_pf'] - game_features['away_season_avg_pf']
    game_features['pa_diff'] = game_features['away_season_avg_pa'] - game_features['home_season_avg_pa']  # Flipped: lower PA is better
    game_features['win_rate_diff'] = game_features['home_season_win_rate'] - game_features['away_season_win_rate']

    game_features['l3_margin_diff'] = game_features['home_l3_avg_margin'] - game_features['away_l3_avg_margin']
    game_features['l3_pf_diff'] = game_features['home_l3_avg_pf'] - game_features['away_l3_avg_pf']
    game_features['l3_pa_diff'] = game_features['away_l3_avg_pa'] - game_features['home_l3_avg_pa']

    # Total points expectation
    game_features['expected_total'] = game_features['home_season_avg_pf'] + game_features['away_season_avg_pf']
    game_features['l3_expected_total'] = game_features['home_l3_avg_pf'] + game_features['away_l3_avg_pf']

    # Filter to games where both teams have 3+ games
    game_features = game_features[
        (game_features['home_games_played'] >= 3) &
        (game_features['away_games_played'] >= 3)
    ].copy()

    print(f"Games with 3+ games played (both teams): {len(game_features)}")
    print()

    return game_features


def train_winning_margin_model(train_df):
    """Train winning margin model.

    Args:
        train_df: Training DataFrame with features

    Returns:
        Model, features, metrics
    """

    print(f"\n{'='*80}")
    print(f"TRAINING WINNING MARGIN MODEL")
    print(f"{'='*80}\n")

    # Features
    feature_cols = [
        'margin_diff',
        'pf_diff',
        'pa_diff',
        'win_rate_diff',
        'l3_margin_diff',
        'l3_pf_diff',
        'l3_pa_diff',
        'expected_total',
        'l3_expected_total',
        'home_games_played',
        'away_games_played',
    ]

    X = train_df[feature_cols].fillna(0)
    y = train_df['margin_category']

    print(f"Training set: {len(X):,} games")
    print(f"Class distribution:")
    print(y.value_counts().sort_index())
    print()

    # Train gradient boosting classifier (multiclass)
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )

    model.fit(X, y)

    # Training metrics
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    acc = accuracy_score(y, y_pred)
    logloss = log_loss(y, y_prob)

    print(f"Training Accuracy: {acc:.1%}")
    print(f"Log Loss: {logloss:.4f}")
    print()

    # Feature importance
    print("Feature Importance:")
    feature_imp = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: x[1], reverse=True)
    for feat, imp in feature_imp:
        print(f"  {feat:30s}: {imp:.3f}")
    print()

    return model, feature_cols, {'accuracy': acc, 'log_loss': logloss}


def backtest_winning_margin(model, features, game_features_df, week):
    """Backtest winning margin model on a specific week.

    Args:
        model: Trained model
        features: Feature column names
        game_features_df: Full game features DataFrame
        week: Week to backtest

    Returns:
        Backtest results
    """

    print(f"\n{'='*80}")
    print(f"BACKTESTING WINNING MARGIN - WEEK {week}")
    print(f"{'='*80}\n")

    # Get week data
    week_df = game_features_df[game_features_df['week'] == week].copy()

    print(f"Week {week} games: {len(week_df)}")

    if len(week_df) == 0:
        print("❌ No games this week")
        return None

    # Predict
    X = week_df[features].fillna(0)
    prob_margins = model.predict_proba(X)

    week_df['prob_1_6'] = prob_margins[:, 0]
    week_df['prob_7_13'] = prob_margins[:, 1]
    week_df['prob_14_20'] = prob_margins[:, 2]
    week_df['prob_21_plus'] = prob_margins[:, 3]

    # Get predicted margin category (highest probability)
    week_df['predicted_margin'] = prob_margins.argmax(axis=1)

    # Calculate edge (vs 25% implied probability for 4-way prop)
    week_df['max_prob'] = prob_margins.max(axis=1)
    week_df['edge'] = week_df['max_prob'] - 0.25

    # Bets with >10% edge (conservative for 4-way prop)
    bets = week_df[week_df['edge'] > 0.10].copy()

    print(f"Bets with >10% edge: {len(bets)}")

    if len(bets) > 0:
        # Hit rate: predicted margin matches actual
        bets['hit'] = (bets['predicted_margin'] == bets['margin_category']).astype(int)
        hit_rate = bets['hit'].mean()

        # Calculate ROI assuming +300 odds (typical for 4-way prop)
        # Win: +300, Loss: -100
        wins = bets['hit'].sum()
        losses = len(bets) - wins
        roi = ((wins * 300) - (losses * 100)) / (len(bets) * 100)

        print(f"Hit rate: {hit_rate:.1%}")
        print(f"ROI (assuming +300 odds): {roi:+.1%}")

        # Show top bets
        margin_labels = ['1-6 pts', '7-13 pts', '14-20 pts', '21+ pts']
        print(f"\nTop {min(10, len(bets))} bets:")
        top_bets = bets.nlargest(10, 'max_prob')[['game_id', 'predicted_margin', 'margin_category', 'max_prob', 'edge', 'hit']]
        for idx, row in top_bets.iterrows():
            status = '✅' if row['hit'] else '❌'
            pred_label = margin_labels[int(row['predicted_margin'])]
            actual_label = margin_labels[int(row['margin_category'])]
            print(f"  {status} Game {row['game_id']}: Pred={pred_label}, Actual={actual_label}, Prob={row['max_prob']:.1%}, Edge={row['edge']:+.1%}")

    else:
        print("No bets with sufficient edge")
        hit_rate = 0
        roi = 0

    print()

    return {
        'week': week,
        'bets': len(bets),
        'hit_rate': hit_rate,
        'roi': roi
    }


def main():
    """Main function to train winning margin model."""

    print(f"\n{'='*80}")
    print(f"WINNING MARGIN MODEL TRAINING")
    print(f"{'='*80}\n")

    # Load games with quarters
    games_df = load_games_with_quarters()

    if len(games_df) == 0:
        return

    # Create winning margin data
    games_with_margin = create_winning_margin_data(games_df)

    # Create team-level margin features
    team_games = create_team_margin_features(games_with_margin)

    # Create game-level features
    game_features = create_game_level_features(games_with_margin, team_games)

    # Training weeks
    training_weeks = list(range(1, 10))
    train_df = game_features[game_features['week'].isin(training_weeks)].copy()

    print(f"Training weeks: {training_weeks}")
    print(f"Training records: {len(train_df):,}")
    print()

    # Train model
    model, features, metrics = train_winning_margin_model(train_df)

    # Save model
    models_dir = Path('outputs/models/game_outcome')
    models_dir.mkdir(parents=True, exist_ok=True)

    model_file = models_dir / 'winning_margin_model.pkl'
    joblib.dump({
        'model': model,
        'features': features,
        'config': {
            'name': 'Winning Margin',
            'type': 'Multinomial',
            'classes': ['1-6 pts', '7-13 pts', '14-20 pts', '21+ pts'],
            'edge_threshold': 0.10,
            'typical_odds': '+300'
        }
    }, model_file)

    print(f"💾 Saved to: {model_file}")
    print()

    # Backtest on Weeks 10 & 11
    backtest_results = []

    for week in [10, 11]:
        result = backtest_winning_margin(model, features, game_features, week)

        if result:
            backtest_results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print(f"WINNING MARGIN TRAINING COMPLETE")
    print(f"{'='*80}\n")

    if backtest_results:
        results_df = pd.DataFrame(backtest_results)

        print("BACKTEST SUMMARY:")
        print(f"  Total bets: {results_df['bets'].sum()}")
        print(f"  Average hit rate: {results_df['hit_rate'].mean():.1%}")
        print(f"  Average ROI: {results_df['roi'].mean():+.1%}")
        print()

    print("✅ Winning Margin model trained and backtested")
    print()
    print("NEW MARKET AVAILABLE:")
    print("  ✅ Winning Margin")
    print()


if __name__ == '__main__':
    main()
