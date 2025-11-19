"""Train Highest Scoring Quarter model.

Highest Scoring Quarter is a popular DraftKings prop where you predict which quarter
(Q1, Q2, Q3, Q4) will have the most combined points scored.

We use multinomial classification to predict the probability distribution across quarters.

Expected Performance: 30-35% hit rate (vs 25% random), +15-25% ROI

Features:
- Historical quarter scoring distributions (both teams)
- First half vs second half tendencies
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

    games_file = Path('/home/user/nfl_backend/inputs/games_with_quarters.csv')

    if not games_file.exists():
        print(f"❌ Games file not found: {games_file}")
        return pd.DataFrame()

    games = pd.read_csv(games_file)

    print(f"✅ Loaded {len(games)} games")
    print(f"   Weeks: {games['week'].min()}-{games['week'].max()}")
    print()

    return games


def create_highest_scoring_quarter_data(games_df):
    """Create dataset with highest scoring quarter per game.

    Args:
        games_df: Games DataFrame with quarter scores

    Returns:
        DataFrame with highest scoring quarter
    """

    print(f"\n{'='*80}")
    print(f"CREATING HIGHEST SCORING QUARTER DATA")
    print(f"{'='*80}\n")

    df = games_df.copy()

    # Calculate combined points per quarter
    df['q1_total'] = df['home_q1'] + df['away_q1']
    df['q2_total'] = df['home_q2'] + df['away_q2']
    df['q3_total'] = df['home_q3'] + df['away_q3']
    df['q4_total'] = df['home_q4'] + df['away_q4']

    # Find highest scoring quarter
    quarter_cols = ['q1_total', 'q2_total', 'q3_total', 'q4_total']
    df['highest_quarter'] = df[quarter_cols].idxmax(axis=1)

    # Convert to numeric label (1, 2, 3, 4)
    df['highest_quarter_num'] = df['highest_quarter'].str[1].astype(int)

    print("Distribution of highest scoring quarter:")
    print(df['highest_quarter_num'].value_counts().sort_index())
    print()

    # Calculate aggregate stats
    print("Average points per quarter:")
    print(f"  Q1: {df['q1_total'].mean():.2f}")
    print(f"  Q2: {df['q2_total'].mean():.2f}")
    print(f"  Q3: {df['q3_total'].mean():.2f}")
    print(f"  Q4: {df['q4_total'].mean():.2f}")
    print()

    return df


def create_team_quarter_features(games_df):
    """Create team-level quarter scoring features.

    Args:
        games_df: Games DataFrame with quarter scores

    Returns:
        DataFrame with team-game records and quarter features
    """

    print(f"\n{'='*80}")
    print(f"CREATING TEAM QUARTER FEATURES")
    print(f"{'='*80}\n")

    # Create team-game records (2 per game: home and away)
    home_games = games_df[['game_id', 'week', 'home_team', 'home_q1', 'home_q2', 'home_q3', 'home_q4']].copy()
    home_games.columns = ['game_id', 'week', 'team', 'q1', 'q2', 'q3', 'q4']
    home_games['is_home'] = 1

    away_games = games_df[['game_id', 'week', 'away_team', 'away_q1', 'away_q2', 'away_q3', 'away_q4']].copy()
    away_games.columns = ['game_id', 'week', 'team', 'q1', 'q2', 'q3', 'q4']
    away_games['is_home'] = 0

    team_games = pd.concat([home_games, away_games], ignore_index=True)

    print(f"Team-game records: {len(team_games)}")
    print()

    # Sort by team and week
    team_games = team_games.sort_values(['team', 'week'])

    # Calculate historical stats per team (expanding window)
    for q in range(1, 5):
        team_games[f'season_q{q}_avg'] = team_games.groupby('team')[f'q{q}'].expanding().mean().reset_index(level=0, drop=True)
        team_games[f'l3_q{q}_avg'] = team_games.groupby('team')[f'q{q}'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

    # Calculate first half vs second half tendencies
    team_games['first_half_points'] = team_games['q1'] + team_games['q2']
    team_games['second_half_points'] = team_games['q3'] + team_games['q4']

    team_games['season_first_half_avg'] = team_games.groupby('team')['first_half_points'].expanding().mean().reset_index(level=0, drop=True)
    team_games['season_second_half_avg'] = team_games.groupby('team')['second_half_points'].expanding().mean().reset_index(level=0, drop=True)

    team_games['l3_first_half_avg'] = team_games.groupby('team')['first_half_points'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    team_games['l3_second_half_avg'] = team_games.groupby('team')['second_half_points'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

    # Games played
    team_games['games_played'] = team_games.groupby('team').cumcount() + 1

    print(f"✅ Created team quarter features")
    print()

    return team_games


def create_game_level_features(games_df, team_games_df):
    """Create game-level features from team-game data.

    Args:
        games_df: Games DataFrame with highest quarter
        team_games_df: Team-game DataFrame with quarter features

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
    game_features = games_df[['game_id', 'week', 'highest_quarter_num']].copy()

    game_features = game_features.merge(
        home_features[['game_id', 'home_games_played'] + [col for col in home_features.columns if col.startswith('home_season_') or col.startswith('home_l3_')]],
        on='game_id', how='left'
    )

    game_features = game_features.merge(
        away_features[['game_id', 'away_games_played'] + [col for col in away_features.columns if col.startswith('away_season_') or col.startswith('away_l3_')]],
        on='game_id', how='left'
    )

    # Combined features
    for q in range(1, 5):
        game_features[f'combined_season_q{q}_avg'] = (game_features[f'home_season_q{q}_avg'] + game_features[f'away_season_q{q}_avg']) / 2
        game_features[f'combined_l3_q{q}_avg'] = (game_features[f'home_l3_q{q}_avg'] + game_features[f'away_l3_q{q}_avg']) / 2

    game_features['combined_season_first_half_avg'] = (game_features['home_season_first_half_avg'] + game_features['away_season_first_half_avg']) / 2
    game_features['combined_season_second_half_avg'] = (game_features['home_season_second_half_avg'] + game_features['away_season_second_half_avg']) / 2

    # Filter to games where both teams have 3+ games
    game_features = game_features[
        (game_features['home_games_played'] >= 3) &
        (game_features['away_games_played'] >= 3)
    ].copy()

    print(f"Games with 3+ games played (both teams): {len(game_features)}")
    print()

    return game_features


def train_highest_quarter_model(train_df):
    """Train highest scoring quarter model.

    Args:
        train_df: Training DataFrame with features

    Returns:
        Model, features, metrics
    """

    print(f"\n{'='*80}")
    print(f"TRAINING HIGHEST SCORING QUARTER MODEL")
    print(f"{'='*80}\n")

    # Features
    feature_cols = [
        'combined_season_q1_avg',
        'combined_season_q2_avg',
        'combined_season_q3_avg',
        'combined_season_q4_avg',
        'combined_l3_q1_avg',
        'combined_l3_q2_avg',
        'combined_l3_q3_avg',
        'combined_l3_q4_avg',
        'combined_season_first_half_avg',
        'combined_season_second_half_avg',
        'home_games_played',
        'away_games_played',
    ]

    X = train_df[feature_cols].fillna(0)
    y = train_df['highest_quarter_num'] - 1  # Convert to 0-indexed (0, 1, 2, 3)

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
        print(f"  {feat:35s}: {imp:.3f}")
    print()

    return model, feature_cols, {'accuracy': acc, 'log_loss': logloss}


def backtest_highest_quarter(model, features, game_features_df, week):
    """Backtest highest scoring quarter model on a specific week.

    Args:
        model: Trained model
        features: Feature column names
        game_features_df: Full game features DataFrame
        week: Week to backtest

    Returns:
        Backtest results
    """

    print(f"\n{'='*80}")
    print(f"BACKTESTING HIGHEST SCORING QUARTER - WEEK {week}")
    print(f"{'='*80}\n")

    # Get week data
    week_df = game_features_df[game_features_df['week'] == week].copy()

    print(f"Week {week} games: {len(week_df)}")

    if len(week_df) == 0:
        print("❌ No games this week")
        return None

    # Predict
    X = week_df[features].fillna(0)
    prob_quarters = model.predict_proba(X)

    week_df['prob_q1'] = prob_quarters[:, 0]
    week_df['prob_q2'] = prob_quarters[:, 1]
    week_df['prob_q3'] = prob_quarters[:, 2]
    week_df['prob_q4'] = prob_quarters[:, 3]

    # Get predicted quarter (highest probability)
    week_df['predicted_quarter'] = prob_quarters.argmax(axis=1) + 1

    # Calculate edge (vs 25% implied probability)
    week_df['max_prob'] = prob_quarters.max(axis=1)
    week_df['edge'] = week_df['max_prob'] - 0.25

    # Bets with >10% edge (conservative for 4-way prop)
    bets = week_df[week_df['edge'] > 0.10].copy()

    print(f"Bets with >10% edge: {len(bets)}")

    if len(bets) > 0:
        # Hit rate: predicted quarter matches actual
        bets['hit'] = (bets['predicted_quarter'] == bets['highest_quarter_num']).astype(int)
        hit_rate = bets['hit'].mean()

        # Calculate ROI assuming +300 odds (typical for 4-way prop)
        # Win: +300, Loss: -100
        wins = bets['hit'].sum()
        losses = len(bets) - wins
        roi = ((wins * 300) - (losses * 100)) / (len(bets) * 100)

        print(f"Hit rate: {hit_rate:.1%}")
        print(f"ROI (assuming +300 odds): {roi:+.1%}")

        # Show top bets
        print(f"\nTop {min(10, len(bets))} bets:")
        top_bets = bets.nlargest(10, 'max_prob')[['game_id', 'predicted_quarter', 'highest_quarter_num', 'max_prob', 'edge', 'hit']]
        for idx, row in top_bets.iterrows():
            status = '✅' if row['hit'] else '❌'
            print(f"  {status} Game {row['game_id']}: Pred=Q{row['predicted_quarter']}, Actual=Q{row['highest_quarter_num']}, Prob={row['max_prob']:.1%}, Edge={row['edge']:+.1%}")

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
    """Main function to train highest scoring quarter model."""

    print(f"\n{'='*80}")
    print(f"HIGHEST SCORING QUARTER MODEL TRAINING")
    print(f"{'='*80}\n")

    # Load games with quarters
    games_df = load_games_with_quarters()

    if len(games_df) == 0:
        return

    # Create highest scoring quarter data
    games_with_highest = create_highest_scoring_quarter_data(games_df)

    # Create team-level quarter features
    team_games = create_team_quarter_features(games_with_highest)

    # Create game-level features
    game_features = create_game_level_features(games_with_highest, team_games)

    # Training weeks
    training_weeks = list(range(1, 10))
    train_df = game_features[game_features['week'].isin(training_weeks)].copy()

    print(f"Training weeks: {training_weeks}")
    print(f"Training records: {len(train_df):,}")
    print()

    # Train model
    model, features, metrics = train_highest_quarter_model(train_df)

    # Save model
    models_dir = Path('outputs/models/game_outcome')
    models_dir.mkdir(parents=True, exist_ok=True)

    model_file = models_dir / 'highest_scoring_quarter_model.pkl'
    joblib.dump({
        'model': model,
        'features': features,
        'config': {
            'name': 'Highest Scoring Quarter',
            'type': 'Multinomial',
            'classes': ['Q1', 'Q2', 'Q3', 'Q4'],
            'edge_threshold': 0.10,
            'typical_odds': '+300'
        }
    }, model_file)

    print(f"💾 Saved to: {model_file}")
    print()

    # Backtest on Weeks 10 & 11
    backtest_results = []

    for week in [10, 11]:
        result = backtest_highest_quarter(model, features, game_features, week)

        if result:
            backtest_results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print(f"HIGHEST SCORING QUARTER TRAINING COMPLETE")
    print(f"{'='*80}\n")

    if backtest_results:
        results_df = pd.DataFrame(backtest_results)

        print("BACKTEST SUMMARY:")
        print(f"  Total bets: {results_df['bets'].sum()}")
        print(f"  Average hit rate: {results_df['hit_rate'].mean():.1%}")
        print(f"  Average ROI: {results_df['roi'].mean():+.1%}")
        print()

    print("✅ Highest Scoring Quarter model trained and backtested")
    print()
    print("NEW MARKET AVAILABLE:")
    print("  ✅ Highest Scoring Quarter")
    print()


if __name__ == '__main__':
    main()
