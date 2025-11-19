"""Train Will Game Go to OT model.

Will Game Go to OT is a DraftKings prop (typically +1000 for Yes, heavy favorite for No).
We use binary classification to predict if the game will go to overtime.

Expected Performance: 5-10% hit rate (vs 6-8% actual OT rate), needs high precision

Features:
- Historical close game rate (both teams)
- Average margin of victory
- Defensive strength (low-scoring games more likely to tie)
- Pace of play
- Clutch performance metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score
import joblib


def load_pbp_data():
    """Load play-by-play data."""

    print(f"\n{'='*80}")
    print(f"LOADING PLAY-BY-PLAY DATA")
    print(f"{'='*80}\n")

    pbp_file = Path('/home/user/nfl_backend/inputs/play_by_play_2025.parquet')

    if not pbp_file.exists():
        print(f"❌ PBP file not found: {pbp_file}")
        return pd.DataFrame()

    pbp = pd.read_parquet(pbp_file)

    print(f"✅ Loaded {len(pbp):,} plays")
    print(f"   Games: {pbp['game_id'].nunique()}")
    print(f"   Weeks: {pbp['week'].min()}-{pbp['week'].max()}")
    print()

    return pbp


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


def identify_ot_games(pbp_df):
    """Identify games that went to overtime.

    Args:
        pbp_df: Play-by-play DataFrame

    Returns:
        DataFrame with game_id and ot flag
    """

    print(f"\n{'='*80}")
    print(f"IDENTIFYING OVERTIME GAMES")
    print(f"{'='*80}\n")

    # Check for plays in quarter 5 (overtime)
    ot_games = pbp_df[pbp_df['qtr'] == 5]['game_id'].unique()

    ot_df = pd.DataFrame({
        'game_id': pbp_df['game_id'].unique()
    })
    ot_df['went_to_ot'] = ot_df['game_id'].isin(ot_games).astype(int)

    ot_count = ot_df['went_to_ot'].sum()
    ot_rate = ot_count / len(ot_df) * 100

    print(f"Total games: {len(ot_df)}")
    print(f"OT games: {ot_count} ({ot_rate:.1f}%)")
    print()

    if ot_count > 0:
        print("OT games:")
        for game_id in ot_games:
            print(f"  {game_id}")
        print()

    return ot_df


def create_team_ot_features(games_df):
    """Create team-level features related to OT probability.

    Args:
        games_df: Games DataFrame with OT flag

    Returns:
        DataFrame with team-game records and OT features
    """

    print(f"\n{'='*80}")
    print(f"CREATING TEAM OT FEATURES")
    print(f"{'='*80}\n")

    # Create team-game records (2 per game: home and away)
    home_games = games_df[['game_id', 'week', 'home_team', 'home_total', 'away_total', 'went_to_ot']].copy()
    home_games.columns = ['game_id', 'week', 'team', 'points_for', 'points_against', 'went_to_ot']
    home_games['is_home'] = 1
    home_games['margin'] = abs(home_games['points_for'] - home_games['points_against'])
    home_games['close_game'] = (home_games['margin'] <= 3).astype(int)

    away_games = games_df[['game_id', 'week', 'away_team', 'away_total', 'home_total', 'went_to_ot']].copy()
    away_games.columns = ['game_id', 'week', 'team', 'points_for', 'points_against', 'went_to_ot']
    away_games['is_home'] = 0
    away_games['margin'] = abs(away_games['points_for'] - away_games['points_against'])
    away_games['close_game'] = (away_games['margin'] <= 3).astype(int)

    team_games = pd.concat([home_games, away_games], ignore_index=True)

    print(f"Team-game records: {len(team_games)}")
    print()

    # Sort by team and week
    team_games = team_games.sort_values(['team', 'week'])

    # Calculate historical stats per team (expanding window)
    team_games['season_close_game_rate'] = team_games.groupby('team')['close_game'].expanding().mean().reset_index(level=0, drop=True)
    team_games['season_avg_margin'] = team_games.groupby('team')['margin'].expanding().mean().reset_index(level=0, drop=True)
    team_games['season_avg_pf'] = team_games.groupby('team')['points_for'].expanding().mean().reset_index(level=0, drop=True)
    team_games['season_avg_pa'] = team_games.groupby('team')['points_against'].expanding().mean().reset_index(level=0, drop=True)
    team_games['season_ot_rate'] = team_games.groupby('team')['went_to_ot'].expanding().mean().reset_index(level=0, drop=True)

    # L3 stats
    team_games['l3_close_game_rate'] = team_games.groupby('team')['close_game'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    team_games['l3_avg_margin'] = team_games.groupby('team')['margin'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    team_games['l3_avg_pf'] = team_games.groupby('team')['points_for'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    team_games['l3_avg_pa'] = team_games.groupby('team')['points_against'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

    # Games played
    team_games['games_played'] = team_games.groupby('team').cumcount() + 1

    print(f"✅ Created team OT features")
    print()

    return team_games


def create_game_level_features(games_df, team_games_df):
    """Create game-level features from team-game data.

    Args:
        games_df: Games DataFrame with OT flag
        team_games_df: Team-game DataFrame with OT features

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
    game_features = games_df[['game_id', 'week', 'went_to_ot']].copy()

    game_features = game_features.merge(
        home_features[['game_id', 'home_games_played', 'home_season_close_game_rate', 'home_season_avg_margin',
                       'home_season_avg_pf', 'home_season_avg_pa', 'home_season_ot_rate',
                       'home_l3_close_game_rate', 'home_l3_avg_margin', 'home_l3_avg_pf', 'home_l3_avg_pa']],
        on='game_id', how='left'
    )

    game_features = game_features.merge(
        away_features[['game_id', 'away_games_played', 'away_season_close_game_rate', 'away_season_avg_margin',
                       'away_season_avg_pf', 'away_season_avg_pa', 'away_season_ot_rate',
                       'away_l3_close_game_rate', 'away_l3_avg_margin', 'away_l3_avg_pf', 'away_l3_avg_pa']],
        on='game_id', how='left'
    )

    # Combined features
    game_features['avg_close_game_rate'] = (game_features['home_season_close_game_rate'] + game_features['away_season_close_game_rate']) / 2
    game_features['avg_margin'] = (game_features['home_season_avg_margin'] + game_features['away_season_avg_margin']) / 2
    game_features['avg_ot_rate'] = (game_features['home_season_ot_rate'] + game_features['away_season_ot_rate']) / 2
    game_features['expected_total'] = game_features['home_season_avg_pf'] + game_features['away_season_avg_pf']

    game_features['l3_avg_close_game_rate'] = (game_features['home_l3_close_game_rate'] + game_features['away_l3_close_game_rate']) / 2
    game_features['l3_avg_margin'] = (game_features['home_l3_avg_margin'] + game_features['away_l3_avg_margin']) / 2
    game_features['l3_expected_total'] = game_features['home_l3_avg_pf'] + game_features['away_l3_avg_pf']

    # Filter to games where both teams have 3+ games
    game_features = game_features[
        (game_features['home_games_played'] >= 3) &
        (game_features['away_games_played'] >= 3)
    ].copy()

    print(f"Games with 3+ games played (both teams): {len(game_features)}")
    print(f"OT rate: {game_features['went_to_ot'].mean():.1%}")
    print()

    return game_features


def train_ot_model(train_df):
    """Train OT prediction model.

    Args:
        train_df: Training DataFrame with features

    Returns:
        Model, features, metrics
    """

    print(f"\n{'='*80}")
    print(f"TRAINING WILL GO TO OT MODEL")
    print(f"{'='*80}\n")

    # Features
    feature_cols = [
        'avg_close_game_rate',
        'avg_margin',
        'avg_ot_rate',
        'expected_total',
        'l3_avg_close_game_rate',
        'l3_avg_margin',
        'l3_expected_total',
        'home_games_played',
        'away_games_played',
    ]

    X = train_df[feature_cols].fillna(0)
    y = train_df['went_to_ot']

    print(f"Training set: {len(X):,} games")
    print(f"OT rate: {y.mean():.1%}")
    print(f"Positive cases: {y.sum()}")
    print()

    # Train gradient boosting classifier
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=42
    )

    model.fit(X, y)

    # Training metrics
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    logloss = log_loss(y, y_prob)

    # Calculate precision/recall (important for rare events)
    if y_pred.sum() > 0:
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        print(f"Training Accuracy: {acc:.1%}")
        print(f"Precision: {precision:.1%}")
        print(f"Recall: {recall:.1%}")
        print(f"Log Loss: {logloss:.4f}")
    else:
        print(f"Training Accuracy: {acc:.1%}")
        print(f"Log Loss: {logloss:.4f}")
        print("No positive predictions")
    print()

    # Feature importance
    print("Feature Importance:")
    feature_imp = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: x[1], reverse=True)
    for feat, imp in feature_imp:
        print(f"  {feat:30s}: {imp:.3f}")
    print()

    return model, feature_cols, {'accuracy': acc, 'log_loss': logloss}


def backtest_ot(model, features, game_features_df, week):
    """Backtest OT prediction model on a specific week.

    Args:
        model: Trained model
        features: Feature column names
        game_features_df: Full game features DataFrame
        week: Week to backtest

    Returns:
        Backtest results
    """

    print(f"\n{'='*80}")
    print(f"BACKTESTING WILL GO TO OT - WEEK {week}")
    print(f"{'='*80}\n")

    # Get week data
    week_df = game_features_df[game_features_df['week'] == week].copy()

    print(f"Week {week} games: {len(week_df)}")

    if len(week_df) == 0:
        print("❌ No games this week")
        return None

    # Predict
    X = week_df[features].fillna(0)
    prob_ot = model.predict_proba(X)[:, 1]
    week_df['prob_ot'] = prob_ot

    # Calculate edge (typical odds +1000 = 9.1% implied probability)
    week_df['edge'] = week_df['prob_ot'] - 0.091

    # Bets with >5% edge (need strong confidence for rare events)
    bets = week_df[week_df['edge'] > 0.05].copy()

    print(f"Bets with >5% edge: {len(bets)}")

    if len(bets) > 0:
        hit_rate = bets['went_to_ot'].mean()

        # Calculate ROI assuming +1000 odds
        # Win: +1000, Loss: -100
        wins = bets['went_to_ot'].sum()
        losses = len(bets) - wins
        roi = ((wins * 1000) - (losses * 100)) / (len(bets) * 100)

        print(f"Hit rate: {hit_rate:.1%}")
        print(f"ROI (assuming +1000 odds): {roi:+.1%}")

        # Show all bets
        print(f"\nAll {len(bets)} bets:")
        for idx, row in bets[['game_id', 'prob_ot', 'edge', 'went_to_ot']].iterrows():
            status = '✅' if row['went_to_ot'] else '❌'
            print(f"  {status} Game {row['game_id']}: Prob={row['prob_ot']:.1%}, Edge={row['edge']:+.1%}")

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
    """Main function to train OT prediction model."""

    print(f"\n{'='*80}")
    print(f"WILL GAME GO TO OT MODEL TRAINING")
    print(f"{'='*80}\n")

    # Load PBP data
    pbp_df = load_pbp_data()

    if len(pbp_df) == 0:
        return

    # Load games with quarters
    games_df = load_games_with_quarters()

    if len(games_df) == 0:
        return

    # Identify OT games
    ot_df = identify_ot_games(pbp_df)

    # Merge OT flag with games
    games_with_ot = games_df.merge(ot_df, on='game_id', how='left')

    # Create team-level OT features
    team_games = create_team_ot_features(games_with_ot)

    # Create game-level features
    game_features = create_game_level_features(games_with_ot, team_games)

    # Training weeks
    training_weeks = list(range(1, 10))
    train_df = game_features[game_features['week'].isin(training_weeks)].copy()

    print(f"Training weeks: {training_weeks}")
    print(f"Training records: {len(train_df):,}")
    print()

    # Train model
    model, features, metrics = train_ot_model(train_df)

    # Save model
    models_dir = Path('outputs/models/game_outcome')
    models_dir.mkdir(parents=True, exist_ok=True)

    model_file = models_dir / 'will_go_to_ot_model.pkl'
    joblib.dump({
        'model': model,
        'features': features,
        'config': {
            'name': 'Will Game Go to OT',
            'type': 'Bernoulli',
            'edge_threshold': 0.05,
            'typical_odds': '+1000'
        }
    }, model_file)

    print(f"💾 Saved to: {model_file}")
    print()

    # Backtest on Weeks 10 & 11
    backtest_results = []

    for week in [10, 11]:
        result = backtest_ot(model, features, game_features, week)

        if result:
            backtest_results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print(f"WILL GO TO OT TRAINING COMPLETE")
    print(f"{'='*80}\n")

    if backtest_results:
        results_df = pd.DataFrame(backtest_results)

        print("BACKTEST SUMMARY:")
        print(f"  Total bets: {results_df['bets'].sum()}")
        if results_df['bets'].sum() > 0:
            print(f"  Average hit rate: {results_df['hit_rate'].mean():.1%}")
            print(f"  Average ROI: {results_df['roi'].mean():+.1%}")
        else:
            print("  No bets placed")
        print()

    print("✅ Will Go to OT model trained and backtested")
    print()
    print("NEW MARKET AVAILABLE:")
    print("  ✅ Will Game Go to OT")
    print()


if __name__ == '__main__':
    main()
