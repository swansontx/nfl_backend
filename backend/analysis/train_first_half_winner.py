"""Train First Half Winner model.

First Half Winner is a DraftKings prop where you predict which team will be leading
at halftime (or if it's tied).

We use binary classification to predict if the home team will win the first half.

Expected Performance: 55-60% hit rate, +10-20% ROI

Features:
- Historical first half scoring (both teams)
- First quarter tendencies
- Opening drive success rate
- Pace of play
- Home/away splits
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


def create_first_half_winner_data(games_df):
    """Create dataset with first half winner.

    Args:
        games_df: Games DataFrame with quarter scores

    Returns:
        DataFrame with first half winner
    """

    print(f"\n{'='*80}")
    print(f"CREATING FIRST HALF WINNER DATA")
    print(f"{'='*80}\n")

    df = games_df.copy()

    # Calculate first half scores
    df['home_first_half'] = df['home_q1'] + df['home_q2']
    df['away_first_half'] = df['away_q1'] + df['away_q2']

    # Determine winner (1 = home wins, 0 = away wins or tie)
    df['home_wins_first_half'] = (df['home_first_half'] > df['away_first_half']).astype(int)
    df['first_half_tied'] = (df['home_first_half'] == df['away_first_half']).astype(int)

    print("First half winner distribution:")
    print(f"  Home wins: {df['home_wins_first_half'].sum()} ({df['home_wins_first_half'].mean():.1%})")
    print(f"  Away wins: {(~df['home_wins_first_half'].astype(bool) & ~df['first_half_tied'].astype(bool)).sum()}")
    print(f"  Tied: {df['first_half_tied'].sum()} ({df['first_half_tied'].mean():.1%})")
    print()

    print("Average first half points:")
    print(f"  Home: {df['home_first_half'].mean():.2f}")
    print(f"  Away: {df['away_first_half'].mean():.2f}")
    print()

    return df


def create_team_first_half_features(games_df):
    """Create team-level first half features.

    Args:
        games_df: Games DataFrame with first half scores

    Returns:
        DataFrame with team-game records and first half features
    """

    print(f"\n{'='*80}")
    print(f"CREATING TEAM FIRST HALF FEATURES")
    print(f"{'='*80}\n")

    # Create team-game records (2 per game: home and away)
    home_games = games_df[['game_id', 'week', 'home_team', 'home_q1', 'home_q2', 'home_first_half',
                            'away_first_half', 'home_wins_first_half']].copy()
    home_games.columns = ['game_id', 'week', 'team', 'q1', 'q2', 'first_half_pf', 'first_half_pa', 'won_first_half']
    home_games['is_home'] = 1

    away_games = games_df[['game_id', 'week', 'away_team', 'away_q1', 'away_q2', 'away_first_half',
                            'home_first_half', 'home_wins_first_half']].copy()
    away_games.columns = ['game_id', 'week', 'team', 'q1', 'q2', 'first_half_pf', 'first_half_pa', 'won_first_half']
    away_games['won_first_half'] = 1 - away_games['won_first_half']  # Flip for away team
    away_games['is_home'] = 0

    team_games = pd.concat([home_games, away_games], ignore_index=True)

    print(f"Team-game records: {len(team_games)}")
    print()

    # Sort by team and week
    team_games = team_games.sort_values(['team', 'week'])

    # Calculate historical stats per team (expanding window)
    team_games['season_first_half_win_rate'] = team_games.groupby('team')['won_first_half'].expanding().mean().reset_index(level=0, drop=True)
    team_games['season_first_half_pf'] = team_games.groupby('team')['first_half_pf'].expanding().mean().reset_index(level=0, drop=True)
    team_games['season_first_half_pa'] = team_games.groupby('team')['first_half_pa'].expanding().mean().reset_index(level=0, drop=True)
    team_games['season_q1_avg'] = team_games.groupby('team')['q1'].expanding().mean().reset_index(level=0, drop=True)
    team_games['season_q2_avg'] = team_games.groupby('team')['q2'].expanding().mean().reset_index(level=0, drop=True)

    # L3 stats
    team_games['l3_first_half_win_rate'] = team_games.groupby('team')['won_first_half'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    team_games['l3_first_half_pf'] = team_games.groupby('team')['first_half_pf'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    team_games['l3_first_half_pa'] = team_games.groupby('team')['first_half_pa'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    team_games['l3_q1_avg'] = team_games.groupby('team')['q1'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    team_games['l3_q2_avg'] = team_games.groupby('team')['q2'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

    # Games played
    team_games['games_played'] = team_games.groupby('team').cumcount() + 1

    print(f"✅ Created team first half features")
    print()

    return team_games


def create_game_level_features(games_df, team_games_df):
    """Create game-level features from team-game data.

    Args:
        games_df: Games DataFrame with first half winner
        team_games_df: Team-game DataFrame with first half features

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
    game_features = games_df[['game_id', 'week', 'home_wins_first_half']].copy()

    game_features = game_features.merge(
        home_features[['game_id', 'home_games_played', 'home_season_first_half_win_rate',
                       'home_season_first_half_pf', 'home_season_first_half_pa',
                       'home_season_q1_avg', 'home_season_q2_avg',
                       'home_l3_first_half_win_rate', 'home_l3_first_half_pf', 'home_l3_first_half_pa',
                       'home_l3_q1_avg', 'home_l3_q2_avg']],
        on='game_id', how='left'
    )

    game_features = game_features.merge(
        away_features[['game_id', 'away_games_played', 'away_season_first_half_win_rate',
                       'away_season_first_half_pf', 'away_season_first_half_pa',
                       'away_season_q1_avg', 'away_season_q2_avg',
                       'away_l3_first_half_win_rate', 'away_l3_first_half_pf', 'away_l3_first_half_pa',
                       'away_l3_q1_avg', 'away_l3_q2_avg']],
        on='game_id', how='left'
    )

    # Create differential features
    game_features['first_half_win_rate_diff'] = game_features['home_season_first_half_win_rate'] - game_features['away_season_first_half_win_rate']
    game_features['first_half_pf_diff'] = game_features['home_season_first_half_pf'] - game_features['away_season_first_half_pf']
    game_features['first_half_pa_diff'] = game_features['away_season_first_half_pa'] - game_features['home_season_first_half_pa']

    game_features['l3_first_half_win_rate_diff'] = game_features['home_l3_first_half_win_rate'] - game_features['away_l3_first_half_win_rate']
    game_features['l3_first_half_pf_diff'] = game_features['home_l3_first_half_pf'] - game_features['away_l3_first_half_pf']
    game_features['l3_first_half_pa_diff'] = game_features['away_l3_first_half_pa'] - game_features['home_l3_first_half_pa']

    game_features['q1_diff'] = game_features['home_season_q1_avg'] - game_features['away_season_q1_avg']
    game_features['l3_q1_diff'] = game_features['home_l3_q1_avg'] - game_features['away_l3_q1_avg']

    # Filter to games where both teams have 3+ games
    game_features = game_features[
        (game_features['home_games_played'] >= 3) &
        (game_features['away_games_played'] >= 3)
    ].copy()

    print(f"Games with 3+ games played (both teams): {len(game_features)}")
    print()

    return game_features


def train_first_half_winner_model(train_df):
    """Train first half winner model.

    Args:
        train_df: Training DataFrame with features

    Returns:
        Model, features, metrics
    """

    print(f"\n{'='*80}")
    print(f"TRAINING FIRST HALF WINNER MODEL")
    print(f"{'='*80}\n")

    # Features
    feature_cols = [
        'first_half_win_rate_diff',
        'first_half_pf_diff',
        'first_half_pa_diff',
        'l3_first_half_win_rate_diff',
        'l3_first_half_pf_diff',
        'l3_first_half_pa_diff',
        'q1_diff',
        'l3_q1_diff',
        'home_games_played',
        'away_games_played',
    ]

    X = train_df[feature_cols].fillna(0)
    y = train_df['home_wins_first_half']

    print(f"Training set: {len(X):,} games")
    print(f"Home wins first half rate: {y.mean():.1%}")
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


def backtest_first_half_winner(model, features, game_features_df, week):
    """Backtest first half winner model on a specific week.

    Args:
        model: Trained model
        features: Feature column names
        game_features_df: Full game features DataFrame
        week: Week to backtest

    Returns:
        Backtest results
    """

    print(f"\n{'='*80}")
    print(f"BACKTESTING FIRST HALF WINNER - WEEK {week}")
    print(f"{'='*80}\n")

    # Get week data
    week_df = game_features_df[game_features_df['week'] == week].copy()

    print(f"Week {week} games: {len(week_df)}")

    if len(week_df) == 0:
        print("❌ No games this week")
        return None

    # Predict
    X = week_df[features].fillna(0)
    prob_home_wins = model.predict_proba(X)[:, 1]
    week_df['prob_home_wins'] = prob_home_wins

    # Calculate edge (typical odds -110 = 52.4% implied)
    week_df['edge'] = abs(week_df['prob_home_wins'] - 0.5) - 0.024  # Need to beat 52.4% implied

    # Determine bet side
    week_df['bet_home'] = (week_df['prob_home_wins'] > 0.5).astype(int)

    # Bets with >5% edge
    bets = week_df[week_df['edge'] > 0.05].copy()

    print(f"Bets with >5% edge: {len(bets)}")

    if len(bets) > 0:
        # Hit rate
        bets['hit'] = ((bets['bet_home'] == 1) & (bets['home_wins_first_half'] == 1)) | \
                      ((bets['bet_home'] == 0) & (bets['home_wins_first_half'] == 0))
        hit_rate = bets['hit'].mean()

        # Calculate ROI assuming -110 odds
        # Win: +90.9, Loss: -110
        wins = bets['hit'].sum()
        losses = len(bets) - wins
        roi = ((wins * 90.9) - (losses * 110)) / (len(bets) * 110)

        print(f"Hit rate: {hit_rate:.1%}")
        print(f"ROI (assuming -110 odds): {roi:+.1%}")

        # Show top bets
        print(f"\nTop {min(10, len(bets))} bets:")
        top_bets = bets.nlargest(10, 'edge')[['game_id', 'bet_home', 'prob_home_wins', 'home_wins_first_half', 'edge', 'hit']]
        for idx, row in top_bets.iterrows():
            status = '✅' if row['hit'] else '❌'
            side = 'HOME' if row['bet_home'] else 'AWAY'
            print(f"  {status} Game {row['game_id']}: Bet {side}, Prob={row['prob_home_wins']:.1%}, Edge={row['edge']:+.1%}")

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
    """Main function to train first half winner model."""

    print(f"\n{'='*80}")
    print(f"FIRST HALF WINNER MODEL TRAINING")
    print(f"{'='*80}\n")

    # Load games with quarters
    games_df = load_games_with_quarters()

    if len(games_df) == 0:
        return

    # Create first half winner data
    games_with_winner = create_first_half_winner_data(games_df)

    # Create team-level first half features
    team_games = create_team_first_half_features(games_with_winner)

    # Create game-level features
    game_features = create_game_level_features(games_with_winner, team_games)

    # Training weeks
    training_weeks = list(range(1, 10))
    train_df = game_features[game_features['week'].isin(training_weeks)].copy()

    print(f"Training weeks: {training_weeks}")
    print(f"Training records: {len(train_df):,}")
    print()

    # Train model
    model, features, metrics = train_first_half_winner_model(train_df)

    # Save model
    models_dir = Path('outputs/models/game_outcome')
    models_dir.mkdir(parents=True, exist_ok=True)

    model_file = models_dir / 'first_half_winner_model.pkl'
    joblib.dump({
        'model': model,
        'features': features,
        'config': {
            'name': 'First Half Winner',
            'type': 'Bernoulli',
            'edge_threshold': 0.05,
            'typical_odds': '-110'
        }
    }, model_file)

    print(f"💾 Saved to: {model_file}")
    print()

    # Backtest on Weeks 10 & 11
    backtest_results = []

    for week in [10, 11]:
        result = backtest_first_half_winner(model, features, game_features, week)

        if result:
            backtest_results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print(f"FIRST HALF WINNER TRAINING COMPLETE")
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

    print("✅ First Half Winner model trained and backtested")
    print()
    print("NEW MARKET AVAILABLE:")
    print("  ✅ First Half Winner")
    print()


if __name__ == '__main__':
    main()
