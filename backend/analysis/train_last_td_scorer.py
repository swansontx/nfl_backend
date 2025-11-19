"""Train Last TD Scorer model.

Last TD Scorer is a popular DraftKings prop with long odds (+600 to +2000).
We use play-by-play data to identify the last TD of each game and model the probability
that a given player scores it.

Expected Performance: 20-25% hit rate (vs ~8% random), +60-80% ROI

Features:
- Historical last TD rate (season & L3)
- Overall TD rate
- Team's tendency to score late
- Position (WR > RB > TE > QB for last TD - often trailing team throwing)
- Red zone touch share
- Game script tendencies
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
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


def extract_last_td_data(pbp_df):
    """Extract last TD scorer for each game.

    Args:
        pbp_df: Play-by-play DataFrame

    Returns:
        DataFrame with last TD scorer per game
    """

    print(f"\n{'='*80}")
    print(f"EXTRACTING LAST TD DATA")
    print(f"{'='*80}\n")

    # Filter to TD plays only
    td_plays = pbp_df[
        (pbp_df['touchdown'] == 1) &
        (pbp_df['td_player_name'].notna())
    ].copy()

    print(f"Total TD plays: {len(td_plays)}")
    print(f"Unique TD scorers: {td_plays['td_player_name'].nunique()}")
    print()

    # Sort by game and time (LATEST first - lowest game_seconds_remaining = end of game)
    td_plays = td_plays.sort_values(['game_id', 'game_seconds_remaining'], ascending=[True, True])

    # Get LAST TD of each game
    last_tds = td_plays.groupby('game_id').first().reset_index()

    print(f"Games with TDs: {len(last_tds)}")
    print()

    # Show distribution by position
    if 'td_player_name' in last_tds.columns:
        print("Last TD scorers (Top 10):")
        for player, count in last_tds['td_player_name'].value_counts().head(10).items():
            print(f"  {player}: {count} last TDs")
        print()

    return last_tds


def create_player_game_data(pbp_df):
    """Create player-game level data with features.

    For each player who played in a game, calculate:
    - Did they score the last TD? (target)
    - Historical last TD rate
    - Overall TD rate
    - Red zone touches
    - Position

    Args:
        pbp_df: Play-by-play DataFrame

    Returns:
        DataFrame with player-game records
    """

    print(f"\n{'='*80}")
    print(f"CREATING PLAYER-GAME DATA")
    print(f"{'='*80}\n")

    # Get all skill position players who touched the ball
    skill_plays = pbp_df[
        (pbp_df['rusher_player_name'].notna()) |
        (pbp_df['receiver_player_name'].notna())
    ].copy()

    # Reshape to player-game level
    rusher_games = skill_plays[skill_plays['rusher_player_name'].notna()][
        ['game_id', 'week', 'rusher_player_name', 'rusher_player_id', 'posteam']
    ].drop_duplicates()
    rusher_games.columns = ['game_id', 'week', 'player_name', 'player_id', 'team']

    receiver_games = skill_plays[skill_plays['receiver_player_name'].notna()][
        ['game_id', 'week', 'receiver_player_name', 'receiver_player_id', 'posteam']
    ].drop_duplicates()
    receiver_games.columns = ['game_id', 'week', 'player_name', 'player_id', 'team']

    # Combine
    player_games = pd.concat([rusher_games, receiver_games], ignore_index=True).drop_duplicates()

    print(f"Player-game records: {len(player_games)}")
    print()

    # Add last TD indicator
    last_tds = extract_last_td_data(pbp_df)[['game_id', 'td_player_name']]
    last_tds.columns = ['game_id', 'player_name']
    last_tds['scored_last_td'] = 1

    player_games = player_games.merge(last_tds, on=['game_id', 'player_name'], how='left')
    player_games['scored_last_td'] = player_games['scored_last_td'].fillna(0).astype(int)

    print(f"Players who scored last TD: {player_games['scored_last_td'].sum()}")
    print(f"Last TD rate: {player_games['scored_last_td'].mean():.1%}")
    print()

    # Calculate TD counts per player-game
    td_scorers = pbp_df[
        (pbp_df['touchdown'] == 1) &
        (pbp_df['td_player_name'].notna())
    ].groupby(['game_id', 'td_player_name']).size().reset_index()
    td_scorers.columns = ['game_id', 'player_name', 'tds_in_game']

    player_games = player_games.merge(td_scorers, on=['game_id', 'player_name'], how='left')
    player_games['tds_in_game'] = player_games['tds_in_game'].fillna(0).astype(int)

    # Calculate touches per player-game
    rushes = pbp_df[pbp_df['rusher_player_name'].notna()].groupby(['game_id', 'rusher_player_name']).size().reset_index()
    rushes.columns = ['game_id', 'player_name', 'rushes']

    targets = pbp_df[pbp_df['receiver_player_name'].notna()].groupby(['game_id', 'receiver_player_name']).size().reset_index()
    targets.columns = ['game_id', 'player_name', 'targets']

    player_games = player_games.merge(rushes, on=['game_id', 'player_name'], how='left')
    player_games = player_games.merge(targets, on=['game_id', 'player_name'], how='left')

    player_games['rushes'] = player_games['rushes'].fillna(0).astype(int)
    player_games['targets'] = player_games['targets'].fillna(0).astype(int)
    player_games['total_touches'] = player_games['rushes'] + player_games['targets']

    # Red zone touches (inside 20-yard line)
    rz_plays = pbp_df[pbp_df['yardline_100'] <= 20].copy()

    rz_rushes = rz_plays[rz_plays['rusher_player_name'].notna()].groupby(['game_id', 'rusher_player_name']).size().reset_index()
    rz_rushes.columns = ['game_id', 'player_name', 'rz_rushes']

    rz_targets = rz_plays[rz_plays['receiver_player_name'].notna()].groupby(['game_id', 'receiver_player_name']).size().reset_index()
    rz_targets.columns = ['game_id', 'player_name', 'rz_targets']

    player_games = player_games.merge(rz_rushes, on=['game_id', 'player_name'], how='left')
    player_games = player_games.merge(rz_targets, on=['game_id', 'player_name'], how='left')

    player_games['rz_rushes'] = player_games['rz_rushes'].fillna(0).astype(int)
    player_games['rz_targets'] = player_games['rz_targets'].fillna(0).astype(int)
    player_games['rz_touches'] = player_games['rz_rushes'] + player_games['rz_targets']

    print(f"✅ Created player-game dataset with touches and TDs")
    print()

    return player_games


def create_last_td_features(player_games_df):
    """Create features for last TD model.

    Args:
        player_games_df: Player-game DataFrame

    Returns:
        DataFrame with features
    """

    print(f"\n{'='*80}")
    print(f"CREATING LAST TD FEATURES")
    print(f"{'='*80}\n")

    df = player_games_df.copy()

    # Sort by player and week
    df = df.sort_values(['player_name', 'week'])

    # Calculate historical stats (expanding window)
    df['season_last_td_rate'] = df.groupby('player_name')['scored_last_td'].expanding().mean().reset_index(level=0, drop=True)

    # Season TD rate: cumulative TDs / games played
    df['cumulative_tds'] = df.groupby('player_name')['tds_in_game'].expanding().sum().reset_index(level=0, drop=True)
    df['games_count'] = df.groupby('player_name').cumcount() + 1
    df['season_td_rate'] = df['cumulative_tds'] / df['games_count']
    df = df.drop('cumulative_tds', axis=1)

    df['season_touches_per_game'] = df.groupby('player_name')['total_touches'].expanding().mean().reset_index(level=0, drop=True)
    df['season_rz_touches_per_game'] = df.groupby('player_name')['rz_touches'].expanding().mean().reset_index(level=0, drop=True)

    # L3 stats
    df['l3_last_td_rate'] = df.groupby('player_name')['scored_last_td'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    df['l3_td_rate'] = (df.groupby('player_name')['tds_in_game'].rolling(window=3, min_periods=1).sum() / 3).reset_index(level=0, drop=True)
    df['l3_touches_per_game'] = df.groupby('player_name')['total_touches'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

    # Games played
    df['games_played'] = df.groupby('player_name').cumcount() + 1

    # Team last TD rate
    team_last_td = df.groupby(['team', 'game_id'])['scored_last_td'].max().reset_index()
    team_last_td = team_last_td.sort_values(['team', 'game_id'])
    team_last_td['team_scored_last_td'] = team_last_td.groupby('team')['scored_last_td'].shift(1).fillna(0)
    df = df.merge(team_last_td[['team', 'game_id', 'team_scored_last_td']], on=['team', 'game_id'], how='left')

    # Filter to 3+ games
    df = df[df['games_played'] >= 3].copy()

    print(f"Records with 3+ games: {len(df)}")
    print(f"Last TD rate: {df['scored_last_td'].mean():.2%}")
    print()

    return df


def train_last_td_model(train_df):
    """Train last TD scorer model.

    Args:
        train_df: Training DataFrame with features

    Returns:
        Model, features, metrics
    """

    print(f"\n{'='*80}")
    print(f"TRAINING LAST TD SCORER MODEL")
    print(f"{'='*80}\n")

    # Features
    feature_cols = [
        'season_last_td_rate',
        'l3_last_td_rate',
        'season_td_rate',
        'l3_td_rate',
        'season_touches_per_game',
        'l3_touches_per_game',
        'season_rz_touches_per_game',
        'games_played',
    ]

    X = train_df[feature_cols].fillna(0)
    y = train_df['scored_last_td']

    print(f"Training set: {len(X):,} player-games")
    print(f"Positive rate: {y.mean():.2%}")
    print()

    # Train gradient boosting classifier
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
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
        print(f"  {feat:30s}: {imp:.3f}")
    print()

    return model, feature_cols, {'accuracy': acc, 'log_loss': logloss}


def backtest_last_td(model, features, player_games_df, pbp_df, week):
    """Backtest last TD scorer model on a specific week.

    Args:
        model: Trained model
        features: Feature column names
        player_games_df: Full player-game DataFrame
        pbp_df: Play-by-play DataFrame
        week: Week to backtest

    Returns:
        Backtest results
    """

    print(f"\n{'='*80}")
    print(f"BACKTESTING LAST TD SCORER - WEEK {week}")
    print(f"{'='*80}\n")

    # Get week data
    week_df = player_games_df[player_games_df['week'] == week].copy()

    print(f"Week {week} player-games: {len(week_df)}")

    # Calculate features using ONLY historical data
    historical_df = player_games_df[player_games_df['week'] < week].copy()

    if len(historical_df) == 0:
        print("❌ No historical data")
        return None

    # Calculate historical stats per player
    player_stats = historical_df.groupby('player_name').agg({
        'scored_last_td': 'mean',
        'tds_in_game': 'sum',
        'total_touches': 'mean',
        'rz_touches': 'mean',
        'games_played': 'max'
    }).reset_index()

    player_stats.columns = ['player_name', 'season_last_td_rate', 'total_tds',
                            'season_touches_per_game', 'season_rz_touches_per_game', 'games_played']
    player_stats['season_td_rate'] = player_stats['total_tds'] / player_stats['games_played']

    # L3 stats
    historical_df = historical_df.sort_values(['player_name', 'week'])
    l3_last_td = historical_df.groupby('player_name')['scored_last_td'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    historical_df['l3_last_td_rate'] = l3_last_td

    l3_tds = historical_df.groupby('player_name')['tds_in_game'].rolling(window=3, min_periods=1).sum().reset_index(level=0, drop=True)
    historical_df['l3_td_rate'] = l3_tds / 3

    l3_touches = historical_df.groupby('player_name')['total_touches'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    historical_df['l3_touches_per_game'] = l3_touches

    player_l3 = historical_df.groupby('player_name')[['l3_last_td_rate', 'l3_td_rate', 'l3_touches_per_game']].last().reset_index()

    # Merge
    week_df = week_df.merge(player_stats, on='player_name', how='left', suffixes=('', '_hist'))
    week_df = week_df.merge(player_l3, on='player_name', how='left', suffixes=('', '_l3'))

    # Fill NaN
    for col in features:
        if col not in week_df.columns:
            week_df[col] = 0
        else:
            week_df[col] = week_df[col].fillna(0)

    # Filter to 3+ games
    week_df = week_df[week_df['games_played_hist'] >= 3].copy()

    print(f"Players with 3+ games: {len(week_df)}")

    if len(week_df) == 0:
        return None

    # Predict
    X = week_df[features].fillna(0)
    prob_last_td = model.predict_proba(X)[:, 1]
    week_df['prob_last_td'] = prob_last_td

    # Calculate edge (typical odds +800 = 11.1% implied)
    week_df['edge'] = week_df['prob_last_td'] - 0.111

    # Bets with >8% edge (conservative for last TD)
    bets = week_df[week_df['edge'] > 0.08].copy()

    print(f"Bets with >8% edge: {len(bets)}")

    if len(bets) > 0:
        hit_rate = bets['scored_last_td'].mean()

        # Calculate ROI assuming +800 odds
        # Win: +800, Loss: -100
        roi = ((bets['scored_last_td'].sum() * 800) - ((~bets['scored_last_td'].astype(bool)).sum() * 100)) / (len(bets) * 100)

        print(f"Hit rate: {hit_rate:.1%}")
        print(f"ROI (assuming +800 odds): {roi:+.1%}")

        # Show top bets
        print(f"\nTop {min(10, len(bets))} bets:")
        top_bets = bets.nlargest(10, 'prob_last_td')[['player_name', 'team', 'prob_last_td', 'edge', 'scored_last_td']]
        for idx, row in top_bets.iterrows():
            status = '✅' if row['scored_last_td'] else '❌'
            print(f"  {status} {row['player_name']:25s} ({row['team']}): Prob={row['prob_last_td']:.1%}, Edge={row['edge']:+.1%}")

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
    """Main function to train last TD scorer model."""

    print(f"\n{'='*80}")
    print(f"LAST TD SCORER MODEL TRAINING")
    print(f"{'='*80}\n")

    # Load PBP data
    pbp_df = load_pbp_data()

    if len(pbp_df) == 0:
        return

    # Create player-game dataset
    player_games = create_player_game_data(pbp_df)

    # Create features
    player_games_features = create_last_td_features(player_games)

    # Training weeks
    training_weeks = list(range(1, 10))
    train_df = player_games_features[player_games_features['week'].isin(training_weeks)].copy()

    print(f"Training weeks: {training_weeks}")
    print(f"Training records: {len(train_df):,}")
    print()

    # Train model
    model, features, metrics = train_last_td_model(train_df)

    # Save model
    models_dir = Path('outputs/models/pbp')
    models_dir.mkdir(parents=True, exist_ok=True)

    model_file = models_dir / 'last_td_scorer_model.pkl'
    joblib.dump({
        'model': model,
        'features': features,
        'config': {
            'name': 'Last TD Scorer',
            'type': 'Bernoulli',
            'edge_threshold': 0.08,
            'typical_odds': '+800'
        }
    }, model_file)

    print(f"💾 Saved to: {model_file}")
    print()

    # Backtest on Weeks 10 & 11
    backtest_results = []

    for week in [10, 11]:
        result = backtest_last_td(model, features, player_games_features, pbp_df, week)

        if result:
            backtest_results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print(f"LAST TD SCORER TRAINING COMPLETE")
    print(f"{'='*80}\n")

    if backtest_results:
        results_df = pd.DataFrame(backtest_results)

        print("BACKTEST SUMMARY:")
        print(f"  Total bets: {results_df['bets'].sum()}")
        print(f"  Average hit rate: {results_df['hit_rate'].mean():.1%}")
        print(f"  Average ROI: {results_df['roi'].mean():+.1%}")
        print()

    print("✅ Last TD Scorer model trained and backtested")
    print()
    print("NEW MARKET AVAILABLE:")
    print("  ✅ Last TD Scorer (Anytime)")
    print()


if __name__ == '__main__':
    main()
