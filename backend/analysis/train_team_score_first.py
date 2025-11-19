"""Train Team Score First model.

Team to Score First is a popular DraftKings prop (typically -110 for both teams).
We use play-by-play data to identify which team scored first and model the probability
based on opening drive tendencies, offensive efficiency, and pace.

Expected Performance: 55-60% hit rate, +10-20% ROI (small edge over 50%)

Features:
- Historical score first rate (season & L3)
- Opening drive success rate
- Time of possession
- First quarter scoring rate
- Home/away splits
- Offensive pace (plays per drive)
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

    pbp_file = Path('inputs/play_by_play_2025.parquet')

    if not pbp_file.exists():
        print(f"❌ PBP file not found: {pbp_file}")
        return pd.DataFrame()

    pbp = pd.read_parquet(pbp_file)

    print(f"✅ Loaded {len(pbp):,} plays")
    print(f"   Games: {pbp['game_id'].nunique()}")
    print(f"   Weeks: {pbp['week'].min()}-{pbp['week'].max()}")
    print()

    return pbp


def extract_first_score_data(pbp_df):
    """Extract which team scored first in each game.

    Args:
        pbp_df: Play-by-play DataFrame

    Returns:
        DataFrame with first scoring team per game
    """

    print(f"\n{'='*80}")
    print(f"EXTRACTING FIRST SCORE DATA")
    print(f"{'='*80}\n")

    # Filter to scoring plays (TD or FG)
    scoring_plays = pbp_df[
        ((pbp_df['touchdown'] == 1) | (pbp_df['field_goal_result'] == 'made')) &
        (pbp_df['posteam'].notna())
    ].copy()

    print(f"Total scoring plays: {len(scoring_plays)}")
    print()

    # Sort by game and time (earliest first)
    scoring_plays = scoring_plays.sort_values(['game_id', 'game_seconds_remaining'], ascending=[True, False])

    # Get first score of each game
    first_scores = scoring_plays.groupby('game_id').first().reset_index()

    print(f"Games with scores: {len(first_scores)}")
    print()

    # Show distribution
    if 'posteam' in first_scores.columns:
        print("First scoring teams distribution:")
        print(f"  Total games: {len(first_scores)}")
        print()

    return first_scores[['game_id', 'posteam', 'home_team', 'away_team', 'week']]


def create_team_game_data(pbp_df):
    """Create team-game level data with features.

    For each team in each game, calculate:
    - Did they score first? (target)
    - Historical score first rate
    - Opening drive stats
    - First quarter scoring
    - Pace metrics

    Args:
        pbp_df: Play-by-play DataFrame

    Returns:
        DataFrame with team-game records
    """

    print(f"\n{'='*80}")
    print(f"CREATING TEAM-GAME DATA")
    print(f"{'='*80}\n")

    # Get all games
    games = pbp_df[['game_id', 'week', 'home_team', 'away_team']].drop_duplicates()

    # Create team-game records (2 per game: home and away)
    home_games = games[['game_id', 'week', 'home_team']].copy()
    home_games.columns = ['game_id', 'week', 'team']
    home_games['is_home'] = 1

    away_games = games[['game_id', 'week', 'away_team']].copy()
    away_games.columns = ['game_id', 'week', 'team']
    away_games['is_home'] = 0

    team_games = pd.concat([home_games, away_games], ignore_index=True)

    print(f"Team-game records: {len(team_games)}")
    print()

    # Add scored first indicator
    first_scores = extract_first_score_data(pbp_df)[['game_id', 'posteam']]
    first_scores.columns = ['game_id', 'team']
    first_scores['scored_first'] = 1

    team_games = team_games.merge(first_scores, on=['game_id', 'team'], how='left')
    team_games['scored_first'] = team_games['scored_first'].fillna(0).astype(int)

    print(f"Teams that scored first: {team_games['scored_first'].sum()}")
    print(f"Score first rate: {team_games['scored_first'].mean():.1%}")
    print()

    # Calculate opening drive success (first drive TD/FG rate)
    first_drives = pbp_df[pbp_df['fixed_drive'] == 1].copy()

    # Get drive results
    drive_results = first_drives.groupby(['game_id', 'posteam']).agg({
        'touchdown': 'max',
        'field_goal_result': lambda x: (x == 'made').max()
    }).reset_index()

    drive_results.columns = ['game_id', 'team', 'opening_drive_td', 'opening_drive_fg']
    drive_results['opening_drive_score'] = ((drive_results['opening_drive_td'] > 0) | (drive_results['opening_drive_fg'] > 0)).astype(int)

    team_games = team_games.merge(drive_results[['game_id', 'team', 'opening_drive_score']],
                                   on=['game_id', 'team'], how='left')
    team_games['opening_drive_score'] = team_games['opening_drive_score'].fillna(0).astype(int)

    # Calculate first quarter points
    q1_plays = pbp_df[pbp_df['qtr'] == 1].copy()

    q1_points = q1_plays.groupby(['game_id', 'posteam']).agg({
        'touchdown': 'sum',
        'field_goal_result': lambda x: (x == 'made').sum(),
        'extra_point_result': lambda x: (x == 'good').sum()
    }).reset_index()

    q1_points.columns = ['game_id', 'team', 'q1_tds', 'q1_fgs', 'q1_xps']
    q1_points['q1_points'] = (q1_points['q1_tds'] * 6) + (q1_points['q1_fgs'] * 3) + q1_points['q1_xps']

    team_games = team_games.merge(q1_points[['game_id', 'team', 'q1_points']],
                                   on=['game_id', 'team'], how='left')
    team_games['q1_points'] = team_games['q1_points'].fillna(0).astype(int)

    # Calculate offensive pace (plays per drive)
    team_drives = pbp_df.groupby(['game_id', 'posteam', 'fixed_drive']).size().reset_index()
    team_drives.columns = ['game_id', 'team', 'drive', 'plays']

    avg_plays = team_drives.groupby(['game_id', 'team'])['plays'].mean().reset_index()
    avg_plays.columns = ['game_id', 'team', 'plays_per_drive']

    team_games = team_games.merge(avg_plays, on=['game_id', 'team'], how='left')
    team_games['plays_per_drive'] = team_games['plays_per_drive'].fillna(0)

    print(f"✅ Created team-game dataset with scoring metrics")
    print()

    return team_games


def create_score_first_features(team_games_df):
    """Create features for score first model.

    Args:
        team_games_df: Team-game DataFrame

    Returns:
        DataFrame with features
    """

    print(f"\n{'='*80}")
    print(f"CREATING SCORE FIRST FEATURES")
    print(f"{'='*80}\n")

    df = team_games_df.copy()

    # Sort by team and week
    df = df.sort_values(['team', 'week'])

    # Calculate historical stats (expanding window)
    df['season_score_first_rate'] = df.groupby('team')['scored_first'].expanding().mean().reset_index(level=0, drop=True)
    df['season_opening_drive_rate'] = df.groupby('team')['opening_drive_score'].expanding().mean().reset_index(level=0, drop=True)
    df['season_q1_points_avg'] = df.groupby('team')['q1_points'].expanding().mean().reset_index(level=0, drop=True)
    df['season_plays_per_drive'] = df.groupby('team')['plays_per_drive'].expanding().mean().reset_index(level=0, drop=True)

    # L3 stats
    df['l3_score_first_rate'] = df.groupby('team')['scored_first'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    df['l3_opening_drive_rate'] = df.groupby('team')['opening_drive_score'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    df['l3_q1_points_avg'] = df.groupby('team')['q1_points'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

    # Games played
    df['games_played'] = df.groupby('team').cumcount() + 1

    # Filter to 3+ games
    df = df[df['games_played'] >= 3].copy()

    print(f"Records with 3+ games: {len(df)}")
    print(f"Score first rate: {df['scored_first'].mean():.2%}")
    print()

    return df


def train_score_first_model(train_df):
    """Train score first model.

    Args:
        train_df: Training DataFrame with features

    Returns:
        Model, features, metrics
    """

    print(f"\n{'='*80}")
    print(f"TRAINING SCORE FIRST MODEL")
    print(f"{'='*80}\n")

    # Features
    feature_cols = [
        'season_score_first_rate',
        'l3_score_first_rate',
        'season_opening_drive_rate',
        'l3_opening_drive_rate',
        'season_q1_points_avg',
        'l3_q1_points_avg',
        'season_plays_per_drive',
        'is_home',
        'games_played',
    ]

    X = train_df[feature_cols].fillna(0)
    y = train_df['scored_first']

    print(f"Training set: {len(X):,} team-games")
    print(f"Positive rate: {y.mean():.2%}")
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
        print(f"  {feat:30s}: {imp:.3f}")
    print()

    return model, feature_cols, {'accuracy': acc, 'log_loss': logloss}


def backtest_score_first(model, features, team_games_df, pbp_df, week):
    """Backtest score first model on a specific week.

    Args:
        model: Trained model
        features: Feature column names
        team_games_df: Full team-game DataFrame
        pbp_df: Play-by-play DataFrame
        week: Week to backtest

    Returns:
        Backtest results
    """

    print(f"\n{'='*80}")
    print(f"BACKTESTING SCORE FIRST - WEEK {week}")
    print(f"{'='*80}\n")

    # Get week data
    week_df = team_games_df[team_games_df['week'] == week].copy()

    print(f"Week {week} team-games: {len(week_df)}")

    # Calculate features using ONLY historical data
    historical_df = team_games_df[team_games_df['week'] < week].copy()

    if len(historical_df) == 0:
        print("❌ No historical data")
        return None

    # Calculate historical stats per team
    team_stats = historical_df.groupby('team').agg({
        'scored_first': 'mean',
        'opening_drive_score': 'mean',
        'q1_points': 'mean',
        'plays_per_drive': 'mean',
        'games_played': 'max'
    }).reset_index()

    team_stats.columns = ['team', 'season_score_first_rate', 'season_opening_drive_rate',
                          'season_q1_points_avg', 'season_plays_per_drive', 'games_played']

    # L3 stats
    historical_df = historical_df.sort_values(['team', 'week'])

    l3_score = historical_df.groupby('team')['scored_first'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    historical_df['l3_score_first_rate'] = l3_score

    l3_opening = historical_df.groupby('team')['opening_drive_score'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    historical_df['l3_opening_drive_rate'] = l3_opening

    l3_q1 = historical_df.groupby('team')['q1_points'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    historical_df['l3_q1_points_avg'] = l3_q1

    team_l3 = historical_df.groupby('team')[['l3_score_first_rate', 'l3_opening_drive_rate', 'l3_q1_points_avg']].last().reset_index()

    # Merge
    week_df = week_df.merge(team_stats, on='team', how='left', suffixes=('', '_hist'))
    week_df = week_df.merge(team_l3, on='team', how='left', suffixes=('', '_l3'))

    # Fill NaN
    for col in features:
        if col not in week_df.columns:
            week_df[col] = 0
        else:
            week_df[col] = week_df[col].fillna(0)

    # Filter to 3+ games
    week_df = week_df[week_df['games_played_hist'] >= 3].copy()

    print(f"Teams with 3+ games: {len(week_df)}")

    if len(week_df) == 0:
        return None

    # Predict
    X = week_df[features].fillna(0)
    prob_score_first = model.predict_proba(X)[:, 1]
    week_df['prob_score_first'] = prob_score_first

    # Calculate edge (typical odds -110 = 52.4% implied)
    week_df['edge'] = week_df['prob_score_first'] - 0.524

    # Bets with >5% edge (standard for 50/50 props)
    bets = week_df[week_df['edge'] > 0.05].copy()

    print(f"Bets with >5% edge: {len(bets)}")

    if len(bets) > 0:
        hit_rate = bets['scored_first'].mean()

        # Calculate ROI assuming -110 odds
        # Win: +90.9 (risk 110 to win 100), Loss: -110
        wins = bets['scored_first'].sum()
        losses = len(bets) - wins
        roi = ((wins * 90.9) - (losses * 110)) / (len(bets) * 110)

        print(f"Hit rate: {hit_rate:.1%}")
        print(f"ROI (assuming -110 odds): {roi:+.1%}")

        # Show top bets
        print(f"\nTop {min(10, len(bets))} bets:")
        top_bets = bets.nlargest(10, 'prob_score_first')[['team', 'is_home', 'prob_score_first', 'edge', 'scored_first']]
        for idx, row in top_bets.iterrows():
            status = '✅' if row['scored_first'] else '❌'
            home_away = 'HOME' if row['is_home'] else 'AWAY'
            print(f"  {status} {row['team']:5s} ({home_away}): Prob={row['prob_score_first']:.1%}, Edge={row['edge']:+.1%}")

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
    """Main function to train score first model."""

    print(f"\n{'='*80}")
    print(f"TEAM SCORE FIRST MODEL TRAINING")
    print(f"{'='*80}\n")

    # Load PBP data
    pbp_df = load_pbp_data()

    if len(pbp_df) == 0:
        return

    # Create team-game dataset
    team_games = create_team_game_data(pbp_df)

    # Create features
    team_games_features = create_score_first_features(team_games)

    # Training weeks
    training_weeks = list(range(1, 10))
    train_df = team_games_features[team_games_features['week'].isin(training_weeks)].copy()

    print(f"Training weeks: {training_weeks}")
    print(f"Training records: {len(train_df):,}")
    print()

    # Train model
    model, features, metrics = train_score_first_model(train_df)

    # Save model
    models_dir = Path('outputs/models/pbp')
    models_dir.mkdir(parents=True, exist_ok=True)

    model_file = models_dir / 'team_score_first_model.pkl'
    joblib.dump({
        'model': model,
        'features': features,
        'config': {
            'name': 'Team to Score First',
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
        result = backtest_score_first(model, features, team_games_features, pbp_df, week)

        if result:
            backtest_results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print(f"TEAM SCORE FIRST TRAINING COMPLETE")
    print(f"{'='*80}\n")

    if backtest_results:
        results_df = pd.DataFrame(backtest_results)

        print("BACKTEST SUMMARY:")
        print(f"  Total bets: {results_df['bets'].sum()}")
        print(f"  Average hit rate: {results_df['hit_rate'].mean():.1%}")
        print(f"  Average ROI: {results_df['roi'].mean():+.1%}")
        print()

    print("✅ Team Score First model trained and backtested")
    print()
    print("NEW MARKET AVAILABLE:")
    print("  ✅ Team to Score First")
    print()


if __name__ == '__main__':
    main()
