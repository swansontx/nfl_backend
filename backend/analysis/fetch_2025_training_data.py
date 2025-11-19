"""Fetch all necessary 2025 season data for model training.

Downloads:
- Games data (schedules, scores, spreads)
- Player stats per game
- Rosters
- Injury data

Strategy: Train on Weeks 1-9, test on Week 10+
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
import time


def fetch_2025_games():
    """Fetch 2025 games data."""

    print(f"\n{'='*80}")
    print(f"FETCHING 2025 SEASON DATA FOR MODEL TRAINING")
    print(f"{'='*80}\n")

    games_file = Path('inputs/games_2025.csv')

    if games_file.exists():
        games = pd.read_csv(games_file)
        print(f"✅ Loaded games data: {len(games)} total games in dataset")

        # Filter to 2025 regular season
        games_2025 = games[
            (games['season'] == 2025) &
            (games['game_type'] == 'REG')
        ].copy()

        print(f"   2025 Regular Season: {len(games_2025)} games")

        # Check weeks available
        weeks_available = sorted(games_2025['week'].unique())
        print(f"   Weeks available: {weeks_available}")

        # Count completed games
        completed = games_2025[games_2025['away_score'].notna()]
        print(f"   Completed games: {len(completed)}")
        print(f"   Incomplete games: {len(games_2025) - len(completed)}")

        return games_2025
    else:
        print(f"❌ Games file not found: {games_file}")
        return pd.DataFrame()


def analyze_available_data(games_df):
    """Analyze what data we have to work with."""

    print(f"\n{'='*80}")
    print(f"DATA AVAILABILITY ANALYSIS")
    print(f"{'='*80}\n")

    # Weeks with completed games
    completed = games_df[games_df['away_score'].notna()].copy()

    weeks_by_count = completed.groupby('week').size().reset_index(name='games')

    print("📊 Completed games by week:")
    for idx, row in weeks_by_count.iterrows():
        week = row['week']
        count = row['games']
        print(f"   Week {week:2d}: {count:2d} games")

    print()

    # Available features in games data
    print("📋 Available features in games data:")
    key_features = [
        'spread_line', 'total_line', 'away_moneyline', 'home_moneyline',
        'away_rest', 'home_rest', 'div_game', 'roof', 'surface',
        'temp', 'wind'
    ]

    for feature in key_features:
        if feature in games_df.columns:
            non_null = games_df[feature].notna().sum()
            pct = non_null / len(games_df) * 100
            print(f"   ✅ {feature}: {pct:.1f}% coverage")
        else:
            print(f"   ❌ {feature}: Not available")

    print()

    return completed


def create_training_features(games_df, training_weeks):
    """Create features for model training from games data.

    Features we can extract:
    - Division game (div_game)
    - Rest advantage (home_rest - away_rest)
    - Spread/total (market expectations)
    - Weather (temp, wind, roof)
    - Home/away records
    """

    print(f"\n{'='*80}")
    print(f"CREATING TRAINING FEATURES")
    print(f"{'='*80}\n")

    # Filter to training weeks
    train_df = games_df[games_df['week'].isin(training_weeks)].copy()

    print(f"Training set: Weeks {min(training_weeks)}-{max(training_weeks)}")
    print(f"Training games: {len(train_df)}")
    print()

    # Feature engineering
    print("🔧 Engineering features:")

    # 1. Rest advantage
    train_df['rest_advantage'] = train_df['home_rest'] - train_df['away_rest']
    print(f"   ✅ rest_advantage (home_rest - away_rest)")

    # 2. Is pick'em game
    train_df['is_pickem'] = (train_df['spread_line'].abs() < 2.0).astype(int)
    print(f"   ✅ is_pickem (spread < 2 points)")

    # 3. Is high total game
    train_df['is_high_total'] = (train_df['total_line'] > 48.0).astype(int)
    print(f"   ✅ is_high_total (total > 48)")

    # 4. Is low total game
    train_df['is_low_total'] = (train_df['total_line'] < 40.0).astype(int)
    print(f"   ✅ is_low_total (total < 40)")

    # 5. Division game
    if 'div_game' in train_df.columns:
        print(f"   ✅ div_game (already present)")
    else:
        print(f"   ⚠️  div_game not available")

    # 6. Implied team totals
    train_df['home_implied_total'] = (train_df['total_line'] - train_df['spread_line']) / 2
    train_df['away_implied_total'] = (train_df['total_line'] + train_df['spread_line']) / 2
    print(f"   ✅ implied_totals (split total by spread)")

    # 7. Weather factors
    if 'roof' in train_df.columns:
        train_df['is_dome'] = (train_df['roof'] == 'dome').astype(int)
        print(f"   ✅ is_dome")

    if 'wind' in train_df.columns:
        train_df['has_wind'] = (train_df['wind'] > 10).astype(int)
        print(f"   ✅ has_wind (>10 mph)")

    print()

    # Target variables
    print("🎯 Creating target variables:")

    train_df['home_score'] = pd.to_numeric(train_df['home_score'], errors='coerce')
    train_df['away_score'] = pd.to_numeric(train_df['away_score'], errors='coerce')

    train_df['actual_total'] = train_df['home_score'] + train_df['away_score']
    train_df['actual_spread'] = train_df['home_score'] - train_df['away_score']
    train_df['home_won'] = (train_df['home_score'] > train_df['away_score']).astype(int)

    print(f"   ✅ actual_total")
    print(f"   ✅ actual_spread")
    print(f"   ✅ home_won")
    print()

    # Remove incomplete games
    train_df = train_df[train_df['actual_total'].notna()].copy()

    print(f"📊 Final training set: {len(train_df)} complete games")
    print()

    return train_df


def generate_insights_from_training_data(train_df):
    """Generate insights from the training data."""

    print(f"\n{'='*80}")
    print(f"TRAINING DATA INSIGHTS")
    print(f"{'='*80}\n")

    # Division game performance
    if 'div_game' in train_df.columns:
        div_games = train_df[train_df['div_game'] == 1]
        non_div_games = train_df[train_df['div_game'] == 0]

        print(f"🏈 Division Games:")
        print(f"   Count: {len(div_games)}")

        # Favorite covering rate
        div_games['favorite_covered'] = (
            ((div_games['spread_line'] < 0) & (div_games['actual_spread'] < div_games['spread_line'])) |
            ((div_games['spread_line'] > 0) & (div_games['actual_spread'] > div_games['spread_line']))
        )

        div_cover_rate = div_games['favorite_covered'].mean()

        non_div_games['favorite_covered'] = (
            ((non_div_games['spread_line'] < 0) & (non_div_games['actual_spread'] < non_div_games['spread_line'])) |
            ((non_div_games['spread_line'] > 0) & (non_div_games['actual_spread'] > non_div_games['spread_line']))
        )

        non_div_cover_rate = non_div_games['favorite_covered'].mean()

        print(f"   Favorite cover rate: {div_cover_rate:.1%}")
        print(f"   Non-division cover rate: {non_div_cover_rate:.1%}")
        print(f"   Difference: {(div_cover_rate - non_div_cover_rate) * 100:+.1f} percentage points")
        print()

    # Rest advantage impact
    if 'rest_advantage' in train_df.columns:
        rest_advantage_games = train_df[train_df['rest_advantage'] >= 3]

        print(f"💤 Rest Advantage (3+ days):")
        print(f"   Count: {len(rest_advantage_games)}")

        if len(rest_advantage_games) > 0:
            rest_adv_win_rate = rest_advantage_games['home_won'].mean()
            overall_home_win_rate = train_df['home_won'].mean()

            print(f"   Home win rate with rest advantage: {rest_adv_win_rate:.1%}")
            print(f"   Overall home win rate: {overall_home_win_rate:.1%}")
            print(f"   Lift: {(rest_adv_win_rate - overall_home_win_rate) * 100:+.1f} percentage points")
        print()

    # High total games
    if 'is_high_total' in train_df.columns:
        high_total_games = train_df[train_df['is_high_total'] == 1]

        print(f"📈 High Total Games (>48):")
        print(f"   Count: {len(high_total_games)}")

        if len(high_total_games) > 0:
            avg_actual = high_total_games['actual_total'].mean()
            avg_predicted = high_total_games['total_line'].mean()

            print(f"   Average predicted total: {avg_predicted:.1f}")
            print(f"   Average actual total: {avg_actual:.1f}")
            print(f"   Bias: {avg_actual - avg_predicted:+.1f} points")
        print()

    # Pick'em games
    if 'is_pickem' in train_df.columns:
        pickem_games = train_df[train_df['is_pickem'] == 1]

        print(f"⚖️  Pick'em Games (<2 pt spread):")
        print(f"   Count: {len(pickem_games)}")

        if len(pickem_games) > 0:
            home_win_rate = pickem_games['home_won'].mean()
            print(f"   Home win rate: {home_win_rate:.1%}")
            print(f"   (Should be ~50% if truly even)")
        print()


def save_training_data(train_df):
    """Save processed training data."""

    outputs_dir = Path('outputs')
    outputs_dir.mkdir(parents=True, exist_ok=True)

    output_file = outputs_dir / 'training_data_weeks_1_9.csv'

    train_df.to_csv(output_file, index=False)

    print(f"💾 Saved training data to: {output_file}")
    print(f"   Rows: {len(train_df)}")
    print(f"   Columns: {len(train_df.columns)}")
    print()


def main():
    """Main data fetching and processing pipeline."""

    # 1. Fetch games data
    games = fetch_2025_games()

    if len(games) == 0:
        print("❌ No games data available")
        return

    # 2. Analyze available data
    completed = analyze_available_data(games)

    # 3. Create training features (Weeks 1-9)
    training_weeks = list(range(1, 10))
    train_df = create_training_features(completed, training_weeks)

    # 4. Generate insights
    generate_insights_from_training_data(train_df)

    # 5. Save training data
    save_training_data(train_df)

    print(f"\n{'='*80}")
    print(f"✅ DATA PREPARATION COMPLETE")
    print(f"{'='*80}\n")

    print(f"Next steps:")
    print(f"  1. Train models on outputs/training_data_weeks_1_9.csv")
    print(f"  2. Make predictions for Week 10")
    print(f"  3. Compare to actual Week 10 results")
    print()


if __name__ == '__main__':
    main()
