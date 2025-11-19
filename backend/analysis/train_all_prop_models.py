"""Train all prop models and backtest on Weeks 10 & 11.

Comprehensive training pipeline:
1. Train quantile regression models for pass_yards, rush_yards, rec_yards
2. Backtest on Week 10 (first out-of-sample test)
3. Backtest on Week 11 (second validation)
4. Calculate hit rates, ROI, calibration
5. Generate API-compatible projection files
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import json
import joblib


def load_training_data():
    """Load training datasets for all prop types."""

    print(f"\n{'='*80}")
    print(f"LOADING TRAINING DATA")
    print(f"{'='*80}\n")

    training_dir = Path('outputs/training')

    datasets = {}

    for prop_type in ['pass_yards', 'rush_yards', 'rec_yards']:
        file_path = training_dir / f'{prop_type}_training.csv'

        if file_path.exists():
            df = pd.read_csv(file_path)
            datasets[prop_type] = df
            print(f"✅ {prop_type}: {len(df)} records")
        else:
            print(f"❌ {prop_type}: File not found")

    print()
    return datasets


def train_quantile_models(prop_type, train_df):
    """Train quantile regression models for a prop type.

    Trains 5 quantile models (10th, 25th, 50th, 75th, 90th percentiles)
    to capture full distribution.
    """

    print(f"\n{'='*80}")
    print(f"TRAINING {prop_type.upper()} MODELS")
    print(f"{'='*80}\n")

    # Map prop_type to actual column name
    target_col_map = {
        'pass_yards': 'passing_yards',
        'rush_yards': 'rushing_yards',
        'rec_yards': 'receiving_yards'
    }

    target_col = target_col_map.get(prop_type, prop_type)

    # Prepare features
    feature_cols = [
        f'season_avg_{prop_type}',
        f'l3_avg_{prop_type}',
        'games_played'
    ]

    # Filter to players with at least 3 games
    train_df = train_df[train_df['games_played'] >= 3].copy()

    X = train_df[feature_cols].fillna(0)
    y = train_df[target_col]

    print(f"Training set: {len(X)} performances")
    print(f"Features: {feature_cols}")
    print()

    # Train quantile models
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

        # Evaluate on training set
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)

        print(f"  MAE: {mae:.2f} yards")

        quantile_models[f'q{int(quantile*100)}'] = model

    print()

    # Feature importance (from median model)
    print("Feature importance (median model):")
    for feat, imp in zip(feature_cols, quantile_models['q50'].feature_importances_):
        print(f"  {feat:30s}: {imp:.4f}")

    print()

    return quantile_models, feature_cols


def calculate_prob_from_quantiles(q10, q25, q50, q75, q90, line):
    """Calculate P(X > line) from quantile predictions.

    Uses linear interpolation between quantiles.
    """

    quantiles = [
        (0.10, q10),
        (0.25, q25),
        (0.50, q50),
        (0.75, q75),
        (0.90, q90)
    ]

    # If line is below q10, very likely to go over
    if line < q10:
        return 0.95

    # If line is above q90, very unlikely to go over
    if line > q90:
        return 0.05

    # Interpolate
    for i in range(len(quantiles) - 1):
        q_low, val_low = quantiles[i]
        q_high, val_high = quantiles[i + 1]

        if val_low <= line <= val_high:
            if val_high == val_low:
                prob_under = (q_low + q_high) / 2
            else:
                # Linear interpolation
                prob_under = q_low + (q_high - q_low) * (line - val_low) / (val_high - val_low)

            return 1 - prob_under

    # Fallback
    return 0.50


def backtest_week(prop_type, models, feature_cols, week, player_stats_df):
    """Backtest predictions for a specific week.

    Args:
        prop_type: 'pass_yards', 'rush_yards', or 'rec_yards'
        models: Dict of quantile models
        feature_cols: List of feature names
        week: Week number to test
        player_stats_df: Full player stats dataframe

    Returns:
        DataFrame of predictions vs actuals
    """

    print(f"\n{'='*80}")
    print(f"BACKTESTING WEEK {week} - {prop_type.upper()}")
    print(f"{'='*80}\n")

    # Map prop_type to actual column name
    target_col_map = {
        'pass_yards': 'passing_yards',
        'rush_yards': 'rushing_yards',
        'rec_yards': 'receiving_yards'
    }

    target_col = target_col_map.get(prop_type, prop_type)

    # Get Week data
    week_df = player_stats_df[player_stats_df['week'] == week].copy()

    # Calculate features (using data up to week-1)
    week_df = week_df.sort_values(['player_id', 'week'])

    # For this week, use stats from previous weeks
    historical_df = player_stats_df[player_stats_df['week'] < week].copy()

    # Calculate season averages up to this point
    player_averages = historical_df.groupby('player_id')[target_col].agg(['mean', 'count']).reset_index()
    player_averages.columns = ['player_id', f'season_avg_{prop_type}', 'games_played']

    # Calculate L3 averages
    historical_df = historical_df.sort_values(['player_id', 'week'])
    l3_avg = historical_df.groupby('player_id')[target_col].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    historical_df[f'l3_avg_{prop_type}'] = l3_avg

    player_l3 = historical_df.groupby('player_id')[f'l3_avg_{prop_type}'].last().reset_index()

    # Merge features
    week_df = week_df.merge(player_averages, on='player_id', how='left')
    week_df = week_df.merge(player_l3, on='player_id', how='left')

    # Filter to players with at least 3 games
    week_df = week_df[week_df['games_played'] >= 3].copy()

    # Fill NaN
    X = week_df[feature_cols].fillna(0)

    print(f"Players to predict: {len(week_df)}")
    print()

    # Make predictions with all quantile models
    predictions = {}
    for model_name, model in models.items():
        predictions[model_name] = model.predict(X)

    # Store predictions in dataframe
    for model_name, preds in predictions.items():
        week_df[model_name] = preds

    # Show sample predictions
    print("Sample predictions:")
    print()

    for idx in week_df.head(10).index:
        player = week_df.loc[idx, 'player_name']
        actual = week_df.loc[idx, target_col]

        q10 = week_df.loc[idx, 'q10']
        q50 = week_df.loc[idx, 'q50']
        q90 = week_df.loc[idx, 'q90']

        print(f"  {player:25s}: Q10={q10:5.1f} Q50={q50:5.1f} Q90={q90:5.1f} | Actual: {actual:5.1f}")

    print()

    # Add target_col to returned dataframe for later use
    week_df['target_value'] = week_df[target_col]

    return week_df


def evaluate_prop_lines(predictions_df, prop_type, typical_lines):
    """Evaluate predictions against typical DraftKings lines.

    Args:
        predictions_df: DataFrame with quantile predictions and actuals
        prop_type: Prop type name
        typical_lines: Dict of player_name -> typical line value

    Returns:
        Dict with hit rates and metrics
    """

    print(f"\n{'='*80}")
    print(f"EVALUATING VS TYPICAL LINES - {prop_type.upper()}")
    print(f"{'='*80}\n")

    results = []

    for idx, row in predictions_df.iterrows():
        player = row['player_name']
        actual = row['target_value']

        # Get quantile predictions
        q10 = row['q10']
        q25 = row['q25']
        q50 = row['q50']
        q75 = row['q75']
        q90 = row['q90']

        # Use typical line (or median prediction if not available)
        if player in typical_lines:
            line = typical_lines[player]
        else:
            line = q50  # Use median as line

        # Calculate P(X > line)
        prob_over = calculate_prob_from_quantiles(q10, q25, q50, q75, q90, line)

        # Determine bet
        if prob_over > 0.60:  # 60% threshold (vs 52.4% break-even at -110)
            bet_side = 'OVER'
            bet_prob = prob_over
            edge = prob_over - 0.524
        elif prob_over < 0.40:  # 40% = 60% for under
            bet_side = 'UNDER'
            bet_prob = 1 - prob_over
            edge = (1 - prob_over) - 0.524
        else:
            bet_side = 'PASS'
            bet_prob = max(prob_over, 1 - prob_over)
            edge = 0

        # Determine if bet hit
        if bet_side == 'OVER':
            hit = actual > line
        elif bet_side == 'UNDER':
            hit = actual < line
        else:
            hit = None

        results.append({
            'player': player,
            'line': line,
            'actual': actual,
            'bet_side': bet_side,
            'bet_prob': bet_prob,
            'edge': edge,
            'hit': hit,
            'q10': q10,
            'q25': q25,
            'q50': q50,
            'q75': q75,
            'q90': q90,
        })

    results_df = pd.DataFrame(results)

    # Calculate metrics (only on bets placed)
    bets = results_df[results_df['bet_side'] != 'PASS'].copy()

    if len(bets) > 0:
        hit_rate = bets['hit'].mean()
        total_bets = len(bets)
        wins = bets['hit'].sum()
        losses = total_bets - wins

        # ROI calculation (assuming -110 odds)
        total_staked = total_bets * 100
        winnings = wins * 190  # Win $90 profit per $100 bet
        net_profit = winnings - total_staked
        roi = net_profit / total_staked

        print(f"📊 Betting Performance:")
        print(f"   Total bets: {total_bets}")
        print(f"   Wins: {wins}")
        print(f"   Losses: {losses}")
        print(f"   Hit rate: {hit_rate:.1%}")
        print(f"   Net profit: ${net_profit:+,.0f}")
        print(f"   ROI: {roi:+.1%}")
        print()

        # Show best bets
        print(f"🎯 Top 5 Bets (by edge):")
        top_bets = bets.nlargest(5, 'edge')
        for idx, bet in top_bets.iterrows():
            status = "✅ HIT" if bet['hit'] else "❌ MISS"
            print(f"  {status} - {bet['player']} {bet['bet_side']} {bet['line']:.1f}")
            print(f"    Actual: {bet['actual']:.1f} | Edge: {bet['edge']:+.1%} | Prob: {bet['bet_prob']:.1%}")

        print()

        return {
            'total_bets': total_bets,
            'wins': wins,
            'losses': losses,
            'hit_rate': hit_rate,
            'roi': roi,
            'net_profit': net_profit,
            'results_df': results_df,
        }

    else:
        print("❌ No bets placed (no edges found)")
        return {'total_bets': 0, 'results_df': results_df}


def main():
    """Main training and backtesting pipeline."""

    print(f"\n{'#'*80}")
    print(f"# COMPREHENSIVE PROP MODEL TRAINING & BACKTESTING")
    print(f"{'#'*80}\n")

    # Load training data
    datasets = load_training_data()

    if len(datasets) == 0:
        print("❌ No training data found. Run fetch_player_stats_2025.py first.")
        return

    # Load full player stats for backtesting
    player_stats = pd.read_csv('inputs/player_stats_2025_synthetic.csv')

    # Train models for each prop type
    all_models = {}
    all_metrics = {}

    for prop_type, train_df in datasets.items():
        # Train models
        models, feature_cols = train_quantile_models(prop_type, train_df)

        all_models[prop_type] = {
            'models': models,
            'features': feature_cols
        }

        # Backtest Week 10
        week10_preds = backtest_week(prop_type, models, feature_cols, 10, player_stats)

        # Backtest Week 11
        week11_preds = backtest_week(prop_type, models, feature_cols, 11, player_stats)

        # Define typical lines (based on our earlier backtesting)
        typical_lines = {}

        if prop_type == 'pass_yards':
            typical_lines = {
                'Josh Allen': 265.5,
                'Tua Tagovailoa': 235.5,
                'Lamar Jackson': 215.5,
                'Jared Goff': 255.5,
                'Sam Darnold': 235.5,
                'Matthew Stafford': 245.5,
            }
        elif prop_type == 'rush_yards':
            typical_lines = {
                'Derrick Henry': 85.5,
                'Bijan Robinson': 75.5,
                'Kenneth Walker': 68.5,
            }
        elif prop_type == 'rec_yards':
            typical_lines = {
                'Amon-Ra St. Brown': 75.5,
                'Puka Nacua': 72.5,
                'Justin Jefferson': 82.5,
            }

        # Evaluate Week 10
        week10_metrics = evaluate_prop_lines(week10_preds, prop_type, typical_lines)

        # Evaluate Week 11
        week11_metrics = evaluate_prop_lines(week11_preds, prop_type, typical_lines)

        all_metrics[prop_type] = {
            'week10': week10_metrics,
            'week11': week11_metrics,
        }

    # Save models
    models_dir = Path('outputs/models')
    models_dir.mkdir(parents=True, exist_ok=True)

    for prop_type, model_data in all_models.items():
        model_file = models_dir / f'{prop_type}_models.pkl'
        joblib.dump(model_data, model_file)
        print(f"💾 Saved {prop_type} models to: {model_file}")

    print()

    # Overall summary
    print(f"\n{'='*80}")
    print(f"📊 OVERALL SUMMARY")
    print(f"{'='*80}\n")

    for prop_type, metrics in all_metrics.items():
        print(f"\n{prop_type.upper()}:")

        for week_name, week_metrics in metrics.items():
            if week_metrics.get('total_bets', 0) > 0:
                print(f"  {week_name}:")
                print(f"    Hit rate: {week_metrics['hit_rate']:.1%}")
                print(f"    ROI: {week_metrics['roi']:+.1%}")
                print(f"    Bets: {week_metrics['wins']}/{week_metrics['total_bets']}")

    print()

    print(f"\n{'='*80}")
    print(f"✅ TRAINING COMPLETE")
    print(f"{'='*80}\n")

    print("Models saved to: outputs/models/")
    print()
    print("Next steps:")
    print("  1. Generate API projection files")
    print("  2. Test betting endpoints")
    print("  3. Integrate with Odds API for real lines")
    print()


if __name__ == '__main__':
    main()
