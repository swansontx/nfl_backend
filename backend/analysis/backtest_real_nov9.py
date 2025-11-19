"""Real backtest for November 9, 2025 (Week 10 Sunday).

This uses ACTUAL game data with REAL results.
Makes predictions using ONLY pre-game information (Vegas lines).
Then compares to actual results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def load_week10_games():
    """Load actual Week 10 Sunday games from Nov 9, 2025."""

    print(f"\n{'='*80}")
    print(f"REAL BACKTEST: November 9, 2025 (Week 10 Sunday)")
    print(f"{'='*80}\n")

    # Load games CSV
    games_file = Path('inputs/games_2025.csv')

    if not games_file.exists():
        print(f"❌ Games file not found: {games_file}")
        return pd.DataFrame()

    # Read all games
    all_games = pd.read_csv(games_file)

    # Filter to Week 10 Sunday 2025
    week10 = all_games[
        (all_games['season'] == 2025) &
        (all_games['week'] == 10) &
        (all_games['weekday'] == 'Sunday') &
        (all_games['game_type'] == 'REG')
    ].copy()

    print(f"📅 Found {len(week10)} real games from November 9, 2025:")
    print()

    for idx, game in week10.iterrows():
        away = game['away_team']
        home = game['home_team']
        away_score = game['away_score']
        home_score = game['home_score']
        spread = game.get('spread_line', np.nan)

        print(f"  {away} @ {home}")
        print(f"    Spread: {spread:+.1f} | Final: {away} {away_score}, {home} {home_score}")

    print()
    return week10


def make_predictions_blind(games_df):
    """Make predictions WITHOUT looking at actual scores.

    Uses ONLY pre-game information (Vegas lines).
    """

    print(f"\n{'='*80}")
    print(f"STEP 1: PRE-GAME PREDICTIONS (No Peeking!)")
    print(f"{'='*80}\n")

    print("📊 Making predictions using ONLY Vegas lines (pre-game):")
    print()

    predictions = []

    for idx, game in games_df.iterrows():
        game_id = game['game_id']
        away = game['away_team']
        home = game['home_team']
        spread = game.get('spread_line', 0)
        total = game.get('total_line', 45)

        # Predict winner from spread
        if spread < 0:
            predicted_winner = away
            home_win_prob = 1 / (1 + np.exp(-spread / 4))
        else:
            predicted_winner = home
            home_win_prob = 1 / (1 + np.exp(spread / 4))

        # Predict scores from total + spread
        # If spread is +6.5 (away favored by 6.5):
        # away_implied = (total + spread) / 2
        # home_implied = (total - spread) / 2

        home_implied = (total - spread) / 2
        away_implied = (total + spread) / 2

        pred_home = round(home_implied)
        pred_away = round(away_implied)

        predictions.append({
            'game_id': game_id,
            'away_team': away,
            'home_team': home,
            'predicted_winner': predicted_winner,
            'home_win_prob': home_win_prob,
            'predicted_home_score': pred_home,
            'predicted_away_score': pred_away,
            'predicted_spread': spread,
            'predicted_total': total,
        })

        print(f"🏈 {away} @ {home}")
        print(f"   Prediction: {away} {pred_away}, {home} {pred_home}")
        print(f"   Winner: {predicted_winner} ({home_win_prob:.1%})")
        print(f"   Line: {spread:+.1f}, Total: {total:.1f}")
        print()

    return pd.DataFrame(predictions)


def compare_to_reality(predictions_df, games_df):
    """Compare predictions to ACTUAL results."""

    print(f"\n{'='*80}")
    print(f"STEP 2: COMPARING TO ACTUAL RESULTS")
    print(f"{'='*80}\n")

    # Merge predictions with actuals
    merged = predictions_df.merge(
        games_df[['game_id', 'away_team', 'home_team', 'away_score', 'home_score']],
        on=['game_id', 'away_team', 'home_team']
    )

    # Calculate actual metrics
    merged['actual_winner'] = merged.apply(
        lambda x: x['home_team'] if x['home_score'] > x['away_score'] else x['away_team'],
        axis=1
    )
    merged['actual_total'] = merged['away_score'] + merged['home_score']
    merged['actual_spread'] = merged['home_score'] - merged['away_score']

    # Calculate accuracy
    winner_correct = (merged['predicted_winner'] == merged['actual_winner']).sum()
    winner_accuracy = winner_correct / len(merged)

    spread_errors = abs(merged['predicted_spread'] - merged['actual_spread'])
    spread_mae = spread_errors.mean()
    spread_within_3 = (spread_errors <= 3).sum() / len(merged)
    spread_within_7 = (spread_errors <= 7).sum() / len(merged)

    total_errors = abs(merged['predicted_total'] - merged['actual_total'])
    total_mae = total_errors.mean()
    total_within_3 = (total_errors <= 3).sum() / len(merged)
    total_within_7 = (total_errors <= 7).sum() / len(merged)

    # Print summary
    print("🎯 PREDICTION ACCURACY:")
    print(f"   Games: {len(merged)}")
    print(f"   Winner picks: {winner_correct}/{len(merged)} ({winner_accuracy:.1%})")
    print()
    print(f"   Spread MAE: {spread_mae:.2f} points")
    print(f"   Spread within 3: {spread_within_3:.1%}")
    print(f"   Spread within 7: {spread_within_7:.1%}")
    print()
    print(f"   Total MAE: {total_mae:.2f} points")
    print(f"   Total within 3: {total_within_3:.1%}")
    print(f"   Total within 7: {total_within_7:.1%}")
    print()

    # Game-by-game breakdown
    print("📋 GAME-BY-GAME RESULTS:")
    print()

    for idx, game in merged.iterrows():
        away = game['away_team']
        home = game['home_team']

        pred_away = game['predicted_away_score']
        pred_home = game['predicted_home_score']
        actual_away = int(game['away_score'])
        actual_home = int(game['home_score'])

        winner_status = "✅" if game['predicted_winner'] == game['actual_winner'] else "❌"
        spread_error = abs(game['predicted_spread'] - game['actual_spread'])
        total_error = abs(game['predicted_total'] - game['actual_total'])

        print(f"  {away} @ {home}")
        print(f"    Predicted: {away} {pred_away}, {home} {pred_home}")
        print(f"    Actual:    {away} {actual_away}, {home} {actual_home}")
        print(f"    Winner: {winner_status} (predicted {game['predicted_winner']}, was {game['actual_winner']})")
        print(f"    Spread error: {spread_error:.1f} pts | Total error: {total_error:.1f} pts")
        print()

    return {
        'games': len(merged),
        'winner_accuracy': winner_accuracy,
        'spread_mae': spread_mae,
        'spread_within_3': spread_within_3,
        'spread_within_7': spread_within_7,
        'total_mae': total_mae,
        'total_within_3': total_within_3,
        'total_within_7': total_within_7,
    }


def analyze_betting_performance(predictions_df, games_df):
    """Analyze how we would have done betting these games."""

    print(f"\n{'='*80}")
    print(f"STEP 3: BETTING PERFORMANCE")
    print(f"{'='*80}\n")

    merged = predictions_df.merge(
        games_df[['game_id', 'away_team', 'home_team', 'away_score', 'home_score',
                  'spread_line', 'away_spread_odds', 'home_spread_odds']],
        on=['game_id', 'away_team', 'home_team']
    )

    merged['actual_winner'] = merged.apply(
        lambda x: x['home_team'] if x['home_score'] > x['away_score'] else x['away_team'],
        axis=1
    )

    # Simulate betting $100 on each predicted winner straight up (moneyline)
    total_bet = len(merged) * 100
    wins = (merged['predicted_winner'] == merged['actual_winner']).sum()
    losses = len(merged) - wins

    # Assuming -110 odds for simplicity
    winnings = wins * 190  # Win $90 profit per winner
    net_profit = winnings - total_bet
    roi = net_profit / total_bet

    print(f"💰 MONEYLINE BETTING (Betting predicted winner):")
    print(f"   Total bets: {len(merged)} @ $100 each")
    print(f"   Total staked: ${total_bet:,}")
    print(f"   Winners: {wins}")
    print(f"   Losers: {losses}")
    print(f"   Net profit: ${net_profit:+,}")
    print(f"   ROI: {roi:+.1%}")
    print()

    # Check spread betting
    # Did the favorite cover?
    merged['actual_spread'] = merged['home_score'] - merged['away_score']
    merged['spread_cover'] = merged['actual_spread'] < merged['spread_line']  # away covered if true

    # If we bet spread (fade the favorite), how would we do?
    # Let's say we bet AGAINST the spread when it's >= 7 points (fade big favorites)
    big_spreads = merged[abs(merged['spread_line']) >= 7].copy()

    if len(big_spreads) > 0:
        print(f"📊 FADING BIG FAVORITES (spread >= 7):")
        print(f"   Games with big spreads: {len(big_spreads)}")

        for idx, game in big_spreads.iterrows():
            away = game['away_team']
            home = game['home_team']
            spread = game['spread_line']
            actual_spread = game['actual_spread']

            # If spread is negative (home favored), did away cover?
            if spread < 0:
                covered = actual_spread > spread  # away covered
                bet_on = away
            else:
                covered = actual_spread < spread  # home covered
                bet_on = home

            status = "✅ WIN" if covered else "❌ LOSS"
            print(f"   {away} @ {home} (line: {spread:+.1f}) - Bet on {bet_on}: {status}")
            print(f"     Actual margin: {actual_spread:+.1f}")

        print()

    return {
        'moneyline_roi': roi,
        'moneyline_wins': wins,
        'moneyline_losses': losses,
        'net_profit': net_profit,
    }


def run_real_backtest():
    """Run complete real backtest."""

    # Load real games
    games = load_week10_games()

    if len(games) == 0:
        print("❌ No games found")
        return

    # Make predictions (blind to results)
    predictions = make_predictions_blind(games)

    # Compare to reality
    accuracy_metrics = compare_to_reality(predictions, games)

    # Analyze betting performance
    betting_metrics = analyze_betting_performance(predictions, games)

    # Summary
    print(f"\n{'='*80}")
    print(f"📊 BACKTEST SUMMARY")
    print(f"{'='*80}\n")

    print(f"✅ Used REAL game data from November 9, 2025")
    print(f"   - Made predictions using ONLY pre-game Vegas lines")
    print(f"   - Compared to ACTUAL results")
    print()

    print(f"🎯 Key Metrics:")
    print(f"   Winner accuracy: {accuracy_metrics['winner_accuracy']:.1%}")
    print(f"   Spread MAE: {accuracy_metrics['spread_mae']:.2f} points")
    print(f"   Total MAE: {accuracy_metrics['total_mae']:.2f} points")
    print()

    print(f"💰 Betting Results:")
    print(f"   Moneyline ROI: {betting_metrics['moneyline_roi']:+.1%}")
    print(f"   Net profit: ${betting_metrics['net_profit']:+,}")
    print()

    print(f"💡 This demonstrates REAL backtesting:")
    print(f"   - Actual NFL games that already happened")
    print(f"   - Predictions made without seeing results")
    print(f"   - Direct comparison to reality")
    print()

    # Save report
    outputs_dir = Path('outputs')
    outputs_dir.mkdir(parents=True, exist_ok=True)

    report_file = outputs_dir / 'backtest_real_nov9_2025.json'

    def convert_types(obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj

    report = {
        'date': '2025-11-09',
        'week': 10,
        'games': convert_types(games[['game_id', 'away_team', 'home_team', 'away_score',
                                      'home_score', 'spread_line', 'total_line']].to_dict(orient='records')),
        'predictions': convert_types(predictions.to_dict(orient='records')),
        'accuracy_metrics': convert_types(accuracy_metrics),
        'betting_metrics': convert_types(betting_metrics),
    }

    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"📄 Full report saved to: {report_file}")
    print()


if __name__ == '__main__':
    run_real_backtest()
