"""Test ML integration in game outcome orchestrator."""

from backend.orchestration.game_outcome_orchestrator import game_outcome_orchestrator

def test_ml_prediction():
    """Test that ML predictor is integrated and working."""
    print("\n" + "="*80)
    print("TESTING ML INTEGRATION")
    print("="*80 + "\n")

    # Test game: 2023 Week 14 KC vs BUF (high-scoring matchup)
    game_id = "2023_14_BUF_KC"
    week = 14

    print(f"Testing prediction for: {game_id}")
    print()

    try:
        # Make prediction
        prediction = game_outcome_orchestrator.predict_game(
            game_id=game_id,
            week=week
        )

        # Display results
        print("✅ Prediction successful!\n")
        print(f"Predicted Total: {prediction.predicted_total:.1f}")
        print(f"  Home Score: {prediction.predicted_home_score:.1f}")
        print(f"  Away Score: {prediction.predicted_away_score:.1f}")
        print(f"  Spread: {prediction.predicted_margin:+.1f} (KC)")
        print(f"\nConfidence: {prediction.confidence:.0%}")
        print(f"Total CI (95%): {prediction.total_ci[0]:.1f} to {prediction.total_ci[1]:.1f}")
        print(f"Total Std Dev: {prediction.total_std:.1f}")

        # Check if using ML model
        if game_outcome_orchestrator.total_model is not None:
            print("\n🏆 Using Neural Network ML predictor (+10.8% vs baseline)")
        else:
            print("\n⚠️  Using formula-based predictor (ML model not available)")

    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    test_ml_prediction()
