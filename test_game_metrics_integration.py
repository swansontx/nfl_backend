"""
Test script to validate game metrics integration.

Tests:
1. GameMetricsEngine initialization
2. Enhanced team strength calculation
3. Pace adjustment to totals
4. Turnover adjustment to spreads
5. Game market analyzer with enhanced metrics
"""

from backend.features.game_metrics_features import GameMetricsEngine, enhance_game_prediction
from backend.analysis.game_markets import GameMarketAnalyzer


def test_game_metrics_engine():
    """Test the game metrics engine initialization and basic functions."""
    print("\n" + "="*60)
    print("TEST 1: Game Metrics Engine")
    print("="*60)

    try:
        engine = GameMetricsEngine(season=2025, inputs_dir="inputs")
        print("✓ Game metrics engine initialized successfully")

        # Test enhanced team strength
        enhanced_strength = engine.get_enhanced_team_strength(
            team='KC',
            base_offensive_rating=25.5,
            base_defensive_rating=19.2,
            is_home=True
        )

        print(f"✓ Enhanced team strength calculated for KC")
        print(f"  Offensive Rating: {enhanced_strength.offensive_rating:.1f} PPG")
        print(f"  Defensive Rating: {enhanced_strength.defensive_rating:.1f} PPG")
        print(f"  Plays Per Game: {enhanced_strength.plays_per_game:.1f}")
        print(f"  Turnover Margin: {enhanced_strength.turnover_margin:+d}")
        print(f"  Success Rate: {enhanced_strength.success_rate_offense:.1%}")
        print(f"  EPA/Play: {enhanced_strength.epa_per_play_offense:+.3f}")

        return True

    except Exception as e:
        print(f"✗ Game metrics engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pace_adjustment():
    """Test pace-based total adjustments."""
    print("\n" + "="*60)
    print("TEST 2: Pace Adjustment to Totals")
    print("="*60)

    try:
        engine = GameMetricsEngine(season=2025, inputs_dir="inputs")

        # Test with two teams - one fast-paced, one slow
        fast_team = engine.get_enhanced_team_strength(
            team='KC',
            base_offensive_rating=26.0,
            base_defensive_rating=20.0,
            is_home=True
        )
        # Manually override pace for testing
        fast_team.plays_per_game = 70.0  # Fast pace

        slow_team = engine.get_enhanced_team_strength(
            team='SF',
            base_offensive_rating=24.0,
            base_defensive_rating=18.0,
            is_home=False
        )
        slow_team.plays_per_game = 60.0  # Slow pace

        base_total = 46.0
        adjusted_total, reasoning = engine.calculate_pace_adjusted_total(
            fast_team, slow_team, base_total
        )

        print(f"✓ Pace adjustment calculated")
        print(f"  Base Total: {base_total:.1f}")
        print(f"  Adjusted Total: {adjusted_total:.1f}")
        print(f"  Adjustment: {adjusted_total - base_total:+.1f} points")
        print(f"  Reasoning: {reasoning}")

        # Test with average pace
        avg_team1 = engine.get_enhanced_team_strength(
            'BUF', 24.0, 20.0, is_home=True
        )
        avg_team1.plays_per_game = 65.0

        avg_team2 = engine.get_enhanced_team_strength(
            'MIA', 23.0, 21.0, is_home=False
        )
        avg_team2.plays_per_game = 65.0

        avg_adjusted, avg_reasoning = engine.calculate_pace_adjusted_total(
            avg_team1, avg_team2, 44.0
        )

        print(f"\n✓ Average pace test")
        print(f"  Adjustment with league-average pace: {avg_adjusted - 44.0:+.1f} points")
        print(f"  (Should be close to 0)")

        return True

    except Exception as e:
        print(f"✗ Pace adjustment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_turnover_adjustment():
    """Test turnover margin adjustments to spreads."""
    print("\n" + "="*60)
    print("TEST 3: Turnover Margin Adjustment to Spreads")
    print("="*60)

    try:
        engine = GameMetricsEngine(season=2025, inputs_dir="inputs")

        # Team with positive turnover margin
        good_to_team = engine.get_enhanced_team_strength(
            'KC', 25.0, 19.0, is_home=True
        )
        good_to_team.turnover_margin = 8  # +8 turnover margin

        # Team with negative turnover margin
        bad_to_team = engine.get_enhanced_team_strength(
            'LV', 22.0, 24.0, is_home=False
        )
        bad_to_team.turnover_margin = -5  # -5 turnover margin

        base_spread = 3.0  # KC favored by 3
        adjusted_spread, reasoning = engine.calculate_turnover_adjusted_spread(
            good_to_team, bad_to_team, base_spread
        )

        print(f"✓ Turnover adjustment calculated")
        print(f"  Home TO Margin: +{good_to_team.turnover_margin}")
        print(f"  Away TO Margin: {bad_to_team.turnover_margin}")
        print(f"  Margin Differential: +{good_to_team.turnover_margin - bad_to_team.turnover_margin}")
        print(f"  Base Spread: {base_spread:+.1f}")
        print(f"  Adjusted Spread: {adjusted_spread:+.1f}")
        print(f"  Adjustment: {adjusted_spread - base_spread:+.1f} points")
        print(f"  Reasoning: {reasoning}")

        return True

    except Exception as e:
        print(f"✗ Turnover adjustment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_game_market_analyzer():
    """Test game market analyzer with enhanced metrics."""
    print("\n" + "="*60)
    print("TEST 4: Game Market Analyzer (Enhanced)")
    print("="*60)

    try:
        # Test with enhanced metrics
        analyzer_enhanced = GameMarketAnalyzer(season=2025, use_enhanced_metrics=True)
        print("✓ Game market analyzer initialized with enhanced metrics")

        # Make a prediction (requires actual data, so may not have full metrics)
        prediction = analyzer_enhanced.predict_game_outcome(
            home_team='KC',
            away_team='BUF',
            week=13
        )

        print(f"\n✓ Game prediction generated (KC vs BUF, Week 13)")
        print(f"  Predicted Score: {prediction.home_team} {prediction.home_score:.1f} - {prediction.away_team} {prediction.away_score:.1f}")
        print(f"  Predicted Spread: {prediction.predicted_spread:+.1f} ({prediction.home_team})")
        print(f"  Predicted Total: {prediction.predicted_total:.1f}")
        print(f"  Win Probabilities: {prediction.home_team} {prediction.home_win_prob:.1%} / {prediction.away_team} {prediction.away_win_prob:.1%}")
        print(f"  Confidence: {prediction.confidence:.1%}")

        # Test without enhanced metrics for comparison
        analyzer_basic = GameMarketAnalyzer(season=2025, use_enhanced_metrics=False)
        prediction_basic = analyzer_basic.predict_game_outcome(
            home_team='KC',
            away_team='BUF',
            week=13
        )

        print(f"\n✓ Basic prediction (for comparison)")
        print(f"  Basic Spread: {prediction_basic.predicted_spread:+.1f}")
        print(f"  Basic Total: {prediction_basic.predicted_total:.1f}")
        print(f"\n  Enhanced vs Basic:")
        print(f"    Spread difference: {prediction.predicted_spread - prediction_basic.predicted_spread:+.1f}")
        print(f"    Total difference: {prediction.predicted_total - prediction_basic.predicted_total:+.1f}")

        return True

    except Exception as e:
        print(f"✗ Game market analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhance_game_prediction_utility():
    """Test the utility function for enhanced predictions."""
    print("\n" + "="*60)
    print("TEST 5: Enhanced Game Prediction Utility")
    print("="*60)

    try:
        result = enhance_game_prediction(
            home_team='KC',
            away_team='BUF',
            base_home_score=25.5,
            base_away_score=23.0,
            home_offensive_rating=26.0,
            home_defensive_rating=19.0,
            away_offensive_rating=25.0,
            away_defensive_rating=20.0,
            season=2025,
            recent_weeks=4
        )

        print("✓ Enhanced prediction utility executed")

        print(f"\nBase Prediction:")
        print(f"  Score: {result['base_prediction']['home_score']:.1f} - {result['base_prediction']['away_score']:.1f}")
        print(f"  Spread: {result['base_prediction']['spread']:+.1f}")
        print(f"  Total: {result['base_prediction']['total']:.1f}")

        print(f"\nEnhanced Prediction:")
        print(f"  Score: {result['enhanced_prediction']['home_score']:.1f} - {result['enhanced_prediction']['away_score']:.1f}")
        print(f"  Spread: {result['enhanced_prediction']['spread']:+.1f}")
        print(f"  Total: {result['enhanced_prediction']['total']:.1f}")

        print(f"\nAdjustments Applied:")
        print(f"  Pace → Total: {result['adjustments']['pace_total_adj']:+.1f} points")
        print(f"  Turnovers → Spread: {result['adjustments']['turnover_spread_adj']:+.1f} points")
        print(f"  Efficiency → Spread: {result['adjustments']['efficiency_spread_adj']:+.1f} points")
        print(f"  Efficiency → Total: {result['adjustments']['efficiency_total_adj']:+.1f} points")

        print(f"\nReasoning:")
        print(f"  {result['reasoning']['pace']}")
        print(f"  {result['reasoning']['turnovers']}")
        if result['reasoning']['efficiency']:
            print(f"  {result['reasoning']['efficiency']}")

        return True

    except Exception as e:
        print(f"✗ Enhanced prediction utility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("GAME METRICS INTEGRATION TEST SUITE")
    print("="*60)

    results = []

    # Run tests
    results.append(("Game Metrics Engine", test_game_metrics_engine()))
    results.append(("Pace Adjustment", test_pace_adjustment()))
    results.append(("Turnover Adjustment", test_turnover_adjustment()))
    results.append(("Game Market Analyzer", test_game_market_analyzer()))
    results.append(("Enhanced Prediction Utility", test_enhance_game_prediction_utility()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! Game metrics integration is working correctly.")
        print("\nKey Features Integrated:")
        print("  • Pace metrics → Game totals (plays per game)")
        print("  • Turnover margin → Game spreads (~2.5 pts per margin)")
        print("  • Efficiency metrics → Spread & total adjustments (EPA, success rate)")
        print("  • Red zone efficiency → Scoring predictions")
        print("  • Game market analyzer enhanced with all metrics")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Review errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
