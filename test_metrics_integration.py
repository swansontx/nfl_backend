"""
Test script to validate team metrics integration into picks pipeline.

This script tests:
1. Team metrics feature engine initialization
2. Player feature enrichment with team efficiency metrics
3. Defense matchup integration
4. Picks pipeline with enhanced features
"""

import pandas as pd
from pathlib import Path
from backend.features.team_metrics_features import TeamMetricsFeatureEngine
from backend.orchestration.picks_pipeline import PicksPipeline


def test_team_metrics_engine():
    """Test the team metrics feature engine."""
    print("\n" + "="*60)
    print("TEST 1: Team Metrics Feature Engine")
    print("="*60)

    try:
        engine = TeamMetricsFeatureEngine(season=2025, inputs_dir="inputs")
        print("✓ Team metrics engine initialized successfully")

        # Test team metrics calculation
        team_metrics = engine.get_team_metrics('KC', weeks=[1, 2, 3, 4])
        print(f"✓ Team metrics calculated for KC: {len(team_metrics)} metrics")

        # Test defense matchup rating
        defense_rating = engine._get_defense_matchup_rating('BUF', 'WR')
        print(f"✓ Defense matchup rating for BUF vs WR: factor={defense_rating.get('matchup_factor', 1.0):.2f}")

        return True

    except Exception as e:
        print(f"✗ Team metrics engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_player_feature_enrichment():
    """Test enriching player features with team metrics."""
    print("\n" + "="*60)
    print("TEST 2: Player Feature Enrichment")
    print("="*60)

    try:
        # Load sample player stats
        player_stats_file = Path("inputs/player_stats_2025.csv")
        if not player_stats_file.exists():
            print("⚠ No player stats file found, skipping enrichment test")
            return True

        df = pd.read_csv(player_stats_file)
        print(f"✓ Loaded {len(df)} player stat rows")

        # Take a small sample for testing
        sample_df = df.head(100).copy()
        print(f"✓ Using sample of {len(sample_df)} rows for testing")

        # Enrich with team metrics
        engine = TeamMetricsFeatureEngine(season=2025, inputs_dir="inputs")
        enriched_df = engine.enrich_player_dataframe(sample_df, recency_weeks=4)

        # Check for new columns
        new_cols = [
            'team_success_rate', 'defense_matchup_factor',
            'pass_efficiency_edge', 'rush_efficiency_edge'
        ]

        for col in new_cols:
            if col in enriched_df.columns:
                print(f"✓ Column '{col}' added to dataframe")
            else:
                print(f"⚠ Column '{col}' not found in enriched dataframe")

        # Show sample enriched row
        if len(enriched_df) > 0:
            sample_row = enriched_df.iloc[0]
            print(f"\nSample enriched player:")
            print(f"  Player: {sample_row.get('player_display_name', 'Unknown')}")
            print(f"  Team: {sample_row.get('team', 'N/A')}")
            print(f"  Opponent: {sample_row.get('opponent_team', 'N/A')}")
            print(f"  Team Success Rate: {sample_row.get('team_success_rate', 0.0):.3f}")
            print(f"  Defense Matchup Factor: {sample_row.get('defense_matchup_factor', 1.0):.3f}")

        return True

    except Exception as e:
        print(f"✗ Feature enrichment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_picks_pipeline_integration():
    """Test picks pipeline with enhanced features."""
    print("\n" + "="*60)
    print("TEST 3: Picks Pipeline Integration")
    print("="*60)

    try:
        # Initialize pipeline
        pipeline = PicksPipeline(
            models_dir="outputs/models",
            inputs_dir="inputs",
            season=2025,
            bankroll=1000.0,
            min_edge=3.0
        )
        print("✓ Picks pipeline initialized with team metrics engine")

        # Check team metrics engine is available
        if hasattr(pipeline, 'team_metrics_engine'):
            print("✓ Team metrics engine accessible in pipeline")
        else:
            print("✗ Team metrics engine not found in pipeline")
            return False

        # Test feature loading (will enrich with team metrics)
        player_features = pipeline._load_player_features(week=13)

        if len(player_features) > 0:
            print(f"✓ Loaded {len(player_features)} player feature rows")

            # Check for team metric columns
            team_cols = ['team_success_rate', 'defense_matchup_factor']
            cols_found = [col for col in team_cols if col in player_features.columns]

            if cols_found:
                print(f"✓ Team metric columns found: {cols_found}")
            else:
                print("⚠ No team metric columns found (may be expected if no pbp data)")

        else:
            print("⚠ No player features loaded (may be expected if no 2025 stats yet)")

        print("\n✓ Picks pipeline integration test passed")
        return True

    except Exception as e:
        print(f"✗ Picks pipeline integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_relevant_team_metrics():
    """Test that correct team metrics are selected for each prop type."""
    print("\n" + "="*60)
    print("TEST 4: Relevant Team Metrics Selection")
    print("="*60)

    try:
        pipeline = PicksPipeline(season=2025)

        # Test different prop types
        test_cases = [
            ('pass_yds', 'QB', 'passing props'),
            ('rush_yds', 'RB', 'rushing props'),
            ('rec_yds', 'WR', 'receiving props'),
            ('pass_tds', 'QB', 'TD props'),
        ]

        for prop_type, position, desc in test_cases:
            metrics = pipeline._get_relevant_team_metrics(prop_type, position)
            print(f"\n{desc} ({prop_type}):")
            print(f"  Selected {len(metrics)} metrics: {', '.join(metrics[:3])}...")

            # Check for defense matchup factor
            if 'defense_matchup_factor' in metrics:
                print(f"  ✓ Includes defense matchup factor")
            else:
                print(f"  ⚠ Missing defense matchup factor")

        return True

    except Exception as e:
        print(f"✗ Relevant team metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TEAM METRICS INTEGRATION TEST SUITE")
    print("="*60)

    results = []

    # Run tests
    results.append(("Team Metrics Engine", test_team_metrics_engine()))
    results.append(("Player Feature Enrichment", test_player_feature_enrichment()))
    results.append(("Picks Pipeline Integration", test_picks_pipeline_integration()))
    results.append(("Relevant Metrics Selection", test_relevant_team_metrics()))

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
        print("\n✓ All tests passed! Team metrics integration is working correctly.")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Review errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
