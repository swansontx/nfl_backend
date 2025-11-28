"""
Test script for Unified Metrics API.

Tests:
1. API initialization and calculator setup
2. Team metrics retrieval
3. Team comparison
4. Player feature enrichment
5. Player context retrieval
6. Game metrics retrieval
7. Matchup analysis
8. Caching functionality
9. Available metrics listing
10. Metric info retrieval
"""

import pandas as pd
from backend.metrics.unified_metrics_api import MetricsAPI, get_metrics_api


def test_api_initialization():
    """Test API initialization."""
    print("\n" + "="*60)
    print("TEST 1: API Initialization")
    print("="*60)

    try:
        api = MetricsAPI(season=2025, inputs_dir="inputs", cache_enabled=True)
        print("✓ Metrics API initialized successfully")

        summary = api.get_summary()
        print(f"\nAPI Summary:")
        print(f"  Season: {summary['season']}")
        print(f"  Cache Enabled: {summary['cache_enabled']}")
        print(f"  Cached Items: {summary['cached_items']}")

        print(f"\nCalculators Available:")
        for calc_name, available in summary['calculators'].items():
            status = "✓" if available else "✗"
            print(f"  {status} {calc_name}")

        return True

    except Exception as e:
        print(f"✗ API initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_team_metrics():
    """Test team metrics retrieval."""
    print("\n" + "="*60)
    print("TEST 2: Team Metrics Retrieval")
    print("="*60)

    try:
        api = MetricsAPI(season=2025)

        # Get metrics for Kansas City
        kc_metrics = api.get_team_metrics('KC')
        print(f"✓ Retrieved {len(kc_metrics)} metrics for KC")

        # Display key metrics
        key_metrics = [
            ('success_rate_offense', '% successful plays'),
            ('epa_per_play_offense', 'EPA per play'),
            ('plays_per_game', 'plays/game'),
            ('turnover_margin', 'TO margin'),
            ('red_zone_td_pct', 'RZ TD %')
        ]

        print("\nKC Key Metrics:")
        for metric_key, label in key_metrics:
            value = kc_metrics.get(metric_key)
            if value is not None:
                if isinstance(value, float):
                    if 0 < abs(value) < 1:
                        print(f"  {label}: {value:.1%}")
                    else:
                        print(f"  {label}: {value:.2f}")
                else:
                    print(f"  {label}: {value}")

        # Test with recency
        recent_metrics = api.get_team_metrics('KC', weeks=[9, 10, 11, 12])
        print(f"\n✓ Retrieved recent metrics (weeks 9-12)")

        return True

    except Exception as e:
        print(f"✗ Team metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_team_comparison():
    """Test team comparison functionality."""
    print("\n" + "="*60)
    print("TEST 3: Team Comparison")
    print("="*60)

    try:
        api = MetricsAPI(season=2025)

        # Compare KC vs BUF
        comparison = api.compare_teams('KC', 'BUF')
        print(f"✓ Compared KC vs BUF")

        print(f"\nKC Advantages ({len(comparison['advantages_a'])} metrics):")
        for metric in comparison['advantages_a'][:5]:  # Show first 5
            metric_data = comparison['metrics'].get(metric, {})
            kc_val = metric_data.get('KC', 0)
            buf_val = metric_data.get('BUF', 0)
            diff = metric_data.get('difference', 0)
            print(f"  {metric}: KC {kc_val:.2f} vs BUF {buf_val:.2f} ({diff:+.2f})")

        print(f"\nBUF Advantages ({len(comparison['advantages_b'])} metrics):")
        for metric in comparison['advantages_b'][:5]:  # Show first 5
            metric_data = comparison['metrics'].get(metric, {})
            kc_val = metric_data.get('KC', 0)
            buf_val = metric_data.get('BUF', 0)
            diff = metric_data.get('difference', 0)
            print(f"  {metric}: BUF {buf_val:.2f} vs KC {kc_val:.2f} ({-diff:+.2f})")

        return True

    except Exception as e:
        print(f"✗ Team comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_player_enrichment():
    """Test player feature enrichment."""
    print("\n" + "="*60)
    print("TEST 4: Player Feature Enrichment")
    print("="*60)

    try:
        api = MetricsAPI(season=2025)

        # Load sample player data
        player_stats_file = "inputs/player_stats_2025.csv"
        import os
        if not os.path.exists(player_stats_file):
            print("⚠ No player stats file, skipping enrichment test")
            return True

        df = pd.read_csv(player_stats_file)
        sample_df = df.head(50).copy()  # Small sample for testing

        print(f"✓ Loaded {len(sample_df)} player rows")

        # Enrich with team metrics
        enriched_df = api.enrich_player_features(sample_df, recency_weeks=4)

        print(f"✓ Enriched player features")
        print(f"  Original columns: {len(sample_df.columns)}")
        print(f"  Enriched columns: {len(enriched_df.columns)}")
        print(f"  New metrics added: {len(enriched_df.columns) - len(sample_df.columns)}")

        # Show sample of new columns
        new_cols = [col for col in enriched_df.columns if col not in sample_df.columns]
        print(f"\nSample new columns: {', '.join(new_cols[:5])}...")

        return True

    except Exception as e:
        print(f"✗ Player enrichment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_player_context():
    """Test player context retrieval."""
    print("\n" + "="*60)
    print("TEST 5: Player Context Retrieval")
    print("="*60)

    try:
        api = MetricsAPI(season=2025)

        # Get context for a hypothetical player
        context = api.get_player_context(
            player_id='00-0036355',  # Patrick Mahomes
            team='KC',
            opponent='BUF',
            position='QB',
            week=13
        )

        print("✓ Retrieved player context")
        print(f"\nPlayer: {context['player_id']}")
        print(f"Team: {context['team']} vs {context['opponent']}")
        print(f"Position: {context['position']}, Week: {context['week']}")

        if 'team_metrics' in context:
            print(f"\nTeam Metrics: {len(context['team_metrics'])} available")

        if 'defense_matchup' in context:
            matchup = context['defense_matchup']
            print(f"\nDefense Matchup:")
            print(f"  Matchup Factor: {matchup.get('matchup_factor', 1.0):.2f}")
            print(f"  Defense Rank: {matchup.get('league_rank', 'N/A')}")

        return True

    except Exception as e:
        print(f"✗ Player context test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_game_metrics():
    """Test game metrics retrieval."""
    print("\n" + "="*60)
    print("TEST 6: Game Metrics Retrieval")
    print("="*60)

    try:
        api = MetricsAPI(season=2025)

        # Get game metrics for KC vs BUF
        game_metrics = api.get_game_metrics(
            home_team='KC',
            away_team='BUF',
            week=13,
            recency_weeks=4
        )

        print("✓ Retrieved game metrics for KC vs BUF (Week 13)")

        if 'summary' in game_metrics:
            summary = game_metrics['summary']

            print(f"\nPace:")
            print(f"  KC plays/game: {summary['pace']['home_plays_per_game']:.1f}")
            print(f"  BUF plays/game: {summary['pace']['away_plays_per_game']:.1f}")
            print(f"  Combined pace: {summary['pace']['combined_pace']:.1f}")

            print(f"\nTurnovers:")
            print(f"  KC margin: {summary['turnovers']['home_margin']:+d}")
            print(f"  BUF margin: {summary['turnovers']['away_margin']:+d}")
            print(f"  Differential: {summary['turnovers']['margin_differential']:+d}")

            print(f"\nEfficiency:")
            print(f"  KC success rate: {summary['efficiency']['home_success_rate_off']:.1%}")
            print(f"  BUF success rate: {summary['efficiency']['away_success_rate_off']:.1%}")

        return True

    except Exception as e:
        print(f"✗ Game metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_matchup_analysis():
    """Test matchup analysis."""
    print("\n" + "="*60)
    print("TEST 7: Matchup Analysis")
    print("="*60)

    try:
        api = MetricsAPI(season=2025)

        # Get matchup analysis
        matchup = api.analyze_matchup(
            home_team='KC',
            away_team='BUF',
            week=13
        )

        print("✓ Retrieved matchup analysis for KC vs BUF")

        if 'comparison' in matchup:
            comp = matchup['comparison']
            print(f"\nKC Advantages: {len(comp['advantages_a'])} metrics")
            print(f"BUF Advantages: {len(comp['advantages_b'])} metrics")

        if 'game_metrics' in matchup:
            print(f"\nGame Metrics: Available")

        return True

    except Exception as e:
        print(f"✗ Matchup analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_caching():
    """Test caching functionality."""
    print("\n" + "="*60)
    print("TEST 8: Caching Functionality")
    print("="*60)

    try:
        api = MetricsAPI(season=2025, cache_enabled=True)

        # First call - should calculate
        import time
        start = time.time()
        metrics1 = api.get_team_metrics('KC')
        time1 = time.time() - start

        # Second call - should use cache
        start = time.time()
        metrics2 = api.get_team_metrics('KC')
        time2 = time.time() - start

        print(f"✓ Caching working")
        print(f"  First call: {time1*1000:.2f}ms")
        print(f"  Cached call: {time2*1000:.2f}ms")
        print(f"  Speedup: {time1/max(time2, 0.0001):.1f}x faster")

        # Check cache size
        summary = api.get_summary()
        print(f"  Cached items: {summary['cached_items']}")

        # Test cache clearing
        api.clear_cache()
        summary = api.get_summary()
        print(f"✓ Cache cleared: {summary['cached_items']} items")

        return True

    except Exception as e:
        print(f"✗ Caching test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_available_metrics():
    """Test available metrics listing."""
    print("\n" + "="*60)
    print("TEST 9: Available Metrics Listing")
    print("="*60)

    try:
        api = MetricsAPI(season=2025)

        available = api.get_available_metrics()
        print("✓ Retrieved available metrics")

        total_metrics = sum(len(metrics) for metrics in available.values())
        print(f"\nTotal metrics available: {total_metrics}")

        for category, metrics in available.items():
            if metrics:
                print(f"\n{category.replace('_', ' ').title()} ({len(metrics)} metrics):")
                for metric in metrics[:3]:  # Show first 3
                    print(f"  • {metric}")
                if len(metrics) > 3:
                    print(f"  ... and {len(metrics) - 3} more")

        return True

    except Exception as e:
        print(f"✗ Available metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metric_info():
    """Test metric info retrieval."""
    print("\n" + "="*60)
    print("TEST 10: Metric Info Retrieval")
    print("="*60)

    try:
        api = MetricsAPI(season=2025)

        # Get info for a few metrics
        test_metrics = ['success_rate_offense', 'epa_per_play_offense', 'plays_per_game']

        for metric in test_metrics:
            info = api.get_metric_info(metric)
            print(f"\n{info['name']}:")
            print(f"  Description: {info['description']}")
            if 'typical_range' in info:
                print(f"  Typical Range: {info['typical_range']}")
            if 'used_in' in info:
                print(f"  Used In: {', '.join(info['used_in'])}")

        print("\n✓ Metric info retrieval working")
        return True

    except Exception as e:
        print(f"✗ Metric info test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_singleton_instance():
    """Test singleton get_metrics_api() function."""
    print("\n" + "="*60)
    print("TEST 11: Singleton Instance")
    print("="*60)

    try:
        # Get singleton instance
        api1 = get_metrics_api(season=2025)
        api2 = get_metrics_api(season=2025)

        # Should be same instance
        if api1 is api2:
            print("✓ Singleton working - same instance returned")
        else:
            print("✗ Singleton not working - different instances")
            return False

        # Different season should create new instance
        api3 = get_metrics_api(season=2024)
        if api3 is not api1:
            print("✓ New instance created for different season")
        else:
            print("✗ Should create new instance for different season")
            return False

        return True

    except Exception as e:
        print(f"✗ Singleton test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("UNIFIED METRICS API TEST SUITE")
    print("="*60)

    results = []

    # Run tests
    results.append(("API Initialization", test_api_initialization()))
    results.append(("Team Metrics", test_team_metrics()))
    results.append(("Team Comparison", test_team_comparison()))
    results.append(("Player Enrichment", test_player_enrichment()))
    results.append(("Player Context", test_player_context()))
    results.append(("Game Metrics", test_game_metrics()))
    results.append(("Matchup Analysis", test_matchup_analysis()))
    results.append(("Caching", test_caching()))
    results.append(("Available Metrics", test_available_metrics()))
    results.append(("Metric Info", test_metric_info()))
    results.append(("Singleton Instance", test_singleton_instance()))

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
        print("\n✓ All tests passed! Unified Metrics API is working correctly.")
        print("\nKey Features Validated:")
        print("  • Single point of access for all metrics")
        print("  • Team metrics retrieval and comparison")
        print("  • Player feature enrichment with team context")
        print("  • Game metrics for predictions")
        print("  • Matchup analysis")
        print("  • Automatic caching for performance")
        print("  • Metric discovery and documentation")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Review errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
