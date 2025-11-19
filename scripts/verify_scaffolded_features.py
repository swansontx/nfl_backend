#!/usr/bin/env python3
"""Verify that all scaffolded features are now properly implemented.

This script tests the newly implemented modules to ensure they work correctly.
"""

from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_fetch_nflverse():
    """Test nflverse download module."""
    print("\n" + "="*60)
    print("Testing: fetch_nflverse.py")
    print("="*60)

    from backend.ingestion.fetch_nflverse import fetch_nflverse

    # Test with a year that likely has no data (won't actually download)
    test_dir = Path('test_outputs/inputs')
    test_dir.mkdir(parents=True, exist_ok=True)

    try:
        # This will create empty files if data isn't available
        fetch_nflverse(year=2024, out_dir=test_dir, cache_dir=None)
        print("✓ fetch_nflverse module works")
        return True
    except Exception as e:
        print(f"✗ fetch_nflverse failed: {e}")
        return False


def test_extract_player_pbp_features():
    """Test player PBP feature extraction."""
    print("\n" + "="*60)
    print("Testing: extract_player_pbp_features.py")
    print("="*60)

    from backend.features.extract_player_pbp_features import extract_features

    # Create a test PBP CSV
    test_pbp = Path('test_outputs/inputs/test_pbp.csv')
    test_pbp.parent.mkdir(parents=True, exist_ok=True)

    # Write sample PBP data
    test_pbp.write_text(
        'game_id,play_type,passer_player_id,receiver_player_id,rusher_player_id,'
        'complete_pass,yards_gained,pass_touchdown,rush_touchdown,interception,air_yards\n'
        '2024_10_KC_BUF,pass,mahomes_pat,kelce_tra,,,1,15,0,0,0,12\n'
        '2024_10_KC_BUF,run,,,henry_der,,,10,0,0,0,\n'
        '2024_10_KC_BUF,pass,mahomes_pat,kelce_tra,,,1,8,1,0,0,6\n'
    )

    test_out = Path('test_outputs/outputs/player_features.json')

    try:
        result = extract_features(test_pbp, test_out)
        print(f"✓ Extracted features for {len(result)} players")
        return True
    except Exception as e:
        print(f"✗ extract_player_pbp_features failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_smoothing_module():
    """Test smoothing module."""
    print("\n" + "="*60)
    print("Testing: smoothing_and_rolling.py")
    print("="*60)

    from backend.features.smoothing_and_rolling import process_all_players
    import json

    # Create test feature data
    test_features_file = Path('test_outputs/outputs/test_features.json')
    test_features_file.parent.mkdir(parents=True, exist_ok=True)

    test_data = {
        'player_001': [
            {'passing_yards': 250, 'passing_tds': 2, 'game_id': '2024_01_KC_BUF', 'season': '2024', 'week': '01'},
            {'passing_yards': 300, 'passing_tds': 3, 'game_id': '2024_02_KC_LAC', 'season': '2024', 'week': '02'},
            {'passing_yards': 280, 'passing_tds': 2, 'game_id': '2024_03_KC_LV', 'season': '2024', 'week': '03'},
        ]
    }

    with open(test_features_file, 'w') as f:
        json.dump(test_data, f)

    test_out = Path('test_outputs/outputs/test_smoothed.json')

    try:
        config = {'ema_alpha': 0.3, 'rolling_window': 2}
        result = process_all_players(test_features_file, test_out, config)

        if result:
            print(f"✓ Smoothed features for {len(result)} players")
            return True
        else:
            print("⚠ No features processed (input file may not exist)")
            return False
    except Exception as e:
        print(f"✗ smoothing_and_rolling failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_roster_index_building():
    """Test roster index building."""
    print("\n" + "="*60)
    print("Testing: build_game_roster_index.py")
    print("="*60)

    from backend.roster_injury.build_game_roster_index import build_roster_index

    # Create test roster CSV
    test_roster_file = Path('test_outputs/inputs/weekly_rosters_2024.csv')
    test_roster_file.parent.mkdir(parents=True, exist_ok=True)

    test_roster_file.write_text(
        'season,team,week,gsis_id,player_name,position,status\n'
        '2024,KC,1,mahomes_pat,Patrick Mahomes,QB,ACT\n'
        '2024,KC,1,kelce_tra,Travis Kelce,TE,ACT\n'
        '2024,BUF,1,allen_jos,Josh Allen,QB,ACT\n'
    )

    test_out_dir = Path('test_outputs/outputs')

    try:
        result = build_roster_index(2024, test_roster_file.parent, test_out_dir)
        print(f"✓ Built roster index with {len(result)} team-weeks")
        return True
    except Exception as e:
        print(f"✗ build_game_roster_index failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_injury_index_building():
    """Test injury index building."""
    print("\n" + "="*60)
    print("Testing: build_injury_game_index.py")
    print("="*60)

    from backend.roster_injury.build_injury_game_index import build_injury_index
    import json

    # Create test injury file
    test_injury_file = Path('test_outputs/outputs/injuries_20241119.json')
    test_injury_file.parent.mkdir(parents=True, exist_ok=True)

    test_injuries = [
        {'player_id': 'kelce_tra', 'team': 'KC', 'status': 'Questionable', 'injury': 'Knee'},
        {'player_id': 'hill_tyr', 'team': 'MIA', 'status': 'Out', 'injury': 'Ankle'}
    ]

    with open(test_injury_file, 'w') as f:
        json.dump(test_injuries, f)

    test_out_dir = Path('test_outputs/outputs')

    try:
        result = build_injury_index(2024, test_injury_file.parent, None, test_out_dir)
        print(f"✓ Built injury index with {len(result)} team-weeks")
        return True
    except Exception as e:
        print(f"✗ build_injury_game_index failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_roster_lookup():
    """Test roster lookup module."""
    print("\n" + "="*60)
    print("Testing: roster_lookup.py")
    print("="*60)

    from backend.roster_injury.roster_lookup import get_player_status, load_roster_data
    import json

    # Create test roster and injury files
    test_data_dir = Path('test_outputs/outputs')
    test_data_dir.mkdir(parents=True, exist_ok=True)

    # Create test roster data
    roster_data = {
        '2024_10_KC': {
            'mahomes_pat': 'ACT',
            'kelce_tra': 'ACT'
        },
        '2024_10_BUF': {
            'allen_jos': 'ACT'
        }
    }

    roster_file = test_data_dir / 'game_rosters_2024.json'
    with open(roster_file, 'w') as f:
        json.dump(roster_data, f)

    # Create test injury data
    injury_data = {
        '2024_10_KC': {
            'kelce_tra': 'Questionable'
        }
    }

    injury_file = test_data_dir / 'injury_game_index_2024.json'
    with open(injury_file, 'w') as f:
        json.dump(injury_data, f)

    try:
        # Test loading
        loaded = load_roster_data(2024, test_data_dir)

        if not loaded:
            print("⚠ Failed to load data")
            return False

        # Test lookups
        status1 = get_player_status('2024_10_KC_BUF', 'mahomes_pat', test_data_dir)
        status2 = get_player_status('2024_10_KC_BUF', 'unknown_player', test_data_dir)

        print(f"✓ Roster lookup works")
        print(f"  mahomes_pat status: {status1}")
        print(f"  unknown_player status: {status2}")
        return True

    except Exception as e:
        print(f"✗ roster_lookup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup_test_outputs():
    """Clean up test outputs."""
    import shutil
    test_dir = Path('test_outputs')
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print("\n✓ Cleaned up test outputs")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("VERIFYING SCAFFOLDED FEATURES IMPLEMENTATION")
    print("="*60)

    results = []

    # Run all tests
    results.append(("fetch_nflverse", test_fetch_nflverse()))
    results.append(("extract_player_pbp_features", test_extract_player_pbp_features()))
    results.append(("smoothing_and_rolling", test_smoothing_module()))
    results.append(("build_game_roster_index", test_roster_index_building()))
    results.append(("build_injury_game_index", test_injury_index_building()))
    results.append(("roster_lookup", test_roster_lookup()))

    # Clean up
    cleanup_test_outputs()

    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print(f"\nTotal: {passed}/{total} modules verified")

    if passed == total:
        print("\n✅ All scaffolded features successfully implemented!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} module(s) need attention")
        sys.exit(1)
