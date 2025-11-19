"""Player play-by-play feature extraction with ADVANCED METRICS

Extracts player-level statistics from nflverse play-by-play data for feature engineering.

Features extracted (BASIC):
- Passing: yards, TDs, completions, attempts, air yards, YAC
- Rushing: yards, TDs, attempts, success rate
- Receiving: yards, TDs, receptions, targets, air yards, YAC

Features extracted (ADVANCED METRICS):
- EPA (Expected Points Added): qb_epa, rushing_epa, total_epa
- CPOE (Completion % Over Expected): per QB
- Success Rate: plays with EPA > 0
- Air Yards & YAC: separation of passing value
- xYAC (Expected YAC): compared to actual YAC
- QB Pressure: hits, hurries, knockdowns
- WPA (Win Probability Added): clutch performance

Output: JSON file mapping player_id -> list of weekly features with advanced metrics
"""

from pathlib import Path
import csv
import json
from typing import Dict, List, Any
from collections import defaultdict


def extract_features(pbp_csv: Path, out_json: Path, player_stats_csv: Path = None) -> Dict[str, List[Dict]]:
    """Extract player features from play-by-play data.

    Args:
        pbp_csv: Path to nflverse play-by-play CSV
        out_json: Path to output JSON file
        player_stats_csv: Optional path to player_stats CSV for validation

    Returns:
        Dictionary mapping player_id -> list of weekly features

    Features are organized by player and week for time-series modeling.
    """
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # Check if input file exists and has data
    if not pbp_csv.exists() or pbp_csv.stat().st_size == 0:
        print(f"⚠ Play-by-play file not found or empty: {pbp_csv}")
        print(f"⚠ Creating empty feature file")
        out_json.write_text('{}')
        return {}

    print(f"\n{'='*60}")
    print(f"Extracting player features from play-by-play data")
    print(f"Input: {pbp_csv}")
    print(f"{'='*60}\n")

    # Initialize feature storage
    # Structure: player_id -> list of game features
    player_features = defaultdict(list)

    try:
        with open(pbp_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            # Track current game context
            current_game = None
            game_features = defaultdict(lambda: {
                # Basic stats
                'passing_yards': 0,
                'passing_tds': 0,
                'completions': 0,
                'attempts': 0,
                'interceptions': 0,
                'sacks': 0,
                'rushing_yards': 0,
                'rushing_tds': 0,
                'rushing_attempts': 0,
                'receptions': 0,
                'targets': 0,
                'receiving_yards': 0,
                'receiving_tds': 0,
                'air_yards': 0,
                'yards_after_catch': 0,
                # ADVANCED METRICS
                'total_epa': 0.0,
                'qb_epa': 0.0,
                'rushing_epa': 0.0,
                'receiving_epa': 0.0,
                'cpoe_sum': 0.0,
                'cpoe_count': 0,
                'success_plays': 0,
                'total_plays': 0,
                'wpa': 0.0,
                'air_epa': 0.0,
                'yac_epa': 0.0,
                'xyac_sum': 0.0,
                'xyac_count': 0,
                'qb_hits': 0,
                'qb_hurries': 0,
                'qb_pressures': 0,
            })

            row_count = 0
            for row in reader:
                row_count += 1

                # Track game changes
                game_id = row.get('game_id', row.get('old_game_id', ''))
                if current_game != game_id:
                    # Save previous game features
                    if current_game:
                        _save_game_features(player_features, game_features, current_game)

                    # Reset for new game
                    current_game = game_id
                    game_features = defaultdict(lambda: {
                        # Basic stats
                        'passing_yards': 0,
                        'passing_tds': 0,
                        'completions': 0,
                        'attempts': 0,
                        'interceptions': 0,
                        'sacks': 0,
                        'rushing_yards': 0,
                        'rushing_tds': 0,
                        'rushing_attempts': 0,
                        'receptions': 0,
                        'targets': 0,
                        'receiving_yards': 0,
                        'receiving_tds': 0,
                        'air_yards': 0,
                        'yards_after_catch': 0,
                        # ADVANCED METRICS
                        'total_epa': 0.0,
                        'qb_epa': 0.0,
                        'rushing_epa': 0.0,
                        'receiving_epa': 0.0,
                        'cpoe_sum': 0.0,
                        'cpoe_count': 0,
                        'success_plays': 0,
                        'total_plays': 0,
                        'wpa': 0.0,
                        'air_epa': 0.0,
                        'yac_epa': 0.0,
                        'xyac_sum': 0.0,
                        'xyac_count': 0,
                        'qb_hits': 0,
                        'qb_hurries': 0,
                        'qb_pressures': 0,
                    })

                # Extract play type
                play_type = row.get('play_type', '')

                # Process passing plays
                if play_type == 'pass':
                    passer_id = row.get('passer_player_id', row.get('passer_id', ''))
                    receiver_id = row.get('receiver_player_id', row.get('receiver_id', ''))

                    if passer_id:
                        game_features[passer_id]['attempts'] += 1
                        game_features[passer_id]['total_plays'] += 1

                        # BASIC STATS
                        if row.get('complete_pass') == '1' or row.get('complete_pass') == 'True':
                            game_features[passer_id]['completions'] += 1
                            yards = float(row.get('yards_gained', row.get('passing_yards', 0)) or 0)
                            game_features[passer_id]['passing_yards'] += yards

                        if row.get('pass_touchdown') == '1' or row.get('pass_touchdown') == 'True':
                            game_features[passer_id]['passing_tds'] += 1

                        if row.get('interception') == '1' or row.get('interception') == 'True':
                            game_features[passer_id]['interceptions'] += 1

                        # ADVANCED METRICS - QB
                        # EPA metrics
                        qb_epa = float(row.get('qb_epa', 0) or 0)
                        total_epa = float(row.get('epa', 0) or 0)
                        game_features[passer_id]['qb_epa'] += qb_epa
                        game_features[passer_id]['total_epa'] += total_epa

                        # CPOE (Completion Percentage Over Expected)
                        cpoe = row.get('cpoe', '')
                        if cpoe and cpoe != '' and cpoe != 'NA':
                            try:
                                game_features[passer_id]['cpoe_sum'] += float(cpoe)
                                game_features[passer_id]['cpoe_count'] += 1
                            except (ValueError, TypeError):
                                pass

                        # Success rate (EPA > 0)
                        if total_epa > 0:
                            game_features[passer_id]['success_plays'] += 1

                        # WPA (Win Probability Added)
                        wpa = float(row.get('wpa', 0) or 0)
                        game_features[passer_id]['wpa'] += wpa

                        # Air EPA and YAC EPA
                        air_epa = float(row.get('air_epa', 0) or 0)
                        yac_epa = float(row.get('yac_epa', 0) or 0)
                        game_features[passer_id]['air_epa'] += air_epa
                        game_features[passer_id]['yac_epa'] += yac_epa

                        # QB Pressure
                        if row.get('qb_hit') == '1':
                            game_features[passer_id]['qb_hits'] += 1
                        if row.get('qb_hurry') == '1':
                            game_features[passer_id]['qb_hurries'] += 1
                        # qb_pressures = hits + hurries + sacks
                        if row.get('sack') == '1':
                            game_features[passer_id]['sacks'] += 1
                            game_features[passer_id]['qb_pressures'] += 1
                        if row.get('qb_hit') == '1' or row.get('qb_hurry') == '1':
                            game_features[passer_id]['qb_pressures'] += 1

                    if receiver_id:
                        game_features[receiver_id]['targets'] += 1
                        game_features[receiver_id]['total_plays'] += 1

                        # BASIC STATS
                        if row.get('complete_pass') == '1' or row.get('complete_pass') == 'True':
                            game_features[receiver_id]['receptions'] += 1
                            yards = float(row.get('yards_gained', row.get('receiving_yards', 0)) or 0)
                            game_features[receiver_id]['receiving_yards'] += yards

                            # Air yards and YAC
                            air_yards = float(row.get('air_yards', 0) or 0)
                            game_features[receiver_id]['air_yards'] += air_yards
                            game_features[receiver_id]['yards_after_catch'] += (yards - air_yards)

                        if row.get('pass_touchdown') == '1' or row.get('pass_touchdown') == 'True':
                            game_features[receiver_id]['receiving_tds'] += 1

                        # ADVANCED METRICS - RECEIVER
                        # Receiving EPA
                        total_epa = float(row.get('epa', 0) or 0)
                        game_features[receiver_id]['receiving_epa'] += total_epa
                        game_features[receiver_id]['total_epa'] += total_epa

                        # Success rate
                        if total_epa > 0:
                            game_features[receiver_id]['success_plays'] += 1

                        # WPA
                        wpa = float(row.get('wpa', 0) or 0)
                        game_features[receiver_id]['wpa'] += wpa

                        # Expected YAC (xYAC)
                        xyac_mean = row.get('xyac_mean_yardage', '')
                        if xyac_mean and xyac_mean != '' and xyac_mean != 'NA':
                            try:
                                game_features[receiver_id]['xyac_sum'] += float(xyac_mean)
                                game_features[receiver_id]['xyac_count'] += 1
                            except (ValueError, TypeError):
                                pass

                        # Air EPA and YAC EPA
                        air_epa = float(row.get('air_epa', 0) or 0)
                        yac_epa = float(row.get('yac_epa', 0) or 0)
                        game_features[receiver_id]['air_epa'] += air_epa
                        game_features[receiver_id]['yac_epa'] += yac_epa

                # Process rushing plays
                elif play_type == 'run':
                    rusher_id = row.get('rusher_player_id', row.get('rusher_id', ''))

                    if rusher_id:
                        game_features[rusher_id]['rushing_attempts'] += 1
                        game_features[rusher_id]['total_plays'] += 1

                        # BASIC STATS
                        yards = float(row.get('yards_gained', row.get('rushing_yards', 0)) or 0)
                        game_features[rusher_id]['rushing_yards'] += yards

                        if row.get('rush_touchdown') == '1' or row.get('rush_touchdown') == 'True':
                            game_features[rusher_id]['rushing_tds'] += 1

                        # ADVANCED METRICS - RUSHING
                        # Rushing EPA
                        total_epa = float(row.get('epa', 0) or 0)
                        game_features[rusher_id]['rushing_epa'] += total_epa
                        game_features[rusher_id]['total_epa'] += total_epa

                        # Success rate (EPA > 0)
                        if total_epa > 0:
                            game_features[rusher_id]['success_plays'] += 1

                        # WPA (Win Probability Added)
                        wpa = float(row.get('wpa', 0) or 0)
                        game_features[rusher_id]['wpa'] += wpa

                # Print progress every 10k rows
                if row_count % 10000 == 0:
                    print(f"  Processed {row_count:,} plays...")

            # Save final game
            if current_game:
                _save_game_features(player_features, game_features, current_game)

        print(f"\n✓ Processed {row_count:,} plays")
        print(f"✓ Extracted features for {len(player_features)} players")

        # Convert defaultdict to regular dict for JSON serialization
        player_features_dict = {k: v for k, v in player_features.items()}

        # Write to JSON
        with open(out_json, 'w') as f:
            json.dump(player_features_dict, f, indent=2)

        print(f"✓ Wrote features to {out_json}")

        # Print sample stats
        if player_features_dict:
            sample_player = list(player_features_dict.keys())[0]
            sample_games = len(player_features_dict[sample_player])
            print(f"\nSample: Player {sample_player} has {sample_games} games of features")

        return player_features_dict

    except Exception as e:
        print(f"✗ Error extracting features: {e}")
        import traceback
        traceback.print_exc()
        out_json.write_text('{}')
        return {}


def _save_game_features(player_features: Dict, game_features: Dict, game_id: str) -> None:
    """Save accumulated game features to player feature dict.

    Args:
        player_features: Main player features dictionary
        game_features: Current game's accumulated features
        game_id: Game identifier
    """
    # Extract season and week from game_id if possible
    # Format: YYYY_WW_AWAY_HOME or old format
    try:
        parts = game_id.split('_')
        if len(parts) >= 2:
            season = parts[0]
            week = parts[1]
        else:
            # Fallback for old game_id format
            season = game_id[:4] if len(game_id) >= 4 else 'unknown'
            week = 'unknown'
    except:
        season = 'unknown'
        week = 'unknown'

    for player_id, features in game_features.items():
        if not player_id or player_id == '':
            continue

        # Add game metadata to features
        features['game_id'] = game_id
        features['season'] = season
        features['week'] = week

        player_features[player_id].append(features)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Extract player features from play-by-play data')
    p.add_argument('--pbp', type=Path,
                   default=Path('inputs/play_by_play_2024.csv'),
                   help='Path to play-by-play CSV')
    p.add_argument('--out', type=Path,
                   default=Path('outputs/player_pbp_features_by_id.json'),
                   help='Path to output JSON file')
    p.add_argument('--player-stats', type=Path,
                   default=None,
                   help='Optional player stats CSV for validation')
    args = p.parse_args()

    extract_features(args.pbp, args.out, args.player_stats)
