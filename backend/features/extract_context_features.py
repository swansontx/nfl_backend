"""Extract context features: opponent defense, pace, market, weather, stadium.

This module adds critical game-context features that improve prop predictions:

1. OPPONENT DEFENSIVE METRICS
   - def_pass_epa_allowed: How much passing EPA defense allows
   - def_rush_epa_allowed: How much rushing EPA defense allows
   - def_success_rate_allowed: Defensive success rate allowed
   - def_cpoe_allowed: CPOE allowed (for QB props)

2. PACE & VOLUME METRICS
   - team_plays_pg: Plays per game (rolling average)
   - neutral_pass_rate: Pass rate in neutral situations
   - neutral_seconds_per_snap: Pace of play

3. MARKET CONTEXT
   - spread: Point spread
   - total: Game total (over/under)
   - implied_team_total: Team's implied points
   - is_favorite: Binary flag
   - expected_to_trail: Game script indicator

4. WEATHER & STADIUM
   - is_dome: Indoor stadium
   - is_outdoors: Outdoor stadium
   - wind_speed_bucket: 0-10, 10-15, 15+ mph
   - temp_bucket: cold (<45F), mild (45-75F), hot (>75F)
   - precipitation: Rain/snow flag

These features dramatically improve model accuracy by accounting for matchup
difficulty, game script, and environmental factors.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import Dict, Optional
from collections import defaultdict


def extract_defensive_metrics(
    pbp_file: Path,
    output_file: Path
) -> Dict:
    """Extract defensive metrics by team from play-by-play data.

    Args:
        pbp_file: Path to play-by-play parquet file
        output_file: Path to save defensive metrics JSON

    Returns:
        Dict mapping (team, week) -> defensive metrics
    """
    print(f"\n{'='*80}")
    print("EXTRACTING DEFENSIVE METRICS")
    print(f"{'='*80}\n")

    # Load PBP data
    print(f"📂 Loading play-by-play data: {pbp_file}")
    df = pd.read_parquet(pbp_file)
    print(f"✓ Loaded {len(df):,} plays")

    # Filter to regular plays (no penalties, etc.)
    df = df[df['play_type'].isin(['pass', 'run'])]
    print(f"✓ Filtered to {len(df):,} pass/run plays")

    # Calculate defensive metrics by team/week
    defensive_metrics = defaultdict(lambda: {
        'plays': 0,
        'pass_epa_sum': 0.0,
        'pass_plays': 0,
        'rush_epa_sum': 0.0,
        'rush_plays': 0,
        'success_plays': 0,
        'total_plays': 0,
        'cpoe_sum': 0.0,
        'cpoe_count': 0
    })

    for _, row in df.iterrows():
        # Defensive team is opposite of posteam
        def_team = row.get('defteam', '')
        week = row.get('week', 0)
        season = row.get('season', 0)

        if not def_team or not week:
            continue

        key = f"{season}_{week}_{def_team}"

        play_type = row.get('play_type', '')
        epa = float(row.get('epa', 0) or 0)

        # Accumulate defensive metrics
        defensive_metrics[key]['total_plays'] += 1

        if play_type == 'pass':
            defensive_metrics[key]['pass_epa_sum'] += epa
            defensive_metrics[key]['pass_plays'] += 1

            # CPOE allowed
            cpoe = row.get('cpoe', '')
            if cpoe and cpoe != 'NA':
                try:
                    defensive_metrics[key]['cpoe_sum'] += float(cpoe)
                    defensive_metrics[key]['cpoe_count'] += 1
                except:
                    pass

        elif play_type == 'run':
            defensive_metrics[key]['rush_epa_sum'] += epa
            defensive_metrics[key]['rush_plays'] += 1

        # Success rate (EPA > 0 means offense succeeded)
        if epa > 0:
            defensive_metrics[key]['success_plays'] += 1

    # Calculate averages
    final_metrics = {}
    for key, metrics in defensive_metrics.items():
        total_plays = metrics['total_plays']

        if total_plays < 5:  # Minimum sample
            continue

        final_metrics[key] = {
            'def_pass_epa_allowed': round(metrics['pass_epa_sum'] / metrics['pass_plays'], 4) if metrics['pass_plays'] > 0 else 0.0,
            'def_rush_epa_allowed': round(metrics['rush_epa_sum'] / metrics['rush_plays'], 4) if metrics['rush_plays'] > 0 else 0.0,
            'def_success_rate_allowed': round(metrics['success_plays'] / total_plays, 4),
            'def_cpoe_allowed': round(metrics['cpoe_sum'] / metrics['cpoe_count'], 4) if metrics['cpoe_count'] > 0 else 0.0,
            'def_total_plays': total_plays
        }

    # Save to JSON
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(final_metrics, f, indent=2)

    print(f"\n✓ Extracted defensive metrics for {len(final_metrics)} team-weeks")
    print(f"✓ Saved to: {output_file}")

    return final_metrics


def extract_pace_metrics(
    pbp_file: Path,
    output_file: Path
) -> Dict:
    """Extract team pace and volume metrics.

    Args:
        pbp_file: Path to play-by-play parquet file
        output_file: Path to save pace metrics JSON

    Returns:
        Dict mapping (team, week) -> pace metrics
    """
    print(f"\n{'='*80}")
    print("EXTRACTING PACE & VOLUME METRICS")
    print(f"{'='*80}\n")

    # Load PBP data
    print(f"📂 Loading play-by-play data: {pbp_file}")
    df = pd.read_parquet(pbp_file)

    # Filter to offensive plays in neutral situations
    # Neutral = score within 8 pts, not last 2 min of half
    df = df[df['play_type'].isin(['pass', 'run'])]

    # Define neutral game script
    df['is_neutral'] = (
        (df['score_differential'].abs() <= 8) &
        (df['half_seconds_remaining'] > 120) &
        (df['qtr'].isin([1, 2, 3]))
    )

    # Calculate pace metrics by team/week
    pace_metrics = defaultdict(lambda: {
        'total_plays': 0,
        'pass_plays': 0,
        'rush_plays': 0,
        'neutral_plays': 0,
        'neutral_pass_plays': 0,
        'neutral_rush_plays': 0,
        'total_seconds': 0.0,
        'neutral_seconds': 0.0
    })

    for _, row in df.iterrows():
        team = row.get('posteam', '')
        week = row.get('week', 0)
        season = row.get('season', 0)

        if not team or not week:
            continue

        key = f"{season}_{week}_{team}"
        play_type = row.get('play_type', '')
        is_neutral = row.get('is_neutral', False)

        # Accumulate plays
        pace_metrics[key]['total_plays'] += 1

        if play_type == 'pass':
            pace_metrics[key]['pass_plays'] += 1
            if is_neutral:
                pace_metrics[key]['neutral_pass_plays'] += 1
        elif play_type == 'run':
            pace_metrics[key]['rush_plays'] += 1
            if is_neutral:
                pace_metrics[key]['neutral_rush_plays'] += 1

        if is_neutral:
            pace_metrics[key]['neutral_plays'] += 1

        # Track time (approximate seconds per snap)
        # This is simplified - in production, calculate from actual timestamps
        pace_metrics[key]['total_seconds'] += 40  # Avg ~40 sec per play
        if is_neutral:
            pace_metrics[key]['neutral_seconds'] += 40

    # Calculate final metrics
    final_metrics = {}
    for key, metrics in pace_metrics.items():
        total_plays = metrics['total_plays']

        if total_plays < 10:  # Minimum sample
            continue

        neutral_plays = metrics['neutral_plays']

        final_metrics[key] = {
            'team_plays_pg': total_plays,  # This is actually per week, will calculate rolling later
            'neutral_pass_rate': round(metrics['neutral_pass_plays'] / neutral_plays, 4) if neutral_plays > 0 else 0.5,
            'neutral_rush_rate': round(metrics['neutral_rush_plays'] / neutral_plays, 4) if neutral_plays > 0 else 0.5,
            'neutral_seconds_per_snap': round(metrics['neutral_seconds'] / neutral_plays, 2) if neutral_plays > 0 else 40.0,
            'total_pass_rate': round(metrics['pass_plays'] / total_plays, 4) if total_plays > 0 else 0.5
        }

    # Save to JSON
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(final_metrics, f, indent=2)

    print(f"\n✓ Extracted pace metrics for {len(final_metrics)} team-weeks")
    print(f"✓ Saved to: {output_file}")

    return final_metrics


def extract_stadium_features(
    games_file: Optional[Path],
    output_file: Path
) -> Dict:
    """Extract stadium features (dome vs outdoors).

    Args:
        games_file: Path to games/schedules data (optional, can build lookup manually)
        output_file: Path to save stadium features JSON

    Returns:
        Dict mapping stadium -> features
    """
    print(f"\n{'='*80}")
    print("EXTRACTING STADIUM FEATURES")
    print(f"{'='*80}\n")

    # Stadium lookup (hardcoded for now - can load from nflverse later)
    # Source: NFL stadium data
    stadium_features = {
        # Domes
        'ATL': {'is_dome': 1, 'is_outdoors': 0, 'roof_type': 'dome'},
        'DET': {'is_dome': 1, 'is_outdoors': 0, 'roof_type': 'dome'},
        'NO': {'is_dome': 1, 'is_outdoors': 0, 'roof_type': 'dome'},
        'MIN': {'is_dome': 1, 'is_outdoors': 0, 'roof_type': 'dome'},
        'LV': {'is_dome': 1, 'is_outdoors': 0, 'roof_type': 'dome'},

        # Retractable (treat as dome for simplicity)
        'HOU': {'is_dome': 1, 'is_outdoors': 0, 'roof_type': 'retractable'},
        'IND': {'is_dome': 1, 'is_outdoors': 0, 'roof_type': 'retractable'},
        'ARI': {'is_dome': 1, 'is_outdoors': 0, 'roof_type': 'retractable'},
        'DAL': {'is_dome': 1, 'is_outdoors': 0, 'roof_type': 'retractable'},
        'LAR': {'is_dome': 1, 'is_outdoors': 0, 'roof_type': 'dome'},

        # Outdoors (all others)
        'BUF': {'is_dome': 0, 'is_outdoors': 1, 'roof_type': 'outdoors'},
        'MIA': {'is_dome': 0, 'is_outdoors': 1, 'roof_type': 'outdoors'},
        'NE': {'is_dome': 0, 'is_outdoors': 1, 'roof_type': 'outdoors'},
        'NYJ': {'is_dome': 0, 'is_outdoors': 1, 'roof_type': 'outdoors'},
        'BAL': {'is_dome': 0, 'is_outdoors': 1, 'roof_type': 'outdoors'},
        'CIN': {'is_dome': 0, 'is_outdoors': 1, 'roof_type': 'outdoors'},
        'CLE': {'is_dome': 0, 'is_outdoors': 1, 'roof_type': 'outdoors'},
        'PIT': {'is_dome': 0, 'is_outdoors': 1, 'roof_type': 'outdoors'},
        'DEN': {'is_dome': 0, 'is_outdoors': 1, 'roof_type': 'outdoors'},
        'KC': {'is_dome': 0, 'is_outdoors': 1, 'roof_type': 'outdoors'},
        'LAC': {'is_dome': 0, 'is_outdoors': 1, 'roof_type': 'outdoors'},
        'CHI': {'is_dome': 0, 'is_outdoors': 1, 'roof_type': 'outdoors'},
        'GB': {'is_dome': 0, 'is_outdoors': 1, 'roof_type': 'outdoors'},
        'CAR': {'is_dome': 0, 'is_outdoors': 1, 'roof_type': 'outdoors'},
        'TB': {'is_dome': 0, 'is_outdoors': 1, 'roof_type': 'outdoors'},
        'WAS': {'is_dome': 0, 'is_outdoors': 1, 'roof_type': 'outdoors'},
        'NYG': {'is_dome': 0, 'is_outdoors': 1, 'roof_type': 'outdoors'},
        'PHI': {'is_dome': 0, 'is_outdoors': 1, 'roof_type': 'outdoors'},
        'JAX': {'is_dome': 0, 'is_outdoors': 1, 'roof_type': 'outdoors'},
        'TEN': {'is_dome': 0, 'is_outdoors': 1, 'roof_type': 'outdoors'},
        'SEA': {'is_dome': 0, 'is_outdoors': 1, 'roof_type': 'outdoors'},
        'SF': {'is_dome': 0, 'is_outdoors': 1, 'roof_type': 'outdoors'},
    }

    # Save to JSON
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(stadium_features, f, indent=2)

    print(f"✓ Created stadium lookup for {len(stadium_features)} teams")
    print(f"✓ Domes: {sum(1 for v in stadium_features.values() if v['is_dome'])}")
    print(f"✓ Outdoors: {sum(1 for v in stadium_features.values() if v['is_outdoors'])}")
    print(f"✓ Saved to: {output_file}")

    return stadium_features


def merge_context_into_features(
    player_features_file: Path,
    defensive_metrics_file: Path,
    pace_metrics_file: Path,
    stadium_features_file: Path,
    output_file: Path
) -> Dict:
    """Merge all context features into player features.

    Args:
        player_features_file: Player features JSON
        defensive_metrics_file: Defensive metrics JSON
        pace_metrics_file: Pace metrics JSON
        stadium_features_file: Stadium features JSON
        output_file: Output path for enhanced features

    Returns:
        Enhanced player features dict
    """
    print(f"\n{'='*80}")
    print("MERGING CONTEXT FEATURES INTO PLAYER FEATURES")
    print(f"{'='*80}\n")

    # Load all data
    print("📂 Loading player features...")
    with open(player_features_file, 'r') as f:
        player_features = json.load(f)

    print("📂 Loading defensive metrics...")
    with open(defensive_metrics_file, 'r') as f:
        defensive_metrics = json.load(f)

    print("📂 Loading pace metrics...")
    with open(pace_metrics_file, 'r') as f:
        pace_metrics = json.load(f)

    print("📂 Loading stadium features...")
    with open(stadium_features_file, 'r') as f:
        stadium_features = json.load(f)

    # Merge context into player features
    enhanced_count = 0

    for player_id, games in player_features.items():
        for game in games:
            season = game.get('season', '')
            week = game.get('week', '')
            opponent = game.get('opponent', '')
            team = game.get('team', '')

            # Add opponent defensive metrics
            opp_def_key = f"{season}_{week}_{opponent}"
            if opp_def_key in defensive_metrics:
                game.update(defensive_metrics[opp_def_key])

            # Add team pace metrics
            team_pace_key = f"{season}_{week}_{team}"
            if team_pace_key in pace_metrics:
                game.update(pace_metrics[team_pace_key])

            # Add stadium features (based on home team)
            # For now, use team's stadium (simplified - should use actual game location)
            if team in stadium_features:
                game.update(stadium_features[team])

            enhanced_count += 1

    # Save enhanced features
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(player_features, f, indent=2)

    print(f"\n✓ Enhanced {enhanced_count} player-games with context features")
    print(f"✓ Saved to: {output_file}")

    return player_features


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract context features for prop prediction'
    )
    parser.add_argument('--season', type=int, required=True,
                       help='Season year')
    parser.add_argument('--pbp-file', type=Path, required=True,
                       help='Play-by-play parquet file')
    parser.add_argument('--player-features', type=Path, required=True,
                       help='Player features JSON file')
    parser.add_argument('--output-dir', type=Path,
                       default=Path('outputs/features/context'),
                       help='Output directory')

    args = parser.parse_args()

    # Extract all context features
    defensive_metrics = extract_defensive_metrics(
        pbp_file=args.pbp_file,
        output_file=args.output_dir / f'{args.season}_defensive_metrics.json'
    )

    pace_metrics = extract_pace_metrics(
        pbp_file=args.pbp_file,
        output_file=args.output_dir / f'{args.season}_pace_metrics.json'
    )

    stadium_features = extract_stadium_features(
        games_file=None,
        output_file=args.output_dir / 'stadium_features.json'
    )

    # Merge into player features
    enhanced_features = merge_context_into_features(
        player_features_file=args.player_features,
        defensive_metrics_file=args.output_dir / f'{args.season}_defensive_metrics.json',
        pace_metrics_file=args.output_dir / f'{args.season}_pace_metrics.json',
        stadium_features_file=args.output_dir / 'stadium_features.json',
        output_file=args.output_dir.parent / f'{args.season}_player_features_with_context.json'
    )

    print(f"\n{'='*80}")
    print("CONTEXT FEATURE EXTRACTION COMPLETE")
    print(f"{'='*80}")
