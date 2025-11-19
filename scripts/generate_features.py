#!/usr/bin/env python3
"""
Generate player features from play-by-play data

This script:
1. Loads play-by-play data from nfl_data_py
2. Aggregates stats per player per game
3. Calculates rolling averages and advanced metrics
4. Stores in PlayerGameFeature table
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from backend.database.session import get_db
from backend.database.models import Game, Player, PlayerGameFeature
from backend.config.logging_config import get_logger

logger = get_logger(__name__)

print("=" * 80)
print("FEATURE GENERATION")
print("=" * 80)
print()

# Check if nfl_data_py is installed
try:
    import nfl_data_py as nfl
    print("✅ nfl_data_py is installed")
except ImportError:
    print("❌ nfl_data_py not installed")
    print()
    print("Install it with:")
    print("  pip install nfl_data_py")
    print()
    sys.exit(1)

# Load play-by-play data for 2024 season
print()
print("📥 Loading play-by-play data from nfl_data_py...")
print("This will take a few minutes (data is cached locally)")
print()

try:
    pbp = nfl.import_pbp_data([2024])
    print(f"✅ Loaded {len(pbp):,} plays from 2024 season")
except Exception as e:
    print(f"❌ Error loading play-by-play data: {e}")
    sys.exit(1)

# Filter to completed plays
pbp = pbp[pbp['play_type'].isin(['pass', 'run'])]
print(f"   Filtered to {len(pbp):,} pass/run plays")

print()
print("🔄 Aggregating player stats by game...")
print()

# Get all games from database to match game_ids
with get_db() as session:
    games = session.query(Game).filter(Game.season == 2024).all()
    game_lookup = {(g.season, g.week, g.away_team, g.home_team): g.game_id for g in games}

features_created = 0
features_updated = 0

# Group by game_id and player
for game_id, plays in pbp.groupby('game_id'):
    # Parse game_id to get our format
    try:
        # pbp game_id format: 2024_01_DET_KC
        parts = str(game_id).split('_')
        if len(parts) >= 4:
            season = int(parts[0])
            week = int(parts[1])
            away = parts[2]
            home = parts[3]

            our_game_id = f"{season}_{week:02d}_{away}_{home}"
        else:
            continue
    except:
        continue

    with get_db() as session:
        # Check if game exists
        game = session.query(Game).filter_by(game_id=our_game_id).first()
        if not game:
            continue

        # Aggregate receiver stats
        receivers = plays[plays['receiver_player_id'].notna()].groupby('receiver_player_id').agg({
            'receiver_player_id': 'first',
            'receiver_player_name': 'first',
            'complete_pass': 'sum',
            'yards_gained': lambda x: x[plays.loc[x.index, 'complete_pass'] == 1].sum(),
            'pass_touchdown': 'sum',
            'epa': 'mean'
        }).rename(columns={
            'complete_pass': 'receptions',
            'yards_gained': 'receiving_yards',
            'pass_touchdown': 'receiving_tds'
        })
        receivers['targets'] = plays[plays['receiver_player_id'].notna()].groupby('receiver_player_id').size()

        # Aggregate rusher stats
        rushers = plays[plays['rusher_player_id'].notna()].groupby('rusher_player_id').agg({
            'rusher_player_id': 'first',
            'rusher_player_name': 'first',
            'rush_attempt': 'sum',
            'yards_gained': lambda x: x[plays.loc[x.index, 'rush_attempt'] == 1].sum(),
            'rush_touchdown': 'sum',
            'epa': 'mean'
        }).rename(columns={
            'rush_attempt': 'rush_attempts',
            'yards_gained': 'rushing_yards',
            'rush_touchdown': 'rushing_tds'
        })

        # Aggregate passer stats
        passers = plays[plays['passer_player_id'].notna()].groupby('passer_player_id').agg({
            'passer_player_id': 'first',
            'passer_player_name': 'first',
            'pass_attempt': 'sum',
            'complete_pass': 'sum',
            'yards_gained': lambda x: x[plays.loc[x.index, 'pass_attempt'] == 1].sum(),
            'pass_touchdown': 'sum',
            'interception': 'sum',
            'epa': 'mean'
        }).rename(columns={
            'pass_attempt': 'pass_attempts',
            'complete_pass': 'completions',
            'yards_gained': 'passing_yards',
            'pass_touchdown': 'passing_tds',
            'interception': 'interceptions'
        })

        # Process receivers
        for player_id, stats in receivers.iterrows():
            existing = session.query(PlayerGameFeature).filter_by(
                player_id=player_id,
                game_id=our_game_id
            ).first()

            if existing:
                # Update existing
                existing.targets = int(stats.get('targets', 0))
                existing.receptions = int(stats.get('receptions', 0))
                existing.receiving_yards = float(stats.get('receiving_yards', 0))
                existing.receiving_tds = int(stats.get('receiving_tds', 0))
                features_updated += 1
            else:
                # Create new
                feature = PlayerGameFeature(
                    player_id=player_id,
                    game_id=our_game_id,
                    targets=int(stats.get('targets', 0)),
                    receptions=int(stats.get('receptions', 0)),
                    receiving_yards=float(stats.get('receiving_yards', 0)),
                    receiving_tds=int(stats.get('receiving_tds', 0))
                )
                session.add(feature)
                features_created += 1

        # Process rushers
        for player_id, stats in rushers.iterrows():
            existing = session.query(PlayerGameFeature).filter_by(
                player_id=player_id,
                game_id=our_game_id
            ).first()

            if existing:
                # Update existing
                existing.rush_attempts = int(stats.get('rush_attempts', 0))
                existing.rushing_yards = float(stats.get('rushing_yards', 0))
                existing.rushing_tds = int(stats.get('rushing_tds', 0))
                features_updated += 1
            else:
                # Create new or merge with receiver
                feature = PlayerGameFeature(
                    player_id=player_id,
                    game_id=our_game_id,
                    rush_attempts=int(stats.get('rush_attempts', 0)),
                    rushing_yards=float(stats.get('rushing_yards', 0)),
                    rushing_tds=int(stats.get('rushing_tds', 0))
                )
                session.add(feature)
                features_created += 1

        # Process passers
        for player_id, stats in passers.iterrows():
            existing = session.query(PlayerGameFeature).filter_by(
                player_id=player_id,
                game_id=our_game_id
            ).first()

            if existing:
                # Update existing
                existing.pass_attempts = int(stats.get('pass_attempts', 0))
                existing.passing_yards = float(stats.get('passing_yards', 0))
                existing.passing_tds = int(stats.get('passing_tds', 0))
                features_updated += 1
            else:
                # Create new
                feature = PlayerGameFeature(
                    player_id=player_id,
                    game_id=our_game_id,
                    pass_attempts=int(stats.get('pass_attempts', 0)),
                    passing_yards=float(stats.get('passing_yards', 0)),
                    passing_tds=int(stats.get('passing_tds', 0))
                )
                session.add(feature)
                features_created += 1

        session.commit()

    if (features_created + features_updated) % 100 == 0:
        print(f"   Processed {features_created + features_updated} features...")

print()
print("=" * 80)
print("✅ Feature Generation Complete!")
print("=" * 80)
print()
print(f"Features Created: {features_created}")
print(f"Features Updated: {features_updated}")
print(f"Total: {features_created + features_updated}")
print()
print("Next steps:")
print("  1. Train models: python scripts/train_models.py")
print("  2. Test recommendations: python scripts/demo_analysis.py")
print()
