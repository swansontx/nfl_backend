#!/usr/bin/env python3
"""
Quick setup script for testing the recommendation system

Creates sample data and simple models so you can test the API
without needing full historical data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from datetime import datetime, timedelta
from backend.database.session import get_db
from backend.database.models import Game, Player, PlayerGameFeatures
from backend.config.logging_config import get_logger

logger = get_logger(__name__)

print("=" * 80)
print("QUICK SETUP FOR TESTING")
print("=" * 80)
print()
print("This creates sample data so you can test the recommendation system")
print()

# Create sample Thursday game
print("1️⃣ Creating Thursday night game (PIT @ CLE)...")

with get_db() as session:
    # Create game
    game_id = "2024_12_PIT_CLE"
    existing_game = session.query(Game).filter_by(game_id=game_id).first()

    if not existing_game:
        game = Game(
            game_id=game_id,
            season=2024,
            week=12,
            game_type='REG',
            away_team='PIT',
            home_team='CLE',
            game_date=datetime.now() + timedelta(days=2),  # Thursday
            gametime='20:15',
            stadium='Huntington Bank Field',
            location='Cleveland',
            roof='outdoor',
            surface='grass',
            completed=False
        )
        session.add(game)
        session.commit()
        print(f"   ✅ Created game: {game_id}")
    else:
        print(f"   ⚠️  Game already exists: {game_id}")

# Create sample players
print()
print("2️⃣ Creating sample players...")

sample_players = [
    # Pittsburgh
    {'player_id': 'PIT_PICKENS_G', 'name': 'George Pickens', 'team': 'PIT', 'position': 'WR'},
    {'player_id': 'PIT_WARREN_J', 'name': 'Jaylen Warren', 'team': 'PIT', 'position': 'RB'},
    {'player_id': 'PIT_WILSON_R', 'name': 'Russell Wilson', 'team': 'PIT', 'position': 'QB'},
    {'player_id': 'PIT_FREIERMUTH_P', 'name': 'Pat Freiermuth', 'team': 'PIT', 'position': 'TE'},

    # Cleveland
    {'player_id': 'CLE_NJOKU_D', 'name': 'David Njoku', 'team': 'CLE', 'position': 'TE'},
    {'player_id': 'CLE_CHUBB_N', 'name': 'Nick Chubb', 'team': 'CLE', 'position': 'RB'},
    {'player_id': 'CLE_WATSON_D', 'name': 'Deshaun Watson', 'team': 'CLE', 'position': 'QB'},
    {'player_id': 'CLE_COOPER_A', 'name': 'Amari Cooper', 'team': 'CLE', 'position': 'WR'},
]

with get_db() as session:
    for p in sample_players:
        existing = session.query(Player).filter_by(player_id=p['player_id']).first()
        if not existing:
            player = Player(
                player_id=p['player_id'],
                player_name=p['name'],
                team=p['team'],
                position=p['position'],
                status='ACT'
            )
            session.add(player)

    session.commit()
    print(f"   ✅ Created {len(sample_players)} players")

# Create sample features
print()
print("3️⃣ Creating sample player features...")

with get_db() as session:
    features_created = 0

    for p in sample_players:
        existing = session.query(PlayerGameFeatures).filter_by(
            player_id=p['player_id'],
            game_id=game_id
        ).first()

        if existing:
            continue

        # Generate realistic mock features based on position
        if p['position'] == 'WR':
            features = PlayerGameFeatures(
                player_id=p['player_id'],
                game_id=game_id,
                season=2024,
                week=12,
                team=p['team'],
                position=p['position'],
                targets_avg=7.5 + np.random.normal(0, 1),
                receptions_avg=4.8 + np.random.normal(0, 0.5),
                rec_yards_avg=68.5 + np.random.normal(0, 10),
                rec_td_avg=0.45 + np.random.normal(0, 0.1),
                snaps_pct=0.75 + np.random.normal(0, 0.05),
                route_pct=0.82 + np.random.normal(0, 0.05)
            )
        elif p['position'] == 'RB':
            features = PlayerGameFeatures(
                player_id=p['player_id'],
                game_id=game_id,
                season=2024,
                week=12,
                team=p['team'],
                position=p['position'],
                carries_avg=12.5 + np.random.normal(0, 2),
                rush_yards_avg=55.5 + np.random.normal(0, 8),
                rush_td_avg=0.35 + np.random.normal(0, 0.1),
                targets_avg=3.2 + np.random.normal(0, 0.5),
                receptions_avg=2.5 + np.random.normal(0, 0.3),
                snaps_pct=0.60 + np.random.normal(0, 0.08)
            )
        elif p['position'] == 'TE':
            features = PlayerGameFeatures(
                player_id=p['player_id'],
                game_id=game_id,
                season=2024,
                week=12,
                team=p['team'],
                position=p['position'],
                targets_avg=5.5 + np.random.normal(0, 0.8),
                receptions_avg=3.8 + np.random.normal(0, 0.5),
                rec_yards_avg=45.5 + np.random.normal(0, 6),
                rec_td_avg=0.30 + np.random.normal(0, 0.08),
                snaps_pct=0.68 + np.random.normal(0, 0.06)
            )
        elif p['position'] == 'QB':
            features = PlayerGameFeatures(
                player_id=p['player_id'],
                game_id=game_id,
                season=2024,
                week=12,
                team=p['team'],
                position=p['position'],
                pass_attempts_avg=32.5 + np.random.normal(0, 3),
                completions_avg=21.5 + np.random.normal(0, 2),
                pass_yards_avg=245.5 + np.random.normal(0, 20),
                pass_td_avg=1.65 + np.random.normal(0, 0.3),
                interceptions_avg=0.75 + np.random.normal(0, 0.2),
                rush_yards_avg=15.5 + np.random.normal(0, 5)
            )

        session.add(features)
        features_created += 1

    session.commit()
    print(f"   ✅ Created {features_created} player feature sets")

print()
print("=" * 80)
print("✅ Setup Complete!")
print("=" * 80)
print()
print("You can now:")
print("  1. Test the API: curl http://localhost:8000/api/v1/games/")
print("  2. Get game recommendations: curl http://localhost:8000/api/v1/recommendations/2024_12_PIT_CLE")
print("  3. View in browser: http://localhost:8000/docs")
print()
print("Note: Recommendations will be based on simple statistical models.")
print("For production, you'll want to train XGBoost models on real historical data.")
print()
