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
from backend.database.models import Game, Player, PlayerGameFeature
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
            stadium='Huntington Bank Field',
            roof='outdoor',
            surface='grass'
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
                display_name=p['name'],
                team=p['team'],
                position=p['position']
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
        existing = session.query(PlayerGameFeature).filter_by(
            player_id=p['player_id'],
            game_id=game_id
        ).first()

        if existing:
            continue

        # Generate realistic mock features based on position
        if p['position'] == 'WR':
            features = PlayerGameFeature(
                player_id=p['player_id'],
                game_id=game_id,
                targets=int(7.5 + np.random.normal(0, 1)),
                receptions=int(4.8 + np.random.normal(0, 0.5)),
                receiving_yards=68.5 + np.random.normal(0, 10),
                receiving_tds=int(np.random.random() < 0.45),  # 45% chance of TD
                snaps=int(55 + np.random.normal(0, 5)),
                routes_run=int(38 + np.random.normal(0, 3)),
                redzone_targets=int(np.random.random() * 3)
            )
        elif p['position'] == 'RB':
            features = PlayerGameFeature(
                player_id=p['player_id'],
                game_id=game_id,
                rush_attempts=int(12.5 + np.random.normal(0, 2)),
                rushing_yards=55.5 + np.random.normal(0, 8),
                rushing_tds=int(np.random.random() < 0.35),  # 35% chance of TD
                targets=int(3.2 + np.random.normal(0, 0.5)),
                receptions=int(2.5 + np.random.normal(0, 0.3)),
                receiving_yards=18.5 + np.random.normal(0, 5),
                snaps=int(42 + np.random.normal(0, 6)),
                redzone_carries=int(np.random.random() * 4)
            )
        elif p['position'] == 'TE':
            features = PlayerGameFeature(
                player_id=p['player_id'],
                game_id=game_id,
                targets=int(5.5 + np.random.normal(0, 0.8)),
                receptions=int(3.8 + np.random.normal(0, 0.5)),
                receiving_yards=45.5 + np.random.normal(0, 6),
                receiving_tds=int(np.random.random() < 0.30),  # 30% chance of TD
                snaps=int(48 + np.random.normal(0, 4)),
                routes_run=int(28 + np.random.normal(0, 3)),
                redzone_targets=int(np.random.random() * 2)
            )
        elif p['position'] == 'QB':
            features = PlayerGameFeature(
                player_id=p['player_id'],
                game_id=game_id,
                pass_attempts=int(32.5 + np.random.normal(0, 3)),
                passing_yards=245.5 + np.random.normal(0, 20),
                passing_tds=int(1.65 + np.random.normal(0, 0.5)),
                rushing_yards=15.5 + np.random.normal(0, 5),
                rush_attempts=int(3.5 + np.random.normal(0, 1)),
                snaps=int(62 + np.random.normal(0, 2))
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
