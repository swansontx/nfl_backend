#!/usr/bin/env python3
"""
Load NFL data into PostgreSQL database

Uses nfl_data_py to fetch:
- Game schedules
- Play-by-play data
- Player statistics
- Rosters
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from backend.database.session import get_db
from backend.database.models import Game, Player
from backend.config.logging_config import get_logger

logger = get_logger(__name__)

print("=" * 80)
print("NFL DATA LOADER")
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

print()
print("📥 Fetching NFL data from nfl_data_py...")
print("This may take a few minutes on first run (data is cached locally)")
print()

# Load 2024 season schedule
print("1️⃣ Loading 2024 season schedule...")
try:
    schedule = nfl.import_schedules([2024])
    print(f"   ✅ Loaded {len(schedule)} games")
except Exception as e:
    print(f"   ❌ Error: {e}")
    schedule = None

# Load rosters
print("2️⃣ Loading player rosters...")
try:
    rosters = nfl.import_seasonal_rosters([2024])
    print(f"   ✅ Loaded {len(rosters)} players")
except Exception as e:
    print(f"   ❌ Error: {e}")
    rosters = None

# Load to database
print()
print("💾 Saving to PostgreSQL database...")
print()

with get_db() as session:
    # Load games
    if schedule is not None:
        print("Loading games...")
        games_added = 0

        for _, game_row in schedule.iterrows():
            # Create game_id in our format: SEASON_WEEK_AWAY_HOME
            game_id = f"{game_row['season']}_{game_row['week']:02d}_{game_row['away_team']}_{game_row['home_team']}"

            # Check if game already exists
            existing = session.query(Game).filter_by(game_id=game_id).first()
            if existing:
                continue

            # Create game
            game = Game(
                game_id=game_id,
                season=int(game_row['season']),
                week=int(game_row['week']),
                game_type=game_row.get('game_type', 'REG'),
                away_team=game_row['away_team'],
                home_team=game_row['home_team'],
                game_date=game_row['gameday'],
                stadium=game_row.get('stadium'),
                roof=game_row.get('roof'),
                surface=game_row.get('surface'),
                temp=game_row.get('temp'),
                wind=game_row.get('wind'),
                away_score=game_row.get('away_score'),
                home_score=game_row.get('home_score')
            )

            session.add(game)
            games_added += 1

        session.commit()
        print(f"   ✅ Added {games_added} new games")

    # Load players
    if rosters is not None:
        print("Loading players...")
        players_added = 0

        for _, player_row in rosters.iterrows():
            player_id = player_row.get('player_id') or player_row.get('gsis_id')
            if not player_id:
                continue

            # Check if player exists
            existing = session.query(Player).filter_by(player_id=player_id).first()
            if existing:
                continue

            # Handle NaT (Not a Time) values from pandas
            birth_date = player_row.get('birth_date')
            if birth_date is not None and str(birth_date) == 'NaT':
                birth_date = None

            # Create player
            player = Player(
                player_id=player_id,
                display_name=player_row.get('player_name') or player_row.get('full_name'),
                first_name=player_row.get('first_name'),
                last_name=player_row.get('last_name'),
                team=player_row.get('team'),
                position=player_row.get('position'),
                height=player_row.get('height'),
                weight=player_row.get('weight'),
                birth_date=birth_date,
                college=player_row.get('college')
            )

            session.add(player)
            players_added += 1

        session.commit()
        print(f"   ✅ Added {players_added} new players")

print()
print("=" * 80)
print("📊 Database Summary")
print("=" * 80)

with get_db() as session:
    game_count = session.query(Game).count()
    player_count = session.query(Player).count()

    print(f"Total Games: {game_count}")
    print(f"Total Players: {player_count}")

    # Show upcoming games
    print()
    print("Upcoming Games:")
    upcoming = session.query(Game).filter(
        Game.away_score == None,
        Game.season == 2024
    ).order_by(Game.week, Game.game_date).limit(5).all()

    for game in upcoming:
        date_str = game.game_date.strftime('%Y-%m-%d') if game.game_date else 'TBD'
        print(f"  Week {game.week}: {game.away_team} @ {game.home_team} ({date_str})")

print()
print("=" * 80)
print("✅ Data loading complete!")
print("=" * 80)
print()
print("Next steps:")
print("  1. Generate player features: python scripts/generate_features.py")
print("  2. Train models: python scripts/train_models.py")
print("  3. Test recommendations: python scripts/demo_analysis.py")
print()
