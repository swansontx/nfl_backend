"""
Data ingestion pipeline for populating database with historical data

Orchestrates:
1. Fetching data from nflverse
2. Parsing and transforming data
3. Populating database tables
4. Fetching and storing odds data
"""

from typing import List, Optional
from datetime import datetime
import pandas as pd

from backend.config import settings
from backend.config.logging_config import get_logger
from backend.database.session import get_db
from backend.database.models import Player, Game, PlayerGameFeature, RosterStatus, Projection
from backend.ingestion import NFLverseClient, OddsAPIClient

logger = get_logger(__name__)


class DataIngestionPipeline:
    """Orchestrates data ingestion from multiple sources"""

    def __init__(self):
        self.nflverse = NFLverseClient()
        self.odds_client = OddsAPIClient()

    def ingest_season(self, season: int, refresh: bool = False) -> None:
        """
        Ingest all data for a season

        Args:
            season: NFL season year
            refresh: Force refresh even if data exists
        """
        logger.info("ingesting_season", season=season, refresh=refresh)

        # Step 1: Fetch and populate players
        self._ingest_players(season, refresh)

        # Step 2: Fetch and populate games
        self._ingest_games(season, refresh)

        # Step 3: Fetch and populate player-game features
        self._ingest_player_game_features(season, refresh)

        # Step 4: Fetch and populate roster status
        self._ingest_roster_status(season, refresh)

        logger.info("season_ingestion_complete", season=season)

    def _ingest_players(self, season: int, refresh: bool) -> None:
        """Ingest player records from rosters"""
        logger.info("ingesting_players", season=season)

        rosters_df = self.nflverse.get_weekly_rosters(season)

        # Get unique players
        players_df = rosters_df[[
            'gsis_id', 'full_name', 'position', 'team', 'birth_date', 'height', 'weight'
        ]].drop_duplicates(subset=['gsis_id'])

        with get_db() as session:
            added = 0
            updated = 0

            for _, row in players_df.iterrows():
                player_id = row['gsis_id']
                if pd.isna(player_id):
                    continue

                existing = session.query(Player).filter(Player.player_id == player_id).first()

                if existing and not refresh:
                    continue

                if existing:
                    # Update
                    existing.display_name = row['full_name']
                    existing.position = row['position']
                    existing.team = row['team']
                    existing.birth_date = pd.to_datetime(row['birth_date']) if not pd.isna(row['birth_date']) else None
                    existing.height = row['height'] if not pd.isna(row['height']) else None
                    existing.weight = row['weight'] if not pd.isna(row['weight']) else None
                    updated += 1
                else:
                    # Create
                    player = Player(
                        player_id=player_id,
                        display_name=row['full_name'],
                        position=row['position'],
                        team=row['team'],
                        birth_date=pd.to_datetime(row['birth_date']) if not pd.isna(row['birth_date']) else None,
                        height=row['height'] if not pd.isna(row['height']) else None,
                        weight=row['weight'] if not pd.isna(row['weight']) else None
                    )
                    session.add(player)
                    added += 1

            session.commit()

        logger.info("players_ingested", season=season, added=added, updated=updated)

    def _ingest_games(self, season: int, refresh: bool) -> None:
        """Ingest game records from play-by-play data"""
        logger.info("ingesting_games", season=season)

        pbp_df = self.nflverse.get_play_by_play(season)

        # Extract unique games
        games_df = pbp_df[[
            'game_id', 'season', 'week', 'game_date',
            'home_team', 'away_team', 'total', 'spread_line'
        ]].drop_duplicates(subset=['game_id'])

        with get_db() as session:
            added = 0
            updated = 0

            for _, row in games_df.iterrows():
                game_id = row['game_id']

                existing = session.query(Game).filter(Game.game_id == game_id).first()

                if existing and not refresh:
                    continue

                game_date = pd.to_datetime(row['game_date'])

                if existing:
                    # Update
                    existing.season = int(row['season'])
                    existing.week = int(row['week'])
                    existing.game_date = game_date
                    existing.home_team = row['home_team']
                    existing.away_team = row['away_team']
                    updated += 1
                else:
                    # Create
                    game = Game(
                        game_id=game_id,
                        season=int(row['season']),
                        week=int(row['week']),
                        game_date=game_date,
                        home_team=row['home_team'],
                        away_team=row['away_team'],
                    )
                    session.add(game)
                    added += 1

            session.commit()

        logger.info("games_ingested", season=season, added=added, updated=updated)

    def _ingest_player_game_features(self, season: int, refresh: bool) -> None:
        """
        Extract and ingest player-game features from play-by-play data

        This is the key step that extracts actual stats for feature engineering
        """
        logger.info("ingesting_player_game_features", season=season)

        pbp_df = self.nflverse.get_play_by_play(season)

        # Get all unique players from PBP
        all_players = set()
        for col in ['passer_player_id', 'rusher_player_id', 'receiver_player_id']:
            all_players.update(pbp_df[col].dropna().unique())

        # Get all games
        all_games = pbp_df['game_id'].unique()

        with get_db() as session:
            added = 0
            skipped = 0

            for player_id in all_players:
                if pd.isna(player_id):
                    continue

                # Check if player exists
                player = session.query(Player).filter(Player.player_id == player_id).first()
                if not player:
                    logger.debug("player_not_found", player_id=player_id)
                    continue

                # Extract stats for each game
                for game_id in all_games:
                    # Check if already exists
                    if not refresh:
                        existing = session.query(PlayerGameFeature).filter(
                            PlayerGameFeature.player_id == player_id,
                            PlayerGameFeature.game_id == game_id
                        ).first()
                        if existing:
                            skipped += 1
                            continue

                    # Extract stats
                    stats = self.nflverse.extract_player_game_stats(pbp_df, player_id, game_id)

                    # Only create if player had some participation
                    if all([
                        stats['pass_attempts'] == 0,
                        stats['rush_attempts'] == 0,
                        stats['targets'] == 0
                    ]):
                        continue

                    # Create or update
                    feature = PlayerGameFeature(
                        player_id=player_id,
                        game_id=game_id,
                        snaps=stats.get('routes_run', 0) + stats.get('rush_attempts', 0),  # Approximate
                        routes_run=stats.get('routes_run', 0),
                        targets=stats.get('targets', 0),
                        receptions=stats.get('receptions', 0),
                        receiving_yards=stats.get('receiving_yards', 0),
                        receiving_tds=stats.get('receiving_tds', 0),
                        carries=stats.get('rush_attempts', 0),
                        rushing_yards=stats.get('rushing_yards', 0),
                        rushing_tds=stats.get('rushing_tds', 0),
                        pass_attempts=stats.get('pass_attempts', 0),
                        completions=stats.get('completions', 0),
                        passing_yards=stats.get('passing_yards', 0),
                        passing_tds=stats.get('passing_tds', 0),
                        interceptions=stats.get('interceptions', 0),
                        redzone_targets=stats.get('redzone_targets', 0),
                        redzone_carries=stats.get('redzone_carries', 0),
                        air_yards=stats.get('air_yards', 0),
                        yac=stats.get('yards_after_catch', 0),
                    )

                    session.add(feature)
                    added += 1

                    # Commit in batches
                    if added % 1000 == 0:
                        session.commit()
                        logger.info("features_batch_committed", added=added)

            session.commit()

        logger.info(
            "player_game_features_ingested",
            season=season,
            added=added,
            skipped=skipped
        )

    def _ingest_roster_status(self, season: int, refresh: bool) -> None:
        """Ingest roster and injury status data"""
        logger.info("ingesting_roster_status", season=season)

        try:
            injuries_df = self.nflverse.get_injuries(season)
        except Exception as e:
            logger.warning("injuries_fetch_failed", error=str(e))
            return

        with get_db() as session:
            added = 0

            for _, row in injuries_df.iterrows():
                player_id = row.get('gsis_id')
                game_id = row.get('game_id')

                if pd.isna(player_id) or pd.isna(game_id):
                    continue

                # Map injury status to our enum
                report_status = row.get('report_status', 'Active')

                status_map = {
                    'Out': 'OUT',
                    'Doubtful': 'DOUBTFUL',
                    'Questionable': 'QUESTIONABLE',
                    'Probable': 'PROBABLE',
                }

                status = status_map.get(report_status, 'ACTIVE')

                # Check if exists
                if not refresh:
                    existing = session.query(RosterStatus).filter(
                        RosterStatus.player_id == player_id,
                        RosterStatus.game_id == game_id
                    ).first()
                    if existing:
                        continue

                roster_status = RosterStatus(
                    player_id=player_id,
                    game_id=game_id,
                    status=status,
                    week=int(row.get('week', 0)),
                )

                session.add(roster_status)
                added += 1

            session.commit()

        logger.info("roster_status_ingested", season=season, added=added)

    def ingest_current_week_odds(self) -> None:
        """Fetch current week's odds and store in database"""
        logger.info("ingesting_current_odds")

        try:
            games = self.odds_client.get_nfl_games()

            # Store odds in database
            # This would update Game records with current lines
            logger.info("current_odds_fetched", games=len(games))

        except Exception as e:
            logger.error("odds_ingestion_failed", error=str(e))

    def close(self):
        """Clean up resources"""
        self.nflverse.close()
        self.odds_client.close()
