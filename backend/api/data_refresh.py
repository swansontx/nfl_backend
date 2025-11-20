"""Data refresh system for auto-populating SQLite on startup and manual refresh.

This module provides:
- Startup data refresh (injuries, schedules, stats, standings)
- Manual refresh endpoints for MCP/Claude to trigger updates
- SQLite storage for fast API access
"""

import asyncio
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import logging

from backend.database.local_db import (
    init_database,
    get_database_status,
    InjuriesRepository,
    SchedulesRepository,
    PlayerStatsRepository,
    TeamStatsRepository,
    RostersRepository,
    OddsRepository
)
from backend.ingestion.fetch_injuries import InjuryFetcher
from backend.api.odds_api import odds_api
from backend.config import settings

logger = logging.getLogger(__name__)


# Play-by-play table creation (append to existing schema)
def ensure_pbp_table():
    """Ensure play-by-play table exists in database."""
    import sqlite3
    from backend.database.local_db import DB_PATH, get_db

    with get_db() as conn:
        cursor = conn.cursor()

        # Play-by-play data - comprehensive game logs with advanced metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS play_by_play (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                play_id TEXT,
                game_id TEXT,
                season INTEGER,
                week INTEGER,
                posteam TEXT,
                defteam TEXT,
                home_team TEXT,
                away_team TEXT,
                quarter INTEGER,
                time TEXT,
                down INTEGER,
                ydstogo INTEGER,
                yardline_100 INTEGER,
                play_type TEXT,
                yards_gained INTEGER,
                air_yards INTEGER,
                yards_after_catch INTEGER,
                pass_length TEXT,
                pass_location TEXT,
                run_location TEXT,
                run_gap TEXT,
                touchdown INTEGER,
                interception INTEGER,
                fumble INTEGER,
                sack INTEGER,
                penalty INTEGER,
                first_down INTEGER,
                epa REAL,
                wpa REAL,
                cpoe REAL,
                qb_hit INTEGER,
                score_differential INTEGER,
                roof TEXT,
                surface TEXT,
                defenders_in_box INTEGER,
                coverage_type TEXT,
                offense_personnel TEXT,
                defense_personnel TEXT,
                passer_player_name TEXT,
                passer_player_id TEXT,
                receiver_player_name TEXT,
                receiver_player_id TEXT,
                rusher_player_name TEXT,
                rusher_player_id TEXT,
                desc TEXT,
                UNIQUE(play_id, game_id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pbp_game
            ON play_by_play(game_id, quarter)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pbp_player
            ON play_by_play(passer_player_name, receiver_player_name, rusher_player_name)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pbp_season_week
            ON play_by_play(season, week)
        """)

        logger.info("Play-by-play table ensured")


class DataRefreshManager:
    """Manages data refresh operations for the NFL backend."""

    def __init__(self):
        self.inputs_dir = settings.inputs_dir
        self.last_refresh: Dict[str, datetime] = {}
        self.refresh_in_progress: Dict[str, bool] = {}

    async def refresh_all(self, force: bool = False) -> Dict:
        """Refresh all data sources.

        Args:
            force: Force refresh even if recently updated

        Returns:
            Dict with refresh results for each data type
        """
        results = {}

        # Initialize database if needed
        init_database()

        # Run all refreshes
        results['injuries'] = await self.refresh_injuries(force)
        results['schedules'] = await self.refresh_schedules(force)
        results['player_stats'] = await self.refresh_player_stats(force)
        results['team_stats'] = await self.refresh_team_stats(force)
        results['rosters'] = await self.refresh_rosters(force)
        results['play_by_play'] = await self.refresh_play_by_play(force)
        results['odds'] = await self.refresh_odds(force)

        results['timestamp'] = datetime.now().isoformat()
        results['database_status'] = get_database_status()

        return results

    async def refresh_odds(self, force: bool = False) -> Dict:
        """Refresh odds/lines from The Odds API.

        Args:
            force: Force refresh even if recently updated

        Returns:
            Dict with refresh results
        """
        if self.refresh_in_progress.get('odds', False):
            return {'status': 'in_progress', 'message': 'Odds refresh already running'}

        self.refresh_in_progress['odds'] = True

        try:
            logger.info("Refreshing odds data from The Odds API...")

            # Get upcoming games
            games = odds_api.get_upcoming_games()

            if not games:
                return {
                    'status': 'warning',
                    'message': 'No upcoming games found or API key not set',
                    'count': 0
                }

            # Get props for each game
            all_props = []
            for game in games:
                event_id = game.get('id')
                if event_id:
                    props = odds_api.get_player_props(event_id)
                    for prop in props:
                        all_props.append({
                            'game_id': event_id,
                            'player_id': getattr(prop, 'player_id', ''),
                            'player_name': getattr(prop, 'player_name', ''),
                            'team': getattr(prop, 'team', ''),
                            'prop_type': getattr(prop, 'prop_type', ''),
                            'line': getattr(prop, 'line', 0),
                            'over_odds': getattr(prop, 'over_odds', 0),
                            'under_odds': getattr(prop, 'under_odds', 0),
                            'book': getattr(prop, 'book', 'draftkings')
                        })

            if not all_props:
                return {
                    'status': 'warning',
                    'message': 'No props fetched from API',
                    'count': 0
                }

            # Determine current week/season
            now = datetime.now()
            season = now.year if now.month >= 9 else now.year - 1
            if now.month >= 9:
                week = min(18, (now - datetime(now.year, 9, 1)).days // 7 + 1)
            else:
                week = min(18, (now - datetime(now.year - 1, 9, 1)).days // 7 + 1)

            # Insert into database
            count = OddsRepository.insert_snapshot(all_props, week, season)

            self.last_refresh['odds'] = datetime.now()

            return {
                'status': 'success',
                'count': count,
                'games_processed': len(games),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error refreshing odds: {e}")
            return {'status': 'error', 'message': str(e)}

        finally:
            self.refresh_in_progress['odds'] = False

    async def refresh_injuries(self, force: bool = False) -> Dict:
        """Refresh injury data from ESPN API and store in SQLite.

        Args:
            force: Force refresh even if recently updated

        Returns:
            Dict with refresh results
        """
        if self.refresh_in_progress.get('injuries', False):
            return {'status': 'in_progress', 'message': 'Injury refresh already running'}

        self.refresh_in_progress['injuries'] = True

        try:
            logger.info("Refreshing injury data from ESPN...")

            # Fetch fresh injuries from ESPN
            fetcher = InjuryFetcher()
            all_injuries = []

            for team in fetcher.nfl_teams:
                try:
                    team_injuries = fetcher.fetch_team_injuries_espn(team)
                    all_injuries.extend(team_injuries)
                except Exception as e:
                    logger.warning(f"Error fetching injuries for {team}: {e}")

            if not all_injuries:
                return {
                    'status': 'warning',
                    'message': 'No injuries fetched from ESPN',
                    'count': 0
                }

            # Determine current week/season
            now = datetime.now()
            season = now.year if now.month >= 9 else now.year - 1
            # Rough week calculation (NFL weeks start in September)
            if now.month >= 9:
                week = min(18, (now - datetime(now.year, 9, 1)).days // 7 + 1)
            else:
                week = min(18, (now - datetime(now.year - 1, 9, 1)).days // 7 + 1)

            # Insert into database
            count = InjuriesRepository.insert_injuries(all_injuries, week, season)

            self.last_refresh['injuries'] = datetime.now()

            return {
                'status': 'success',
                'count': count,
                'season': season,
                'week': week,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error refreshing injuries: {e}")
            return {'status': 'error', 'message': str(e)}

        finally:
            self.refresh_in_progress['injuries'] = False

    async def refresh_schedules(self, force: bool = False) -> Dict:
        """Refresh schedule data from CSV files.

        Args:
            force: Force refresh

        Returns:
            Dict with refresh results
        """
        if self.refresh_in_progress.get('schedules', False):
            return {'status': 'in_progress', 'message': 'Schedule refresh already running'}

        self.refresh_in_progress['schedules'] = True

        try:
            logger.info("Refreshing schedule data...")

            # Find schedule files
            schedule_files = list(self.inputs_dir.glob('*schedule*.csv')) + \
                           list(self.inputs_dir.glob('*schedule*.parquet'))

            if not schedule_files:
                return {
                    'status': 'warning',
                    'message': 'No schedule files found in inputs/',
                    'count': 0
                }

            all_schedules = []
            for file in schedule_files:
                try:
                    if file.suffix == '.csv':
                        df = pd.read_csv(file)
                    else:
                        df = pd.read_parquet(file)

                    schedules = df.to_dict('records')
                    all_schedules.extend(schedules)
                except Exception as e:
                    logger.warning(f"Error reading schedule file {file}: {e}")

            if not all_schedules:
                return {
                    'status': 'warning',
                    'message': 'No schedule data loaded',
                    'count': 0
                }

            # Insert into database
            count = SchedulesRepository.upsert_schedules(all_schedules)

            self.last_refresh['schedules'] = datetime.now()

            return {
                'status': 'success',
                'count': count,
                'files_processed': len(schedule_files),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error refreshing schedules: {e}")
            return {'status': 'error', 'message': str(e)}

        finally:
            self.refresh_in_progress['schedules'] = False

    async def refresh_player_stats(self, force: bool = False) -> Dict:
        """Refresh player stats from CSV files.

        Args:
            force: Force refresh

        Returns:
            Dict with refresh results
        """
        if self.refresh_in_progress.get('player_stats', False):
            return {'status': 'in_progress', 'message': 'Player stats refresh already running'}

        self.refresh_in_progress['player_stats'] = True

        try:
            logger.info("Refreshing player stats...")

            # Find player stats files
            stats_files = list(self.inputs_dir.glob('player_stats*.csv'))

            if not stats_files:
                return {
                    'status': 'warning',
                    'message': 'No player stats files found',
                    'count': 0
                }

            all_stats = []
            for file in stats_files:
                try:
                    df = pd.read_csv(file)
                    stats = df.to_dict('records')
                    all_stats.extend(stats)
                except Exception as e:
                    logger.warning(f"Error reading stats file {file}: {e}")

            if not all_stats:
                return {
                    'status': 'warning',
                    'message': 'No player stats loaded',
                    'count': 0
                }

            # Insert into database
            count = PlayerStatsRepository.upsert_player_stats(all_stats)

            self.last_refresh['player_stats'] = datetime.now()

            return {
                'status': 'success',
                'count': count,
                'files_processed': len(stats_files),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error refreshing player stats: {e}")
            return {'status': 'error', 'message': str(e)}

        finally:
            self.refresh_in_progress['player_stats'] = False

    async def refresh_team_stats(self, force: bool = False) -> Dict:
        """Refresh team stats from CSV files.

        Args:
            force: Force refresh

        Returns:
            Dict with refresh results
        """
        if self.refresh_in_progress.get('team_stats', False):
            return {'status': 'in_progress', 'message': 'Team stats refresh already running'}

        self.refresh_in_progress['team_stats'] = True

        try:
            logger.info("Refreshing team stats...")

            # Find team stats files
            stats_files = list(self.inputs_dir.glob('*team_stats*.csv')) + \
                         list(self.inputs_dir.glob('defensive_stats*.csv'))

            if not stats_files:
                return {
                    'status': 'warning',
                    'message': 'No team stats files found',
                    'count': 0
                }

            all_stats = []
            for file in stats_files:
                try:
                    df = pd.read_csv(file)
                    stats = df.to_dict('records')
                    all_stats.extend(stats)
                except Exception as e:
                    logger.warning(f"Error reading team stats file {file}: {e}")

            if not all_stats:
                return {
                    'status': 'warning',
                    'message': 'No team stats loaded',
                    'count': 0
                }

            # Insert into database
            count = TeamStatsRepository.upsert_team_stats(all_stats)

            self.last_refresh['team_stats'] = datetime.now()

            return {
                'status': 'success',
                'count': count,
                'files_processed': len(stats_files),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error refreshing team stats: {e}")
            return {'status': 'error', 'message': str(e)}

        finally:
            self.refresh_in_progress['team_stats'] = False

    async def refresh_rosters(self, force: bool = False) -> Dict:
        """Refresh roster/depth chart data.

        Args:
            force: Force refresh

        Returns:
            Dict with refresh results
        """
        if self.refresh_in_progress.get('rosters', False):
            return {'status': 'in_progress', 'message': 'Roster refresh already running'}

        self.refresh_in_progress['rosters'] = True

        try:
            logger.info("Refreshing roster data...")

            # Find roster/depth chart files
            roster_files = list(self.inputs_dir.glob('rosters*.csv')) + \
                          list(self.inputs_dir.glob('depth_charts*.csv'))

            if not roster_files:
                return {
                    'status': 'warning',
                    'message': 'No roster files found',
                    'count': 0
                }

            all_rosters = []
            for file in roster_files:
                try:
                    df = pd.read_csv(file)
                    rosters = df.to_dict('records')
                    all_rosters.extend(rosters)
                except Exception as e:
                    logger.warning(f"Error reading roster file {file}: {e}")

            if not all_rosters:
                return {
                    'status': 'warning',
                    'message': 'No roster data loaded',
                    'count': 0
                }

            # Insert into database
            count = RostersRepository.upsert_rosters(all_rosters)

            self.last_refresh['rosters'] = datetime.now()

            return {
                'status': 'success',
                'count': count,
                'files_processed': len(roster_files),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error refreshing rosters: {e}")
            return {'status': 'error', 'message': str(e)}

        finally:
            self.refresh_in_progress['rosters'] = False

    async def refresh_play_by_play(self, force: bool = False) -> Dict:
        """Refresh play-by-play data from nflverse parquet files.

        Args:
            force: Force refresh

        Returns:
            Dict with refresh results
        """
        if self.refresh_in_progress.get('play_by_play', False):
            return {'status': 'in_progress', 'message': 'Play-by-play refresh already running'}

        self.refresh_in_progress['play_by_play'] = True

        try:
            logger.info("Refreshing play-by-play data...")

            # Ensure PBP table exists
            ensure_pbp_table()

            # Find play-by-play files
            pbp_files = list(self.inputs_dir.glob('play_by_play*.parquet')) + \
                       list(self.inputs_dir.glob('pbp*.parquet')) + \
                       list(self.inputs_dir.glob('*pbp*.csv'))

            if not pbp_files:
                return {
                    'status': 'warning',
                    'message': 'No play-by-play files found in inputs/',
                    'count': 0
                }

            from backend.database.local_db import get_db

            total_plays = 0
            for file in pbp_files:
                try:
                    if file.suffix == '.parquet':
                        df = pd.read_parquet(file)
                    else:
                        df = pd.read_csv(file)

                    # Select key columns for our PBP table
                    key_cols = [
                        'play_id', 'game_id', 'season', 'week', 'posteam', 'defteam',
                        'qtr', 'time', 'down', 'ydstogo', 'yardline_100', 'play_type',
                        'yards_gained', 'air_yards', 'yards_after_catch', 'pass_length',
                        'pass_location', 'run_location', 'run_gap', 'touchdown',
                        'interception', 'fumble', 'sack', 'penalty', 'first_down',
                        'epa', 'wpa', 'passer_player_name', 'passer_player_id',
                        'receiver_player_name', 'receiver_player_id',
                        'rusher_player_name', 'rusher_player_id', 'desc'
                    ]

                    # Filter to columns that exist
                    available_cols = [c for c in key_cols if c in df.columns]
                    df_filtered = df[available_cols].copy()

                    # Rename qtr to quarter if needed
                    if 'qtr' in df_filtered.columns:
                        df_filtered = df_filtered.rename(columns={'qtr': 'quarter'})

                    plays = df_filtered.to_dict('records')

                    # Insert into database
                    with get_db() as conn:
                        cursor = conn.cursor()
                        for play in plays:
                            try:
                                cursor.execute("""
                                    INSERT OR REPLACE INTO play_by_play
                                    (play_id, game_id, season, week, posteam, defteam,
                                     home_team, away_team, quarter, time, down, ydstogo,
                                     yardline_100, play_type, yards_gained, air_yards,
                                     yards_after_catch, pass_length, pass_location,
                                     run_location, run_gap, touchdown, interception,
                                     fumble, sack, penalty, first_down, epa, wpa, cpoe,
                                     qb_hit, score_differential, roof, surface,
                                     defenders_in_box, coverage_type, offense_personnel,
                                     defense_personnel, passer_player_name, passer_player_id,
                                     receiver_player_name, receiver_player_id,
                                     rusher_player_name, rusher_player_id, desc)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    play.get('play_id'),
                                    play.get('game_id'),
                                    play.get('season'),
                                    play.get('week'),
                                    play.get('posteam'),
                                    play.get('defteam'),
                                    play.get('home_team'),
                                    play.get('away_team'),
                                    play.get('qtr', play.get('quarter')),
                                    play.get('time'),
                                    play.get('down'),
                                    play.get('ydstogo'),
                                    play.get('yardline_100'),
                                    play.get('play_type'),
                                    play.get('yards_gained'),
                                    play.get('air_yards'),
                                    play.get('yards_after_catch'),
                                    play.get('pass_length'),
                                    play.get('pass_location'),
                                    play.get('run_location'),
                                    play.get('run_gap'),
                                    play.get('touchdown'),
                                    play.get('interception'),
                                    play.get('fumble'),
                                    play.get('sack'),
                                    play.get('penalty'),
                                    play.get('first_down'),
                                    play.get('epa'),
                                    play.get('wpa'),
                                    play.get('cpoe'),
                                    play.get('qb_hit'),
                                    play.get('score_differential'),
                                    play.get('roof'),
                                    play.get('surface'),
                                    play.get('defenders_in_box'),
                                    play.get('coverage_type'),
                                    play.get('offense_personnel'),
                                    play.get('defense_personnel'),
                                    play.get('passer_player_name'),
                                    play.get('passer_player_id'),
                                    play.get('receiver_player_name'),
                                    play.get('receiver_player_id'),
                                    play.get('rusher_player_name'),
                                    play.get('rusher_player_id'),
                                    play.get('desc')
                                ))
                                total_plays += 1
                            except Exception as e:
                                pass  # Skip duplicate plays

                except Exception as e:
                    logger.warning(f"Error reading PBP file {file}: {e}")

            self.last_refresh['play_by_play'] = datetime.now()

            return {
                'status': 'success',
                'count': total_plays,
                'files_processed': len(pbp_files),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error refreshing play-by-play: {e}")
            return {'status': 'error', 'message': str(e)}

        finally:
            self.refresh_in_progress['play_by_play'] = False

    def get_refresh_status(self) -> Dict:
        """Get current refresh status for all data types.

        Returns:
            Dict with last refresh times and in-progress status
        """
        return {
            'last_refresh': {
                k: v.isoformat() if v else None
                for k, v in self.last_refresh.items()
            },
            'in_progress': self.refresh_in_progress,
            'database_status': get_database_status()
        }


# Singleton instance
data_refresh_manager = DataRefreshManager()


async def startup_refresh():
    """Run data refresh on server startup.

    This is called by FastAPI's lifespan/startup event.
    """
    logger.info("Starting data refresh on server startup...")

    try:
        # Initialize database schema
        init_database()
        logger.info("Database initialized")

        # Refresh all data
        results = await data_refresh_manager.refresh_all()

        # Log results
        for data_type, result in results.items():
            if isinstance(result, dict) and 'status' in result:
                if result['status'] == 'success':
                    logger.info(f"  {data_type}: {result.get('count', 0)} records loaded")
                elif result['status'] == 'warning':
                    logger.warning(f"  {data_type}: {result.get('message', 'warning')}")
                elif result['status'] == 'error':
                    logger.error(f"  {data_type}: {result.get('message', 'error')}")

        logger.info("Startup data refresh complete")
        return results

    except Exception as e:
        logger.error(f"Startup refresh failed: {e}")
        raise
