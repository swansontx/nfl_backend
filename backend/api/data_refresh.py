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
from backend.ingestion.fetch_nflverse import fetch_nflverse
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


def ensure_advanced_tables():
    """Ensure advanced nflverse tables exist in database."""
    from backend.database.local_db import get_db

    with get_db() as conn:
        cursor = conn.cursor()

        # Snap counts - player participation percentages
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS snap_counts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                season INTEGER,
                week INTEGER,
                game_id TEXT,
                player_id TEXT,
                player_name TEXT,
                team TEXT,
                position TEXT,
                offense_snaps INTEGER,
                offense_pct REAL,
                defense_snaps INTEGER,
                defense_pct REAL,
                st_snaps INTEGER,
                st_pct REAL,
                UNIQUE(season, week, player_id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_snaps_player
            ON snap_counts(player_id, season)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_snaps_team
            ON snap_counts(team, season, week)
        """)

        # FTN charting - coverage schemes and formations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ftn_charting (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                season INTEGER,
                week INTEGER,
                game_id TEXT,
                play_id TEXT,
                offense_formation TEXT,
                offense_personnel TEXT,
                defense_coverage TEXT,
                defense_man_zone TEXT,
                defenders_in_box INTEGER,
                pass_rushers INTEGER,
                blitz INTEGER,
                UNIQUE(season, week, game_id, play_id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ftn_game
            ON ftn_charting(game_id, season)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ftn_coverage
            ON ftn_charting(defense_man_zone, season)
        """)

        # PFR advanced stats - passing, rushing, receiving, defense
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pfr_advanced (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                season INTEGER,
                player_id TEXT,
                player_name TEXT,
                team TEXT,
                stat_type TEXT,
                games INTEGER,
                attempts INTEGER,
                yards INTEGER,
                tds INTEGER,
                first_downs INTEGER,
                ybc REAL,
                yac REAL,
                broken_tackles INTEGER,
                drops INTEGER,
                drop_pct REAL,
                bad_throws INTEGER,
                bad_throw_pct REAL,
                pocket_time REAL,
                blitzed INTEGER,
                hurried INTEGER,
                pressured_pct REAL,
                UNIQUE(season, player_id, stat_type)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pfr_player
            ON pfr_advanced(player_id, season)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pfr_team
            ON pfr_advanced(team, stat_type, season)
        """)

        logger.info("Advanced nflverse tables ensured")


# Staleness thresholds (in hours)
STALENESS_THRESHOLDS = {
    'injuries': 6,      # Refresh if > 6 hours old
    'odds': 2,          # Refresh if > 2 hours old (lines move)
    'schedules': 24,    # Once per day
    'player_stats': 12, # Twice per day
    'team_stats': 12,
    'rosters': 24,
    'play_by_play': 24,
    'snap_counts': 24,
    'ftn_charting': 24,
    'pfr_advanced': 168  # Once per week
}


class DataRefreshManager:
    """Manages data refresh operations for the NFL backend."""

    def __init__(self):
        self.inputs_dir = settings.inputs_dir
        self.last_refresh: Dict[str, datetime] = {}
        self.refresh_in_progress: Dict[str, bool] = {}
        self._ensure_timestamps_table()
        self._load_timestamps()

    def _ensure_timestamps_table(self):
        """Create refresh_timestamps table if it doesn't exist."""
        from backend.database.local_db import get_db

        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS refresh_timestamps (
                        data_type TEXT PRIMARY KEY,
                        last_refresh TEXT,
                        record_count INTEGER DEFAULT 0
                    )
                """)
                logger.info("Refresh timestamps table ensured")
        except Exception as e:
            logger.warning(f"Could not create timestamps table: {e}")

    def _load_timestamps(self):
        """Load last refresh timestamps from SQLite."""
        from backend.database.local_db import get_db

        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT data_type, last_refresh FROM refresh_timestamps")
                rows = cursor.fetchall()

                for data_type, last_refresh_str in rows:
                    if last_refresh_str:
                        try:
                            self.last_refresh[data_type] = datetime.fromisoformat(last_refresh_str)
                        except:
                            pass

                if self.last_refresh:
                    logger.info(f"Loaded {len(self.last_refresh)} refresh timestamps from database")
        except Exception as e:
            logger.warning(f"Could not load timestamps: {e}")

    def _save_timestamp(self, data_type: str, count: int = 0):
        """Save refresh timestamp to SQLite."""
        from backend.database.local_db import get_db

        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO refresh_timestamps (data_type, last_refresh, record_count)
                    VALUES (?, ?, ?)
                """, (data_type, datetime.now().isoformat(), count))
        except Exception as e:
            logger.warning(f"Could not save timestamp for {data_type}: {e}")

    def _has_data(self, data_type: str, season: int = None) -> bool:
        """Check if data already exists in SQLite tables for current season.

        Args:
            data_type: Type of data to check
            season: Season to check for (defaults to current season)
        """
        from backend.database.local_db import get_db

        if season is None:
            now = datetime.now()
            season = now.year if now.month >= 9 else now.year - 1

        table_map = {
            'injuries': ('injuries', 'season'),
            'schedules': ('schedules', 'season'),
            'player_stats': ('player_stats', 'season'),
            'team_stats': ('team_stats', 'season'),
            'rosters': ('rosters', 'season'),
            'play_by_play': ('play_by_play', 'season'),
            'odds': ('odds_snapshots', 'season'),
            'snap_counts': ('snap_counts', 'season'),
            'ftn_charting': ('ftn_charting', 'season'),
            'pfr_advanced': ('pfr_advanced', 'season')
        }

        table_info = table_map.get(data_type)
        if not table_info:
            return False

        table_name, season_col = table_info

        try:
            with get_db() as conn:
                cursor = conn.cursor()
                # Check for data in current season
                cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {season_col} = ?", (season,))
                count = cursor.fetchone()[0]

                if count > 0:
                    logger.info(f"{data_type}: Found {count} records for {season} season")
                    return True
                return False
        except Exception as e:
            logger.debug(f"Could not check {data_type}: {e}")
            return False

    def is_stale(self, data_type: str) -> bool:
        """Check if data type is stale and needs refresh.

        Args:
            data_type: Type of data to check

        Returns:
            True if data is stale, missing, or never refreshed
        """
        # First check if data exists at all for current season
        if not self._has_data(data_type):
            logger.info(f"{data_type}: No data found for current season, needs refresh")
            return True

        # Check timestamp
        last = self.last_refresh.get(data_type)
        if not last:
            # Data exists but no timestamp - set timestamp and skip refresh
            logger.info(f"{data_type}: Data exists but no timestamp, marking as fresh")
            self._save_timestamp(data_type, 0)
            self.last_refresh[data_type] = datetime.now()
            return False

        threshold_hours = STALENESS_THRESHOLDS.get(data_type, 24)
        age_hours = (datetime.now() - last).total_seconds() / 3600

        if age_hours > threshold_hours:
            logger.info(f"{data_type}: Data is {age_hours:.1f}h old (threshold: {threshold_hours}h), needs refresh")
            return True

        logger.info(f"{data_type}: Data is fresh ({age_hours:.1f}h old), skipping")
        return False

    def download_nflverse_data(self, season: int = None) -> Dict:
        """Download nflverse data files if they don't exist.

        This downloads from GitHub releases directly, no special libraries needed.

        Args:
            season: NFL season year (defaults to current season)

        Returns:
            Dict with download results
        """
        if season is None:
            now = datetime.now()
            season = now.year if now.month >= 9 else now.year - 1

        logger.info(f"Checking for nflverse data files for {season} season...")

        # Check if key files exist
        key_files = [
            f'player_stats_{season}.csv',
            f'play_by_play_{season}.csv',
            f'weekly_rosters_{season}.csv',
            f'snap_counts_{season}.csv',
            f'schedules_{season}.csv'
        ]

        missing_files = []
        for filename in key_files:
            filepath = self.inputs_dir / filename
            if not filepath.exists():
                missing_files.append(filename)

        if not missing_files:
            logger.info(f"All nflverse data files exist for {season}")
            return {
                'status': 'skipped',
                'reason': 'files already exist',
                'season': season
            }

        logger.info(f"Missing {len(missing_files)} nflverse files, downloading...")

        try:
            # Call fetch_nflverse to download missing files
            fetch_nflverse(
                year=season,
                out_dir=self.inputs_dir,
                cache_dir=None,
                include_all=True
            )

            return {
                'status': 'success',
                'season': season,
                'downloaded': len(missing_files),
                'files': missing_files
            }

        except Exception as e:
            logger.error(f"Error downloading nflverse data: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'season': season
            }

    async def refresh_all(self, force: bool = False) -> Dict:
        """Refresh all data sources with smart staleness checking.

        Args:
            force: Force refresh even if recently updated

        Returns:
            Dict with refresh results for each data type
        """
        results = {}

        # Initialize database if needed
        init_database()
        ensure_advanced_tables()

        # Download nflverse data files if missing
        results['nflverse_download'] = self.download_nflverse_data()

        # Run all refreshes (with staleness check if not forced)
        if force or self.is_stale('injuries'):
            results['injuries'] = await self.refresh_injuries(force)
        else:
            results['injuries'] = {'status': 'skipped', 'reason': 'not stale'}

        if force or self.is_stale('schedules'):
            results['schedules'] = await self.refresh_schedules(force)
        else:
            results['schedules'] = {'status': 'skipped', 'reason': 'not stale'}

        if force or self.is_stale('player_stats'):
            results['player_stats'] = await self.refresh_player_stats(force)
        else:
            results['player_stats'] = {'status': 'skipped', 'reason': 'not stale'}

        if force or self.is_stale('team_stats'):
            results['team_stats'] = await self.refresh_team_stats(force)
        else:
            results['team_stats'] = {'status': 'skipped', 'reason': 'not stale'}

        if force or self.is_stale('rosters'):
            results['rosters'] = await self.refresh_rosters(force)
        else:
            results['rosters'] = {'status': 'skipped', 'reason': 'not stale'}

        if force or self.is_stale('play_by_play'):
            results['play_by_play'] = await self.refresh_play_by_play(force)
        else:
            results['play_by_play'] = {'status': 'skipped', 'reason': 'not stale'}

        if force or self.is_stale('odds'):
            results['odds'] = await self.refresh_odds(force)
        else:
            results['odds'] = {'status': 'skipped', 'reason': 'not stale'}

        # New advanced tables
        if force or self.is_stale('snap_counts'):
            results['snap_counts'] = await self.refresh_snap_counts(force)
        else:
            results['snap_counts'] = {'status': 'skipped', 'reason': 'not stale'}

        if force or self.is_stale('ftn_charting'):
            results['ftn_charting'] = await self.refresh_ftn_charting(force)
        else:
            results['ftn_charting'] = {'status': 'skipped', 'reason': 'not stale'}

        if force or self.is_stale('pfr_advanced'):
            results['pfr_advanced'] = await self.refresh_pfr_advanced(force)
        else:
            results['pfr_advanced'] = {'status': 'skipped', 'reason': 'not stale'}

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
            self._save_timestamp('odds', count)

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
            self._save_timestamp('injuries', count)

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
            self._save_timestamp('schedules', count)

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
            self._save_timestamp('player_stats', count)

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
            self._save_timestamp('team_stats', count)

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
            self._save_timestamp('rosters', count)

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
            self._save_timestamp('play_by_play', total_plays)

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

    async def refresh_snap_counts(self, force: bool = False) -> Dict:
        """Refresh snap count data from nflverse.

        Args:
            force: Force refresh

        Returns:
            Dict with refresh results
        """
        if self.refresh_in_progress.get('snap_counts', False):
            return {'status': 'in_progress', 'message': 'Snap counts refresh already running'}

        self.refresh_in_progress['snap_counts'] = True

        try:
            logger.info("Refreshing snap counts data...")

            # Find snap count files
            snap_files = list(self.inputs_dir.glob('snap_counts*.csv'))

            if not snap_files:
                return {
                    'status': 'warning',
                    'message': 'No snap count files found',
                    'count': 0
                }

            from backend.database.local_db import get_db

            total_records = 0
            for file in snap_files:
                try:
                    df = pd.read_csv(file)

                    with get_db() as conn:
                        cursor = conn.cursor()
                        for _, row in df.iterrows():
                            try:
                                cursor.execute("""
                                    INSERT OR REPLACE INTO snap_counts
                                    (season, week, game_id, player_id, player_name, team,
                                     position, offense_snaps, offense_pct, defense_snaps,
                                     defense_pct, st_snaps, st_pct)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    row.get('season'),
                                    row.get('week'),
                                    row.get('game_id'),
                                    row.get('player_id', row.get('pfr_player_id')),
                                    row.get('player', row.get('player_name')),
                                    row.get('team'),
                                    row.get('position'),
                                    row.get('offense_snaps'),
                                    row.get('offense_pct'),
                                    row.get('defense_snaps'),
                                    row.get('defense_pct'),
                                    row.get('st_snaps'),
                                    row.get('st_pct')
                                ))
                                total_records += 1
                            except:
                                pass
                except Exception as e:
                    logger.warning(f"Error reading snap counts file {file}: {e}")

            self.last_refresh['snap_counts'] = datetime.now()
            self._save_timestamp('snap_counts', total_records)

            return {
                'status': 'success',
                'count': total_records,
                'files_processed': len(snap_files),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error refreshing snap counts: {e}")
            return {'status': 'error', 'message': str(e)}

        finally:
            self.refresh_in_progress['snap_counts'] = False

    async def refresh_ftn_charting(self, force: bool = False) -> Dict:
        """Refresh FTN charting data (coverage schemes, formations).

        Args:
            force: Force refresh

        Returns:
            Dict with refresh results
        """
        if self.refresh_in_progress.get('ftn_charting', False):
            return {'status': 'in_progress', 'message': 'FTN charting refresh already running'}

        self.refresh_in_progress['ftn_charting'] = True

        try:
            logger.info("Refreshing FTN charting data...")

            # Find FTN charting files
            ftn_files = list(self.inputs_dir.glob('ftn_charting*.csv'))

            if not ftn_files:
                return {
                    'status': 'warning',
                    'message': 'No FTN charting files found',
                    'count': 0
                }

            from backend.database.local_db import get_db

            total_records = 0
            for file in ftn_files:
                try:
                    df = pd.read_csv(file)

                    with get_db() as conn:
                        cursor = conn.cursor()
                        for _, row in df.iterrows():
                            try:
                                cursor.execute("""
                                    INSERT OR REPLACE INTO ftn_charting
                                    (season, week, game_id, play_id, offense_formation,
                                     offense_personnel, defense_coverage, defense_man_zone,
                                     defenders_in_box, pass_rushers, blitz)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    row.get('season'),
                                    row.get('week'),
                                    row.get('nflverse_game_id', row.get('game_id')),
                                    row.get('play_id'),
                                    row.get('offense_formation'),
                                    row.get('offense_personnel'),
                                    row.get('coverage'),
                                    row.get('coverage_type', row.get('man_zone')),
                                    row.get('n_defense_box'),
                                    row.get('n_pass_rushers'),
                                    row.get('is_blitz', 0)
                                ))
                                total_records += 1
                            except:
                                pass
                except Exception as e:
                    logger.warning(f"Error reading FTN charting file {file}: {e}")

            self.last_refresh['ftn_charting'] = datetime.now()
            self._save_timestamp('ftn_charting', total_records)

            return {
                'status': 'success',
                'count': total_records,
                'files_processed': len(ftn_files),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error refreshing FTN charting: {e}")
            return {'status': 'error', 'message': str(e)}

        finally:
            self.refresh_in_progress['ftn_charting'] = False

    async def refresh_pfr_advanced(self, force: bool = False) -> Dict:
        """Refresh PFR advanced stats.

        Args:
            force: Force refresh

        Returns:
            Dict with refresh results
        """
        if self.refresh_in_progress.get('pfr_advanced', False):
            return {'status': 'in_progress', 'message': 'PFR advanced refresh already running'}

        self.refresh_in_progress['pfr_advanced'] = True

        try:
            logger.info("Refreshing PFR advanced stats...")

            # Find PFR advanced stats files
            pfr_files = list(self.inputs_dir.glob('pfr_advstats*.csv'))

            if not pfr_files:
                return {
                    'status': 'warning',
                    'message': 'No PFR advanced stats files found',
                    'count': 0
                }

            from backend.database.local_db import get_db

            total_records = 0
            for file in pfr_files:
                try:
                    df = pd.read_csv(file)

                    # Determine stat type from filename
                    stat_type = 'unknown'
                    if 'pass' in file.name:
                        stat_type = 'passing'
                    elif 'rush' in file.name:
                        stat_type = 'rushing'
                    elif 'rec' in file.name:
                        stat_type = 'receiving'
                    elif 'def' in file.name:
                        stat_type = 'defense'

                    with get_db() as conn:
                        cursor = conn.cursor()
                        for _, row in df.iterrows():
                            try:
                                cursor.execute("""
                                    INSERT OR REPLACE INTO pfr_advanced
                                    (season, player_id, player_name, team, stat_type,
                                     games, attempts, yards, tds, first_downs,
                                     ybc, yac, broken_tackles, drops, drop_pct,
                                     bad_throws, bad_throw_pct, pocket_time,
                                     blitzed, hurried, pressured_pct)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    row.get('season'),
                                    row.get('pfr_id', row.get('player_id')),
                                    row.get('player_name', row.get('player')),
                                    row.get('team'),
                                    stat_type,
                                    row.get('g', row.get('games')),
                                    row.get('att', row.get('attempts')),
                                    row.get('yds', row.get('yards')),
                                    row.get('td', row.get('tds')),
                                    row.get('first_down', row.get('first_downs')),
                                    row.get('ybc', row.get('yards_before_contact')),
                                    row.get('yac', row.get('yards_after_contact')),
                                    row.get('brk_tkl', row.get('broken_tackles')),
                                    row.get('drop', row.get('drops')),
                                    row.get('drop_pct'),
                                    row.get('bad_throw', row.get('bad_throws')),
                                    row.get('bad_pct', row.get('bad_throw_pct')),
                                    row.get('pocket_time'),
                                    row.get('blitzed'),
                                    row.get('hurried'),
                                    row.get('prss_pct', row.get('pressured_pct'))
                                ))
                                total_records += 1
                            except:
                                pass
                except Exception as e:
                    logger.warning(f"Error reading PFR file {file}: {e}")

            self.last_refresh['pfr_advanced'] = datetime.now()
            self._save_timestamp('pfr_advanced', total_records)

            return {
                'status': 'success',
                'count': total_records,
                'files_processed': len(pfr_files),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error refreshing PFR advanced: {e}")
            return {'status': 'error', 'message': str(e)}

        finally:
            self.refresh_in_progress['pfr_advanced'] = False

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
