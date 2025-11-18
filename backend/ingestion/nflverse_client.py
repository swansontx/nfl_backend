"""
NFLverse data client for fetching play-by-play, player stats, and rosters

Data sources:
- https://github.com/nflverse/nflverse-data/releases
- Play-by-play: https://github.com/nflverse/nflverse-data/releases/download/pbp/
- Player stats: https://github.com/nflverse/nflverse-data/releases/download/player_stats/
- Rosters: https://github.com/nflverse/nflverse-data/releases/download/rosters/
- Injuries: https://github.com/nflverse/nflverse-data/releases/download/injuries/
"""

import pandas as pd
import httpx
from typing import Optional, List, Dict
from datetime import datetime
from pathlib import Path

from backend.config import settings
from backend.config.logging_config import get_logger

logger = get_logger(__name__)


class NFLverseClient:
    """Client for fetching NFL data from nflverse"""

    BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download"

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or settings.data_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client(timeout=60.0)

    def get_play_by_play(self, season: int, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch play-by-play data for a season

        Args:
            season: NFL season year (e.g., 2024)
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with play-by-play data
        """
        cache_file = self.cache_dir / f"pbp_{season}.parquet"

        if use_cache and cache_file.exists():
            logger.info("loading_pbp_from_cache", season=season)
            return pd.read_parquet(cache_file)

        url = f"{self.BASE_URL}/pbp/play_by_play_{season}.parquet"
        logger.info("fetching_pbp", season=season, url=url)

        try:
            response = self.client.get(url)
            response.raise_for_status()

            # Write to cache
            cache_file.write_bytes(response.content)

            # Load and return
            df = pd.read_parquet(cache_file)
            logger.info("pbp_loaded", season=season, rows=len(df))

            return df

        except Exception as e:
            logger.error("pbp_fetch_failed", season=season, error=str(e))
            raise

    def get_player_stats(self, season: int, stat_type: str = "offense", use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch player stats for a season

        Args:
            season: NFL season year
            stat_type: Type of stats ('offense', 'defense', 'kicking')
            use_cache: Whether to use cached data

        Returns:
            DataFrame with player stats
        """
        cache_file = self.cache_dir / f"player_stats_{stat_type}_{season}.parquet"

        if use_cache and cache_file.exists():
            logger.info("loading_stats_from_cache", season=season, stat_type=stat_type)
            return pd.read_parquet(cache_file)

        url = f"{self.BASE_URL}/player_stats/player_stats_{season}.parquet"
        logger.info("fetching_player_stats", season=season, url=url)

        try:
            response = self.client.get(url)
            response.raise_for_status()

            # Write to cache
            cache_file.write_bytes(response.content)

            df = pd.read_parquet(cache_file)
            logger.info("player_stats_loaded", season=season, rows=len(df))

            return df

        except Exception as e:
            logger.error("player_stats_fetch_failed", season=season, error=str(e))
            raise

    def get_weekly_rosters(self, season: int, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch weekly roster data

        Args:
            season: NFL season year
            use_cache: Whether to use cached data

        Returns:
            DataFrame with weekly roster data
        """
        cache_file = self.cache_dir / f"rosters_{season}.parquet"

        if use_cache and cache_file.exists():
            logger.info("loading_rosters_from_cache", season=season)
            return pd.read_parquet(cache_file)

        url = f"{self.BASE_URL}/rosters/roster_{season}.parquet"
        logger.info("fetching_rosters", season=season, url=url)

        try:
            response = self.client.get(url)
            response.raise_for_status()

            cache_file.write_bytes(response.content)
            df = pd.read_parquet(cache_file)
            logger.info("rosters_loaded", season=season, rows=len(df))

            return df

        except Exception as e:
            logger.error("rosters_fetch_failed", season=season, error=str(e))
            raise

    def get_injuries(self, season: int, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch injury report data

        Args:
            season: NFL season year
            use_cache: Whether to use cached data

        Returns:
            DataFrame with injury data
        """
        cache_file = self.cache_dir / f"injuries_{season}.parquet"

        if use_cache and cache_file.exists():
            logger.info("loading_injuries_from_cache", season=season)
            return pd.read_parquet(cache_file)

        url = f"{self.BASE_URL}/injuries/injuries_{season}.parquet"
        logger.info("fetching_injuries", season=season, url=url)

        try:
            response = self.client.get(url)
            response.raise_for_status()

            cache_file.write_bytes(response.content)
            df = pd.read_parquet(cache_file)
            logger.info("injuries_loaded", season=season, rows=len(df))

            return df

        except Exception as e:
            logger.error("injuries_fetch_failed", season=season, error=str(e))
            raise

    def get_depth_charts(self, season: int, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch depth chart data

        Args:
            season: NFL season year
            use_cache: Whether to use cached data

        Returns:
            DataFrame with depth chart data
        """
        cache_file = self.cache_dir / f"depth_charts_{season}.parquet"

        if use_cache and cache_file.exists():
            logger.info("loading_depth_charts_from_cache", season=season)
            return pd.read_parquet(cache_file)

        url = f"{self.BASE_URL}/depth_charts/depth_charts_{season}.parquet"
        logger.info("fetching_depth_charts", season=season, url=url)

        try:
            response = self.client.get(url)
            response.raise_for_status()

            cache_file.write_bytes(response.content)
            df = pd.read_parquet(cache_file)
            logger.info("depth_charts_loaded", season=season, rows=len(df))

            return df

        except Exception as e:
            logger.error("depth_charts_fetch_failed", season=season, error=str(e))
            raise

    def extract_player_game_stats(
        self,
        pbp_df: pd.DataFrame,
        player_id: str,
        game_id: str
    ) -> Dict:
        """
        Extract detailed stats for a player in a specific game from PBP data

        Args:
            pbp_df: Play-by-play DataFrame
            player_id: Player's gsis_id
            game_id: Game ID

        Returns:
            Dict with detailed game stats
        """
        game_plays = pbp_df[pbp_df['game_id'] == game_id].copy()

        stats = {
            'player_id': player_id,
            'game_id': game_id,

            # Passing
            'completions': 0,
            'pass_attempts': 0,
            'passing_yards': 0,
            'passing_tds': 0,
            'interceptions': 0,
            'pass_longest': 0,

            # Rushing
            'rush_attempts': 0,
            'rushing_yards': 0,
            'rushing_tds': 0,
            'rush_longest': 0,

            # Receiving
            'targets': 0,
            'receptions': 0,
            'receiving_yards': 0,
            'receiving_tds': 0,
            'rec_longest': 0,

            # Advanced metrics
            'routes_run': 0,
            'redzone_targets': 0,
            'redzone_carries': 0,
            'goal_line_carries': 0,
            'air_yards': 0,
            'yards_after_catch': 0,
        }

        # Passing stats
        pass_plays = game_plays[game_plays['passer_player_id'] == player_id]
        if not pass_plays.empty:
            stats['completions'] = pass_plays['complete_pass'].sum()
            stats['pass_attempts'] = pass_plays['pass_attempt'].sum()
            stats['passing_yards'] = pass_plays['passing_yards'].sum()
            stats['passing_tds'] = pass_plays['pass_touchdown'].sum()
            stats['interceptions'] = pass_plays['interception'].sum()
            completed = pass_plays[pass_plays['complete_pass'] == 1]
            if not completed.empty:
                stats['pass_longest'] = completed['passing_yards'].max()

        # Rushing stats
        rush_plays = game_plays[game_plays['rusher_player_id'] == player_id]
        if not rush_plays.empty:
            stats['rush_attempts'] = len(rush_plays)
            stats['rushing_yards'] = rush_plays['rushing_yards'].sum()
            stats['rushing_tds'] = rush_plays['rush_touchdown'].sum()
            stats['rush_longest'] = rush_plays['rushing_yards'].max()

            # Redzone/goal line
            stats['redzone_carries'] = rush_plays[rush_plays['yardline_100'] <= 20].shape[0]
            stats['goal_line_carries'] = rush_plays[rush_plays['yardline_100'] <= 5].shape[0]

        # Receiving stats
        rec_plays = game_plays[game_plays['receiver_player_id'] == player_id]
        if not rec_plays.empty:
            stats['targets'] = len(rec_plays)
            stats['receptions'] = rec_plays['complete_pass'].sum()
            stats['receiving_yards'] = rec_plays['receiving_yards'].sum()
            stats['receiving_tds'] = rec_plays['pass_touchdown'].sum()

            completed_rec = rec_plays[rec_plays['complete_pass'] == 1]
            if not completed_rec.empty:
                stats['rec_longest'] = completed_rec['receiving_yards'].max()
                stats['yards_after_catch'] = completed_rec['yards_after_catch'].sum()

            stats['air_yards'] = rec_plays['air_yards'].sum()
            stats['redzone_targets'] = rec_plays[rec_plays['yardline_100'] <= 20].shape[0]

            # Routes run (approximate from targets + non-targeted pass plays)
            # This is approximate - true routes require participation data
            stats['routes_run'] = stats['targets']  # Minimum

        logger.debug("player_game_stats_extracted", **stats)

        return stats

    def get_player_historical_stats(
        self,
        player_id: str,
        n_games: int = 10,
        season: Optional[int] = None
    ) -> List[Dict]:
        """
        Get historical game-by-game stats for a player

        Args:
            player_id: Player's gsis_id
            n_games: Number of recent games to fetch
            season: Season to fetch from (None = current)

        Returns:
            List of game stat dicts
        """
        if season is None:
            season = datetime.now().year

        pbp = self.get_play_by_play(season)

        # Get all games this player appeared in
        player_games = pbp[
            (pbp['passer_player_id'] == player_id) |
            (pbp['rusher_player_id'] == player_id) |
            (pbp['receiver_player_id'] == player_id)
        ]['game_id'].unique()

        # Sort by game date (game_id format: YYYY_WW_AWAY_HOME)
        player_games = sorted(player_games, reverse=True)[:n_games]

        stats_list = []
        for game_id in player_games:
            stats = self.extract_player_game_stats(pbp, player_id, game_id)
            stats_list.append(stats)

        logger.info(
            "player_historical_stats_fetched",
            player_id=player_id,
            games=len(stats_list)
        )

        return stats_list

    def close(self):
        """Close the HTTP client"""
        self.client.close()
