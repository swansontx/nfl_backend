"""NFL Schedule Loader - Load and manage game schedules.

Loads schedule data from nflverse and provides APIs for:
- Team schedules (all games for a team)
- Week schedules (all games in a week)
- Game lookups by ID
- Upcoming games
"""

from pathlib import Path
from typing import Dict, List, Optional
import csv
from dataclasses import dataclass
from datetime import datetime
from backend.api.cache import cached, CACHE_TTL


@dataclass
class Game:
    """NFL Game data."""
    game_id: str
    season: int
    week: int
    game_type: str  # REG, POST, PRE
    gameday: str  # YYYY-MM-DD
    weekday: str
    gametime: str  # HH:MM
    away_team: str
    home_team: str
    away_score: Optional[int] = None
    home_score: Optional[int] = None
    result: Optional[int] = None  # 1 = home win, 0 = tie, -1 = away win
    total: Optional[int] = None
    overtime: int = 0
    old_game_id: Optional[str] = None
    gsis_id: Optional[str] = None
    espn_game_id: Optional[str] = None
    pfr_game_id: Optional[str] = None
    location: Optional[str] = None
    roof: Optional[str] = None
    surface: Optional[str] = None
    temp: Optional[int] = None
    wind: Optional[int] = None
    home_coach: Optional[str] = None
    away_coach: Optional[str] = None
    stadium: Optional[str] = None


class ScheduleLoader:
    """Load and manage NFL game schedules."""

    def __init__(self, schedules_dir: Path = Path('inputs')):
        """Initialize schedule loader.

        Args:
            schedules_dir: Directory containing schedule CSV files
        """
        self.schedules_dir = schedules_dir
        self._schedule_cache: Dict[int, List[Game]] = {}

    @cached(ttl_seconds=CACHE_TTL['schedule'])  # 1 hour
    def load_schedule(self, year: int) -> List[Game]:
        """Load schedule for a season.

        Args:
            year: NFL season year

        Returns:
            List of Game objects

        Expected file: inputs/schedule_{year}.csv
        """
        if year in self._schedule_cache:
            return self._schedule_cache[year]

        schedule_file = self.schedules_dir / f'schedule_{year}.csv'

        if not schedule_file.exists():
            print(f"⚠ Schedule file not found: {schedule_file}")
            return []

        games = []

        try:
            with open(schedule_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    # Parse required fields
                    game_id = row.get('game_id', '')
                    season = int(row.get('season', year))
                    week = int(row.get('week', 0))
                    game_type = row.get('game_type', 'REG')
                    gameday = row.get('gameday', '')
                    weekday = row.get('weekday', '')
                    gametime = row.get('gametime', '')
                    away_team = row.get('away_team', '')
                    home_team = row.get('home_team', '')

                    # Parse optional fields
                    away_score = int(row.get('away_score')) if row.get('away_score') else None
                    home_score = int(row.get('home_score')) if row.get('home_score') else None
                    result = int(row.get('result')) if row.get('result') else None
                    total = int(row.get('total')) if row.get('total') else None
                    overtime = int(row.get('overtime', 0))

                    game = Game(
                        game_id=game_id,
                        season=season,
                        week=week,
                        game_type=game_type,
                        gameday=gameday,
                        weekday=weekday,
                        gametime=gametime,
                        away_team=away_team,
                        home_team=home_team,
                        away_score=away_score,
                        home_score=home_score,
                        result=result,
                        total=total,
                        overtime=overtime,
                        old_game_id=row.get('old_game_id'),
                        gsis_id=row.get('gsis'),
                        espn_game_id=row.get('espn'),
                        pfr_game_id=row.get('pfr'),
                        location=row.get('location'),
                        roof=row.get('roof'),
                        surface=row.get('surface'),
                        temp=int(row.get('temp')) if row.get('temp') else None,
                        wind=int(row.get('wind')) if row.get('wind') else None,
                        home_coach=row.get('home_coach'),
                        away_coach=row.get('away_coach'),
                        stadium=row.get('stadium')
                    )

                    games.append(game)

            self._schedule_cache[year] = games
            print(f"✓ Loaded {len(games)} games for {year} season")
            return games

        except Exception as e:
            print(f"✗ Error loading schedule: {e}")
            return []

    def get_team_schedule(self, team: str, year: int, game_type: str = 'REG') -> List[Game]:
        """Get all games for a team in a season.

        Args:
            team: Team abbreviation
            year: Season year
            game_type: Game type filter ('REG', 'POST', 'PRE', or 'ALL')

        Returns:
            List of Game objects for this team
        """
        all_games = self.load_schedule(year)
        team = team.upper()

        team_games = []
        for game in all_games:
            # Check if team is in this game
            if game.away_team == team or game.home_team == team:
                # Filter by game type if not ALL
                if game_type == 'ALL' or game.game_type == game_type:
                    team_games.append(game)

        # Sort by week
        team_games.sort(key=lambda g: g.week)
        return team_games

    def get_week_schedule(self, year: int, week: int, game_type: str = 'REG') -> List[Game]:
        """Get all games in a specific week.

        Args:
            year: Season year
            week: Week number
            game_type: Game type ('REG', 'POST', 'PRE')

        Returns:
            List of Game objects for this week
        """
        all_games = self.load_schedule(year)

        week_games = [
            game for game in all_games
            if game.week == week and game.game_type == game_type
        ]

        # Sort by gameday and gametime
        week_games.sort(key=lambda g: (g.gameday, g.gametime))
        return week_games

    def get_game(self, game_id: str) -> Optional[Game]:
        """Get a specific game by ID.

        Args:
            game_id: Game ID (format: YYYY_WW_AWAY_HOME)

        Returns:
            Game object or None if not found
        """
        # Parse year from game_id
        try:
            parts = game_id.split('_')
            if len(parts) >= 1:
                year = int(parts[0])
                all_games = self.load_schedule(year)

                for game in all_games:
                    if game.game_id == game_id:
                        return game
        except:
            pass

        return None

    def get_upcoming_games(self, team: Optional[str] = None, limit: int = 10) -> List[Game]:
        """Get upcoming games.

        Args:
            team: Optional team filter
            limit: Max number of games to return

        Returns:
            List of upcoming Game objects
        """
        # Get current year and next year schedules
        current_year = datetime.now().year
        all_games = []

        for year in [current_year, current_year + 1]:
            all_games.extend(self.load_schedule(year))

        # Filter to games without scores (not played yet)
        upcoming = [game for game in all_games if game.away_score is None]

        # Filter by team if specified
        if team:
            team = team.upper()
            upcoming = [
                game for game in upcoming
                if game.away_team == team or game.home_team == team
            ]

        # Sort by gameday
        upcoming.sort(key=lambda g: (g.gameday, g.gametime))

        return upcoming[:limit]

    def get_completed_games(self, team: Optional[str] = None, year: Optional[int] = None, limit: int = 10) -> List[Game]:
        """Get recently completed games.

        Args:
            team: Optional team filter
            year: Optional year filter (defaults to current year)
            limit: Max number of games to return

        Returns:
            List of completed Game objects
        """
        if year is None:
            year = datetime.now().year

        all_games = self.load_schedule(year)

        # Filter to games with scores (played)
        completed = [game for game in all_games if game.away_score is not None]

        # Filter by team if specified
        if team:
            team = team.upper()
            completed = [
                game for game in completed
                if game.away_team == team or game.home_team == team
            ]

        # Sort by gameday (most recent first)
        completed.sort(key=lambda g: (g.gameday, g.gametime), reverse=True)

        return completed[:limit]


# Singleton instance
schedule_loader = ScheduleLoader()
