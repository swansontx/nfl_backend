"""NFL Season and Week Calculator.

Automatically determines the current NFL season and week based on the current date.
"""

from datetime import datetime, timedelta
from typing import Tuple

# NFL season start dates (Tuesday after Labor Day, first Sunday)
# These are the dates of Week 1 games
NFL_SEASON_STARTS = {
    2023: datetime(2023, 9, 7),   # Thursday opener
    2024: datetime(2024, 9, 5),   # Thursday opener
    2025: datetime(2025, 9, 4),   # Thursday opener (projected)
    2026: datetime(2026, 9, 10),  # Thursday opener (projected)
}

# Regular season is 18 weeks (17 games + 1 bye)
REGULAR_SEASON_WEEKS = 18


def get_current_nfl_season() -> int:
    """Get the current NFL season year.

    NFL season spans two calendar years (Sept-Feb).
    Returns the year the season started in.

    Returns:
        Season year (e.g., 2025 for 2025-2026 season)
    """
    now = datetime.now()
    year = now.year
    month = now.month

    # If we're in Jan-Aug, we're in the previous year's season
    # (playoffs/offseason from last year's season)
    if month < 9:
        # Check if we're before August - definitely previous season
        if month < 8:
            return year - 1
        # August - check if season has started
        if year in NFL_SEASON_STARTS:
            if now < NFL_SEASON_STARTS[year]:
                return year - 1
        else:
            # Default: assume season starts in September
            return year - 1

    # September onwards - current year's season
    return year


def get_current_nfl_week() -> int:
    """Get the current NFL week number.

    Returns:
        Week number (1-18 for regular season, 19+ for playoffs)
    """
    now = datetime.now()
    season = get_current_nfl_season()

    # Get season start date
    if season in NFL_SEASON_STARTS:
        season_start = NFL_SEASON_STARTS[season]
    else:
        # Default to first Thursday of September
        season_start = datetime(season, 9, 5)

    # If before season start, return week 1
    if now < season_start:
        return 1

    # Calculate weeks elapsed
    days_elapsed = (now - season_start).days
    week = (days_elapsed // 7) + 1

    # Cap at reasonable max (postseason goes to ~22)
    return min(week, 22)


def get_current_season_and_week() -> Tuple[int, int]:
    """Get both current season and week.

    Returns:
        Tuple of (season_year, week_number)
    """
    return get_current_nfl_season(), get_current_nfl_week()


def get_week_dates(season: int, week: int) -> Tuple[datetime, datetime]:
    """Get the start and end dates for a specific week.

    Args:
        season: NFL season year
        week: Week number

    Returns:
        Tuple of (week_start, week_end) datetimes
    """
    if season in NFL_SEASON_STARTS:
        season_start = NFL_SEASON_STARTS[season]
    else:
        season_start = datetime(season, 9, 5)

    # Week starts on Tuesday (for betting purposes)
    week_start = season_start + timedelta(days=(week - 1) * 7 - 2)
    week_end = week_start + timedelta(days=7)

    return week_start, week_end


def is_regular_season(week: int) -> bool:
    """Check if week is in regular season.

    Args:
        week: Week number

    Returns:
        True if regular season week
    """
    return 1 <= week <= REGULAR_SEASON_WEEKS


def format_game_id(season: int, week: int, away_team: str, home_team: str) -> str:
    """Format a standard game ID.

    Args:
        season: NFL season year
        week: Week number
        away_team: Away team abbreviation
        home_team: Home team abbreviation

    Returns:
        Game ID string (e.g., '2025_12_BUF_MIA')
    """
    return f"{season}_{week}_{away_team}_{home_team}"


if __name__ == "__main__":
    # Test the functions
    season, week = get_current_season_and_week()
    print(f"Current NFL Season: {season}")
    print(f"Current NFL Week: {week}")
    print(f"Is Regular Season: {is_regular_season(week)}")

    start, end = get_week_dates(season, week)
    print(f"Week {week} dates: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
