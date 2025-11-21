"""NFL Calendar Utilities - Dynamic season and week detection.

Automatically determines the current NFL season and week based on the date.
No more hardcoded defaults!

Usage:
    from backend.utils.nfl_calendar import get_current_season, get_current_week

    season = get_current_season()  # e.g., 2025
    week = get_current_week()      # e.g., 12
"""

from datetime import datetime, date, timedelta
from typing import Tuple


# NFL Season start dates (Thursday of Week 1)
# These are approximate - the first Thursday after Labor Day
NFL_SEASON_STARTS = {
    2023: date(2023, 9, 7),
    2024: date(2024, 9, 5),
    2025: date(2025, 9, 4),
    2026: date(2026, 9, 10),
    2027: date(2027, 9, 9),
}


def get_current_season(reference_date: date = None) -> int:
    """Get the current NFL season year.

    NFL seasons span two calendar years (Sept-Feb).
    The season year is the year it starts in.

    Args:
        reference_date: Date to check (defaults to today)

    Returns:
        NFL season year (e.g., 2025)

    Examples:
        - November 21, 2025 -> 2025 (regular season)
        - February 10, 2026 -> 2025 (playoffs/Super Bowl)
        - August 15, 2025 -> 2025 (preseason)
    """
    if reference_date is None:
        reference_date = date.today()

    year = reference_date.year
    month = reference_date.month

    # If we're in January-July, we're in the previous year's season
    # (playoffs, Super Bowl, or offseason before new season starts)
    if month <= 7:
        # Check if we're still in the previous season's playoffs
        # Super Bowl is typically in early February
        if month <= 2:
            return year - 1
        else:
            # March-July: offseason, but closer to next season
            # Return the upcoming season
            return year
    else:
        # August-December: current year's season
        return year


def get_current_week(reference_date: date = None) -> int:
    """Get the current NFL week number.

    Args:
        reference_date: Date to check (defaults to today)

    Returns:
        NFL week number (1-18 for regular season, 19-22 for playoffs)
        Returns 1 if before season starts, 18 if after regular season

    Week mapping:
        - Weeks 1-18: Regular season
        - Week 19: Wild Card
        - Week 20: Divisional
        - Week 21: Conference Championships
        - Week 22: Super Bowl
    """
    if reference_date is None:
        reference_date = date.today()

    season = get_current_season(reference_date)

    # Get season start date
    if season in NFL_SEASON_STARTS:
        season_start = NFL_SEASON_STARTS[season]
    else:
        # Estimate: first Thursday after Labor Day
        # Labor Day is first Monday of September
        sept_1 = date(season, 9, 1)
        days_until_monday = (7 - sept_1.weekday()) % 7
        if sept_1.weekday() == 0:  # Already Monday
            days_until_monday = 0
        labor_day = sept_1 + timedelta(days=days_until_monday)
        # First Thursday after Labor Day
        season_start = labor_day + timedelta(days=3)

    # Calculate days since season start
    days_since_start = (reference_date - season_start).days

    if days_since_start < 0:
        # Before season starts - preseason or previous season
        if days_since_start > -30:
            return 1  # Preseason, return week 1 as upcoming
        else:
            # More than a month before - likely in previous season's playoffs
            # or deep offseason
            return 1

    # Calculate week (each week is 7 days)
    week = (days_since_start // 7) + 1

    # Cap at reasonable values
    if week > 22:
        week = 22  # Super Bowl week max
    elif week > 18:
        # Playoff weeks (19-22)
        pass

    return week


def get_current_season_and_week(reference_date: date = None) -> Tuple[int, int]:
    """Get both current season and week.

    Args:
        reference_date: Date to check (defaults to today)

    Returns:
        Tuple of (season, week)
    """
    if reference_date is None:
        reference_date = date.today()

    return get_current_season(reference_date), get_current_week(reference_date)


def get_week_dates(season: int, week: int) -> Tuple[date, date]:
    """Get the start and end dates for a specific NFL week.

    Args:
        season: NFL season year
        week: Week number

    Returns:
        Tuple of (start_date, end_date) for the week
    """
    if season in NFL_SEASON_STARTS:
        season_start = NFL_SEASON_STARTS[season]
    else:
        # Estimate
        sept_1 = date(season, 9, 1)
        days_until_monday = (7 - sept_1.weekday()) % 7
        if sept_1.weekday() == 0:
            days_until_monday = 0
        labor_day = sept_1 + timedelta(days=days_until_monday)
        season_start = labor_day + timedelta(days=3)

    # Week starts on Thursday (for TNF) but main games are Sunday
    week_start = season_start + timedelta(days=(week - 1) * 7)
    week_end = week_start + timedelta(days=6)

    return week_start, week_end


# Convenience functions for common use
def current_season() -> int:
    """Shorthand for get_current_season()."""
    return get_current_season()


def current_week() -> int:
    """Shorthand for get_current_week()."""
    return get_current_week()


if __name__ == "__main__":
    # Test the functions
    today = date.today()
    season = get_current_season()
    week = get_current_week()

    print(f"Today: {today}")
    print(f"Current NFL Season: {season}")
    print(f"Current NFL Week: {week}")

    # Show week dates
    start, end = get_week_dates(season, week)
    print(f"Week {week} dates: {start} to {end}")

    # Test some dates
    test_dates = [
        date(2025, 11, 21),  # Today
        date(2025, 9, 4),    # Week 1 start
        date(2025, 9, 10),   # Week 1
        date(2026, 2, 8),    # Super Bowl
        date(2025, 8, 15),   # Preseason
    ]

    print("\nTest dates:")
    for d in test_dates:
        s, w = get_current_season_and_week(d)
        print(f"  {d}: Season {s}, Week {w}")
