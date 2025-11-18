"""Date and NFL calendar utilities"""
from datetime import datetime
from typing import Optional
import pytz


def parse_nfl_date(date_str: str) -> datetime:
    """Parse various NFL date formats to datetime"""
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%m/%d/%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    raise ValueError(f"Unable to parse date: {date_str}")


def get_season_from_date(date: datetime) -> int:
    """
    Determine NFL season from date.
    NFL season runs Sept-Feb, with Feb games counting as previous calendar year's season
    """
    if date.month >= 9:  # Sept-Dec
        return date.year
    elif date.month <= 2:  # Jan-Feb (playoffs)
        return date.year - 1
    elif date.month >= 3 and date.month <= 8:  # Offseason
        return date.year  # Return current year for offseason
    else:
        return date.year


def get_week_from_date(date: datetime, season: int) -> Optional[int]:
    """
    Estimate NFL week from date and season.
    This is approximate - better to use official schedule data.
    """
    # NFL regular season typically starts first Thursday after Labor Day (first Mon in Sept)
    # Week 1 is usually ~Sept 5-10
    season_start = datetime(season, 9, 1)

    # Find first Thursday
    while season_start.weekday() != 3:  # Thursday
        season_start = season_start.replace(day=season_start.day + 1)

    delta = (date - season_start).days
    week = (delta // 7) + 1

    if week < 1:
        return None
    if week > 18:
        # Playoffs
        return week

    return week
