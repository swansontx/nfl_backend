"""NFL stadium database with coordinates for weather lookups.

Contains stadium location data for all 32 NFL teams including:
- GPS coordinates (latitude, longitude)
- Stadium name
- Whether it's a dome/retractable roof
- City and state
"""

from typing import Dict, Optional

# NFL stadium data (2024-2025 season)
STADIUMS: Dict[str, Dict] = {
    'ARI': {
        'name': 'State Farm Stadium',
        'city': 'Glendale',
        'state': 'AZ',
        'lat': 33.5276,
        'lon': -112.2626,
        'is_dome': True,
        'roof_type': 'retractable'
    },
    'ATL': {
        'name': 'Mercedes-Benz Stadium',
        'city': 'Atlanta',
        'state': 'GA',
        'lat': 33.7554,
        'lon': -84.4008,
        'is_dome': True,
        'roof_type': 'retractable'
    },
    'BAL': {
        'name': 'M&T Bank Stadium',
        'city': 'Baltimore',
        'state': 'MD',
        'lat': 39.2780,
        'lon': -76.6227,
        'is_dome': False,
        'roof_type': 'open'
    },
    'BUF': {
        'name': 'Highmark Stadium',
        'city': 'Orchard Park',
        'state': 'NY',
        'lat': 42.7738,
        'lon': -78.7870,
        'is_dome': False,
        'roof_type': 'open'
    },
    'CAR': {
        'name': 'Bank of America Stadium',
        'city': 'Charlotte',
        'state': 'NC',
        'lat': 35.2258,
        'lon': -80.8528,
        'is_dome': False,
        'roof_type': 'open'
    },
    'CHI': {
        'name': 'Soldier Field',
        'city': 'Chicago',
        'state': 'IL',
        'lat': 41.8623,
        'lon': -87.6167,
        'is_dome': False,
        'roof_type': 'open'
    },
    'CIN': {
        'name': 'Paycor Stadium',
        'city': 'Cincinnati',
        'state': 'OH',
        'lat': 39.0954,
        'lon': -84.5160,
        'is_dome': False,
        'roof_type': 'open'
    },
    'CLE': {
        'name': 'Cleveland Browns Stadium',
        'city': 'Cleveland',
        'state': 'OH',
        'lat': 41.5061,
        'lon': -81.6995,
        'is_dome': False,
        'roof_type': 'open'
    },
    'DAL': {
        'name': 'AT&T Stadium',
        'city': 'Arlington',
        'state': 'TX',
        'lat': 32.7473,
        'lon': -97.0945,
        'is_dome': True,
        'roof_type': 'retractable'
    },
    'DEN': {
        'name': 'Empower Field at Mile High',
        'city': 'Denver',
        'state': 'CO',
        'lat': 39.7439,
        'lon': -105.0201,
        'is_dome': False,
        'roof_type': 'open'
    },
    'DET': {
        'name': 'Ford Field',
        'city': 'Detroit',
        'state': 'MI',
        'lat': 42.3400,
        'lon': -83.0456,
        'is_dome': True,
        'roof_type': 'dome'
    },
    'GB': {
        'name': 'Lambeau Field',
        'city': 'Green Bay',
        'state': 'WI',
        'lat': 44.5013,
        'lon': -88.0622,
        'is_dome': False,
        'roof_type': 'open'
    },
    'HOU': {
        'name': 'NRG Stadium',
        'city': 'Houston',
        'state': 'TX',
        'lat': 29.6847,
        'lon': -95.4107,
        'is_dome': True,
        'roof_type': 'retractable'
    },
    'IND': {
        'name': 'Lucas Oil Stadium',
        'city': 'Indianapolis',
        'state': 'IN',
        'lat': 39.7601,
        'lon': -86.1639,
        'is_dome': True,
        'roof_type': 'retractable'
    },
    'JAX': {
        'name': 'TIAA Bank Field',
        'city': 'Jacksonville',
        'state': 'FL',
        'lat': 30.3240,
        'lon': -81.6373,
        'is_dome': False,
        'roof_type': 'open'
    },
    'KC': {
        'name': 'GEHA Field at Arrowhead Stadium',
        'city': 'Kansas City',
        'state': 'MO',
        'lat': 39.0489,
        'lon': -94.4839,
        'is_dome': False,
        'roof_type': 'open'
    },
    'LAC': {
        'name': 'SoFi Stadium',
        'city': 'Inglewood',
        'state': 'CA',
        'lat': 33.9535,
        'lon': -118.3392,
        'is_dome': False,
        'roof_type': 'open'  # Has roof but considered open-air
    },
    'LAR': {
        'name': 'SoFi Stadium',
        'city': 'Inglewood',
        'state': 'CA',
        'lat': 33.9535,
        'lon': -118.3392,
        'is_dome': False,
        'roof_type': 'open'
    },
    'LV': {
        'name': 'Allegiant Stadium',
        'city': 'Las Vegas',
        'state': 'NV',
        'lat': 36.0909,
        'lon': -115.1833,
        'is_dome': True,
        'roof_type': 'dome'
    },
    'MIA': {
        'name': 'Hard Rock Stadium',
        'city': 'Miami Gardens',
        'state': 'FL',
        'lat': 25.9580,
        'lon': -80.2389,
        'is_dome': False,
        'roof_type': 'open'
    },
    'MIN': {
        'name': 'U.S. Bank Stadium',
        'city': 'Minneapolis',
        'state': 'MN',
        'lat': 44.9738,
        'lon': -93.2575,
        'is_dome': True,
        'roof_type': 'dome'
    },
    'NE': {
        'name': 'Gillette Stadium',
        'city': 'Foxborough',
        'state': 'MA',
        'lat': 42.0909,
        'lon': -71.2643,
        'is_dome': False,
        'roof_type': 'open'
    },
    'NO': {
        'name': 'Caesars Superdome',
        'city': 'New Orleans',
        'state': 'LA',
        'lat': 29.9511,
        'lon': -90.0812,
        'is_dome': True,
        'roof_type': 'dome'
    },
    'NYG': {
        'name': 'MetLife Stadium',
        'city': 'East Rutherford',
        'state': 'NJ',
        'lat': 40.8128,
        'lon': -74.0742,
        'is_dome': False,
        'roof_type': 'open'
    },
    'NYJ': {
        'name': 'MetLife Stadium',
        'city': 'East Rutherford',
        'state': 'NJ',
        'lat': 40.8128,
        'lon': -74.0742,
        'is_dome': False,
        'roof_type': 'open'
    },
    'PHI': {
        'name': 'Lincoln Financial Field',
        'city': 'Philadelphia',
        'state': 'PA',
        'lat': 39.9008,
        'lon': -75.1675,
        'is_dome': False,
        'roof_type': 'open'
    },
    'PIT': {
        'name': 'Acrisure Stadium',
        'city': 'Pittsburgh',
        'state': 'PA',
        'lat': 40.4468,
        'lon': -80.0158,
        'is_dome': False,
        'roof_type': 'open'
    },
    'SEA': {
        'name': 'Lumen Field',
        'city': 'Seattle',
        'state': 'WA',
        'lat': 47.5952,
        'lon': -122.3316,
        'is_dome': False,
        'roof_type': 'open'
    },
    'SF': {
        'name': "Levi's Stadium",
        'city': 'Santa Clara',
        'state': 'CA',
        'lat': 37.4032,
        'lon': -121.9698,
        'is_dome': False,
        'roof_type': 'open'
    },
    'TB': {
        'name': 'Raymond James Stadium',
        'city': 'Tampa',
        'state': 'FL',
        'lat': 27.9759,
        'lon': -82.5033,
        'is_dome': False,
        'roof_type': 'open'
    },
    'TEN': {
        'name': 'Nissan Stadium',
        'city': 'Nashville',
        'state': 'TN',
        'lat': 36.1665,
        'lon': -86.7713,
        'is_dome': False,
        'roof_type': 'open'
    },
    'WAS': {
        'name': 'FedExField',
        'city': 'Landover',
        'state': 'MD',
        'lat': 38.9076,
        'lon': -76.8645,
        'is_dome': False,
        'roof_type': 'open'
    }
}


def get_stadium(team_abbr: str) -> Optional[Dict]:
    """Get stadium data for a team.

    Args:
        team_abbr: Team abbreviation (e.g., 'KC', 'BUF')

    Returns:
        Stadium data dictionary or None if not found
    """
    return STADIUMS.get(team_abbr.upper())


def get_stadium_for_game(game_id: str) -> Optional[Dict]:
    """Get stadium data for a game based on game_id.

    Args:
        game_id: Game ID in format {season}_{week}_{away}_{home}

    Returns:
        Stadium data for the home team
    """
    try:
        parts = game_id.split('_')
        if len(parts) != 4:
            return None

        # Home team is the 4th part
        home_team = parts[3]
        return get_stadium(home_team)

    except Exception:
        return None


def is_weather_relevant(team_abbr: str) -> bool:
    """Check if weather is relevant for a team's stadium.

    Args:
        team_abbr: Team abbreviation

    Returns:
        False if dome, True if open-air stadium
    """
    stadium = get_stadium(team_abbr)
    if not stadium:
        return True  # Assume weather matters if unknown

    return not stadium['is_dome']
