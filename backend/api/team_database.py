"""NFL Team Database - Comprehensive team information.

Provides detailed information about all 32 NFL teams including:
- Team metadata (colors, logos, established year)
- Division/conference info
- Stadium info (integrated with stadium_database)
- Historical records
- Current roster links
"""

from typing import Dict, Optional, List
from backend.api.stadium_database import STADIUMS


# NFL Team comprehensive database
TEAMS = {
    # AFC East
    'BUF': {
        'team_id': 'BUF',
        'name': 'Buffalo Bills',
        'city': 'Buffalo',
        'state': 'NY',
        'abbreviation': 'BUF',
        'conference': 'AFC',
        'division': 'East',
        'established': 1960,
        'colors': {
            'primary': '#00338D',  # Royal blue
            'secondary': '#C60C30',  # Red
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/buf.png',
        'espn_id': 'buf',
        'super_bowl_wins': 0,
        'super_bowl_appearances': 4,
        'playoff_appearances': 23,
    },
    'MIA': {
        'team_id': 'MIA',
        'name': 'Miami Dolphins',
        'city': 'Miami',
        'state': 'FL',
        'abbreviation': 'MIA',
        'conference': 'AFC',
        'division': 'East',
        'established': 1966,
        'colors': {
            'primary': '#008E97',  # Aqua
            'secondary': '#FC4C02',  # Orange
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/mia.png',
        'espn_id': 'mia',
        'super_bowl_wins': 2,
        'super_bowl_appearances': 5,
        'playoff_appearances': 23,
    },
    'NE': {
        'team_id': 'NE',
        'name': 'New England Patriots',
        'city': 'Foxborough',
        'state': 'MA',
        'abbreviation': 'NE',
        'conference': 'AFC',
        'division': 'East',
        'established': 1960,
        'colors': {
            'primary': '#002244',  # Navy
            'secondary': '#C60C30',  # Red
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/ne.png',
        'espn_id': 'ne',
        'super_bowl_wins': 6,
        'super_bowl_appearances': 11,
        'playoff_appearances': 29,
    },
    'NYJ': {
        'team_id': 'NYJ',
        'name': 'New York Jets',
        'city': 'East Rutherford',
        'state': 'NJ',
        'abbreviation': 'NYJ',
        'conference': 'AFC',
        'division': 'East',
        'established': 1960,
        'colors': {
            'primary': '#125740',  # Green
            'secondary': '#FFFFFF',  # White
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/nyj.png',
        'espn_id': 'nyj',
        'super_bowl_wins': 1,
        'super_bowl_appearances': 1,
        'playoff_appearances': 14,
    },

    # AFC North
    'BAL': {
        'team_id': 'BAL',
        'name': 'Baltimore Ravens',
        'city': 'Baltimore',
        'state': 'MD',
        'abbreviation': 'BAL',
        'conference': 'AFC',
        'division': 'North',
        'established': 1996,
        'colors': {
            'primary': '#241773',  # Purple
            'secondary': '#000000',  # Black
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/bal.png',
        'espn_id': 'bal',
        'super_bowl_wins': 2,
        'super_bowl_appearances': 2,
        'playoff_appearances': 15,
    },
    'CIN': {
        'team_id': 'CIN',
        'name': 'Cincinnati Bengals',
        'city': 'Cincinnati',
        'state': 'OH',
        'abbreviation': 'CIN',
        'conference': 'AFC',
        'division': 'North',
        'established': 1968,
        'colors': {
            'primary': '#FB4F14',  # Orange
            'secondary': '#000000',  # Black
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/cin.png',
        'espn_id': 'cin',
        'super_bowl_wins': 0,
        'super_bowl_appearances': 3,
        'playoff_appearances': 15,
    },
    'CLE': {
        'team_id': 'CLE',
        'name': 'Cleveland Browns',
        'city': 'Cleveland',
        'state': 'OH',
        'abbreviation': 'CLE',
        'conference': 'AFC',
        'division': 'North',
        'established': 1946,
        'colors': {
            'primary': '#311D00',  # Brown
            'secondary': '#FF3C00',  # Orange
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/cle.png',
        'espn_id': 'cle',
        'super_bowl_wins': 0,
        'super_bowl_appearances': 0,
        'playoff_appearances': 17,
    },
    'PIT': {
        'team_id': 'PIT',
        'name': 'Pittsburgh Steelers',
        'city': 'Pittsburgh',
        'state': 'PA',
        'abbreviation': 'PIT',
        'conference': 'AFC',
        'division': 'North',
        'established': 1933,
        'colors': {
            'primary': '#FFB612',  # Gold
            'secondary': '#000000',  # Black
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/pit.png',
        'espn_id': 'pit',
        'super_bowl_wins': 6,
        'super_bowl_appearances': 8,
        'playoff_appearances': 33,
    },

    # AFC South
    'HOU': {
        'team_id': 'HOU',
        'name': 'Houston Texans',
        'city': 'Houston',
        'state': 'TX',
        'abbreviation': 'HOU',
        'conference': 'AFC',
        'division': 'South',
        'established': 2002,
        'colors': {
            'primary': '#03202F',  # Deep steel blue
            'secondary': '#A71930',  # Battle red
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/hou.png',
        'espn_id': 'hou',
        'super_bowl_wins': 0,
        'super_bowl_appearances': 0,
        'playoff_appearances': 6,
    },
    'IND': {
        'team_id': 'IND',
        'name': 'Indianapolis Colts',
        'city': 'Indianapolis',
        'state': 'IN',
        'abbreviation': 'IND',
        'conference': 'AFC',
        'division': 'South',
        'established': 1953,
        'colors': {
            'primary': '#002C5F',  # Royal blue
            'secondary': '#FFFFFF',  # White
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/ind.png',
        'espn_id': 'ind',
        'super_bowl_wins': 2,
        'super_bowl_appearances': 4,
        'playoff_appearances': 27,
    },
    'JAX': {
        'team_id': 'JAX',
        'name': 'Jacksonville Jaguars',
        'city': 'Jacksonville',
        'state': 'FL',
        'abbreviation': 'JAX',
        'conference': 'AFC',
        'division': 'South',
        'established': 1995,
        'colors': {
            'primary': '#006778',  # Teal
            'secondary': '#D7A22A',  # Gold
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/jax.png',
        'espn_id': 'jax',
        'super_bowl_wins': 0,
        'super_bowl_appearances': 0,
        'playoff_appearances': 7,
    },
    'TEN': {
        'team_id': 'TEN',
        'name': 'Tennessee Titans',
        'city': 'Nashville',
        'state': 'TN',
        'abbreviation': 'TEN',
        'conference': 'AFC',
        'division': 'South',
        'established': 1960,
        'colors': {
            'primary': '#0C2340',  # Navy
            'secondary': '#4B92DB',  # Titans blue
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/ten.png',
        'espn_id': 'ten',
        'super_bowl_wins': 0,
        'super_bowl_appearances': 1,
        'playoff_appearances': 22,
    },

    # AFC West
    'DEN': {
        'team_id': 'DEN',
        'name': 'Denver Broncos',
        'city': 'Denver',
        'state': 'CO',
        'abbreviation': 'DEN',
        'conference': 'AFC',
        'division': 'West',
        'established': 1960,
        'colors': {
            'primary': '#FB4F14',  # Orange
            'secondary': '#002244',  # Navy
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/den.png',
        'espn_id': 'den',
        'super_bowl_wins': 3,
        'super_bowl_appearances': 8,
        'playoff_appearances': 22,
    },
    'KC': {
        'team_id': 'KC',
        'name': 'Kansas City Chiefs',
        'city': 'Kansas City',
        'state': 'MO',
        'abbreviation': 'KC',
        'conference': 'AFC',
        'division': 'West',
        'established': 1960,
        'colors': {
            'primary': '#E31837',  # Red
            'secondary': '#FFB81C',  # Gold
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/kc.png',
        'espn_id': 'kc',
        'super_bowl_wins': 3,
        'super_bowl_appearances': 4,
        'playoff_appearances': 25,
    },
    'LAC': {
        'team_id': 'LAC',
        'name': 'Los Angeles Chargers',
        'city': 'Inglewood',
        'state': 'CA',
        'abbreviation': 'LAC',
        'conference': 'AFC',
        'division': 'West',
        'established': 1960,
        'colors': {
            'primary': '#0080C6',  # Powder blue
            'secondary': '#FFC20E',  # Gold
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/lac.png',
        'espn_id': 'lac',
        'super_bowl_wins': 0,
        'super_bowl_appearances': 0,
        'playoff_appearances': 19,
    },
    'LV': {
        'team_id': 'LV',
        'name': 'Las Vegas Raiders',
        'city': 'Las Vegas',
        'state': 'NV',
        'abbreviation': 'LV',
        'conference': 'AFC',
        'division': 'West',
        'established': 1960,
        'colors': {
            'primary': '#000000',  # Black
            'secondary': '#A5ACAF',  # Silver
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/lv.png',
        'espn_id': 'lv',
        'super_bowl_wins': 3,
        'super_bowl_appearances': 5,
        'playoff_appearances': 22,
    },

    # NFC East
    'DAL': {
        'team_id': 'DAL',
        'name': 'Dallas Cowboys',
        'city': 'Arlington',
        'state': 'TX',
        'abbreviation': 'DAL',
        'conference': 'NFC',
        'division': 'East',
        'established': 1960,
        'colors': {
            'primary': '#041E42',  # Navy
            'secondary': '#869397',  # Silver
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/dal.png',
        'espn_id': 'dal',
        'super_bowl_wins': 5,
        'super_bowl_appearances': 8,
        'playoff_appearances': 35,
    },
    'NYG': {
        'team_id': 'NYG',
        'name': 'New York Giants',
        'city': 'East Rutherford',
        'state': 'NJ',
        'abbreviation': 'NYG',
        'conference': 'NFC',
        'division': 'East',
        'established': 1925,
        'colors': {
            'primary': '#0B2265',  # Blue
            'secondary': '#A71930',  # Red
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/nyg.png',
        'espn_id': 'nyg',
        'super_bowl_wins': 4,
        'super_bowl_appearances': 5,
        'playoff_appearances': 33,
    },
    'PHI': {
        'team_id': 'PHI',
        'name': 'Philadelphia Eagles',
        'city': 'Philadelphia',
        'state': 'PA',
        'abbreviation': 'PHI',
        'conference': 'NFC',
        'division': 'East',
        'established': 1933,
        'colors': {
            'primary': '#004C54',  # Midnight green
            'secondary': '#A5ACAF',  # Silver
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/phi.png',
        'espn_id': 'phi',
        'super_bowl_wins': 1,
        'super_bowl_appearances': 3,
        'playoff_appearances': 28,
    },
    'WAS': {
        'team_id': 'WAS',
        'name': 'Washington Commanders',
        'city': 'Landover',
        'state': 'MD',
        'abbreviation': 'WAS',
        'conference': 'NFC',
        'division': 'East',
        'established': 1932,
        'colors': {
            'primary': '#5A1414',  # Burgundy
            'secondary': '#FFB612',  # Gold
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/wsh.png',
        'espn_id': 'wsh',
        'super_bowl_wins': 3,
        'super_bowl_appearances': 5,
        'playoff_appearances': 24,
    },

    # NFC North
    'CHI': {
        'team_id': 'CHI',
        'name': 'Chicago Bears',
        'city': 'Chicago',
        'state': 'IL',
        'abbreviation': 'CHI',
        'conference': 'NFC',
        'division': 'North',
        'established': 1920,
        'colors': {
            'primary': '#0B162A',  # Navy
            'secondary': '#C83803',  # Orange
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/chi.png',
        'espn_id': 'chi',
        'super_bowl_wins': 1,
        'super_bowl_appearances': 2,
        'playoff_appearances': 27,
    },
    'DET': {
        'team_id': 'DET',
        'name': 'Detroit Lions',
        'city': 'Detroit',
        'state': 'MI',
        'abbreviation': 'DET',
        'conference': 'NFC',
        'division': 'North',
        'established': 1930,
        'colors': {
            'primary': '#0076B6',  # Honolulu blue
            'secondary': '#B0B7BC',  # Silver
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/det.png',
        'espn_id': 'det',
        'super_bowl_wins': 0,
        'super_bowl_appearances': 0,
        'playoff_appearances': 17,
    },
    'GB': {
        'team_id': 'GB',
        'name': 'Green Bay Packers',
        'city': 'Green Bay',
        'state': 'WI',
        'abbreviation': 'GB',
        'conference': 'NFC',
        'division': 'North',
        'established': 1921,
        'colors': {
            'primary': '#203731',  # Dark green
            'secondary': '#FFB612',  # Gold
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/gb.png',
        'espn_id': 'gb',
        'super_bowl_wins': 4,
        'super_bowl_appearances': 5,
        'playoff_appearances': 36,
    },
    'MIN': {
        'team_id': 'MIN',
        'name': 'Minnesota Vikings',
        'city': 'Minneapolis',
        'state': 'MN',
        'abbreviation': 'MIN',
        'conference': 'NFC',
        'division': 'North',
        'established': 1961,
        'colors': {
            'primary': '#4F2683',  # Purple
            'secondary': '#FFC62F',  # Gold
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/min.png',
        'espn_id': 'min',
        'super_bowl_wins': 0,
        'super_bowl_appearances': 4,
        'playoff_appearances': 30,
    },

    # NFC South
    'ATL': {
        'team_id': 'ATL',
        'name': 'Atlanta Falcons',
        'city': 'Atlanta',
        'state': 'GA',
        'abbreviation': 'ATL',
        'conference': 'NFC',
        'division': 'South',
        'established': 1966,
        'colors': {
            'primary': '#A71930',  # Red
            'secondary': '#000000',  # Black
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/atl.png',
        'espn_id': 'atl',
        'super_bowl_wins': 0,
        'super_bowl_appearances': 2,
        'playoff_appearances': 15,
    },
    'CAR': {
        'team_id': 'CAR',
        'name': 'Carolina Panthers',
        'city': 'Charlotte',
        'state': 'NC',
        'abbreviation': 'CAR',
        'conference': 'NFC',
        'division': 'South',
        'established': 1995,
        'colors': {
            'primary': '#0085CA',  # Process blue
            'secondary': '#101820',  # Black
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/car.png',
        'espn_id': 'car',
        'super_bowl_wins': 0,
        'super_bowl_appearances': 2,
        'playoff_appearances': 9,
    },
    'NO': {
        'team_id': 'NO',
        'name': 'New Orleans Saints',
        'city': 'New Orleans',
        'state': 'LA',
        'abbreviation': 'NO',
        'conference': 'NFC',
        'division': 'South',
        'established': 1967,
        'colors': {
            'primary': '#D3BC8D',  # Gold
            'secondary': '#101820',  # Black
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/no.png',
        'espn_id': 'no',
        'super_bowl_wins': 1,
        'super_bowl_appearances': 1,
        'playoff_appearances': 16,
    },
    'TB': {
        'team_id': 'TB',
        'name': 'Tampa Bay Buccaneers',
        'city': 'Tampa',
        'state': 'FL',
        'abbreviation': 'TB',
        'conference': 'NFC',
        'division': 'South',
        'established': 1976,
        'colors': {
            'primary': '#D50A0A',  # Red
            'secondary': '#34302B',  # Pewter
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/tb.png',
        'espn_id': 'tb',
        'super_bowl_wins': 2,
        'super_bowl_appearances': 2,
        'playoff_appearances': 15,
    },

    # NFC West
    'ARI': {
        'team_id': 'ARI',
        'name': 'Arizona Cardinals',
        'city': 'Glendale',
        'state': 'AZ',
        'abbreviation': 'ARI',
        'conference': 'NFC',
        'division': 'West',
        'established': 1898,
        'colors': {
            'primary': '#97233F',  # Cardinal red
            'secondary': '#000000',  # Black
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/ari.png',
        'espn_id': 'ari',
        'super_bowl_wins': 0,
        'super_bowl_appearances': 1,
        'playoff_appearances': 10,
    },
    'LAR': {
        'team_id': 'LAR',
        'name': 'Los Angeles Rams',
        'city': 'Inglewood',
        'state': 'CA',
        'abbreviation': 'LAR',
        'conference': 'NFC',
        'division': 'West',
        'established': 1937,
        'colors': {
            'primary': '#003594',  # Royal blue
            'secondary': '#FFA300',  # Sol
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/lar.png',
        'espn_id': 'lar',
        'super_bowl_wins': 2,
        'super_bowl_appearances': 5,
        'playoff_appearances': 29,
    },
    'SF': {
        'team_id': 'SF',
        'name': 'San Francisco 49ers',
        'city': 'Santa Clara',
        'state': 'CA',
        'abbreviation': 'SF',
        'conference': 'NFC',
        'division': 'West',
        'established': 1946,
        'colors': {
            'primary': '#AA0000',  # Red
            'secondary': '#B3995D',  # Gold
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/sf.png',
        'espn_id': 'sf',
        'super_bowl_wins': 5,
        'super_bowl_appearances': 7,
        'playoff_appearances': 28,
    },
    'SEA': {
        'team_id': 'SEA',
        'name': 'Seattle Seahawks',
        'city': 'Seattle',
        'state': 'WA',
        'abbreviation': 'SEA',
        'conference': 'NFC',
        'division': 'West',
        'established': 1976,
        'colors': {
            'primary': '#002244',  # College navy
            'secondary': '#69BE28',  # Action green
        },
        'logo_url': 'https://a.espncdn.com/i/teamlogos/nfl/500/sea.png',
        'espn_id': 'sea',
        'super_bowl_wins': 1,
        'super_bowl_appearances': 3,
        'playoff_appearances': 19,
    },
}


def get_team(team_id: str) -> Optional[Dict]:
    """Get team information by team ID.

    Args:
        team_id: Team abbreviation (e.g., 'KC', 'BUF')

    Returns:
        Team dict with full info, or None if not found
    """
    team_info = TEAMS.get(team_id.upper())

    if team_info:
        # Merge with stadium info if available
        stadium_info = STADIUMS.get(team_id.upper())
        if stadium_info:
            team_info = {**team_info, 'stadium': stadium_info}

    return team_info


def get_all_teams() -> Dict[str, Dict]:
    """Get all teams.

    Returns:
        Dictionary of all teams with stadium info merged
    """
    all_teams = {}
    for team_id, team_info in TEAMS.items():
        team_data = team_info.copy()
        stadium_info = STADIUMS.get(team_id)
        if stadium_info:
            team_data['stadium'] = stadium_info
        all_teams[team_id] = team_data

    return all_teams


def get_division_teams(division: str) -> List[Dict]:
    """Get all teams in a division.

    Args:
        division: Division name (e.g., 'AFC East', 'NFC North')

    Returns:
        List of team dicts
    """
    # Parse division string (e.g., 'AFC East' -> conference='AFC', division='East')
    parts = division.split()
    if len(parts) != 2:
        return []

    conference, div_name = parts

    teams = []
    for team_id, team_info in TEAMS.items():
        if team_info['conference'] == conference and team_info['division'] == div_name:
            team_data = get_team(team_id)
            if team_data:
                teams.append(team_data)

    return teams


def get_conference_teams(conference: str) -> List[Dict]:
    """Get all teams in a conference.

    Args:
        conference: Conference name ('AFC' or 'NFC')

    Returns:
        List of team dicts
    """
    teams = []
    for team_id, team_info in TEAMS.items():
        if team_info['conference'] == conference:
            team_data = get_team(team_id)
            if team_data:
                teams.append(team_data)

    return teams
