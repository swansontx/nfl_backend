"""
Team name normalization and mapping

Handles various team name formats across data sources:
- nflverse: "KC", "SF", "LAR", "TB"
- Sportsbooks: "Kansas City Chiefs", "KC Chiefs"
- Historical changes: "OAK" -> "LV", "SD" -> "LAC", "STL" -> "LAR"
"""

from typing import Optional, Dict
from backend.config.logging_config import get_logger

logger = get_logger(__name__)


class TeamMapper:
    """Maps various team name formats to canonical abbreviations"""

    # Canonical team abbreviations (current as of 2024)
    CANONICAL_TEAMS = {
        'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
        'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
        'LAC', 'LAR', 'LV', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
        'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS'
    }

    # Historical team relocations/rebrands
    TEAM_RELOCATIONS = {
        'OAK': 'LV',   # Oakland -> Las Vegas (2020)
        'SD': 'LAC',   # San Diego -> LA Chargers (2017)
        'STL': 'LAR',  # St. Louis -> LA Rams (2016)
    }

    # Alternative abbreviations used by various data sources
    ABBREVIATION_MAP = {
        # Standard
        'ARI': 'ARI', 'ARZ': 'ARI',
        'ATL': 'ATL',
        'BAL': 'BAL',
        'BUF': 'BUF',
        'CAR': 'CAR',
        'CHI': 'CHI',
        'CIN': 'CIN',
        'CLE': 'CLE',
        'DAL': 'DAL',
        'DEN': 'DEN',
        'DET': 'DET',
        'GB': 'GB', 'GNB': 'GB',
        'HOU': 'HOU',
        'IND': 'IND',
        'JAX': 'JAX', 'JAC': 'JAX',
        'KC': 'KC', 'KAN': 'KC',
        'LAC': 'LAC', 'LA': 'LAC',  # Ambiguous - context needed
        'LAR': 'LAR',
        'LV': 'LV', 'LVR': 'LV',
        'MIA': 'MIA',
        'MIN': 'MIN',
        'NE': 'NE', 'NEP': 'NE',
        'NO': 'NO', 'NOR': 'NO',
        'NYG': 'NYG',
        'NYJ': 'NYJ',
        'PHI': 'PHI',
        'PIT': 'PIT',
        'SEA': 'SEA',
        'SF': 'SF', 'SFO': 'SF',
        'TB': 'TB', 'TAM': 'TB',
        'TEN': 'TEN',
        'WAS': 'WAS', 'WSH': 'WAS',

        # Historical
        'OAK': 'LV',
        'SD': 'LAC',
        'STL': 'LAR',
    }

    # Full team names to abbreviations
    FULL_NAME_MAP = {
        # AFC East
        'buffalo bills': 'BUF',
        'miami dolphins': 'MIA',
        'new england patriots': 'NE',
        'new york jets': 'NYJ',

        # AFC North
        'baltimore ravens': 'BAL',
        'cincinnati bengals': 'CIN',
        'cleveland browns': 'CLE',
        'pittsburgh steelers': 'PIT',

        # AFC South
        'houston texans': 'HOU',
        'indianapolis colts': 'IND',
        'jacksonville jaguars': 'JAX',
        'tennessee titans': 'TEN',

        # AFC West
        'denver broncos': 'DEN',
        'kansas city chiefs': 'KC',
        'las vegas raiders': 'LV',
        'los angeles chargers': 'LAC',

        # NFC East
        'dallas cowboys': 'DAL',
        'new york giants': 'NYG',
        'philadelphia eagles': 'PHI',
        'washington commanders': 'WAS',

        # NFC North
        'chicago bears': 'CHI',
        'detroit lions': 'DET',
        'green bay packers': 'GB',
        'minnesota vikings': 'MIN',

        # NFC South
        'atlanta falcons': 'ATL',
        'carolina panthers': 'CAR',
        'new orleans saints': 'NO',
        'tampa bay buccaneers': 'TB',

        # NFC West
        'arizona cardinals': 'ARI',
        'los angeles rams': 'LAR',
        'san francisco 49ers': 'SF',
        'seattle seahawks': 'SEA',

        # Historical
        'oakland raiders': 'LV',
        'san diego chargers': 'LAC',
        'st louis rams': 'LAR',
        'st. louis rams': 'LAR',
        'washington redskins': 'WAS',
        'washington football team': 'WAS',
    }

    # City/nickname variations
    CITY_NICKNAME_MAP = {
        'arizona': 'ARI',
        'atlanta': 'ATL',
        'baltimore': 'BAL',
        'buffalo': 'BUF',
        'carolina': 'CAR',
        'chicago': 'CHI',
        'cincinnati': 'CIN',
        'cleveland': 'CLE',
        'dallas': 'DAL',
        'denver': 'DEN',
        'detroit': 'DET',
        'green bay': 'GB',
        'houston': 'HOU',
        'indianapolis': 'IND',
        'jacksonville': 'JAX',
        'kansas city': 'KC',
        'los angeles chargers': 'LAC',
        'los angeles rams': 'LAR',
        'las vegas': 'LV',
        'miami': 'MIA',
        'minnesota': 'MIN',
        'new england': 'NE',
        'new orleans': 'NO',
        'new york giants': 'NYG',
        'new york jets': 'NYJ',
        'philadelphia': 'PHI',
        'pittsburgh': 'PIT',
        'seattle': 'SEA',
        'san francisco': 'SF',
        'tampa bay': 'TB',
        'tennessee': 'TEN',
        'washington': 'WAS',
    }

    @classmethod
    def normalize_team(cls, team: str, year: Optional[int] = None) -> Optional[str]:
        """
        Normalize team name to canonical abbreviation

        Args:
            team: Team name in any format
            year: Optional year for handling relocations

        Returns:
            Canonical team abbreviation or None if not found
        """
        if not team:
            return None

        team = team.strip()

        # Already canonical
        if team in cls.CANONICAL_TEAMS:
            return team

        # Try abbreviation map
        team_upper = team.upper()
        if team_upper in cls.ABBREVIATION_MAP:
            canonical = cls.ABBREVIATION_MAP[team_upper]
            logger.debug("team_normalized", input=team, output=canonical)
            return canonical

        # Try full name
        team_lower = team.lower()
        if team_lower in cls.FULL_NAME_MAP:
            canonical = cls.FULL_NAME_MAP[team_lower]
            logger.debug("team_normalized_fullname", input=team, output=canonical)
            return canonical

        # Try city/nickname
        if team_lower in cls.CITY_NICKNAME_MAP:
            canonical = cls.CITY_NICKNAME_MAP[team_lower]
            logger.debug("team_normalized_city", input=team, output=canonical)
            return canonical

        # Try partial matching
        for full_name, abbr in cls.FULL_NAME_MAP.items():
            if team_lower in full_name or full_name in team_lower:
                logger.debug("team_normalized_partial", input=team, output=abbr)
                return abbr

        logger.warning("team_not_found", team=team)
        return None

    @classmethod
    def handle_relocation(cls, team: str, year: int) -> str:
        """
        Handle team relocations based on year

        Args:
            team: Team abbreviation
            year: Season year

        Returns:
            Appropriate team abbreviation for that year
        """
        # OAK -> LV in 2020
        if team == 'OAK' and year >= 2020:
            return 'LV'
        elif team == 'LV' and year < 2020:
            return 'OAK'

        # SD -> LAC in 2017
        if team == 'SD' and year >= 2017:
            return 'LAC'
        elif team == 'LAC' and year < 2017:
            return 'SD'

        # STL -> LAR in 2016
        if team == 'STL' and year >= 2016:
            return 'LAR'
        elif team == 'LAR' and year < 2016:
            return 'STL'

        return team

    @classmethod
    def get_team_full_name(cls, abbr: str) -> Optional[str]:
        """
        Get full team name from abbreviation

        Args:
            abbr: Team abbreviation

        Returns:
            Full team name
        """
        # Reverse lookup
        for full_name, team_abbr in cls.FULL_NAME_MAP.items():
            if team_abbr == abbr:
                return full_name.title()

        return None

    @classmethod
    def are_same_team(cls, team1: str, team2: str, year1: Optional[int] = None, year2: Optional[int] = None) -> bool:
        """
        Check if two team names refer to the same team

        Args:
            team1: First team name
            team2: Second team name
            year1: Year for first team (for relocations)
            year2: Year for second team

        Returns:
            True if same team
        """
        norm1 = cls.normalize_team(team1, year1)
        norm2 = cls.normalize_team(team2, year2)

        if not norm1 or not norm2:
            return False

        # Handle relocations
        if year1:
            norm1 = cls.handle_relocation(norm1, year1)
        if year2:
            norm2 = cls.handle_relocation(norm2, year2)

        return norm1 == norm2

    @classmethod
    def get_division(cls, team: str) -> Optional[str]:
        """
        Get division for a team

        Args:
            team: Team abbreviation

        Returns:
            Division ("AFC East", etc.) or None
        """
        canonical = cls.normalize_team(team)
        if not canonical:
            return None

        divisions = {
            'AFC East': ['BUF', 'MIA', 'NE', 'NYJ'],
            'AFC North': ['BAL', 'CIN', 'CLE', 'PIT'],
            'AFC South': ['HOU', 'IND', 'JAX', 'TEN'],
            'AFC West': ['DEN', 'KC', 'LAC', 'LV'],
            'NFC East': ['DAL', 'NYG', 'PHI', 'WAS'],
            'NFC North': ['CHI', 'DET', 'GB', 'MIN'],
            'NFC South': ['ATL', 'CAR', 'NO', 'TB'],
            'NFC West': ['ARI', 'LAR', 'SF', 'SEA'],
        }

        for division, teams in divisions.items():
            if canonical in teams:
                return division

        return None

    @classmethod
    def is_divisional_matchup(cls, team1: str, team2: str) -> bool:
        """
        Check if two teams are in the same division

        Args:
            team1: First team
            team2: Second team

        Returns:
            True if divisional rivals
        """
        div1 = cls.get_division(team1)
        div2 = cls.get_division(team2)

        return div1 is not None and div1 == div2
