"""Home field advantage features for NFL prop modeling.

Calculates various home field advantage (HFA) metrics that can be used
as features in ML models. Leverages stadium database for stadium-specific factors.

HFA Factors:
- Standard home/away indicator (binary)
- Stadium-specific advantages (altitude, noise, weather)
- Travel distance and time zone changes
- Division/rivalry games (familiarity reduces HFA)
- Rest days differential
"""

from typing import Dict, Optional
from pathlib import Path
import math

from backend.api.stadium_database import get_stadium, STADIUMS


class HomeFieldAdvantageCalculator:
    """Calculate home field advantage features."""

    # Stadium-specific HFA modifiers (based on historical data)
    STADIUM_HFA_MULTIPLIERS = {
        'SEA': 1.25,  # Seattle - 12th man, extreme noise
        'KC': 1.20,   # Kansas City - Arrowhead noise
        'DEN': 1.18,  # Denver - Mile High altitude
        'GB': 1.15,   # Green Bay - Lambeau mystique, cold weather
        'NO': 1.12,   # New Orleans - Superdome noise
        'BUF': 1.10,  # Buffalo - weather advantage
        'NE': 1.08,   # New England - historically strong HFA
        # Neutral/average stadiums
        'LAR': 0.95,  # LA - lots of away fans
        'LAC': 0.95,  # LA Chargers - lots of away fans
        'LV': 0.98,   # Las Vegas - neutral
        'MIA': 1.05,  # Miami - heat advantage
        'TB': 1.02,   # Tampa - moderate
    }

    # Default for teams not listed
    DEFAULT_HFA_MULTIPLIER = 1.0

    @staticmethod
    def calculate_basic_hfa(is_home_team: bool) -> Dict[str, float]:
        """Calculate basic home field advantage feature.

        Args:
            is_home_team: Whether this is the home team

        Returns:
            Dictionary with HFA features
        """
        return {
            'is_home': 1.0 if is_home_team else 0.0,
            'is_away': 0.0 if is_home_team else 1.0,
            'base_hfa_multiplier': 1.0 if is_home_team else 1.0
        }

    @classmethod
    def calculate_stadium_hfa(cls, team: str, is_home_team: bool) -> Dict[str, float]:
        """Calculate stadium-specific home field advantage.

        Args:
            team: Team abbreviation
            is_home_team: Whether this team is playing at home

        Returns:
            Dictionary with stadium-specific HFA features
        """
        stadium = get_stadium(team)

        features = {
            'is_home': 1.0 if is_home_team else 0.0,
            'stadium_hfa_multiplier': 1.0,
            'altitude_advantage': 0.0,
            'dome_advantage': 0.0,
            'weather_exposure': 1.0  # 1.0 = outdoor, 0.0 = dome
        }

        if not is_home_team:
            # Away team gets no stadium advantages
            return features

        # Get stadium-specific multiplier
        features['stadium_hfa_multiplier'] = cls.STADIUM_HFA_MULTIPLIERS.get(
            team,
            cls.DEFAULT_HFA_MULTIPLIER
        )

        if not stadium:
            return features

        # Altitude advantage (Denver)
        if team == 'DEN':
            features['altitude_advantage'] = 1.0  # 5,280 ft elevation

        # Dome advantage (controlled conditions, noise)
        if stadium['is_dome']:
            features['dome_advantage'] = 1.0
            features['weather_exposure'] = 0.0

        return features

    @staticmethod
    def calculate_travel_distance(away_team: str, home_team: str) -> Dict[str, float]:
        """Calculate travel distance impact on performance.

        Args:
            away_team: Away team abbreviation
            home_team: Home team abbreviation

        Returns:
            Dictionary with travel-related features

        Travel impact factors:
            - Distance traveled (miles)
            - Time zone changes (EST, CST, MST, PST)
            - Cross-country travel penalty
        """
        away_stadium = get_stadium(away_team)
        home_stadium = get_stadium(home_team)

        features = {
            'travel_distance_miles': 0.0,
            'time_zone_change': 0,  # Number of time zones crossed
            'cross_country_travel': 0.0,  # 1.0 if coast-to-coast
            'travel_penalty': 0.0  # Combined travel impact (0-1 scale)
        }

        if not away_stadium or not home_stadium:
            return features

        # Calculate great circle distance
        distance = cls._calculate_distance(
            away_stadium['lat'], away_stadium['lon'],
            home_stadium['lat'], home_stadium['lon']
        )
        features['travel_distance_miles'] = round(distance, 1)

        # Estimate time zone changes (rough approximation)
        away_lon = away_stadium['lon']
        home_lon = home_stadium['lon']
        tz_change = abs(int((away_lon - home_lon) / 15))  # ~15 degrees per timezone
        features['time_zone_change'] = min(tz_change, 3)

        # Cross-country travel (>2000 miles)
        if distance > 2000:
            features['cross_country_travel'] = 1.0

        # Combined travel penalty (normalized 0-1)
        # Factors: distance (max ~3000 mi) + timezone (max 3)
        penalty = min(
            (distance / 3000) * 0.7 +  # Distance component (0-0.7)
            (tz_change / 3) * 0.3,      # Timezone component (0-0.3)
            1.0
        )
        features['travel_penalty'] = round(penalty, 3)

        return features

    @staticmethod
    def calculate_rivalry_factor(away_team: str, home_team: str) -> Dict[str, float]:
        """Calculate rivalry/familiarity factor.

        Args:
            away_team: Away team abbreviation
            home_team: Home team abbreviation

        Returns:
            Dictionary with rivalry features

        Division games reduce HFA due to familiarity.
        Rivalry games may increase intensity.
        """
        # NFL divisions
        divisions = {
            'AFC_EAST': ['BUF', 'MIA', 'NE', 'NYJ'],
            'AFC_NORTH': ['BAL', 'CIN', 'CLE', 'PIT'],
            'AFC_SOUTH': ['HOU', 'IND', 'JAX', 'TEN'],
            'AFC_WEST': ['DEN', 'KC', 'LAC', 'LV'],
            'NFC_EAST': ['DAL', 'NYG', 'PHI', 'WAS'],
            'NFC_NORTH': ['CHI', 'DET', 'GB', 'MIN'],
            'NFC_SOUTH': ['ATL', 'CAR', 'NO', 'TB'],
            'NFC_WEST': ['ARI', 'LAR', 'SF', 'SEA']
        }

        features = {
            'is_division_game': 0.0,
            'same_conference': 0.0,
            'hfa_familiarity_reduction': 0.0  # HFA reduced by ~10-15% in division games
        }

        # Check if division game
        for division, teams in divisions.items():
            if away_team in teams and home_team in teams:
                features['is_division_game'] = 1.0
                features['hfa_familiarity_reduction'] = 0.12  # 12% reduction
                break

        # Check if same conference
        away_conf = 'AFC' if any(away_team in div for div, _ in divisions.items() if 'AFC' in div) else 'NFC'
        home_conf = 'AFC' if any(home_team in div for div, _ in divisions.items() if 'AFC' in div) else 'NFC'

        if away_conf == home_conf:
            features['same_conference'] = 1.0

        return features

    @staticmethod
    def _calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great circle distance between two points (Haversine formula).

        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates

        Returns:
            Distance in miles
        """
        # Earth radius in miles
        R = 3959.0

        # Convert to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))

        return R * c

    @classmethod
    def get_all_hfa_features(
        cls,
        game_id: str,
        team: str,
        is_home_team: bool
    ) -> Dict[str, float]:
        """Get all HFA features for a team in a game.

        Args:
            game_id: Game ID in format {season}_{week}_{away}_{home}
            team: Team abbreviation
            is_home_team: Whether this team is home

        Returns:
            Dictionary with all HFA features combined
        """
        # Parse game_id to get both teams
        try:
            parts = game_id.split('_')
            if len(parts) != 4:
                raise ValueError(f"Invalid game_id: {game_id}")

            away_team = parts[2]
            home_team = parts[3]

        except Exception as e:
            print(f"Error parsing game_id {game_id}: {e}")
            return cls.calculate_basic_hfa(is_home_team)

        # Combine all HFA features
        features = {}

        # Basic HFA
        features.update(cls.calculate_basic_hfa(is_home_team))

        # Stadium-specific HFA
        features.update(cls.calculate_stadium_hfa(team, is_home_team))

        # Travel distance (for away team)
        if not is_home_team:
            features.update(cls.calculate_travel_distance(away_team, home_team))

        # Rivalry/division game
        features.update(cls.calculate_rivalry_factor(away_team, home_team))

        return features


# Convenience instance
hfa_calculator = HomeFieldAdvantageCalculator()
