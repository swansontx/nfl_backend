"""External API integrations for weather and injury data.

Integrations:
- OpenWeather API for game weather conditions
- Sleeper API for NFL player injury data
"""

import os
import requests
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import json


class WeatherAPI:
    """Integration with OpenWeather API for game weather data."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize weather API client.

        Args:
            api_key: OpenWeather API key (defaults to env var OPENWEATHER_API_KEY)
        """
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "https://api.openweathermap.org/data/2.5"

    def get_game_weather(self,
                        stadium_lat: float,
                        stadium_lon: float,
                        game_time: str) -> Dict:
        """Fetch weather forecast for a game.

        Args:
            stadium_lat: Stadium latitude
            stadium_lon: Stadium longitude
            game_time: Game time in ISO format (e.g., "2025-11-24T18:00:00Z")

        Returns:
            Weather data dictionary

        Example response:
            {
                "temperature": 45,
                "temp_unit": "F",
                "condition": "Clear",
                "wind_speed": 12,
                "wind_unit": "mph",
                "humidity": 65,
                "precipitation_chance": 10,
                "is_dome": False
            }
        """
        if not self.api_key:
            return self._get_placeholder_weather()

        try:
            url = f"{self.base_url}/forecast"
            params = {
                'lat': stadium_lat,
                'lon': stadium_lon,
                'appid': self.api_key,
                'units': 'imperial'  # Fahrenheit
            }

            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            # Parse forecast data for game time
            # TODO: Match forecast to game_time
            forecast = data['list'][0] if data.get('list') else {}

            return {
                'temperature': int(forecast.get('main', {}).get('temp', 70)),
                'temp_unit': 'F',
                'condition': forecast.get('weather', [{}])[0].get('main', 'Clear'),
                'wind_speed': int(forecast.get('wind', {}).get('speed', 0)),
                'wind_unit': 'mph',
                'humidity': forecast.get('main', {}).get('humidity', 50),
                'precipitation_chance': int(forecast.get('pop', 0) * 100),
                'is_dome': False  # TODO: Determine from stadium data
            }

        except Exception as e:
            print(f"Weather API error: {e}")
            return self._get_placeholder_weather()

    def _get_placeholder_weather(self) -> Dict:
        """Return placeholder weather when API is unavailable."""
        return {
            'temperature': 70,
            'temp_unit': 'F',
            'condition': 'Clear',
            'wind_speed': 5,
            'wind_unit': 'mph',
            'humidity': 50,
            'precipitation_chance': 0,
            'is_dome': False
        }


class SleeperAPI:
    """Integration with Sleeper API for NFL player data and injuries."""

    def __init__(self):
        """Initialize Sleeper API client."""
        self.base_url = "https://api.sleeper.app/v1"
        self._players_cache = None
        self._cache_timestamp = None

    def get_all_players(self, force_refresh: bool = False) -> Dict:
        """Fetch all NFL players from Sleeper API.

        Args:
            force_refresh: Force cache refresh

        Returns:
            Dictionary mapping player_id -> player data

        Note: Sleeper uses their own player IDs, not nflverse IDs.
              You'll need to map between them using player names.
        """
        # Cache for 1 hour
        cache_valid = (
            self._players_cache is not None and
            self._cache_timestamp is not None and
            (datetime.now() - self._cache_timestamp).seconds < 3600
        )

        if not force_refresh and cache_valid:
            return self._players_cache

        try:
            url = f"{self.base_url}/players/nfl"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            self._players_cache = response.json()
            self._cache_timestamp = datetime.now()
            return self._players_cache

        except Exception as e:
            print(f"Sleeper API error: {e}")
            return {}

    def get_injuries(self) -> List[Dict]:
        """Get all current NFL player injuries.

        Returns:
            List of injury records

        Example:
            [
                {
                    "player_id": "sleeper_id",
                    "player_name": "Patrick Mahomes",
                    "team": "KC",
                    "position": "QB",
                    "injury_status": "Questionable",
                    "injury_body_part": "Ankle",
                    "injury_notes": "Limited in practice"
                },
                ...
            ]
        """
        players = self.get_all_players()
        injuries = []

        for player_id, player_data in players.items():
            injury_status = player_data.get('injury_status')

            if injury_status and injury_status != 'Healthy':
                injuries.append({
                    'player_id': player_id,
                    'player_name': player_data.get('full_name', ''),
                    'team': player_data.get('team', ''),
                    'position': player_data.get('position', ''),
                    'injury_status': injury_status,
                    'injury_body_part': player_data.get('injury_body_part', ''),
                    'injury_notes': player_data.get('injury_notes', ''),
                    'last_updated': datetime.now().isoformat()
                })

        return injuries

    def get_injuries_for_team(self, team_abbr: str) -> List[Dict]:
        """Get injuries for a specific team.

        Args:
            team_abbr: Team abbreviation (e.g., "KC", "BUF")

        Returns:
            List of injury records for the team
        """
        all_injuries = self.get_injuries()
        return [
            injury for injury in all_injuries
            if injury['team'] == team_abbr
        ]

    def get_injuries_for_game(self, game_id: str) -> Dict[str, List[Dict]]:
        """Get injuries for both teams in a game.

        Args:
            game_id: Game ID in format {season}_{week}_{away}_{home}

        Returns:
            Dictionary with 'away_team' and 'home_team' injury lists

        Example:
            {
                "away_team": "KC",
                "home_team": "BUF",
                "away_injuries": [...],
                "home_injuries": [...]
            }
        """
        # Parse game_id to get teams
        from backend.canonical.game_id_utils import parse_game_id

        try:
            game_info = parse_game_id(game_id)
            away_team = game_info['away_team']
            home_team = game_info['home_team']

            return {
                'away_team': away_team,
                'home_team': home_team,
                'away_injuries': self.get_injuries_for_team(away_team),
                'home_injuries': self.get_injuries_for_team(home_team)
            }

        except Exception as e:
            print(f"Error getting injuries for game {game_id}: {e}")
            return {
                'away_team': '',
                'home_team': '',
                'away_injuries': [],
                'home_injuries': []
            }


# Singleton instances
weather_api = WeatherAPI()
sleeper_api = SleeperAPI()
