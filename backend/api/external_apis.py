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

from backend.api.cache import cached, CACHE_TTL
from backend.api.stadium_database import get_stadium_for_game


class OpenMeteoAPI:
    """Integration with Open-Meteo API for game weather data (FREE, no API key needed)."""

    def __init__(self):
        """Initialize Open-Meteo API client."""
        self.base_url = "https://api.open-meteo.com/v1/forecast"

    @cached(ttl_seconds=CACHE_TTL['weather'])  # 1 hour
    def get_game_weather(self,
                        stadium_lat: float,
                        stadium_lon: float,
                        game_time: str) -> Dict:
        """Fetch weather forecast for a game using Open-Meteo (free).

        Args:
            stadium_lat: Stadium latitude
            stadium_lon: Stadium longitude
            game_time: Game time in ISO format (e.g., "2025-11-24T18:00:00Z")

        Returns:
            Weather data dictionary with impact analysis

        Example response:
            {
                "temperature": 45,
                "temp_unit": "F",
                "condition": "Clear",
                "wind_speed": 12,
                "wind_unit": "mph",
                "wind_gusts": 20,
                "humidity": 65,
                "precipitation_chance": 10,
                "precipitation_mm": 0.0,
                "is_dome": False,
                "weather_impact": {
                    "passing_impact": "neutral",
                    "kicking_impact": "slight_negative",
                    "overall_score_impact": "neutral"
                }
            }
        """
        try:
            params = {
                'latitude': stadium_lat,
                'longitude': stadium_lon,
                'hourly': 'temperature_2m,relative_humidity_2m,precipitation_probability,precipitation,wind_speed_10m,wind_gusts_10m,weather_code',
                'temperature_unit': 'fahrenheit',
                'wind_speed_unit': 'mph',
                'precipitation_unit': 'mm',
                'forecast_days': 7,
                'timezone': 'America/New_York'
            }

            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Parse forecast data for game time
            hourly = data.get('hourly', {})
            times = hourly.get('time', [])

            # Find the closest time to game_time
            game_dt = datetime.fromisoformat(game_time.replace('Z', '+00:00')) if game_time else datetime.now()
            target_hour = game_dt.strftime('%Y-%m-%dT%H:00')

            idx = 0
            for i, t in enumerate(times):
                if t >= target_hour:
                    idx = i
                    break

            # Extract weather data
            temp = hourly.get('temperature_2m', [70])[idx] if hourly.get('temperature_2m') else 70
            humidity = hourly.get('relative_humidity_2m', [50])[idx] if hourly.get('relative_humidity_2m') else 50
            precip_prob = hourly.get('precipitation_probability', [0])[idx] if hourly.get('precipitation_probability') else 0
            precip_mm = hourly.get('precipitation', [0])[idx] if hourly.get('precipitation') else 0
            wind_speed = hourly.get('wind_speed_10m', [5])[idx] if hourly.get('wind_speed_10m') else 5
            wind_gusts = hourly.get('wind_gusts_10m', [10])[idx] if hourly.get('wind_gusts_10m') else 10
            weather_code = hourly.get('weather_code', [0])[idx] if hourly.get('weather_code') else 0

            # Determine condition from weather code
            condition = self._weather_code_to_condition(weather_code)

            # Calculate weather impact on props
            impact = self._calculate_weather_impact(temp, wind_speed, wind_gusts, precip_prob, precip_mm)

            return {
                'temperature': int(temp),
                'temp_unit': 'F',
                'condition': condition,
                'wind_speed': int(wind_speed),
                'wind_gusts': int(wind_gusts),
                'wind_unit': 'mph',
                'humidity': int(humidity),
                'precipitation_chance': int(precip_prob),
                'precipitation_mm': round(precip_mm, 1),
                'is_dome': False,
                'weather_impact': impact,
                'source': 'open-meteo'
            }

        except Exception as e:
            print(f"Open-Meteo API error: {e}")
            return self._get_placeholder_weather()

    def _weather_code_to_condition(self, code: int) -> str:
        """Convert WMO weather code to condition string."""
        conditions = {
            0: 'Clear',
            1: 'Mainly Clear', 2: 'Partly Cloudy', 3: 'Overcast',
            45: 'Fog', 48: 'Depositing Fog',
            51: 'Light Drizzle', 53: 'Drizzle', 55: 'Heavy Drizzle',
            61: 'Light Rain', 63: 'Rain', 65: 'Heavy Rain',
            66: 'Freezing Rain', 67: 'Heavy Freezing Rain',
            71: 'Light Snow', 73: 'Snow', 75: 'Heavy Snow',
            77: 'Snow Grains',
            80: 'Light Showers', 81: 'Showers', 82: 'Heavy Showers',
            85: 'Light Snow Showers', 86: 'Heavy Snow Showers',
            95: 'Thunderstorm', 96: 'Thunderstorm with Hail', 99: 'Heavy Thunderstorm'
        }
        return conditions.get(code, 'Unknown')

    def _calculate_weather_impact(self, temp: float, wind: float, gusts: float,
                                   precip_prob: float, precip_mm: float) -> Dict:
        """Calculate weather impact on different prop types.

        Returns impact assessments for passing, kicking, rushing, and overall scoring.
        """
        impact = {
            'passing_impact': 'neutral',
            'passing_adjustment': 0,  # Percentage adjustment to passing props
            'kicking_impact': 'neutral',
            'kicking_adjustment': 0,
            'rushing_impact': 'neutral',
            'rushing_adjustment': 0,
            'total_impact': 'neutral',
            'total_adjustment': 0,  # Expected points adjustment
            'notes': []
        }

        # Wind impact (major factor for passing and kicking)
        if gusts >= 30 or wind >= 20:
            impact['passing_impact'] = 'strong_negative'
            impact['passing_adjustment'] = -15  # -15% to passing yards
            impact['kicking_impact'] = 'strong_negative'
            impact['kicking_adjustment'] = -20
            impact['notes'].append(f"High winds ({wind} mph, gusts {gusts}) will significantly impact passing/kicking")
        elif gusts >= 20 or wind >= 15:
            impact['passing_impact'] = 'moderate_negative'
            impact['passing_adjustment'] = -8
            impact['kicking_impact'] = 'moderate_negative'
            impact['kicking_adjustment'] = -10
            impact['notes'].append(f"Moderate winds ({wind} mph) may affect deep passes and long FGs")
        elif gusts >= 15 or wind >= 10:
            impact['passing_impact'] = 'slight_negative'
            impact['passing_adjustment'] = -3
            impact['kicking_impact'] = 'slight_negative'
            impact['kicking_adjustment'] = -5
            impact['notes'].append(f"Light winds ({wind} mph) - minimal impact")

        # Precipitation impact
        if precip_mm >= 5 or precip_prob >= 80:
            impact['passing_impact'] = 'strong_negative' if impact['passing_impact'] != 'strong_negative' else impact['passing_impact']
            impact['passing_adjustment'] -= 10
            impact['rushing_impact'] = 'slight_positive'
            impact['rushing_adjustment'] = 5  # Teams run more in bad weather
            impact['notes'].append(f"Heavy precipitation expected - favors rushing game")
        elif precip_mm >= 2 or precip_prob >= 50:
            impact['passing_adjustment'] -= 5
            impact['rushing_adjustment'] += 3
            impact['notes'].append(f"Precipitation likely ({precip_prob}%) - may see increased rushing")

        # Temperature extremes
        if temp <= 32:
            impact['passing_adjustment'] -= 5
            impact['kicking_adjustment'] -= 5
            impact['notes'].append(f"Cold weather ({temp}F) - grip/ball flight affected")
        elif temp <= 40:
            impact['passing_adjustment'] -= 2
            impact['notes'].append(f"Cool weather ({temp}F)")
        elif temp >= 90:
            impact['notes'].append(f"Hot weather ({temp}F) - fatigue factor in 4th quarter")

        # Calculate overall total impact
        total_adj = (impact['passing_adjustment'] + impact['kicking_adjustment']) / 3
        if total_adj <= -10:
            impact['total_impact'] = 'strong_negative'
            impact['total_adjustment'] = -3  # Expected 3 fewer points
        elif total_adj <= -5:
            impact['total_impact'] = 'moderate_negative'
            impact['total_adjustment'] = -1.5
        elif total_adj >= 3:
            impact['total_impact'] = 'slight_positive'
            impact['total_adjustment'] = 1
        else:
            impact['total_impact'] = 'neutral'
            impact['total_adjustment'] = 0

        return impact

    def get_weather_for_game(self, game_id: str, game_time: Optional[str] = None) -> Dict:
        """Get weather for a game using stadium database.

        Args:
            game_id: Game ID in format {season}_{week}_{away}_{home}
            game_time: Game time in ISO format (optional)

        Returns:
            Weather data dictionary with impact analysis
        """
        # Get stadium from game_id
        stadium = get_stadium_for_game(game_id)

        if not stadium:
            return self._get_placeholder_weather()

        # If it's a dome, return dome weather
        if stadium['is_dome']:
            return {
                'temperature': 72,
                'temp_unit': 'F',
                'condition': 'Dome',
                'wind_speed': 0,
                'wind_gusts': 0,
                'wind_unit': 'mph',
                'humidity': 50,
                'precipitation_chance': 0,
                'precipitation_mm': 0,
                'is_dome': True,
                'stadium_name': stadium['name'],
                'weather_impact': {
                    'passing_impact': 'neutral',
                    'passing_adjustment': 0,
                    'kicking_impact': 'neutral',
                    'kicking_adjustment': 0,
                    'rushing_impact': 'neutral',
                    'rushing_adjustment': 0,
                    'total_impact': 'neutral',
                    'total_adjustment': 0,
                    'notes': ['Indoor dome - weather not a factor']
                },
                'source': 'dome'
            }

        # Get actual weather forecast
        game_time = game_time or datetime.now().isoformat()
        weather = self.get_game_weather(
            stadium['lat'],
            stadium['lon'],
            game_time
        )
        weather['stadium_name'] = stadium['name']
        weather['is_dome'] = False

        return weather

    def _get_placeholder_weather(self) -> Dict:
        """Return placeholder weather when API is unavailable."""
        return {
            'temperature': 70,
            'temp_unit': 'F',
            'condition': 'Clear',
            'wind_speed': 5,
            'wind_gusts': 10,
            'wind_unit': 'mph',
            'humidity': 50,
            'precipitation_chance': 0,
            'precipitation_mm': 0,
            'is_dome': False,
            'weather_impact': {
                'passing_impact': 'neutral',
                'passing_adjustment': 0,
                'kicking_impact': 'neutral',
                'kicking_adjustment': 0,
                'rushing_impact': 'neutral',
                'rushing_adjustment': 0,
                'total_impact': 'neutral',
                'total_adjustment': 0,
                'notes': ['Weather data unavailable - using defaults']
            },
            'source': 'placeholder'
        }


class WeatherAPI:
    """Integration with OpenWeather API for game weather data."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize weather API client.

        Args:
            api_key: OpenWeather API key (defaults to env var OPENWEATHER_API_KEY)
        """
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "https://api.openweathermap.org/data/2.5"

    @cached(ttl_seconds=CACHE_TTL['weather'])  # 1 hour
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

    def get_weather_for_game(self, game_id: str, game_time: Optional[str] = None) -> Dict:
        """Get weather for a game using stadium database.

        Args:
            game_id: Game ID in format {season}_{week}_{away}_{home}
            game_time: Game time in ISO format (optional, uses placeholder if not provided)

        Returns:
            Weather data dictionary
        """
        # Get stadium from game_id
        stadium = get_stadium_for_game(game_id)

        if not stadium:
            return self._get_placeholder_weather()

        # If it's a dome, return dome weather
        if stadium['is_dome']:
            return {
                'temperature': 72,
                'temp_unit': 'F',
                'condition': 'Clear',
                'wind_speed': 0,
                'wind_unit': 'mph',
                'humidity': 50,
                'precipitation_chance': 0,
                'is_dome': True,
                'stadium_name': stadium['name']
            }

        # Get actual weather forecast
        game_time = game_time or datetime.now().isoformat()
        weather = self.get_game_weather(
            stadium['lat'],
            stadium['lon'],
            game_time
        )
        weather['stadium_name'] = stadium['name']
        weather['is_dome'] = False

        return weather

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

    @cached(ttl_seconds=CACHE_TTL['injuries'])  # 15 minutes
    def get_all_players(self, force_refresh: bool = False) -> Dict:
        """Fetch all NFL players from Sleeper API.

        Args:
            force_refresh: Force cache refresh (note: caching handled by decorator)

        Returns:
            Dictionary mapping player_id -> player data

        Note: Sleeper uses their own player IDs, not nflverse IDs.
              You'll need to map between them using player names.
        """
        try:
            url = f"{self.base_url}/players/nfl"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()

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
open_meteo_api = OpenMeteoAPI()  # FREE - no API key needed (recommended)
weather_api = WeatherAPI()       # OpenWeather - requires API key
sleeper_api = SleeperAPI()
