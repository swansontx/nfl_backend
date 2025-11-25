"""Weather Data Scraper.

Fetches weather data for NFL game locations to support weather-based adjustments.
"""

from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime, timedelta
import requests
import json
from pathlib import Path


@dataclass
class WeatherData:
    """Weather data for a game location."""
    location: str
    game_time: Optional[datetime] = None

    temperature: Optional[float] = None  # Fahrenheit
    feels_like: Optional[float] = None
    wind_speed: Optional[float] = None  # MPH
    wind_direction: Optional[str] = None
    precipitation: Optional[str] = None  # 'none', 'rain', 'snow'
    humidity: Optional[float] = None
    conditions: Optional[str] = None  # Description

    # Stadium info
    is_dome: bool = False
    is_retractable: bool = False

    timestamp: Optional[datetime] = None


class WeatherScraper:
    """Scraper for weather data with caching."""

    def __init__(self):
        """Initialize scraper."""
        self.cache_dir = Path('.cache/weather')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_duration = timedelta(hours=3)  # Cache for 3 hours

        # Stadium locations (for weather lookup)
        self.stadium_locations = {
            'ARI': {'city': 'Glendale', 'state': 'AZ', 'dome': True, 'retractable': True},
            'ATL': {'city': 'Atlanta', 'state': 'GA', 'dome': True, 'retractable': False},
            'BAL': {'city': 'Baltimore', 'state': 'MD', 'dome': False, 'retractable': False},
            'BUF': {'city': 'Orchard Park', 'state': 'NY', 'dome': False, 'retractable': False},
            'CAR': {'city': 'Charlotte', 'state': 'NC', 'dome': False, 'retractable': False},
            'CHI': {'city': 'Chicago', 'state': 'IL', 'dome': False, 'retractable': False},
            'CIN': {'city': 'Cincinnati', 'state': 'OH', 'dome': False, 'retractable': False},
            'CLE': {'city': 'Cleveland', 'state': 'OH', 'dome': False, 'retractable': False},
            'DAL': {'city': 'Arlington', 'state': 'TX', 'dome': True, 'retractable': True},
            'DEN': {'city': 'Denver', 'state': 'CO', 'dome': False, 'retractable': False},
            'DET': {'city': 'Detroit', 'state': 'MI', 'dome': True, 'retractable': False},
            'GB': {'city': 'Green Bay', 'state': 'WI', 'dome': False, 'retractable': False},
            'HOU': {'city': 'Houston', 'state': 'TX', 'dome': True, 'retractable': True},
            'IND': {'city': 'Indianapolis', 'state': 'IN', 'dome': True, 'retractable': False},
            'JAX': {'city': 'Jacksonville', 'state': 'FL', 'dome': False, 'retractable': False},
            'KC': {'city': 'Kansas City', 'state': 'MO', 'dome': False, 'retractable': False},
            'LAC': {'city': 'Inglewood', 'state': 'CA', 'dome': False, 'retractable': False},
            'LAR': {'city': 'Inglewood', 'state': 'CA', 'dome': False, 'retractable': False},
            'LV': {'city': 'Las Vegas', 'state': 'NV', 'dome': True, 'retractable': False},
            'MIA': {'city': 'Miami Gardens', 'state': 'FL', 'dome': False, 'retractable': False},
            'MIN': {'city': 'Minneapolis', 'state': 'MN', 'dome': True, 'retractable': False},
            'NE': {'city': 'Foxborough', 'state': 'MA', 'dome': False, 'retractable': False},
            'NO': {'city': 'New Orleans', 'state': 'LA', 'dome': True, 'retractable': False},
            'NYG': {'city': 'East Rutherford', 'state': 'NJ', 'dome': False, 'retractable': False},
            'NYJ': {'city': 'East Rutherford', 'state': 'NJ', 'dome': False, 'retractable': False},
            'PHI': {'city': 'Philadelphia', 'state': 'PA', 'dome': False, 'retractable': False},
            'PIT': {'city': 'Pittsburgh', 'state': 'PA', 'dome': False, 'retractable': False},
            'SF': {'city': 'Santa Clara', 'state': 'CA', 'dome': False, 'retractable': False},
            'SEA': {'city': 'Seattle', 'state': 'WA', 'dome': False, 'retractable': False},
            'TB': {'city': 'Tampa', 'state': 'FL', 'dome': False, 'retractable': False},
            'TEN': {'city': 'Nashville', 'state': 'TN', 'dome': False, 'retractable': False},
            'WAS': {'city': 'Landover', 'state': 'MD', 'dome': False, 'retractable': False}
        }

    def fetch_weather(
        self,
        team: str,
        game_time: Optional[datetime] = None,
        use_cache: bool = True
    ) -> WeatherData:
        """Fetch weather data for a team's home stadium.

        Args:
            team: Team abbreviation
            game_time: Game date/time (for forecast)
            use_cache: Whether to use cached data

        Returns:
            WeatherData object
        """
        # Get stadium info
        stadium_info = self.stadium_locations.get(team)
        if not stadium_info:
            return self._get_default_weather(team)

        # If dome, return immediately
        if stadium_info['dome'] and not stadium_info['retractable']:
            return WeatherData(
                location=f"{stadium_info['city']}, {stadium_info['state']}",
                game_time=game_time,
                is_dome=True,
                timestamp=datetime.now()
            )

        # Check cache
        cache_key = f"{team}_{game_time.strftime('%Y%m%d') if game_time else 'current'}"
        if use_cache:
            cached = self._load_from_cache(cache_key)
            if cached:
                return cached

        # Fetch fresh weather data
        weather = self._fetch_weather_api(stadium_info, game_time)

        # Save to cache
        if use_cache:
            self._save_to_cache(cache_key, weather)

        return weather

    def _fetch_weather_api(
        self,
        stadium_info: Dict,
        game_time: Optional[datetime]
    ) -> WeatherData:
        """Fetch weather from API.

        Args:
            stadium_info: Stadium location info
            game_time: Game date/time

        Returns:
            WeatherData object
        """
        # NOTE: This requires a weather API key
        # Popular options:
        # - OpenWeatherMap (free tier available)
        # - WeatherAPI.com (free tier available)
        # - Weather.gov (free, no key needed, US only)

        # For now, return mock data
        # In production, would call actual weather API

        return WeatherData(
            location=f"{stadium_info['city']}, {stadium_info['state']}",
            game_time=game_time,
            temperature=55.0,  # Would fetch actual
            feels_like=52.0,
            wind_speed=8.0,
            wind_direction="NW",
            precipitation='none',
            humidity=65.0,
            conditions="Partly Cloudy",
            is_dome=stadium_info['dome'],
            is_retractable=stadium_info['retractable'],
            timestamp=datetime.now()
        )

    def _get_default_weather(self, team: str) -> WeatherData:
        """Get default weather (moderate conditions)."""
        return WeatherData(
            location=team,
            temperature=55.0,
            wind_speed=5.0,
            precipitation='none',
            is_dome=False,
            timestamp=datetime.now()
        )

    def _load_from_cache(self, cache_key: str) -> Optional[WeatherData]:
        """Load weather data from cache."""
        cache_file = self.cache_dir / f'{cache_key}.json'

        if not cache_file.exists():
            return None

        # Check if cache is still valid
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - mtime > self.cache_duration:
            return None

        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)

            return WeatherData(
                location=data['location'],
                game_time=datetime.fromisoformat(data['game_time']) if data.get('game_time') else None,
                temperature=data.get('temperature'),
                feels_like=data.get('feels_like'),
                wind_speed=data.get('wind_speed'),
                wind_direction=data.get('wind_direction'),
                precipitation=data.get('precipitation'),
                humidity=data.get('humidity'),
                conditions=data.get('conditions'),
                is_dome=data.get('is_dome', False),
                is_retractable=data.get('is_retractable', False),
                timestamp=datetime.fromisoformat(data['timestamp']) if data.get('timestamp') else None
            )

        except Exception as e:
            print(f"Error loading weather cache: {e}")
            return None

    def _save_to_cache(self, cache_key: str, weather: WeatherData):
        """Save weather data to cache."""
        cache_file = self.cache_dir / f'{cache_key}.json'

        try:
            data = {
                'location': weather.location,
                'game_time': weather.game_time.isoformat() if weather.game_time else None,
                'temperature': weather.temperature,
                'feels_like': weather.feels_like,
                'wind_speed': weather.wind_speed,
                'wind_direction': weather.wind_direction,
                'precipitation': weather.precipitation,
                'humidity': weather.humidity,
                'conditions': weather.conditions,
                'is_dome': weather.is_dome,
                'is_retractable': weather.is_retractable,
                'timestamp': weather.timestamp.isoformat() if weather.timestamp else None
            }

            with open(cache_file, 'w') as f:
                json.dump(data, f)

        except Exception as e:
            print(f"Error saving weather cache: {e}")


# Singleton instance
weather_scraper = WeatherScraper()


if __name__ == "__main__":
    # Test scraper
    scraper = WeatherScraper()

    # Test for outdoor stadium
    weather = scraper.fetch_weather('GB', game_time=datetime(2025, 12, 15, 13, 0))
    print(f"Weather for {weather.location}:")
    print(f"  Temperature: {weather.temperature}°F")
    print(f"  Wind: {weather.wind_speed} MPH {weather.wind_direction}")
    print(f"  Conditions: {weather.conditions}")
    print(f"  Dome: {weather.is_dome}")

    # Test for dome
    weather_dome = scraper.fetch_weather('NO')
    print(f"\nWeather for {weather_dome.location}:")
    print(f"  Dome: {weather_dome.is_dome}")
