"""
Weather API integration for game conditions

Fetches weather data for NFL stadiums to assess impact on prop performance:
- Temperature, wind speed, precipitation
- Impact on passing vs rushing
- Impact on scoring totals

Uses OpenWeatherMap and Weather.gov APIs.
"""

from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import requests

from backend.config import settings
from backend.config.logging_config import get_logger

logger = get_logger(__name__)


class WeatherCondition(Enum):
    """Weather condition categories"""
    CLEAR = "clear"
    CLOUDY = "cloudy"
    RAIN = "rain"
    SNOW = "snow"
    WIND = "wind"  # High wind
    EXTREME = "extreme"  # Extreme cold/heat


@dataclass
class WeatherData:
    """Weather conditions for a game"""
    # Location
    stadium: str
    city: str
    latitude: float
    longitude: float

    # Weather conditions
    temperature: float  # Fahrenheit
    feels_like: float  # With wind chill/heat index
    wind_speed: float  # MPH
    wind_gust: Optional[float]  # MPH
    wind_direction: Optional[int]  # Degrees
    humidity: float  # Percent
    precipitation_prob: float  # Percent chance
    precipitation_amount: float  # Inches
    condition: str  # Description (clear, rain, snow, etc.)
    is_dome: bool  # Indoor stadium

    # Timestamp
    game_time: datetime
    forecast_time: datetime

    # Derived impact scores (0-1, higher = worse conditions)
    passing_difficulty: float
    kicking_difficulty: float
    overall_difficulty: float


# NFL stadium coordinates and dome status
STADIUM_DATA = {
    "Arrowhead Stadium": {"city": "Kansas City", "lat": 39.0489, "lon": -94.4839, "dome": False},
    "Lambeau Field": {"city": "Green Bay", "lat": 44.5013, "lon": -88.0622, "dome": False},
    "Soldier Field": {"city": "Chicago", "lat": 41.8623, "lon": -87.6167, "dome": False},
    "MetLife Stadium": {"city": "East Rutherford", "lat": 40.8128, "lon": -74.0742, "dome": False},
    "Gillette Stadium": {"city": "Foxborough", "lat": 42.0909, "lon": -71.2643, "dome": False},
    "M&T Bank Stadium": {"city": "Baltimore", "lat": 39.2780, "lon": -76.6227, "dome": False},
    "Highmark Stadium": {"city": "Buffalo", "lat": 42.7738, "lon": -78.7870, "dome": False},
    "Lincoln Financial Field": {"city": "Philadelphia", "lat": 39.9008, "lon": -75.1675, "dome": False},
    "Mercedes-Benz Stadium": {"city": "Atlanta", "lat": 33.7553, "lon": -84.4006, "dome": True},
    "AT&T Stadium": {"city": "Arlington", "lat": 32.7473, "lon": -97.0945, "dome": True},
    "SoFi Stadium": {"city": "Inglewood", "lat": 33.9535, "lon": -118.3390, "dome": True},
    "Allegiant Stadium": {"city": "Las Vegas", "lat": 36.0909, "lon": -115.1833, "dome": True},
    "U.S. Bank Stadium": {"city": "Minneapolis", "lat": 44.9738, "lon": -93.2575, "dome": True},
    "Ford Field": {"city": "Detroit", "lat": 42.3400, "lon": -83.0456, "dome": True},
    "Caesars Superdome": {"city": "New Orleans", "lat": 29.9511, "lon": -90.0812, "dome": True},
    "State Farm Stadium": {"city": "Glendale", "lat": 33.5276, "lon": -112.2626, "dome": True},
    "Lumen Field": {"city": "Seattle", "lat": 47.5952, "lon": -122.3316, "dome": False},
    "Empower Field": {"city": "Denver", "lat": 39.7439, "lon": -105.0201, "dome": False},
    "Hard Rock Stadium": {"city": "Miami", "lat": 25.9580, "lon": -80.2389, "dome": False},
    "Acrisure Stadium": {"city": "Pittsburgh", "lat": 40.4468, "lon": -80.0158, "dome": False},
    "Paycor Stadium": {"city": "Cincinnati", "lat": 39.0954, "lon": -84.5160, "dome": False},
    "Cleveland Browns Stadium": {"city": "Cleveland", "lat": 41.5061, "lon": -81.6995, "dome": False},
    "Lucas Oil Stadium": {"city": "Indianapolis", "lat": 39.7601, "lon": -86.1639, "dome": True},
    "Nissan Stadium": {"city": "Nashville", "lat": 36.1665, "lon": -86.7713, "dome": False},
    "TIAA Bank Field": {"city": "Jacksonville", "lat": 30.3239, "lon": -81.6373, "dome": False},
    "NRG Stadium": {"city": "Houston", "lat": 29.6847, "lon": -95.4107, "dome": True},
    "Raymond James Stadium": {"city": "Tampa", "lat": 27.9759, "lon": -82.5033, "dome": False},
    "Bank of America Stadium": {"city": "Charlotte", "lat": 35.2258, "lon": -80.8528, "dome": False},
    "FedExField": {"city": "Landover", "lat": 38.9076, "lon": -76.8645, "dome": False},
    "Levis Stadium": {"city": "Santa Clara", "lat": 37.4030, "lon": -121.9698, "dome": False},
}


class OpenWeatherMapClient:
    """
    Client for OpenWeatherMap API

    Fetches current and forecast weather data
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenWeatherMap client

        Args:
            api_key: OpenWeatherMap API key
        """
        self.api_key = api_key or settings.openweather_api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"

    def get_stadium_weather(
        self,
        stadium_name: str,
        game_time: Optional[datetime] = None
    ) -> Optional[WeatherData]:
        """
        Get weather for a stadium

        Args:
            stadium_name: Stadium name
            game_time: Game time (if future, uses forecast)

        Returns:
            WeatherData or None
        """
        if stadium_name not in STADIUM_DATA:
            logger.warning("unknown_stadium", stadium=stadium_name)
            return None

        stadium = STADIUM_DATA[stadium_name]

        # If dome, weather doesn't matter
        if stadium["dome"]:
            return WeatherData(
                stadium=stadium_name,
                city=stadium["city"],
                latitude=stadium["lat"],
                longitude=stadium["lon"],
                temperature=72.0,  # Perfect indoor conditions
                feels_like=72.0,
                wind_speed=0.0,
                wind_gust=None,
                wind_direction=None,
                humidity=50.0,
                precipitation_prob=0.0,
                precipitation_amount=0.0,
                condition="dome",
                is_dome=True,
                game_time=game_time or datetime.utcnow(),
                forecast_time=datetime.utcnow(),
                passing_difficulty=0.0,
                kicking_difficulty=0.0,
                overall_difficulty=0.0
            )

        # Determine if current or forecast
        now = datetime.utcnow()
        use_forecast = game_time and (game_time - now).total_seconds() > 3600  # More than 1 hour away

        if use_forecast:
            weather = self._get_forecast_weather(stadium["lat"], stadium["lon"], game_time)
        else:
            weather = self._get_current_weather(stadium["lat"], stadium["lon"])

        if not weather:
            return None

        # Calculate difficulty scores
        passing_difficulty = self._calculate_passing_difficulty(weather)
        kicking_difficulty = self._calculate_kicking_difficulty(weather)
        overall_difficulty = (passing_difficulty + kicking_difficulty) / 2

        # Build WeatherData
        weather_data = WeatherData(
            stadium=stadium_name,
            city=stadium["city"],
            latitude=stadium["lat"],
            longitude=stadium["lon"],
            temperature=weather["temp"],
            feels_like=weather["feels_like"],
            wind_speed=weather["wind_speed"],
            wind_gust=weather.get("wind_gust"),
            wind_direction=weather.get("wind_deg"),
            humidity=weather["humidity"],
            precipitation_prob=weather.get("pop", 0) * 100,
            precipitation_amount=weather.get("rain", 0) + weather.get("snow", 0),
            condition=weather["condition"],
            is_dome=False,
            game_time=game_time or now,
            forecast_time=now,
            passing_difficulty=passing_difficulty,
            kicking_difficulty=kicking_difficulty,
            overall_difficulty=overall_difficulty
        )

        return weather_data

    def _get_current_weather(self, lat: float, lon: float) -> Optional[Dict]:
        """Get current weather for coordinates"""
        if not self.api_key:
            logger.warning("no_openweather_api_key")
            return None

        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "imperial"  # Fahrenheit, MPH
        }

        try:
            response = requests.get(
                f"{self.base_url}/weather",
                params=params,
                timeout=10
            )
            response.raise_for_status()

            data = response.json()

            return {
                "temp": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "wind_speed": data["wind"]["speed"],
                "wind_gust": data["wind"].get("gust"),
                "wind_deg": data["wind"].get("deg"),
                "humidity": data["main"]["humidity"],
                "rain": data.get("rain", {}).get("1h", 0) / 25.4,  # mm to inches
                "snow": data.get("snow", {}).get("1h", 0) / 25.4,
                "condition": data["weather"][0]["main"].lower() if data.get("weather") else "clear"
            }

        except Exception as e:
            logger.error("weather_fetch_failed", error=str(e))
            return None

    def _get_forecast_weather(
        self,
        lat: float,
        lon: float,
        target_time: datetime
    ) -> Optional[Dict]:
        """Get forecast weather for coordinates at target time"""
        if not self.api_key:
            logger.warning("no_openweather_api_key")
            return None

        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "imperial"
        }

        try:
            response = requests.get(
                f"{self.base_url}/forecast",
                params=params,
                timeout=10
            )
            response.raise_for_status()

            data = response.json()

            # Find forecast closest to target time
            forecasts = data.get("list", [])

            if not forecasts:
                return None

            # Find closest forecast
            closest = min(
                forecasts,
                key=lambda f: abs(
                    (datetime.fromtimestamp(f["dt"]) - target_time).total_seconds()
                )
            )

            return {
                "temp": closest["main"]["temp"],
                "feels_like": closest["main"]["feels_like"],
                "wind_speed": closest["wind"]["speed"],
                "wind_gust": closest["wind"].get("gust"),
                "wind_deg": closest["wind"].get("deg"),
                "humidity": closest["main"]["humidity"],
                "pop": closest.get("pop", 0),  # Probability of precipitation
                "rain": closest.get("rain", {}).get("3h", 0) / 25.4 / 3,  # Per hour
                "snow": closest.get("snow", {}).get("3h", 0) / 25.4 / 3,
                "condition": closest["weather"][0]["main"].lower() if closest.get("weather") else "clear"
            }

        except Exception as e:
            logger.error("forecast_fetch_failed", error=str(e))
            return None

    def _calculate_passing_difficulty(self, weather: Dict) -> float:
        """
        Calculate passing difficulty (0-1 scale)

        Factors:
        - Wind speed (>15 MPH = difficult)
        - Precipitation
        - Extreme temps
        """
        difficulty = 0.0

        # Wind impact (most important for passing)
        wind = weather["wind_speed"]
        if wind > 20:
            difficulty += 0.5
        elif wind > 15:
            difficulty += 0.3
        elif wind > 10:
            difficulty += 0.1

        # Precipitation
        precip = weather.get("rain", 0) + weather.get("snow", 0)
        if precip > 0.2:
            difficulty += 0.3
        elif precip > 0.1:
            difficulty += 0.2

        # Snow is worse than rain
        if weather.get("snow", 0) > 0.1:
            difficulty += 0.2

        # Extreme cold (< 20°F)
        temp = weather["temp"]
        if temp < 20:
            difficulty += 0.2
        elif temp < 32:
            difficulty += 0.1

        # Cap at 1.0
        return min(difficulty, 1.0)

    def _calculate_kicking_difficulty(self, weather: Dict) -> float:
        """
        Calculate kicking difficulty (0-1 scale)

        Factors:
        - Wind (most important)
        - Cold (affects ball)
        """
        difficulty = 0.0

        # Wind impact (critical for kicking)
        wind = weather["wind_speed"]
        if wind > 25:
            difficulty += 0.6
        elif wind > 20:
            difficulty += 0.4
        elif wind > 15:
            difficulty += 0.2

        # Extreme cold
        temp = weather["temp"]
        if temp < 20:
            difficulty += 0.3
        elif temp < 32:
            difficulty += 0.1

        return min(difficulty, 1.0)


class WeatherImpactAnalyzer:
    """
    Analyzes weather impact on prop markets

    Provides adjustment factors based on weather conditions
    """

    def analyze_impact(self, weather: WeatherData) -> Dict[str, float]:
        """
        Calculate weather impact adjustments for each market

        Args:
            weather: WeatherData for game

        Returns:
            Dict mapping market to adjustment factor (1.0 = no impact, <1 = decrease, >1 = increase)
        """
        if weather.is_dome:
            # No weather impact in dome
            return {market: 1.0 for market in self._get_all_markets()}

        adjustments = {}

        # Passing markets (negatively impacted by weather)
        passing_factor = 1.0 - (weather.passing_difficulty * 0.3)  # Up to 30% decrease
        adjustments["player_pass_yds"] = passing_factor
        adjustments["player_pass_tds"] = passing_factor
        adjustments["player_pass_completions"] = passing_factor * 1.1  # Slightly less affected
        adjustments["player_rec_yds"] = passing_factor
        adjustments["player_receptions"] = passing_factor * 1.1

        # Rushing markets (benefit from bad weather)
        rushing_factor = 1.0 + (weather.passing_difficulty * 0.15)  # Up to 15% increase
        adjustments["player_rush_yds"] = rushing_factor
        adjustments["player_rush_attempts"] = rushing_factor

        # Kicking markets
        kicking_factor = 1.0 - (weather.kicking_difficulty * 0.4)  # Up to 40% decrease
        adjustments["player_field_goals"] = kicking_factor

        # TD markets (slightly affected)
        td_factor = 1.0 - (weather.overall_difficulty * 0.1)
        adjustments["player_anytime_td"] = td_factor
        adjustments["player_total_tds"] = td_factor

        return adjustments

    def _get_all_markets(self) -> List[str]:
        """Get list of all supported markets"""
        return [
            "player_pass_yds", "player_pass_tds", "player_pass_completions",
            "player_rec_yds", "player_receptions",
            "player_rush_yds", "player_rush_attempts",
            "player_anytime_td", "player_total_tds",
            "player_field_goals"
        ]

    def get_weather_flag(self, weather: WeatherData) -> Optional[str]:
        """
        Get weather flag for display

        Args:
            weather: WeatherData

        Returns:
            Weather flag string or None
        """
        if weather.is_dome:
            return None

        if weather.overall_difficulty >= 0.6:
            return "EXTREME_WEATHER"
        elif weather.wind_speed >= 20:
            return "HIGH_WIND"
        elif weather.precipitation_amount >= 0.2:
            return "HEAVY_PRECIP"
        elif weather.temperature <= 20:
            return "EXTREME_COLD"
        elif weather.overall_difficulty >= 0.3:
            return "MODERATE_WEATHER"
        else:
            return None


# Convenience function
def get_game_weather(
    stadium: str,
    game_time: Optional[datetime] = None
) -> Optional[WeatherData]:
    """
    Get weather for a game

    Args:
        stadium: Stadium name
        game_time: Game time

    Returns:
        WeatherData or None
    """
    client = OpenWeatherMapClient()
    return client.get_stadium_weather(stadium, game_time)
