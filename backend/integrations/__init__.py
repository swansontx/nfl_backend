"""Real-time data integrations for odds, news, and weather"""
from .odds_api import OddsAPIClient, PrizePicksClient, PropOdds, AggregatedPropOdds, Sportsbook
from .news_api import (
    AggregatedNewsClient,
    ESPNNewsClient,
    TwitterClient,
    RotoworldClient,
    FantasyProsClient,
    RawNewsItem,
    NewsSource,
)
from .weather_api import OpenWeatherMapClient, WeatherData, WeatherImpactAnalyzer, WeatherCondition

__all__ = [
    # Odds
    "OddsAPIClient",
    "PrizePicksClient",
    "PropOdds",
    "AggregatedPropOdds",
    "Sportsbook",
    # News
    "AggregatedNewsClient",
    "ESPNNewsClient",
    "TwitterClient",
    "RotoworldClient",
    "FantasyProsClient",
    "RawNewsItem",
    "NewsSource",
    # Weather
    "OpenWeatherMapClient",
    "WeatherData",
    "WeatherImpactAnalyzer",
    "WeatherCondition",
]
