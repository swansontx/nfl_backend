"""Extract weather features for games.

Weather significantly impacts prop performance:
- Wind >15mph: Crushes passing yards, passing TDs, field goals
- Cold <32F: Reduces passing efficiency
- Rain/Snow: Lowers passing volume, increases rush attempts
- Dome: Eliminates weather variance (consistent scoring)

Research shows weather-adjusted models have 5-10% lower RMSE for outdoor props.
"""

from pathlib import Path
import json
import requests
from typing import Dict, Optional
from datetime import datetime


def fetch_weather_from_api(
    game_location: str,
    game_time: str,
    weather_api_key: Optional[str] = None
) -> Dict:
    """Fetch weather from external API (e.g., OpenWeather, WeatherAPI).

    Args:
        game_location: City name or coordinates
        game_time: Game datetime
        weather_api_key: API key for weather service

    Returns:
        Weather dict with temp, wind, precipitation
    """
    # Placeholder for actual weather API integration
    # In production, integrate with OpenWeather API, WeatherAPI, etc.

    weather = {
        'temp_f': 65,
        'wind_mph': 8,
        'precipitation': False,
        'conditions': 'Clear'
    }

    if weather_api_key:
        # TODO: Implement actual API call
        # Example: OpenWeather API
        # url = f"https://api.openweathermap.org/data/2.5/forecast"
        # params = {'q': game_location, 'appid': weather_api_key}
        # response = requests.get(url, params=params)
        pass

    return weather


def load_weather_from_nflverse(
    season: int,
    pbp_file: Path
) -> Dict:
    """Extract weather from nflverse play-by-play data.

    nflverse includes weather fields:
    - weather: Description (e.g., "Clear", "Rain")
    - temp: Temperature
    - wind: Wind speed

    Args:
        season: Season year
        pbp_file: Play-by-play parquet file

    Returns:
        Dict mapping game_id -> weather features
    """
    print(f"\n{'='*80}")
    print(f"EXTRACTING WEATHER FROM NFLVERSE - {season}")
    print(f"{'='*80}\n")

    import pandas as pd

    # Load PBP data
    print(f"📂 Loading play-by-play data: {pbp_file}")
    df = pd.read_parquet(pbp_file)

    # Extract unique games with weather data
    weather_data = {}

    for game_id in df['game_id'].unique():
        game_df = df[df['game_id'] == game_id]

        # Get first play's weather (should be consistent throughout game)
        first_play = game_df.iloc[0]

        weather_desc = first_play.get('weather', '')
        temp = first_play.get('temp', '')
        wind = first_play.get('wind', '')
        roof = first_play.get('roof', '')

        # Parse temperature
        try:
            temp_f = float(str(temp).replace('°', '').replace('F', '').strip()) if temp else 65
        except (ValueError, TypeError, AttributeError):
            temp_f = 65  # Default to mild temperature if parsing fails

        # Parse wind speed
        try:
            # Wind often formatted like "10 mph" or "10-15 mph"
            wind_str = str(wind).lower().replace('mph', '').strip()
            if '-' in wind_str:
                # Take average of range
                wind_parts = wind_str.split('-')
                wind_mph = (float(wind_parts[0]) + float(wind_parts[1])) / 2
            else:
                wind_mph = float(wind_str) if wind_str and wind_str != 'nan' else 5
        except (ValueError, TypeError, AttributeError, IndexError):
            wind_mph = 5  # Default to light wind if parsing fails

        # Check for precipitation
        precipitation = any(
            term in str(weather_desc).lower()
            for term in ['rain', 'snow', 'sleet', 'storm']
        )

        # Determine if dome/outdoors
        is_dome = roof in ['dome', 'closed'] if roof else False
        is_outdoors = not is_dome

        # Create weather features
        weather_data[game_id] = {
            'temp_f': round(temp_f, 1),
            'wind_mph': round(wind_mph, 1),
            'precipitation': 1 if precipitation else 0,
            'weather_desc': str(weather_desc) if weather_desc else 'Clear',

            # Buckets for model features
            'temp_bucket': _temp_bucket(temp_f),
            'wind_bucket': _wind_bucket(wind_mph),
            'wind_high': 1 if wind_mph >= 15 else 0,  # Critical threshold
            'temp_cold': 1 if temp_f < 45 else 0,
            'temp_hot': 1 if temp_f > 85 else 0,

            # Roof
            'is_dome': 1 if is_dome else 0,
            'is_outdoors': 1 if is_outdoors else 0
        }

    print(f"✓ Extracted weather for {len(weather_data)} games")

    return weather_data


def _temp_bucket(temp_f: float) -> str:
    """Bucket temperature into categories."""
    if temp_f < 32:
        return 'freezing'
    elif temp_f < 45:
        return 'cold'
    elif temp_f < 60:
        return 'cool'
    elif temp_f < 75:
        return 'mild'
    elif temp_f < 85:
        return 'warm'
    else:
        return 'hot'


def _wind_bucket(wind_mph: float) -> str:
    """Bucket wind speed into categories."""
    if wind_mph < 5:
        return 'calm'
    elif wind_mph < 10:
        return 'light'
    elif wind_mph < 15:
        return 'moderate'
    elif wind_mph < 20:
        return 'strong'
    else:
        return 'very_strong'


def add_weather_to_player_features(
    player_features: Dict,
    weather_data: Dict,
    stadium_features: Dict
) -> Dict:
    """Add weather features to player game records.

    Args:
        player_features: Player features dict
        weather_data: Weather data dict
        stadium_features: Stadium features dict

    Returns:
        Enhanced player features
    """
    print(f"\n{'='*80}")
    print("ADDING WEATHER FEATURES TO PLAYER GAMES")
    print(f"{'='*80}\n")

    enhanced_count = 0

    for player_id, games in player_features.items():
        for game in games:
            game_id = game.get('game_id', '')
            team = game.get('team', '')

            # Add weather if available
            if game_id and game_id in weather_data:
                game.update(weather_data[game_id])
                enhanced_count += 1
            else:
                # Fallback: Use stadium type
                if team in stadium_features:
                    is_dome = stadium_features[team]['is_dome']
                    game['is_dome'] = is_dome
                    game['is_outdoors'] = 1 - is_dome

                    if is_dome:
                        # Domes have no weather impact
                        game['temp_f'] = 72  # Climate controlled
                        game['wind_mph'] = 0
                        game['precipitation'] = 0
                        game['temp_bucket'] = 'mild'
                        game['wind_bucket'] = 'calm'
                        game['wind_high'] = 0
                        game['temp_cold'] = 0
                        game['temp_hot'] = 0

    print(f"✓ Added weather features to {enhanced_count} player-games")

    return player_features


def export_weather_features(
    season: int,
    pbp_file: Path,
    output_file: Path
) -> Dict:
    """Export weather features to JSON.

    Args:
        season: Season year
        pbp_file: Play-by-play parquet file
        output_file: Output JSON path

    Returns:
        Weather data dict
    """
    weather_data = load_weather_from_nflverse(
        season=season,
        pbp_file=pbp_file
    )

    # Save to JSON
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(weather_data, f, indent=2)

    print(f"\n✓ Saved weather features to: {output_file}")

    # Print summary stats
    wind_high_count = sum(1 for w in weather_data.values() if w['wind_high'])
    cold_count = sum(1 for w in weather_data.values() if w['temp_cold'])
    precip_count = sum(1 for w in weather_data.values() if w['precipitation'])
    dome_count = sum(1 for w in weather_data.values() if w['is_dome'])

    print(f"\nWeather Summary:")
    print(f"  High wind games (≥15mph): {wind_high_count}")
    print(f"  Cold games (<45F): {cold_count}")
    print(f"  Precipitation games: {precip_count}")
    print(f"  Dome games: {dome_count}")

    return weather_data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract weather features from nflverse'
    )
    parser.add_argument('--season', type=int, required=True)
    parser.add_argument('--pbp-file', type=Path, required=True,
                       help='Play-by-play parquet file')
    parser.add_argument('--player-features', type=Path, required=True,
                       help='Player features JSON')
    parser.add_argument('--stadium-features', type=Path,
                       help='Stadium features JSON (optional)')
    parser.add_argument('--output-dir', type=Path,
                       default=Path('outputs/features/weather'),
                       help='Output directory')

    args = parser.parse_args()

    # Extract weather
    weather_data = export_weather_features(
        season=args.season,
        pbp_file=args.pbp_file,
        output_file=args.output_dir / f'{args.season}_weather.json'
    )

    # Load player features
    with open(args.player_features, 'r') as f:
        player_features = json.load(f)

    # Load stadium features if provided
    stadium_features = {}
    if args.stadium_features and args.stadium_features.exists():
        with open(args.stadium_features, 'r') as f:
            stadium_features = json.load(f)

    # Add weather to player features
    enhanced_features = add_weather_to_player_features(
        player_features=player_features,
        weather_data=weather_data,
        stadium_features=stadium_features
    )

    # Save
    output_path = args.player_features.parent / f"{args.season}_player_features_with_weather.json"
    with open(output_path, 'w') as f:
        json.dump(enhanced_features, f, indent=2)

    print(f"\n✓ Saved enhanced features to: {output_path}")
