"""Ingestion: fetch odds from OddsAPI

This script fetches NFL odds data from the OddsAPI and caches responses
as JSON files in cache/web_event_<id>.json format.

TODOs:
- Add OddsAPI key management (env var or config)
- Add CLI args (sport, markets, regions)
- Add retry/backoff for API calls
- Add rate limiting to respect API quotas
- Wire into orchestration/orchestrator
- Add data validation/schema checking
"""

from pathlib import Path
import argparse
import json
import os
import requests
from typing import List, Dict


def fetch_odds_api(sport: str = 'americanfootball_nfl',
                   markets: str = 'h2h,spreads,totals',
                   cache_dir: Path = Path('cache')) -> List[Dict]:
    """Fetch odds data from OddsAPI and cache results.

    Args:
        sport: Sport key (default: americanfootball_nfl)
        markets: Comma-separated market types
        cache_dir: Directory to cache API responses

    Returns:
        List of event dictionaries from the API

    TODO: Implement actual API call using requests
    Example API endpoint:
        https://api.the-odds-api.com/v4/sports/{sport}/odds
        ?apiKey={key}&regions=us&markets={markets}
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get('ODDS_API_KEY')

    if not api_key:
        print("ERROR: ODDS_API_KEY not set - cannot fetch odds data")
        print("Set environment variable: export ODDS_API_KEY=your_key_here")
        return []

    try:
        # Fetch odds from The Odds API
        url = f'https://api.the-odds-api.com/v4/sports/{sport}/odds'
        params = {
            'apiKey': api_key,
            'regions': 'us',
            'markets': markets,
            'oddsFormat': 'american'
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        events = response.json()

        # Show API quota usage
        requests_remaining = response.headers.get('x-requests-remaining', 'unknown')
        requests_used = response.headers.get('x-requests-used', 'unknown')
        print(f"API quota: {requests_used} used, {requests_remaining} remaining")

        # Cache each event
        for event in events:
            event_id = event.get('id', 'unknown')
            event_file = cache_dir / f"web_event_{event_id}.json"
            event_file.write_text(json.dumps(event, indent=2))

        print(f"Cached {len(events)} events to {cache_dir}")
        return events

    except requests.exceptions.RequestException as e:
        print(f"OddsAPI error: {e}")
        return []
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse API response: {e}")
        return []


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Fetch NFL odds from OddsAPI')
    p.add_argument('--sport', type=str, default='americanfootball_nfl',
                   help='Sport key (default: americanfootball_nfl)')
    p.add_argument('--markets', type=str, default='h2h,spreads,totals',
                   help='Comma-separated market types')
    p.add_argument('--cache', type=Path, default=Path('cache'),
                   help='Cache directory for API responses')
    args = p.parse_args()

    events = fetch_odds_api(args.sport, args.markets, args.cache)
    print(f"Fetched {len(events)} events")
