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

    # TODO: Replace with actual API call
    # import requests
    # response = requests.get(
    #     f'https://api.the-odds-api.com/v4/sports/{sport}/odds',
    #     params={
    #         'apiKey': os.environ.get('ODDS_API_KEY'),
    #         'regions': 'us',
    #         'markets': markets
    #     }
    # )
    # events = response.json()

    # Placeholder: create sample event
    sample_event = {
        'id': 'sample_event_001',
        'sport_key': sport,
        'commence_time': '2025-11-24T18:00:00Z',
        'home_team': 'Buffalo Bills',
        'away_team': 'Kansas City Chiefs',
        'bookmakers': []
    }

    # Cache the event
    event_file = cache_dir / f"web_event_{sample_event['id']}.json"
    event_file.write_text(json.dumps(sample_event, indent=2))
    print(f"Cached event to {event_file}")

    return [sample_event]


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
