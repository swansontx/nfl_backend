"""Ingestion: fetch odds from OddsAPI

This script fetches NFL player prop odds from the OddsAPI.

Requires:
- ODDS_API_KEY environment variable set
- pip install requests

Usage:
    python -m backend.ingestion.fetch_odds --markets player_pass_yds,player_rush_yds
"""

from pathlib import Path
import argparse
import json
import os
import time
from typing import List, Dict, Optional
from datetime import datetime

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    requests = None


# Available player prop markets
PLAYER_PROP_MARKETS = [
    'player_pass_yds',
    'player_pass_tds',
    'player_pass_completions',
    'player_pass_attempts',
    'player_interceptions',
    'player_rush_yds',
    'player_rush_attempts',
    'player_receptions',
    'player_reception_yds',
    'player_anytime_td',
]


def fetch_nfl_events(api_key: str) -> List[Dict]:
    """Fetch upcoming NFL events/games."""
    if not requests:
        return []

    url = 'https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events'
    response = requests.get(url, params={'apiKey': api_key})

    if response.status_code != 200:
        print(f"Error fetching events: {response.status_code} - {response.text}")
        return []

    return response.json()


def fetch_player_props(api_key: str,
                       event_id: str,
                       markets: List[str],
                       regions: str = 'us',
                       bookmakers: Optional[str] = None) -> Dict:
    """Fetch player prop odds for a specific game."""
    if not requests:
        return {}

    url = f'https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}/odds'

    params = {
        'apiKey': api_key,
        'regions': regions,
        'markets': ','.join(markets),
        'oddsFormat': 'american'
    }

    if bookmakers:
        params['bookmakers'] = bookmakers

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"Error fetching props for {event_id}: {response.status_code}")
        return {}

    remaining = response.headers.get('x-requests-remaining', 'unknown')
    print(f"API requests remaining: {remaining}")

    return response.json()


def fetch_all_props(api_key: str,
                    markets: List[str] = None,
                    regions: str = 'us',
                    bookmakers: str = 'draftkings,fanduel',
                    output_dir: Path = Path('inputs/odds'),
                    rate_limit_delay: float = 1.0) -> Dict:
    """Fetch player props for all upcoming NFL games."""
    if markets is None:
        markets = PLAYER_PROP_MARKETS

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching NFL events...")
    events = fetch_nfl_events(api_key)

    if not events:
        print("No events found")
        return {}

    print(f"Found {len(events)} upcoming games")

    all_odds = {
        'fetched_at': datetime.utcnow().isoformat(),
        'events': [],
        'props_by_player': {}
    }

    for event in events:
        event_id = event['id']
        home = event.get('home_team', 'Unknown')
        away = event.get('away_team', 'Unknown')

        print(f"\nFetching props for {away} @ {home}...")
        time.sleep(rate_limit_delay)

        props = fetch_player_props(api_key, event_id, markets, regions, bookmakers)

        if not props:
            continue

        event_data = {
            'event_id': event_id,
            'home_team': home,
            'away_team': away,
            'commence_time': event.get('commence_time'),
            'bookmakers': props.get('bookmakers', [])
        }

        all_odds['events'].append(event_data)

        # Organize by player
        for bookmaker in props.get('bookmakers', []):
            book_name = bookmaker.get('key', 'unknown')

            for market in bookmaker.get('markets', []):
                market_key = market.get('key', '')

                for outcome in market.get('outcomes', []):
                    player_name = outcome.get('description', '')

                    if not player_name:
                        continue

                    if player_name not in all_odds['props_by_player']:
                        all_odds['props_by_player'][player_name] = {}

                    if market_key not in all_odds['props_by_player'][player_name]:
                        all_odds['props_by_player'][player_name][market_key] = []

                    all_odds['props_by_player'][player_name][market_key].append({
                        'bookmaker': book_name,
                        'event_id': event_id,
                        'name': outcome.get('name'),
                        'point': outcome.get('point'),
                        'price': outcome.get('price'),
                    })

    # Save to file
    output_file = output_dir / f"player_props_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(output_file, 'w') as f:
        json.dump(all_odds, f, indent=2)

    print(f"\nSaved odds to {output_file}")

    # Also save latest version
    latest_file = output_dir / "player_props_latest.json"
    with open(latest_file, 'w') as f:
        json.dump(all_odds, f, indent=2)

    return all_odds


def load_latest_odds(odds_dir: Path = Path('inputs/odds')) -> Dict:
    """Load the most recent odds data."""
    latest_file = odds_dir / "player_props_latest.json"

    if not latest_file.exists():
        return {}

    with open(latest_file) as f:
        return json.load(f)


def get_player_line(player_name: str,
                    market: str,
                    odds_data: Dict = None,
                    bookmaker: str = 'draftkings') -> Optional[Dict]:
    """Get a player's line from odds data.

    Args:
        player_name: Player name
        market: Market type (e.g., 'player_pass_yds')
        odds_data: Loaded odds dict (loads latest if None)
        bookmaker: Preferred bookmaker

    Returns:
        Dict with line and odds or None
    """
    if odds_data is None:
        odds_data = load_latest_odds()

    props_by_player = odds_data.get('props_by_player', {})

    # Try exact match
    player_props = props_by_player.get(player_name, {})

    # Fuzzy match if not found
    if not player_props:
        player_lower = player_name.lower()
        for name, props in props_by_player.items():
            if player_lower in name.lower() or name.lower() in player_lower:
                player_props = props
                break

    if not player_props or market not in player_props:
        return None

    market_props = player_props[market]

    over_line = None
    under_line = None

    for prop in market_props:
        if prop['bookmaker'] == bookmaker or bookmaker is None:
            if prop['name'] == 'Over':
                over_line = {'line': prop['point'], 'odds': prop['price']}
            elif prop['name'] == 'Under':
                under_line = {'line': prop['point'], 'odds': prop['price']}

    if over_line and under_line:
        return {
            'over_line': over_line['line'],
            'over_odds': over_line['odds'],
            'under_line': under_line['line'],
            'under_odds': under_line['odds']
        }

    return None


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Fetch NFL odds from OddsAPI')
    p.add_argument('--markets', type=str, default=None,
                   help='Comma-separated player prop markets')
    p.add_argument('--bookmakers', type=str, default='draftkings,fanduel',
                   help='Comma-separated bookmakers')
    p.add_argument('--output', type=Path, default=Path('inputs/odds'),
                   help='Output directory')
    p.add_argument('--list-markets', action='store_true',
                   help='List available markets')
    args = p.parse_args()

    if args.list_markets:
        print("Available player prop markets:")
        for market in PLAYER_PROP_MARKETS:
            print(f"  - {market}")
        exit(0)

    api_key = os.environ.get('ODDS_API_KEY')

    if not api_key:
        print("ERROR: ODDS_API_KEY environment variable not set")
        print("\nTo get an API key:")
        print("1. Go to https://the-odds-api.com/")
        print("2. Sign up for free (500 requests/month)")
        print("3. Set: export ODDS_API_KEY=your_key_here")
        exit(1)

    markets = args.markets.split(',') if args.markets else None

    fetch_all_props(
        api_key,
        markets=markets,
        bookmakers=args.bookmakers,
        output_dir=args.output
    )
