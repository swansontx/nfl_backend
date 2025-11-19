"""Fetch prop betting lines from The Odds API.

Downloads player prop lines (passing yards, rushing yards, receiving yards, TDs, etc.)
from major sportsbooks for backtesting and live betting.

API Documentation: https://the-odds-api.com/

Prop Markets Available:
- player_pass_yds (QB passing yards)
- player_pass_tds (QB passing TDs)
- player_pass_completions (QB completions)
- player_rush_yds (RB/QB rushing yards)
- player_rush_tds (Rushing TDs)
- player_receptions (Receptions)
- player_reception_yds (Receiving yards)
- player_reception_tds (Receiving TDs)
- player_anytime_td (Anytime TD scorer)

Output: JSON files with prop lines by game and player
"""

from pathlib import Path
import argparse
import requests
import json
import time
from typing import Dict, List, Optional
from datetime import datetime
import os


class PropLineFetcher:
    """Fetches prop betting lines from The Odds API."""

    def __init__(self, api_key: str):
        """Initialize with API key.

        Args:
            api_key: The Odds API key
        """
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4"
        self.sport = "americanfootball_nfl"

        # Prop markets to fetch
        self.prop_markets = [
            'player_pass_yds',
            'player_pass_tds',
            'player_pass_completions',
            'player_rush_yds',
            'player_rush_tds',
            'player_receptions',
            'player_reception_yds',
            'player_anytime_td'
        ]

        # Sportsbooks to include (major US books)
        self.bookmakers = [
            'draftkings',
            'fanduel',
            'williamhill_us',  # Caesars
            'betmgm',
            'pointsbetus',
            'unibet_us'
        ]

    def fetch_upcoming_games(self) -> List[Dict]:
        """Fetch list of upcoming NFL games.

        Returns:
            List of game dictionaries with game_id, commence_time, teams
        """
        url = f"{self.base_url}/sports/{self.sport}/events"
        params = {
            'apiKey': self.api_key,
            'dateFormat': 'iso'
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            games = response.json()
            print(f"✓ Found {len(games)} upcoming NFL games")

            return games

        except requests.exceptions.RequestException as e:
            print(f"✗ Error fetching games: {e}")
            return []

    def fetch_prop_odds_for_game(
        self,
        game_id: str,
        market: str,
        bookmakers: Optional[List[str]] = None
    ) -> Dict:
        """Fetch prop odds for a specific game and market.

        Args:
            game_id: The Odds API game ID
            market: Prop market (e.g., 'player_pass_yds')
            bookmakers: List of bookmaker keys (optional)

        Returns:
            Dict with prop odds data
        """
        url = f"{self.base_url}/sports/{self.sport}/events/{game_id}/odds"

        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': market,
            'oddsFormat': 'american',
            'dateFormat': 'iso'
        }

        if bookmakers:
            params['bookmakers'] = ','.join(bookmakers)

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Check remaining requests
            remaining = response.headers.get('x-requests-remaining', 'unknown')
            print(f"  {market}: {remaining} requests remaining")

            return data

        except requests.exceptions.RequestException as e:
            print(f"  ✗ Error fetching {market} for {game_id}: {e}")
            return {}

    def process_prop_data(self, raw_data: Dict, market: str) -> List[Dict]:
        """Process raw API response into structured prop data.

        Args:
            raw_data: Raw API response
            market: Market type

        Returns:
            List of processed prop dictionaries
        """
        if not raw_data or 'bookmakers' not in raw_data:
            return []

        processed_props = []

        for bookmaker in raw_data.get('bookmakers', []):
            bookmaker_name = bookmaker.get('key', '')

            for market_data in bookmaker.get('markets', []):
                for outcome in market_data.get('outcomes', []):
                    player_name = outcome.get('description', '')
                    point = outcome.get('point')
                    price = outcome.get('price')  # American odds

                    if player_name and point is not None:
                        processed_props.append({
                            'player_name': player_name,
                            'market': market,
                            'bookmaker': bookmaker_name,
                            'line': point,
                            'odds': price,
                            'timestamp': datetime.now().isoformat()
                        })

        return processed_props

    def fetch_all_props_for_week(
        self,
        output_dir: Path,
        week: Optional[int] = None
    ) -> Dict:
        """Fetch all prop lines for upcoming week.

        Args:
            output_dir: Directory to save prop data
            week: Week number (for filename)

        Returns:
            Dict with summary of fetched props
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Fetching NFL Player Props")
        print(f"{'='*60}\n")

        # Get upcoming games
        games = self.fetch_upcoming_games()

        if not games:
            print("⚠️  No upcoming games found")
            return {'games': 0, 'props': 0}

        all_props_by_game = {}
        total_props = 0

        for game in games:
            game_id = game.get('id')
            home_team = game.get('home_team', 'Unknown')
            away_team = game.get('away_team', 'Unknown')
            commence_time = game.get('commence_time', '')

            print(f"\n{away_team} @ {home_team}")
            print(f"  Game ID: {game_id}")
            print(f"  Kickoff: {commence_time}")

            game_props = {
                'game_id': game_id,
                'home_team': home_team,
                'away_team': away_team,
                'commence_time': commence_time,
                'props': {}
            }

            # Fetch each prop market
            for market in self.prop_markets:
                raw_data = self.fetch_prop_odds_for_game(
                    game_id,
                    market,
                    self.bookmakers
                )

                if raw_data:
                    props = self.process_prop_data(raw_data, market)
                    if props:
                        game_props['props'][market] = props
                        total_props += len(props)
                        print(f"    ✓ {market}: {len(props)} props")

                # Rate limit: sleep between requests
                time.sleep(0.5)

            all_props_by_game[game_id] = game_props

        # Save to file
        week_str = f"week_{week}" if week else "current"
        output_file = output_dir / f"prop_lines_{week_str}.json"

        with open(output_file, 'w') as f:
            json.dump(all_props_by_game, f, indent=2)

        print(f"\n{'='*60}")
        print(f"✓ Fetched props for {len(games)} games")
        print(f"✓ Total props: {total_props}")
        print(f"✓ Saved to: {output_file}")
        print(f"{'='*60}\n")

        return {
            'games': len(games),
            'props': total_props,
            'output_file': str(output_file)
        }

    def get_player_prop_history(
        self,
        player_name: str,
        prop_type: str,
        historical_files: List[Path]
    ) -> List[Dict]:
        """Get historical prop lines for a specific player.

        Args:
            player_name: Player name
            prop_type: Prop market (e.g., 'player_pass_yds')
            historical_files: List of historical prop JSON files

        Returns:
            List of historical prop lines
        """
        history = []

        for file_path in historical_files:
            if not file_path.exists():
                continue

            with open(file_path, 'r') as f:
                data = json.load(f)

            for game_id, game_data in data.items():
                props = game_data.get('props', {}).get(prop_type, [])

                for prop in props:
                    if prop.get('player_name', '').lower() == player_name.lower():
                        prop['game_id'] = game_id
                        prop['game_date'] = game_data.get('commence_time', '')
                        history.append(prop)

        return sorted(history, key=lambda x: x.get('game_date', ''))


def fetch_prop_lines(
    api_key: str,
    output_dir: Path,
    week: Optional[int] = None
) -> Dict:
    """Main function to fetch prop betting lines.

    Args:
        api_key: The Odds API key
        output_dir: Output directory
        week: Week number (optional)

    Returns:
        Summary dict
    """
    if not api_key:
        print("⚠️  No API key provided")
        print("Set ODDS_API_KEY environment variable or pass --api-key")
        return {'error': 'No API key'}

    fetcher = PropLineFetcher(api_key)
    return fetcher.fetch_all_props_for_week(output_dir, week)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fetch NFL player prop betting lines from The Odds API'
    )
    parser.add_argument('--api-key', type=str, default=None,
                       help='The Odds API key (or use ODDS_API_KEY env var)')
    parser.add_argument('--output', type=Path, default=Path('inputs/prop_lines'),
                       help='Output directory (default: inputs/prop_lines/)')
    parser.add_argument('--week', type=int, default=None,
                       help='Week number (for filename)')
    args = parser.parse_args()

    # Get API key from args or environment
    api_key = args.api_key or os.getenv('ODDS_API_KEY')

    if not api_key:
        print("\n⚠️  ERROR: No API key provided")
        print("\nTo use The Odds API:")
        print("1. Get a free API key at https://the-odds-api.com/")
        print("2. Set environment variable: export ODDS_API_KEY='your_key'")
        print("3. Or pass as argument: --api-key your_key")
        print("\nNote: Free tier includes 500 requests/month")
        exit(1)

    fetch_prop_lines(api_key, args.output, args.week)
