"""Integration with The Odds API for real-time sportsbook lines.

The Odds API provides live betting odds from multiple sportsbooks including:
- DraftKings
- FanDuel
- BetMGM
- Caesars
- PointsBet
- And more

API Documentation: https://the-odds-api.com/liveapi/guides/v4/
"""

import os
import requests
from typing import List, Dict, Optional
from datetime import datetime
from backend.api.cache import cached, CACHE_TTL
from backend.api.prop_analyzer import PropLine


class OddsAPI:
    """Integration with The Odds API for NFL prop betting lines."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Odds API client.

        Args:
            api_key: The Odds API key (defaults to env var ODDS_API_KEY)
        """
        self.api_key = api_key or os.getenv('ODDS_API_KEY')
        self.base_url = "https://api.the-odds-api.com/v4"
        self.sport = "americanfootball_nfl"

    @cached(ttl_seconds=CACHE_TTL['sportsbook_lines'])  # 1 minute
    def get_upcoming_games(self) -> List[Dict]:
        """Get list of upcoming NFL games.

        Returns:
            List of game events with IDs and teams
        """
        if not self.api_key:
            print("Warning: ODDS_API_KEY not set, returning empty list")
            return []

        try:
            url = f"{self.base_url}/sports/{self.sport}/events"
            params = {
                'apiKey': self.api_key,
                'dateFormat': 'iso'
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            print(f"Odds API error (get_upcoming_games): {e}")
            return []

    @cached(ttl_seconds=CACHE_TTL['sportsbook_lines'])  # 1 minute
    def get_player_props(self, event_id: Optional[str] = None) -> List[PropLine]:
        """Get player prop lines from The Odds API.

        Args:
            event_id: Specific event ID (optional, gets all if not provided)

        Returns:
            List of PropLine objects

        Player Prop Markets:
            - player_pass_yds: Passing yards
            - player_pass_tds: Passing touchdowns
            - player_pass_completions: Pass completions
            - player_rush_yds: Rushing yards
            - player_receptions: Receptions
            - player_reception_yds: Receiving yards
            - player_anytime_td: Anytime touchdown scorer
        """
        if not self.api_key:
            print("Warning: ODDS_API_KEY not set, returning empty list")
            return []

        try:
            # Player prop markets to fetch
            markets = [
                'player_pass_yds',
                'player_pass_tds',
                'player_pass_completions',
                'player_rush_yds',
                'player_receptions',
                'player_reception_yds'
            ]

            url = f"{self.base_url}/sports/{self.sport}/events"
            if event_id:
                url = f"{url}/{event_id}/odds"

            params = {
                'apiKey': self.api_key,
                'regions': 'us',  # US sportsbooks
                'markets': ','.join(markets),
                'oddsFormat': 'american',
                'dateFormat': 'iso'
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Parse response into PropLine objects
            prop_lines = []

            # Handle both single event and multiple events
            events = [data] if event_id else data

            for event in events:
                if not isinstance(event, dict):
                    continue

                bookmakers = event.get('bookmakers', [])

                for bookmaker in bookmakers:
                    book_name = bookmaker.get('title', 'Unknown')
                    markets_data = bookmaker.get('markets', [])

                    for market in markets_data:
                        market_key = market.get('key', '')
                        outcomes = market.get('outcomes', [])

                        # Convert market key to our prop type format
                        prop_type = self._market_to_prop_type(market_key)

                        for outcome in outcomes:
                            player_name = outcome.get('description', '')
                            point = outcome.get('point')
                            price = outcome.get('price')

                            if not all([player_name, point, price]):
                                continue

                            # The Odds API gives us one side at a time
                            # We need to fetch both over and under
                            # For now, create a PropLine with what we have
                            prop_lines.append(PropLine(
                                player_id=self._normalize_player_name(player_name),
                                player_name=player_name,
                                prop_type=prop_type,
                                line=float(point),
                                over_odds=price if outcome.get('name') == 'Over' else -110,
                                under_odds=price if outcome.get('name') == 'Under' else -110,
                                book=book_name,
                                timestamp=datetime.now().isoformat()
                            ))

            return prop_lines

        except Exception as e:
            print(f"Odds API error (get_player_props): {e}")
            return []

    def get_props_for_game(self, game_id: str) -> List[PropLine]:
        """Get player props for a specific game using our game_id format.

        Args:
            game_id: Game ID in format {season}_{week}_{away}_{home}

        Returns:
            List of PropLine objects for that game

        Note: This requires mapping our game_id to The Odds API event_id.
              For now, returns all props and you can filter by team.
        """
        # TODO: Map game_id to Odds API event_id
        # For now, get all props
        all_props = self.get_player_props()

        # Extract teams from game_id
        try:
            parts = game_id.split('_')
            if len(parts) == 4:
                away_team = parts[2]
                home_team = parts[3]

                # Filter props for players on these teams
                # (This is a simplified approach; ideal would be exact event matching)
                # For now, return all props
                return all_props
        except Exception:
            pass

        return all_props

    @staticmethod
    def _market_to_prop_type(market_key: str) -> str:
        """Convert Odds API market key to our prop_type format.

        Args:
            market_key: Market key from Odds API (e.g., 'player_pass_yds')

        Returns:
            Our prop_type format (e.g., 'passing_yards')
        """
        market_map = {
            'player_pass_yds': 'passing_yards',
            'player_pass_tds': 'passing_tds',
            'player_pass_completions': 'completions',
            'player_rush_yds': 'rushing_yards',
            'player_receptions': 'receptions',
            'player_reception_yds': 'receiving_yards',
            'player_anytime_td': 'anytime_td'
        }
        return market_map.get(market_key, market_key)

    @staticmethod
    def _normalize_player_name(name: str) -> str:
        """Normalize player name to create a consistent player_id.

        Args:
            name: Player name from Odds API

        Returns:
            Normalized name to use as player_id
        """
        # Convert to lowercase, replace spaces with underscores
        return name.lower().replace(' ', '_').replace('.', '')

    def get_best_line(self, player_name: str, prop_type: str) -> Optional[PropLine]:
        """Get the best available line for a player prop across all sportsbooks.

        Args:
            player_name: Player name
            prop_type: Type of prop (e.g., 'passing_yards')

        Returns:
            PropLine with the best odds, or None if not found
        """
        all_props = self.get_player_props()

        # Filter for this player and prop type
        matching_props = [
            prop for prop in all_props
            if prop.player_name.lower() == player_name.lower()
            and prop.prop_type == prop_type
        ]

        if not matching_props:
            return None

        # Find prop with best combined odds (highest over + highest under)
        best_prop = max(
            matching_props,
            key=lambda p: p.over_odds + p.under_odds
        )

        return best_prop

    def check_usage(self) -> Dict:
        """Check API usage and remaining requests.

        Returns:
            Dictionary with usage information
        """
        if not self.api_key:
            return {'error': 'No API key configured'}

        try:
            # The Odds API includes usage info in response headers
            url = f"{self.base_url}/sports/{self.sport}/events"
            params = {'apiKey': self.api_key}

            response = requests.get(url, params=params, timeout=10)

            return {
                'requests_remaining': response.headers.get('x-requests-remaining', 'unknown'),
                'requests_used': response.headers.get('x-requests-used', 'unknown'),
                'status': 'ok' if response.status_code == 200 else 'error'
            }

        except Exception as e:
            return {'error': str(e)}


# Singleton instance
odds_api = OddsAPI()
