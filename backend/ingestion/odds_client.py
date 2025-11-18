"""
Odds API client for fetching sportsbook lines and props

API Documentation: https://the-odds-api.com/liveapi/guides/v4/
"""

import httpx
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json

from backend.config import settings
from backend.config.logging_config import get_logger

logger = get_logger(__name__)


class OddsAPIClient:
    """
    Client for The Odds API

    Fetches:
    - Game lines (spread, total, moneyline)
    - Player props (passing yards, rushing yards, etc.)
    - Historical odds and line movements
    """

    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.odds_api_key
        if not self.api_key:
            logger.warning("odds_api_key_not_set")

        self.client = httpx.Client(timeout=30.0)
        self.cache_dir = Path(settings.data_cache_dir) / "odds"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_nfl_games(
        self,
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        date_format: str = "iso"
    ) -> List[Dict]:
        """
        Fetch NFL game odds

        Args:
            regions: Bookmaker regions ('us', 'uk', 'eu', 'au')
            markets: Markets to fetch ('h2h', 'spreads', 'totals')
            date_format: Date format ('iso' or 'unix')

        Returns:
            List of games with odds
        """
        url = f"{self.BASE_URL}/sports/americanfootball_nfl/odds"

        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "dateFormat": date_format,
        }

        try:
            response = self.client.get(url, params=params)
            response.raise_for_status()

            games = response.json()

            # Log remaining requests
            remaining = response.headers.get("x-requests-remaining")
            used = response.headers.get("x-requests-used")
            logger.info(
                "odds_fetched",
                games=len(games),
                requests_remaining=remaining,
                requests_used=used
            )

            return games

        except Exception as e:
            logger.error("odds_fetch_failed", error=str(e))
            raise

    def get_player_props(
        self,
        regions: str = "us",
        markets: Optional[List[str]] = None,
        use_cache: bool = True,
        cache_minutes: int = 5
    ) -> List[Dict]:
        """
        Fetch player prop odds

        Args:
            regions: Bookmaker regions
            markets: Specific prop markets to fetch (None = all)
            use_cache: Whether to use cached data
            cache_minutes: Cache TTL in minutes

        Returns:
            List of player props with odds from multiple books
        """
        # Default to common prop markets
        if markets is None:
            markets = [
                "player_pass_yds",
                "player_pass_tds",
                "player_pass_completions",
                "player_rush_yds",
                "player_rush_attempts",
                "player_rec_yds",
                "player_receptions",
                "player_anytime_td",
                "player_first_td",
            ]

        # Check cache
        cache_file = self.cache_dir / "player_props_latest.json"
        if use_cache and cache_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age < timedelta(minutes=cache_minutes):
                logger.info("loading_props_from_cache", age_seconds=cache_age.seconds)
                return json.loads(cache_file.read_text())

        all_props = []

        for market in markets:
            url = f"{self.BASE_URL}/sports/americanfootball_nfl/events/GAME_ID/odds"

            # Note: The Odds API requires event IDs for player props
            # In production, you'd first fetch events, then fetch props for each event
            # This is a simplified version

            logger.info("fetching_player_props", market=market)

            # For now, structure the expected response format
            # In production, make actual API calls per event
            props = self._fetch_market_props(market, regions)
            all_props.extend(props)

        # Cache results
        cache_file.write_text(json.dumps(all_props, indent=2))

        logger.info("player_props_fetched", count=len(all_props))

        return all_props

    def _fetch_market_props(self, market: str, regions: str) -> List[Dict]:
        """
        Fetch props for a specific market

        Note: This is a placeholder - actual implementation would fetch from API
        """
        # In production, make actual API calls here
        # For now, return empty list as placeholder
        return []

    def get_historical_odds(
        self,
        game_id: str,
        market: str = "spreads"
    ) -> List[Dict]:
        """
        Fetch historical odds for a game (line movement)

        Args:
            game_id: The Odds API game ID
            market: Market type

        Returns:
            List of odds snapshots showing line movement
        """
        # The Odds API has limited historical data in free tier
        # This would require premium subscription or separate data provider

        logger.warning("historical_odds_not_implemented", game_id=game_id)
        return []

    def parse_prop_line(
        self,
        prop_data: Dict,
        player_name: str,
        market: str
    ) -> Optional[Dict]:
        """
        Parse a player prop line from multiple books to find best line

        Args:
            prop_data: Raw prop data from API
            player_name: Player name to filter for
            market: Market type

        Returns:
            Dict with best available line info
        """
        # Find the player's prop across different books
        best_over = {"line": None, "odds": None, "book": None}
        best_under = {"line": None, "odds": None, "book": None}

        for bookmaker in prop_data.get("bookmakers", []):
            book_name = bookmaker["key"]

            for market_data in bookmaker.get("markets", []):
                if market_data["key"] != market:
                    continue

                for outcome in market_data.get("outcomes", []):
                    if outcome.get("description") != player_name:
                        continue

                    name = outcome["name"]  # "Over" or "Under"
                    line = outcome.get("point")
                    odds = outcome.get("price")  # American odds

                    if name == "Over":
                        if best_over["odds"] is None or odds > best_over["odds"]:
                            best_over = {"line": line, "odds": odds, "book": book_name}
                    elif name == "Under":
                        if best_under["odds"] is None or odds > best_under["odds"]:
                            best_under = {"line": line, "odds": odds, "book": book_name}

        if best_over["line"] is None:
            return None

        return {
            "player_name": player_name,
            "market": market,
            "line": best_over["line"],
            "over_odds": best_over["odds"],
            "over_book": best_over["book"],
            "under_odds": best_under["odds"],
            "under_book": best_under["book"],
            "timestamp": datetime.now().isoformat(),
        }

    def calculate_implied_probability(self, american_odds: int) -> float:
        """
        Convert American odds to implied probability

        Args:
            american_odds: American odds format (e.g., -110, +150)

        Returns:
            Implied probability (0-1)
        """
        if american_odds > 0:
            # Underdog: 100 / (odds + 100)
            return 100 / (american_odds + 100)
        else:
            # Favorite: |odds| / (|odds| + 100)
            return abs(american_odds) / (abs(american_odds) + 100)

    def calculate_fair_line(self, over_odds: int, under_odds: int) -> float:
        """
        Calculate fair line by removing vig

        Args:
            over_odds: American odds for over
            under_odds: American odds for under

        Returns:
            Fair probability for over
        """
        over_prob = self.calculate_implied_probability(over_odds)
        under_prob = self.calculate_implied_probability(under_odds)

        # Total implied probability includes vig
        total = over_prob + under_prob
        vig = total - 1.0

        # Remove vig proportionally
        fair_over = over_prob / total

        logger.debug(
            "fair_line_calculated",
            over_prob=over_prob,
            under_prob=under_prob,
            vig=vig,
            fair_over=fair_over
        )

        return fair_over

    def get_closing_line_value(
        self,
        bet_line: float,
        bet_odds: int,
        closing_line: float,
        closing_odds: int
    ) -> Dict:
        """
        Calculate closing line value (CLV)

        CLV is a key metric - if you consistently beat the closing line,
        you're likely to be profitable long-term

        Args:
            bet_line: Line when bet was placed
            bet_odds: Odds when bet was placed
            closing_line: Closing line before game
            closing_odds: Closing odds

        Returns:
            Dict with CLV metrics
        """
        bet_prob = self.calculate_implied_probability(bet_odds)
        closing_prob = self.calculate_implied_probability(closing_odds)

        # CLV is the difference in implied probability
        clv = closing_prob - bet_prob

        # Also calculate line value
        line_diff = closing_line - bet_line

        return {
            "bet_implied_prob": bet_prob,
            "closing_implied_prob": closing_prob,
            "clv": clv,
            "clv_pct": (clv / bet_prob) * 100 if bet_prob > 0 else 0,
            "line_diff": line_diff,
            "beat_closing": clv > 0,
        }

    def close(self):
        """Close HTTP client"""
        self.client.close()
