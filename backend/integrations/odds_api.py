"""
Odds API integration for sportsbook data

Fetches real-time odds from multiple sportsbooks:
- DraftKings
- FanDuel
- PrizePicks
- BetMGM
- Caesars

Uses The Odds API (https://the-odds-api.com/) for aggregated data.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import requests
from time import sleep

from backend.config import settings
from backend.config.logging_config import get_logger
from backend.database.session import get_db
from backend.database.models import MarketOdds

logger = get_logger(__name__)


class Sportsbook(Enum):
    """Supported sportsbooks"""
    DRAFTKINGS = "draftkings"
    FANDUEL = "fanduel"
    PRIZEPICKS = "prizepicks"
    BETMGM = "betmgm"
    CAESARS = "caesars"
    PINNACLE = "pinnacle"  # Sharpest book


@dataclass
class PropOdds:
    """Single prop odds from a sportsbook"""
    sportsbook: str
    player_id: str
    player_name: str
    market: str
    line: float
    over_odds: int  # American odds
    under_odds: int  # American odds
    timestamp: datetime

    # Derived
    over_prob: float  # Implied probability
    under_prob: float
    vig: float  # Vigorish (sportsbook edge)


@dataclass
class AggregatedPropOdds:
    """Aggregated odds across multiple sportsbooks"""
    player_id: str
    player_name: str
    market: str

    # Best available odds
    best_over_line: float
    best_over_odds: int
    best_over_book: str

    best_under_line: float
    best_under_odds: int
    best_under_book: str

    # Consensus (average across books)
    consensus_line: float
    consensus_prob: float

    # All book odds
    all_odds: List[PropOdds]

    # Metadata
    n_books: int
    line_spread: float  # Max - min line
    timestamp: datetime


class OddsAPIClient:
    """
    Client for The Odds API

    Fetches props, player props, and futures for NFL
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize odds API client

        Args:
            api_key: The Odds API key (get from settings if not provided)
        """
        self.api_key = api_key or settings.odds_api_key
        self.base_url = "https://api.the-odds-api.com/v4"

        self.sport = "americanfootball_nfl"
        self.regions = "us"
        self.markets = "player_pass_yds,player_pass_tds,player_rush_yds,player_rec_yds,player_receptions"

    def fetch_player_props(
        self,
        event_id: Optional[str] = None,
        bookmakers: Optional[List[str]] = None
    ) -> List[PropOdds]:
        """
        Fetch player props from The Odds API

        Args:
            event_id: Specific game ID (None = all upcoming games)
            bookmakers: List of bookmaker names (None = all)

        Returns:
            List of PropOdds
        """
        if not self.api_key:
            logger.warning("no_odds_api_key_configured")
            return []

        # Build endpoint
        if event_id:
            url = f"{self.base_url}/sports/{self.sport}/events/{event_id}/odds"
        else:
            url = f"{self.base_url}/sports/{self.sport}/odds"

        # Build params
        params = {
            "apiKey": self.api_key,
            "regions": self.regions,
            "markets": self.markets,
            "oddsFormat": "american",
        }

        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)

        try:
            logger.info("fetching_odds", event_id=event_id)

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Parse response
            props = self._parse_odds_response(data)

            logger.info("odds_fetched", count=len(props))

            return props

        except requests.RequestException as e:
            logger.error("odds_fetch_failed", error=str(e))
            return []

    def _parse_odds_response(self, data: List[Dict]) -> List[PropOdds]:
        """Parse The Odds API response into PropOdds"""
        props = []

        for event in data:
            event_id = event.get("id")
            event_time = datetime.fromisoformat(
                event.get("commence_time", "").replace("Z", "+00:00")
            ) if event.get("commence_time") else datetime.utcnow()

            # Iterate bookmakers
            for bookmaker in event.get("bookmakers", []):
                book_name = bookmaker.get("key")

                # Iterate markets
                for market in bookmaker.get("markets", []):
                    market_key = market.get("key")

                    # Iterate outcomes (players)
                    for outcome in market.get("outcomes", []):
                        player_name = outcome.get("description", "Unknown")
                        point = outcome.get("point")  # Line
                        price = outcome.get("price")  # Odds
                        name = outcome.get("name")  # Over/Under

                        # We need to pair Over and Under
                        # For now, create separate entries
                        # In production, you'd group by player+market

                        props.append(PropOdds(
                            sportsbook=book_name,
                            player_id="",  # Would map from player_name
                            player_name=player_name,
                            market=market_key,
                            line=point or 0,
                            over_odds=price if name == "Over" else 0,
                            under_odds=price if name == "Under" else 0,
                            timestamp=event_time,
                            over_prob=self._odds_to_prob(price) if name == "Over" else 0,
                            under_prob=self._odds_to_prob(price) if name == "Under" else 0,
                            vig=0  # Calculate later
                        ))

        return props

    def aggregate_props(self, props: List[PropOdds]) -> List[AggregatedPropOdds]:
        """
        Aggregate props across sportsbooks

        Finds best odds and consensus lines for each player-market
        """
        # Group by player + market
        grouped: Dict[tuple, List[PropOdds]] = {}

        for prop in props:
            key = (prop.player_name, prop.market)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(prop)

        # Aggregate
        aggregated = []

        for (player_name, market), book_props in grouped.items():
            if not book_props:
                continue

            # Get all lines and odds
            over_lines = [(p.line, p.over_odds, p.sportsbook) for p in book_props if p.over_odds != 0]
            under_lines = [(p.line, p.under_odds, p.sportsbook) for p in book_props if p.under_odds != 0]

            if not over_lines and not under_lines:
                continue

            # Best over
            if over_lines:
                # Best over = lowest line at good odds, or highest odds
                best_over = max(over_lines, key=lambda x: x[1])  # Highest odds
                best_over_line, best_over_odds, best_over_book = best_over
            else:
                best_over_line, best_over_odds, best_over_book = 0, 0, ""

            # Best under
            if under_lines:
                best_under = max(under_lines, key=lambda x: x[1])
                best_under_line, best_under_odds, best_under_book = best_under
            else:
                best_under_line, best_under_odds, best_under_book = 0, 0, ""

            # Consensus line (median)
            all_lines = [p.line for p in book_props if p.line > 0]
            consensus_line = sorted(all_lines)[len(all_lines) // 2] if all_lines else 0

            # Consensus prob (average implied prob)
            all_probs = [p.over_prob for p in book_props if p.over_prob > 0]
            consensus_prob = sum(all_probs) / len(all_probs) if all_probs else 0.5

            # Line spread
            line_spread = max(all_lines) - min(all_lines) if all_lines else 0

            agg = AggregatedPropOdds(
                player_id="",  # Would map
                player_name=player_name,
                market=market,
                best_over_line=best_over_line,
                best_over_odds=best_over_odds,
                best_over_book=best_over_book,
                best_under_line=best_under_line,
                best_under_odds=best_under_odds,
                best_under_book=best_under_book,
                consensus_line=consensus_line,
                consensus_prob=consensus_prob,
                all_odds=book_props,
                n_books=len(set(p.sportsbook for p in book_props)),
                line_spread=line_spread,
                timestamp=datetime.utcnow()
            )

            aggregated.append(agg)

        logger.info("props_aggregated", count=len(aggregated))

        return aggregated

    def save_odds_to_db(self, props: List[PropOdds]) -> int:
        """
        Save odds to database

        Args:
            props: List of PropOdds to save

        Returns:
            Number of records saved
        """
        saved = 0

        with get_db() as session:
            for prop in props:
                # Create MarketOdds record
                odds_record = MarketOdds(
                    player_id=prop.player_id,
                    market=prop.market,
                    sportsbook=prop.sportsbook,
                    line=prop.line,
                    over_odds=prop.over_odds,
                    under_odds=prop.under_odds,
                    over_prob=prop.over_prob,
                    under_prob=prop.under_prob,
                    timestamp=prop.timestamp
                )

                session.add(odds_record)
                saved += 1

            session.commit()

        logger.info("odds_saved_to_db", count=saved)

        return saved

    @staticmethod
    def _odds_to_prob(odds: int) -> float:
        """
        Convert American odds to implied probability

        Args:
            odds: American odds (e.g., -110, +150)

        Returns:
            Implied probability (0-1)
        """
        if odds < 0:
            # Favorite: -110 = 110/(110+100) = 0.524
            return -odds / (-odds + 100)
        elif odds > 0:
            # Underdog: +150 = 100/(150+100) = 0.40
            return 100 / (odds + 100)
        else:
            return 0.5

    @staticmethod
    def _prob_to_odds(prob: float) -> int:
        """
        Convert probability to American odds

        Args:
            prob: Probability (0-1)

        Returns:
            American odds
        """
        if prob >= 0.5:
            # Favorite
            return int(-(prob * 100) / (1 - prob))
        else:
            # Underdog
            return int(((1 - prob) * 100) / prob)

    def calculate_edge(
        self,
        model_prob: float,
        market_odds: int
    ) -> float:
        """
        Calculate edge (model prob vs market prob)

        Args:
            model_prob: Model's probability estimate
            market_odds: Market odds (American format)

        Returns:
            Edge (positive = +EV, negative = -EV)
        """
        market_prob = self._odds_to_prob(market_odds)
        edge = model_prob - market_prob

        return edge

    def calculate_closing_line_value(
        self,
        bet_odds: int,
        closing_odds: int
    ) -> float:
        """
        Calculate Closing Line Value (CLV)

        CLV > 0 = beat the market (good bet)
        CLV < 0 = lost to market (bad bet)

        Args:
            bet_odds: Odds when bet was placed
            closing_odds: Odds at game time

        Returns:
            CLV in probability points
        """
        bet_prob = self._odds_to_prob(bet_odds)
        closing_prob = self._odds_to_prob(closing_odds)

        clv = bet_prob - closing_prob

        return clv


class PrizePicksClient:
    """
    Specialized client for PrizePicks

    PrizePicks doesn't provide traditional odds, but rather
    "player projections" that users pick over/under.

    The "payout multiplier" implicitly encodes probability.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize PrizePicks client

        Note: PrizePicks doesn't have a public API
        This would require web scraping or unofficial endpoints
        """
        self.api_key = api_key
        self.base_url = "https://api.prizepicks.com"  # Unofficial

    def fetch_projections(self) -> List[Dict]:
        """
        Fetch player projections from PrizePicks

        Returns:
            List of player projection dicts
        """
        # Note: This is a placeholder
        # In production, you'd scrape their site or use unofficial API

        logger.warning("prizepicks_api_not_implemented")

        return []

    def convert_to_prop_odds(self, projections: List[Dict]) -> List[PropOdds]:
        """
        Convert PrizePicks projections to PropOdds format

        PrizePicks typically uses a multiplier system:
        - 2-pick: 3x payout
        - 3-pick: 5x payout
        - 4-pick: 10x payout

        Implied prob = 1 / (payout_multiplier ^ (1/n_picks))
        """
        props = []

        for proj in projections:
            # Extract projection data
            player_name = proj.get("player_name")
            stat_type = proj.get("stat_type")
            line = proj.get("line_score")

            # Assume 50/50 (no odds provided by PrizePicks)
            # They use multipliers instead
            implied_prob = 0.5

            prop = PropOdds(
                sportsbook="prizepicks",
                player_id=proj.get("player_id", ""),
                player_name=player_name,
                market=stat_type,
                line=line,
                over_odds=100,  # Even odds
                under_odds=100,
                timestamp=datetime.utcnow(),
                over_prob=implied_prob,
                under_prob=implied_prob,
                vig=0.05  # Estimated ~5% vig
            )

            props.append(prop)

        return props


# Example usage
def fetch_and_save_odds(game_id: Optional[str] = None) -> int:
    """
    Convenience function to fetch and save odds

    Args:
        game_id: Optional specific game

    Returns:
        Number of odds records saved
    """
    client = OddsAPIClient()

    # Fetch props
    props = client.fetch_player_props(event_id=game_id)

    if not props:
        logger.warning("no_props_fetched")
        return 0

    # Save to DB
    saved = client.save_odds_to_db(props)

    return saved
