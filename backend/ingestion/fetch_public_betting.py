"""Public Betting Data Scraper.

Scrapes public betting percentages (bet% and money%) from SportsBettingDime.
Data includes spreads, totals, and moneylines for all NFL games.
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time
import json
from pathlib import Path
import re


@dataclass
class PublicBettingMarket:
    """Public betting data for a single market (spread/total/ML)."""
    market_type: str  # 'spread', 'total', 'moneyline'
    home_bet_pct: Optional[float] = None  # % of bets on home team
    home_money_pct: Optional[float] = None  # % of money on home team
    away_bet_pct: Optional[float] = None  # % of bets on away team
    away_money_pct: Optional[float] = None  # % of money on away team
    over_bet_pct: Optional[float] = None  # % of bets on over
    over_money_pct: Optional[float] = None  # % of money on over
    under_bet_pct: Optional[float] = None  # % of bets on under
    under_money_pct: Optional[float] = None  # % of money on under
    line: Optional[float] = None  # Current line


@dataclass
class PublicBettingData:
    """Complete public betting data for a game."""
    game_id: str
    home_team: str
    away_team: str

    spread: Optional[PublicBettingMarket] = None
    total: Optional[PublicBettingMarket] = None
    moneyline: Optional[PublicBettingMarket] = None

    # Sharp money indicators
    spread_sharp_on_home: bool = False
    spread_sharp_on_away: bool = False
    total_sharp_on_over: bool = False
    total_sharp_on_under: bool = False
    ml_sharp_on_home: bool = False
    ml_sharp_on_away: bool = False

    # Contrarian opportunities
    spread_contrarian_home: bool = False
    spread_contrarian_away: bool = False
    total_contrarian_over: bool = False
    total_contrarian_under: bool = False

    timestamp: Optional[datetime] = None


class PublicBettingScraper:
    """Scraper for public betting percentages."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize scraper.

        Args:
            cache_dir: Directory for caching responses (default: backend/.cache)
        """
        self.cache_dir = cache_dir or Path(__file__).parent.parent / '.cache'
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_duration = timedelta(minutes=15)  # Cache for 15 minutes

        # Team abbreviation mapping
        self.team_mapping = {
            'Arizona Cardinals': 'ARI', 'Arizona': 'ARI', 'Cardinals': 'ARI',
            'Atlanta Falcons': 'ATL', 'Atlanta': 'ATL', 'Falcons': 'ATL',
            'Baltimore Ravens': 'BAL', 'Baltimore': 'BAL', 'Ravens': 'BAL',
            'Buffalo Bills': 'BUF', 'Buffalo': 'BUF', 'Bills': 'BUF',
            'Carolina Panthers': 'CAR', 'Carolina': 'CAR', 'Panthers': 'CAR',
            'Chicago Bears': 'CHI', 'Chicago': 'CHI', 'Bears': 'CHI',
            'Cincinnati Bengals': 'CIN', 'Cincinnati': 'CIN', 'Bengals': 'CIN',
            'Cleveland Browns': 'CLE', 'Cleveland': 'CLE', 'Browns': 'CLE',
            'Dallas Cowboys': 'DAL', 'Dallas': 'DAL', 'Cowboys': 'DAL',
            'Denver Broncos': 'DEN', 'Denver': 'DEN', 'Broncos': 'DEN',
            'Detroit Lions': 'DET', 'Detroit': 'DET', 'Lions': 'DET',
            'Green Bay Packers': 'GB', 'Green Bay': 'GB', 'Packers': 'GB',
            'Houston Texans': 'HOU', 'Houston': 'HOU', 'Texans': 'HOU',
            'Indianapolis Colts': 'IND', 'Indianapolis': 'IND', 'Colts': 'IND',
            'Jacksonville Jaguars': 'JAX', 'Jacksonville': 'JAX', 'Jaguars': 'JAX',
            'Kansas City Chiefs': 'KC', 'Kansas City': 'KC', 'Chiefs': 'KC',
            'Las Vegas Raiders': 'LV', 'Las Vegas': 'LV', 'Raiders': 'LV',
            'Los Angeles Chargers': 'LAC', 'LA Chargers': 'LAC', 'Chargers': 'LAC',
            'Los Angeles Rams': 'LAR', 'LA Rams': 'LAR', 'Rams': 'LAR',
            'Miami Dolphins': 'MIA', 'Miami': 'MIA', 'Dolphins': 'MIA',
            'Minnesota Vikings': 'MIN', 'Minnesota': 'MIN', 'Vikings': 'MIN',
            'New England Patriots': 'NE', 'New England': 'NE', 'Patriots': 'NE',
            'New Orleans Saints': 'NO', 'New Orleans': 'NO', 'Saints': 'NO',
            'New York Giants': 'NYG', 'NY Giants': 'NYG', 'Giants': 'NYG',
            'New York Jets': 'NYJ', 'NY Jets': 'NYJ', 'Jets': 'NYJ',
            'Philadelphia Eagles': 'PHI', 'Philadelphia': 'PHI', 'Eagles': 'PHI',
            'Pittsburgh Steelers': 'PIT', 'Pittsburgh': 'PIT', 'Steelers': 'PIT',
            'San Francisco 49ers': 'SF', 'San Francisco': 'SF', '49ers': 'SF',
            'Seattle Seahawks': 'SEA', 'Seattle': 'SEA', 'Seahawks': 'SEA',
            'Tampa Bay Buccaneers': 'TB', 'Tampa Bay': 'TB', 'Buccaneers': 'TB',
            'Tennessee Titans': 'TEN', 'Tennessee': 'TEN', 'Titans': 'TEN',
            'Washington Commanders': 'WAS', 'Washington': 'WAS', 'Commanders': 'WAS'
        }

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f'public_betting_{cache_key}.json'

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache is still valid."""
        if not cache_path.exists():
            return False

        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - mtime < self.cache_duration

    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load data from cache if valid."""
        cache_path = self._get_cache_path(cache_key)

        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
                return None

        return None

    def _save_to_cache(self, cache_key: str, data: Dict):
        """Save data to cache."""
        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error saving cache: {e}")

    def _normalize_team_name(self, team_name: str) -> Optional[str]:
        """Convert team name to abbreviation."""
        team_name = team_name.strip()

        # Direct match
        if team_name in self.team_mapping:
            return self.team_mapping[team_name]

        # Try partial match
        for full_name, abbrev in self.team_mapping.items():
            if team_name.lower() in full_name.lower() or full_name.lower() in team_name.lower():
                return abbrev

        return None

    def _detect_sharp_money(
        self,
        bet_pct: Optional[float],
        money_pct: Optional[float],
        threshold: float = 15.0
    ) -> bool:
        """Detect sharp money (money% significantly higher than bet%).

        Args:
            bet_pct: Percentage of bets
            money_pct: Percentage of money
            threshold: Minimum difference to indicate sharp money

        Returns:
            True if sharp money detected
        """
        if bet_pct is None or money_pct is None:
            return False

        # Sharp money = fewer bets but more money
        return money_pct - bet_pct > threshold

    def _detect_contrarian_opportunity(
        self,
        bet_pct: Optional[float],
        threshold: float = 75.0
    ) -> bool:
        """Detect contrarian betting opportunity (heavy public on one side).

        Args:
            bet_pct: Percentage of bets
            threshold: Minimum percentage to indicate heavy public action

        Returns:
            True if contrarian opportunity exists
        """
        if bet_pct is None:
            return False

        return bet_pct > threshold

    def fetch_sportsbettingdime(
        self,
        week: Optional[int] = None,
        use_cache: bool = True
    ) -> Dict[str, PublicBettingData]:
        """Fetch public betting data from SportsBettingDime.

        Args:
            week: NFL week number (for cache key)
            use_cache: Whether to use cached data

        Returns:
            Dictionary mapping game_id to PublicBettingData
        """
        cache_key = f"sbd_week_{week or 'current'}"

        # Try cache first
        if use_cache:
            cached = self._load_from_cache(cache_key)
            if cached:
                print(f"Using cached public betting data")
                return self._deserialize_data(cached)

        print("Fetching fresh public betting data from SportsBettingDime...")

        url = "https://www.sportsbettingdime.com/nfl/public-betting-trends/"

        try:
            # Fetch with retry logic
            response = self._fetch_with_retry(url)

            if response.status_code != 200:
                print(f"Failed to fetch public betting data: HTTP {response.status_code}")
                return {}

            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract game data (this will need to be adjusted based on actual HTML structure)
            games_data = self._parse_sportsbettingdime_html(soup)

            # Cache the results
            if use_cache:
                serialized = self._serialize_data(games_data)
                self._save_to_cache(cache_key, serialized)

            return games_data

        except Exception as e:
            print(f"Error fetching public betting data: {e}")
            return {}

    def _fetch_with_retry(
        self,
        url: str,
        max_retries: int = 3,
        backoff: float = 2.0
    ) -> requests.Response:
        """Fetch URL with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    url,
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'},
                    timeout=10
                )
                return response
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = backoff ** attempt
                print(f"Retry {attempt + 1}/{max_retries} after {wait_time}s...")
                time.sleep(wait_time)

        raise requests.RequestException("Max retries exceeded")

    def _parse_sportsbettingdime_html(self, soup: BeautifulSoup) -> Dict[str, PublicBettingData]:
        """Parse SportsBettingDime HTML to extract public betting data.

        Note: This is a template. The actual parsing logic needs to be
        implemented based on the real HTML structure of the website.
        """
        games_data = {}

        # TODO: Implement actual parsing based on SportsBettingDime's HTML structure
        # For now, this is a placeholder that returns empty data

        # The structure typically involves finding tables or divs containing:
        # - Team names
        # - Bet percentages for spread/total/ML
        # - Money percentages for spread/total/ML
        # - Current lines

        # Example pseudo-code:
        # games = soup.find_all('div', class_='game-row')
        # for game in games:
        #     home_team = extract_home_team(game)
        #     away_team = extract_away_team(game)
        #     spread_data = extract_spread_betting(game)
        #     total_data = extract_total_betting(game)
        #     ml_data = extract_ml_betting(game)
        #
        #     Create PublicBettingData object and add to games_data

        print("Warning: HTML parsing not yet implemented. Using placeholder.")
        return games_data

    def fetch_covers(
        self,
        week: Optional[int] = None,
        use_cache: bool = True
    ) -> Dict[str, PublicBettingData]:
        """Fetch public betting data from Covers.com (alternative source).

        Args:
            week: NFL week number (for cache key)
            use_cache: Whether to use cached data

        Returns:
            Dictionary mapping game_id to PublicBettingData
        """
        cache_key = f"covers_week_{week or 'current'}"

        # Try cache first
        if use_cache:
            cached = self._load_from_cache(cache_key)
            if cached:
                print(f"Using cached public betting data from Covers")
                return self._deserialize_data(cached)

        print("Fetching fresh public betting data from Covers.com...")

        url = "https://contests.covers.com/consensus/topconsensus/nfl/overall"

        try:
            response = self._fetch_with_retry(url)

            if response.status_code != 200:
                print(f"Failed to fetch from Covers: HTTP {response.status_code}")
                return {}

            soup = BeautifulSoup(response.content, 'html.parser')
            games_data = self._parse_covers_html(soup)

            if use_cache:
                serialized = self._serialize_data(games_data)
                self._save_to_cache(cache_key, serialized)

            return games_data

        except Exception as e:
            print(f"Error fetching from Covers: {e}")
            return {}

    def _parse_covers_html(self, soup: BeautifulSoup) -> Dict[str, PublicBettingData]:
        """Parse Covers.com HTML."""
        # TODO: Implement based on actual Covers HTML structure
        print("Warning: Covers parsing not yet implemented. Using placeholder.")
        return {}

    def _serialize_data(self, games_data: Dict[str, PublicBettingData]) -> Dict:
        """Serialize PublicBettingData objects to JSON-compatible dict."""
        serialized = {}

        for game_id, data in games_data.items():
            serialized[game_id] = {
                'game_id': data.game_id,
                'home_team': data.home_team,
                'away_team': data.away_team,
                'spread': self._serialize_market(data.spread),
                'total': self._serialize_market(data.total),
                'moneyline': self._serialize_market(data.moneyline),
                'spread_sharp_on_home': data.spread_sharp_on_home,
                'spread_sharp_on_away': data.spread_sharp_on_away,
                'total_sharp_on_over': data.total_sharp_on_over,
                'total_sharp_on_under': data.total_sharp_on_under,
                'ml_sharp_on_home': data.ml_sharp_on_home,
                'ml_sharp_on_away': data.ml_sharp_on_away,
                'spread_contrarian_home': data.spread_contrarian_home,
                'spread_contrarian_away': data.spread_contrarian_away,
                'total_contrarian_over': data.total_contrarian_over,
                'total_contrarian_under': data.total_contrarian_under,
                'timestamp': data.timestamp.isoformat() if data.timestamp else None
            }

        return serialized

    def _serialize_market(self, market: Optional[PublicBettingMarket]) -> Optional[Dict]:
        """Serialize a single market."""
        if market is None:
            return None

        return {
            'market_type': market.market_type,
            'home_bet_pct': market.home_bet_pct,
            'home_money_pct': market.home_money_pct,
            'away_bet_pct': market.away_bet_pct,
            'away_money_pct': market.away_money_pct,
            'over_bet_pct': market.over_bet_pct,
            'over_money_pct': market.over_money_pct,
            'under_bet_pct': market.under_bet_pct,
            'under_money_pct': market.under_money_pct,
            'line': market.line
        }

    def _deserialize_data(self, serialized: Dict) -> Dict[str, PublicBettingData]:
        """Deserialize JSON dict back to PublicBettingData objects."""
        games_data = {}

        for game_id, data in serialized.items():
            games_data[game_id] = PublicBettingData(
                game_id=data['game_id'],
                home_team=data['home_team'],
                away_team=data['away_team'],
                spread=self._deserialize_market(data.get('spread')),
                total=self._deserialize_market(data.get('total')),
                moneyline=self._deserialize_market(data.get('moneyline')),
                spread_sharp_on_home=data.get('spread_sharp_on_home', False),
                spread_sharp_on_away=data.get('spread_sharp_on_away', False),
                total_sharp_on_over=data.get('total_sharp_on_over', False),
                total_sharp_on_under=data.get('total_sharp_on_under', False),
                ml_sharp_on_home=data.get('ml_sharp_on_home', False),
                ml_sharp_on_away=data.get('ml_sharp_on_away', False),
                spread_contrarian_home=data.get('spread_contrarian_home', False),
                spread_contrarian_away=data.get('spread_contrarian_away', False),
                total_contrarian_over=data.get('total_contrarian_over', False),
                total_contrarian_under=data.get('total_contrarian_under', False),
                timestamp=datetime.fromisoformat(data['timestamp']) if data.get('timestamp') else None
            )

        return games_data

    def _deserialize_market(self, data: Optional[Dict]) -> Optional[PublicBettingMarket]:
        """Deserialize a single market."""
        if data is None:
            return None

        return PublicBettingMarket(
            market_type=data['market_type'],
            home_bet_pct=data.get('home_bet_pct'),
            home_money_pct=data.get('home_money_pct'),
            away_bet_pct=data.get('away_bet_pct'),
            away_money_pct=data.get('away_money_pct'),
            over_bet_pct=data.get('over_bet_pct'),
            over_money_pct=data.get('over_money_pct'),
            under_bet_pct=data.get('under_bet_pct'),
            under_money_pct=data.get('under_money_pct'),
            line=data.get('line')
        )

    def create_mock_data(
        self,
        game_id: str,
        home_team: str,
        away_team: str
    ) -> PublicBettingData:
        """Create mock public betting data for testing.

        Args:
            game_id: Game identifier
            home_team: Home team abbreviation
            away_team: Away team abbreviation

        Returns:
            PublicBettingData with realistic mock values
        """
        import random

        # Generate realistic percentages
        spread_home_bet = random.uniform(45, 75)
        spread_home_money = spread_home_bet + random.uniform(-15, 15)

        total_over_bet = random.uniform(45, 75)
        total_over_money = total_over_bet + random.uniform(-15, 15)

        ml_home_bet = random.uniform(45, 75)
        ml_home_money = ml_home_bet + random.uniform(-15, 15)

        # Create markets
        spread = PublicBettingMarket(
            market_type='spread',
            home_bet_pct=round(spread_home_bet, 1),
            home_money_pct=round(spread_home_money, 1),
            away_bet_pct=round(100 - spread_home_bet, 1),
            away_money_pct=round(100 - spread_home_money, 1),
            line=-3.0
        )

        total = PublicBettingMarket(
            market_type='total',
            over_bet_pct=round(total_over_bet, 1),
            over_money_pct=round(total_over_money, 1),
            under_bet_pct=round(100 - total_over_bet, 1),
            under_money_pct=round(100 - total_over_money, 1),
            line=47.5
        )

        moneyline = PublicBettingMarket(
            market_type='moneyline',
            home_bet_pct=round(ml_home_bet, 1),
            home_money_pct=round(ml_home_money, 1),
            away_bet_pct=round(100 - ml_home_bet, 1),
            away_money_pct=round(100 - ml_home_money, 1)
        )

        # Detect sharp money and contrarian opportunities
        data = PublicBettingData(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            spread=spread,
            total=total,
            moneyline=moneyline,
            timestamp=datetime.now()
        )

        # Detect indicators
        data.spread_sharp_on_home = self._detect_sharp_money(
            spread.home_bet_pct, spread.home_money_pct
        )
        data.spread_sharp_on_away = self._detect_sharp_money(
            spread.away_bet_pct, spread.away_money_pct
        )
        data.total_sharp_on_over = self._detect_sharp_money(
            total.over_bet_pct, total.over_money_pct
        )
        data.total_sharp_on_under = self._detect_sharp_money(
            total.under_bet_pct, total.under_money_pct
        )

        data.spread_contrarian_home = self._detect_contrarian_opportunity(
            spread.away_bet_pct  # Contrarian = fade the public
        )
        data.spread_contrarian_away = self._detect_contrarian_opportunity(
            spread.home_bet_pct
        )
        data.total_contrarian_over = self._detect_contrarian_opportunity(
            total.under_bet_pct
        )
        data.total_contrarian_under = self._detect_contrarian_opportunity(
            total.over_bet_pct
        )

        return data


# Singleton instance
public_betting_scraper = PublicBettingScraper()


if __name__ == "__main__":
    # Test scraper
    scraper = PublicBettingScraper()

    # Create mock data for testing
    mock = scraper.create_mock_data("2025_12_BUF_KC", "KC", "BUF")
    print(f"Mock data created for {mock.game_id}")
    print(f"Spread: {mock.spread.home_bet_pct}% bets, {mock.spread.home_money_pct}% money on {mock.home_team}")
    print(f"Sharp on home: {mock.spread_sharp_on_home}")
    print(f"Contrarian opportunity: {mock.spread_contrarian_away}")
