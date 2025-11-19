"""
News API integration for real-time player news

Fetches news from multiple sources:
- ESPN API
- Twitter/X (via API)
- FantasyPros
- Rotoworld
- NFL.com
- ProFootballTalk

Provides structured news data for sentiment analysis.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import requests
import feedparser

from backend.config import settings
from backend.config.logging_config import get_logger

logger = get_logger(__name__)


class NewsSource(Enum):
    """Supported news sources"""
    ESPN = "espn"
    TWITTER = "twitter"
    FANTASYPROS = "fantasypros"
    ROTOWORLD = "rotoworld"
    NFL_COM = "nfl_com"
    PFT = "profootballtalk"


@dataclass
class RawNewsItem:
    """Raw news item before sentiment analysis"""
    source: str
    headline: str
    description: str
    url: str
    published_at: datetime
    author: Optional[str] = None

    # Player mentions (extracted)
    mentioned_players: List[str] = None


class ESPNNewsClient:
    """
    Client for ESPN News API

    ESPN provides a free API for NFL news
    """

    def __init__(self):
        self.base_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/news"

    def fetch_news(
        self,
        limit: int = 50,
        player_id: Optional[str] = None
    ) -> List[RawNewsItem]:
        """
        Fetch latest NFL news from ESPN

        Args:
            limit: Max news items to fetch
            player_id: Optional ESPN player ID to filter

        Returns:
            List of RawNewsItem
        """
        params = {
            "limit": limit
        }

        if player_id:
            params["player"] = player_id

        try:
            response = requests.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()

            data = response.json()

            news_items = []

            for article in data.get("articles", []):
                item = RawNewsItem(
                    source="espn",
                    headline=article.get("headline", ""),
                    description=article.get("description", ""),
                    url=article.get("links", {}).get("web", {}).get("href", ""),
                    published_at=datetime.fromisoformat(
                        article.get("published", "").replace("Z", "+00:00")
                    ) if article.get("published") else datetime.utcnow(),
                    author=article.get("byline", None)
                )

                news_items.append(item)

            logger.info("espn_news_fetched", count=len(news_items))

            return news_items

        except Exception as e:
            logger.error("espn_news_fetch_failed", error=str(e))
            return []


class TwitterClient:
    """
    Client for Twitter/X API v2

    Fetches tweets from NFL insiders for breaking news
    """

    def __init__(self, bearer_token: Optional[str] = None):
        """
        Initialize Twitter client

        Args:
            bearer_token: Twitter API v2 bearer token
        """
        self.bearer_token = bearer_token or settings.twitter_bearer_token
        self.base_url = "https://api.twitter.com/2"

        # Notable NFL insiders to monitor
        self.insiders = [
            "AdamSchefter",  # ESPN
            "RapSheet",  # Ian Rapoport
            "JFowlerESPN",  # Jeremy Fowler
            "TomPelissero",  # Tom Pelissero
            "JayGlazer",  # Jay Glazer
            "MikeGarafolo",  # Mike Garafolo
        ]

    def fetch_tweets(
        self,
        query: str,
        max_results: int = 50,
        hours_back: int = 24
    ) -> List[RawNewsItem]:
        """
        Fetch recent tweets matching query

        Args:
            query: Twitter search query
            max_results: Max tweets to fetch (10-100)
            hours_back: How far back to search

        Returns:
            List of RawNewsItem from tweets
        """
        if not self.bearer_token:
            logger.warning("no_twitter_bearer_token")
            return []

        # Build query
        # Example: "Mahomes injury" from:AdamSchefter OR from:RapSheet
        insider_clause = " OR ".join([f"from:{user}" for user in self.insiders])
        full_query = f"({query}) ({insider_clause}) -is:retweet"

        # Calculate start time
        start_time = (datetime.utcnow() - timedelta(hours=hours_back)).isoformat() + "Z"

        params = {
            "query": full_query,
            "max_results": min(max_results, 100),
            "start_time": start_time,
            "tweet.fields": "created_at,author_id,text",
            "expansions": "author_id",
            "user.fields": "username"
        }

        headers = {
            "Authorization": f"Bearer {self.bearer_token}"
        }

        try:
            response = requests.get(
                f"{self.base_url}/tweets/search/recent",
                headers=headers,
                params=params,
                timeout=15
            )
            response.raise_for_status()

            data = response.json()

            # Parse tweets
            tweets = data.get("data", [])
            users = {u["id"]: u for u in data.get("includes", {}).get("users", [])}

            news_items = []

            for tweet in tweets:
                author_id = tweet.get("author_id")
                author = users.get(author_id, {})

                item = RawNewsItem(
                    source="twitter",
                    headline=tweet.get("text", "")[:200],  # First 200 chars
                    description=tweet.get("text", ""),
                    url=f"https://twitter.com/{author.get('username')}/status/{tweet.get('id')}",
                    published_at=datetime.fromisoformat(
                        tweet.get("created_at", "").replace("Z", "+00:00")
                    ) if tweet.get("created_at") else datetime.utcnow(),
                    author=author.get("username")
                )

                news_items.append(item)

            logger.info("twitter_news_fetched", count=len(news_items))

            return news_items

        except Exception as e:
            logger.error("twitter_fetch_failed", error=str(e))
            return []

    def fetch_insider_timelines(self, max_per_insider: int = 10) -> List[RawNewsItem]:
        """
        Fetch recent tweets from all NFL insiders

        Args:
            max_per_insider: Max tweets per insider

        Returns:
            Combined list of news items
        """
        all_news = []

        for username in self.insiders:
            try:
                # Get user ID
                user_response = requests.get(
                    f"{self.base_url}/users/by/username/{username}",
                    headers={"Authorization": f"Bearer {self.bearer_token}"},
                    timeout=10
                )
                user_data = user_response.json()
                user_id = user_data.get("data", {}).get("id")

                if not user_id:
                    continue

                # Get recent tweets
                tweets_response = requests.get(
                    f"{self.base_url}/users/{user_id}/tweets",
                    headers={"Authorization": f"Bearer {self.bearer_token}"},
                    params={
                        "max_results": max_per_insider,
                        "tweet.fields": "created_at,text"
                    },
                    timeout=10
                )
                tweets_data = tweets_response.json()

                for tweet in tweets_data.get("data", []):
                    item = RawNewsItem(
                        source="twitter",
                        headline=tweet.get("text", "")[:200],
                        description=tweet.get("text", ""),
                        url=f"https://twitter.com/{username}/status/{tweet.get('id')}",
                        published_at=datetime.fromisoformat(
                            tweet.get("created_at", "").replace("Z", "+00:00")
                        ) if tweet.get("created_at") else datetime.utcnow(),
                        author=username
                    )
                    all_news.append(item)

            except Exception as e:
                logger.error("insider_timeline_failed", username=username, error=str(e))
                continue

        logger.info("insider_timelines_fetched", count=len(all_news))

        return all_news


class RotoworldClient:
    """
    Client for Rotoworld/NBC Sports Edge RSS feeds

    Rotoworld provides player news via RSS
    """

    def __init__(self):
        self.rss_url = "https://www.nbcsportsedge.com/football/nfl/player-news"

    def fetch_news(self, max_items: int = 50) -> List[RawNewsItem]:
        """
        Fetch latest player news from Rotoworld RSS

        Args:
            max_items: Max news items to fetch

        Returns:
            List of RawNewsItem
        """
        try:
            feed = feedparser.parse(self.rss_url)

            news_items = []

            for entry in feed.entries[:max_items]:
                item = RawNewsItem(
                    source="rotoworld",
                    headline=entry.get("title", ""),
                    description=entry.get("summary", ""),
                    url=entry.get("link", ""),
                    published_at=datetime(*entry.published_parsed[:6]) if hasattr(entry, "published_parsed") else datetime.utcnow()
                )

                news_items.append(item)

            logger.info("rotoworld_news_fetched", count=len(news_items))

            return news_items

        except Exception as e:
            logger.error("rotoworld_fetch_failed", error=str(e))
            return []


class FantasyProsClient:
    """
    Client for FantasyPros news

    FantasyPros aggregates news from multiple sources
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FantasyPros client

        Args:
            api_key: FantasyPros API key
        """
        self.api_key = api_key or settings.fantasypros_api_key
        self.base_url = "https://api.fantasypros.com/v2"

    def fetch_news(
        self,
        position: Optional[str] = None,
        player_id: Optional[str] = None
    ) -> List[RawNewsItem]:
        """
        Fetch news from FantasyPros

        Args:
            position: Optional position filter (QB, RB, WR, TE)
            player_id: Optional player ID filter

        Returns:
            List of RawNewsItem
        """
        if not self.api_key:
            logger.warning("no_fantasypros_api_key")
            return []

        params = {
            "api_key": self.api_key,
            "sport": "NFL"
        }

        if position:
            params["position"] = position

        if player_id:
            params["player_id"] = player_id

        try:
            response = requests.get(
                f"{self.base_url}/news",
                params=params,
                timeout=15
            )
            response.raise_for_status()

            data = response.json()

            news_items = []

            for article in data.get("news", []):
                item = RawNewsItem(
                    source="fantasypros",
                    headline=article.get("headline", ""),
                    description=article.get("analysis", ""),
                    url=article.get("source_url", ""),
                    published_at=datetime.fromisoformat(
                        article.get("published_at", "")
                    ) if article.get("published_at") else datetime.utcnow(),
                    mentioned_players=article.get("players", [])
                )

                news_items.append(item)

            logger.info("fantasypros_news_fetched", count=len(news_items))

            return news_items

        except Exception as e:
            logger.error("fantasypros_fetch_failed", error=str(e))
            return []


class AggregatedNewsClient:
    """
    Aggregated news client that combines multiple sources

    Deduplicates and ranks news by recency and source quality
    """

    def __init__(self):
        self.espn = ESPNNewsClient()
        self.twitter = TwitterClient()
        self.rotoworld = RotoworldClient()
        self.fantasypros = FantasyProsClient()

        # Source quality weights (higher = more trustworthy)
        self.source_weights = {
            "twitter": 1.0,  # If from verified insiders
            "espn": 0.9,
            "fantasypros": 0.8,
            "rotoworld": 0.7,
        }

    def fetch_all_news(
        self,
        hours_back: int = 24,
        max_per_source: int = 50
    ) -> List[RawNewsItem]:
        """
        Fetch news from all sources

        Args:
            hours_back: How far back to fetch
            max_per_source: Max items per source

        Returns:
            Aggregated and deduplicated news items
        """
        all_news = []

        # ESPN
        try:
            espn_news = self.espn.fetch_news(limit=max_per_source)
            all_news.extend(espn_news)
        except Exception as e:
            logger.error("espn_aggregation_failed", error=str(e))

        # Twitter (insider timelines)
        try:
            twitter_news = self.twitter.fetch_insider_timelines(max_per_insider=10)
            all_news.extend(twitter_news)
        except Exception as e:
            logger.error("twitter_aggregation_failed", error=str(e))

        # Rotoworld
        try:
            roto_news = self.rotoworld.fetch_news(max_items=max_per_source)
            all_news.extend(roto_news)
        except Exception as e:
            logger.error("rotoworld_aggregation_failed", error=str(e))

        # FantasyPros
        try:
            fp_news = self.fantasypros.fetch_news()
            all_news.extend(fp_news)
        except Exception as e:
            logger.error("fantasypros_aggregation_failed", error=str(e))

        # Filter by recency
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        recent_news = [n for n in all_news if n.published_at >= cutoff_time]

        # Deduplicate by headline similarity
        deduplicated = self._deduplicate_news(recent_news)

        # Sort by recency
        deduplicated.sort(key=lambda x: x.published_at, reverse=True)

        logger.info(
            "news_aggregated",
            total=len(all_news),
            recent=len(recent_news),
            deduplicated=len(deduplicated)
        )

        return deduplicated

    def _deduplicate_news(self, news_items: List[RawNewsItem]) -> List[RawNewsItem]:
        """
        Deduplicate news items by headline similarity

        Args:
            news_items: List of news items

        Returns:
            Deduplicated list
        """
        seen_headlines = set()
        deduplicated = []

        for item in news_items:
            # Normalize headline
            normalized = item.headline.lower().strip()

            # Skip if very similar headline already seen
            # (Simple dedup - in production, use fuzzy matching)
            if normalized not in seen_headlines:
                seen_headlines.add(normalized)
                deduplicated.append(item)

        return deduplicated

    def fetch_player_news(
        self,
        player_name: str,
        hours_back: int = 48
    ) -> List[RawNewsItem]:
        """
        Fetch news for a specific player

        Args:
            player_name: Player name to search
            hours_back: Hours to search back

        Returns:
            Player-specific news items
        """
        all_news = []

        # ESPN (no player-specific endpoint, filter after)
        espn_news = self.espn.fetch_news(limit=100)
        player_espn = [n for n in espn_news if player_name.lower() in n.headline.lower()]
        all_news.extend(player_espn)

        # Twitter
        try:
            twitter_news = self.twitter.fetch_tweets(
                query=player_name,
                max_results=50,
                hours_back=hours_back
            )
            all_news.extend(twitter_news)
        except Exception:
            pass

        # Sort by recency
        all_news.sort(key=lambda x: x.published_at, reverse=True)

        logger.info("player_news_fetched", player=player_name, count=len(all_news))

        return all_news


# Convenience function
def fetch_latest_news(hours_back: int = 24) -> List[RawNewsItem]:
    """
    Fetch latest NFL news from all sources

    Args:
        hours_back: How many hours to look back

    Returns:
        List of news items
    """
    client = AggregatedNewsClient()
    return client.fetch_all_news(hours_back=hours_back)
