"""
News fetching and sentiment analysis for player updates

Sources:
- NFL.com injury reports
- ESPN player news
- Twitter/X for breaking news
- Beat reporters

Analyzes:
- Injury severity and timeline
- Coaching comments (snap count, role changes)
- Lineup changes
- News sentiment (positive/negative/neutral)
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import re

from backend.config import settings
from backend.config.logging_config import get_logger

logger = get_logger(__name__)


class NewsSentiment(str, Enum):
    """News sentiment classification"""
    VERY_POSITIVE = "very_positive"  # "Expected to have increased role"
    POSITIVE = "positive"  # "Practicing fully"
    NEUTRAL = "neutral"  # "Questionable for Sunday"
    NEGATIVE = "negative"  # "Limited in practice"
    VERY_NEGATIVE = "very_negative"  # "Ruled out"


class NewsImpact(str, Enum):
    """Expected impact on prop value"""
    MAJOR_BOOST = "major_boost"  # +20%+
    MODERATE_BOOST = "moderate_boost"  # +10-20%
    SLIGHT_BOOST = "slight_boost"  # +5-10%
    NEUTRAL = "neutral"  # No change
    SLIGHT_DECREASE = "slight_decrease"  # -5-10%
    MODERATE_DECREASE = "moderate_decrease"  # -10-20%
    MAJOR_DECREASE = "major_decrease"  # -20%+


@dataclass
class NewsItem:
    """Single news item for a player"""
    player_id: str
    player_name: str

    headline: str
    content: str
    source: str

    published_at: datetime
    fetched_at: datetime

    # Analysis
    sentiment: NewsSentiment
    impact: NewsImpact
    confidence: float  # 0-1, confidence in sentiment/impact

    # Keywords extracted
    keywords: List[str]

    # Categories
    is_injury: bool = False
    is_role_change: bool = False
    is_lineup_change: bool = False
    is_trade: bool = False


@dataclass
class PlayerNewsDigest:
    """Aggregated news for a player"""
    player_id: str
    player_name: str

    # Recent news items
    news_items: List[NewsItem]

    # Overall assessment
    overall_sentiment: NewsSentiment
    overall_impact: NewsImpact

    # Key insights
    injury_status: Optional[str] = None  # "OUT", "DOUBTFUL", "QUESTIONABLE", "PROBABLE"
    snap_count_trend: Optional[str] = None  # "increasing", "stable", "decreasing"
    role_change: Optional[str] = None  # Description of role change

    # Metadata
    last_updated: datetime = None
    news_count: int = 0


class NewsAnalyzer:
    """
    Analyzes news for impact on player props

    Uses keyword matching and rules-based sentiment for now.
    Future: Could use LLM for better understanding.
    """

    def __init__(self):
        """Initialize news analyzer"""
        # Positive keywords
        self.positive_keywords = {
            'full participant', 'full practice', 'no injury designation',
            'increased role', 'more touches', 'expanded role',
            'starting', 'feature back', 'wr1', 'cleared',
            'healthy', 'activated', 'return'
        }

        # Negative keywords
        self.negative_keywords = {
            'ruled out', 'doubtful', 'dnp', 'did not practice',
            'limited', 'sidelined', 'injury', 'hurt', 'IR',
            'reduced role', 'backup', 'depth', 'snap count down'
        }

        # Impact keywords
        self.major_impact_keywords = {
            'ruled out', 'IR', 'suspended', 'traded',
            'starting', 'bellcow', 'featured'
        }

    def analyze_news_item(
        self,
        player_id: str,
        player_name: str,
        headline: str,
        content: str,
        source: str,
        published_at: datetime
    ) -> NewsItem:
        """
        Analyze a single news item

        Args:
            player_id: Player ID
            player_name: Player name
            headline: News headline
            content: News content
            source: News source
            published_at: Publication timestamp

        Returns:
            NewsItem with sentiment and impact analysis
        """
        # Combine headline and content for analysis
        full_text = f"{headline} {content}".lower()

        # Extract keywords
        keywords = self._extract_keywords(full_text)

        # Classify sentiment
        sentiment = self._classify_sentiment(full_text, keywords)

        # Estimate impact
        impact = self._estimate_impact(full_text, keywords, sentiment)

        # Categorize
        is_injury = self._is_injury_news(full_text)
        is_role_change = self._is_role_change(full_text)
        is_lineup_change = self._is_lineup_change(full_text)
        is_trade = self._is_trade_news(full_text)

        # Confidence based on source and clarity
        confidence = self._calculate_confidence(source, sentiment, keywords)

        news_item = NewsItem(
            player_id=player_id,
            player_name=player_name,
            headline=headline,
            content=content,
            source=source,
            published_at=published_at,
            fetched_at=datetime.utcnow(),
            sentiment=sentiment,
            impact=impact,
            confidence=confidence,
            keywords=keywords,
            is_injury=is_injury,
            is_role_change=is_role_change,
            is_lineup_change=is_lineup_change,
            is_trade=is_trade
        )

        logger.debug(
            "news_analyzed",
            player_id=player_id,
            sentiment=sentiment.value,
            impact=impact.value
        )

        return news_item

    def create_player_digest(
        self,
        player_id: str,
        player_name: str,
        news_items: List[NewsItem]
    ) -> PlayerNewsDigest:
        """
        Create aggregated news digest for a player

        Args:
            player_id: Player ID
            player_name: Player name
            news_items: List of recent news items

        Returns:
            PlayerNewsDigest with aggregated insights
        """
        if not news_items:
            return PlayerNewsDigest(
                player_id=player_id,
                player_name=player_name,
                news_items=[],
                overall_sentiment=NewsSentiment.NEUTRAL,
                overall_impact=NewsImpact.NEUTRAL,
                last_updated=datetime.utcnow(),
                news_count=0
            )

        # Sort by recency
        news_items.sort(key=lambda x: x.published_at, reverse=True)

        # Overall sentiment (weighted by recency and confidence)
        overall_sentiment = self._aggregate_sentiment(news_items)

        # Overall impact
        overall_impact = self._aggregate_impact(news_items)

        # Extract injury status from most recent news
        injury_status = self._extract_injury_status(news_items)

        # Snap count trend
        snap_count_trend = self._extract_snap_trend(news_items)

        # Role change
        role_change = self._extract_role_change(news_items)

        digest = PlayerNewsDigest(
            player_id=player_id,
            player_name=player_name,
            news_items=news_items,
            overall_sentiment=overall_sentiment,
            overall_impact=overall_impact,
            injury_status=injury_status,
            snap_count_trend=snap_count_trend,
            role_change=role_change,
            last_updated=datetime.utcnow(),
            news_count=len(news_items)
        )

        return digest

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        keywords = []

        # Check positive keywords
        for kw in self.positive_keywords:
            if kw in text:
                keywords.append(kw)

        # Check negative keywords
        for kw in self.negative_keywords:
            if kw in text:
                keywords.append(kw)

        return keywords

    def _classify_sentiment(
        self,
        text: str,
        keywords: List[str]
    ) -> NewsSentiment:
        """Classify news sentiment"""
        # Count positive vs negative keywords
        positive_count = sum(1 for kw in keywords if kw in self.positive_keywords)
        negative_count = sum(1 for kw in keywords if kw in self.negative_keywords)

        # Special cases
        if 'ruled out' in text or 'ir' in text:
            return NewsSentiment.VERY_NEGATIVE
        if 'starting' in text or 'full participant' in text:
            return NewsSentiment.VERY_POSITIVE

        # Based on keyword balance
        if positive_count > negative_count + 1:
            return NewsSentiment.POSITIVE
        elif negative_count > positive_count + 1:
            return NewsSentiment.NEGATIVE
        elif positive_count > negative_count:
            return NewsSentiment.POSITIVE
        elif negative_count > positive_count:
            return NewsSentiment.NEGATIVE
        else:
            return NewsSentiment.NEUTRAL

    def _estimate_impact(
        self,
        text: str,
        keywords: List[str],
        sentiment: NewsSentiment
    ) -> NewsImpact:
        """Estimate impact on prop value"""
        # Major impacts
        if any(kw in text for kw in self.major_impact_keywords):
            if sentiment in [NewsSentiment.VERY_POSITIVE, NewsSentiment.POSITIVE]:
                return NewsImpact.MAJOR_BOOST
            elif sentiment in [NewsSentiment.VERY_NEGATIVE, NewsSentiment.NEGATIVE]:
                return NewsImpact.MAJOR_DECREASE

        # Map sentiment to impact
        sentiment_impact_map = {
            NewsSentiment.VERY_POSITIVE: NewsImpact.MODERATE_BOOST,
            NewsSentiment.POSITIVE: NewsImpact.SLIGHT_BOOST,
            NewsSentiment.NEUTRAL: NewsImpact.NEUTRAL,
            NewsSentiment.NEGATIVE: NewsImpact.SLIGHT_DECREASE,
            NewsSentiment.VERY_NEGATIVE: NewsImpact.MODERATE_DECREASE,
        }

        return sentiment_impact_map[sentiment]

    def _is_injury_news(self, text: str) -> bool:
        """Check if news is injury-related"""
        injury_keywords = ['injury', 'injured', 'practice', 'questionable', 'doubtful', 'out', 'dnp']
        return any(kw in text for kw in injury_keywords)

    def _is_role_change(self, text: str) -> bool:
        """Check if news indicates role change"""
        role_keywords = ['role', 'snaps', 'touches', 'starting', 'backup', 'depth chart']
        return any(kw in text for kw in role_keywords)

    def _is_lineup_change(self, text: str) -> bool:
        """Check if news is lineup-related"""
        lineup_keywords = ['lineup', 'starting', 'inactive', 'active', 'roster']
        return any(kw in text for kw in lineup_keywords)

    def _is_trade_news(self, text: str) -> bool:
        """Check if news is trade-related"""
        return 'trade' in text or 'traded' in text

    def _calculate_confidence(
        self,
        source: str,
        sentiment: NewsSentiment,
        keywords: List[str]
    ) -> float:
        """Calculate confidence in analysis"""
        # Base confidence
        confidence = 0.5

        # Official sources higher confidence
        if 'nfl.com' in source.lower() or 'official' in source.lower():
            confidence += 0.2
        elif 'espn' in source.lower():
            confidence += 0.1

        # Clear sentiment increases confidence
        if sentiment in [NewsSentiment.VERY_POSITIVE, NewsSentiment.VERY_NEGATIVE]:
            confidence += 0.2

        # More keywords = more confidence
        if len(keywords) >= 3:
            confidence += 0.1

        return min(confidence, 1.0)

    def _aggregate_sentiment(self, news_items: List[NewsItem]) -> NewsSentiment:
        """Aggregate sentiment across multiple news items"""
        if not news_items:
            return NewsSentiment.NEUTRAL

        # Weight recent news more heavily
        total_weight = 0
        weighted_score = 0

        sentiment_scores = {
            NewsSentiment.VERY_NEGATIVE: -2,
            NewsSentiment.NEGATIVE: -1,
            NewsSentiment.NEUTRAL: 0,
            NewsSentiment.POSITIVE: 1,
            NewsSentiment.VERY_POSITIVE: 2,
        }

        for i, item in enumerate(news_items):
            # Recency weight (most recent = 1.0, decays)
            weight = 1.0 / (i + 1)
            # Confidence weight
            weight *= item.confidence

            score = sentiment_scores[item.sentiment]
            weighted_score += score * weight
            total_weight += weight

        if total_weight == 0:
            return NewsSentiment.NEUTRAL

        avg_score = weighted_score / total_weight

        # Map back to sentiment
        if avg_score >= 1.5:
            return NewsSentiment.VERY_POSITIVE
        elif avg_score >= 0.5:
            return NewsSentiment.POSITIVE
        elif avg_score >= -0.5:
            return NewsSentiment.NEUTRAL
        elif avg_score >= -1.5:
            return NewsSentiment.NEGATIVE
        else:
            return NewsSentiment.VERY_NEGATIVE

    def _aggregate_impact(self, news_items: List[NewsItem]) -> NewsImpact:
        """Aggregate impact across news items"""
        # Use most impactful recent news
        if not news_items:
            return NewsImpact.NEUTRAL

        # Check for major impacts in recent news
        for item in news_items[:3]:  # Last 3 items
            if item.impact in [NewsImpact.MAJOR_BOOST, NewsImpact.MAJOR_DECREASE]:
                return item.impact

        # Otherwise aggregate
        return news_items[0].impact if news_items else NewsImpact.NEUTRAL

    def _extract_injury_status(self, news_items: List[NewsItem]) -> Optional[str]:
        """Extract current injury status"""
        for item in news_items:
            if not item.is_injury:
                continue

            text = item.content.lower()
            if 'ruled out' in text or 'out for' in text:
                return "OUT"
            if 'doubtful' in text:
                return "DOUBTFUL"
            if 'questionable' in text:
                return "QUESTIONABLE"
            if 'probable' in text or 'expected to play' in text:
                return "PROBABLE"
            if 'full participant' in text or 'no designation' in text:
                return "HEALTHY"

        return None

    def _extract_snap_trend(self, news_items: List[NewsItem]) -> Optional[str]:
        """Extract snap count trend"""
        for item in news_items:
            text = item.content.lower()
            if 'snap' in text or 'playing time' in text:
                if 'increas' in text or 'more' in text or 'expanded' in text:
                    return "increasing"
                if 'decreas' in text or 'less' in text or 'reduced' in text:
                    return "decreasing"
                return "stable"

        return None

    def _extract_role_change(self, news_items: List[NewsItem]) -> Optional[str]:
        """Extract role change description"""
        for item in news_items:
            if item.is_role_change:
                # Return the headline as role description
                return item.headline

        return None
