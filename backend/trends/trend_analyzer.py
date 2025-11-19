"""
Trend analysis for player performance patterns

Analyzes:
- Recent form (last 3-5 games)
- Streaks (scoring, yardage, etc.)
- Hot/cold players (performance vs expectation)
- Matchup history (vs specific teams/defenses)
- Home/away splits
- Weather performance
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from backend.config import settings
from backend.config.logging_config import get_logger
from backend.database.session import get_db
from backend.database.models import PlayerGameFeature, Game, Player

logger = get_logger(__name__)


@dataclass
class TrendSignal:
    """Trend signal for a player-market combination"""
    player_id: str
    market: str

    # Trend indicators
    recent_form_score: float  # -1 to +1 (cold to hot)
    streak_count: int  # Positive for over streak, negative for under
    consistency_score: float  # 0 to 1 (boom/bust vs consistent)

    # Matchup trends
    vs_opponent_avg: Optional[float] = None  # Performance vs this opponent
    home_away_diff: Optional[float] = None  # Home vs away difference

    # Volatility
    recent_volatility: float = 0.0  # Coefficient of variation

    # Overall signal
    signal_strength: float = 0.0  # Combined signal (-1 to +1)
    confidence: float = 0.5  # How confident in the signal

    # Supporting data
    recent_games: int = 0
    sample_values: List[float] = None

    def __post_init__(self):
        if self.sample_values is None:
            self.sample_values = []


class TrendAnalyzer:
    """
    Analyzes player performance trends

    Identifies:
    - Players on hot streaks (consistently beating projections)
    - Players on cold streaks (consistently underperforming)
    - Matchup advantages (performs well vs certain teams)
    - Situational trends (home/road, weather, etc.)
    """

    def __init__(
        self,
        recent_window: int = 5,
        streak_threshold: float = 0.1  # 10% above/below mean
    ):
        """
        Initialize trend analyzer

        Args:
            recent_window: Number of recent games to analyze
            streak_threshold: Threshold for streak detection
        """
        self.recent_window = recent_window
        self.streak_threshold = streak_threshold

    def analyze_player_trends(
        self,
        player_id: str,
        market: str,
        current_game_id: Optional[str] = None
    ) -> TrendSignal:
        """
        Analyze all trends for a player-market combination

        Args:
            player_id: Player ID
            market: Market name (e.g., 'player_rec_yds')
            current_game_id: Exclude this game (for prediction context)

        Returns:
            TrendSignal with comprehensive trend analysis
        """
        # Get recent performance data
        recent_data = self._get_recent_performance(
            player_id, market, current_game_id
        )

        if len(recent_data) < 3:
            logger.debug(
                "insufficient_trend_data",
                player_id=player_id,
                market=market,
                games=len(recent_data)
            )
            return TrendSignal(
                player_id=player_id,
                market=market,
                recent_form_score=0.0,
                streak_count=0,
                consistency_score=0.5,
                recent_games=len(recent_data)
            )

        values = np.array([d['value'] for d in recent_data])

        # Calculate trend components
        recent_form_score = self._calculate_recent_form(values)
        streak_count = self._calculate_streak(values)
        consistency_score = self._calculate_consistency(values)
        recent_volatility = self._calculate_volatility(values)

        # Matchup trends (if opponent info available)
        vs_opponent_avg = None
        home_away_diff = None

        # Combined signal
        signal_strength = self._combine_signals(
            recent_form_score,
            streak_count,
            consistency_score
        )

        # Confidence based on sample size and consistency
        confidence = self._calculate_confidence(
            len(recent_data),
            consistency_score
        )

        signal = TrendSignal(
            player_id=player_id,
            market=market,
            recent_form_score=recent_form_score,
            streak_count=streak_count,
            consistency_score=consistency_score,
            vs_opponent_avg=vs_opponent_avg,
            home_away_diff=home_away_diff,
            recent_volatility=recent_volatility,
            signal_strength=signal_strength,
            confidence=confidence,
            recent_games=len(recent_data),
            sample_values=values.tolist()
        )

        logger.debug(
            "trend_analyzed",
            player_id=player_id,
            market=market,
            signal_strength=signal_strength,
            confidence=confidence
        )

        return signal

    def find_hot_players(
        self,
        market: str,
        min_signal_strength: float = 0.3,
        min_confidence: float = 0.6,
        limit: int = 10
    ) -> List[TrendSignal]:
        """
        Find players on hot streaks for a specific market

        Args:
            market: Market to analyze
            min_signal_strength: Minimum signal strength
            min_confidence: Minimum confidence
            limit: Max number of results

        Returns:
            List of TrendSignals for hot players
        """
        hot_players = []

        with get_db() as session:
            # Get all active players
            players = session.query(Player).filter(
                Player.position.in_(['QB', 'RB', 'WR', 'TE'])
            ).all()

            for player in players:
                signal = self.analyze_player_trends(player.player_id, market)

                if (signal.signal_strength >= min_signal_strength and
                    signal.confidence >= min_confidence):
                    hot_players.append(signal)

        # Sort by signal strength
        hot_players.sort(key=lambda x: x.signal_strength, reverse=True)

        logger.info(
            "hot_players_found",
            market=market,
            count=len(hot_players[:limit])
        )

        return hot_players[:limit]

    def analyze_matchup_history(
        self,
        player_id: str,
        opponent_team: str,
        market: str,
        n_games: int = 5
    ) -> Dict:
        """
        Analyze player's historical performance vs specific opponent

        Args:
            player_id: Player ID
            opponent_team: Opponent team code
            market: Market to analyze
            n_games: Number of games to look back

        Returns:
            Dict with matchup analysis
        """
        # Get games vs this opponent
        with get_db() as session:
            player = session.query(Player).filter(
                Player.player_id == player_id
            ).first()

            if not player:
                return {}

            # Find games vs opponent
            games_vs_opp = (
                session.query(PlayerGameFeature, Game)
                .join(Game, PlayerGameFeature.game_id == Game.game_id)
                .filter(
                    PlayerGameFeature.player_id == player_id,
                    ((Game.home_team == opponent_team) & (Game.away_team == player.team)) |
                    ((Game.away_team == opponent_team) & (Game.home_team == player.team))
                )
                .order_by(Game.game_date.desc())
                .limit(n_games)
                .all()
            )

            if not games_vs_opp:
                return {'games': 0}

            # Extract values based on market
            values = []
            for feature, game in games_vs_opp:
                value = self._extract_market_value(feature, market)
                if value is not None:
                    values.append(value)

            if not values:
                return {'games': 0}

            return {
                'games': len(values),
                'avg': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'recent_values': values
            }

    def _get_recent_performance(
        self,
        player_id: str,
        market: str,
        exclude_game_id: Optional[str] = None
    ) -> List[Dict]:
        """Get recent performance data for a player-market"""
        with get_db() as session:
            query = (
                session.query(PlayerGameFeature, Game)
                .join(Game, PlayerGameFeature.game_id == Game.game_id)
                .filter(PlayerGameFeature.player_id == player_id)
                .order_by(Game.game_date.desc())
                .limit(self.recent_window)
            )

            if exclude_game_id:
                query = query.filter(PlayerGameFeature.game_id != exclude_game_id)

            results = query.all()

            performance_data = []
            for feature, game in results:
                value = self._extract_market_value(feature, market)
                if value is not None:
                    performance_data.append({
                        'value': value,
                        'game_id': game.game_id,
                        'game_date': game.game_date
                    })

            return performance_data

    def _extract_market_value(
        self,
        feature: PlayerGameFeature,
        market: str
    ) -> Optional[float]:
        """Extract stat value for a market from PlayerGameFeature"""
        market_stat_map = {
            'player_rec_yds': 'receiving_yards',
            'player_receptions': 'receptions',
            'player_rush_yds': 'rushing_yards',
            'player_rush_attempts': 'rush_attempts',
            'player_pass_yds': 'passing_yards',
            'player_pass_tds': 'passing_tds',
            'player_rec_tds': 'receiving_tds',
            'player_rush_tds': 'rushing_tds',
        }

        stat_name = market_stat_map.get(market)
        if not stat_name:
            return None

        value = getattr(feature, stat_name, None)
        return float(value) if value is not None else None

    def _calculate_recent_form(self, values: np.ndarray) -> float:
        """
        Calculate recent form score (-1 to +1)

        Positive = trending up (hot)
        Negative = trending down (cold)
        """
        if len(values) < 3:
            return 0.0

        # Calculate linear trend
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)

        # Normalize by mean
        mean_value = values.mean()
        if mean_value == 0:
            return 0.0

        # Slope as % of mean per game
        normalized_slope = slope / mean_value

        # Clip to -1, +1
        form_score = np.clip(normalized_slope * 5, -1, 1)

        return float(form_score)

    def _calculate_streak(self, values: np.ndarray) -> int:
        """
        Calculate streak count

        Positive = consecutive overs
        Negative = consecutive unders
        Zero = no streak
        """
        if len(values) < 2:
            return 0

        mean = values.mean()
        threshold = mean * self.streak_threshold

        # Check for over/under streak
        streak = 0
        for i in range(len(values) - 1, -1, -1):
            if values[i] > mean + threshold:
                if streak >= 0:
                    streak += 1
                else:
                    break
            elif values[i] < mean - threshold:
                if streak <= 0:
                    streak -= 1
                else:
                    break
            else:
                break

        return int(streak)

    def _calculate_consistency(self, values: np.ndarray) -> float:
        """
        Calculate consistency score (0 to 1)

        1 = very consistent
        0 = very volatile (boom/bust)
        """
        if len(values) < 2:
            return 0.5

        cv = values.std() / values.mean() if values.mean() > 0 else 1.0

        # Map CV to consistency score
        # CV < 0.3 = very consistent
        # CV > 1.0 = very volatile
        consistency = 1 / (1 + cv)

        return float(consistency)

    def _calculate_volatility(self, values: np.ndarray) -> float:
        """Calculate coefficient of variation"""
        if len(values) < 2:
            return 0.0

        mean = values.mean()
        if mean == 0:
            return 0.0

        cv = values.std() / mean
        return float(cv)

    def _combine_signals(
        self,
        recent_form: float,
        streak_count: int,
        consistency: float
    ) -> float:
        """
        Combine trend signals into overall signal strength

        Returns: -1 (strong sell) to +1 (strong buy)
        """
        # Weight components
        form_weight = 0.5
        streak_weight = 0.3
        consistency_weight = 0.2

        # Normalize streak to -1, +1
        streak_normalized = np.clip(streak_count / 3, -1, 1)

        # Combine (consistency amplifies signal)
        signal = (
            form_weight * recent_form +
            streak_weight * streak_normalized
        ) * (0.5 + 0.5 * consistency)

        return float(np.clip(signal, -1, 1))

    def _calculate_confidence(
        self,
        sample_size: int,
        consistency: float
    ) -> float:
        """
        Calculate confidence in trend signal

        Higher sample size + higher consistency = higher confidence
        """
        # Sample size component
        size_confidence = min(sample_size / 10, 1.0)

        # Consistency component
        consistency_confidence = consistency

        # Combined
        confidence = 0.6 * size_confidence + 0.4 * consistency_confidence

        return float(confidence)
