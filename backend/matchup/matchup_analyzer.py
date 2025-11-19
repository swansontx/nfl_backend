"""
Dynamic matchup analysis for prop betting

Calculates opponent defensive strength and game script factors:
- Defensive yards allowed by position
- Touchdowns allowed
- Pace of play metrics
- Home/away splits
- Recent defensive trends

Used to generate matchup_signal in recommendation system.
"""

from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

from backend.database.session import get_db
from backend.database.models import Game, Player, PlayerGameFeature
from backend.config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MatchupMetrics:
    """Matchup analysis for a player-game combination"""

    # Opponent strength (0-1, higher = easier matchup)
    opponent_rank_percentile: float  # 0 = toughest defense, 1 = easiest

    # Specific metrics
    avg_yards_allowed: float  # Average yards allowed to this position
    avg_tds_allowed: float  # Average TDs allowed
    games_analyzed: int  # Sample size

    # Pace factors
    opponent_pace: float  # Plays per game
    opponent_points_allowed: float  # Points per game

    # Confidence
    confidence: float  # 0-1, based on sample size and recency

    # Metadata
    opponent_team: str
    position: str
    market: str


class MatchupAnalyzer:
    """
    Analyzes opponent defensive matchups

    Calculates how favorable a matchup is based on:
    - Historical performance vs this opponent
    - Opponent's defensive stats vs this position
    - Game pace and scoring environment
    """

    def __init__(self, lookback_games: int = 8):
        """
        Initialize matchup analyzer

        Args:
            lookback_games: Number of recent games to analyze
        """
        self.lookback_games = lookback_games

    def analyze_matchup(
        self,
        player_id: str,
        game_id: str,
        market: str
    ) -> Optional[MatchupMetrics]:
        """
        Analyze matchup for a player-game-market combination

        Args:
            player_id: Player ID
            game_id: Game ID
            market: Market type (e.g., 'receiving_yards', 'rushing_yards')

        Returns:
            MatchupMetrics with matchup analysis
        """
        with get_db() as session:
            # Get game and player info
            game = session.query(Game).filter(Game.game_id == game_id).first()
            if not game:
                logger.warning("game_not_found", game_id=game_id)
                return None

            player = session.query(Player).filter(Player.player_id == player_id).first()
            if not player:
                logger.warning("player_not_found", player_id=player_id)
                return None

            # Determine opponent
            opponent_team = game.away_team if player.team == game.home_team else game.home_team

            # Get opponent's defensive stats vs this position
            metrics = self._calculate_defensive_stats(
                session=session,
                opponent_team=opponent_team,
                position=player.position,
                market=market,
                before_date=game.game_date
            )

            if not metrics:
                logger.debug(
                    "insufficient_matchup_data",
                    player_id=player_id,
                    opponent=opponent_team
                )
                return None

            # Add opponent team info
            metrics.opponent_team = opponent_team
            metrics.position = player.position
            metrics.market = market

            logger.debug(
                "matchup_analyzed",
                player_id=player_id,
                opponent=opponent_team,
                rank_percentile=metrics.opponent_rank_percentile,
                confidence=metrics.confidence
            )

            return metrics

    def _calculate_defensive_stats(
        self,
        session,
        opponent_team: str,
        position: str,
        market: str,
        before_date: datetime
    ) -> Optional[MatchupMetrics]:
        """
        Calculate opponent defensive stats against this position

        Args:
            session: Database session
            opponent_team: Opponent team abbreviation
            position: Player position (QB, RB, WR, TE)
            market: Market type
            before_date: Only use games before this date

        Returns:
            MatchupMetrics or None
        """
        # Get recent games where opponent_team played
        recent_games = (
            session.query(Game)
            .filter(
                (Game.home_team == opponent_team) | (Game.away_team == opponent_team),
                Game.game_date < before_date
            )
            .order_by(Game.game_date.desc())
            .limit(self.lookback_games)
            .all()
        )

        if not recent_games:
            return None

        # Get all opposing players in those games
        game_ids = [g.game_id for g in recent_games]

        # Find players who played AGAINST opponent_team
        opposing_stats = []

        for game in recent_games:
            # Determine which team opposed opponent_team
            opposing_team = game.away_team if game.home_team == opponent_team else game.home_team

            # Get all players from opposing team with this position
            features = (
                session.query(PlayerGameFeature, Player)
                .join(Player, PlayerGameFeature.player_id == Player.player_id)
                .filter(
                    PlayerGameFeature.game_id == game.game_id,
                    Player.team == opposing_team,
                    Player.position == position
                )
                .all()
            )

            for feature, player in features:
                stat_value = self._extract_market_stat(feature, market)
                if stat_value is not None:
                    opposing_stats.append(stat_value)

        if not opposing_stats:
            return None

        # Calculate defensive metrics
        avg_allowed = float(np.mean(opposing_stats))
        std_allowed = float(np.std(opposing_stats))
        games_analyzed = len(opposing_stats)

        # Calculate TD stats if applicable
        td_stats = []
        if 'touchdown' in market.lower() or market in ['receiving_tds', 'rushing_tds', 'passing_tds']:
            for game in recent_games:
                opposing_team = game.away_team if game.home_team == opponent_team else game.home_team

                features = (
                    session.query(PlayerGameFeature, Player)
                    .join(Player, PlayerGameFeature.player_id == Player.player_id)
                    .filter(
                        PlayerGameFeature.game_id == game.game_id,
                        Player.team == opposing_team,
                        Player.position == position
                    )
                    .all()
                )

                for feature, player in features:
                    tds = self._extract_td_stat(feature, market)
                    if tds is not None:
                        td_stats.append(tds)

        avg_tds_allowed = float(np.mean(td_stats)) if td_stats else 0.0

        # Calculate opponent rank percentile
        # Compare this opponent to league average
        league_avg = self._get_league_average(session, position, market, before_date)

        if league_avg > 0:
            # Higher yards allowed = easier matchup (higher percentile)
            rank_percentile = min(avg_allowed / (league_avg * 1.5), 1.0)
        else:
            rank_percentile = 0.5  # Neutral

        # Confidence based on sample size
        confidence = min(games_analyzed / self.lookback_games, 1.0)

        # Get pace metrics
        opponent_pace = self._calculate_pace(session, opponent_team, before_date)
        opponent_points_allowed = self._calculate_points_allowed(session, opponent_team, before_date)

        return MatchupMetrics(
            opponent_rank_percentile=rank_percentile,
            avg_yards_allowed=avg_allowed,
            avg_tds_allowed=avg_tds_allowed,
            games_analyzed=games_analyzed,
            opponent_pace=opponent_pace,
            opponent_points_allowed=opponent_points_allowed,
            confidence=confidence,
            opponent_team=opponent_team,
            position=position,
            market=market
        )

    def _extract_market_stat(self, feature: PlayerGameFeature, market: str) -> Optional[float]:
        """Extract the relevant stat from PlayerGameFeature based on market"""
        market_map = {
            'receiving_yards': feature.receiving_yards,
            'rushing_yards': feature.rushing_yards,
            'passing_yards': feature.passing_yards,
            'receptions': feature.receptions,
            'targets': feature.targets,
            'rush_attempts': feature.rush_attempts,
            'pass_attempts': feature.pass_attempts,
        }

        value = market_map.get(market)
        return float(value) if value is not None else None

    def _extract_td_stat(self, feature: PlayerGameFeature, market: str) -> Optional[int]:
        """Extract touchdown stat from PlayerGameFeature"""
        td_map = {
            'receiving_tds': feature.receiving_tds,
            'rushing_tds': feature.rushing_tds,
            'passing_tds': feature.passing_tds,
        }

        value = td_map.get(market)
        return int(value) if value is not None else None

    def _get_league_average(
        self,
        session,
        position: str,
        market: str,
        before_date: datetime
    ) -> float:
        """Calculate league average for this position/market"""
        # Get all games before this date
        games = (
            session.query(Game)
            .filter(Game.game_date < before_date)
            .order_by(Game.game_date.desc())
            .limit(100)  # Recent games
            .all()
        )

        if not games:
            return 0.0

        game_ids = [g.game_id for g in games]

        # Get all features for this position
        features = (
            session.query(PlayerGameFeature)
            .join(Player, PlayerGameFeature.player_id == Player.player_id)
            .filter(
                PlayerGameFeature.game_id.in_(game_ids),
                Player.position == position
            )
            .all()
        )

        stats = []
        for feature in features:
            stat_value = self._extract_market_stat(feature, market)
            if stat_value is not None and stat_value > 0:
                stats.append(stat_value)

        return float(np.mean(stats)) if stats else 0.0

    def _calculate_pace(self, session, team: str, before_date: datetime) -> float:
        """Calculate team's offensive pace (plays per game)"""
        # For now, return neutral value
        # In production, calculate from play-by-play data
        return 65.0  # League average

    def _calculate_points_allowed(self, session, team: str, before_date: datetime) -> float:
        """Calculate points allowed per game"""
        games = (
            session.query(Game)
            .filter(
                (Game.home_team == team) | (Game.away_team == team),
                Game.game_date < before_date,
                Game.home_score.isnot(None)
            )
            .order_by(Game.game_date.desc())
            .limit(self.lookback_games)
            .all()
        )

        if not games:
            return 22.0  # League average

        points_allowed = []
        for game in games:
            if game.home_team == team:
                points_allowed.append(game.away_score or 0)
            else:
                points_allowed.append(game.home_score or 0)

        return float(np.mean(points_allowed)) if points_allowed else 22.0

    def get_matchup_signal(self, metrics: MatchupMetrics) -> float:
        """
        Convert matchup metrics to 0-1 signal for recommendation system

        Args:
            metrics: MatchupMetrics from analyze_matchup

        Returns:
            Signal value 0-1 (0.5 = neutral, higher = better matchup)
        """
        # Start with rank percentile (0-1 scale)
        signal = metrics.opponent_rank_percentile

        # Adjust for pace (higher pace = more opportunities)
        pace_factor = min(metrics.opponent_pace / 70.0, 1.2)  # 70 = fast pace
        signal *= pace_factor

        # Adjust for points allowed (higher scoring games = more volume)
        scoring_factor = min(metrics.opponent_points_allowed / 25.0, 1.2)  # 25 = high scoring
        signal *= scoring_factor

        # Weight by confidence
        signal = signal * metrics.confidence + 0.5 * (1 - metrics.confidence)

        # Ensure 0-1 range
        return max(0.0, min(1.0, signal))
