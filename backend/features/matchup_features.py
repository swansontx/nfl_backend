"""
Matchup and environment feature extraction

Provides context beyond player stats:
- Opponent defensive strength
- Game environment (home/away, weather, etc.)
- Game script (Vegas lines, implied totals)
- Team context (pace, play calling tendencies)
"""

from dataclasses import dataclass
from typing import Optional, Dict
from datetime import datetime

from backend.config.logging_config import get_logger
from backend.database.session import get_db
from backend.database.models import Game, Player, PlayerGameFeature

logger = get_logger(__name__)


@dataclass
class MatchupFeatures:
    """Matchup and environment features for a player-game"""

    player_id: str
    game_id: str

    # Game context
    is_home: bool = False
    is_divisional: bool = False
    is_primetime: bool = False
    rest_days: int = 7

    # Vegas lines (if available)
    game_total: Optional[float] = None
    spread: Optional[float] = None  # Positive = favorite
    team_implied_total: Optional[float] = None

    # Weather (for outdoor games)
    temperature: Optional[float] = None
    wind_speed: Optional[float] = None
    precipitation: Optional[str] = None  # "none", "rain", "snow"
    is_dome: bool = False

    # Opponent strength
    opp_team: str = ""
    opp_pass_def_rank: Optional[int] = None  # 1-32 rank
    opp_rush_def_rank: Optional[int] = None
    opp_yds_allowed_per_game: Optional[float] = None
    opp_points_allowed_per_game: Optional[float] = None

    # Team pace and tendencies
    team_pace: Optional[float] = None  # Plays per game
    team_pass_rate: Optional[float] = None  # Pass play %
    team_redzone_rate: Optional[float] = None  # RZ trips per game

    # Position-specific matchups (advanced)
    cb_matchup_quality: Optional[str] = None  # "elite", "good", "average", "weak"
    expected_coverage: Optional[str] = None  # "man", "zone", "mixed"


class MatchupFeatureExtractor:
    """Extract matchup and environment features"""

    def __init__(self):
        """Initialize extractor"""
        # Cache for defensive rankings
        self._def_rankings_cache: Dict = {}

    def extract_matchup_features(
        self,
        player_id: str,
        game_id: str
    ) -> MatchupFeatures:
        """
        Extract full matchup feature set for a player-game

        Args:
            player_id: Player ID
            game_id: Game ID

        Returns:
            MatchupFeatures object
        """
        with get_db() as session:
            # Get game
            game = session.query(Game).filter(Game.game_id == game_id).first()
            if not game:
                logger.warning("game_not_found", game_id=game_id)
                return MatchupFeatures(player_id=player_id, game_id=game_id)

            # Get player
            player = session.query(Player).filter(Player.player_id == player_id).first()
            if not player:
                logger.warning("player_not_found", player_id=player_id)
                return MatchupFeatures(player_id=player_id, game_id=game_id)

            # Determine home/away
            is_home = player.team == game.home_team
            opponent = game.away_team if is_home else game.home_team

            # Build features
            features = MatchupFeatures(
                player_id=player_id,
                game_id=game_id,
                is_home=is_home,
                opp_team=opponent,
            )

            # Add game context
            self._add_game_context(features, game)

            # Add vegas lines
            self._add_vegas_context(features, game, player.team)

            # Add weather
            self._add_weather_context(features, game)

            # Add opponent defense
            self._add_opponent_defense(features, opponent, game.season, player.position)

            # Add team pace
            self._add_team_pace(features, player.team, game.season)

            logger.debug("matchup_features_extracted", player_id=player_id, game_id=game_id)

            return features

    def _add_game_context(self, features: MatchupFeatures, game: Game) -> None:
        """Add basic game context"""
        # Divisional game detection (would need division data)
        # For now, placeholder
        features.is_divisional = False

        # Primetime detection (Sunday/Monday/Thursday night)
        # Would need game time data
        features.is_primetime = False

        # Rest days (would need previous game date)
        features.rest_days = 7  # Default

    def _add_vegas_context(self, features: MatchupFeatures, game: Game, team: str) -> None:
        """Add Vegas line context"""
        # Game total (if stored in DB)
        # features.game_total = game.total  # Would come from odds data

        # Spread (positive = favorite)
        # features.spread = game.spread if team == game.home_team else -game.spread

        # Implied total = (total / 2) + (spread / 2)
        # if features.game_total and features.spread:
        #     features.team_implied_total = (features.game_total / 2) + (features.spread / 2)

        # Placeholder for now
        features.game_total = 45.0  # Average NFL total
        features.spread = 0.0
        features.team_implied_total = 22.5

    def _add_weather_context(self, features: MatchupFeatures, game: Game) -> None:
        """Add weather context"""
        # Weather data would come from external API
        # Domes: ATL, DET, HOU, IND, LV, LAR, MIN, NO
        dome_teams = {'ATL', 'DET', 'HOU', 'IND', 'LV', 'LAR', 'MIN', 'NO'}

        features.is_dome = game.home_team in dome_teams

        if features.is_dome:
            features.temperature = 72.0
            features.wind_speed = 0.0
            features.precipitation = "none"
        else:
            # Would fetch from weather API
            features.temperature = 60.0  # Placeholder
            features.wind_speed = 5.0
            features.precipitation = "none"

    def _add_opponent_defense(
        self,
        features: MatchupFeatures,
        opponent: str,
        season: int,
        position: str
    ) -> None:
        """Add opponent defensive strength metrics"""
        # This would compute from historical data
        # For now, use placeholders

        # Cache key
        cache_key = f"{opponent}_{season}"

        if cache_key in self._def_rankings_cache:
            rankings = self._def_rankings_cache[cache_key]
            features.opp_pass_def_rank = rankings.get('pass_def_rank')
            features.opp_rush_def_rank = rankings.get('rush_def_rank')
            features.opp_yds_allowed_per_game = rankings.get('yds_allowed')
            features.opp_points_allowed_per_game = rankings.get('pts_allowed')
            return

        # Would compute from actual game data
        # For now, placeholder
        features.opp_pass_def_rank = 16  # Mid-tier
        features.opp_rush_def_rank = 16
        features.opp_yds_allowed_per_game = 350.0
        features.opp_points_allowed_per_game = 22.0

    def _add_team_pace(
        self,
        features: MatchupFeatures,
        team: str,
        season: int
    ) -> None:
        """Add team pace and tendency metrics"""
        # Would compute from actual play-by-play data
        # For now, placeholder

        features.team_pace = 65.0  # Plays per game (league average)
        features.team_pass_rate = 0.60  # 60% pass plays
        features.team_redzone_rate = 3.5  # RZ trips per game

    def compute_defensive_rankings(self, season: int) -> Dict[str, Dict]:
        """
        Compute defensive rankings for all teams in a season

        This should be run periodically to update the cache

        Args:
            season: NFL season

        Returns:
            Dict mapping team to defensive metrics
        """
        with get_db() as session:
            # Get all games for the season
            games = session.query(Game).filter(Game.season == season).all()

            team_stats: Dict[str, Dict] = {}

            # Initialize
            for game in games:
                for team in [game.home_team, game.away_team]:
                    if team not in team_stats:
                        team_stats[team] = {
                            'pass_yds_allowed': [],
                            'rush_yds_allowed': [],
                            'total_yds_allowed': [],
                            'points_allowed': []
                        }

            # Aggregate stats from games
            # This would need actual game results stored
            # For now, return empty

            logger.info("defensive_rankings_computed", season=season, teams=len(team_stats))

            return team_stats

    def adjust_projection_for_matchup(
        self,
        base_projection: float,
        matchup_features: MatchupFeatures,
        stat_type: str
    ) -> float:
        """
        Adjust a base projection based on matchup context

        Args:
            base_projection: Base projection value
            matchup_features: Matchup context
            stat_type: Type of stat ("passing_yards", "rushing_yards", etc.)

        Returns:
            Adjusted projection
        """
        adjustment = 1.0

        # Home/away adjustment
        if matchup_features.is_home:
            adjustment *= 1.05  # 5% boost for home
        else:
            adjustment *= 0.95

        # Weather adjustments (for passing)
        if stat_type in ["passing_yards", "receiving_yards"]:
            if matchup_features.wind_speed and matchup_features.wind_speed > 15:
                adjustment *= 0.90  # 10% penalty for high wind
            if matchup_features.precipitation in ["rain", "snow"]:
                adjustment *= 0.85  # 15% penalty for precipitation

        # Opponent defense adjustment
        if stat_type == "passing_yards":
            if matchup_features.opp_pass_def_rank:
                # Top 10 defense = penalty, bottom 10 = boost
                if matchup_features.opp_pass_def_rank <= 10:
                    adjustment *= 0.90
                elif matchup_features.opp_pass_def_rank >= 23:
                    adjustment *= 1.10

        elif stat_type == "rushing_yards":
            if matchup_features.opp_rush_def_rank:
                if matchup_features.opp_rush_def_rank <= 10:
                    adjustment *= 0.90
                elif matchup_features.opp_rush_def_rank >= 23:
                    adjustment *= 1.10

        # Game total adjustment (high totals = more volume)
        if matchup_features.game_total:
            if matchup_features.game_total >= 50:
                adjustment *= 1.10  # High scoring game
            elif matchup_features.game_total <= 40:
                adjustment *= 0.90  # Low scoring game

        # Spread adjustment (favorites pass more when winning)
        if matchup_features.spread:
            if stat_type == "passing_yards" and matchup_features.spread > 7:
                adjustment *= 1.05  # Big favorites pass more
            elif stat_type == "rushing_yards" and matchup_features.spread > 7:
                adjustment *= 1.10  # Big favorites run more late

        adjusted = base_projection * adjustment

        logger.debug(
            "projection_adjusted",
            base=base_projection,
            adjusted=adjusted,
            adjustment_factor=adjustment,
            stat_type=stat_type
        )

        return adjusted
