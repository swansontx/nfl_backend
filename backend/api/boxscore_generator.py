"""NFL Boxscore Generator - Generate boxscores from play-by-play data.

Creates comprehensive boxscores similar to ESPN:
- Final score and quarter scores
- Team statistics (rushing, passing, turnovers, etc.)
- Player statistics (passing, rushing, receiving)
- Scoring summary (touchdowns, field goals)
"""

from pathlib import Path
from typing import Dict, List, Optional
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from backend.api.cache import cached, CACHE_TTL


@dataclass
class PlayerStats:
    """Player statistics for a game."""
    player_id: str
    player_name: str
    team: str

    # Passing
    pass_attempts: int = 0
    pass_completions: int = 0
    pass_yards: int = 0
    pass_tds: int = 0
    interceptions: int = 0
    sacks: int = 0
    sack_yards: int = 0

    # Rushing
    rush_attempts: int = 0
    rush_yards: int = 0
    rush_tds: int = 0
    fumbles: int = 0
    fumbles_lost: int = 0

    # Receiving
    receptions: int = 0
    targets: int = 0
    rec_yards: int = 0
    rec_tds: int = 0


@dataclass
class TeamStats:
    """Team statistics for a game."""
    team: str

    # Scoring
    total_points: int = 0
    q1_points: int = 0
    q2_points: int = 0
    q3_points: int = 0
    q4_points: int = 0
    ot_points: int = 0

    # Offense
    total_yards: int = 0
    passing_yards: int = 0
    rushing_yards: int = 0
    first_downs: int = 0
    third_down_conversions: int = 0
    third_down_attempts: int = 0
    fourth_down_conversions: int = 0
    fourth_down_attempts: int = 0

    # Turnovers
    turnovers: int = 0
    fumbles_lost: int = 0
    interceptions_thrown: int = 0

    # Penalties
    penalties: int = 0
    penalty_yards: int = 0

    # Time of possession
    time_of_possession: int = 0  # seconds


@dataclass
class ScoringPlay:
    """A scoring play."""
    quarter: int
    time: str
    team: str
    description: str
    away_score: int
    home_score: int
    play_type: str  # 'TD', 'FG', '2PT', 'PAT', 'SAFETY'


@dataclass
class Boxscore:
    """Complete game boxscore."""
    game_id: str
    away_team: str
    home_team: str
    away_score: int
    home_score: int

    away_stats: TeamStats
    home_stats: TeamStats

    away_players: Dict[str, PlayerStats] = field(default_factory=dict)
    home_players: Dict[str, PlayerStats] = field(default_factory=dict)

    scoring_plays: List[ScoringPlay] = field(default_factory=list)


class BoxscoreGenerator:
    """Generate boxscores from play-by-play data."""

    def __init__(self, pbp_dir: Path = Path('inputs')):
        """Initialize boxscore generator.

        Args:
            pbp_dir: Directory containing play-by-play CSV files
        """
        self.pbp_dir = pbp_dir

    @cached(ttl_seconds=CACHE_TTL['boxscore'])  # 30 minutes
    def generate_boxscore(self, game_id: str) -> Optional[Boxscore]:
        """Generate boxscore for a game.

        Args:
            game_id: Game ID (format: YYYY_WW_AWAY_HOME)

        Returns:
            Boxscore object or None if not found
        """
        # Parse game_id to get year
        try:
            parts = game_id.split('_')
            if len(parts) < 4:
                return None

            year = parts[0]
            away_team = parts[2]
            home_team = parts[3]

        except:
            return None

        # Load play-by-play data
        pbp_file = self.pbp_dir / f'play_by_play_{year}.csv'

        if not pbp_file.exists():
            print(f"⚠ Play-by-play file not found: {pbp_file}")
            return None

        # Initialize stats
        away_stats = TeamStats(team=away_team)
        home_stats = TeamStats(team=home_team)
        away_players = {}
        home_players = {}
        scoring_plays = []

        try:
            with open(pbp_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    # Only process plays for this game
                    if row.get('game_id') != game_id and row.get('old_game_id') != game_id:
                        continue

                    self._process_play(row, away_team, home_team, away_stats, home_stats,
                                     away_players, home_players, scoring_plays)

            # Create boxscore
            boxscore = Boxscore(
                game_id=game_id,
                away_team=away_team,
                home_team=home_team,
                away_score=away_stats.total_points,
                home_score=home_stats.total_points,
                away_stats=away_stats,
                home_stats=home_stats,
                away_players=away_players,
                home_players=home_players,
                scoring_plays=scoring_plays
            )

            return boxscore

        except Exception as e:
            print(f"✗ Error generating boxscore: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _process_play(self, play: Dict, away_team: str, home_team: str,
                     away_stats: TeamStats, home_stats: TeamStats,
                     away_players: Dict, home_players: Dict,
                     scoring_plays: List[ScoringPlay]):
        """Process a single play and update stats."""

        posteam = play.get('posteam', '')
        defteam = play.get('defteam', '')

        # Determine which team's stats to update
        if posteam == away_team:
            off_stats = away_stats
            def_stats = home_stats
            off_players = away_players
        elif posteam == home_team:
            off_stats = home_stats
            def_stats = away_stats
            off_players = home_players
        else:
            return  # Not a play with possession

        # Update yards
        yards_gained = int(play.get('yards_gained', 0) or 0)
        off_stats.total_yards += yards_gained

        play_type = play.get('play_type', '')

        # Process passing plays
        if play_type == 'pass':
            passer_id = play.get('passer_player_id', play.get('passer_id', ''))
            receiver_id = play.get('receiver_player_id', play.get('receiver_id', ''))

            if passer_id:
                # Get or create passer stats
                if passer_id not in off_players:
                    off_players[passer_id] = PlayerStats(
                        player_id=passer_id,
                        player_name=play.get('passer_player_name', passer_id),
                        team=posteam
                    )

                passer = off_players[passer_id]
                passer.pass_attempts += 1

                if play.get('complete_pass') == '1':
                    passer.pass_completions += 1
                    passer.pass_yards += yards_gained
                    off_stats.passing_yards += yards_gained

                if play.get('pass_touchdown') == '1':
                    passer.pass_tds += 1

                if play.get('interception') == '1':
                    passer.interceptions += 1
                    off_stats.interceptions_thrown += 1
                    off_stats.turnovers += 1

                if play.get('sack') == '1':
                    passer.sacks += 1
                    sack_yds = abs(yards_gained)  # Sacks are negative yards
                    passer.sack_yards += sack_yds

            if receiver_id:
                # Get or create receiver stats
                if receiver_id not in off_players:
                    off_players[receiver_id] = PlayerStats(
                        player_id=receiver_id,
                        player_name=play.get('receiver_player_name', receiver_id),
                        team=posteam
                    )

                receiver = off_players[receiver_id]
                receiver.targets += 1

                if play.get('complete_pass') == '1':
                    receiver.receptions += 1
                    receiver.rec_yards += yards_gained

                if play.get('pass_touchdown') == '1':
                    receiver.rec_tds += 1

        # Process rushing plays
        elif play_type == 'run':
            rusher_id = play.get('rusher_player_id', play.get('rusher_id', ''))

            if rusher_id:
                # Get or create rusher stats
                if rusher_id not in off_players:
                    off_players[rusher_id] = PlayerStats(
                        player_id=rusher_id,
                        player_name=play.get('rusher_player_name', rusher_id),
                        team=posteam
                    )

                rusher = off_players[rusher_id]
                rusher.rush_attempts += 1
                rusher.rush_yards += yards_gained
                off_stats.rushing_yards += yards_gained

                if play.get('rush_touchdown') == '1':
                    rusher.rush_tds += 1

        # Track first downs
        if play.get('first_down') == '1':
            off_stats.first_downs += 1

        # Track third down conversions
        if play.get('third_down_converted') == '1':
            off_stats.third_down_conversions += 1
        if play.get('third_down_failed') == '1':
            off_stats.third_down_attempts += 1

        # Track fourth down conversions
        if play.get('fourth_down_converted') == '1':
            off_stats.fourth_down_conversions += 1
        if play.get('fourth_down_failed') == '1':
            off_stats.fourth_down_attempts += 1

        # Track fumbles
        if play.get('fumble_lost') == '1':
            off_stats.fumbles_lost += 1
            off_stats.turnovers += 1

        # Track penalties
        if play.get('penalty') == '1':
            penalty_team = play.get('penalty_team', '')
            penalty_yards = int(play.get('penalty_yards', 0) or 0)

            if penalty_team == away_team:
                away_stats.penalties += 1
                away_stats.penalty_yards += penalty_yards
            elif penalty_team == home_team:
                home_stats.penalties += 1
                home_stats.penalty_yards += penalty_yards

        # Track scoring plays
        if play.get('touchdown') == '1' or play.get('field_goal_result') == 'made':
            quarter = int(play.get('qtr', 0))
            time_remaining = play.get('time', play.get('game_seconds_remaining', ''))
            description = play.get('desc', '')
            posteam_score = int(play.get('posteam_score_post', 0))
            defteam_score = int(play.get('defteam_score_post', 0))

            # Determine scores
            if posteam == away_team:
                away_score = posteam_score
                home_score = defteam_score
            else:
                away_score = defteam_score
                home_score = posteam_score

            play_type_str = 'TD' if play.get('touchdown') == '1' else 'FG'

            scoring_play = ScoringPlay(
                quarter=quarter,
                time=str(time_remaining),
                team=posteam,
                description=description,
                away_score=away_score,
                home_score=home_score,
                play_type=play_type_str
            )
            scoring_plays.append(scoring_play)

            # Update quarter scores
            points = int(play.get('posteam_score_post', 0)) - int(play.get('posteam_score', 0))

            if quarter == 1:
                off_stats.q1_points += points
            elif quarter == 2:
                off_stats.q2_points += points
            elif quarter == 3:
                off_stats.q3_points += points
            elif quarter == 4:
                off_stats.q4_points += points
            else:
                off_stats.ot_points += points

            off_stats.total_points += points


# Singleton instance
boxscore_generator = BoxscoreGenerator()
