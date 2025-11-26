"""EPA (Expected Points Added) calculation utilities.

EPA is the #1 advanced metric in NFL analytics. It measures the value of each play
in terms of expected points, accounting for down, distance, and field position.
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class TeamEPA:
    """EPA metrics for a team."""
    team: str
    season: int
    week: int

    # Overall EPA
    total_epa: float
    epa_per_play: float

    # Offense EPA
    off_epa: float
    off_epa_per_play: float
    off_pass_epa: float
    off_rush_epa: float

    # Defense EPA (opponent EPA when defending)
    def_epa: float
    def_epa_per_play: float
    def_pass_epa: float
    def_rush_epa: float

    # Sample size
    plays: int
    off_plays: int
    def_plays: int


class EPACalculator:
    """Calculate EPA metrics for teams from play-by-play data."""

    def __init__(self, inputs_dir: str = 'inputs'):
        """Initialize EPA calculator.

        Args:
            inputs_dir: Directory containing play-by-play data
        """
        self.inputs_dir = Path(inputs_dir)
        self._pbp_cache = {}  # Cache loaded play-by-play data

    def load_pbp_data(self, season: int) -> Optional[pd.DataFrame]:
        """Load play-by-play data for a season.

        Args:
            season: Season year

        Returns:
            Play-by-play DataFrame or None if not available
        """
        # Check cache
        if season in self._pbp_cache:
            return self._pbp_cache[season]

        # Try to load from inputs
        pbp_file = self.inputs_dir / f'play_by_play_{season}.parquet'
        if pbp_file.exists():
            pbp = pd.read_parquet(pbp_file)
            self._pbp_cache[season] = pbp
            return pbp

        # Try historical directory
        pbp_file = self.inputs_dir / 'historical' / f'play_by_play_{season}.parquet'
        if pbp_file.exists():
            pbp = pd.read_parquet(pbp_file)
            self._pbp_cache[season] = pbp
            return pbp

        # Try to fetch using nfl_data_py
        try:
            import nfl_data_py as nfl
            print(f"  Fetching play-by-play data for {season} from nfl_data_py...")
            pbp = nfl.import_pbp_data([season])

            # Save for future use
            historical_dir = self.inputs_dir / 'historical'
            historical_dir.mkdir(parents=True, exist_ok=True)
            pbp_file = historical_dir / f'play_by_play_{season}.parquet'
            pbp.to_parquet(pbp_file)
            print(f"    ✓ Saved to {pbp_file}")

            self._pbp_cache[season] = pbp
            return pbp
        except Exception as e:
            print(f"  ! Could not load play-by-play for {season}: {e}")
            return None

    def calculate_team_epa(
        self,
        team: str,
        season: int,
        through_week: int,
        last_n_games: Optional[int] = None
    ) -> Optional[TeamEPA]:
        """Calculate EPA metrics for a team through a given week.

        Args:
            team: Team abbreviation
            season: Season year
            through_week: Calculate through this week (exclusive)
            last_n_games: If provided, only use last N games

        Returns:
            TeamEPA object or None if data not available
        """
        pbp = self.load_pbp_data(season)
        if pbp is None:
            return None

        # Filter to completed games before through_week
        pbp_filtered = pbp[
            (pbp['week'] < through_week) &
            (pbp['week'] > 0) &  # Regular season only
            (pd.notna(pbp['epa']))  # Valid EPA
        ].copy()

        if len(pbp_filtered) == 0:
            return None

        # Offensive plays (team has the ball)
        off_plays = pbp_filtered[pbp_filtered['posteam'] == team].copy()

        # Defensive plays (team is defending)
        def_plays = pbp_filtered[pbp_filtered['defteam'] == team].copy()

        # If last_n_games specified, filter to recent games
        if last_n_games is not None and len(off_plays) > 0:
            # Get unique game IDs for this team
            team_games = pbp_filtered[
                (pbp_filtered['posteam'] == team) | (pbp_filtered['defteam'] == team)
            ]['game_id'].unique()

            # Take last N games
            recent_games = sorted(team_games)[-last_n_games:]

            off_plays = off_plays[off_plays['game_id'].isin(recent_games)]
            def_plays = def_plays[def_plays['game_id'].isin(recent_games)]

        if len(off_plays) == 0 and len(def_plays) == 0:
            return None

        # Calculate offensive EPA
        off_epa_total = off_plays['epa'].sum() if len(off_plays) > 0 else 0.0
        off_epa_per_play = off_plays['epa'].mean() if len(off_plays) > 0 else 0.0

        # Pass vs rush EPA
        pass_plays = off_plays[off_plays['pass'] == 1]
        rush_plays = off_plays[off_plays['rush'] == 1]

        off_pass_epa = pass_plays['epa'].mean() if len(pass_plays) > 0 else 0.0
        off_rush_epa = rush_plays['epa'].mean() if len(rush_plays) > 0 else 0.0

        # Calculate defensive EPA (opponent EPA when this team is defending)
        def_epa_total = def_plays['epa'].sum() if len(def_plays) > 0 else 0.0
        def_epa_per_play = def_plays['epa'].mean() if len(def_plays) > 0 else 0.0

        # Defense pass vs rush EPA
        def_pass_plays = def_plays[def_plays['pass'] == 1]
        def_rush_plays = def_plays[def_plays['rush'] == 1]

        def_pass_epa = def_pass_plays['epa'].mean() if len(def_pass_plays) > 0 else 0.0
        def_rush_epa = def_rush_plays['epa'].mean() if len(def_rush_plays) > 0 else 0.0

        # Total EPA (offense - defense, since lower def EPA is better)
        total_epa = off_epa_total - def_epa_total
        total_plays = len(off_plays) + len(def_plays)
        epa_per_play = total_epa / total_plays if total_plays > 0 else 0.0

        return TeamEPA(
            team=team,
            season=season,
            week=through_week,
            total_epa=total_epa,
            epa_per_play=epa_per_play,
            off_epa=off_epa_total,
            off_epa_per_play=off_epa_per_play,
            off_pass_epa=off_pass_epa,
            off_rush_epa=off_rush_epa,
            def_epa=def_epa_total,
            def_epa_per_play=def_epa_per_play,
            def_pass_epa=def_pass_epa,
            def_rush_epa=def_rush_epa,
            plays=total_plays,
            off_plays=len(off_plays),
            def_plays=len(def_plays)
        )

    def get_epa_adjustment_for_game(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: int,
        last_n_games: int = 6
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate EPA-based adjustment for game total prediction.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: Season year
            week: Week number
            last_n_games: Number of recent games to use

        Returns:
            (adjustment, details_dict)
            adjustment: Points adjustment to add to baseline prediction
            details: Dictionary with EPA breakdown
        """
        home_epa = self.calculate_team_epa(home_team, season, week, last_n_games)
        away_epa = self.calculate_team_epa(away_team, season, week, last_n_games)

        details = {
            'home_off_epa': 0.0,
            'home_def_epa': 0.0,
            'away_off_epa': 0.0,
            'away_def_epa': 0.0,
            'epa_adjustment': 0.0,
            'home_epa_advantage': 0.0,
            'away_epa_advantage': 0.0,
        }

        if home_epa is None or away_epa is None:
            return 0.0, details

        details['home_off_epa'] = home_epa.off_epa_per_play
        details['home_def_epa'] = home_epa.def_epa_per_play
        details['away_off_epa'] = away_epa.off_epa_per_play
        details['away_def_epa'] = away_epa.def_epa_per_play

        # EPA matchup advantages
        # Home offense vs away defense
        home_epa_advantage = home_epa.off_epa_per_play - away_epa.def_epa_per_play

        # Away offense vs home defense
        away_epa_advantage = away_epa.off_epa_per_play - home_epa.def_epa_per_play

        details['home_epa_advantage'] = home_epa_advantage
        details['away_epa_advantage'] = away_epa_advantage

        # Convert EPA to scoring adjustment
        # EPA is already in "points added" so we don't need to multiply by plays
        # Instead, use a conservative weight factor
        #
        # Reasoning:
        # - EPA differences are already in terms of points per play
        # - Typical game: 130 total plays, so EPA/play * 130 = EPA/game
        # - But recent EPA is highly correlated with recent PPG (which baseline uses)
        # - So we only want the MARGINAL value of EPA beyond what PPG captures
        #
        # Use 20% weight to capture EPA insight without double-counting recent form

        weight = 0.20  # Conservative weight for EPA adjustment

        # Calculate expected scoring adjustment (marginal improvement over PPG)
        home_expected = home_epa_advantage * weight
        away_expected = away_epa_advantage * weight

        # Total game adjustment
        # Cap at ±10 points to prevent extreme predictions
        adjustment = np.clip(home_expected + away_expected, -10.0, 10.0)

        details['epa_adjustment'] = adjustment

        return adjustment, details


def get_rest_differential(
    home_team: str,
    away_team: str,
    season: int,
    week: int,
    schedule: pd.DataFrame
) -> int:
    """Calculate rest differential (home rest days - away rest days).

    Args:
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        season: Season year
        week: Week number
        schedule: Schedule DataFrame

    Returns:
        Rest differential in days (positive = home team more rested)
    """
    # Get previous games for each team
    home_prev = schedule[
        ((schedule['home_team'] == home_team) | (schedule['away_team'] == home_team)) &
        (schedule['week'] < week) &
        (schedule['season'] == season)
    ].sort_values('week').tail(1)

    away_prev = schedule[
        ((schedule['home_team'] == away_team) | (schedule['away_team'] == away_team)) &
        (schedule['week'] < week) &
        (schedule['season'] == season)
    ].sort_values('week').tail(1)

    if len(home_prev) == 0 or len(away_prev) == 0:
        return 0

    home_last_week = home_prev['week'].iloc[0]
    away_last_week = away_prev['week'].iloc[0]

    # Calculate days (7 days per week on average)
    home_rest = (week - home_last_week) * 7
    away_rest = (week - away_last_week) * 7

    return home_rest - away_rest
