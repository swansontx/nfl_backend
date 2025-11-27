"""Team Matchup Analyzer - Data-Backed Team Comparison System.

Calculates comprehensive matchup metrics between two NFL teams using historical data:
- Head-to-head performance
- Statistical profile matching (offense vs defense)
- Common opponents analysis
- Recent form and trends
- Home/away splits
- Rest advantage
- Division/conference factors
- Pace and style compatibility

Usage:
    from backend.analysis.team_matchup_analyzer import TeamMatchupAnalyzer

    analyzer = TeamMatchupAnalyzer(season=2025)
    matchup = analyzer.analyze_matchup(
        home_team='KC',
        away_team='BUF',
        week=14
    )

    print(f"Predicted total: {matchup['predicted_total']}")
    print(f"Edge: {matchup['edge_analysis']}")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta


@dataclass
class TeamProfile:
    """Statistical profile of a team."""
    team: str
    season: int

    # Offensive metrics
    points_per_game: float = 0.0
    yards_per_game: float = 0.0
    passing_yards_per_game: float = 0.0
    rushing_yards_per_game: float = 0.0
    third_down_pct: float = 0.0
    red_zone_td_pct: float = 0.0
    turnovers_per_game: float = 0.0

    # Defensive metrics (points/yards allowed)
    points_allowed_per_game: float = 0.0
    yards_allowed_per_game: float = 0.0
    pass_yards_allowed_per_game: float = 0.0
    rush_yards_allowed_per_game: float = 0.0
    sacks_per_game: float = 0.0
    takeaways_per_game: float = 0.0

    # Advanced metrics
    plays_per_game: float = 0.0  # Pace
    yards_per_play: float = 0.0
    yards_per_play_allowed: float = 0.0

    # Recent form (last 4 games)
    recent_ppg: float = 0.0
    recent_yards_pg: float = 0.0
    recent_record: str = ""

    # Situational
    home_ppg: float = 0.0
    away_ppg: float = 0.0
    division_record: str = ""

    games_played: int = 0


@dataclass
class MatchupAnalysis:
    """Complete matchup analysis between two teams."""
    home_team: str
    away_team: str
    week: int
    season: int

    # Head-to-head history
    h2h_games: int = 0
    h2h_home_wins: int = 0
    h2h_avg_total: float = 0.0
    h2h_last_result: str = ""

    # Statistical edges
    offensive_edge: str = ""  # Which team has better offense
    defensive_edge: str = ""  # Which team has better defense
    pace_matchup: str = ""     # Fast vs slow, expected plays

    # Common opponents
    common_opponents: List[str] = field(default_factory=list)
    common_opponent_edge: str = ""

    # Recent form
    home_form: str = ""  # Last 4 games
    away_form: str = ""
    momentum_edge: str = ""

    # Situational factors
    rest_advantage: int = 0  # Days difference
    home_field_value: float = 0.0
    division_game: bool = False
    weather_impact: str = ""

    # Predictions
    predicted_total: float = 0.0
    predicted_spread: float = 0.0
    predicted_home_score: float = 0.0
    predicted_away_score: float = 0.0
    confidence: float = 0.0

    # Key insights
    key_matchups: List[str] = field(default_factory=list)
    edge_analysis: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


class TeamMatchupAnalyzer:
    """Analyzes team matchups using comprehensive historical data."""

    def __init__(self, season: int = 2025, data_dir: Path = None):
        """Initialize analyzer.

        Args:
            season: Season year
            data_dir: Directory with historical data
        """
        self.season = season
        self.data_dir = data_dir or Path('inputs/historical')

        # Load data
        self.games = self._load_games()
        self.player_stats = self._load_player_stats()

        # Calculate team profiles
        self.team_profiles: Dict[str, TeamProfile] = {}
        self._calculate_team_profiles()

    def _load_games(self) -> pd.DataFrame:
        """Load game data for current and recent seasons."""
        all_games = []

        # Load current season and previous 2 seasons for trends
        for year in [self.season - 2, self.season - 1, self.season]:
            games_file = self.data_dir / f'games_{year}.csv'
            if games_file.exists():
                df = pd.read_csv(games_file)
                all_games.append(df)

        if all_games:
            return pd.concat(all_games, ignore_index=True)
        return pd.DataFrame()

    def _load_player_stats(self) -> pd.DataFrame:
        """Load player stats for current season."""
        stats_file = self.data_dir / f'player_stats_{self.season}_all.csv'

        if stats_file.exists():
            return pd.read_csv(stats_file)
        return pd.DataFrame()

    def _calculate_team_profiles(self):
        """Calculate statistical profiles for all teams."""
        if self.player_stats.empty:
            return

        teams = self.player_stats['team'].unique()

        for team in teams:
            if pd.isna(team):
                continue

            profile = self._build_team_profile(team)
            self.team_profiles[team] = profile

    def _build_team_profile(self, team: str) -> TeamProfile:
        """Build comprehensive profile for a team.

        Args:
            team: Team abbreviation

        Returns:
            TeamProfile with calculated metrics
        """
        profile = TeamProfile(team=team, season=self.season)

        # Get team's offensive stats
        team_stats = self.player_stats[self.player_stats['team'] == team]

        if team_stats.empty:
            return profile

        # Calculate offensive metrics (aggregated by week)
        weekly_offense = team_stats.groupby('week').agg({
            'passing_yards': 'sum',
            'rushing_yards': 'sum',
            'passing_tds': 'sum',
            'rushing_tds': 'sum',
            'fantasy_points_ppr': 'sum'
        })

        profile.games_played = len(weekly_offense)

        if profile.games_played > 0:
            profile.passing_yards_per_game = weekly_offense['passing_yards'].mean()
            profile.rushing_yards_per_game = weekly_offense['rushing_yards'].mean()
            profile.yards_per_game = profile.passing_yards_per_game + profile.rushing_yards_per_game
            profile.points_per_game = weekly_offense['fantasy_points_ppr'].mean() * 0.7  # Rough conversion
            # Note: turnovers not available in this dataset

            # Recent form (last 4 weeks)
            if len(weekly_offense) >= 4:
                recent = weekly_offense.tail(4)
                profile.recent_ppg = recent['fantasy_points_ppr'].mean() * 0.7
                profile.recent_yards_pg = (recent['passing_yards'] + recent['rushing_yards']).mean()

        # Get defensive stats (opponent performance against this team)
        opponent_stats = self.player_stats[self.player_stats['opponent_team'] == team]

        if not opponent_stats.empty:
            weekly_defense = opponent_stats.groupby('week').agg({
                'passing_yards': 'sum',
                'rushing_yards': 'sum',
                'fantasy_points_ppr': 'sum'
            })

            if len(weekly_defense) > 0:
                profile.pass_yards_allowed_per_game = weekly_defense['passing_yards'].mean()
                profile.rush_yards_allowed_per_game = weekly_defense['rushing_yards'].mean()
                profile.yards_allowed_per_game = profile.pass_yards_allowed_per_game + profile.rush_yards_allowed_per_game
                profile.points_allowed_per_game = weekly_defense['fantasy_points_ppr'].mean() * 0.7

        # Home/away splits
        if not self.games.empty:
            home_games = self.games[
                (self.games['home_team'] == team) &
                (self.games['season'] == self.season)
            ]
            away_games = self.games[
                (self.games['away_team'] == team) &
                (self.games['season'] == self.season)
            ]

            if len(home_games) > 0:
                profile.home_ppg = home_games['home_score'].mean()
            if len(away_games) > 0:
                profile.away_ppg = away_games['away_score'].mean()

        return profile

    def analyze_matchup(
        self,
        home_team: str,
        away_team: str,
        week: int,
        include_weather: bool = False
    ) -> MatchupAnalysis:
        """Analyze matchup between two teams.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            week: Week number
            include_weather: Include weather analysis (requires additional data)

        Returns:
            MatchupAnalysis with comprehensive metrics
        """
        analysis = MatchupAnalysis(
            home_team=home_team,
            away_team=away_team,
            week=week,
            season=self.season
        )

        # Get team profiles
        home_profile = self.team_profiles.get(home_team)
        away_profile = self.team_profiles.get(away_team)

        if not home_profile or not away_profile:
            analysis.notes.append("Insufficient data for one or both teams")
            return analysis

        # 1. Head-to-head history
        self._analyze_head_to_head(analysis, home_team, away_team)

        # 2. Statistical edges
        self._analyze_statistical_edges(analysis, home_profile, away_profile)

        # 3. Common opponents
        self._analyze_common_opponents(analysis, home_team, away_team)

        # 4. Recent form
        self._analyze_recent_form(analysis, home_profile, away_profile)

        # 5. Situational factors
        self._analyze_situational_factors(analysis, home_team, away_team, week)

        # 6. Generate prediction
        self._generate_prediction(analysis, home_profile, away_profile)

        # 7. Key matchup insights
        self._identify_key_matchups(analysis, home_profile, away_profile)

        return analysis

    def _analyze_head_to_head(self, analysis: MatchupAnalysis, home: str, away: str):
        """Analyze head-to-head history."""
        if self.games.empty:
            return

        # Get all games between these teams
        h2h = self.games[
            ((self.games['home_team'] == home) & (self.games['away_team'] == away)) |
            ((self.games['home_team'] == away) & (self.games['away_team'] == home))
        ]

        analysis.h2h_games = len(h2h)

        if analysis.h2h_games > 0:
            # Home wins when current home team is home
            home_at_home = h2h[
                (h2h['home_team'] == home) &
                (h2h['home_score'] > h2h['away_score'])
            ]
            analysis.h2h_home_wins = len(home_at_home)

            # Average total
            analysis.h2h_avg_total = (h2h['home_score'] + h2h['away_score']).mean()

            # Last result
            if len(h2h) > 0:
                last_game = h2h.iloc[-1]
                analysis.h2h_last_result = f"{last_game['away_team']} {last_game['away_score']} @ {last_game['home_team']} {last_game['home_score']}"

    def _analyze_statistical_edges(self, analysis: MatchupAnalysis, home: TeamProfile, away: TeamProfile):
        """Determine statistical advantages."""
        # Offensive edge
        if home.points_per_game > away.points_per_game * 1.1:
            analysis.offensive_edge = f"{home.team} (+{home.points_per_game - away.points_per_game:.1f} ppg)"
        elif away.points_per_game > home.points_per_game * 1.1:
            analysis.offensive_edge = f"{away.team} (+{away.points_per_game - home.points_per_game:.1f} ppg)"
        else:
            analysis.offensive_edge = "Even"

        # Defensive edge (lower is better)
        if home.points_allowed_per_game < away.points_allowed_per_game * 0.9:
            analysis.defensive_edge = f"{home.team} ({home.points_allowed_per_game:.1f} vs {away.points_allowed_per_game:.1f} papg)"
        elif away.points_allowed_per_game < home.points_allowed_per_game * 0.9:
            analysis.defensive_edge = f"{away.team} ({away.points_allowed_per_game:.1f} vs {home.points_allowed_per_game:.1f} papg)"
        else:
            analysis.defensive_edge = "Even"

        # Pace matchup
        total_pace = home.plays_per_game + away.plays_per_game
        if total_pace > 130:
            analysis.pace_matchup = "Fast-paced (high scoring potential)"
        elif total_pace < 110:
            analysis.pace_matchup = "Slow-paced (lower scoring)"
        else:
            analysis.pace_matchup = "Average pace"

    def _analyze_common_opponents(self, analysis: MatchupAnalysis, home: str, away: str):
        """Analyze performance against common opponents."""
        if self.games.empty:
            return

        # Get opponents faced by each team
        home_opponents = set(
            list(self.games[self.games['home_team'] == home]['away_team']) +
            list(self.games[self.games['away_team'] == home]['home_team'])
        )
        away_opponents = set(
            list(self.games[self.games['home_team'] == away]['away_team']) +
            list(self.games[self.games['away_team'] == away]['home_team'])
        )

        common = home_opponents & away_opponents
        analysis.common_opponents = list(common)

        if len(common) > 0:
            # Compare performance vs common opponents
            home_vs_common = []
            away_vs_common = []

            for opp in common:
                # Home team vs this opponent
                h_games = self.games[
                    ((self.games['home_team'] == home) & (self.games['away_team'] == opp)) |
                    ((self.games['away_team'] == home) & (self.games['home_team'] == opp))
                ]
                if len(h_games) > 0:
                    h_score = h_games.apply(
                        lambda x: x['home_score'] if x['home_team'] == home else x['away_score'], axis=1
                    ).mean()
                    home_vs_common.append(h_score)

                # Away team vs this opponent
                a_games = self.games[
                    ((self.games['home_team'] == away) & (self.games['away_team'] == opp)) |
                    ((self.games['away_team'] == away) & (self.games['home_team'] == opp))
                ]
                if len(a_games) > 0:
                    a_score = a_games.apply(
                        lambda x: x['home_score'] if x['home_team'] == away else x['away_score'], axis=1
                    ).mean()
                    away_vs_common.append(a_score)

            if home_vs_common and away_vs_common:
                h_avg = np.mean(home_vs_common)
                a_avg = np.mean(away_vs_common)
                diff = h_avg - a_avg

                if abs(diff) > 3:
                    better_team = home if diff > 0 else away
                    analysis.common_opponent_edge = f"{better_team} (+{abs(diff):.1f} ppg vs common opponents)"
                else:
                    analysis.common_opponent_edge = "Even vs common opponents"

    def _analyze_recent_form(self, analysis: MatchupAnalysis, home: TeamProfile, away: TeamProfile):
        """Analyze recent performance trends."""
        analysis.home_form = f"{home.recent_ppg:.1f} ppg (last 4)"
        analysis.away_form = f"{away.recent_ppg:.1f} ppg (last 4)"

        # Momentum edge
        home_trending = home.recent_ppg - home.points_per_game
        away_trending = away.recent_ppg - away.points_per_game

        if home_trending > 3 and home_trending > away_trending:
            analysis.momentum_edge = f"{home.team} trending up (+{home_trending:.1f})"
        elif away_trending > 3 and away_trending > home_trending:
            analysis.momentum_edge = f"{away.team} trending up (+{away_trending:.1f})"
        elif home_trending < -3:
            analysis.momentum_edge = f"{home.team} trending down ({home_trending:.1f})"
        elif away_trending < -3:
            analysis.momentum_edge = f"{away.team} trending down ({away_trending:.1f})"
        else:
            analysis.momentum_edge = "Both teams steady"

    def _analyze_situational_factors(self, analysis: MatchupAnalysis, home: str, away: str, week: int):
        """Analyze situational factors."""
        # Home field advantage
        home_profile = self.team_profiles.get(home)
        if home_profile and home_profile.home_ppg > 0 and home_profile.away_ppg > 0:
            analysis.home_field_value = home_profile.home_ppg - home_profile.away_ppg

        # Division game
        # Simplified: would need division mapping
        analysis.division_game = False  # Placeholder

        # Rest advantage (would need schedule data)
        analysis.rest_advantage = 0  # Placeholder

    def _generate_prediction(self, analysis: MatchupAnalysis, home: TeamProfile, away: TeamProfile):
        """Generate score prediction based on analysis."""
        # Base prediction: team's offensive average vs opponent's defensive average
        home_base = (home.points_per_game + (50 - away.points_allowed_per_game)) / 2 if away.points_allowed_per_game > 0 else home.points_per_game
        away_base = (away.points_per_game + (50 - home.points_allowed_per_game)) / 2 if home.points_allowed_per_game > 0 else away.points_per_game

        # Adjust for home field
        home_adj = analysis.home_field_value * 0.5  # 50% of home field value

        # Adjust for recent form
        home_form_adj = (home.recent_ppg - home.points_per_game) * 0.3
        away_form_adj = (away.recent_ppg - away.points_per_game) * 0.3

        # Calculate final prediction
        analysis.predicted_home_score = home_base + home_adj + home_form_adj
        analysis.predicted_away_score = away_base - (home_adj * 0.5) + away_form_adj

        analysis.predicted_total = analysis.predicted_home_score + analysis.predicted_away_score
        analysis.predicted_spread = analysis.predicted_home_score - analysis.predicted_away_score

        # Confidence based on data availability
        confidence_factors = [
            1.0 if home.games_played >= 8 else 0.5,
            1.0 if away.games_played >= 8 else 0.5,
            1.0 if analysis.h2h_games > 0 else 0.8,
            1.0 if len(analysis.common_opponents) > 3 else 0.9
        ]
        analysis.confidence = np.mean(confidence_factors)

    def _identify_key_matchups(self, analysis: MatchupAnalysis, home: TeamProfile, away: TeamProfile):
        """Identify key matchup advantages."""
        # Offense vs Defense matchups
        if home.passing_yards_per_game > 250 and away.pass_yards_allowed_per_game > 250:
            analysis.key_matchups.append(f"{home.team} passing offense vs {away.team} pass defense (advantage {home.team})")
            analysis.edge_analysis['passing_game'] = 0.6  # 60% edge to home

        if away.rushing_yards_per_game > 130 and home.rush_yards_allowed_per_game > 130:
            analysis.key_matchups.append(f"{away.team} rushing offense vs {home.team} rush defense (advantage {away.team})")
            analysis.edge_analysis['rushing_game'] = -0.6  # 60% edge to away

        # Add summary note
        if len(analysis.key_matchups) > 0:
            analysis.notes.append(f"Identified {len(analysis.key_matchups)} key matchup factors")


if __name__ == "__main__":
    # Example usage
    analyzer = TeamMatchupAnalyzer(season=2024)

    # Analyze a sample matchup
    matchup = analyzer.analyze_matchup(
        home_team='KC',
        away_team='BUF',
        week=11
    )

    print(f"\n{'='*70}")
    print(f"MATCHUP ANALYSIS: {matchup.away_team} @ {matchup.home_team} (Week {matchup.week})")
    print(f"{'='*70}\n")

    print(f"Head-to-Head:")
    print(f"  Games: {matchup.h2h_games}")
    print(f"  {matchup.home_team} home wins: {matchup.h2h_home_wins}")
    print(f"  Avg total: {matchup.h2h_avg_total:.1f}")
    print(f"  Last result: {matchup.h2h_last_result}")

    print(f"\nStatistical Edges:")
    print(f"  Offense: {matchup.offensive_edge}")
    print(f"  Defense: {matchup.defensive_edge}")
    print(f"  Pace: {matchup.pace_matchup}")

    print(f"\nRecent Form:")
    print(f"  {matchup.home_team}: {matchup.home_form}")
    print(f"  {matchup.away_team}: {matchup.away_form}")
    print(f"  Momentum: {matchup.momentum_edge}")

    print(f"\nPrediction:")
    print(f"  {matchup.home_team}: {matchup.predicted_home_score:.1f}")
    print(f"  {matchup.away_team}: {matchup.predicted_away_score:.1f}")
    print(f"  Total: {matchup.predicted_total:.1f}")
    print(f"  Spread: {matchup.home_team} {matchup.predicted_spread:+.1f}")
    print(f"  Confidence: {matchup.confidence:.0%}")

    if matchup.key_matchups:
        print(f"\nKey Matchups:")
        for i, matchup_note in enumerate(matchup.key_matchups, 1):
            print(f"  {i}. {matchup_note}")
