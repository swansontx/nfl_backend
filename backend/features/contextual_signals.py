"""
Contextual Signals Extraction

Extracts game context signals from available data sources:
- Weather (wind, temperature, precipitation)
- Rest & Travel (days rest, time zones)
- Injuries (key player absences)
- Officials (referee tendencies)
- Situational (divisional games, primetime, etc.)

Uses data-driven approach: Extract signals, measure impact, optimize weights.
"""

from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


@dataclass
class GameContext:
    """Contextual signals for a game."""
    game_id: str

    # Weather signals
    temperature: Optional[float] = None  # Fahrenheit
    wind_speed: Optional[float] = None   # mph
    roof_type: Optional[str] = None      # 'dome', 'outdoors', 'retractable'

    # Rest & Travel signals
    home_rest_days: int = 7
    away_rest_days: int = 7
    rest_differential: int = 0  # home_rest - away_rest

    # Game situation signals
    is_divisional: bool = False
    is_primetime: bool = False  # Thursday/Sunday/Monday night
    week: int = 1

    # Injury signals (to be populated)
    home_injury_impact: float = 0.0  # Calculated injury severity score
    away_injury_impact: float = 0.0

    # Official signals (to be populated)
    referee_penalty_rate: Optional[float] = None  # Penalties per game


class ContextualSignalsExtractor:
    """
    Extracts contextual signals from available data sources.

    Data-driven approach: Extract everything, then measure what matters.
    """

    def __init__(self, season: int = 2025, inputs_dir: str = "inputs"):
        self.season = season
        self.inputs_dir = Path(inputs_dir)

        # Load base data
        self.schedules = self._load_schedules()
        self.injuries = self._load_injuries()
        self.officials = self._load_officials()

    def _load_schedules(self) -> pd.DataFrame:
        """Load schedule data with weather and rest info."""
        schedule_file = self.inputs_dir / "schedules_2024_2025.csv"

        if not schedule_file.exists():
            print(f"Warning: {schedule_file} not found")
            return pd.DataFrame()

        df = pd.read_csv(schedule_file)

        # Filter to this season
        df = df[df['season'] == self.season].copy()

        return df

    def _load_injuries(self) -> pd.DataFrame:
        """Load injury data."""
        injury_file = self.inputs_dir / "injuries_2024_2025.csv"

        if not injury_file.exists():
            print(f"Warning: {injury_file} not found")
            return pd.DataFrame()

        df = pd.read_csv(injury_file)

        # Filter to this season
        df = df[df['season'] == self.season].copy()

        return df

    def _load_officials(self) -> pd.DataFrame:
        """Load officials data."""
        officials_file = self.inputs_dir / "officials_2024_2025.csv"

        if not officials_file.exists():
            print(f"Warning: {officials_file} not found")
            return pd.DataFrame()

        df = pd.read_csv(officials_file)

        # Filter to this season
        df = df[df['season'] == self.season].copy()

        return df

    def extract_game_context(self, game_id: str) -> GameContext:
        """
        Extract all contextual signals for a game.

        Args:
            game_id: Game identifier (e.g., "2025_05_KC_BUF")

        Returns:
            GameContext with all available signals
        """
        if self.schedules.empty:
            return GameContext(game_id=game_id)

        # Find game in schedule
        game = self.schedules[self.schedules['game_id'] == game_id]

        if game.empty:
            return GameContext(game_id=game_id)

        game = game.iloc[0]

        # Extract weather signals
        temperature = game.get('temp')
        if pd.notna(temperature):
            temperature = float(temperature)
        else:
            temperature = None

        wind_speed = game.get('wind')
        if pd.notna(wind_speed):
            wind_speed = float(wind_speed)
        else:
            wind_speed = None

        roof_type = game.get('roof')
        if pd.notna(roof_type):
            roof_type = str(roof_type).lower()
        else:
            roof_type = None

        # Extract rest signals
        home_rest = game.get('home_rest', 7)
        away_rest = game.get('away_rest', 7)

        if pd.isna(home_rest):
            home_rest = 7
        if pd.isna(away_rest):
            away_rest = 7

        home_rest = int(home_rest)
        away_rest = int(away_rest)
        rest_diff = home_rest - away_rest

        # Extract situational signals
        is_divisional = bool(game.get('div_game', False))
        week = int(game.get('week', 1))

        # Detect primetime games (Thursday/Sunday/Monday night)
        gametime = str(game.get('gametime', ''))
        weekday = str(game.get('weekday', ''))

        is_primetime = (
            weekday == 'Thursday' or
            weekday == 'Monday' or
            ('20:' in gametime or '21:' in gametime)  # Night games
        )

        # Extract injury signals (to be implemented)
        home_team = game.get('home_team')
        away_team = game.get('away_team')

        home_injury_impact = self._calculate_injury_impact(home_team, week)
        away_injury_impact = self._calculate_injury_impact(away_team, week)

        # Extract referee signals (to be implemented)
        referee_penalty_rate = self._get_referee_penalty_rate(game_id)

        return GameContext(
            game_id=game_id,
            temperature=temperature,
            wind_speed=wind_speed,
            roof_type=roof_type,
            home_rest_days=home_rest,
            away_rest_days=away_rest,
            rest_differential=rest_diff,
            is_divisional=is_divisional,
            is_primetime=is_primetime,
            week=week,
            home_injury_impact=home_injury_impact,
            away_injury_impact=away_injury_impact,
            referee_penalty_rate=referee_penalty_rate
        )

    def _calculate_injury_impact(self, team: str, week: int) -> float:
        """
        Calculate injury impact score for a team in a given week.

        Position weights (to be optimized via backtest):
        - QB: Most important
        - RB1, WR1: High importance
        - OL: Medium importance (cumulative)
        - Defense: Lower importance (unless star)

        Returns:
            Injury impact score (higher = more impacted by injuries)
        """
        if self.injuries.empty or pd.isna(team):
            return 0.0

        # Get injuries for this team and week
        team_injuries = self.injuries[
            (self.injuries['team'] == team) &
            (self.injuries['week'] == week)
        ]

        if team_injuries.empty:
            return 0.0

        # Position importance weights (starting values, to be optimized)
        position_weights = {
            'QB': 10.0,   # Quarterback most critical
            'RB': 3.0,    # Running back
            'WR': 2.5,    # Wide receiver
            'TE': 2.0,    # Tight end
            'T': 1.5,     # Tackle
            'G': 1.0,     # Guard
            'C': 1.5,     # Center
            'DE': 1.5,    # Defensive end
            'DT': 1.0,    # Defensive tackle
            'LB': 1.5,    # Linebacker
            'CB': 2.0,    # Cornerback
            'S': 1.5,     # Safety
            'K': 0.5,     # Kicker
            'P': 0.3      # Punter
        }

        # Status severity (to be optimized)
        status_multipliers = {
            'Out': 1.0,           # Definitely not playing
            'Doubtful': 0.8,      # Unlikely to play
            'Questionable': 0.4,  # 50/50
            'Probable': 0.2,      # Likely to play
        }

        total_impact = 0.0

        for _, injury in team_injuries.iterrows():
            position = injury.get('position', '')
            status = injury.get('report_status', '')

            # Get base weight for position
            base_weight = position_weights.get(position, 0.5)

            # Apply status multiplier
            status_mult = status_multipliers.get(status, 0.5)

            impact = base_weight * status_mult
            total_impact += impact

        return total_impact

    def _get_referee_penalty_rate(self, game_id: str) -> Optional[float]:
        """
        Get referee's historical penalty rate.

        Returns average penalties per game for the referee crew.
        (To be implemented with historical official data)
        """
        if self.officials.empty:
            return None

        # Find officials for this game
        game_officials = self.officials[self.officials['game_id'] == game_id]

        if game_officials.empty:
            return None

        # For now, return None - would need historical penalty data
        # to calculate actual penalty rates per referee
        return None


class SignalImpactAnalyzer:
    """
    Analyzes the historical impact of contextual signals.

    Measures correlation between signals and:
    - Actual spreads vs expected
    - Actual totals vs expected
    - Prediction errors

    Used to determine which signals matter and how much to weight them.
    """

    def __init__(self, games_df: pd.DataFrame, extractor: ContextualSignalsExtractor):
        """
        Initialize analyzer.

        Args:
            games_df: Historical games with results
            extractor: Signal extractor instance
        """
        self.games_df = games_df
        self.extractor = extractor

    def measure_signal_correlations(self) -> pd.DataFrame:
        """
        Measure correlation of each signal with game outcomes.

        Returns:
            DataFrame with signal correlations
        """
        results = []

        for _, game in self.games_df.iterrows():
            game_id = game['game_id']

            # Extract signals
            context = self.extractor.extract_game_context(game_id)

            # Get actual outcomes
            actual_total = game.get('home_score', 0) + game.get('away_score', 0)
            actual_spread = game.get('home_score', 0) - game.get('away_score', 0)

            # Get betting lines
            line_total = game.get('total_line')
            line_spread = game.get('spread_line')

            # Calculate over/under outcome (actual vs line)
            total_diff = actual_total - line_total if pd.notna(line_total) else None
            spread_diff = actual_spread - line_spread if pd.notna(line_spread) else None

            results.append({
                'game_id': game_id,
                'temperature': context.temperature,
                'wind_speed': context.wind_speed,
                'is_outdoor': context.roof_type == 'outdoors' if context.roof_type else None,
                'rest_differential': context.rest_differential,
                'is_divisional': context.is_divisional,
                'is_primetime': context.is_primetime,
                'home_injury_impact': context.home_injury_impact,
                'away_injury_impact': context.away_injury_impact,
                'total_injury_impact': context.home_injury_impact + context.away_injury_impact,
                'actual_total': actual_total,
                'actual_spread': actual_spread,
                'line_total': line_total,
                'line_spread': line_spread,
                'total_diff': total_diff,  # Positive = went over
                'spread_diff': spread_diff
            })

        df = pd.DataFrame(results)

        # Calculate correlations
        print("\n" + "="*70)
        print("SIGNAL IMPACT ANALYSIS")
        print("="*70)

        # Weather signals
        if df['temperature'].notna().sum() > 10:
            temp_corr = df[['temperature', 'total_diff']].corr().iloc[0, 1]
            print(f"\nTemperature → Total Diff: {temp_corr:.3f}")

            # Break down by temperature ranges
            cold_games = df[df['temperature'] < 32]['total_diff']
            mild_games = df[(df['temperature'] >= 32) & (df['temperature'] < 70)]['total_diff']
            hot_games = df[df['temperature'] >= 70]['total_diff']

            if len(cold_games) > 5:
                print(f"  Cold (<32°F): Avg diff = {cold_games.mean():.2f} pts (n={len(cold_games)})")
            if len(mild_games) > 5:
                print(f"  Mild (32-70°F): Avg diff = {mild_games.mean():.2f} pts (n={len(mild_games)})")
            if len(hot_games) > 5:
                print(f"  Hot (>70°F): Avg diff = {hot_games.mean():.2f} pts (n={len(hot_games)})")

        if df['wind_speed'].notna().sum() > 10:
            wind_corr = df[['wind_speed', 'total_diff']].corr().iloc[0, 1]
            print(f"\nWind Speed → Total Diff: {wind_corr:.3f}")

            # Break down by wind ranges
            calm = df[df['wind_speed'] < 10]['total_diff']
            moderate = df[(df['wind_speed'] >= 10) & (df['wind_speed'] < 15)]['total_diff']
            windy = df[df['wind_speed'] >= 15]['total_diff']

            if len(calm) > 5:
                print(f"  Calm (<10mph): Avg diff = {calm.mean():.2f} pts (n={len(calm)})")
            if len(moderate) > 5:
                print(f"  Moderate (10-15mph): Avg diff = {moderate.mean():.2f} pts (n={len(moderate)})")
            if len(windy) > 5:
                print(f"  Windy (>15mph): Avg diff = {windy.mean():.2f} pts (n={len(windy)})")

        # Rest signals
        rest_spread_corr = df[['rest_differential', 'spread_diff']].corr().iloc[0, 1]
        print(f"\nRest Differential → Spread Diff: {rest_spread_corr:.3f}")

        short_rest_home = df[df['rest_differential'] < -3]['spread_diff']
        normal_rest = df[(df['rest_differential'] >= -3) & (df['rest_differential'] <= 3)]['spread_diff']
        extra_rest_home = df[df['rest_differential'] > 3]['spread_diff']

        if len(short_rest_home) > 5:
            print(f"  Home on short rest: Avg diff = {short_rest_home.mean():.2f} pts (n={len(short_rest_home)})")
        if len(normal_rest) > 5:
            print(f"  Normal rest: Avg diff = {normal_rest.mean():.2f} pts (n={len(normal_rest)})")
        if len(extra_rest_home) > 5:
            print(f"  Home extra rest: Avg diff = {extra_rest_home.mean():.2f} pts (n={len(extra_rest_home)})")

        # Divisional games
        div_total = df[df['is_divisional']]['total_diff']
        non_div_total = df[~df['is_divisional']]['total_diff']

        if len(div_total) > 5 and len(non_div_total) > 5:
            print(f"\nDivisional Games → Total Diff:")
            print(f"  Divisional: Avg diff = {div_total.mean():.2f} pts (n={len(div_total)})")
            print(f"  Non-divisional: Avg diff = {non_div_total.mean():.2f} pts (n={len(non_div_total)})")
            print(f"  Difference: {div_total.mean() - non_div_total.mean():.2f} pts")

        # Primetime games
        prime_total = df[df['is_primetime']]['total_diff']
        regular_total = df[~df['is_primetime']]['total_diff']

        if len(prime_total) > 5 and len(regular_total) > 5:
            print(f"\nPrimetime Games → Total Diff:")
            print(f"  Primetime: Avg diff = {prime_total.mean():.2f} pts (n={len(prime_total)})")
            print(f"  Regular: Avg diff = {regular_total.mean():.2f} pts (n={len(regular_total)})")
            print(f"  Difference: {prime_total.mean() - regular_total.mean():.2f} pts")

        # Injury impact
        if df['total_injury_impact'].max() > 0:
            high_injury = df[df['total_injury_impact'] > 5]['spread_diff']
            low_injury = df[df['total_injury_impact'] <= 5]['spread_diff']

            if len(high_injury) > 5 and len(low_injury) > 5:
                print(f"\nInjury Impact → Spread Diff:")
                print(f"  High injury impact (>5): Avg diff = {high_injury.mean():.2f} pts (n={len(high_injury)})")
                print(f"  Low injury impact (≤5): Avg diff = {low_injury.mean():.2f} pts (n={len(low_injury)})")

        print("\n" + "="*70)

        return df


def analyze_signals_from_backtest(season: int = 2025, inputs_dir: str = "inputs"):
    """
    Convenience function to analyze signal impact from historical data.

    Usage:
        from backend.features.contextual_signals import analyze_signals_from_backtest
        analyze_signals_from_backtest(season=2025)
    """
    # Load schedule with results
    schedule_file = f"{inputs_dir}/schedules_2024_2025.csv"
    games_df = pd.read_csv(schedule_file)
    games_df = games_df[
        (games_df['season'] == season) &
        (games_df['game_type'] == 'REG') &
        (games_df['away_score'].notna())
    ]

    # Create extractor
    extractor = ContextualSignalsExtractor(season=season, inputs_dir=inputs_dir)

    # Analyze
    analyzer = SignalImpactAnalyzer(games_df, extractor)
    results_df = analyzer.measure_signal_correlations()

    return results_df
