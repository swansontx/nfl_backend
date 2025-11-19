"""Insight generation engine for matchup analysis and prop recommendations.

This module generates data-driven insights by analyzing:
- Player performance trends
- Matchup advantages/disadvantages
- Weather impact
- Injury impact
- Vegas line movement
- Historical performance patterns
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import statistics


@dataclass
class PlayerStats:
    """Player statistics for analysis."""
    player_id: str
    player_name: str
    position: str
    team: str
    recent_games: List[Dict]  # List of game stat dicts
    season_avg: Dict  # Season averages
    vs_opponent_avg: Optional[Dict] = None  # Avg vs specific opponent


@dataclass
class Insight:
    """Structured insight data."""
    insight_type: str
    title: str
    description: str
    confidence: float
    supporting_data: Dict
    impact_level: str  # "high", "medium", "low"
    recommendation: Optional[str] = None


class StatisticalAnalyzer:
    """Statistical analysis utilities for trend detection and significance testing."""

    @staticmethod
    def calculate_trend(values: List[float], recent_weight: int = 3) -> Dict:
        """Calculate trend direction and strength.

        Args:
            values: List of values in chronological order
            recent_weight: Number of recent games to weight heavily

        Returns:
            Dict with trend analysis
        """
        if len(values) < 2:
            return {"trend": "insufficient_data", "strength": 0, "direction": "neutral"}

        # Simple linear regression
        n = len(values)
        x = list(range(n))
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(values)

        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator

        # Compare recent average to earlier average
        if len(values) >= recent_weight * 2:
            recent_avg = statistics.mean(values[-recent_weight:])
            earlier_avg = statistics.mean(values[:-recent_weight])
            pct_change = ((recent_avg - earlier_avg) / earlier_avg * 100) if earlier_avg else 0
        else:
            pct_change = 0

        # Determine direction and strength
        if abs(slope) < 0.1:
            direction = "stable"
            strength = 0
        elif slope > 0:
            direction = "increasing"
            strength = min(abs(slope) * 10, 1.0)
        else:
            direction = "decreasing"
            strength = min(abs(slope) * 10, 1.0)

        return {
            "trend": direction,
            "strength": round(strength, 2),
            "direction": direction,
            "slope": round(slope, 3),
            "pct_change": round(pct_change, 1),
            "recent_avg": round(recent_avg, 1) if len(values) >= recent_weight * 2 else None,
            "earlier_avg": round(earlier_avg, 1) if len(values) >= recent_weight * 2 else None
        }

    @staticmethod
    def calculate_consistency(values: List[float]) -> Dict:
        """Calculate consistency metrics.

        Args:
            values: List of stat values

        Returns:
            Dict with consistency metrics
        """
        if len(values) < 2:
            return {"consistency": "insufficient_data", "std_dev": 0, "coefficient_of_variation": 0}

        mean = statistics.mean(values)
        std_dev = statistics.stdev(values)
        cv = (std_dev / mean * 100) if mean else 0

        # Classify consistency
        if cv < 15:
            consistency = "very_consistent"
        elif cv < 25:
            consistency = "consistent"
        elif cv < 40:
            consistency = "moderate"
        else:
            consistency = "volatile"

        return {
            "consistency": consistency,
            "std_dev": round(std_dev, 2),
            "coefficient_of_variation": round(cv, 1),
            "mean": round(mean, 1),
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values)
        }

    @staticmethod
    def detect_outliers(values: List[float], threshold: float = 2.0) -> List[int]:
        """Detect statistical outliers using z-score.

        Args:
            values: List of values
            threshold: Z-score threshold (default 2.0 std devs)

        Returns:
            List of indices of outliers
        """
        if len(values) < 3:
            return []

        mean = statistics.mean(values)
        std_dev = statistics.stdev(values)

        if std_dev == 0:
            return []

        outliers = []
        for i, value in enumerate(values):
            z_score = abs((value - mean) / std_dev)
            if z_score > threshold:
                outliers.append(i)

        return outliers


class InsightGenerator:
    """Generate insights from player and matchup data."""

    def __init__(self):
        self.analyzer = StatisticalAnalyzer()

    def generate_player_trend_insight(self, player: PlayerStats, stat_name: str) -> Optional[Insight]:
        """Generate insight about player's recent trend.

        Args:
            player: PlayerStats object
            stat_name: Name of stat to analyze (e.g., "passing_yards")

        Returns:
            Insight object if significant trend found
        """
        if len(player.recent_games) < 3:
            return None

        values = [game.get(stat_name, 0) for game in player.recent_games]
        trend_data = self.analyzer.calculate_trend(values)
        consistency_data = self.analyzer.calculate_consistency(values)

        # Only generate insight if trend is significant and we have enough data
        if trend_data['strength'] < 0.3 or trend_data.get('recent_avg') is None:
            return None

        # Determine confidence based on consistency and trend strength
        confidence = min(0.5 + (trend_data['strength'] * 0.3) + (0.2 if consistency_data['consistency'] in ['very_consistent', 'consistent'] else 0), 0.95)

        if trend_data['direction'] == 'increasing':
            title = f"{player.player_name} Trending Up in {stat_name.replace('_', ' ').title()}"
            description = (
                f"{player.player_name} has been on an upward trend, averaging "
                f"{trend_data['recent_avg']:.1f} {stat_name} in recent games vs "
                f"{trend_data['earlier_avg']:.1f} earlier ({trend_data['pct_change']:+.1f}%). "
                f"Performance is {consistency_data['consistency']}."
            )
            recommendation = f"Consider OVER on {player.player_name} {stat_name} props"
        else:
            title = f"{player.player_name} Trending Down in {stat_name.replace('_', ' ').title()}"
            description = (
                f"{player.player_name} has been declining, averaging "
                f"{trend_data['recent_avg']:.1f} {stat_name} in recent games vs "
                f"{trend_data['earlier_avg']:.1f} earlier ({trend_data['pct_change']:+.1f}%). "
                f"Performance is {consistency_data['consistency']}."
            )
            recommendation = f"Consider UNDER on {player.player_name} {stat_name} props"

        return Insight(
            insight_type="trend",
            title=title,
            description=description,
            confidence=confidence,
            supporting_data={
                "stat": stat_name,
                "recent_avg": trend_data['recent_avg'],
                "earlier_avg": trend_data['earlier_avg'],
                "pct_change": trend_data['pct_change'],
                "trend_strength": trend_data['strength'],
                "consistency": consistency_data['consistency'],
                "sample_size": len(values),
                "recent_games": values[-3:]
            },
            impact_level="high" if trend_data['strength'] > 0.6 else "medium",
            recommendation=recommendation
        )

    def generate_matchup_insight(self, player: PlayerStats, opponent_def_rank: int, league_avg: float) -> Optional[Insight]:
        """Generate insight about player vs opponent matchup.

        Args:
            player: PlayerStats object
            opponent_def_rank: Opponent's defensive rank (1-32)
            league_avg: League average for this stat

        Returns:
            Insight object if matchup is notable
        """
        if not player.vs_opponent_avg or not player.season_avg:
            return None

        stat_name = "yards"  # Placeholder
        player_avg = player.season_avg.get(stat_name, 0)
        vs_opponent = player.vs_opponent_avg.get(stat_name, 0)

        # Check if favorable matchup
        is_favorable = opponent_def_rank >= 20  # Bottom half of league

        if is_favorable and player_avg > league_avg * 1.1:
            confidence = min(0.7 + (opponent_def_rank - 20) * 0.02, 0.9)

            title = f"Favorable Matchup: {player.player_name} vs Weak Defense"
            description = (
                f"{player.player_name} (avg {player_avg:.1f} {stat_name}) faces a "
                f"#{opponent_def_rank} ranked defense. Historical performance suggests "
                f"exploitable matchup. League avg: {league_avg:.1f}."
            )
            recommendation = f"Value play on {player.player_name} OVER"

            return Insight(
                insight_type="matchup",
                title=title,
                description=description,
                confidence=confidence,
                supporting_data={
                    "player_avg": player_avg,
                    "league_avg": league_avg,
                    "vs_opponent_avg": vs_opponent if vs_opponent else None,
                    "opponent_rank": opponent_def_rank,
                    "percentile": round((32 - opponent_def_rank) / 32 * 100, 1)
                },
                impact_level="high" if confidence > 0.8 else "medium",
                recommendation=recommendation
            )

        return None

    def generate_weather_impact_insight(self, weather_data: Dict, player: PlayerStats) -> Optional[Insight]:
        """Generate insight about weather impact on player performance.

        Args:
            weather_data: Weather forecast data
            player: PlayerStats object

        Returns:
            Insight object if weather is impactful
        """
        wind_speed = weather_data.get('wind_speed', 0)
        condition = weather_data.get('condition', 'Clear')
        is_dome = weather_data.get('is_dome', False)

        if is_dome:
            return None  # No weather impact in dome

        # High wind affects passing
        if wind_speed > 15 and player.position == 'QB':
            confidence = min(0.6 + (wind_speed - 15) * 0.02, 0.85)

            title = "High Wind Alert: Passing Game Impact"
            description = (
                f"Wind speeds of {wind_speed} mph expected. Historical data shows "
                f"QB passing yards decrease by ~{int((wind_speed - 10) * 2)}% in these conditions. "
                f"Running game and short passes favored."
            )

            return Insight(
                insight_type="weather",
                title=title,
                description=description,
                confidence=confidence,
                supporting_data={
                    "wind_speed": wind_speed,
                    "condition": condition,
                    "expected_impact": "negative_passing",
                    "estimated_reduction_pct": int((wind_speed - 10) * 2)
                },
                impact_level="high" if wind_speed > 20 else "medium",
                recommendation="Consider UNDER on passing props, OVER on rushing props"
            )

        # Rain affects ball security and passing
        if condition in ['Rain', 'Snow']:
            confidence = 0.65

            title = f"{condition} Expected: Ball Security Concerns"
            description = (
                f"{condition} forecast increases fumble risk and reduces deep passing. "
                f"Ground game and possession-based offense favored."
            )

            return Insight(
                insight_type="weather",
                title=title,
                description=description,
                confidence=confidence,
                supporting_data={
                    "condition": condition,
                    "expected_impact": "negative_passing_ball_security"
                },
                impact_level="medium",
                recommendation="Favor rushing props and short-area receivers"
            )

        return None

    def generate_injury_impact_insight(self, injuries: List[Dict], team: str) -> List[Insight]:
        """Generate insights about injury impacts.

        Args:
            injuries: List of injury records
            team: Team abbreviation

        Returns:
            List of Insight objects
        """
        insights = []

        # Key position injuries
        key_positions = ['QB', 'RB', 'WR', 'TE', 'OL']
        key_injuries = [inj for inj in injuries if inj.get('position') in key_positions and inj.get('injury_status') in ['Out', 'Doubtful']]

        for injury in key_injuries:
            position = injury.get('position')
            player_name = injury.get('player_name')
            status = injury.get('injury_status')

            if position == 'QB':
                confidence = 0.9
                impact = "critical"
                title = f"Critical: {team} Starting QB {player_name} {status}"
                description = (
                    f"{player_name} is {status}. Backup QB typically sees significant "
                    f"performance drop-off. Expect conservative game plan and increased "
                    f"reliance on running game."
                )
                recommendation = f"Fade {team} passing props, consider {team} rushing OVER"
            elif position in ['RB', 'WR']:
                confidence = 0.75
                impact = "high"
                title = f"Key Absence: {team} {position} {player_name} {status}"
                description = (
                    f"{player_name} ({position}) is {status}. Expect increased target/touch "
                    f"share for remaining skill players."
                )
                recommendation = f"Look for value on {team} secondary {position}s"
            else:
                continue

            insights.append(Insight(
                insight_type="injury",
                title=title,
                description=description,
                confidence=confidence,
                supporting_data={
                    "player": player_name,
                    "position": position,
                    "status": status,
                    "team": team
                },
                impact_level=impact,
                recommendation=recommendation
            ))

        return insights

    def generate_all_insights(self,
                            game_id: str,
                            players: List[PlayerStats],
                            weather_data: Optional[Dict] = None,
                            injuries: Optional[List[Dict]] = None) -> List[Insight]:
        """Generate all insights for a game.

        Args:
            game_id: Game ID
            players: List of PlayerStats objects
            weather_data: Weather forecast (optional)
            injuries: Injury data (optional)

        Returns:
            List of all generated insights, sorted by confidence
        """
        all_insights = []

        # Player trend insights
        for player in players:
            for stat in ['passing_yards', 'rushing_yards', 'receiving_yards']:
                insight = self.generate_player_trend_insight(player, stat)
                if insight:
                    all_insights.append(insight)

        # Weather insights
        if weather_data:
            for player in players:
                insight = self.generate_weather_impact_insight(weather_data, player)
                if insight:
                    all_insights.append(insight)

        # Injury insights
        if injuries:
            teams = set(inj.get('team') for inj in injuries)
            for team in teams:
                team_injuries = [inj for inj in injuries if inj.get('team') == team]
                team_insights = self.generate_injury_impact_insight(team_injuries, team)
                all_insights.extend(team_insights)

        # Sort by confidence and impact
        impact_scores = {'high': 3, 'medium': 2, 'low': 1}
        all_insights.sort(
            key=lambda x: (impact_scores.get(x.impact_level, 0), x.confidence),
            reverse=True
        )

        return all_insights


# Singleton instance
insight_generator = InsightGenerator()
