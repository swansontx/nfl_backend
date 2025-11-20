"""Defense Performance Analyzer - answers questions about team defense performance.

This module provides quick intelligent responses to questions like:
- "How has X team defense done against the run recently?"
- "How does X team pass defense look?"
- "How did specific RBs perform against this defense?"

Returns structured data with:
- Per-game performance vs individual players
- Comparison to player averages (+/- difference)
- Trend analysis (improving/declining)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from backend.database.local_db import get_db

logger = logging.getLogger(__name__)


@dataclass
class PlayerVsDefense:
    """Individual player performance against a defense."""
    player_name: str
    team: str
    game_id: str
    week: int

    # Actual performance
    yards: int = 0
    attempts: int = 0
    touchdowns: int = 0
    ypc: float = 0.0

    # Player's season average
    season_avg_yards: float = 0.0
    season_avg_ypc: float = 0.0

    # Differential
    yards_diff: int = 0  # +5 means 5 more than usual
    ypc_diff: float = 0.0

    # Result
    held_under: bool = False
    breakout: bool = False


@dataclass
class DefensePerformance:
    """Team defense performance analysis."""
    team: str
    defense_type: str  # 'rush' or 'pass'
    season: int
    weeks_analyzed: int

    # Overall stats
    total_yards_allowed: int = 0
    avg_yards_allowed: float = 0.0
    total_tds_allowed: int = 0

    # Rankings
    yards_rank: int = 16
    epa_rank: int = 16

    # Trend
    recent_avg: float = 0.0  # Last 3 games
    season_avg: float = 0.0
    trending: str = "neutral"  # improving, declining, neutral

    # Individual matchups
    player_matchups: List[PlayerVsDefense] = field(default_factory=list)

    # Summary
    held_under_pct: float = 0.0  # % of players held under average
    insight: str = ""


class DefenseAnalyzer:
    """Analyzes team defense performance against specific positions."""

    def __init__(self):
        """Initialize analyzer."""
        pass

    def get_rush_defense_performance(self, team: str, season: int = 2024,
                                     last_n_games: int = 5) -> DefensePerformance:
        """Get rush defense performance with individual RB matchups.

        Args:
            team: Team abbreviation (e.g., 'BUF')
            season: NFL season year
            last_n_games: Number of recent games to analyze

        Returns:
            DefensePerformance with RB matchups and comparisons
        """
        result = DefensePerformance(
            team=team.upper(),
            defense_type='rush',
            season=season,
            weeks_analyzed=last_n_games
        )

        try:
            with get_db() as conn:
                cursor = conn.cursor()

                # Get rush plays against this defense
                cursor.execute("""
                    SELECT
                        p.game_id,
                        p.week,
                        p.rusher_player_name,
                        p.posteam,
                        SUM(p.yards_gained) as yards,
                        COUNT(*) as attempts,
                        SUM(p.touchdown) as tds,
                        AVG(p.epa) as epa
                    FROM play_by_play p
                    WHERE p.defteam = ?
                        AND p.season = ?
                        AND p.play_type = 'run'
                        AND p.rusher_player_name IS NOT NULL
                        AND p.rusher_player_name != ''
                    GROUP BY p.game_id, p.week, p.rusher_player_name, p.posteam
                    ORDER BY p.week DESC
                """, (team.upper(), season))

                matchups = [dict(row) for row in cursor.fetchall()]

                if not matchups:
                    result.insight = f"No rush defense data found for {team.upper()}"
                    return result

                # Get unique games
                games = list(set(m['game_id'] for m in matchups))
                result.weeks_analyzed = min(len(games), last_n_games)

                # Filter to last N games
                recent_weeks = sorted(set(m['week'] for m in matchups), reverse=True)[:last_n_games]
                matchups = [m for m in matchups if m['week'] in recent_weeks]

                # Calculate overall defense stats
                total_yards = sum(m['yards'] for m in matchups)
                total_attempts = sum(m['attempts'] for m in matchups)
                total_tds = sum(m['tds'] or 0 for m in matchups)

                result.total_yards_allowed = total_yards
                result.avg_yards_allowed = total_yards / result.weeks_analyzed if result.weeks_analyzed > 0 else 0
                result.total_tds_allowed = total_tds

                # Analyze individual player matchups
                player_matchups = []
                held_under_count = 0

                for m in matchups:
                    player = m['rusher_player_name']
                    if not player or m['attempts'] < 5:  # Skip players with < 5 carries
                        continue

                    # Get player's season average
                    cursor.execute("""
                        SELECT
                            AVG(total_yards) as avg_yards,
                            AVG(total_yards * 1.0 / NULLIF(attempts, 0)) as avg_ypc
                        FROM (
                            SELECT
                                game_id,
                                SUM(yards_gained) as total_yards,
                                COUNT(*) as attempts
                            FROM play_by_play
                            WHERE rusher_player_name = ?
                                AND season = ?
                                AND play_type = 'run'
                            GROUP BY game_id
                        )
                    """, (player, season))

                    avg_row = cursor.fetchone()
                    season_avg = dict(avg_row) if avg_row else {'avg_yards': 0, 'avg_ypc': 0}

                    ypc = m['yards'] / m['attempts'] if m['attempts'] > 0 else 0
                    season_avg_yards = season_avg.get('avg_yards') or 0
                    season_avg_ypc = season_avg.get('avg_ypc') or 0

                    yards_diff = m['yards'] - season_avg_yards
                    ypc_diff = ypc - season_avg_ypc

                    held_under = yards_diff < -5  # 5+ yards below average
                    breakout = yards_diff > 15  # 15+ yards above average

                    if held_under:
                        held_under_count += 1

                    pm = PlayerVsDefense(
                        player_name=player,
                        team=m['posteam'],
                        game_id=m['game_id'],
                        week=m['week'],
                        yards=m['yards'],
                        attempts=m['attempts'],
                        touchdowns=m['tds'] or 0,
                        ypc=round(ypc, 1),
                        season_avg_yards=round(season_avg_yards, 1),
                        season_avg_ypc=round(season_avg_ypc, 1),
                        yards_diff=round(yards_diff),
                        ypc_diff=round(ypc_diff, 1),
                        held_under=held_under,
                        breakout=breakout
                    )
                    player_matchups.append(pm)

                # Sort by week (most recent first) then by yards
                player_matchups.sort(key=lambda x: (-x.week, -x.yards))
                result.player_matchups = player_matchups[:15]  # Top 15 matchups

                # Calculate held under percentage
                if player_matchups:
                    result.held_under_pct = round(100 * held_under_count / len(player_matchups), 1)

                # Calculate trend
                if len(recent_weeks) >= 3:
                    last_3 = [m for m in matchups if m['week'] in recent_weeks[:3]]
                    earlier = [m for m in matchups if m['week'] in recent_weeks[3:]]

                    if last_3:
                        result.recent_avg = sum(m['yards'] for m in last_3) / 3
                    if earlier:
                        result.season_avg = sum(m['yards'] for m in earlier) / len(set(e['week'] for e in earlier))

                    if result.recent_avg < result.season_avg * 0.9:
                        result.trending = "improving"
                    elif result.recent_avg > result.season_avg * 1.1:
                        result.trending = "declining"
                    else:
                        result.trending = "neutral"

                # Build insight
                if result.held_under_pct >= 60:
                    result.insight = f"{team.upper()} run defense is STRONG - held {result.held_under_pct}% of RBs below average"
                elif result.held_under_pct <= 40:
                    result.insight = f"{team.upper()} run defense is WEAK - only held {result.held_under_pct}% of RBs below average"
                else:
                    result.insight = f"{team.upper()} run defense is AVERAGE - held {result.held_under_pct}% of RBs below average"

                if result.trending == "improving":
                    result.insight += " (trending better recently)"
                elif result.trending == "declining":
                    result.insight += " (trending worse recently)"

        except Exception as e:
            logger.error(f"Error analyzing rush defense: {e}")
            result.insight = f"Error analyzing defense: {str(e)}"

        return result

    def get_pass_defense_performance(self, team: str, season: int = 2024,
                                     last_n_games: int = 5) -> DefensePerformance:
        """Get pass defense performance with individual QB/WR matchups.

        Args:
            team: Team abbreviation
            season: NFL season year
            last_n_games: Number of recent games to analyze

        Returns:
            DefensePerformance with passing matchups
        """
        result = DefensePerformance(
            team=team.upper(),
            defense_type='pass',
            season=season,
            weeks_analyzed=last_n_games
        )

        try:
            with get_db() as conn:
                cursor = conn.cursor()

                # Get QB performance against this defense
                cursor.execute("""
                    SELECT
                        p.game_id,
                        p.week,
                        p.passer_player_name,
                        p.posteam,
                        SUM(p.yards_gained) as yards,
                        COUNT(*) as attempts,
                        SUM(CASE WHEN p.touchdown = 1 THEN 1 ELSE 0 END) as tds,
                        SUM(CASE WHEN p.interception = 1 THEN 1 ELSE 0 END) as ints,
                        AVG(p.epa) as epa,
                        AVG(p.cpoe) as cpoe
                    FROM play_by_play p
                    WHERE p.defteam = ?
                        AND p.season = ?
                        AND p.play_type = 'pass'
                        AND p.passer_player_name IS NOT NULL
                        AND p.passer_player_name != ''
                    GROUP BY p.game_id, p.week, p.passer_player_name, p.posteam
                    ORDER BY p.week DESC
                """, (team.upper(), season))

                matchups = [dict(row) for row in cursor.fetchall()]

                if not matchups:
                    result.insight = f"No pass defense data found for {team.upper()}"
                    return result

                # Get unique games
                games = list(set(m['game_id'] for m in matchups))
                result.weeks_analyzed = min(len(games), last_n_games)

                # Filter to last N games
                recent_weeks = sorted(set(m['week'] for m in matchups), reverse=True)[:last_n_games]
                matchups = [m for m in matchups if m['week'] in recent_weeks]

                # Calculate overall defense stats
                total_yards = sum(m['yards'] for m in matchups)
                total_attempts = sum(m['attempts'] for m in matchups)
                total_tds = sum(m['tds'] or 0 for m in matchups)

                result.total_yards_allowed = total_yards
                result.avg_yards_allowed = total_yards / result.weeks_analyzed if result.weeks_analyzed > 0 else 0
                result.total_tds_allowed = total_tds

                # Analyze individual QB matchups
                player_matchups = []
                held_under_count = 0

                for m in matchups:
                    player = m['passer_player_name']
                    if not player or m['attempts'] < 10:  # Skip QBs with < 10 attempts
                        continue

                    # Get QB's season average
                    cursor.execute("""
                        SELECT
                            AVG(total_yards) as avg_yards,
                            AVG(total_yards * 1.0 / NULLIF(attempts, 0)) as avg_ypa
                        FROM (
                            SELECT
                                game_id,
                                SUM(yards_gained) as total_yards,
                                COUNT(*) as attempts
                            FROM play_by_play
                            WHERE passer_player_name = ?
                                AND season = ?
                                AND play_type = 'pass'
                            GROUP BY game_id
                        )
                    """, (player, season))

                    avg_row = cursor.fetchone()
                    season_avg = dict(avg_row) if avg_row else {'avg_yards': 0, 'avg_ypa': 0}

                    ypa = m['yards'] / m['attempts'] if m['attempts'] > 0 else 0
                    season_avg_yards = season_avg.get('avg_yards') or 0
                    season_avg_ypa = season_avg.get('avg_ypa') or 0

                    yards_diff = m['yards'] - season_avg_yards
                    ypa_diff = ypa - season_avg_ypa

                    held_under = yards_diff < -20  # 20+ yards below average
                    breakout = yards_diff > 50  # 50+ yards above average

                    if held_under:
                        held_under_count += 1

                    pm = PlayerVsDefense(
                        player_name=player,
                        team=m['posteam'],
                        game_id=m['game_id'],
                        week=m['week'],
                        yards=m['yards'],
                        attempts=m['attempts'],
                        touchdowns=m['tds'] or 0,
                        ypc=round(ypa, 1),  # Using ypc field for ypa
                        season_avg_yards=round(season_avg_yards, 1),
                        season_avg_ypc=round(season_avg_ypa, 1),  # Using for ypa
                        yards_diff=round(yards_diff),
                        ypc_diff=round(ypa_diff, 1),
                        held_under=held_under,
                        breakout=breakout
                    )
                    player_matchups.append(pm)

                # Sort by week then yards
                player_matchups.sort(key=lambda x: (-x.week, -x.yards))
                result.player_matchups = player_matchups[:10]

                # Calculate held under percentage
                if player_matchups:
                    result.held_under_pct = round(100 * held_under_count / len(player_matchups), 1)

                # Build insight
                if result.held_under_pct >= 60:
                    result.insight = f"{team.upper()} pass defense is STRONG - held {result.held_under_pct}% of QBs below average"
                elif result.held_under_pct <= 40:
                    result.insight = f"{team.upper()} pass defense is WEAK - only held {result.held_under_pct}% of QBs below average"
                else:
                    result.insight = f"{team.upper()} pass defense is AVERAGE - held {result.held_under_pct}% of QBs below average"

        except Exception as e:
            logger.error(f"Error analyzing pass defense: {e}")
            result.insight = f"Error analyzing defense: {str(e)}"

        return result

    def get_defense_summary(self, team: str, season: int = 2024) -> Dict:
        """Get complete defense summary with both rush and pass analysis.

        Args:
            team: Team abbreviation
            season: NFL season year

        Returns:
            Dict with complete defense analysis
        """
        rush = self.get_rush_defense_performance(team, season)
        pass_def = self.get_pass_defense_performance(team, season)

        return {
            'team': team.upper(),
            'season': season,
            'rush_defense': {
                'insight': rush.insight,
                'yards_allowed_per_game': round(rush.avg_yards_allowed, 1),
                'tds_allowed': rush.total_tds_allowed,
                'held_under_pct': rush.held_under_pct,
                'trending': rush.trending,
                'matchups': [
                    {
                        'player': m.player_name,
                        'team': m.team,
                        'week': m.week,
                        'yards': m.yards,
                        'attempts': m.attempts,
                        'ypc': m.ypc,
                        'vs_avg': f"{'+' if m.yards_diff >= 0 else ''}{m.yards_diff}",
                        'held_under': m.held_under
                    }
                    for m in rush.player_matchups[:10]
                ]
            },
            'pass_defense': {
                'insight': pass_def.insight,
                'yards_allowed_per_game': round(pass_def.avg_yards_allowed, 1),
                'tds_allowed': pass_def.total_tds_allowed,
                'held_under_pct': pass_def.held_under_pct,
                'matchups': [
                    {
                        'player': m.player_name,
                        'team': m.team,
                        'week': m.week,
                        'yards': m.yards,
                        'attempts': m.attempts,
                        'ypa': m.ypc,
                        'vs_avg': f"{'+' if m.yards_diff >= 0 else ''}{m.yards_diff}",
                        'held_under': m.held_under
                    }
                    for m in pass_def.player_matchups[:10]
                ]
            }
        }


# Singleton instance
defense_analyzer = DefenseAnalyzer()
