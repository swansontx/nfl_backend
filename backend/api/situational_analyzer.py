"""Situational Intelligence Analyzer - compounds all data to prevent LLM shortcuts.

This module pre-computes specific betting situations so the LLM can't skip analysis:
- Trending form (last 3 games vs season average)
- Weather impact analysis
- Rest/schedule advantages
- Team strength differentials
- Positional matchup grades

ADDITIONAL DATA TO ADD (for ever-growing database):
==================================================

1. COVERAGE ANALYSIS
   - Zone vs Man coverage rates by team
   - QB performance vs zone/man
   - WR separation by coverage type
   - Blitz rates and pressure rates

2. ADVANCED SPLITS
   - Home/away performance
   - Indoor/outdoor splits
   - Grass/turf performance
   - Day/night game splits

3. GAME SCRIPT
   - Performance when leading
   - Performance when trailing
   - Garbage time production
   - 2-minute drill efficiency

4. PERSONNEL DATA
   - Snap counts by player
   - Route participation %
   - Target share trends
   - Carry share trends

5. OFFENSIVE LINE
   - Pressure rate allowed
   - Time to throw
   - Run blocking grade

6. HISTORICAL MATCHUP
   - Head-to-head history
   - Revenge game flags
   - Divisional rivalry boost

7. VEGAS IMPLIED
   - Implied team totals
   - Implied pace
   - Line movement direction
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from backend.database.local_db import get_db

logger = logging.getLogger(__name__)


@dataclass
class TrendingForm:
    """Recent form analysis (last 3 games vs season)."""
    team: str

    # Passing trends
    recent_pass_yards_avg: float = 0.0
    season_pass_yards_avg: float = 0.0
    pass_trend: str = "neutral"  # up, down, neutral
    pass_trend_pct: float = 0.0

    # Rushing trends
    recent_rush_yards_avg: float = 0.0
    season_rush_yards_avg: float = 0.0
    rush_trend: str = "neutral"
    rush_trend_pct: float = 0.0

    # Scoring trends
    recent_points_avg: float = 0.0
    season_points_avg: float = 0.0
    scoring_trend: str = "neutral"
    scoring_trend_pct: float = 0.0

    # Defense trends
    recent_points_allowed_avg: float = 0.0
    season_points_allowed_avg: float = 0.0
    defense_trend: str = "neutral"

    # Overall form
    form_grade: str = "C"
    momentum: str = "neutral"  # hot, cold, neutral

    # Narrative
    form_narrative: str = ""


@dataclass
class WeatherImpact:
    """Weather conditions and impact on gameplay."""
    temperature: int = 70
    wind_speed: int = 0
    precipitation_chance: int = 0
    is_dome: bool = False
    roof_type: str = "open"

    # Impact flags
    high_wind: bool = False
    extreme_cold: bool = False
    extreme_heat: bool = False
    precipitation: bool = False

    # Impact assessments
    pass_impact: str = "neutral"
    kick_impact: str = "neutral"

    # Narrative
    weather_narrative: str = ""

    # Prop recommendations
    weather_props: List[str] = field(default_factory=list)


@dataclass
class ScheduleAdvantage:
    """Schedule and rest factors."""
    team: str

    days_rest: int = 7
    opponent_rest: int = 7
    rest_advantage: int = 0

    is_home: bool = False
    is_division: bool = False
    is_primetime: bool = False
    is_short_week: bool = False
    is_long_week: bool = False

    # Travel factors
    timezone_diff: int = 0
    is_cross_country: bool = False

    # Narrative
    schedule_narrative: str = ""


@dataclass
class PositionalEdge:
    """Positional matchup edge."""
    position: str
    team: str
    opponent: str

    off_rank: int = 16
    def_rank: int = 16
    rank_diff: int = 0

    grade: str = "C"
    edge_score: float = 0.0

    insight: str = ""
    target_props: List[str] = field(default_factory=list)


@dataclass
class GameSituation:
    """Complete situational analysis for a game."""
    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str

    # Form analysis
    home_form: TrendingForm = None
    away_form: TrendingForm = None

    # Weather
    weather: WeatherImpact = None

    # Schedule
    home_schedule: ScheduleAdvantage = None
    away_schedule: ScheduleAdvantage = None

    # Positional edges
    positional_edges: List[PositionalEdge] = field(default_factory=list)

    # Key situations
    key_situations: List[Dict] = field(default_factory=list)

    # Top targets
    prop_targets: List[Dict] = field(default_factory=list)


class SituationalAnalyzer:
    """Analyzes situational factors for betting edges."""

    def __init__(self):
        """Initialize analyzer."""
        self.team_rankings = self._load_team_rankings()

    def _load_team_rankings(self) -> Dict[str, Dict]:
        """Load team rankings from database."""
        rankings = {}
        try:
            with get_db() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT t1.*
                    FROM team_stats t1
                    INNER JOIN (
                        SELECT team, MAX(week) as max_week
                        FROM team_stats
                        WHERE season = 2024
                        GROUP BY team
                    ) t2 ON t1.team = t2.team AND t1.week = t2.max_week
                    WHERE t1.season = 2024
                """)

                teams = [dict(row) for row in cursor.fetchall()]

                if teams:
                    # Sort for rankings
                    off_pts = sorted(teams, key=lambda x: x.get('points_scored', 0) or 0, reverse=True)
                    off_pass = sorted(teams, key=lambda x: x.get('pass_yards', 0) or 0, reverse=True)
                    off_rush = sorted(teams, key=lambda x: x.get('rush_yards', 0) or 0, reverse=True)
                    def_pts = sorted(teams, key=lambda x: x.get('points_allowed', 0) or 0)
                    def_pass = sorted(teams, key=lambda x: x.get('pass_yards_allowed', 0) or 0)
                    def_rush = sorted(teams, key=lambda x: x.get('rush_yards_allowed', 0) or 0)

                    for t in teams:
                        team = t.get('team')
                        if not team:
                            continue

                        rankings[team] = {
                            'off_points_rank': next((i+1 for i, x in enumerate(off_pts) if x.get('team') == team), 16),
                            'off_pass_rank': next((i+1 for i, x in enumerate(off_pass) if x.get('team') == team), 16),
                            'off_rush_rank': next((i+1 for i, x in enumerate(off_rush) if x.get('team') == team), 16),
                            'def_points_rank': next((i+1 for i, x in enumerate(def_pts) if x.get('team') == team), 16),
                            'def_pass_rank': next((i+1 for i, x in enumerate(def_pass) if x.get('team') == team), 16),
                            'def_rush_rank': next((i+1 for i, x in enumerate(def_rush) if x.get('team') == team), 16),
                        }

        except Exception as e:
            logger.error(f"Error loading rankings: {e}")

        return rankings

    def get_trending_form(self, team: str, season: int = 2024, week: int = 12) -> TrendingForm:
        """Get trending form for a team."""
        form = TrendingForm(team=team)

        try:
            with get_db() as conn:
                cursor = conn.cursor()

                # Get last 3 games
                cursor.execute("""
                    SELECT points_scored, points_allowed, pass_yards, rush_yards
                    FROM team_stats
                    WHERE team = ? AND season = ? AND week < ?
                    ORDER BY week DESC
                    LIMIT 3
                """, (team, season, week))

                recent = [dict(row) for row in cursor.fetchall()]

                # Get season averages
                cursor.execute("""
                    SELECT
                        AVG(points_scored) as pts,
                        AVG(points_allowed) as pts_allowed,
                        AVG(pass_yards) as pass_yds,
                        AVG(rush_yards) as rush_yds
                    FROM team_stats
                    WHERE team = ? AND season = ?
                """, (team, season))

                season_avg = cursor.fetchone()

                if recent and season_avg:
                    s = dict(season_avg)

                    # Calculate recent averages
                    n = len(recent)
                    recent_pts = sum(r.get('points_scored', 0) or 0 for r in recent) / n
                    recent_pass = sum(r.get('pass_yards', 0) or 0 for r in recent) / n
                    recent_rush = sum(r.get('rush_yards', 0) or 0 for r in recent) / n
                    recent_allowed = sum(r.get('points_allowed', 0) or 0 for r in recent) / n

                    form.recent_points_avg = recent_pts
                    form.season_points_avg = s.get('pts', 0) or 0
                    form.recent_pass_yards_avg = recent_pass
                    form.season_pass_yards_avg = s.get('pass_yds', 0) or 0
                    form.recent_rush_yards_avg = recent_rush
                    form.season_rush_yards_avg = s.get('rush_yds', 0) or 0
                    form.recent_points_allowed_avg = recent_allowed
                    form.season_points_allowed_avg = s.get('pts_allowed', 0) or 0

                    # Calculate trends
                    form.scoring_trend, form.scoring_trend_pct = self._calc_trend(recent_pts, form.season_points_avg)
                    form.pass_trend, form.pass_trend_pct = self._calc_trend(recent_pass, form.season_pass_yards_avg)
                    form.rush_trend, form.rush_trend_pct = self._calc_trend(recent_rush, form.season_rush_yards_avg)
                    form.defense_trend, _ = self._calc_trend(form.season_points_allowed_avg, recent_allowed)

                    # Form grade
                    score = 0
                    if form.scoring_trend == "up": score += 2
                    elif form.scoring_trend == "down": score -= 2
                    if form.pass_trend == "up": score += 1
                    if form.rush_trend == "up": score += 1
                    if form.defense_trend == "up": score += 1
                    elif form.defense_trend == "down": score -= 1

                    if score >= 3: form.form_grade = "A"
                    elif score >= 1: form.form_grade = "B"
                    elif score >= -1: form.form_grade = "C"
                    elif score >= -3: form.form_grade = "D"
                    else: form.form_grade = "F"

                    # Momentum
                    if form.form_grade in ["A", "B"] and form.scoring_trend == "up":
                        form.momentum = "hot"
                        form.form_narrative = f"{team} is HOT - scoring {recent_pts:.1f} ppg last 3 games vs {form.season_points_avg:.1f} season average (+{form.scoring_trend_pct:.0f}%)"
                    elif form.form_grade in ["D", "F"] and form.scoring_trend == "down":
                        form.momentum = "cold"
                        form.form_narrative = f"{team} is COLD - scoring just {recent_pts:.1f} ppg last 3 games vs {form.season_points_avg:.1f} season average ({form.scoring_trend_pct:.0f}%)"
                    else:
                        form.momentum = "neutral"
                        form.form_narrative = f"{team} steady - {recent_pts:.1f} ppg last 3 games vs {form.season_points_avg:.1f} season"

        except Exception as e:
            logger.warning(f"Error getting form: {e}")

        return form

    def _calc_trend(self, recent: float, season: float) -> Tuple[str, float]:
        """Calculate trend direction and percentage."""
        if season == 0:
            return "neutral", 0.0
        pct = ((recent - season) / season) * 100
        if pct >= 10:
            return "up", pct
        elif pct <= -10:
            return "down", pct
        return "neutral", pct

    def get_weather_impact(self, game_id: str) -> WeatherImpact:
        """Get weather impact for a game."""
        impact = WeatherImpact()

        try:
            with get_db() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT temp, wind, roof
                    FROM schedules
                    WHERE game_id = ?
                """, (game_id,))

                game = cursor.fetchone()

                if game:
                    g = dict(game)
                    temp = g.get('temp') or 70
                    wind = g.get('wind') or 0
                    roof = g.get('roof', '')

                    impact.temperature = temp
                    impact.wind_speed = wind
                    impact.roof_type = roof
                    impact.is_dome = roof in ['dome', 'closed']

                    if impact.is_dome:
                        impact.pass_impact = "favorable"
                        impact.kick_impact = "favorable"
                        impact.weather_narrative = "Dome game - optimal passing conditions"
                    else:
                        if wind >= 20:
                            impact.high_wind = True
                            impact.pass_impact = "unfavorable"
                            impact.kick_impact = "unfavorable"
                            impact.weather_narrative = f"HIGH WINDS ({wind} mph) - FADE passing props, favor rushing"
                            impact.weather_props = ["rushing_yards OVER", "passing_yards UNDER", "field_goals UNDER"]
                        elif wind >= 15:
                            impact.high_wind = True
                            impact.kick_impact = "unfavorable"
                            impact.weather_narrative = f"Windy ({wind} mph) - kicker props affected"
                            impact.weather_props = ["field_goals UNDER"]

                        if temp <= 32:
                            impact.extreme_cold = True
                            if not impact.weather_narrative:
                                impact.weather_narrative = f"Cold weather ({temp}F) - grip issues possible"
                        elif temp >= 90:
                            impact.extreme_heat = True
                            if not impact.weather_narrative:
                                impact.weather_narrative = f"Heat ({temp}F) - watch late-game fatigue"

                        if not impact.weather_narrative:
                            impact.weather_narrative = f"Normal conditions ({temp}F, {wind} mph)"

        except Exception as e:
            logger.warning(f"Error getting weather: {e}")

        return impact

    def get_schedule_advantage(self, team: str, opponent: str,
                                season: int, week: int, is_home: bool) -> ScheduleAdvantage:
        """Get schedule advantages for a team."""
        adv = ScheduleAdvantage(team=team, is_home=is_home)

        try:
            with get_db() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT away_rest, home_rest, away_team, home_team
                    FROM schedules
                    WHERE season = ? AND week = ?
                    AND (home_team = ? OR away_team = ?)
                """, (season, week, team, team))

                game = cursor.fetchone()

                if game:
                    g = dict(game)
                    if g.get('home_team') == team:
                        adv.days_rest = g.get('home_rest', 7) or 7
                        adv.opponent_rest = g.get('away_rest', 7) or 7
                    else:
                        adv.days_rest = g.get('away_rest', 7) or 7
                        adv.opponent_rest = g.get('home_rest', 7) or 7

                    adv.rest_advantage = adv.days_rest - adv.opponent_rest
                    adv.is_short_week = adv.days_rest <= 4
                    adv.is_long_week = adv.days_rest >= 10

                # Check division
                divisions = {
                    'AFC East': ['BUF', 'MIA', 'NE', 'NYJ'],
                    'AFC North': ['BAL', 'CIN', 'CLE', 'PIT'],
                    'AFC South': ['HOU', 'IND', 'JAX', 'TEN'],
                    'AFC West': ['DEN', 'KC', 'LV', 'LAC'],
                    'NFC East': ['DAL', 'NYG', 'PHI', 'WAS'],
                    'NFC North': ['CHI', 'DET', 'GB', 'MIN'],
                    'NFC South': ['ATL', 'CAR', 'NO', 'TB'],
                    'NFC West': ['ARI', 'LAR', 'SF', 'SEA']
                }

                for div, teams in divisions.items():
                    if team in teams and opponent in teams:
                        adv.is_division = True
                        break

                # Build narrative
                parts = []
                if adv.rest_advantage >= 3:
                    parts.append(f"{team} has {adv.rest_advantage} more days rest")
                elif adv.rest_advantage <= -3:
                    parts.append(f"{team} at rest disadvantage ({adv.rest_advantage} days)")

                if adv.is_short_week:
                    parts.append("short week")
                if adv.is_long_week:
                    parts.append("extra rest (bye)")
                if adv.is_home:
                    parts.append("home")
                if adv.is_division:
                    parts.append("division game")

                adv.schedule_narrative = ", ".join(parts) if parts else "neutral schedule"

        except Exception as e:
            logger.warning(f"Error getting schedule: {e}")

        return adv

    def get_positional_edges(self, home_team: str, away_team: str) -> List[PositionalEdge]:
        """Get positional matchup edges."""
        edges = []

        home_ranks = self.team_rankings.get(home_team, {})
        away_ranks = self.team_rankings.get(away_team, {})

        # Away QB vs Home Pass D
        away_pass_off = away_ranks.get('off_pass_rank', 16)
        home_pass_def = home_ranks.get('def_pass_rank', 16)
        diff = home_pass_def - away_pass_off

        edges.append(PositionalEdge(
            position="QB",
            team=away_team,
            opponent=home_team,
            off_rank=away_pass_off,
            def_rank=home_pass_def,
            rank_diff=diff,
            grade=self._diff_to_grade(diff),
            edge_score=min(100, max(-100, diff * 3.3)),
            insight=self._qb_insight(away_team, home_team, away_pass_off, home_pass_def),
            target_props=self._qb_props(diff)
        ))

        # Away RB vs Home Rush D
        away_rush_off = away_ranks.get('off_rush_rank', 16)
        home_rush_def = home_ranks.get('def_rush_rank', 16)
        diff = home_rush_def - away_rush_off

        edges.append(PositionalEdge(
            position="RB",
            team=away_team,
            opponent=home_team,
            off_rank=away_rush_off,
            def_rank=home_rush_def,
            rank_diff=diff,
            grade=self._diff_to_grade(diff),
            edge_score=min(100, max(-100, diff * 3.3)),
            insight=self._rb_insight(away_team, home_team, away_rush_off, home_rush_def),
            target_props=self._rb_props(diff)
        ))

        # Home QB vs Away Pass D
        home_pass_off = home_ranks.get('off_pass_rank', 16)
        away_pass_def = away_ranks.get('def_pass_rank', 16)
        diff = away_pass_def - home_pass_off

        edges.append(PositionalEdge(
            position="QB",
            team=home_team,
            opponent=away_team,
            off_rank=home_pass_off,
            def_rank=away_pass_def,
            rank_diff=diff,
            grade=self._diff_to_grade(diff),
            edge_score=min(100, max(-100, diff * 3.3)),
            insight=self._qb_insight(home_team, away_team, home_pass_off, away_pass_def),
            target_props=self._qb_props(diff)
        ))

        # Home RB vs Away Rush D
        home_rush_off = home_ranks.get('off_rush_rank', 16)
        away_rush_def = away_ranks.get('def_rush_rank', 16)
        diff = away_rush_def - home_rush_off

        edges.append(PositionalEdge(
            position="RB",
            team=home_team,
            opponent=away_team,
            off_rank=home_rush_off,
            def_rank=away_rush_def,
            rank_diff=diff,
            grade=self._diff_to_grade(diff),
            edge_score=min(100, max(-100, diff * 3.3)),
            insight=self._rb_insight(home_team, away_team, home_rush_off, away_rush_def),
            target_props=self._rb_props(diff)
        ))

        return edges

    def _diff_to_grade(self, diff: int) -> str:
        if diff >= 15: return "A+"
        elif diff >= 10: return "A"
        elif diff >= 5: return "B+"
        elif diff >= 0: return "B"
        elif diff >= -5: return "C"
        elif diff >= -10: return "D"
        else: return "F"

    def _qb_insight(self, team: str, opp: str, off: int, def_: int) -> str:
        if off <= 10 and def_ >= 20:
            return f"{team} top-10 pass offense vs {opp} bottom-12 pass D - SMASH SPOT"
        elif def_ >= 25:
            return f"{opp} allows most pass yards (#{def_}) - TARGET {team} passing props"
        elif off >= 25 and def_ <= 10:
            return f"{team} weak passing (#{off}) vs {opp} stout D (#{def_}) - FADE"
        return f"{team} pass (#{off}) vs {opp} pass D (#{def_})"

    def _rb_insight(self, team: str, opp: str, off: int, def_: int) -> str:
        if off <= 10 and def_ >= 20:
            return f"{team} top-10 rush offense vs {opp} weak run D - SMASH SPOT"
        elif def_ >= 25:
            return f"{opp} allows most rush yards (#{def_}) - TARGET {team} RB props"
        elif off >= 25 and def_ <= 10:
            return f"{team} weak rush (#{off}) vs {opp} stout D (#{def_}) - FADE"
        return f"{team} rush (#{off}) vs {opp} rush D (#{def_})"

    def _qb_props(self, diff: int) -> List[str]:
        if diff >= 10:
            return ["passing_yards OVER", "passing_tds OVER", "completions OVER"]
        elif diff >= 5:
            return ["passing_yards OVER"]
        elif diff <= -10:
            return ["passing_yards UNDER", "interceptions OVER"]
        return []

    def _rb_props(self, diff: int) -> List[str]:
        if diff >= 10:
            return ["rushing_yards OVER", "rushing_attempts OVER"]
        elif diff >= 5:
            return ["rushing_yards OVER"]
        elif diff <= -10:
            return ["rushing_yards UNDER"]
        return []

    def analyze_game(self, game_id: str, home_team: str, away_team: str,
                     season: int = 2024, week: int = 12) -> GameSituation:
        """Analyze complete situational factors for a game."""

        # Get all components
        home_form = self.get_trending_form(home_team, season, week)
        away_form = self.get_trending_form(away_team, season, week)
        weather = self.get_weather_impact(game_id)
        home_schedule = self.get_schedule_advantage(home_team, away_team, season, week, True)
        away_schedule = self.get_schedule_advantage(away_team, home_team, season, week, False)
        positional_edges = self.get_positional_edges(home_team, away_team)

        # Find key situations
        key_situations = []

        # Positional smash spots
        for edge in positional_edges:
            if edge.edge_score >= 25:
                key_situations.append({
                    'type': 'SMASH_SPOT',
                    'team': edge.team,
                    'position': edge.position,
                    'grade': edge.grade,
                    'edge_score': edge.edge_score,
                    'description': edge.insight,
                    'props': edge.target_props,
                    'confidence': 'HIGH'
                })
            elif edge.edge_score >= 15:
                key_situations.append({
                    'type': 'FAVORABLE',
                    'team': edge.team,
                    'position': edge.position,
                    'grade': edge.grade,
                    'edge_score': edge.edge_score,
                    'description': edge.insight,
                    'props': edge.target_props,
                    'confidence': 'MEDIUM'
                })

        # Hot/cold teams
        for team, form in [(home_team, home_form), (away_team, away_form)]:
            if form.momentum == "hot":
                key_situations.append({
                    'type': 'HOT_TEAM',
                    'team': team,
                    'edge_score': 15,
                    'description': form.form_narrative,
                    'confidence': 'MEDIUM'
                })
            elif form.momentum == "cold":
                key_situations.append({
                    'type': 'COLD_TEAM',
                    'team': team,
                    'edge_score': -15,
                    'description': form.form_narrative,
                    'confidence': 'MEDIUM'
                })

        # Weather impacts
        if weather.high_wind:
            key_situations.append({
                'type': 'WEATHER',
                'edge_score': 20,
                'description': weather.weather_narrative,
                'props': weather.weather_props,
                'confidence': 'HIGH'
            })

        # Rest advantages
        if home_schedule.rest_advantage >= 3:
            key_situations.append({
                'type': 'REST_EDGE',
                'team': home_team,
                'edge_score': home_schedule.rest_advantage * 5,
                'description': home_schedule.schedule_narrative,
                'confidence': 'MEDIUM'
            })
        elif away_schedule.rest_advantage >= 3:
            key_situations.append({
                'type': 'REST_EDGE',
                'team': away_team,
                'edge_score': away_schedule.rest_advantage * 5,
                'description': away_schedule.schedule_narrative,
                'confidence': 'MEDIUM'
            })

        # Generate prop targets from situations
        prop_targets = []
        for sit in key_situations:
            if sit.get('type') in ['SMASH_SPOT', 'FAVORABLE'] and sit.get('props'):
                prop_targets.append({
                    'team': sit.get('team'),
                    'position': sit.get('position', 'N/A'),
                    'props': sit['props'],
                    'rationale': sit['description'],
                    'confidence': sit['confidence'],
                    'edge_score': sit['edge_score']
                })

        prop_targets.sort(key=lambda x: x['edge_score'], reverse=True)

        return GameSituation(
            game_id=game_id,
            season=season,
            week=week,
            home_team=home_team,
            away_team=away_team,
            home_form=home_form,
            away_form=away_form,
            weather=weather,
            home_schedule=home_schedule,
            away_schedule=away_schedule,
            positional_edges=positional_edges,
            key_situations=key_situations,
            prop_targets=prop_targets
        )

    def get_week_situations(self, season: int = 2024, week: int = 12,
                            min_edge: float = 15.0) -> List[Dict]:
        """Get all high-edge situations for a week."""
        all_situations = []

        try:
            with get_db() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT game_id, home_team, away_team
                    FROM schedules
                    WHERE season = ? AND week = ?
                """, (season, week))

                games = cursor.fetchall()

                for game in games:
                    g = dict(game)
                    analysis = self.analyze_game(
                        g['game_id'], g['home_team'], g['away_team'],
                        season, week
                    )

                    for sit in analysis.key_situations:
                        if sit.get('edge_score', 0) >= min_edge:
                            sit['game_id'] = g['game_id']
                            sit['matchup'] = f"{g['away_team']} @ {g['home_team']}"
                            all_situations.append(sit)

        except Exception as e:
            logger.error(f"Error getting week situations: {e}")

        all_situations.sort(key=lambda x: x.get('edge_score', 0), reverse=True)
        return all_situations


# Singleton
situational_analyzer = SituationalAnalyzer()


def analyze_game_situation(game_id: str, home_team: str, away_team: str,
                           season: int = 2024, week: int = 12) -> GameSituation:
    """Get complete situational analysis for a game."""
    return situational_analyzer.analyze_game(game_id, home_team, away_team, season, week)


def get_top_situations(season: int = 2024, week: int = 12,
                       min_edge: float = 15.0) -> List[Dict]:
    """Get top betting situations for a week."""
    return situational_analyzer.get_week_situations(season, week, min_edge)
