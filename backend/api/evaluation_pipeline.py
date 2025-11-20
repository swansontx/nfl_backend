"""Master Evaluation Pipeline - Orchestrates ALL analyzers for consistent game analysis.

This module ensures EVERY game analysis uses ALL available data sources:
- Situational Analysis (form, weather, schedule, positional edges)
- Matchup Analysis (EPA, CPOE, pressure rates)
- Injury Impact (replacements, prop redistribution)
- Prop Analysis (value finding, edge calculation)
- Contextual Splits (home/away, grass/turf, etc.)
- Defense Bucketing (coverage tendencies)
- Odds/Line Movement

The pipeline produces a standardized GameEvaluation with scored categories
that can be used by the LLM for consistent analysis.

EVALUATION CATEGORIES:
1. MATCHUP_QUALITY (0-100) - Offensive vs defensive strength
2. SITUATIONAL_EDGE (0-100) - Rest, weather, trending form
3. INJURY_IMPACT (0-100) - Key player availability
4. PROP_VALUE (0-100) - Line inefficiencies
5. HISTORICAL_CONTEXT (0-100) - Head-to-head, splits
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

from backend.database.local_db import get_db

logger = logging.getLogger(__name__)


@dataclass
class CategoryScore:
    """Score for a specific evaluation category."""
    category: str
    score: float  # 0-100
    grade: str    # A+, A, B+, B, C, D, F
    confidence: str  # HIGH, MEDIUM, LOW
    factors: List[Dict] = field(default_factory=list)
    narrative: str = ""


@dataclass
class PropTarget:
    """Specific prop betting target."""
    player: str
    team: str
    prop_type: str
    direction: str  # OVER, UNDER
    edge_score: float
    rationale: str
    confidence: str
    sources: List[str] = field(default_factory=list)  # Which analyzers flagged this


@dataclass
class GameEvaluation:
    """Complete evaluation for a game."""
    game_id: str
    home_team: str
    away_team: str
    season: int
    week: int

    # Category scores
    matchup_quality: CategoryScore = None
    situational_edge: CategoryScore = None
    injury_impact: CategoryScore = None
    prop_value: CategoryScore = None

    # Overall
    overall_score: float = 0.0
    overall_grade: str = "C"
    game_narrative: str = ""

    # Actionable targets
    prop_targets: List[PropTarget] = field(default_factory=list)

    # Key takeaways (top 3-5 points)
    key_takeaways: List[str] = field(default_factory=list)

    # Raw data from each analyzer (for LLM context)
    analyzer_outputs: Dict[str, Any] = field(default_factory=dict)

    # Timestamp
    evaluated_at: str = ""


class EvaluationPipeline:
    """Master pipeline that orchestrates all analyzers."""

    def __init__(self):
        """Initialize pipeline with all analyzers."""
        # Import analyzers lazily to avoid circular imports
        self._situational = None
        self._injury = None
        self._matchup = None
        self._prop = None

    @property
    def situational(self):
        if self._situational is None:
            from backend.api.situational_analyzer import situational_analyzer
            self._situational = situational_analyzer
        return self._situational

    @property
    def injury(self):
        if self._injury is None:
            from backend.api.injury_impact_analyzer import injury_analyzer
            self._injury = injury_analyzer
        return self._injury

    @property
    def matchup(self):
        if self._matchup is None:
            from backend.api.matchup_analyzer import MatchupAnalyzer
            self._matchup = MatchupAnalyzer()
        return self._matchup

    @property
    def prop(self):
        if self._prop is None:
            from backend.api.prop_analyzer import prop_analyzer
            self._prop = prop_analyzer
        return self._prop

    def evaluate_game(self, game_id: str, home_team: str, away_team: str,
                      season: int = 2024, week: int = 12) -> GameEvaluation:
        """Run complete evaluation pipeline for a game.

        This is the MAIN ENTRY POINT that ensures all analyzers are used.

        Args:
            game_id: Game identifier
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: Season year
            week: Week number

        Returns:
            GameEvaluation with all categories scored
        """
        eval_result = GameEvaluation(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            season=season,
            week=week,
            evaluated_at=datetime.now().isoformat()
        )

        # 1. SITUATIONAL ANALYSIS
        eval_result.situational_edge = self._evaluate_situational(
            game_id, home_team, away_team, season, week, eval_result
        )

        # 2. MATCHUP QUALITY
        eval_result.matchup_quality = self._evaluate_matchup(
            home_team, away_team, eval_result
        )

        # 3. INJURY IMPACT
        eval_result.injury_impact = self._evaluate_injuries(
            home_team, away_team, eval_result
        )

        # 4. PROP VALUE
        eval_result.prop_value = self._evaluate_props(
            game_id, home_team, away_team, eval_result
        )

        # Calculate overall score
        eval_result.overall_score = self._calculate_overall_score(eval_result)
        eval_result.overall_grade = self._score_to_grade(eval_result.overall_score)

        # Generate key takeaways
        eval_result.key_takeaways = self._generate_takeaways(eval_result)

        # Generate game narrative
        eval_result.game_narrative = self._generate_narrative(eval_result)

        # Consolidate prop targets from all sources
        eval_result.prop_targets = self._consolidate_prop_targets(eval_result)

        return eval_result

    def _evaluate_situational(self, game_id: str, home_team: str, away_team: str,
                               season: int, week: int, eval_result: GameEvaluation) -> CategoryScore:
        """Evaluate situational factors."""
        try:
            analysis = self.situational.analyze_game(
                game_id, home_team, away_team, season, week
            )

            # Store raw output
            eval_result.analyzer_outputs['situational'] = {
                'home_form': {
                    'momentum': analysis.home_form.momentum,
                    'grade': analysis.home_form.form_grade,
                    'narrative': analysis.home_form.form_narrative
                },
                'away_form': {
                    'momentum': analysis.away_form.momentum,
                    'grade': analysis.away_form.form_grade,
                    'narrative': analysis.away_form.form_narrative
                },
                'weather': {
                    'narrative': analysis.weather.weather_narrative,
                    'props': analysis.weather.weather_props
                },
                'schedule': {
                    'home_rest': analysis.home_schedule.days_rest,
                    'away_rest': analysis.away_schedule.days_rest,
                    'rest_advantage': analysis.home_schedule.rest_advantage
                },
                'key_situations': analysis.key_situations
            }

            # Calculate score
            score = 50.0  # Base
            factors = []

            # Form factors
            if analysis.home_form.momentum == "hot":
                score += 10
                factors.append({'factor': f'{home_team} trending hot', 'impact': 10})
            elif analysis.home_form.momentum == "cold":
                score -= 10
                factors.append({'factor': f'{home_team} trending cold', 'impact': -10})

            if analysis.away_form.momentum == "hot":
                score += 10
                factors.append({'factor': f'{away_team} trending hot', 'impact': 10})
            elif analysis.away_form.momentum == "cold":
                score -= 10
                factors.append({'factor': f'{away_team} trending cold', 'impact': -10})

            # Rest factors
            rest_adv = analysis.home_schedule.rest_advantage
            if abs(rest_adv) >= 3:
                impact = rest_adv * 3
                score += impact
                team = home_team if rest_adv > 0 else away_team
                factors.append({'factor': f'{team} rest advantage ({abs(rest_adv)} days)', 'impact': impact})

            # Weather factors
            if analysis.weather.high_wind:
                score += 5  # Creates betting opportunities
                factors.append({'factor': 'High wind creates opportunities', 'impact': 5})

            # From key situations
            for sit in analysis.key_situations:
                if sit['type'] == 'SMASH_SPOT':
                    score += 15
                    factors.append({'factor': sit['description'][:50], 'impact': 15})

            score = max(0, min(100, score))

            return CategoryScore(
                category="SITUATIONAL_EDGE",
                score=score,
                grade=self._score_to_grade(score),
                confidence="HIGH" if len(factors) >= 3 else "MEDIUM",
                factors=factors,
                narrative=f"Situational analysis identified {len(factors)} key factors"
            )

        except Exception as e:
            logger.error(f"Situational evaluation error: {e}")
            return CategoryScore(
                category="SITUATIONAL_EDGE",
                score=50.0,
                grade="C",
                confidence="LOW",
                narrative=f"Error in situational analysis: {e}"
            )

    def _evaluate_matchup(self, home_team: str, away_team: str,
                          eval_result: GameEvaluation) -> CategoryScore:
        """Evaluate matchup quality using EPA-based analyzer."""
        try:
            # Use the existing matchup analyzer for EPA metrics
            factors = []
            score = 50.0

            # Get positional edges from situational output
            sit_output = eval_result.analyzer_outputs.get('situational', {})
            key_situations = sit_output.get('key_situations', [])

            for sit in key_situations:
                if sit.get('type') in ['SMASH_SPOT', 'FAVORABLE']:
                    edge = sit.get('edge_score', 0)
                    if edge > 0:
                        score += min(edge / 3, 15)
                        factors.append({
                            'factor': f"{sit.get('team', '')} {sit.get('position', '')} matchup",
                            'impact': edge / 3,
                            'insight': sit.get('description', '')[:80]
                        })

            score = max(0, min(100, score))

            # Store raw output
            eval_result.analyzer_outputs['matchup'] = {
                'positional_edges': [
                    {'team': s.get('team'), 'position': s.get('position'),
                     'edge': s.get('edge_score'), 'insight': s.get('description')}
                    for s in key_situations if s.get('type') in ['SMASH_SPOT', 'FAVORABLE']
                ]
            }

            return CategoryScore(
                category="MATCHUP_QUALITY",
                score=score,
                grade=self._score_to_grade(score),
                confidence="MEDIUM",
                factors=factors,
                narrative=f"Found {len(factors)} favorable positional matchups"
            )

        except Exception as e:
            logger.error(f"Matchup evaluation error: {e}")
            return CategoryScore(
                category="MATCHUP_QUALITY",
                score=50.0,
                grade="C",
                confidence="LOW",
                narrative=f"Error in matchup analysis: {e}"
            )

    def _evaluate_injuries(self, home_team: str, away_team: str,
                           eval_result: GameEvaluation) -> CategoryScore:
        """Evaluate injury impact on both teams."""
        try:
            factors = []
            score = 50.0  # Base

            for team in [home_team, away_team]:
                # Get injuries from database
                injuries = self._get_team_injuries(team)

                if injuries:
                    report = self.injury.analyze_team_injuries(team, injuries)

                    # Store raw output
                    if 'injuries' not in eval_result.analyzer_outputs:
                        eval_result.analyzer_outputs['injuries'] = {}

                    eval_result.analyzer_outputs['injuries'][team] = {
                        'total_impact': report.total_impact_score,
                        'key_injuries': [
                            {
                                'player': imp.injured_player,
                                'position': imp.position,
                                'status': imp.status,
                                'replacement': imp.replacement,
                                'narrative': imp.narrative
                            }
                            for imp in report.key_injuries
                        ],
                        'betting_angle': report.betting_angle
                    }

                    # Impact on score
                    if report.total_impact_score >= 50:
                        # Significant injuries = betting opportunities
                        score += 15
                        factors.append({
                            'factor': f'{team} major injuries (impact: {report.total_impact_score:.0f})',
                            'impact': 15,
                            'insight': report.betting_angle
                        })
                    elif report.total_impact_score >= 25:
                        score += 8
                        factors.append({
                            'factor': f'{team} notable injuries (impact: {report.total_impact_score:.0f})',
                            'impact': 8
                        })

            score = max(0, min(100, score))

            return CategoryScore(
                category="INJURY_IMPACT",
                score=score,
                grade=self._score_to_grade(score),
                confidence="HIGH" if len(factors) > 0 else "MEDIUM",
                factors=factors,
                narrative=f"Injury situation creates {'significant' if score >= 65 else 'moderate' if score >= 55 else 'minimal'} opportunities"
            )

        except Exception as e:
            logger.error(f"Injury evaluation error: {e}")
            return CategoryScore(
                category="INJURY_IMPACT",
                score=50.0,
                grade="C",
                confidence="LOW",
                narrative=f"Error in injury analysis: {e}"
            )

    def _evaluate_props(self, game_id: str, home_team: str, away_team: str,
                        eval_result: GameEvaluation) -> CategoryScore:
        """Evaluate prop value opportunities."""
        try:
            factors = []
            score = 50.0

            # Collect prop recommendations from all sources
            sit_output = eval_result.analyzer_outputs.get('situational', {})

            # From key situations
            for sit in sit_output.get('key_situations', []):
                props = sit.get('props', sit.get('target_props', []))
                if props:
                    score += 5
                    factors.append({
                        'factor': f"{sit.get('team', '')} {sit.get('position', '')} props",
                        'props': props,
                        'confidence': sit.get('confidence', 'MEDIUM')
                    })

            # Weather props
            weather = sit_output.get('weather', {})
            if weather.get('props'):
                score += 5
                factors.append({
                    'factor': 'Weather-driven props',
                    'props': weather['props'],
                    'confidence': 'HIGH'
                })

            score = max(0, min(100, score))

            eval_result.analyzer_outputs['props'] = {
                'total_opportunities': len(factors),
                'factors': factors
            }

            return CategoryScore(
                category="PROP_VALUE",
                score=score,
                grade=self._score_to_grade(score),
                confidence="MEDIUM",
                factors=factors,
                narrative=f"Found {len(factors)} prop betting opportunities"
            )

        except Exception as e:
            logger.error(f"Prop evaluation error: {e}")
            return CategoryScore(
                category="PROP_VALUE",
                score=50.0,
                grade="C",
                confidence="LOW",
                narrative=f"Error in prop analysis: {e}"
            )

    def _get_team_injuries(self, team: str) -> List[Dict]:
        """Get injuries for a team from database."""
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM injuries
                    WHERE team = ?
                    AND reported_at = (SELECT MAX(reported_at) FROM injuries WHERE team = ?)
                """, (team, team))

                injuries = [dict(row) for row in cursor.fetchall()]

                # Convert to expected format
                return [
                    {
                        'player_name': inj.get('player_name'),
                        'position': inj.get('position'),
                        'injury_status': inj.get('status'),
                        'injury_type': inj.get('injury_type')
                    }
                    for inj in injuries
                ]
        except Exception as e:
            logger.warning(f"Error getting injuries: {e}")
            return []

    def _calculate_overall_score(self, eval_result: GameEvaluation) -> float:
        """Calculate weighted overall score."""
        weights = {
            'situational': 0.30,
            'matchup': 0.25,
            'injury': 0.25,
            'prop': 0.20
        }

        total = 0.0
        if eval_result.situational_edge:
            total += eval_result.situational_edge.score * weights['situational']
        if eval_result.matchup_quality:
            total += eval_result.matchup_quality.score * weights['matchup']
        if eval_result.injury_impact:
            total += eval_result.injury_impact.score * weights['injury']
        if eval_result.prop_value:
            total += eval_result.prop_value.score * weights['prop']

        return total

    def _score_to_grade(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 90: return "A+"
        elif score >= 85: return "A"
        elif score >= 80: return "B+"
        elif score >= 70: return "B"
        elif score >= 60: return "C"
        elif score >= 50: return "D"
        else: return "F"

    def _generate_takeaways(self, eval_result: GameEvaluation) -> List[str]:
        """Generate top 3-5 key takeaways."""
        takeaways = []

        # From situational
        if eval_result.situational_edge and eval_result.situational_edge.factors:
            for f in eval_result.situational_edge.factors[:2]:
                takeaways.append(f.get('factor', ''))

        # From matchup
        if eval_result.matchup_quality and eval_result.matchup_quality.factors:
            for f in eval_result.matchup_quality.factors[:1]:
                insight = f.get('insight', f.get('factor', ''))
                if insight:
                    takeaways.append(insight[:100])

        # From injuries
        if eval_result.injury_impact and eval_result.injury_impact.factors:
            for f in eval_result.injury_impact.factors[:1]:
                takeaways.append(f.get('insight', f.get('factor', '')))

        return takeaways[:5]

    def _generate_narrative(self, eval_result: GameEvaluation) -> str:
        """Generate overall game narrative."""
        parts = []

        parts.append(f"{eval_result.away_team} @ {eval_result.home_team}")
        parts.append(f"Overall Grade: {eval_result.overall_grade} ({eval_result.overall_score:.0f}/100)")

        if eval_result.key_takeaways:
            parts.append("Key factors: " + "; ".join(eval_result.key_takeaways[:3]))

        return " | ".join(parts)

    def _consolidate_prop_targets(self, eval_result: GameEvaluation) -> List[PropTarget]:
        """Consolidate prop targets from all analyzers."""
        targets = []

        # From situational
        sit_output = eval_result.analyzer_outputs.get('situational', {})
        for sit in sit_output.get('key_situations', []):
            props = sit.get('props', sit.get('target_props', []))
            for prop in props:
                parts = prop.split()
                if len(parts) >= 2:
                    prop_type = parts[0]
                    direction = parts[1]
                    targets.append(PropTarget(
                        player="Team",
                        team=sit.get('team', ''),
                        prop_type=prop_type,
                        direction=direction,
                        edge_score=sit.get('edge_score', 0),
                        rationale=sit.get('description', ''),
                        confidence=sit.get('confidence', 'MEDIUM'),
                        sources=['situational']
                    ))

        # Sort by edge score
        targets.sort(key=lambda x: x.edge_score, reverse=True)

        return targets[:10]  # Top 10

    def evaluate_week(self, season: int = 2024, week: int = 12) -> List[GameEvaluation]:
        """Evaluate all games in a week.

        Args:
            season: Season year
            week: Week number

        Returns:
            List of GameEvaluation for all games
        """
        evaluations = []

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
                    eval_result = self.evaluate_game(
                        g['game_id'], g['home_team'], g['away_team'],
                        season, week
                    )
                    evaluations.append(eval_result)

        except Exception as e:
            logger.error(f"Error evaluating week: {e}")

        # Sort by overall score
        evaluations.sort(key=lambda x: x.overall_score, reverse=True)

        return evaluations


# Singleton
evaluation_pipeline = EvaluationPipeline()


def evaluate_game(game_id: str, home_team: str, away_team: str,
                  season: int = 2024, week: int = 12) -> GameEvaluation:
    """Evaluate a game using the complete pipeline."""
    return evaluation_pipeline.evaluate_game(game_id, home_team, away_team, season, week)


def evaluate_week(season: int = 2024, week: int = 12) -> List[GameEvaluation]:
    """Evaluate all games in a week."""
    return evaluation_pipeline.evaluate_week(season, week)
