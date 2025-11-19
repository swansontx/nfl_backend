"""Narrative generation for game previews and betting angles.

This module generates data-driven narratives using templates and statistical analysis.
Enhanced with LLM integration (OpenAI GPT-4, Anthropic Claude) for sophisticated narratives.

Environment Variables:
    LLM_PROVIDER: 'openai' or 'anthropic' (default: None, uses templates only)
    OPENAI_API_KEY: OpenAI API key (required if using OpenAI)
    ANTHROPIC_API_KEY: Anthropic API key (required if using Anthropic Claude)
    LLM_MODEL: Model to use (default: 'gpt-4' or 'claude-3-sonnet-20240229')
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import os
import json


@dataclass
class Narrative:
    """Structured narrative data."""
    narrative_type: str
    content: str
    generated_at: str
    confidence: float = 0.8
    supporting_stats: Optional[Dict] = None


class NarrativeTemplates:
    """Template-based narrative generation."""

    @staticmethod
    def game_preview(home_team: str, away_team: str, key_stats: Dict) -> Narrative:
        """Generate game preview narrative.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            key_stats: Dict with key stats for both teams

        Returns:
            Narrative object
        """
        home_off_rank = key_stats.get('home_offense_rank', 16)
        away_off_rank = key_stats.get('away_offense_rank', 16)
        home_def_rank = key_stats.get('home_defense_rank', 16)
        away_def_rank = key_stats.get('away_defense_rank', 16)

        # Determine narrative angle based on stats
        if home_off_rank <= 10 and away_off_rank <= 10:
            narrative_angle = "high_scoring"
            content = (
                f"This matchup features two of the league's top offenses. "
                f"{away_team} ranks #{away_off_rank} in total offense, while "
                f"{home_team} comes in at #{home_off_rank}. With both defenses "
                f"ranked in the middle of the pack (#{away_def_rank}, #{home_def_rank}), "
                f"expect an offensive showcase with the over looking attractive."
            )
        elif home_def_rank <= 8 or away_def_rank <= 8:
            better_def = home_team if home_def_rank < away_def_rank else away_team
            better_def_rank = min(home_def_rank, away_def_rank)
            narrative_angle = "defensive_battle"
            content = (
                f"Defense should dictate this contest. {better_def}'s #{better_def_rank} "
                f"ranked defense has been dominant, limiting opponents consistently. "
                f"The total may be inflated - look for value on the under and "
                f"lower-scoring props."
            )
        elif abs(home_off_rank - away_off_rank) > 15:
            better_off = home_team if home_off_rank < away_off_rank else away_team
            worse_off = away_team if home_off_rank < away_off_rank else home_team
            better_rank = min(home_off_rank, away_off_rank)
            worse_rank = max(home_off_rank, away_off_rank)
            narrative_angle = "mismatch"
            content = (
                f"Clear offensive mismatch here. {better_off}'s #{better_rank} offense "
                f"faces {worse_off}'s struggling #{worse_rank} unit. "
                f"Look for {better_off} to control tempo and exploit the weaker opponent. "
                f"Player props for {better_off}'s stars should have solid floors."
            )
        else:
            narrative_angle = "balanced"
            content = (
                f"Evenly matched teams battle it out. {away_team} (#{away_off_rank} offense) "
                f"travels to face {home_team} (#{home_off_rank} offense). "
                f"Game script could swing either way, making in-game betting and "
                f"situational props more attractive than pregame positions."
            )

        return Narrative(
            narrative_type="preview",
            content=content,
            generated_at=datetime.now().isoformat(),
            confidence=0.85,
            supporting_stats={
                "home_offense_rank": home_off_rank,
                "away_offense_rank": away_off_rank,
                "home_defense_rank": home_def_rank,
                "away_defense_rank": away_def_rank,
                "narrative_angle": narrative_angle
            }
        )

    @staticmethod
    def key_matchup(player: Dict, opponent_unit: Dict) -> Narrative:
        """Generate key matchup narrative.

        Args:
            player: Player data with recent performance
            opponent_unit: Opponent defensive unit stats

        Returns:
            Narrative object
        """
        player_name = player.get('name', 'Player')
        position = player.get('position', 'POS')
        avg_stat = player.get('avg_stat', 0)
        opponent_rank = opponent_unit.get('rank', 16)
        stat_allowed = opponent_unit.get('avg_allowed', 0)

        if opponent_rank >= 25:  # Weak defense
            content = (
                f"Exploit Alert: {player_name} ({position}) should thrive against "
                f"the #{opponent_rank} ranked defensive unit. Averaging {avg_stat:.1f} "
                f"on the season, he faces a defense allowing {stat_allowed:.1f} per game "
                f"to the position. This matchup screams OVER opportunity."
            )
            angle = "exploit"
        elif opponent_rank <= 8:  # Elite defense
            content = (
                f"Tough Test: {player_name} meets an elite defensive challenge. "
                f"The #{opponent_rank} unit has been stifling, allowing just "
                f"{stat_allowed:.1f} per game. While {player_name} averages {avg_stat:.1f}, "
                f"expect regression in this spot. Under looks safer than over."
            )
            angle = "challenge"
        else:
            content = (
                f"Neutral Matchup: {player_name}'s {avg_stat:.1f} average meets a "
                f"middle-of-the-road #{opponent_rank} defense allowing {stat_allowed:.1f}. "
                f"Stick close to season averages - no major edge either way."
            )
            angle = "neutral"

        return Narrative(
            narrative_type="key_matchups",
            content=content,
            generated_at=datetime.now().isoformat(),
            confidence=0.8 if angle in ['exploit', 'challenge'] else 0.6,
            supporting_stats={
                "player_avg": avg_stat,
                "opponent_rank": opponent_rank,
                "allowed_per_game": stat_allowed,
                "matchup_angle": angle
            }
        )

    @staticmethod
    def betting_angle(insights: List[Dict], line_data: Optional[Dict] = None) -> Narrative:
        """Generate betting angle narrative from insights.

        Args:
            insights: List of insight dicts
            line_data: Optional Vegas line data

        Returns:
            Narrative object
        """
        if not insights:
            content = "No standout betting angles identified. Consider live betting opportunities."
            return Narrative(
                narrative_type="betting_angle",
                content=content,
                generated_at=datetime.now().isoformat(),
                confidence=0.5
            )

        # Find highest confidence insight
        top_insight = max(insights, key=lambda x: x.get('confidence', 0))
        insight_type = top_insight.get('insight_type', 'trend')
        title = top_insight.get('title', '')
        recommendation = top_insight.get('recommendation', '')
        confidence = top_insight.get('confidence', 0.7)

        if insight_type == 'trend':
            content = (
                f"Value Play Identified: {title}. {recommendation}. "
                f"Trend analysis shows {confidence:.0%} confidence in continued pattern. "
                f"Books may be slow to adjust to recent performance shifts."
            )
        elif insight_type == 'matchup':
            content = (
                f"Matchup Advantage: {title}. {recommendation}. "
                f"Historical data and defensive metrics point to exploitable situation. "
                f"Look for props that haven't fully priced in the matchup dynamics."
            )
        elif insight_type == 'weather':
            content = (
                f"Weather Factor: {title}. {recommendation}. "
                f"Environmental conditions significantly impact performance. "
                f"Early birds may find value before markets adjust."
            )
        else:
            content = (
                f"Sharp Angle: {title}. {recommendation}. "
                f"Multiple data points converge on this opportunity."
            )

        return Narrative(
            narrative_type="betting_angle",
            content=content,
            generated_at=datetime.now().isoformat(),
            confidence=confidence,
            supporting_stats={
                "insight_type": insight_type,
                "confidence": confidence,
                "recommendation": recommendation
            }
        )

    @staticmethod
    def contrarian_angle(public_betting: Dict, sharp_money: Dict) -> Narrative:
        """Generate contrarian betting narrative.

        Args:
            public_betting: Public betting percentages
            sharp_money: Sharp money indicators

        Returns:
            Narrative object
        """
        public_side = public_betting.get('side', 'over')
        public_pct = public_betting.get('percentage', 70)
        sharp_side = sharp_money.get('side', 'under')
        line_movement = sharp_money.get('line_movement', 'down')

        if public_pct >= 70 and public_side != sharp_side:
            content = (
                f"Contrarian Alert: {public_pct}% of public is on the {public_side}, "
                f"but sharp money has moved the line {line_movement}. "
                f"Classic fade-the-public spot - sharps are on the {sharp_side}. "
                f"When the public is heavily on one side but the line moves the other way, "
                f"it typically signals professional money taking the opposite position."
            )
            confidence = 0.75
        else:
            content = (
                f"Public and sharp money aligned on the {public_side}. "
                f"Consensus plays can still win, but less contrarian value available."
            )
            confidence = 0.6

        return Narrative(
            narrative_type="betting_angle",
            content=content,
            generated_at=datetime.now().isoformat(),
            confidence=confidence,
            supporting_stats={
                "public_percentage": public_pct,
                "public_side": public_side,
                "sharp_side": sharp_side,
                "line_movement": line_movement
            }
        )


class NarrativeGenerator:
    """Main narrative generation class with LLM enhancement."""

    def __init__(self):
        self.templates = NarrativeTemplates()
        self.llm_provider = os.getenv('LLM_PROVIDER')  # 'openai' or 'anthropic'
        self.llm_model = os.getenv('LLM_MODEL')

        # Initialize LLM client if configured
        self.llm_client = None
        if self.llm_provider == 'openai':
            self._init_openai()
        elif self.llm_provider == 'anthropic':
            self._init_anthropic()

    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.llm_client = OpenAI(api_key=api_key)
                if not self.llm_model:
                    self.llm_model = 'gpt-4'
                print(f"✓ OpenAI LLM initialized with model: {self.llm_model}")
            else:
                print("⚠ OpenAI API key not found - using templates only")
        except ImportError:
            print("⚠ OpenAI package not installed - using templates only")
            self.llm_provider = None

    def _init_anthropic(self):
        """Initialize Anthropic Claude client."""
        try:
            from anthropic import Anthropic
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if api_key:
                self.llm_client = Anthropic(api_key=api_key)
                if not self.llm_model:
                    self.llm_model = 'claude-3-sonnet-20240229'
                print(f"✓ Anthropic Claude initialized with model: {self.llm_model}")
            else:
                print("⚠ Anthropic API key not found - using templates only")
        except ImportError:
            print("⚠ Anthropic package not installed - using templates only")
            self.llm_provider = None

    def generate_game_narratives(self,
                                game_id: str,
                                team_stats: Dict,
                                player_matchups: List[Dict],
                                insights: List[Dict],
                                weather: Optional[Dict] = None,
                                betting_data: Optional[Dict] = None) -> List[Narrative]:
        """Generate all narratives for a game.

        Args:
            game_id: Game ID
            team_stats: Team statistics
            player_matchups: List of key player matchup data
            insights: Generated insights
            weather: Weather data (optional)
            betting_data: Betting lines and percentages (optional)

        Returns:
            List of Narrative objects
        """
        narratives = []

        # Game preview
        home_team = team_stats.get('home_team', 'HOME')
        away_team = team_stats.get('away_team', 'AWAY')
        preview = self.templates.game_preview(home_team, away_team, team_stats)
        narratives.append(preview)

        # Key matchups (top 2-3)
        for matchup in player_matchups[:3]:
            matchup_narrative = self.templates.key_matchup(
                matchup.get('player', {}),
                matchup.get('opponent_unit', {})
            )
            narratives.append(matchup_narrative)

        # Betting angles
        if insights:
            betting_narrative = self.templates.betting_angle(insights, betting_data)
            narratives.append(betting_narrative)

        # Contrarian angle if betting data available
        if betting_data and betting_data.get('public_betting'):
            contrarian = self.templates.contrarian_angle(
                betting_data.get('public_betting', {}),
                betting_data.get('sharp_money', {})
            )
            narratives.append(contrarian)

        # Weather narrative if impactful
        if weather and not weather.get('is_dome', False):
            if weather.get('wind_speed', 0) > 15 or weather.get('condition') in ['Rain', 'Snow']:
                weather_narrative = Narrative(
                    narrative_type="weather",
                    content=(
                        f"Weather Watch: {weather.get('condition', 'Clear')} with "
                        f"{weather.get('wind_speed', 0)} mph winds expected. "
                        f"These conditions historically favor the ground game and "
                        f"reduce deep passing efficiency. Under totals and rushing "
                        f"props gain appeal in these elements."
                    ),
                    generated_at=datetime.now().isoformat(),
                    confidence=0.75,
                    supporting_stats=weather
                )
                narratives.append(weather_narrative)

        return narratives

    def enhance_with_llm(self, narrative: Narrative, context: Dict) -> str:
        """Enhance narrative with LLM.

        Args:
            narrative: Base narrative
            context: Additional context for LLM (stats, trends, historical data)

        Returns:
            Enhanced narrative content (LLM-generated or original if LLM not available)
        """
        # Return original if no LLM configured
        if not self.llm_client or not self.llm_provider:
            return narrative.content

        try:
            # Build LLM prompt
            system_prompt = self._build_system_prompt(narrative.narrative_type)
            user_prompt = self._build_user_prompt(narrative, context)

            # Call appropriate LLM
            if self.llm_provider == 'openai':
                enhanced = self._enhance_with_openai(system_prompt, user_prompt)
            elif self.llm_provider == 'anthropic':
                enhanced = self._enhance_with_claude(system_prompt, user_prompt)
            else:
                enhanced = narrative.content

            return enhanced if enhanced else narrative.content

        except Exception as e:
            print(f"⚠ LLM enhancement failed: {e}")
            # Fallback to original content
            return narrative.content

    def _build_system_prompt(self, narrative_type: str) -> str:
        """Build system prompt based on narrative type."""
        base = (
            "You are an expert NFL analyst and betting strategist with deep knowledge of "
            "player performance, team dynamics, and betting markets. Your analysis is "
            "data-driven, insightful, and actionable for bettors."
        )

        type_specific = {
            'preview': " Focus on game flow, key matchups, and strategic angles.",
            'key_matchups': " Focus on player vs defense dynamics and statistical trends.",
            'betting_angle': " Focus on value identification, line analysis, and contrarian opportunities.",
            'weather': " Focus on environmental impact on game script and player performance."
        }

        return base + type_specific.get(narrative_type, "")

    def _build_user_prompt(self, narrative: Narrative, context: Dict) -> str:
        """Build user prompt with narrative and context."""
        prompt = f"""Enhance this NFL betting narrative with more depth and insight:

BASE NARRATIVE:
{narrative.content}

SUPPORTING STATS:
{json.dumps(narrative.supporting_stats, indent=2)}

ADDITIONAL CONTEXT:
{json.dumps(context, indent=2)}

Instructions:
- Maintain the core insight and betting angle from the base narrative
- Add statistical depth and historical context where relevant
- Keep it concise (2-4 sentences max)
- Use specific numbers and trends
- Be actionable for bettors
- Don't add speculation without data support
- Maintain professional, analytical tone

Enhanced narrative:"""
        return prompt

    def _enhance_with_openai(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Enhance narrative using OpenAI GPT."""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠ OpenAI API error: {e}")
            return None

    def _enhance_with_claude(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Enhance narrative using Anthropic Claude."""
        try:
            message = self.llm_client.messages.create(
                model=self.llm_model,
                max_tokens=300,
                temperature=0.7,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            return message.content[0].text.strip()
        except Exception as e:
            print(f"⚠ Anthropic API error: {e}")
            return None


# Singleton instance
narrative_generator = NarrativeGenerator()
