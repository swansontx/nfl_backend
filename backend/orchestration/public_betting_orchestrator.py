"""Public Betting Orchestrator - Weekly Sharp Money & Contrarian Analysis.

Orchestrates public betting analysis across all games in a week to identify:
- Strongest sharp money plays (where pros are betting)
- Best contrarian opportunities (fade the public)
- Line movement vs public betting divergence
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd


@dataclass
class SharpMoneyPlay:
    """A sharp money betting opportunity."""
    game_id: str
    home_team: str
    away_team: str
    market: str  # 'spread', 'total', 'moneyline'
    side: str  # Team name or 'OVER'/'UNDER'

    bet_pct: float
    money_pct: float
    sharp_differential: float  # money_pct - bet_pct

    line: Optional[float] = None
    recommended_play: str = ""
    confidence: str = "MEDIUM"  # LOW, MEDIUM, HIGH, VERY_HIGH
    reasoning: str = ""


@dataclass
class ContrarianPlay:
    """A contrarian betting opportunity (fade the public)."""
    game_id: str
    home_team: str
    away_team: str
    market: str
    side: str  # Team name or 'OVER'/'UNDER'

    public_pct: float  # How much public is on the OTHER side
    bet_pct: float
    money_pct: float

    line: Optional[float] = None
    recommended_play: str = ""
    confidence: str = "MEDIUM"
    reasoning: str = ""


@dataclass
class PublicBettingWeekSummary:
    """Weekly summary of public betting intelligence."""
    week: int
    season: int
    total_games: int

    # Sharp money plays
    sharp_money_plays: List[SharpMoneyPlay]
    top_sharp_play: Optional[SharpMoneyPlay] = None

    # Contrarian plays
    contrarian_plays: List[ContrarianPlay]
    top_contrarian_play: Optional[ContrarianPlay] = None

    # Summary stats
    games_with_sharp_money: int = 0
    games_with_contrarian_opps: int = 0
    avg_sharp_differential: float = 0.0


class PublicBettingOrchestrator:
    """Orchestrate public betting analysis across a week."""

    def __init__(self, season: int = 2025):
        """Initialize orchestrator.

        Args:
            season: NFL season year
        """
        self.season = season
        self.inputs_dir = Path('inputs')

    def analyze_week(
        self,
        week: int,
        min_sharp_diff: float = 15.0,
        min_contrarian_pct: float = 75.0
    ) -> PublicBettingWeekSummary:
        """Analyze public betting for an entire week.

        Args:
            week: Week number
            min_sharp_diff: Minimum differential to flag sharp money (default 15%)
            min_contrarian_pct: Minimum public % to flag contrarian (default 75%)

        Returns:
            PublicBettingWeekSummary with all opportunities
        """
        # Get all games for the week
        games = self._get_week_games(week)

        sharp_plays = []
        contrarian_plays = []

        # Analyze each game
        for _, game in games.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']
            game_id = f"{self.season}_{week}_{away_team}_{home_team}"

            # Get public betting data for this game
            public_data = self._get_public_betting_data(game_id, home_team, away_team)

            if not public_data:
                continue

            # Find sharp money plays
            game_sharp_plays = self._find_sharp_money(
                game_id, home_team, away_team, public_data, min_sharp_diff
            )
            sharp_plays.extend(game_sharp_plays)

            # Find contrarian opportunities
            game_contrarian_plays = self._find_contrarian_opps(
                game_id, home_team, away_team, public_data, min_contrarian_pct
            )
            contrarian_plays.extend(game_contrarian_plays)

        # Rank and filter
        sharp_plays = self._rank_sharp_plays(sharp_plays)
        contrarian_plays = self._rank_contrarian_plays(contrarian_plays)

        # Create summary
        summary = PublicBettingWeekSummary(
            week=week,
            season=self.season,
            total_games=len(games),
            sharp_money_plays=sharp_plays,
            contrarian_plays=contrarian_plays,
            top_sharp_play=sharp_plays[0] if sharp_plays else None,
            top_contrarian_play=contrarian_plays[0] if contrarian_plays else None,
            games_with_sharp_money=len(set(p.game_id for p in sharp_plays)),
            games_with_contrarian_opps=len(set(p.game_id for p in contrarian_plays)),
            avg_sharp_differential=sum(p.sharp_differential for p in sharp_plays) / len(sharp_plays) if sharp_plays else 0
        )

        return summary

    def _get_week_games(self, week: int) -> pd.DataFrame:
        """Get all games for a week."""
        try:
            schedule_file = self.inputs_dir / f'{self.season}_schedule.parquet'
            if not schedule_file.exists():
                return pd.DataFrame()

            schedule = pd.read_parquet(schedule_file)
            week_games = schedule[schedule['week'] == week].copy()
            return week_games

        except Exception as e:
            print(f"Error loading schedule: {e}")
            return pd.DataFrame()

    def _get_public_betting_data(
        self,
        game_id: str,
        home_team: str,
        away_team: str
    ) -> Optional[Dict]:
        """Get public betting data for a game."""
        try:
            from backend.ingestion.fetch_public_betting import public_betting_scraper

            # For now, use mock data
            # In production: public_betting_scraper.fetch_sportsbettingdime(week)
            public_data = public_betting_scraper.create_mock_data(
                game_id=game_id,
                home_team=home_team,
                away_team=away_team
            )

            return {
                'spread': public_data.spread,
                'total': public_data.total,
                'moneyline': public_data.moneyline,
                'spread_sharp_on_home': public_data.spread_sharp_on_home,
                'spread_sharp_on_away': public_data.spread_sharp_on_away,
                'total_sharp_on_over': public_data.total_sharp_on_over,
                'total_sharp_on_under': public_data.total_sharp_on_under,
                'spread_contrarian_home': public_data.spread_contrarian_home,
                'spread_contrarian_away': public_data.spread_contrarian_away,
                'total_contrarian_over': public_data.total_contrarian_over,
                'total_contrarian_under': public_data.total_contrarian_under
            }

        except Exception as e:
            print(f"Error fetching public betting data for {game_id}: {e}")
            return None

    def _find_sharp_money(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        public_data: Dict,
        min_differential: float
    ) -> List[SharpMoneyPlay]:
        """Find sharp money plays in a game."""
        plays = []

        # Spread sharp money
        if public_data['spread']:
            spread = public_data['spread']

            # Home team sharp money
            if spread.home_bet_pct and spread.home_money_pct:
                diff = spread.home_money_pct - spread.home_bet_pct
                if diff >= min_differential:
                    plays.append(SharpMoneyPlay(
                        game_id=game_id,
                        home_team=home_team,
                        away_team=away_team,
                        market='spread',
                        side=home_team,
                        bet_pct=spread.home_bet_pct,
                        money_pct=spread.home_money_pct,
                        sharp_differential=diff,
                        line=spread.line,
                        recommended_play=f"{home_team} {spread.line if spread.line else ''}",
                        confidence=self._calculate_sharp_confidence(diff),
                        reasoning=f"{spread.home_bet_pct:.0f}% bets but {spread.home_money_pct:.0f}% money on {home_team}. Sharp money differential: {diff:.1f}%"
                    ))

            # Away team sharp money
            if spread.away_bet_pct and spread.away_money_pct:
                diff = spread.away_money_pct - spread.away_bet_pct
                if diff >= min_differential:
                    plays.append(SharpMoneyPlay(
                        game_id=game_id,
                        home_team=home_team,
                        away_team=away_team,
                        market='spread',
                        side=away_team,
                        bet_pct=spread.away_bet_pct,
                        money_pct=spread.away_money_pct,
                        sharp_differential=diff,
                        line=-spread.line if spread.line else None,
                        recommended_play=f"{away_team} {-spread.line if spread.line else ''}",
                        confidence=self._calculate_sharp_confidence(diff),
                        reasoning=f"{spread.away_bet_pct:.0f}% bets but {spread.away_money_pct:.0f}% money on {away_team}. Sharp money differential: {diff:.1f}%"
                    ))

        # Total sharp money
        if public_data['total']:
            total = public_data['total']

            # Over sharp money
            if total.over_bet_pct and total.over_money_pct:
                diff = total.over_money_pct - total.over_bet_pct
                if diff >= min_differential:
                    plays.append(SharpMoneyPlay(
                        game_id=game_id,
                        home_team=home_team,
                        away_team=away_team,
                        market='total',
                        side='OVER',
                        bet_pct=total.over_bet_pct,
                        money_pct=total.over_money_pct,
                        sharp_differential=diff,
                        line=total.line,
                        recommended_play=f"OVER {total.line if total.line else ''}",
                        confidence=self._calculate_sharp_confidence(diff),
                        reasoning=f"{total.over_bet_pct:.0f}% bets but {total.over_money_pct:.0f}% money on OVER. Sharp money differential: {diff:.1f}%"
                    ))

            # Under sharp money
            if total.under_bet_pct and total.under_money_pct:
                diff = total.under_money_pct - total.under_bet_pct
                if diff >= min_differential:
                    plays.append(SharpMoneyPlay(
                        game_id=game_id,
                        home_team=home_team,
                        away_team=away_team,
                        market='total',
                        side='UNDER',
                        bet_pct=total.under_bet_pct,
                        money_pct=total.under_money_pct,
                        sharp_differential=diff,
                        line=total.line,
                        recommended_play=f"UNDER {total.line if total.line else ''}",
                        confidence=self._calculate_sharp_confidence(diff),
                        reasoning=f"{total.under_bet_pct:.0f}% bets but {total.under_money_pct:.0f}% money on UNDER. Sharp money differential: {diff:.1f}%"
                    ))

        return plays

    def _find_contrarian_opps(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        public_data: Dict,
        min_public_pct: float
    ) -> List[ContrarianPlay]:
        """Find contrarian opportunities in a game."""
        plays = []

        # Spread contrarian
        if public_data['spread']:
            spread = public_data['spread']

            # Bet home (public on away)
            if spread.away_bet_pct and spread.away_bet_pct >= min_public_pct:
                plays.append(ContrarianPlay(
                    game_id=game_id,
                    home_team=home_team,
                    away_team=away_team,
                    market='spread',
                    side=home_team,
                    public_pct=spread.away_bet_pct,
                    bet_pct=spread.home_bet_pct or 0,
                    money_pct=spread.home_money_pct or 0,
                    line=spread.line,
                    recommended_play=f"{home_team} {spread.line if spread.line else ''}",
                    confidence=self._calculate_contrarian_confidence(spread.away_bet_pct),
                    reasoning=f"{spread.away_bet_pct:.0f}% of public on {away_team}. Fade the public and bet {home_team}."
                ))

            # Bet away (public on home)
            if spread.home_bet_pct and spread.home_bet_pct >= min_public_pct:
                plays.append(ContrarianPlay(
                    game_id=game_id,
                    home_team=home_team,
                    away_team=away_team,
                    market='spread',
                    side=away_team,
                    public_pct=spread.home_bet_pct,
                    bet_pct=spread.away_bet_pct or 0,
                    money_pct=spread.away_money_pct or 0,
                    line=-spread.line if spread.line else None,
                    recommended_play=f"{away_team} {-spread.line if spread.line else ''}",
                    confidence=self._calculate_contrarian_confidence(spread.home_bet_pct),
                    reasoning=f"{spread.home_bet_pct:.0f}% of public on {home_team}. Fade the public and bet {away_team}."
                ))

        # Total contrarian
        if public_data['total']:
            total = public_data['total']

            # Bet over (public on under)
            if total.under_bet_pct and total.under_bet_pct >= min_public_pct:
                plays.append(ContrarianPlay(
                    game_id=game_id,
                    home_team=home_team,
                    away_team=away_team,
                    market='total',
                    side='OVER',
                    public_pct=total.under_bet_pct,
                    bet_pct=total.over_bet_pct or 0,
                    money_pct=total.over_money_pct or 0,
                    line=total.line,
                    recommended_play=f"OVER {total.line if total.line else ''}",
                    confidence=self._calculate_contrarian_confidence(total.under_bet_pct),
                    reasoning=f"{total.under_bet_pct:.0f}% of public on UNDER. Fade the public and bet OVER."
                ))

            # Bet under (public on over)
            if total.over_bet_pct and total.over_bet_pct >= min_public_pct:
                plays.append(ContrarianPlay(
                    game_id=game_id,
                    home_team=home_team,
                    away_team=away_team,
                    market='total',
                    side='UNDER',
                    public_pct=total.over_bet_pct,
                    bet_pct=total.under_bet_pct or 0,
                    money_pct=total.under_money_pct or 0,
                    line=total.line,
                    recommended_play=f"UNDER {total.line if total.line else ''}",
                    confidence=self._calculate_contrarian_confidence(total.over_bet_pct),
                    reasoning=f"{total.over_bet_pct:.0f}% of public on OVER. Fade the public and bet UNDER."
                ))

        return plays

    def _calculate_sharp_confidence(self, differential: float) -> str:
        """Calculate confidence level for sharp money play."""
        if differential >= 30:
            return "VERY_HIGH"
        elif differential >= 22:
            return "HIGH"
        elif differential >= 15:
            return "MEDIUM"
        else:
            return "LOW"

    def _calculate_contrarian_confidence(self, public_pct: float) -> str:
        """Calculate confidence level for contrarian play."""
        if public_pct >= 85:
            return "VERY_HIGH"
        elif public_pct >= 80:
            return "HIGH"
        elif public_pct >= 75:
            return "MEDIUM"
        else:
            return "LOW"

    def _rank_sharp_plays(self, plays: List[SharpMoneyPlay]) -> List[SharpMoneyPlay]:
        """Rank sharp money plays by differential."""
        return sorted(plays, key=lambda x: x.sharp_differential, reverse=True)

    def _rank_contrarian_plays(self, plays: List[ContrarianPlay]) -> List[ContrarianPlay]:
        """Rank contrarian plays by public percentage."""
        return sorted(plays, key=lambda x: x.public_pct, reverse=True)


# Singleton instance
public_betting_orchestrator = PublicBettingOrchestrator()


if __name__ == "__main__":
    # Test orchestrator
    orchestrator = PublicBettingOrchestrator(season=2025)
    summary = orchestrator.analyze_week(week=12)

    print(f"Week {summary.week} Public Betting Analysis")
    print(f"Total Games: {summary.total_games}")
    print(f"Games with Sharp Money: {summary.games_with_sharp_money}")
    print(f"Games with Contrarian Opps: {summary.games_with_contrarian_opps}")

    if summary.top_sharp_play:
        print(f"\nTop Sharp Money Play:")
        print(f"  {summary.top_sharp_play.recommended_play}")
        print(f"  Differential: {summary.top_sharp_play.sharp_differential:.1f}%")
        print(f"  Confidence: {summary.top_sharp_play.confidence}")

    if summary.top_contrarian_play:
        print(f"\nTop Contrarian Play:")
        print(f"  {summary.top_contrarian_play.recommended_play}")
        print(f"  Public %: {summary.top_contrarian_play.public_pct:.0f}%")
        print(f"  Confidence: {summary.top_contrarian_play.confidence}")
