"""Game-level betting market analysis.

Analyzes spreads, moneylines, and over/unders for game outcomes.
Provides win probabilities, edge calculations, and value assessments.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import pandas as pd
from pathlib import Path
from backend.features.game_metrics_features import GameMetricsEngine, EnhancedTeamStrength


@dataclass
class TeamStrength:
    """Team strength metrics."""
    team: str
    offensive_rating: float  # Points per game
    defensive_rating: float  # Points allowed per game
    net_rating: float  # Point differential per game
    recent_form_rating: float  # Last 3 games performance
    home_advantage: float = 2.5  # Points


@dataclass
class GamePrediction:
    """Game outcome prediction."""
    home_team: str
    away_team: str
    home_score: float
    away_score: float
    home_win_prob: float
    away_win_prob: float
    predicted_spread: float  # Positive = home favored
    predicted_total: float
    confidence: float  # 0-1


@dataclass
class MarketAnalysis:
    """Analysis of a betting market."""
    market_type: str  # 'spread', 'moneyline', 'total'
    market_line: float  # The line (spread or total)
    market_odds: int  # American odds
    predicted_value: float  # Our prediction
    edge: float  # Edge percentage
    ev: float  # Expected value
    recommendation: str  # 'BET', 'PASS', 'FADE'
    confidence: float  # 0-1
    reasoning: str


@dataclass
class GameMarketAnalysis:
    """Complete market analysis for a game."""
    game_id: str
    home_team: str
    away_team: str
    prediction: GamePrediction

    # Market analyses
    spread_analysis: Optional[MarketAnalysis] = None
    total_analysis: Optional[MarketAnalysis] = None
    moneyline_home_analysis: Optional[MarketAnalysis] = None
    moneyline_away_analysis: Optional[MarketAnalysis] = None

    # Best bets
    best_bet: Optional[str] = None
    best_bet_ev: Optional[float] = None


class GameMarketAnalyzer:
    """Analyze game-level betting markets."""

    def __init__(self, season: int = 2025, use_enhanced_metrics: bool = True):
        """Initialize analyzer.

        Args:
            season: NFL season year
            use_enhanced_metrics: Whether to use pace/turnover/efficiency metrics
        """
        self.season = season
        self.inputs_dir = Path('inputs')
        self.use_enhanced_metrics = use_enhanced_metrics

        # Initialize game metrics engine for advanced predictions
        if use_enhanced_metrics:
            self.metrics_engine = GameMetricsEngine(season=season, inputs_dir=str(self.inputs_dir))
        else:
            self.metrics_engine = None

    def calculate_team_strength(
        self,
        team: str,
        week: int,
        is_home: bool = False
    ) -> TeamStrength:
        """Calculate team strength metrics.

        Args:
            team: Team abbreviation
            week: Current week
            is_home: Whether team is home

        Returns:
            TeamStrength object with ratings
        """
        # Load team stats
        try:
            team_stats_file = self.inputs_dir / f'{self.season}_team_stats_offense.csv'
            if team_stats_file.exists():
                team_stats = pd.read_csv(team_stats_file)
                team_data = team_stats[team_stats['team'] == team]

                if len(team_data) > 0:
                    # Offensive rating (points per game)
                    offensive_rating = team_data['points'].sum() / max(team_data['games'].iloc[0], 1)

                    # Load defensive stats
                    def_stats_file = self.inputs_dir / f'{self.season}_team_stats_defense.csv'
                    if def_stats_file.exists():
                        def_stats = pd.read_csv(def_stats_file)
                        def_data = def_stats[def_stats['team'] == team]

                        if len(def_data) > 0:
                            defensive_rating = def_data['points_allowed'].sum() / max(def_data['games'].iloc[0], 1)
                        else:
                            defensive_rating = 21.5  # League average
                    else:
                        defensive_rating = 21.5

                    # Net rating
                    net_rating = offensive_rating - defensive_rating

                    # Recent form (last 3 games)
                    recent_form_rating = self._calculate_recent_form(team, week)

                    # Home advantage
                    home_advantage = 2.5 if is_home else 0.0

                    return TeamStrength(
                        team=team,
                        offensive_rating=offensive_rating,
                        defensive_rating=defensive_rating,
                        net_rating=net_rating,
                        recent_form_rating=recent_form_rating,
                        home_advantage=home_advantage
                    )

            # Fallback to league average if no data
            return TeamStrength(
                team=team,
                offensive_rating=21.5,
                defensive_rating=21.5,
                net_rating=0.0,
                recent_form_rating=0.0,
                home_advantage=2.5 if is_home else 0.0
            )

        except Exception as e:
            print(f"Error calculating team strength: {e}")
            # Return league average
            return TeamStrength(
                team=team,
                offensive_rating=21.5,
                defensive_rating=21.5,
                net_rating=0.0,
                recent_form_rating=0.0,
                home_advantage=2.5 if is_home else 0.0
            )

    def _calculate_recent_form(self, team: str, week: int) -> float:
        """Calculate recent form adjustment (last 3 games).

        Args:
            team: Team abbreviation
            week: Current week

        Returns:
            Recent form rating (positive = hot, negative = cold)
        """
        try:
            schedule_file = self.inputs_dir / f'{self.season}_schedule.parquet'
            if not schedule_file.exists():
                return 0.0

            schedule = pd.read_parquet(schedule_file)

            # Get team's recent games (completed games before current week)
            team_games = schedule[
                ((schedule['home_team'] == team) | (schedule['away_team'] == team)) &
                (schedule['week'] < week) &
                (pd.notna(schedule['home_score']))
            ].tail(3)

            if len(team_games) == 0:
                return 0.0

            # Calculate average point differential in last 3 games
            point_diffs = []
            for _, game in team_games.iterrows():
                if game['home_team'] == team:
                    diff = game['home_score'] - game['away_score']
                else:
                    diff = game['away_score'] - game['home_score']
                point_diffs.append(diff)

            # Average point differential (positive = winning, negative = losing)
            avg_diff = sum(point_diffs) / len(point_diffs)

            # Convert to form rating (-5 to +5 range)
            form_rating = max(-5.0, min(5.0, avg_diff / 3.0))

            return form_rating

        except Exception as e:
            print(f"Error calculating recent form: {e}")
            return 0.0

    def predict_game_outcome(
        self,
        home_team: str,
        away_team: str,
        week: int,
        recent_weeks: int = 4
    ) -> GamePrediction:
        """Predict game outcome with optional enhanced metrics.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            week: Week number
            recent_weeks: Number of recent weeks for advanced metrics

        Returns:
            GamePrediction with scores and probabilities
        """
        # Calculate basic team strengths
        home_strength = self.calculate_team_strength(home_team, week, is_home=True)
        away_strength = self.calculate_team_strength(away_team, week, is_home=False)

        # Base prediction using strength ratings
        # Home score = home offensive rating vs away defensive rating + home advantage + form
        base_home_score = (
            (home_strength.offensive_rating + away_strength.defensive_rating) / 2.0
            + home_strength.home_advantage
            + home_strength.recent_form_rating
        )

        # Away score = away offensive rating vs home defensive rating + form
        base_away_score = (
            (away_strength.offensive_rating + home_strength.defensive_rating) / 2.0
            + away_strength.recent_form_rating
        )

        # Apply enhanced metrics if enabled
        if self.use_enhanced_metrics and self.metrics_engine:
            # Get enhanced team strengths with pace/turnover/efficiency metrics
            weeks_list = list(range(max(1, week - recent_weeks), week)) if week > 1 else None

            enhanced_home = self.metrics_engine.get_enhanced_team_strength(
                home_team,
                home_strength.offensive_rating,
                home_strength.defensive_rating,
                home_strength.recent_form_rating,
                is_home=True,
                weeks=weeks_list
            )

            enhanced_away = self.metrics_engine.get_enhanced_team_strength(
                away_team,
                away_strength.offensive_rating,
                away_strength.defensive_rating,
                away_strength.recent_form_rating,
                is_home=False,
                weeks=weeks_list
            )

            # Calculate pace adjustment to total
            base_total = base_home_score + base_away_score
            pace_adjusted_total, _ = self.metrics_engine.calculate_pace_adjusted_total(
                enhanced_home, enhanced_away, base_total
            )

            # Calculate turnover adjustment to spread
            base_spread = base_home_score - base_away_score
            to_adjusted_spread, _ = self.metrics_engine.calculate_turnover_adjusted_spread(
                enhanced_home, enhanced_away, base_spread
            )

            # Get efficiency adjustments
            efficiency_adjs = self.metrics_engine.calculate_efficiency_adjustments(
                enhanced_home, enhanced_away
            )

            # Apply all adjustments
            predicted_spread = to_adjusted_spread + efficiency_adjs['spread_adj']
            predicted_total = pace_adjusted_total + efficiency_adjs['total_adj']

            # Recalculate scores from adjusted spread and total
            home_score = (predicted_total + predicted_spread) / 2.0
            away_score = (predicted_total - predicted_spread) / 2.0

        else:
            # Use base predictions
            home_score = base_home_score
            away_score = base_away_score
            predicted_spread = home_score - away_score
            predicted_total = home_score + away_score

        # Win probabilities using point spread to probability conversion
        # Rule of thumb: 1 point spread ≈ 2.8% win probability shift from 50%
        spread_prob_shift = predicted_spread * 0.028
        home_win_prob = 0.50 + spread_prob_shift
        away_win_prob = 1.0 - home_win_prob

        # Clamp probabilities
        home_win_prob = max(0.01, min(0.99, home_win_prob))
        away_win_prob = max(0.01, min(0.99, away_win_prob))

        # Confidence based on data quality and enhanced metrics usage
        confidence = 0.80 if self.use_enhanced_metrics else 0.75

        return GamePrediction(
            home_team=home_team,
            away_team=away_team,
            home_score=round(home_score, 1),
            away_score=round(away_score, 1),
            home_win_prob=round(home_win_prob, 3),
            away_win_prob=round(away_win_prob, 3),
            predicted_spread=round(predicted_spread, 1),
            predicted_total=round(predicted_total, 1),
            confidence=confidence
        )

    def analyze_spread_market(
        self,
        prediction: GamePrediction,
        market_spread: float,
        market_odds: int = -110
    ) -> MarketAnalysis:
        """Analyze spread betting market.

        Args:
            prediction: Game prediction
            market_spread: Market spread line (positive = home favored)
            market_odds: American odds (default -110)

        Returns:
            MarketAnalysis for spread
        """
        # Edge = difference between predicted spread and market spread
        edge = prediction.predicted_spread - market_spread

        # Determine which side to bet
        if edge > 1.5:
            side = "HOME"
            recommendation = "BET"
            reasoning = f"Model predicts {prediction.home_team} by {abs(prediction.predicted_spread):.1f}, market has them at {abs(market_spread):.1f}. {abs(edge):.1f} point edge."
        elif edge < -1.5:
            side = "AWAY"
            recommendation = "BET"
            reasoning = f"Model predicts {prediction.away_team} covers by {abs(edge):.1f} points."
        else:
            side = "NONE"
            recommendation = "PASS"
            reasoning = f"Edge too small ({abs(edge):.1f} points). Market is efficient."

        # Calculate EV
        # Implied probability from odds
        if market_odds < 0:
            implied_prob = abs(market_odds) / (abs(market_odds) + 100)
        else:
            implied_prob = 100 / (market_odds + 100)

        # True probability (simplified - using edge magnitude)
        true_prob = 0.5 + (abs(edge) * 0.03)  # ~3% per point of edge
        true_prob = max(0.01, min(0.99, true_prob))

        # EV calculation
        if recommendation == "BET":
            if market_odds < 0:
                payout = 100 / abs(market_odds)
            else:
                payout = market_odds / 100

            ev = (true_prob * payout) - ((1 - true_prob) * 1.0)
        else:
            ev = 0.0

        return MarketAnalysis(
            market_type="spread",
            market_line=market_spread,
            market_odds=market_odds,
            predicted_value=prediction.predicted_spread,
            edge=round(edge, 2),
            ev=round(ev, 3),
            recommendation=recommendation,
            confidence=prediction.confidence,
            reasoning=reasoning
        )

    def analyze_total_market(
        self,
        prediction: GamePrediction,
        market_total: float,
        market_odds: int = -110
    ) -> MarketAnalysis:
        """Analyze total (over/under) betting market.

        Args:
            prediction: Game prediction
            market_total: Market total line
            market_odds: American odds (default -110)

        Returns:
            MarketAnalysis for total
        """
        # Edge = difference between predicted total and market total
        edge = prediction.predicted_total - market_total

        # Determine which side to bet
        if edge > 3.0:
            side = "OVER"
            recommendation = "BET"
            reasoning = f"Model predicts {prediction.predicted_total:.1f} points, market at {market_total:.1f}. {abs(edge):.1f} point edge on OVER."
        elif edge < -3.0:
            side = "UNDER"
            recommendation = "BET"
            reasoning = f"Model predicts {prediction.predicted_total:.1f} points, market at {market_total:.1f}. {abs(edge):.1f} point edge on UNDER."
        else:
            side = "NONE"
            recommendation = "PASS"
            reasoning = f"Edge too small ({abs(edge):.1f} points). Market is efficient."

        # Calculate EV (similar to spread)
        if market_odds < 0:
            implied_prob = abs(market_odds) / (abs(market_odds) + 100)
        else:
            implied_prob = 100 / (market_odds + 100)

        true_prob = 0.5 + (abs(edge) * 0.02)  # ~2% per point of edge
        true_prob = max(0.01, min(0.99, true_prob))

        if recommendation == "BET":
            if market_odds < 0:
                payout = 100 / abs(market_odds)
            else:
                payout = market_odds / 100

            ev = (true_prob * payout) - ((1 - true_prob) * 1.0)
        else:
            ev = 0.0

        return MarketAnalysis(
            market_type="total",
            market_line=market_total,
            market_odds=market_odds,
            predicted_value=prediction.predicted_total,
            edge=round(edge, 2),
            ev=round(ev, 3),
            recommendation=recommendation,
            confidence=prediction.confidence,
            reasoning=reasoning
        )

    def analyze_moneyline_market(
        self,
        prediction: GamePrediction,
        home_ml_odds: int,
        away_ml_odds: int
    ) -> Tuple[MarketAnalysis, MarketAnalysis]:
        """Analyze moneyline betting markets.

        Args:
            prediction: Game prediction
            home_ml_odds: Home team moneyline odds
            away_ml_odds: Away team moneyline odds

        Returns:
            Tuple of (home_analysis, away_analysis)
        """
        # Convert odds to implied probabilities
        if home_ml_odds < 0:
            home_implied_prob = abs(home_ml_odds) / (abs(home_ml_odds) + 100)
        else:
            home_implied_prob = 100 / (home_ml_odds + 100)

        if away_ml_odds < 0:
            away_implied_prob = abs(away_ml_odds) / (abs(away_ml_odds) + 100)
        else:
            away_implied_prob = 100 / (away_ml_odds + 100)

        # Compare to predicted probabilities
        home_edge = prediction.home_win_prob - home_implied_prob
        away_edge = prediction.away_win_prob - away_implied_prob

        # Home ML analysis
        if home_edge > 0.05:  # 5% edge threshold
            home_rec = "BET"
            home_reason = f"Model gives {prediction.home_team} {prediction.home_win_prob:.1%} to win, market implies {home_implied_prob:.1%}. {home_edge:.1%} edge."
        else:
            home_rec = "PASS"
            home_reason = f"No significant edge. Model: {prediction.home_win_prob:.1%}, Market: {home_implied_prob:.1%}."

        # Calculate home EV
        if home_rec == "BET":
            if home_ml_odds < 0:
                home_payout = 100 / abs(home_ml_odds)
            else:
                home_payout = home_ml_odds / 100

            home_ev = (prediction.home_win_prob * home_payout) - ((1 - prediction.home_win_prob) * 1.0)
        else:
            home_ev = 0.0

        home_analysis = MarketAnalysis(
            market_type="moneyline",
            market_line=home_ml_odds,
            market_odds=home_ml_odds,
            predicted_value=prediction.home_win_prob,
            edge=round(home_edge * 100, 2),  # Convert to percentage
            ev=round(home_ev, 3),
            recommendation=home_rec,
            confidence=prediction.confidence,
            reasoning=home_reason
        )

        # Away ML analysis
        if away_edge > 0.05:
            away_rec = "BET"
            away_reason = f"Model gives {prediction.away_team} {prediction.away_win_prob:.1%} to win, market implies {away_implied_prob:.1%}. {away_edge:.1%} edge."
        else:
            away_rec = "PASS"
            away_reason = f"No significant edge. Model: {prediction.away_win_prob:.1%}, Market: {away_implied_prob:.1%}."

        # Calculate away EV
        if away_rec == "BET":
            if away_ml_odds < 0:
                away_payout = 100 / abs(away_ml_odds)
            else:
                away_payout = away_ml_odds / 100

            away_ev = (prediction.away_win_prob * away_payout) - ((1 - prediction.away_win_prob) * 1.0)
        else:
            away_ev = 0.0

        away_analysis = MarketAnalysis(
            market_type="moneyline",
            market_line=away_ml_odds,
            market_odds=away_ml_odds,
            predicted_value=prediction.away_win_prob,
            edge=round(away_edge * 100, 2),
            ev=round(away_ev, 3),
            recommendation=away_rec,
            confidence=prediction.confidence,
            reasoning=away_reason
        )

        return home_analysis, away_analysis

    def analyze_game(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        week: int,
        market_data: Optional[Dict] = None
    ) -> GameMarketAnalysis:
        """Complete game market analysis.

        Args:
            game_id: Game ID
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            week: Week number
            market_data: Optional market data with spread, total, moneylines

        Returns:
            GameMarketAnalysis with all markets
        """
        # Generate prediction
        prediction = self.predict_game_outcome(home_team, away_team, week)

        # Initialize analysis
        analysis = GameMarketAnalysis(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            prediction=prediction
        )

        # Analyze markets if data provided
        if market_data:
            # Spread
            if 'spread' in market_data and market_data['spread'] is not None:
                spread_odds = market_data.get('spread_odds', -110)
                analysis.spread_analysis = self.analyze_spread_market(
                    prediction,
                    market_data['spread'],
                    spread_odds
                )

            # Total
            if 'total' in market_data and market_data['total'] is not None:
                total_odds = market_data.get('total_odds', -110)
                analysis.total_analysis = self.analyze_total_market(
                    prediction,
                    market_data['total'],
                    total_odds
                )

            # Moneylines
            if 'home_ml' in market_data and 'away_ml' in market_data:
                home_ml_analysis, away_ml_analysis = self.analyze_moneyline_market(
                    prediction,
                    market_data['home_ml'],
                    market_data['away_ml']
                )
                analysis.moneyline_home_analysis = home_ml_analysis
                analysis.moneyline_away_analysis = away_ml_analysis

            # Determine best bet
            bets = []
            if analysis.spread_analysis and analysis.spread_analysis.recommendation == "BET":
                bets.append(("spread", analysis.spread_analysis.ev))
            if analysis.total_analysis and analysis.total_analysis.recommendation == "BET":
                bets.append(("total", analysis.total_analysis.ev))
            if analysis.moneyline_home_analysis and analysis.moneyline_home_analysis.recommendation == "BET":
                bets.append(("home_ml", analysis.moneyline_home_analysis.ev))
            if analysis.moneyline_away_analysis and analysis.moneyline_away_analysis.recommendation == "BET":
                bets.append(("away_ml", analysis.moneyline_away_analysis.ev))

            if bets:
                best = max(bets, key=lambda x: x[1])
                analysis.best_bet = best[0]
                analysis.best_bet_ev = best[1]

        return analysis
