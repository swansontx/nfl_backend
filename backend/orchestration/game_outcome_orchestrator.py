"""Game Outcome Orchestrator - ML Pipeline for Game Predictions.

Full machine learning pipeline for predicting game outcomes:
- Spreads (point differential)
- Totals (combined score)
- Moneylines (win probability)

Uses advanced team metrics, historical data, and situational factors.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import json

# Import ML predictors
try:
    from backend.ml.unified_game_predictor import UnifiedGamePredictor
    UNIFIED_ML_AVAILABLE = True
except ImportError:
    UNIFIED_ML_AVAILABLE = False
    print("⚠️  Unified ML predictor not available, using formula-based predictions")


@dataclass
class GameFeatures:
    """Features for game outcome prediction."""
    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str

    # Basic team stats
    home_off_ppg: float
    home_def_ppg: float
    away_off_ppg: float
    away_def_ppg: float

    # Advanced metrics (when available)
    home_off_epa: Optional[float] = None
    home_def_epa: Optional[float] = None
    away_off_epa: Optional[float] = None
    away_def_epa: Optional[float] = None

    # Recent form (last 3 games)
    home_l3_margin: float = 0.0
    away_l3_margin: float = 0.0
    home_l3_total: float = 0.0
    away_l3_total: float = 0.0

    # Situational factors
    rest_differential: int = 0  # Home rest days - away rest days
    is_division_game: bool = False
    is_primetime: bool = False
    temperature: Optional[float] = None
    wind_speed: Optional[float] = None
    precipitation: Optional[str] = None
    is_dome: bool = False

    # Historical matchup
    h2h_home_margin_avg: float = 0.0  # Historical home team margin
    h2h_total_avg: float = 0.0  # Historical total points

    # Market data
    opening_spread: Optional[float] = None
    current_spread: Optional[float] = None
    opening_total: Optional[float] = None
    current_total: Optional[float] = None
    line_movement_spread: Optional[float] = None
    line_movement_total: Optional[float] = None

    # Public betting data (bet % and money %)
    spread_bet_pct_home: Optional[float] = None
    spread_money_pct_home: Optional[float] = None
    total_bet_pct_over: Optional[float] = None
    total_money_pct_over: Optional[float] = None
    ml_bet_pct_home: Optional[float] = None
    ml_money_pct_home: Optional[float] = None

    # Sharp money indicators (money % >> bet %)
    spread_sharp_on_home: bool = False
    spread_sharp_on_away: bool = False
    total_sharp_on_over: bool = False
    total_sharp_on_under: bool = False

    # Contrarian opportunities (heavy public on one side)
    spread_contrarian_home: bool = False  # Bet home (public on away)
    spread_contrarian_away: bool = False  # Bet away (public on home)
    total_contrarian_over: bool = False  # Bet over (public on under)
    total_contrarian_under: bool = False  # Bet under (public on over)


@dataclass
class GameOutcomePrediction:
    """Game outcome prediction from ML models."""
    game_id: str
    home_team: str
    away_team: str

    # Predictions
    predicted_home_score: float
    predicted_away_score: float
    predicted_margin: float  # Positive = home favored
    predicted_total: float
    home_win_prob: float

    # Uncertainty
    margin_std: float
    total_std: float
    margin_ci: Tuple[float, float]  # 95% confidence interval
    total_ci: Tuple[float, float]

    # Model confidence
    confidence: float  # 0-1 based on feature quality

    # Edge vs market (if available)
    spread_edge: Optional[float] = None
    total_edge: Optional[float] = None
    ml_edge: Optional[float] = None


class GameOutcomeOrchestrator:
    """Orchestrator for game outcome predictions using ML models."""

    def __init__(self, season: int = 2025):
        """Initialize orchestrator.

        Args:
            season: NFL season year
        """
        self.season = season
        self.inputs_dir = Path('inputs')
        self.models_dir = Path('models/game_outcomes')
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Models (loaded on demand)
        self.ml_model = None

        # Initialize unified ML predictor for spreads and totals
        self.unified_predictor = None
        if UNIFIED_ML_AVAILABLE:
            try:
                self.unified_predictor = UnifiedGamePredictor()
                print("✓ Loaded Unified Neural Network (Spreads +4.2%, Totals +1.3%)")
            except Exception as e:
                print(f"⚠️  Could not load unified ML predictor: {e}")
                self.unified_predictor = None

        # Calibrators
        self.margin_calibrator = None
        self.total_calibrator = None

    # ========================================================================
    # PHASE 1: Feature Collection
    # ========================================================================

    def collect_features(self, game_id: str, week: int) -> GameFeatures:
        """Collect all features for a game.

        Args:
            game_id: Game ID (format: {season}_{week}_{away}_{home})
            week: Week number

        Returns:
            GameFeatures object
        """
        # Parse game_id
        parts = game_id.split('_')
        if len(parts) < 4:
            raise ValueError(f"Invalid game_id: {game_id}")

        season = int(parts[0])
        week = int(parts[1])
        away_team = parts[2]
        home_team = parts[3]

        features = GameFeatures(
            game_id=game_id,
            season=season,
            week=week,
            home_team=home_team,
            away_team=away_team,
            home_off_ppg=0.0,
            home_def_ppg=0.0,
            away_off_ppg=0.0,
            away_def_ppg=0.0
        )

        # Collect team stats
        features = self._add_team_stats(features)

        # Collect recent form
        features = self._add_recent_form(features)

        # Collect situational factors
        features = self._add_situational_factors(features)

        # Collect historical matchup data
        features = self._add_historical_matchup(features)

        # Collect market data (if available)
        features = self._add_market_data(features)

        # Collect public betting data (if available)
        features = self._add_public_betting_data(features)

        return features

    def _add_team_stats(self, features: GameFeatures) -> GameFeatures:
        """Add basic team statistics.

        Args:
            features: GameFeatures object

        Returns:
            Updated features
        """
        try:
            # Load offensive stats
            off_stats_file = self.inputs_dir / f'{features.season}_team_stats_offense.csv'
            if off_stats_file.exists():
                off_stats = pd.read_csv(off_stats_file)

                home_off = off_stats[off_stats['team'] == features.home_team]
                away_off = off_stats[off_stats['team'] == features.away_team]

                if len(home_off) > 0:
                    features.home_off_ppg = home_off['points'].sum() / max(home_off['games'].iloc[0], 1)

                if len(away_off) > 0:
                    features.away_off_ppg = away_off['points'].sum() / max(away_off['games'].iloc[0], 1)

            # Load defensive stats
            def_stats_file = self.inputs_dir / f'{features.season}_team_stats_defense.csv'
            if def_stats_file.exists():
                def_stats = pd.read_csv(def_stats_file)

                home_def = def_stats[def_stats['team'] == features.home_team]
                away_def = def_stats[def_stats['team'] == features.away_team]

                if len(home_def) > 0:
                    features.home_def_ppg = home_def['points_allowed'].sum() / max(home_def['games'].iloc[0], 1)

                if len(away_def) > 0:
                    features.away_def_ppg = away_def['points_allowed'].sum() / max(away_def['games'].iloc[0], 1)

        except Exception as e:
            print(f"Error loading team stats: {e}")

        return features

    def _add_recent_form(self, features: GameFeatures) -> GameFeatures:
        """Add recent form features (last 3 games).

        Args:
            features: GameFeatures object

        Returns:
            Updated features
        """
        try:
            schedule_file = self.inputs_dir / f'{features.season}_schedule.parquet'
            if not schedule_file.exists():
                return features

            schedule = pd.read_parquet(schedule_file)

            # Home team recent form
            home_games = schedule[
                ((schedule['home_team'] == features.home_team) |
                 (schedule['away_team'] == features.home_team)) &
                (schedule['week'] < features.week) &
                (pd.notna(schedule['home_score']))
            ].tail(3)

            if len(home_games) > 0:
                margins = []
                totals = []
                for _, game in home_games.iterrows():
                    if game['home_team'] == features.home_team:
                        margin = game['home_score'] - game['away_score']
                        total = game['home_score'] + game['away_score']
                    else:
                        margin = game['away_score'] - game['home_score']
                        total = game['home_score'] + game['away_score']
                    margins.append(margin)
                    totals.append(total)

                features.home_l3_margin = np.mean(margins)
                features.home_l3_total = np.mean(totals)

            # Away team recent form
            away_games = schedule[
                ((schedule['home_team'] == features.away_team) |
                 (schedule['away_team'] == features.away_team)) &
                (schedule['week'] < features.week) &
                (pd.notna(schedule['home_score']))
            ].tail(3)

            if len(away_games) > 0:
                margins = []
                totals = []
                for _, game in away_games.iterrows():
                    if game['home_team'] == features.away_team:
                        margin = game['home_score'] - game['away_score']
                        total = game['home_score'] + game['away_score']
                    else:
                        margin = game['away_score'] - game['home_score']
                        total = game['home_score'] + game['away_score']
                    margins.append(margin)
                    totals.append(total)

                features.away_l3_margin = np.mean(margins)
                features.away_l3_total = np.mean(totals)

        except Exception as e:
            print(f"Error calculating recent form: {e}")

        return features

    def _add_situational_factors(self, features: GameFeatures) -> GameFeatures:
        """Add situational factors.

        Args:
            features: GameFeatures object

        Returns:
            Updated features
        """
        # Check if division game
        division_map = {
            'AFC East': ['BUF', 'MIA', 'NE', 'NYJ'],
            'AFC North': ['BAL', 'CIN', 'CLE', 'PIT'],
            'AFC South': ['HOU', 'IND', 'JAX', 'TEN'],
            'AFC West': ['DEN', 'KC', 'LV', 'LAC'],
            'NFC East': ['DAL', 'NYG', 'PHI', 'WAS'],
            'NFC North': ['CHI', 'DET', 'GB', 'MIN'],
            'NFC South': ['ATL', 'CAR', 'NO', 'TB'],
            'NFC West': ['ARI', 'LAR', 'SF', 'SEA']
        }

        for division, teams in division_map.items():
            if features.home_team in teams and features.away_team in teams:
                features.is_division_game = True
                break

        # Check if dome game
        from backend.api.stadium_database import get_stadium_for_game
        try:
            stadium = get_stadium_for_game(features.game_id)
            if stadium and stadium.get('is_dome'):
                features.is_dome = True
        except:
            pass

        return features

    def _add_historical_matchup(self, features: GameFeatures) -> GameFeatures:
        """Add historical head-to-head data.

        Args:
            features: GameFeatures object

        Returns:
            Updated features
        """
        try:
            # Look back 3 seasons for historical matchups
            margins = []
            totals = []

            for past_season in range(features.season - 3, features.season):
                schedule_file = self.inputs_dir / f'{past_season}_schedule.parquet'
                if schedule_file.exists():
                    schedule = pd.read_parquet(schedule_file)

                    # Find games between these teams
                    h2h = schedule[
                        ((schedule['home_team'] == features.home_team) &
                         (schedule['away_team'] == features.away_team)) |
                        ((schedule['home_team'] == features.away_team) &
                         (schedule['away_team'] == features.home_team))
                    ]

                    for _, game in h2h.iterrows():
                        if pd.notna(game['home_score']):
                            if game['home_team'] == features.home_team:
                                margin = game['home_score'] - game['away_score']
                            else:
                                margin = game['away_score'] - game['home_score']

                            total = game['home_score'] + game['away_score']
                            margins.append(margin)
                            totals.append(total)

            if margins:
                features.h2h_home_margin_avg = np.mean(margins)
                features.h2h_total_avg = np.mean(totals)

        except Exception as e:
            print(f"Error loading historical matchup: {e}")

        return features

    def _add_market_data(self, features: GameFeatures) -> GameFeatures:
        """Add betting market data.

        Args:
            features: GameFeatures object

        Returns:
            Updated features
        """
        # Fetch current odds
        try:
            from backend.ingestion.fetch_odds import fetch_odds_api
            odds_events = fetch_odds_api()

            for event in odds_events:
                event_home = event.get('home_team', '').upper()
                event_away = event.get('away_team', '').upper()

                if (features.home_team.upper() in event_home and
                    features.away_team.upper() in event_away):

                    if event.get('bookmakers'):
                        bookmaker = event['bookmakers'][0]
                        markets = bookmaker.get('markets', [])

                        # Spread
                        spread_market = next((m for m in markets if m.get('key') == 'spreads'), None)
                        if spread_market and spread_market.get('outcomes'):
                            home_spread = next((o for o in spread_market['outcomes']
                                              if features.home_team.upper() in o.get('name', '').upper()), None)
                            if home_spread:
                                features.current_spread = home_spread.get('point')

                        # Total
                        totals_market = next((m for m in markets if m.get('key') == 'totals'), None)
                        if totals_market and totals_market.get('outcomes'):
                            features.current_total = totals_market['outcomes'][0].get('point')

                    break

        except Exception as e:
            print(f"Error fetching market data: {e}")

        return features

    def _add_public_betting_data(self, features: GameFeatures) -> GameFeatures:
        """Add public betting percentages (bet % and money %).

        Args:
            features: GameFeatures object

        Returns:
            Updated features with public betting data
        """
        try:
            from backend.ingestion.fetch_public_betting import public_betting_scraper

            # For now, use mock data (until HTML parsing implemented)
            # In production, would call: scraper.fetch_sportsbettingdime(week=features.week)
            public_data = public_betting_scraper.create_mock_data(
                game_id=features.game_id,
                home_team=features.home_team,
                away_team=features.away_team
            )

            # Add spread betting data
            if public_data.spread:
                features.spread_bet_pct_home = public_data.spread.home_bet_pct
                features.spread_money_pct_home = public_data.spread.home_money_pct

            # Add total betting data
            if public_data.total:
                features.total_bet_pct_over = public_data.total.over_bet_pct
                features.total_money_pct_over = public_data.total.over_money_pct

            # Add moneyline betting data
            if public_data.moneyline:
                features.ml_bet_pct_home = public_data.moneyline.home_bet_pct
                features.ml_money_pct_home = public_data.moneyline.home_money_pct

            # Add sharp money indicators
            features.spread_sharp_on_home = public_data.spread_sharp_on_home
            features.spread_sharp_on_away = public_data.spread_sharp_on_away
            features.total_sharp_on_over = public_data.total_sharp_on_over
            features.total_sharp_on_under = public_data.total_sharp_on_under

            # Add contrarian opportunities
            features.spread_contrarian_home = public_data.spread_contrarian_home
            features.spread_contrarian_away = public_data.spread_contrarian_away
            features.total_contrarian_over = public_data.total_contrarian_over
            features.total_contrarian_under = public_data.total_contrarian_under

        except Exception as e:
            print(f"Error fetching public betting data: {e}")
            # Continue without public betting data

        return features

    # ========================================================================
    # PHASE 2: Feature Engineering
    # ========================================================================

    def engineer_features(self, features: GameFeatures) -> Dict[str, float]:
        """Engineer features for model input.

        Args:
            features: GameFeatures object

        Returns:
            Dictionary of engineered features
        """
        engineered = {}

        # Basic team strength
        engineered['home_net_rating'] = features.home_off_ppg - features.home_def_ppg
        engineered['away_net_rating'] = features.away_off_ppg - features.away_def_ppg
        engineered['net_rating_diff'] = engineered['home_net_rating'] - engineered['away_net_rating']

        # Matchup-specific
        engineered['home_off_vs_away_def'] = features.home_off_ppg - features.away_def_ppg
        engineered['away_off_vs_home_def'] = features.away_off_ppg - features.home_def_ppg

        # Recent form
        engineered['form_diff'] = features.home_l3_margin - features.away_l3_margin
        engineered['recent_total_avg'] = (features.home_l3_total + features.away_l3_total) / 2

        # Situational
        engineered['is_division_game'] = 1.0 if features.is_division_game else 0.0
        engineered['is_dome'] = 1.0 if features.is_dome else 0.0
        engineered['rest_differential'] = float(features.rest_differential)

        # Historical
        engineered['h2h_margin'] = features.h2h_home_margin_avg
        engineered['h2h_total'] = features.h2h_total_avg

        # Home advantage (baseline)
        engineered['home_field_advantage'] = 2.5

        return engineered

    # ========================================================================
    # PHASE 3: Prediction
    # ========================================================================

    def predict_game(
        self,
        game_id: str,
        week: int,
        market_spread: Optional[float] = None,
        market_total: Optional[float] = None
    ) -> GameOutcomePrediction:
        """Generate prediction for a game.

        Args:
            game_id: Game ID
            week: Week number
            market_spread: Current market spread (optional)
            market_total: Current market total (optional)

        Returns:
            GameOutcomePrediction
        """
        # Collect features
        features = self.collect_features(game_id, week)

        # Try unified ML predictor first (predicts spreads AND totals)
        if self.unified_predictor is not None:
            try:
                from backend.backtesting.framework import BacktestingFramework

                # Use unified ML predictor
                framework = BacktestingFramework(seasons=[features.season])
                ml_prediction = self.unified_predictor.predict_game(
                    home_team=features.home_team,
                    away_team=features.away_team,
                    season=features.season,
                    week=features.week,
                    framework=framework
                )

                # Extract predictions from ML model
                predicted_home_score = ml_prediction.predicted_home_score
                predicted_away_score = ml_prediction.predicted_away_score
                predicted_margin = ml_prediction.predicted_spread
                predicted_total = ml_prediction.predicted_total
                home_win_prob = ml_prediction.home_win_prob
                margin_std = ml_prediction.spread_std
                total_std = ml_prediction.total_std

            except Exception as e:
                print(f"⚠️  Unified ML prediction failed, falling back to formula: {e}")
                # Fall back to formula
                X = self.engineer_features(features)
                predicted_margin, margin_std = self._predict_margin_formula(X, features)
                predicted_total, total_std = self._predict_total_formula(X, features)
                predicted_home_score = (predicted_total + predicted_margin) / 2
                predicted_away_score = (predicted_total - predicted_margin) / 2
                home_win_prob = 0.5 + (predicted_margin * 0.028)
                home_win_prob = max(0.01, min(0.99, home_win_prob))

        else:
            # FALLBACK: Use formula-based prediction
            X = self.engineer_features(features)
            predicted_margin, margin_std = self._predict_margin_formula(X, features)
            predicted_total, total_std = self._predict_total_formula(X, features)
            predicted_home_score = (predicted_total + predicted_margin) / 2
            predicted_away_score = (predicted_total - predicted_margin) / 2
            home_win_prob = 0.5 + (predicted_margin * 0.028)
            home_win_prob = max(0.01, min(0.99, home_win_prob))

        # Confidence intervals
        margin_ci = (
            predicted_margin - 1.96 * margin_std,
            predicted_margin + 1.96 * margin_std
        )
        total_ci = (
            predicted_total - 1.96 * total_std,
            predicted_total + 1.96 * total_std
        )

        # Model confidence (higher if more data available)
        confidence = self._calculate_confidence(features)

        # Create prediction
        prediction = GameOutcomePrediction(
            game_id=game_id,
            home_team=features.home_team,
            away_team=features.away_team,
            predicted_home_score=round(predicted_home_score, 1),
            predicted_away_score=round(predicted_away_score, 1),
            predicted_margin=round(predicted_margin, 1),
            predicted_total=round(predicted_total, 1),
            home_win_prob=round(home_win_prob, 3),
            margin_std=round(margin_std, 1),
            total_std=round(total_std, 1),
            margin_ci=(round(margin_ci[0], 1), round(margin_ci[1], 1)),
            total_ci=(round(total_ci[0], 1), round(total_ci[1], 1)),
            confidence=confidence
        )

        # Calculate edge vs market
        if market_spread is not None:
            prediction.spread_edge = round(predicted_margin - market_spread, 2)

        if market_total is not None:
            prediction.total_edge = round(predicted_total - market_total, 2)

        return prediction

    def _predict_margin_formula(
        self,
        X: Dict[str, float],
        features: GameFeatures
    ) -> Tuple[float, float]:
        """Predict margin using formula (until ML model trained).

        Args:
            X: Engineered features
            features: Raw features

        Returns:
            (predicted_margin, std_dev)
        """
        # Base prediction from team stats
        margin = X['net_rating_diff']

        # Add home field advantage
        margin += X['home_field_advantage']

        # Adjust for recent form
        margin += X['form_diff'] * 0.3

        # Adjust for historical matchup
        if X['h2h_margin'] != 0:
            margin += X['h2h_margin'] * 0.15

        # Division game adjustment (typically tighter)
        if X['is_division_game']:
            margin *= 0.9

        # PUBLIC BETTING ADJUSTMENTS
        # Sharp money adjustment (follow the smart money)
        if features.spread_sharp_on_home:
            margin += 0.5  # Boost home team (sharp money on them)
        elif features.spread_sharp_on_away:
            margin -= 0.5  # Reduce home team (sharp money on away)

        # Contrarian adjustment (fade heavy public)
        if features.spread_contrarian_away:
            # Public heavily on home, fade them (bet away)
            margin -= 0.3
        elif features.spread_contrarian_home:
            # Public heavily on away, fade them (bet home)
            margin += 0.3

        # Standard deviation (typical NFL game variance)
        std = 13.5

        return margin, std

    def _predict_total_formula(
        self,
        X: Dict[str, float],
        features: GameFeatures
    ) -> Tuple[float, float]:
        """Predict total using formula (fallback when unified ML unavailable).

        Args:
            X: Engineered features
            features: Raw features

        Returns:
            (predicted_total, std_dev)
        """
        # Formula-based prediction
        # Average of offense vs defense matchups
        total = (
            (features.home_off_ppg + features.away_def_ppg) / 2 +
            (features.away_off_ppg + features.home_def_ppg) / 2
        )

        # Adjust for recent form
        if X['recent_total_avg'] > 0:
            total = total * 0.7 + X['recent_total_avg'] * 0.3

        # Adjust for historical matchup
        if X['h2h_total'] > 0:
            total = total * 0.85 + X['h2h_total'] * 0.15

        # Dome boost
        if X['is_dome']:
            total += 2.0

        # Division game adjustment (usually lower scoring)
        if X['is_division_game']:
            total -= 1.5

        # PUBLIC BETTING ADJUSTMENTS
        # Sharp money adjustment (follow the smart money)
        if features.total_sharp_on_over:
            total += 1.0  # Boost total (sharp money on over)
        elif features.total_sharp_on_under:
            total -= 1.0  # Reduce total (sharp money on under)

        # Contrarian adjustment (fade heavy public)
        if features.total_contrarian_under:
            # Public heavily on over, fade them (bet under)
            total -= 0.7
        elif features.total_contrarian_over:
            # Public heavily on under, fade them (bet over)
            total += 0.7

        # Standard deviation
        std = 11.0

        return total, std

    def _calculate_confidence(self, features: GameFeatures) -> float:
        """Calculate prediction confidence.

        Args:
            features: GameFeatures

        Returns:
            Confidence score (0-1)
        """
        confidence = 0.5  # Base

        # Boost for team stats available
        if features.home_off_ppg > 0 and features.away_off_ppg > 0:
            confidence += 0.1

        # Boost for recent form data
        if features.home_l3_margin != 0 and features.away_l3_margin != 0:
            confidence += 0.15

        # Boost for historical matchup data
        if features.h2h_home_margin_avg != 0:
            confidence += 0.1

        # Boost for market data
        if features.current_spread is not None:
            confidence += 0.15

        return min(1.0, confidence)

    # ========================================================================
    # PHASE 4: Integration
    # ========================================================================

    def run_week_analysis(
        self,
        week: int,
        season: Optional[int] = None
    ) -> List[GameOutcomePrediction]:
        """Generate predictions for all games in a week.

        Args:
            week: Week number
            season: Season year (default: current)

        Returns:
            List of GameOutcomePrediction objects
        """
        season = season or self.season

        # Load schedule
        schedule_file = self.inputs_dir / f'{season}_schedule.parquet'
        if not schedule_file.exists():
            print(f"Schedule not found for season {season}")
            return []

        schedule = pd.read_parquet(schedule_file)
        week_games = schedule[schedule['week'] == week]

        predictions = []
        for _, game in week_games.iterrows():
            game_id = f"{season}_{week}_{game['away_team']}_{game['home_team']}"

            try:
                prediction = self.predict_game(game_id, week)
                predictions.append(prediction)
            except Exception as e:
                print(f"Error predicting {game_id}: {e}")

        return predictions


# Singleton instance
game_outcome_orchestrator = GameOutcomeOrchestrator()
