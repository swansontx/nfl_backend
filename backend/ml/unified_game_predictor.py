"""Unified game outcome predictor using trained Neural Networks.

Predicts home_score and away_score, then derives:
- Spread (margin)
- Total
- Win probability (for moneylines)

This ensures all predictions are interconnected and consistent.
"""

import numpy as np
import joblib
import json
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass

from backend.backtesting.framework import BacktestingFramework


@dataclass
class UnifiedGamePrediction:
    """Unified game prediction with all derived metrics."""
    # Score predictions
    predicted_home_score: float
    predicted_away_score: float

    # Derived predictions
    predicted_spread: float  # home - away (positive = home favored)
    predicted_total: float   # home + away

    # Win probability
    home_win_prob: float
    away_win_prob: float

    # Confidence and uncertainty
    confidence: float  # 0-1 based on data quality
    spread_std: float  # Standard deviation for spread
    total_std: float   # Standard deviation for total

    # Breakdowns
    features_used: Dict[str, float]


class UnifiedGamePredictor:
    """Production predictor using trained Neural Networks for spreads and totals."""

    def __init__(self, model_dir: str = 'models/unified_game_outcomes'):
        """Initialize predictor.

        Args:
            model_dir: Directory containing trained models
        """
        model_path = Path(model_dir)

        # Load models
        self.home_model = joblib.load(model_path / 'home_score_nn.pkl')
        self.away_model = joblib.load(model_path / 'away_score_nn.pkl')

        # Load metadata
        with open(model_path / 'model_metadata.json', 'r') as f:
            self.metadata = json.load(f)

        self.feature_names = self.metadata['features']

        print(f"✓ Loaded Unified Neural Network models from {model_dir}")
        print(f"  Spread improvement: {self.metadata['spread_improvement']:+.1f}%")
        print(f"  Total improvement: {self.metadata['total_improvement']:+.1f}%")
        print(f"  Features: {len(self.feature_names)}")

    def predict_game(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: int,
        framework: Optional[BacktestingFramework] = None
    ) -> UnifiedGamePrediction:
        """Predict game outcome (spread, total, win probability).

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: Season year
            week: Week number
            framework: Backtesting framework (optional, will create if None)

        Returns:
            UnifiedGamePrediction with all metrics
        """
        if framework is None:
            framework = BacktestingFramework(seasons=[season])

        # Extract features
        features = self._extract_features(home_team, away_team, season, week, framework)

        # Prepare feature vector
        feature_vector = np.array([features[name] for name in self.feature_names])

        # Predict scores
        predicted_home_score = self.home_model.predict(feature_vector.reshape(1, -1))[0]
        predicted_away_score = self.away_model.predict(feature_vector.reshape(1, -1))[0]

        # Derive spread and total
        predicted_spread = predicted_home_score - predicted_away_score
        predicted_total = predicted_home_score + predicted_away_score

        # Win probability from spread
        # Using NFL standard: 1 point spread ≈ 2.8% shift from 50%
        home_win_prob = 0.5 + (predicted_spread * 0.028)
        home_win_prob = max(0.01, min(0.99, home_win_prob))  # Clamp to [0.01, 0.99]
        away_win_prob = 1.0 - home_win_prob

        # Confidence based on data quality
        confidence = self._calculate_confidence(features)

        # Uncertainty (std dev)
        # Based on typical NFL variance, adjusted by confidence
        base_spread_std = 13.5
        base_total_std = 11.0
        spread_std = base_spread_std * (1.0 - 0.3 * confidence)
        total_std = base_total_std * (1.0 - 0.3 * confidence)

        return UnifiedGamePrediction(
            predicted_home_score=round(predicted_home_score, 1),
            predicted_away_score=round(predicted_away_score, 1),
            predicted_spread=round(predicted_spread, 1),
            predicted_total=round(predicted_total, 1),
            home_win_prob=round(home_win_prob, 3),
            away_win_prob=round(away_win_prob, 3),
            confidence=round(confidence, 2),
            spread_std=round(spread_std, 1),
            total_std=round(total_std, 1),
            features_used=features
        )

    def _extract_features(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: int,
        framework: BacktestingFramework
    ) -> Dict[str, float]:
        """Extract features for prediction."""
        features = {}

        # Get team stats
        home_stats = self._get_team_stats(home_team, season, week, framework)
        away_stats = self._get_team_stats(away_team, season, week, framework)

        # Recent form
        features['home_recent_ppg'] = home_stats.get('recent_ppg', 21.0)
        features['away_recent_ppg'] = away_stats.get('recent_ppg', 21.0)

        # Season averages
        features['home_season_ppg'] = home_stats.get('season_ppg', 21.0)
        features['away_season_ppg'] = away_stats.get('season_ppg', 21.0)

        # Defense
        features['home_def_ppg_allowed'] = home_stats.get('def_ppg', 21.0)
        features['away_def_ppg_allowed'] = away_stats.get('def_ppg', 21.0)

        # Defense matchups
        features['home_off_vs_away_def'] = (
            features['home_season_ppg'] - features['away_def_ppg_allowed']
        )
        features['away_off_vs_home_def'] = (
            features['away_season_ppg'] - features['home_def_ppg_allowed']
        )

        # Last 3 games
        features['home_l3_ppg'] = home_stats.get('l3_ppg', 21.0)
        features['away_l3_ppg'] = away_stats.get('l3_ppg', 21.0)
        features['home_l3_margin'] = home_stats.get('l3_margin', 0.0)
        features['away_l3_margin'] = away_stats.get('l3_margin', 0.0)

        # Momentum (trend)
        features['home_trend'] = features['home_l3_ppg'] - features['home_season_ppg']
        features['away_trend'] = features['away_l3_ppg'] - features['away_season_ppg']

        # Volatility
        features['home_std'] = home_stats.get('std', 10.0)
        features['away_std'] = away_stats.get('std', 10.0)

        # Rest differential
        features['rest_differential'] = 0.0  # Could enhance with schedule data
        features['home_off_bye'] = 0.0
        features['away_off_bye'] = 0.0

        # Weather (use defaults for now)
        features['temperature'] = 60.0
        features['wind_speed'] = 5.0
        features['is_cold'] = 0.0
        features['is_windy'] = 0.0

        # Situational
        features['is_primetime'] = 0.0  # Could enhance
        features['is_division_game'] = 1.0 if self._is_division_game(home_team, away_team) else 0.0
        features['week'] = float(week)

        return features

    def _get_team_stats(
        self,
        team: str,
        season: int,
        through_week: int,
        framework: BacktestingFramework
    ) -> Dict[str, float]:
        """Get team stats through a given week."""
        games = framework.load_historical_games(season)

        # Get games for this team before this week
        team_games = [
            g for g in games
            if g.week < through_week and (g.home_team == team or g.away_team == team)
        ]

        if not team_games:
            # Return NFL averages
            return {
                'recent_ppg': 21.0,
                'season_ppg': 21.0,
                'def_ppg': 21.0,
                'l3_ppg': 21.0,
                'l3_margin': 0.0,
                'std': 10.0
            }

        # Calculate stats
        scores = []
        margins = []
        def_scores = []

        for g in team_games:
            if g.home_team == team:
                scores.append(g.home_score)
                margins.append(g.home_score - g.away_score)
                def_scores.append(g.away_score)
            else:
                scores.append(g.away_score)
                margins.append(g.away_score - g.home_score)
                def_scores.append(g.home_score)

        # Recent (last 4 games)
        recent_ppg = np.mean(scores[-4:]) if len(scores) >= 4 else np.mean(scores)

        # Last 3 games
        l3_ppg = np.mean(scores[-3:]) if len(scores) >= 3 else np.mean(scores)
        l3_margin = np.mean(margins[-3:]) if len(margins) >= 3 else np.mean(margins)

        return {
            'recent_ppg': recent_ppg,
            'season_ppg': np.mean(scores),
            'def_ppg': np.mean(def_scores),
            'l3_ppg': l3_ppg,
            'l3_margin': l3_margin,
            'std': np.std(scores) if len(scores) > 1 else 10.0
        }

    def _is_division_game(self, home_team: str, away_team: str) -> bool:
        """Check if teams are in same division."""
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

        for div_teams in divisions.values():
            if home_team in div_teams and away_team in div_teams:
                return True
        return False

    def _calculate_confidence(self, features: Dict[str, float]) -> float:
        """Calculate prediction confidence based on data quality."""
        confidence = 1.0

        # Reduce confidence if using defaults (21.0 = NFL average)
        if features['home_recent_ppg'] == 21.0:
            confidence *= 0.7
        if features['away_recent_ppg'] == 21.0:
            confidence *= 0.7

        # Boost confidence for primetime/division games (more scouted)
        if features['is_division_game'] == 1.0:
            confidence *= 1.1

        return min(1.0, confidence)
