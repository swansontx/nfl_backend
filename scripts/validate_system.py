#!/usr/bin/env python3
"""
System Validation Script

Tests the entire prop modeling system to verify it works:
1. Data pipeline (can we load data?)
2. Feature engineering (do features generate?)
3. Model predictions (do models run?)
4. Calibration (are probabilities accurate?)
5. Recommendations (does scoring work?)
6. Historical backtest (would we have made money?)

Usage:
    python scripts/validate_system.py --season 2024 --full
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from datetime import datetime, timedelta
from typing import Dict, List
import numpy as np
import pandas as pd

from backend.config import settings
from backend.config.logging_config import get_logger
from backend.database.session import get_db
from backend.database.models import Game, Player, PlayerGameFeature, Projection, Outcome

# Import all components
from backend.features.feature_engineer import FeatureEngineer
from backend.models.prop_model_runner import PropModelRunner
from backend.calibration import ProbabilityCalibrator, OutcomeExtractor, CalibrationValidator
from backend.recommendations import RecommendationScorer
from backend.backtest import BacktestEngine, SignalEffectivenessAnalyzer
from backend.trends import TrendAnalyzer
from backend.news import NewsAnalyzer

logger = get_logger(__name__)


class SystemValidator:
    """
    Comprehensive system validation

    Tests each component and the full pipeline
    """

    def __init__(self, season: int = 2024):
        self.season = season
        self.results = {}

    def run_full_validation(self) -> Dict:
        """Run all validation tests"""
        print("=" * 80)
        print("NFL PROP SYSTEM VALIDATION")
        print("=" * 80)
        print(f"Season: {self.season}")
        print(f"Started: {datetime.now()}")
        print()

        # Test each component
        tests = [
            ("Data Availability", self.test_data_availability),
            ("Feature Engineering", self.test_feature_engineering),
            ("Model Predictions", self.test_model_predictions),
            ("Calibration", self.test_calibration),
            ("Trend Analysis", self.test_trend_analysis),
            ("Recommendations", self.test_recommendations),
            ("Historical Backtest", self.test_historical_backtest),
        ]

        passed = 0
        failed = 0

        for test_name, test_func in tests:
            print(f"\n{'=' * 80}")
            print(f"TEST: {test_name}")
            print('=' * 80)

            try:
                result = test_func()
                self.results[test_name] = result

                if result['passed']:
                    print(f"✅ PASSED")
                    passed += 1
                else:
                    print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
                    failed += 1

                # Print details
                if result.get('details'):
                    print("\nDetails:")
                    for key, value in result['details'].items():
                        print(f"  {key}: {value}")

            except Exception as e:
                print(f"❌ FAILED: {str(e)}")
                self.results[test_name] = {'passed': False, 'error': str(e)}
                failed += 1

        # Summary
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Passed: {passed}/{len(tests)}")
        print(f"Failed: {failed}/{len(tests)}")
        print()

        if failed == 0:
            print("🎉 ALL TESTS PASSED! System is working correctly.")
        else:
            print("⚠️  SOME TESTS FAILED. Review errors above.")

        print("=" * 80)

        return self.results

    def test_data_availability(self) -> Dict:
        """Test 1: Can we access the data?"""
        with get_db() as session:
            # Check games
            games = session.query(Game).filter(
                Game.season == self.season
            ).all()

            if not games:
                return {
                    'passed': False,
                    'error': f'No games found for season {self.season}'
                }

            # Check players
            players = session.query(Player).limit(100).all()

            if not players:
                return {
                    'passed': False,
                    'error': 'No players found in database'
                }

            # Check features
            features = session.query(PlayerGameFeature).limit(100).all()

            if not features:
                return {
                    'passed': False,
                    'error': 'No player game features found'
                }

            return {
                'passed': True,
                'details': {
                    'games': len(games),
                    'players': len(players),
                    'feature_records': len(features),
                }
            }

    def test_feature_engineering(self) -> Dict:
        """Test 2: Does feature engineering work?"""
        # Get a random game
        with get_db() as session:
            game = session.query(Game).filter(
                Game.season == self.season
            ).first()

            if not game:
                return {
                    'passed': False,
                    'error': 'No games available for testing'
                }

            # Get a player
            player = session.query(Player).filter(
                Player.position == 'QB'
            ).first()

            if not player:
                return {
                    'passed': False,
                    'error': 'No QB found for testing'
                }

        # Try to engineer features
        try:
            engineer = FeatureEngineer()

            # Test smoothed features
            smoothed = engineer.get_smoothed_features(
                player.player_id,
                n_games=5
            )

            if not smoothed:
                return {
                    'passed': False,
                    'error': 'Smoothed features returned empty'
                }

            # Test matchup features
            matchup = engineer.extract_matchup_features(
                game.game_id,
                player.team
            )

            return {
                'passed': True,
                'details': {
                    'player': player.full_name,
                    'game': game.game_id,
                    'smoothed_features': len(smoothed.__dict__),
                    'matchup_features': bool(matchup),
                }
            }

        except Exception as e:
            return {
                'passed': False,
                'error': f'Feature engineering failed: {str(e)}'
            }

    def test_model_predictions(self) -> Dict:
        """Test 3: Can models generate predictions?"""
        # Get a game and player
        with get_db() as session:
            game = session.query(Game).filter(
                Game.season == self.season
            ).first()

            player = session.query(Player).filter(
                Player.position == 'WR'
            ).first()

            if not game or not player:
                return {
                    'passed': False,
                    'error': 'No game or player found'
                }

        try:
            runner = PropModelRunner()

            # Test a single projection
            projection = runner.generate_projection(
                player_id=player.player_id,
                game_id=game.game_id,
                market='player_rec_yds'
            )

            if not projection:
                return {
                    'passed': False,
                    'error': 'Projection returned None'
                }

            # Check projection has required fields
            required_fields = ['mu', 'sigma', 'model_prob', 'score']
            missing = [f for f in required_fields if getattr(projection, f, None) is None]

            if missing:
                return {
                    'passed': False,
                    'error': f'Missing fields: {missing}'
                }

            return {
                'passed': True,
                'details': {
                    'player': player.full_name,
                    'market': 'player_rec_yds',
                    'projection_mu': projection.mu,
                    'projection_sigma': projection.sigma,
                    'model_prob': projection.model_prob,
                    'score': projection.score,
                }
            }

        except Exception as e:
            return {
                'passed': False,
                'error': f'Model prediction failed: {str(e)}'
            }

    def test_calibration(self) -> Dict:
        """Test 4: Is calibration working?"""
        # Check if we have outcomes and projections
        with get_db() as session:
            projections = session.query(Projection).filter(
                Projection.model_prob.isnot(None)
            ).limit(100).all()

            outcomes = session.query(Outcome).limit(100).all()

            if len(projections) < 10:
                return {
                    'passed': False,
                    'error': f'Only {len(projections)} projections with model_prob found (need 10+)'
                }

            if len(outcomes) < 10:
                return {
                    'passed': False,
                    'error': f'Only {len(outcomes)} outcomes found (need 10+)'
                }

        # Try calibration validation
        try:
            validator = CalibrationValidator()

            # Get one market to test
            market = 'player_rec_yds'

            metrics = validator.evaluate_market(
                market=market,
                season=self.season
            )

            if not metrics:
                return {
                    'passed': False,
                    'error': 'Calibration metrics returned None'
                }

            return {
                'passed': True,
                'details': {
                    'market': market,
                    'brier_score': metrics.brier_score,
                    'log_loss': metrics.log_loss,
                    'roc_auc': metrics.roc_auc,
                    'samples': metrics.n_samples,
                }
            }

        except Exception as e:
            return {
                'passed': False,
                'error': f'Calibration test failed: {str(e)}'
            }

    def test_trend_analysis(self) -> Dict:
        """Test 5: Does trend analysis work?"""
        # Get a player
        with get_db() as session:
            player = session.query(Player).filter(
                Player.position == 'RB'
            ).first()

            if not player:
                return {
                    'passed': False,
                    'error': 'No RB found'
                }

        try:
            analyzer = TrendAnalyzer()

            trends = analyzer.analyze_player_trends(
                player_id=player.player_id,
                market='player_rush_yds',
                n_games=10
            )

            if not trends:
                return {
                    'passed': False,
                    'error': 'Trend analysis returned None'
                }

            return {
                'passed': True,
                'details': {
                    'player': player.full_name,
                    'recent_form': trends.recent_form_score,
                    'streak': trends.streak_count,
                    'consistency': trends.consistency_score,
                    'signal_strength': trends.signal_strength,
                }
            }

        except Exception as e:
            return {
                'passed': False,
                'error': f'Trend analysis failed: {str(e)}'
            }

    def test_recommendations(self) -> Dict:
        """Test 6: Can we generate recommendations?"""
        # Get a game
        with get_db() as session:
            game = session.query(Game).filter(
                Game.season == self.season
            ).first()

            if not game:
                return {
                    'passed': False,
                    'error': 'No games found'
                }

        try:
            scorer = RecommendationScorer()

            recs = scorer.recommend_props(
                game_id=game.game_id,
                limit=5
            )

            if not recs:
                return {
                    'passed': False,
                    'error': 'No recommendations generated'
                }

            # Check recommendation quality
            top_rec = recs[0]

            return {
                'passed': True,
                'details': {
                    'game': game.game_id,
                    'recommendations_count': len(recs),
                    'top_player': top_rec.player_name,
                    'top_market': top_rec.market,
                    'top_score': top_rec.overall_score,
                    'top_confidence': top_rec.confidence,
                }
            }

        except Exception as e:
            return {
                'passed': False,
                'error': f'Recommendations failed: {str(e)}'
            }

    def test_historical_backtest(self) -> Dict:
        """Test 7: Historical backtest performance"""
        # Need completed games with outcomes
        with get_db() as session:
            completed_games = session.query(Game).filter(
                Game.season == self.season,
                Game.game_date < datetime.utcnow()
            ).limit(10).all()

            if not completed_games:
                return {
                    'passed': False,
                    'error': 'No completed games for backtest'
                }

            # Check for outcomes
            outcomes = session.query(Outcome).filter(
                Outcome.game_id.in_([g.game_id for g in completed_games])
            ).all()

            if len(outcomes) < 10:
                return {
                    'passed': False,
                    'error': f'Only {len(outcomes)} outcomes found (need 10+ for backtest)'
                }

        try:
            backtester = BacktestEngine()

            # Run backtest on completed games
            result = backtester.run_backtest(
                start_date=completed_games[0].game_date,
                end_date=completed_games[-1].game_date,
                markets=['player_rec_yds', 'player_rush_yds'],
                simulate_betting=True
            )

            # Check results
            if not result.strategy_results:
                return {
                    'passed': False,
                    'error': 'No strategy results from backtest'
                }

            strat = result.strategy_results

            return {
                'passed': True,
                'details': {
                    'games': result.total_games,
                    'projections': result.total_projections,
                    'total_bets': strat['total_bets'],
                    'win_rate': f"{strat['hit_rate']*100:.1f}%",
                    'roi': f"{strat['roi_percent']:.2f}%",
                    'final_bankroll': f"${strat['final_bankroll']:.2f}",
                    'profit': f"${strat['total_profit']:.2f}",
                }
            }

        except Exception as e:
            return {
                'passed': False,
                'error': f'Backtest failed: {str(e)}'
            }


def main():
    parser = argparse.ArgumentParser(description='Validate NFL prop system')
    parser.add_argument('--season', type=int, default=2024, help='Season to test')
    parser.add_argument('--quick', action='store_true', help='Run quick validation (skip backtest)')

    args = parser.parse_args()

    validator = SystemValidator(season=args.season)

    if args.quick:
        print("Running quick validation (no backtest)...")
        # Run only fast tests
        tests = [
            validator.test_data_availability,
            validator.test_feature_engineering,
            validator.test_model_predictions,
        ]

        for test in tests:
            result = test()
            print(f"{test.__name__}: {'✅ PASS' if result['passed'] else '❌ FAIL'}")
    else:
        results = validator.run_full_validation()

        # Exit code based on results
        failed = sum(1 for r in results.values() if not r['passed'])
        sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
