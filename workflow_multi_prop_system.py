#!/usr/bin/env python3
"""Master workflow for multi-prop prediction system.

This script orchestrates the complete pipeline for training and deploying ALL 60+ prop models:

PHASE 1: Data Ingestion & Feature Extraction
- Ingest nflverse data (PBP + rosters)
- Extract player features with advanced metrics (EPA, CPOE, success rate, etc.)
- Track DNP/inactive instances via is_active field

PHASE 2: Injury Data Integration
- Fetch injury data from nflverse
- Merge into player features
- Enhance DNP tracking with actual injury statuses

PHASE 3: Multi-Prop Model Training
- Train separate models for ALL 60+ prop types
- Full game props: passing, rushing, receiving, kicking, defense, TDs
- Quarter/half props: 1H, 1Q, 2H, 3Q, 4Q with proportional modeling
- Combo props: pass+rush yards, etc.

PHASE 4: Comprehensive Backtesting
- Test all models on historical data
- Identify best/worst performing markets
- Generate deployment recommendations

PHASE 5: Value Detection
- Generate projections for upcoming games
- Compare vs current DraftKings odds
- Find +EV betting opportunities

Usage:
    # Full pipeline (train all models + backtest)
    python workflow_multi_prop_system.py --season 2024

    # Skip data ingestion (use existing data)
    python workflow_multi_prop_system.py --season 2024 --skip-ingestion

    # Train only (no backtest)
    python workflow_multi_prop_system.py --season 2024 --train-only

    # Backtest only (requires trained models)
    python workflow_multi_prop_system.py --season 2024 --backtest-only
"""

import argparse
import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import List

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.ingestion.fetch_nflverse import fetch_nflverse
from backend.ingestion.fetch_injury_data import fetch_injury_data, merge_injury_into_features
from backend.features.extract_player_pbp_features import extract_features
from backend.modeling.train_multi_prop_models import train_multi_prop_models
from backend.calib_backtest.run_comprehensive_backtest import run_comprehensive_backtest
from backend.analysis.detect_prop_value import detect_prop_value

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('workflow_multi_prop.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MultiPropWorkflowRunner:
    """Orchestrates the full multi-prop system workflow."""

    def __init__(self, config: dict):
        self.config = config
        self.inputs_dir = Path(config.get('inputs_dir', 'inputs'))
        self.outputs_dir = Path(config.get('outputs_dir', 'outputs'))
        self.cache_dir = Path(config.get('cache_dir', 'cache'))

        # Create directories
        for dir_path in [
            self.inputs_dir,
            self.outputs_dir,
            self.cache_dir,
            self.outputs_dir / 'features',
            self.outputs_dir / 'models' / 'multi_prop',
            self.outputs_dir / 'injuries',
            self.outputs_dir / 'backtest' / 'comprehensive',
            self.outputs_dir / 'analysis'
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def run_full_pipeline(
        self,
        season: int,
        skip_ingestion: bool = False,
        train_only: bool = False,
        backtest_only: bool = False
    ) -> dict:
        """Run the full multi-prop pipeline.

        Args:
            season: Season year
            skip_ingestion: Skip data ingestion
            train_only: Only train, skip backtest
            backtest_only: Only backtest (requires trained models)

        Returns:
            Results dict
        """
        workflow_start = datetime.now()
        results = {
            'workflow_start': workflow_start.isoformat(),
            'season': season,
            'steps_completed': [],
            'errors': []
        }

        logger.info("=" * 80)
        logger.info("MULTI-PROP PREDICTION SYSTEM - FULL WORKFLOW")
        logger.info("=" * 80)
        logger.info(f"Season: {season}")
        logger.info(f"Skip Ingestion: {skip_ingestion}")
        logger.info(f"Train Only: {train_only}")
        logger.info(f"Backtest Only: {backtest_only}")
        logger.info("=" * 80)

        try:
            # PHASE 1: Data Ingestion & Feature Extraction
            if not skip_ingestion and not backtest_only:
                logger.info("\n[PHASE 1/5] Data Ingestion & Feature Extraction...")
                self._phase1_ingestion_and_features(season)
                results['steps_completed'].append('phase1_ingestion_features')
            else:
                logger.info("\n[PHASE 1/5] Skipping ingestion & features (using existing data)")

            # PHASE 2: Injury Data Integration
            if not backtest_only:
                logger.info("\n[PHASE 2/5] Injury Data Integration...")
                self._phase2_injury_integration(season)
                results['steps_completed'].append('phase2_injury_integration')
            else:
                logger.info("\n[PHASE 2/5] Skipping injury integration (backtest only mode)")

            # PHASE 3: Multi-Prop Model Training
            if not backtest_only:
                logger.info("\n[PHASE 3/5] Multi-Prop Model Training...")
                training_results = self._phase3_train_all_models(season)
                results['training_results'] = training_results
                results['steps_completed'].append('phase3_training')
            else:
                logger.info("\n[PHASE 3/5] Skipping training (backtest only mode)")

            # PHASE 4: Comprehensive Backtesting
            if not train_only:
                logger.info("\n[PHASE 4/5] Comprehensive Backtesting...")
                backtest_results = self._phase4_comprehensive_backtest(season)
                results['backtest_results'] = backtest_results
                results['steps_completed'].append('phase4_backtest')
            else:
                logger.info("\n[PHASE 4/5] Skipping backtest (train only mode)")

            # PHASE 5: Value Detection (optional, requires current odds)
            if not train_only and not backtest_only:
                logger.info("\n[PHASE 5/5] Value Detection (skipped - requires current odds)")
                # This would be run separately when needed:
                # python -m backend.analysis.detect_prop_value --current-odds outputs/prop_lines/snapshot.json
            else:
                logger.info("\n[PHASE 5/5] Value Detection (skipped)")

            # Workflow complete
            workflow_end = datetime.now()
            duration = (workflow_end - workflow_start).total_seconds()

            results['workflow_end'] = workflow_end.isoformat()
            results['duration_seconds'] = duration
            results['status'] = 'success'

            logger.info("\n" + "=" * 80)
            logger.info("WORKFLOW COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            logger.info(f"Steps Completed: {', '.join(results['steps_completed'])}")
            logger.info("=" * 80)

            return results

        except Exception as e:
            logger.error(f"Workflow failed: {e}", exc_info=True)
            results['status'] = 'failed'
            results['errors'].append(str(e))
            raise

    def _phase1_ingestion_and_features(self, season: int):
        """Phase 1: Ingest data and extract features."""
        logger.info(f"  [1.1] Ingesting nflverse data for {season}...")
        fetch_nflverse(
            year=season,
            out_dir=self.inputs_dir,
            cache_dir=self.cache_dir,
            include_all=True
        )

        logger.info(f"  [1.2] Extracting player features...")
        pbp_file = self.inputs_dir / f"{season}_play_by_play.parquet"
        roster_file = self.inputs_dir / f"{season}_weekly_rosters.parquet"
        output_file = self.outputs_dir / 'features' / f"{season}_player_features.json"

        extract_features(
            pbp_csv=pbp_file,
            out_json=output_file,
            player_stats_csv=roster_file if roster_file.exists() else None
        )

        logger.info(f"  ✓ Phase 1 complete")

    def _phase2_injury_integration(self, season: int):
        """Phase 2: Fetch and merge injury data."""
        logger.info(f"  [2.1] Fetching injury data...")
        injury_map = fetch_injury_data(
            season=season,
            output_dir=self.outputs_dir / 'injuries',
            cache_dir=self.cache_dir
        )

        logger.info(f"  [2.2] Merging injury data into features...")
        features_file = self.outputs_dir / 'features' / f"{season}_player_features.json"
        injury_file = self.outputs_dir / 'injuries' / f"{season}_injuries.json"
        merged_output = self.outputs_dir / 'features' / f"{season}_player_features_with_injuries.json"

        if features_file.exists():
            merge_injury_into_features(
                features_file=features_file,
                injury_file=injury_file,
                output_file=merged_output
            )

        logger.info(f"  ✓ Phase 2 complete")

    def _phase3_train_all_models(self, season: int) -> dict:
        """Phase 3: Train all 60+ prop models."""
        logger.info(f"  [3.1] Training all multi-prop models...")

        # Use merged features with injury data
        features_file = self.outputs_dir / 'features' / f"{season}_player_features_with_injuries.json"
        if not features_file.exists():
            features_file = self.outputs_dir / 'features' / f"{season}_player_features.json"

        output_dir = self.outputs_dir / 'models' / 'multi_prop'

        training_results = train_multi_prop_models(
            season=season,
            features_file=features_file,
            output_dir=output_dir,
            model_type=self.config.get('model_type', 'xgboost')
        )

        logger.info(f"  ✓ Phase 3 complete")
        logger.info(f"  ✓ Trained {len([r for r in training_results.values() if 'error' not in r])}/{len(training_results)} models")

        return training_results

    def _phase4_comprehensive_backtest(self, season: int) -> dict:
        """Phase 4: Run comprehensive backtest on all models."""
        logger.info(f"  [4.1] Running comprehensive backtest...")

        # Use merged features with injury data
        features_file = self.outputs_dir / 'features' / f"{season}_player_features_with_injuries.json"
        if not features_file.exists():
            features_file = self.outputs_dir / 'features' / f"{season}_player_features.json"

        injury_file = self.outputs_dir / 'injuries' / f"{season}_injuries.json"
        models_dir = self.outputs_dir / 'models' / 'multi_prop'
        output_dir = self.outputs_dir / 'backtest' / 'comprehensive'

        backtest_results = run_comprehensive_backtest(
            season=season,
            models_dir=models_dir,
            features_file=features_file,
            injury_file=injury_file if injury_file.exists() else None,
            output_dir=output_dir,
            min_r2=self.config.get('min_r2', 0.3),
            min_samples=self.config.get('min_samples', 10)
        )

        logger.info(f"  ✓ Phase 4 complete")
        logger.info(f"  ✓ Pass Rate: {backtest_results['pass_rate']*100:.1f}%")

        return backtest_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Multi-Prop Prediction System - Full Workflow'
    )
    parser.add_argument('--season', type=int, required=True,
                       help='Season year (e.g., 2024)')
    parser.add_argument('--skip-ingestion', action='store_true',
                       help='Skip data ingestion (use existing data)')
    parser.add_argument('--train-only', action='store_true',
                       help='Only train models, skip backtest')
    parser.add_argument('--backtest-only', action='store_true',
                       help='Only run backtest (requires trained models)')
    parser.add_argument('--model-type', default='xgboost',
                       choices=['xgboost', 'lightgbm'],
                       help='Model type (default: xgboost)')
    parser.add_argument('--min-r2', type=float, default=0.3,
                       help='Minimum R² threshold for deployment (default: 0.3)')
    parser.add_argument('--min-samples', type=int, default=10,
                       help='Minimum samples per model (default: 10)')
    parser.add_argument('--inputs-dir', type=Path, default=Path('inputs'),
                       help='Input data directory')
    parser.add_argument('--outputs-dir', type=Path, default=Path('outputs'),
                       help='Output directory')
    parser.add_argument('--cache-dir', type=Path, default=Path('cache'),
                       help='Cache directory')

    args = parser.parse_args()

    # Build config
    config = {
        'model_type': args.model_type,
        'min_r2': args.min_r2,
        'min_samples': args.min_samples,
        'inputs_dir': str(args.inputs_dir),
        'outputs_dir': str(args.outputs_dir),
        'cache_dir': str(args.cache_dir)
    }

    # Run workflow
    runner = MultiPropWorkflowRunner(config)
    results = runner.run_full_pipeline(
        season=args.season,
        skip_ingestion=args.skip_ingestion,
        train_only=args.train_only,
        backtest_only=args.backtest_only
    )

    # Print final summary
    print("\n" + "=" * 80)
    print("WORKFLOW SUMMARY")
    print("=" * 80)
    print(f"Status: {results['status']}")
    print(f"Duration: {results.get('duration_seconds', 0):.1f}s ({results.get('duration_seconds', 0)/60:.1f} minutes)")
    print(f"Steps: {', '.join(results['steps_completed'])}")

    if 'backtest_results' in results:
        br = results['backtest_results']
        print(f"\nBacktest Results:")
        print(f"  Models Tested: {br['total_models']}")
        print(f"  Passed: {br['passed_models']} ({br['pass_rate']*100:.1f}%)")
        print(f"  Failed: {br['failed_models']}")
        print(f"  Avg R²: {br['summary']['avg_r2']:.3f}")

    print("=" * 80)


if __name__ == '__main__':
    main()
