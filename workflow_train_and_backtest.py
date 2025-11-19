#!/usr/bin/env python3
"""Master workflow for training and backtesting NFL prop prediction models.

This script orchestrates the entire pipeline:
1. Ingest nflverse data (PBP + advanced metrics) for multiple seasons
2. Extract player features with EPA, CPOE, success rate, pressure metrics
3. Train XGBoost/LightGBM models for passing yards predictions
4. Run comprehensive backtest with accuracy metrics
5. Generate reports and visualizations

Usage:
    # Full pipeline (ingest + train + backtest)
    python workflow_train_and_backtest.py --seasons 2020 2021 2022 2023 2024

    # Skip ingestion (if data already exists)
    python workflow_train_and_backtest.py --skip-ingestion --seasons 2023 2024

    # Train only (no backtest)
    python workflow_train_and_backtest.py --train-only --seasons 2023

    # Backtest only (requires trained model)
    python workflow_train_and_backtest.py --backtest-only --model-path outputs/models/passing_model_2024.pkl
"""

import argparse
import sys
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import List, Optional

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.ingestion.fetch_nflverse import fetch_nflverse
from backend.features.extract_player_pbp_features import extract_player_features
from backend.modeling.train_passing_model import train_passing_model
from backend.calib_backtest.run_backtest import run_backtest

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('workflow.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class WorkflowRunner:
    """Orchestrates the full training and backtesting workflow."""

    def __init__(self, config: dict):
        """Initialize workflow runner.

        Args:
            config: Configuration dict with paths and parameters
        """
        self.config = config
        self.inputs_dir = Path(config.get('inputs_dir', 'inputs'))
        self.outputs_dir = Path(config.get('outputs_dir', 'outputs'))
        self.cache_dir = Path(config.get('cache_dir', 'cache'))

        # Create directories
        self.inputs_dir.mkdir(exist_ok=True)
        self.outputs_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)

        (self.outputs_dir / 'features').mkdir(exist_ok=True)
        (self.outputs_dir / 'models').mkdir(exist_ok=True)
        (self.outputs_dir / 'backtest').mkdir(exist_ok=True)

    def run_full_pipeline(
        self,
        seasons: List[int],
        skip_ingestion: bool = False,
        train_only: bool = False,
        backtest_only: bool = False,
        model_path: Optional[Path] = None
    ) -> dict:
        """Run the full training and backtesting pipeline.

        Args:
            seasons: List of seasons to process (e.g., [2020, 2021, 2022])
            skip_ingestion: Skip data ingestion step
            train_only: Only train models, skip backtest
            backtest_only: Only run backtest (requires trained model)
            model_path: Path to trained model (required for backtest_only)

        Returns:
            Dict with workflow results and metrics
        """
        workflow_start = datetime.now()
        results = {
            'workflow_start': workflow_start.isoformat(),
            'seasons': seasons,
            'steps_completed': [],
            'errors': []
        }

        logger.info("=" * 80)
        logger.info("NFL PROP PREDICTION MODEL - TRAINING & BACKTEST WORKFLOW")
        logger.info("=" * 80)
        logger.info(f"Seasons: {seasons}")
        logger.info(f"Skip Ingestion: {skip_ingestion}")
        logger.info(f"Train Only: {train_only}")
        logger.info(f"Backtest Only: {backtest_only}")
        logger.info("=" * 80)

        try:
            # Step 1: Ingest nflverse data
            if not skip_ingestion and not backtest_only:
                logger.info("\n[STEP 1/4] Ingesting nflverse data...")
                self._ingest_data(seasons)
                results['steps_completed'].append('ingestion')
            else:
                logger.info("\n[STEP 1/4] Skipping ingestion (using existing data)")

            # Step 2: Extract features
            if not backtest_only:
                logger.info("\n[STEP 2/4] Extracting player features with advanced metrics...")
                feature_files = self._extract_features(seasons)
                results['feature_files'] = [str(f) for f in feature_files]
                results['steps_completed'].append('feature_extraction')
            else:
                logger.info("\n[STEP 2/4] Skipping feature extraction (backtest only mode)")

            # Step 3: Train models
            if not backtest_only:
                logger.info("\n[STEP 3/4] Training passing yards prediction model...")
                model_info = self._train_models(seasons)
                results['model_info'] = model_info
                results['steps_completed'].append('training')
                trained_model_path = Path(model_info['model_path'])
            else:
                if not model_path:
                    raise ValueError("model_path required for backtest_only mode")
                logger.info("\n[STEP 3/4] Skipping training (using existing model)")
                trained_model_path = model_path

            # Step 4: Run backtest
            if not train_only:
                logger.info("\n[STEP 4/4] Running comprehensive backtest...")
                backtest_results = self._run_backtest(seasons, trained_model_path)
                results['backtest_results'] = backtest_results
                results['steps_completed'].append('backtest')
            else:
                logger.info("\n[STEP 4/4] Skipping backtest (train only mode)")

            # Workflow complete
            workflow_end = datetime.now()
            duration = (workflow_end - workflow_start).total_seconds()

            results['workflow_end'] = workflow_end.isoformat()
            results['duration_seconds'] = duration
            results['status'] = 'success'

            logger.info("\n" + "=" * 80)
            logger.info("WORKFLOW COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Duration: {duration:.1f} seconds")
            logger.info(f"Steps Completed: {', '.join(results['steps_completed'])}")

            # Save workflow summary
            summary_file = self.outputs_dir / 'workflow_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Workflow summary saved to: {summary_file}")

            return results

        except Exception as e:
            logger.error(f"Workflow failed: {e}", exc_info=True)
            results['status'] = 'failed'
            results['errors'].append(str(e))
            raise

    def _ingest_data(self, seasons: List[int]):
        """Ingest nflverse data for all seasons.

        Args:
            seasons: List of seasons to ingest
        """
        for season in seasons:
            logger.info(f"  Ingesting {season} season...")
            try:
                fetch_nflverse(
                    year=season,
                    out_dir=self.inputs_dir,
                    cache_dir=self.cache_dir,
                    include_all=True  # Get all advanced metrics
                )
                logger.info(f"  ✓ {season} season ingested successfully")
            except Exception as e:
                logger.error(f"  ✗ Failed to ingest {season}: {e}")
                raise

    def _extract_features(self, seasons: List[int]) -> List[Path]:
        """Extract player features from play-by-play data.

        Args:
            seasons: List of seasons to process

        Returns:
            List of feature file paths
        """
        feature_files = []

        for season in seasons:
            logger.info(f"  Extracting features for {season}...")

            pbp_file = self.inputs_dir / f"{season}_play_by_play.parquet"
            if not pbp_file.exists():
                logger.warning(f"  ⚠️  Play-by-play file not found: {pbp_file}")
                continue

            roster_file = self.inputs_dir / f"{season}_weekly_rosters.parquet"
            output_file = self.outputs_dir / 'features' / f"{season}_player_features.json"

            try:
                extract_player_features(
                    pbp_file=pbp_file,
                    roster_file=roster_file if roster_file.exists() else None,
                    output_file=output_file
                )
                feature_files.append(output_file)
                logger.info(f"  ✓ Features extracted: {output_file}")
            except Exception as e:
                logger.error(f"  ✗ Failed to extract features for {season}: {e}")
                raise

        return feature_files

    def _train_models(self, seasons: List[int]) -> dict:
        """Train passing yards prediction model.

        Args:
            seasons: List of seasons (will use latest for training)

        Returns:
            Dict with model info
        """
        # Use latest season for training
        train_season = max(seasons)
        logger.info(f"  Training on {train_season} season data...")

        features_file = self.outputs_dir / 'features' / f"{train_season}_player_features.json"
        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found: {features_file}")

        model_output = self.outputs_dir / 'models' / f"passing_model_{train_season}.pkl"
        metrics_output = self.outputs_dir / 'models' / f"passing_model_{train_season}_metrics.json"

        try:
            metrics = train_passing_model(
                season=train_season,
                features_file=features_file,
                output_model_path=model_output,
                model_type=self.config.get('model_type', 'xgboost')
            )

            # Save training metrics
            with open(metrics_output, 'w') as f:
                json.dump(metrics, f, indent=2)

            logger.info(f"  ✓ Model trained: {model_output}")
            logger.info(f"  ✓ Val RMSE: {metrics.get('val_rmse', 0):.2f}")
            logger.info(f"  ✓ Val R²: {metrics.get('val_r2', 0):.3f}")

            return {
                'model_path': str(model_output),
                'metrics_path': str(metrics_output),
                'metrics': metrics
            }

        except Exception as e:
            logger.error(f"  ✗ Training failed: {e}")
            raise

    def _run_backtest(self, seasons: List[int], model_path: Path) -> dict:
        """Run backtest on trained model.

        Args:
            seasons: List of seasons to backtest
            model_path: Path to trained model

        Returns:
            Dict with backtest results
        """
        # Use latest season for backtest
        backtest_season = max(seasons)
        logger.info(f"  Backtesting on {backtest_season} season...")

        features_file = self.outputs_dir / 'features' / f"{backtest_season}_player_features.json"
        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found: {features_file}")

        # For actuals, we'll use the same features file (it contains actual yards)
        actuals_file = features_file

        report_output = self.outputs_dir / 'backtest' / f"backtest_report_{backtest_season}.json"

        try:
            results = run_backtest(
                season=backtest_season,
                model_path=model_path,
                features_file=features_file,
                actuals_file=actuals_file,
                output_report=report_output,
                weeks=None  # All weeks
            )

            logger.info(f"  ✓ Backtest complete: {report_output}")
            logger.info(f"  ✓ Test RMSE: {results.get('accuracy_metrics', {}).get('rmse', 0):.2f}")
            logger.info(f"  ✓ Test MAE: {results.get('accuracy_metrics', {}).get('mae', 0):.2f}")
            logger.info(f"  ✓ Test R²: {results.get('accuracy_metrics', {}).get('r_squared', 0):.3f}")

            return results

        except Exception as e:
            logger.error(f"  ✗ Backtest failed: {e}")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='NFL Prop Prediction Model - Training & Backtest Workflow'
    )
    parser.add_argument('--seasons', type=int, nargs='+', required=True,
                       help='Seasons to process (e.g., 2020 2021 2022 2023)')
    parser.add_argument('--skip-ingestion', action='store_true',
                       help='Skip data ingestion (use existing data)')
    parser.add_argument('--train-only', action='store_true',
                       help='Only train models, skip backtest')
    parser.add_argument('--backtest-only', action='store_true',
                       help='Only run backtest (requires --model-path)')
    parser.add_argument('--model-path', type=Path,
                       help='Path to trained model (for backtest-only mode)')
    parser.add_argument('--model-type', default='xgboost',
                       choices=['xgboost', 'lightgbm'],
                       help='Model type to use (default: xgboost)')
    parser.add_argument('--inputs-dir', type=Path, default=Path('inputs'),
                       help='Input data directory (default: inputs/)')
    parser.add_argument('--outputs-dir', type=Path, default=Path('outputs'),
                       help='Output directory (default: outputs/)')
    parser.add_argument('--cache-dir', type=Path, default=Path('cache'),
                       help='Cache directory (default: cache/)')

    args = parser.parse_args()

    # Validate arguments
    if args.backtest_only and not args.model_path:
        parser.error("--backtest-only requires --model-path")

    # Build config
    config = {
        'model_type': args.model_type,
        'inputs_dir': str(args.inputs_dir),
        'outputs_dir': str(args.outputs_dir),
        'cache_dir': str(args.cache_dir)
    }

    # Run workflow
    runner = WorkflowRunner(config)
    results = runner.run_full_pipeline(
        seasons=args.seasons,
        skip_ingestion=args.skip_ingestion,
        train_only=args.train_only,
        backtest_only=args.backtest_only,
        model_path=args.model_path
    )

    # Print summary
    print("\n" + "=" * 80)
    print("WORKFLOW SUMMARY")
    print("=" * 80)
    print(f"Status: {results['status']}")
    print(f"Duration: {results.get('duration_seconds', 0):.1f}s")
    print(f"Steps: {', '.join(results['steps_completed'])}")

    if 'backtest_results' in results:
        metrics = results['backtest_results'].get('accuracy_metrics', {})
        print(f"\nBacktest Accuracy:")
        print(f"  RMSE: {metrics.get('rmse', 0):.2f} yards")
        print(f"  MAE: {metrics.get('mae', 0):.2f} yards")
        print(f"  R²: {metrics.get('r_squared', 0):.3f}")

    print("=" * 80)


if __name__ == '__main__':
    main()
