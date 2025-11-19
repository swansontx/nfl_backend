"""Orchestrator for running the full NFL props pipeline

Coordinates execution of all pipeline stages:
1. Data Ingestion (nflverse, schedules, odds, injuries)
2. Feature Extraction (player PBP features from play-by-play)
3. Feature Engineering (smoothing, rolling windows)
4. Roster/Injury Indexing (build game-level indexes)
5. Model Training (train passing/rushing/receiving models)
6. Prediction Generation (generate prop projections)
7. Backtest/Calibration (validate model accuracy)

Can run full pipeline or individual stages.

Usage:
    # Run full pipeline with training
    python -m backend.orchestration.orchestrator --season 2024 --train

    # Run predictions only (skip data/features)
    python -m backend.orchestration.orchestrator --season 2024 --predict-only

    # Run backtest on historical data
    python -m backend.orchestration.orchestrator --season 2023 --backtest
"""

from pathlib import Path
import argparse
from datetime import datetime
from typing import List, Optional
import subprocess
import sys


class PipelineStage:
    """Represents a single stage in the pipeline."""

    def __init__(self, name: str, script_path: str, args: List[str] = None):
        self.name = name
        self.script_path = script_path
        self.args = args or []
        self.completed = False

    def run(self) -> bool:
        """Execute this pipeline stage.

        Returns:
            True if successful, False otherwise
        """
        print(f"\n{'='*60}")
        print(f"Running stage: {self.name}")
        print(f"{'='*60}")

        try:
            cmd = [sys.executable, self.script_path] + self.args
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(f"Warnings: {result.stderr}")

            self.completed = True
            print(f"✓ Stage '{self.name}' completed successfully")
            return True

        except subprocess.CalledProcessError as e:
            print(f"✗ Stage '{self.name}' failed with error:")
            print(e.stderr)
            return False


class NFLPropsPipeline:
    """Orchestrator for the full NFL props pipeline."""

    def __init__(self, season: int = 2025, week: Optional[int] = None):
        self.season = season
        self.week = week
        self.stages = []
        self._build_pipeline()

    def _build_pipeline(self):
        """Build the pipeline stages."""
        backend_dir = Path(__file__).parent.parent

        # Stage 1: Data Ingestion
        self.stages.append(PipelineStage(
            name="Ingest nflverse data (PBP, player stats, rosters)",
            script_path=str(backend_dir / "ingestion" / "fetch_nflverse.py"),
            args=["--year", str(self.season), "--out", "inputs"]
        ))

        self.stages.append(PipelineStage(
            name="Ingest NFL schedules",
            script_path=str(backend_dir / "ingestion" / "fetch_nflverse_schedules.py"),
            args=["--year", str(self.season), "--out", "inputs"]
        ))

        # Stage 2: Feature Extraction
        self.stages.append(PipelineStage(
            name="Extract player PBP features",
            script_path=str(backend_dir / "features" / "extract_player_pbp_features.py"),
            args=["--pbp", f"inputs/play_by_play_{self.season}.csv",
                  "--out", "outputs/player_pbp_features_by_id.json"]
        ))

        # Stage 3: Feature Engineering
        self.stages.append(PipelineStage(
            name="Apply smoothing and rolling windows",
            script_path=str(backend_dir / "features" / "smoothing_and_rolling.py"),
            args=["--input", "outputs/player_pbp_features_by_id.json",
                  "--output", "outputs/player_features_smoothed.json"]
        ))

        # Stage 4: Roster/Injury Indexing
        self.stages.append(PipelineStage(
            name="Build roster index",
            script_path=str(backend_dir / "roster_injury" / "build_game_roster_index.py"),
            args=["--year", str(self.season),
                  "--source", "inputs",
                  "--output", "outputs"]
        ))

        self.stages.append(PipelineStage(
            name="Build injury index",
            script_path=str(backend_dir / "roster_injury" / "build_injury_game_index.py"),
            args=["--year", str(self.season),
                  "--injuries-dir", "outputs",
                  "--output", "outputs"]
        ))

        # Stage 5: Model Training (optional, controlled by --train flag)
        # Note: Model training scripts to be implemented
        # Would run: backend/modeling/train_passing_model.py, etc.

        # Stage 6: Prediction Generation (for specific week/games)
        # Note: Prediction scripts to be implemented
        # Would run: backend/modeling/generate_predictions.py

        # Stage 7: Backtest/Calibration (optional, controlled by --backtest flag)
        # Note: Backtest framework to be implemented
        # Would run: backend/calib_backtest/run_backtest.py

    def run_full_pipeline(self) -> bool:
        """Run the complete pipeline.

        Returns:
            True if all stages completed successfully
        """
        print(f"\n{'#'*60}")
        print(f"# Starting NFL Props Pipeline")
        print(f"# Season: {self.season}, Week: {self.week or 'N/A'}")
        print(f"# Timestamp: {datetime.now().isoformat()}")
        print(f"{'#'*60}\n")

        for stage in self.stages:
            if not stage.run():
                print(f"\n✗ Pipeline failed at stage: {stage.name}")
                return False

        print(f"\n{'#'*60}")
        print(f"# ✓ Pipeline completed successfully!")
        print(f"# {len(self.stages)} stages executed")
        print(f"{'#'*60}\n")
        return True

    def run_stage(self, stage_name: str) -> bool:
        """Run a specific pipeline stage by name.

        Args:
            stage_name: Name of stage to run

        Returns:
            True if stage completed successfully
        """
        for stage in self.stages:
            if stage.name == stage_name:
                return stage.run()

        print(f"Error: Stage '{stage_name}' not found")
        return False

    def list_stages(self):
        """Print all available pipeline stages."""
        print("Available pipeline stages:")
        for i, stage in enumerate(self.stages, 1):
            status = "✓" if stage.completed else " "
            print(f"  [{status}] {i}. {stage.name}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='NFL Props Pipeline Orchestrator')
    p.add_argument('--season', type=int, default=2024,
                   help='NFL season year (default: 2024)')
    p.add_argument('--week', type=int, default=None,
                   help='Week number for prediction generation')
    p.add_argument('--list-stages', action='store_true',
                   help='List all pipeline stages')
    p.add_argument('--stage', type=str, default=None,
                   help='Run specific stage by name')
    p.add_argument('--train', action='store_true',
                   help='Include model training step (not yet implemented)')
    p.add_argument('--backtest', action='store_true',
                   help='Include backtest validation step (not yet implemented)')
    p.add_argument('--predict-only', action='store_true',
                   help='Skip data ingestion/features, only run predictions (not yet implemented)')
    args = p.parse_args()

    # Create pipeline
    pipeline = NFLPropsPipeline(season=args.season, week=args.week)

    if args.list_stages:
        pipeline.list_stages()
        print(f"\nNotes:")
        print(f"  - Model training: Use --train flag (not yet implemented)")
        print(f"  - Backtest validation: Use --backtest flag (not yet implemented)")
        print(f"  - Predictions only: Use --predict-only flag (not yet implemented)")
    elif args.stage:
        success = pipeline.run_stage(args.stage)
        sys.exit(0 if success else 1)
    else:
        # Run full pipeline
        if args.train:
            print("\n⚠️  Model training requested but not yet implemented")
        if args.backtest:
            print("\n⚠️  Backtest validation requested but not yet implemented")
        if args.predict_only:
            print("\n⚠️  Predict-only mode requested but not yet implemented")

        success = pipeline.run_full_pipeline()
        sys.exit(0 if success else 1)
