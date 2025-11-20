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

    def _add_training_stages(self):
        """Add model training stages to pipeline."""
        backend_dir = Path(__file__).parent.parent

        # Stage 5a: Train multi-prop models (main training)
        self.stages.append(PipelineStage(
            name="Train multi-prop models (all prop types)",
            script_path=str(backend_dir / "modeling" / "train_multi_prop_models.py"),
            args=["--season", str(self.season),
                  "--output-dir", "outputs/models"]
        ))

        # Stage 5b: Train passing model (detailed QB model)
        self.stages.append(PipelineStage(
            name="Train passing model (QB projections)",
            script_path=str(backend_dir / "modeling" / "train_passing_model.py"),
            args=["--year", str(self.season),
                  "--output", "outputs/models"]
        ))

        # Stage 5c: Train usage efficiency models
        self.stages.append(PipelineStage(
            name="Train usage efficiency models",
            script_path=str(backend_dir / "modeling" / "train_usage_efficiency_models.py"),
            args=["--season", str(self.season)]
        ))

        # Stage 5d: Train quantile models (confidence intervals)
        self.stages.append(PipelineStage(
            name="Train quantile models (confidence intervals)",
            script_path=str(backend_dir / "modeling" / "train_quantile_models.py"),
            args=["--season", str(self.season)]
        ))

    def _add_prediction_stages(self):
        """Add prediction generation stages to pipeline."""
        backend_dir = Path(__file__).parent.parent

        # Stage 6: Generate projections for upcoming games
        week_args = ["--week", str(self.week)] if self.week else []
        self.stages.append(PipelineStage(
            name=f"Generate projections for week {self.week or 'current'}",
            script_path=str(backend_dir / "modeling" / "generate_projections.py"),
            args=["--season", str(self.season),
                  "--output", "outputs/projections"] + week_args
        ))

    def _add_backtest_stages(self):
        """Add backtest/calibration stages to pipeline."""
        backend_dir = Path(__file__).parent.parent

        # Stage 7a: Run enhanced backtest
        self.stages.append(PipelineStage(
            name="Run enhanced backtest validation",
            script_path=str(backend_dir / "calib_backtest" / "run_enhanced_backtest.py"),
            args=["--season", str(self.season),
                  "--output", "outputs/backtest"]
        ))

        # Stage 7b: Calibrate probabilities
        self.stages.append(PipelineStage(
            name="Calibrate probabilities",
            script_path=str(backend_dir / "calib_backtest" / "calibrate.py"),
            args=["--season", str(self.season)]
        ))

    def _add_recommendation_stages(self, team1: str, team2: str):
        """Add recommendation/picks generation stages to pipeline."""
        backend_dir = Path(__file__).parent.parent

        # Stage 8: Generate game picks with narratives and justifications
        self.stages.append(PipelineStage(
            name=f"Generate value picks for {team1} @ {team2}",
            script_path=str(backend_dir / "recommendations" / "generate_game_picks.py"),
            args=["--team1", team1, "--team2", team2]
        ))

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
                   help='Include model training stages')
    p.add_argument('--backtest', action='store_true',
                   help='Include backtest validation stages')
    p.add_argument('--predict', action='store_true',
                   help='Include prediction generation stage')
    p.add_argument('--predict-only', action='store_true',
                   help='Skip data ingestion/features, only run predictions')
    p.add_argument('--full', action='store_true',
                   help='Run full pipeline (data + train + predict + backtest)')
    p.add_argument('--picks', action='store_true',
                   help='Generate game picks after predictions')
    p.add_argument('--team1', type=str, default=None,
                   help='Away team abbreviation (e.g., HOU)')
    p.add_argument('--team2', type=str, default=None,
                   help='Home team abbreviation (e.g., BUF)')
    args = p.parse_args()

    # Create pipeline
    pipeline = NFLPropsPipeline(season=args.season, week=args.week)

    # Add optional stages based on flags
    if args.full or args.train:
        pipeline._add_training_stages()
    if args.full or args.predict or args.predict_only:
        pipeline._add_prediction_stages()
    if args.full or args.backtest:
        pipeline._add_backtest_stages()
    if args.picks and args.team1 and args.team2:
        pipeline._add_recommendation_stages(args.team1, args.team2)

    if args.list_stages:
        pipeline.list_stages()
        print(f"\nUsage examples:")
        print(f"  python -m backend.orchestration.orchestrator --season 2024")
        print(f"  python -m backend.orchestration.orchestrator --season 2024 --train")
        print(f"  python -m backend.orchestration.orchestrator --season 2024 --full")
        print(f"  python -m backend.orchestration.orchestrator --season 2024 --week 12 --predict")
        print(f"  python -m backend.orchestration.orchestrator --season 2024 --backtest")
    elif args.stage:
        success = pipeline.run_stage(args.stage)
        sys.exit(0 if success else 1)
    elif args.predict_only:
        # Only run prediction stage (skip data ingestion)
        print(f"\n{'#'*60}")
        print(f"# Running prediction-only mode")
        print(f"# Season: {args.season}, Week: {args.week or 'current'}")
        print(f"{'#'*60}\n")

        # Find and run only prediction stage
        for stage in pipeline.stages:
            if "Generate projections" in stage.name:
                success = stage.run()
                sys.exit(0 if success else 1)

        print("Error: Prediction stage not found. Use --predict flag.")
        sys.exit(1)
    else:
        # Run full pipeline
        success = pipeline.run_full_pipeline()
        sys.exit(0 if success else 1)
