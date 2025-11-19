"""Orchestrator for running the full NFL props pipeline

Coordinates execution of all pipeline stages:
1. Ingestion (nflverse, odds, injuries)
2. Canonicalization (player/game mapping)
3. Feature extraction
4. Modeling
5. Calibration (if needed)

Can run full pipeline or individual stages.

TODOs:
- Add pipeline DAG/dependency management
- Add error handling and retry logic
- Add logging and monitoring
- Support parallel execution where possible
- Add CLI for running specific stages or full pipeline
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

        # Stage 1: Ingestion
        self.stages.append(PipelineStage(
            name="Ingest nflverse data",
            script_path=str(backend_dir / "ingestion" / "fetch_nflverse.py"),
            args=["--year", str(self.season)]
        ))

        self.stages.append(PipelineStage(
            name="Ingest odds data",
            script_path=str(backend_dir / "ingestion" / "fetch_odds.py"),
            args=[]
        ))

        self.stages.append(PipelineStage(
            name="Ingest injury data",
            script_path=str(backend_dir / "ingestion" / "fetch_injuries.py"),
            args=["--date", datetime.now().strftime('%Y%m%d')]
        ))

        # Stage 2: Feature extraction
        self.stages.append(PipelineStage(
            name="Extract player PBP features",
            script_path=str(backend_dir / "features" / "extract_player_pbp_features.py"),
            args=[]
        ))

        # TODO: Add smoothing/rolling features stage when implemented

        # Stage 3: Roster/injury indexing
        # TODO: Add roster and injury index building when implemented

        # Stage 4: Modeling
        # Note: This would typically run for each upcoming game
        if self.week:
            # Example game_id - would need to build from actual schedule
            game_id = f"{self.season}_{self.week:02d}_KC_BUF"
            self.stages.append(PipelineStage(
                name=f"Run models for week {self.week}",
                script_path=str(backend_dir / "modeling" / "model_runner.py"),
                args=["--game-id", game_id]
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
    p.add_argument('--season', type=int, default=2025,
                   help='NFL season year')
    p.add_argument('--week', type=int, default=None,
                   help='Week number for modeling stage')
    p.add_argument('--list-stages', action='store_true',
                   help='List all pipeline stages')
    p.add_argument('--stage', type=str, default=None,
                   help='Run specific stage by name')
    args = p.parse_args()

    # Create pipeline
    pipeline = NFLPropsPipeline(season=args.season, week=args.week)

    if args.list_stages:
        pipeline.list_stages()
    elif args.stage:
        success = pipeline.run_stage(args.stage)
        sys.exit(0 if success else 1)
    else:
        # Run full pipeline
        success = pipeline.run_full_pipeline()
        sys.exit(0 if success else 1)
