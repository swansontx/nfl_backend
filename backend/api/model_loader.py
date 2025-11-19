"""Load ML model outputs for prop predictions.

This module loads prop projections from the outputs/predictions/ directory
where the ML models write their predictions.

Expected file structure:
    outputs/predictions/props_{game_id}.csv
    outputs/predictions/props_{season}_{week}.csv

CSV Format:
    player_id,player_name,prop_type,projection,std_dev,confidence_lower,confidence_upper
"""

from pathlib import Path
from typing import List, Optional, Dict
import csv
from dataclasses import dataclass

from backend.api.prop_analyzer import PropProjection
from backend.api.cache import cached, CACHE_TTL


@dataclass
class ModelMetadata:
    """Metadata about model predictions."""
    model_version: str
    generated_at: str
    game_id: str
    total_projections: int


class ModelOutputLoader:
    """Load ML model prop projections from file system."""

    def __init__(self, predictions_dir: Path = Path('outputs/predictions')):
        """Initialize model output loader.

        Args:
            predictions_dir: Directory containing prediction CSV files
        """
        self.predictions_dir = predictions_dir

    @cached(ttl_seconds=CACHE_TTL['projections'])  # 30 minutes
    def load_projections_for_game(self, game_id: str) -> List[PropProjection]:
        """Load prop projections for a specific game.

        Args:
            game_id: Game ID in format {season}_{week}_{away}_{home}

        Returns:
            List of PropProjection objects

        Looks for files in this order:
            1. props_{game_id}.csv (exact match)
            2. props_{season}_{week}.csv (all games in that week)
        """
        projections = []

        # Try exact game match first
        game_file = self.predictions_dir / f"props_{game_id}.csv"
        if game_file.exists():
            projections.extend(self._load_csv(game_file))
            return projections

        # Try week-level file
        try:
            parts = game_id.split('_')
            if len(parts) >= 2:
                season = parts[0]
                week = parts[1]
                week_file = self.predictions_dir / f"props_{season}_{week}.csv"

                if week_file.exists():
                    # Load all and filter by game_id if column exists
                    all_projections = self._load_csv(week_file)

                    # Filter for this specific game (if game_id column exists)
                    projections = [
                        proj for proj in all_projections
                        if not hasattr(proj, 'game_id') or proj.game_id == game_id
                    ]
                    return projections

        except Exception as e:
            print(f"Error loading week-level projections: {e}")

        # If no files found, return empty list
        if not projections:
            print(f"No projection files found for {game_id}")

        return projections

    @cached(ttl_seconds=CACHE_TTL['projections'])  # 30 minutes
    def load_projection_for_player(
        self,
        player_id: str,
        prop_type: str,
        game_id: Optional[str] = None
    ) -> Optional[PropProjection]:
        """Load projection for a specific player and prop type.

        Args:
            player_id: Player ID
            prop_type: Type of prop (e.g., 'passing_yards')
            game_id: Optional game ID to narrow search

        Returns:
            PropProjection or None if not found
        """
        if game_id:
            projections = self.load_projections_for_game(game_id)
        else:
            # Load all available projections
            projections = self._load_all_recent_projections()

        # Find matching projection
        for proj in projections:
            if proj.player_id == player_id and proj.prop_type == prop_type:
                return proj

        return None

    def _load_csv(self, file_path: Path) -> List[PropProjection]:
        """Load projections from a CSV file.

        Args:
            file_path: Path to CSV file

        Returns:
            List of PropProjection objects

        Expected CSV columns:
            - player_id (required)
            - player_name (required)
            - prop_type (required)
            - projection (required)
            - std_dev (optional, defaults to projection * 0.15)
            - confidence_lower (optional, calculated if not provided)
            - confidence_upper (optional, calculated if not provided)
            - hit_probability_over (optional, calculated if not provided)
            - hit_probability_under (optional, calculated if not provided)
        """
        projections = []

        try:
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    # Required fields
                    player_id = row.get('player_id')
                    player_name = row.get('player_name')
                    prop_type = row.get('prop_type')
                    projection = row.get('projection')

                    if not all([player_id, player_name, prop_type, projection]):
                        print(f"Skipping row with missing required fields: {row}")
                        continue

                    projection_val = float(projection)

                    # Optional fields with defaults
                    std_dev = float(row.get('std_dev', projection_val * 0.15))

                    # Calculate confidence interval if not provided
                    conf_lower = float(row.get('confidence_lower', projection_val - 1.96 * std_dev))
                    conf_upper = float(row.get('confidence_upper', projection_val + 1.96 * std_dev))

                    # Probabilities (will be calculated by prop_analyzer if not provided)
                    hit_prob_over = float(row.get('hit_probability_over', 0.5))
                    hit_prob_under = float(row.get('hit_probability_under', 0.5))

                    projections.append(PropProjection(
                        player_id=player_id,
                        player_name=player_name,
                        prop_type=prop_type,
                        projection=projection_val,
                        std_dev=std_dev,
                        confidence_interval=(conf_lower, conf_upper),
                        hit_probability_over=hit_prob_over,
                        hit_probability_under=hit_prob_under
                    ))

        except Exception as e:
            print(f"Error loading CSV {file_path}: {e}")

        return projections

    def _load_all_recent_projections(self) -> List[PropProjection]:
        """Load all recent projection files.

        Returns:
            Combined list of all recent projections
        """
        projections = []

        if not self.predictions_dir.exists():
            print(f"Predictions directory not found: {self.predictions_dir}")
            return projections

        # Find all CSV files
        for csv_file in self.predictions_dir.glob('props_*.csv'):
            projections.extend(self._load_csv(csv_file))

        return projections

    def get_available_games(self) -> List[str]:
        """Get list of games with available projections.

        Returns:
            List of game_ids
        """
        games = set()

        if not self.predictions_dir.exists():
            return []

        for csv_file in self.predictions_dir.glob('props_*.csv'):
            # Extract game_id from filename
            filename = csv_file.stem  # e.g., 'props_2025_10_KC_BUF'
            parts = filename.split('_')

            if len(parts) >= 4:
                # Full game_id format
                game_id = '_'.join(parts[1:])  # Remove 'props' prefix
                games.add(game_id)
            elif len(parts) == 3:
                # Week-level file (props_2025_10)
                # We'll list it as the season_week combo
                games.add('_'.join(parts[1:]))

        return sorted(list(games))

    def create_sample_projections(self, game_id: str) -> None:
        """Create sample projections file for development/testing.

        Args:
            game_id: Game ID in format {season}_{week}_{away}_{home}
        """
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

        sample_data = [
            {
                'player_id': 'mahomes_patrick',
                'player_name': 'Patrick Mahomes',
                'prop_type': 'passing_yards',
                'projection': 295.3,
                'std_dev': 42.5,
                'confidence_lower': 252.8,
                'confidence_upper': 337.8,
                'hit_probability_over': 0.68,
                'hit_probability_under': 0.32
            },
            {
                'player_id': 'kelce_travis',
                'player_name': 'Travis Kelce',
                'prop_type': 'receiving_yards',
                'projection': 78.2,
                'std_dev': 18.3,
                'confidence_lower': 59.9,
                'confidence_upper': 96.5,
                'hit_probability_over': 0.55,
                'hit_probability_under': 0.45
            },
            {
                'player_id': 'allen_josh',
                'player_name': 'Josh Allen',
                'prop_type': 'passing_yards',
                'projection': 285.7,
                'std_dev': 38.2,
                'confidence_lower': 248.5,
                'confidence_upper': 322.9,
                'hit_probability_over': 0.62,
                'hit_probability_under': 0.38
            }
        ]

        output_file = self.predictions_dir / f"props_{game_id}.csv"

        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sample_data[0].keys())
            writer.writeheader()
            writer.writerows(sample_data)

        print(f"Created sample projections: {output_file}")


# Singleton instance
model_loader = ModelOutputLoader()
