"""Injury adjustments for prop projections.

This module applies multipliers to projections based on injury status.
Injury data comes from Sleeper API and ESPN (via fetch_injuries.py).

Key insight: Injured players have reduced snap expectations, which directly
impacts their stat projections. This is especially important for:
- Volume props (receptions, carries, targets)
- Yardage props (receiving_yards, rushing_yards)
- TD props (lower opportunity = lower TD probability)
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime


# Injury status multipliers for snap count expectations
# These represent expected reduction in playing time
INJURY_MULTIPLIERS = {
    # Active statuses
    'Active': 1.0,
    'active': 1.0,
    None: 1.0,
    '': 1.0,

    # Questionable - typically plays but may have reduced snaps
    'Questionable': 0.85,
    'Q': 0.85,
    'questionable': 0.85,

    # Doubtful - rarely plays, if so very limited
    'Doubtful': 0.25,
    'D': 0.25,
    'doubtful': 0.25,

    # Out - will not play
    'Out': 0.0,
    'O': 0.0,
    'out': 0.0,

    # IR designations
    'IR': 0.0,
    'IR-R': 0.0,  # IR-Designated for Return
    'PUP': 0.0,   # Physically Unable to Perform
    'NFI': 0.0,   # Non-Football Injury
    'Suspended': 0.0,
    'Reserve/COVID-19': 0.0,

    # Practice statuses (less severe)
    'DNP': 0.5,        # Did Not Practice
    'Limited': 0.75,   # Limited Practice
    'Full': 1.0,       # Full Practice
}

# Body part multipliers - some injuries affect certain stats more
BODY_PART_ADJUSTMENTS = {
    # Leg/lower body injuries affect rushing more
    'Ankle': {'rushing_yards': 0.9, 'rushing_tds': 0.9, 'carries': 0.9},
    'Knee': {'rushing_yards': 0.85, 'rushing_tds': 0.85, 'carries': 0.85},
    'Hamstring': {'rushing_yards': 0.85, 'carries': 0.9},
    'Foot': {'rushing_yards': 0.9, 'carries': 0.9},
    'Calf': {'rushing_yards': 0.9, 'carries': 0.9},
    'Quadricep': {'rushing_yards': 0.85, 'carries': 0.9},
    'Hip': {'rushing_yards': 0.85, 'carries': 0.85},
    'Groin': {'rushing_yards': 0.85, 'carries': 0.85},

    # Upper body injuries affect passing/catching
    'Shoulder': {'passing_yards': 0.9, 'passing_tds': 0.9, 'receiving_yards': 0.9},
    'Hand': {'passing_yards': 0.85, 'receptions': 0.85, 'receiving_yards': 0.85},
    'Finger': {'passing_yards': 0.9, 'receptions': 0.9},
    'Wrist': {'passing_yards': 0.85, 'receptions': 0.85},
    'Elbow': {'passing_yards': 0.85, 'passing_tds': 0.9},
    'Thumb': {'passing_yards': 0.85, 'receptions': 0.85},

    # Ribs affect everything
    'Ribs': {'passing_yards': 0.85, 'rushing_yards': 0.85, 'receiving_yards': 0.85},

    # Head injuries - conservative approach
    'Concussion': {'all': 0.0},  # DNP until cleared
    'Head': {'all': 0.0},
}


class InjuryAdjuster:
    """Adjust projections based on injury data."""

    def __init__(self, injuries_dir: str = "inputs/injuries"):
        self.injuries_dir = Path(injuries_dir)
        self.injuries: Dict[str, Dict] = {}
        self._load_injuries()

    def _load_injuries(self):
        """Load injury data from files."""
        # Try to load from most recent injury file
        injury_files = list(self.injuries_dir.glob("injuries_*.json"))

        if not injury_files:
            # Try alternative locations
            alt_file = Path("inputs/injuries.json")
            if alt_file.exists():
                injury_files = [alt_file]

        if not injury_files:
            print("No injury data found. Run fetch_injuries.py to get latest data.")
            return

        # Load most recent file
        latest_file = max(injury_files, key=lambda f: f.stat().st_mtime)

        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)

            # Index by player name (lowercase for matching)
            for player in data:
                name = player.get('player_name', '').lower()
                if name:
                    self.injuries[name] = player

            print(f"Loaded {len(self.injuries)} player injuries from {latest_file}")

        except Exception as e:
            print(f"Error loading injuries: {e}")

    def get_injury_status(self, player_name: str) -> Optional[Dict]:
        """Get injury status for a player.

        Args:
            player_name: Player name (case-insensitive)

        Returns:
            Injury dict or None if not found/healthy
        """
        name_lower = player_name.lower()

        # Try exact match
        if name_lower in self.injuries:
            return self.injuries[name_lower]

        # Try partial match (handle "Jr.", "III", etc.)
        for key, injury in self.injuries.items():
            if name_lower in key or key in name_lower:
                return injury

        return None

    def get_snap_multiplier(self, player_name: str) -> float:
        """Get snap count multiplier based on injury status.

        Args:
            player_name: Player name

        Returns:
            Multiplier (0.0 to 1.0)
        """
        injury = self.get_injury_status(player_name)

        if not injury:
            return 1.0

        status = injury.get('injury_status') or injury.get('status', '')

        return INJURY_MULTIPLIERS.get(status, 1.0)

    def adjust_projection(
        self,
        player_name: str,
        prop_type: str,
        projection: float,
        std_dev: float
    ) -> Tuple[float, float]:
        """Adjust projection based on injury status.

        Args:
            player_name: Player name
            prop_type: Type of prop (e.g., 'passing_yards')
            projection: Base projection
            std_dev: Base standard deviation

        Returns:
            (adjusted_projection, adjusted_std_dev)
        """
        injury = self.get_injury_status(player_name)

        if not injury:
            return projection, std_dev

        # Get base multiplier from injury status
        status = injury.get('injury_status') or injury.get('status', '')
        base_multiplier = INJURY_MULTIPLIERS.get(status, 1.0)

        if base_multiplier == 0.0:
            # Player is out
            return 0.0, 0.0

        # Get body part adjustment if available
        body_part = injury.get('injury_body_part') or injury.get('body_part', '')
        body_adjustments = BODY_PART_ADJUSTMENTS.get(body_part, {})

        # Check for 'all' modifier (e.g., concussion)
        if 'all' in body_adjustments:
            return 0.0, 0.0

        # Get prop-specific adjustment
        prop_adjustment = body_adjustments.get(prop_type, 1.0)

        # Apply multipliers
        total_multiplier = base_multiplier * prop_adjustment
        adjusted_projection = projection * total_multiplier

        # Increase uncertainty for injured players
        uncertainty_multiplier = 1.0 + (1.0 - total_multiplier) * 0.5
        adjusted_std_dev = std_dev * uncertainty_multiplier

        return round(adjusted_projection, 1), round(adjusted_std_dev, 1)

    def get_injury_report(self) -> Dict:
        """Get summary of all injuries by status.

        Returns:
            Dict with injury counts by status
        """
        status_counts = {}

        for name, injury in self.injuries.items():
            status = injury.get('injury_status') or injury.get('status', 'Unknown')
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            'total_injuries': len(self.injuries),
            'by_status': status_counts,
        }

    def get_questionable_players(self) -> list:
        """Get list of questionable players.

        Returns:
            List of player dicts
        """
        questionable = []

        for name, injury in self.injuries.items():
            status = injury.get('injury_status') or injury.get('status', '')
            if status.lower() in ['questionable', 'q']:
                questionable.append({
                    'name': injury.get('player_name', name),
                    'team': injury.get('team', 'UNK'),
                    'position': injury.get('position', 'UNK'),
                    'body_part': injury.get('injury_body_part', injury.get('body_part', 'UNK')),
                })

        return questionable

    def get_out_players(self) -> list:
        """Get list of players ruled out.

        Returns:
            List of player dicts
        """
        out_players = []

        for name, injury in self.injuries.items():
            status = injury.get('injury_status') or injury.get('status', '')
            if status.lower() in ['out', 'o', 'ir', 'ir-r', 'pup', 'nfi', 'suspended']:
                out_players.append({
                    'name': injury.get('player_name', name),
                    'team': injury.get('team', 'UNK'),
                    'position': injury.get('position', 'UNK'),
                    'status': status,
                })

        return out_players


# Singleton instance
injury_adjuster = InjuryAdjuster()


def adjust_projection_for_injury(
    player_name: str,
    prop_type: str,
    projection: float,
    std_dev: float
) -> Tuple[float, float]:
    """Convenience function to adjust projection for injury.

    Args:
        player_name: Player name
        prop_type: Prop type
        projection: Base projection
        std_dev: Base std dev

    Returns:
        (adjusted_projection, adjusted_std_dev)
    """
    return injury_adjuster.adjust_projection(
        player_name, prop_type, projection, std_dev
    )
