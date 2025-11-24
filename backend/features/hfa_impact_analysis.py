"""Analyze home field advantage impact on player props.

This module tests how much HFA affects different positions and prop types
to determine appropriate weighting in models.

Research Questions:
1. How much does HFA affect QB passing yards? (expect: moderate, ~5-10 yards)
2. How much does HFA affect RB rushing yards? (expect: small, ~2-5 yards)
3. How much does HFA affect WR receiving yards? (expect: moderate, ~3-8 yards)
4. Does dome vs outdoor matter more for certain positions?
5. Does travel distance affect positions differently?
"""

from typing import Dict, List, Optional
from pathlib import Path
import csv
from dataclasses import dataclass
import pandas as pd

from backend.features.home_field_advantage import hfa_calculator


@dataclass
class HFAImpact:
    """HFA impact analysis result."""
    position: str
    prop_type: str
    home_avg: float
    away_avg: float
    hfa_effect: float  # Yards/stat difference
    hfa_effect_pct: float  # Percentage difference
    sample_size: int
    confidence: float


class HFAImpactAnalyzer:
    """Analyze HFA impact on player props by position and prop type."""

    # Expected HFA impacts based on historical NFL data (these are estimates)
    # TODO: Replace with actual analysis from your data
    EXPECTED_IMPACTS = {
        'QB': {
            'passing_yards': 8.5,      # ~8-9 yards home advantage
            'passing_tds': 0.15,       # ~0.15 TD advantage
            'completions': 1.2,        # ~1.2 completion advantage
            'interceptions': -0.08     # Fewer INTs at home
        },
        'RB': {
            'rushing_yards': 4.2,      # ~4-5 yards home advantage
            'rushing_tds': 0.08,       # Small TD advantage
            'receptions': 0.3,         # Minimal receiving advantage
            'receiving_yards': 2.1     # Small receiving advantage
        },
        'WR': {
            'receptions': 0.5,         # ~0.5 reception advantage
            'receiving_yards': 6.3,    # ~6-7 yards home advantage
            'receiving_tds': 0.12,     # ~0.12 TD advantage
            'targets': 0.8             # ~0.8 target advantage
        },
        'TE': {
            'receptions': 0.4,         # Similar to WR
            'receiving_yards': 4.8,    # Less than WR
            'receiving_tds': 0.10,     # Similar to WR
            'targets': 0.7
        }
    }

    # Position-specific HFA sensitivity to factors
    HFA_SENSITIVITIES = {
        'QB': {
            'dome_advantage': 0.15,      # 15% boost in dome (no weather)
            'crowd_noise_penalty': 0.10,  # 10% penalty in loud stadiums (road)
            'travel_penalty_weight': 0.12 # 12% penalty per travel unit
        },
        'RB': {
            'dome_advantage': 0.05,      # Small dome advantage
            'crowd_noise_penalty': 0.05, # Less affected by noise
            'travel_penalty_weight': 0.08
        },
        'WR': {
            'dome_advantage': 0.12,      # Moderate dome advantage
            'crowd_noise_penalty': 0.08, # Moderate noise impact
            'travel_penalty_weight': 0.10
        },
        'TE': {
            'dome_advantage': 0.08,
            'crowd_noise_penalty': 0.06,
            'travel_penalty_weight': 0.09
        }
    }

    def get_expected_hfa_adjustment(
        self,
        position: str,
        prop_type: str,
        game_id: str,
        team: str,
        is_home_team: bool
    ) -> Dict[str, float]:
        """Get expected HFA adjustment for a player's prop.

        Args:
            position: Player position (QB, RB, WR, TE)
            prop_type: Prop type (passing_yards, rushing_yards, etc.)
            game_id: Game ID for HFA features
            team: Player's team
            is_home_team: Whether player's team is home

        Returns:
            Dictionary with adjustment details
        """
        # Get base HFA impact for this position/prop
        base_impact = self.EXPECTED_IMPACTS.get(position, {}).get(prop_type, 0.0)

        # Get HFA features for this game
        hfa_features = hfa_calculator.get_all_hfa_features(game_id, team, is_home_team)

        # Get position sensitivities
        sensitivities = self.HFA_SENSITIVITIES.get(position, {})

        # Calculate adjusted impact
        adjusted_impact = base_impact

        if is_home_team:
            # Positive adjustments for home team
            if hfa_features.get('dome_advantage', 0) > 0:
                dome_boost = base_impact * sensitivities.get('dome_advantage', 0)
                adjusted_impact += dome_boost

            # Stadium-specific HFA multiplier
            stadium_mult = hfa_features.get('stadium_hfa_multiplier', 1.0)
            adjusted_impact *= stadium_mult

        else:
            # Negative adjustments for away team
            # Travel penalty
            travel_penalty_pct = hfa_features.get('travel_penalty', 0)
            travel_impact = base_impact * sensitivities.get('travel_penalty_weight', 0) * travel_penalty_pct
            adjusted_impact -= travel_impact

            # Crowd noise penalty (away team)
            stadium_mult = hfa_features.get('stadium_hfa_multiplier', 1.0)
            if stadium_mult > 1.1:  # Loud stadiums (SEA, KC, NO)
                noise_penalty = base_impact * sensitivities.get('crowd_noise_penalty', 0) * (stadium_mult - 1.0)
                adjusted_impact -= noise_penalty

        # Division game familiarity reduces HFA
        if hfa_features.get('is_division_game', 0) > 0:
            familiarity_reduction = hfa_features.get('hfa_familiarity_reduction', 0)
            adjusted_impact *= (1.0 - familiarity_reduction)

        return {
            'base_impact': round(base_impact, 2),
            'adjusted_impact': round(adjusted_impact, 2),
            'adjustment_factors': {
                'is_home': is_home_team,
                'dome_advantage': hfa_features.get('dome_advantage', 0),
                'stadium_multiplier': hfa_features.get('stadium_hfa_multiplier', 1.0),
                'travel_penalty': hfa_features.get('travel_penalty', 0),
                'division_game': hfa_features.get('is_division_game', 0)
            }
        }

    def apply_hfa_to_projection(
        self,
        base_projection: float,
        position: str,
        prop_type: str,
        game_id: str,
        team: str,
        is_home_team: bool
    ) -> Dict[str, float]:
        """Apply HFA adjustment to a base projection.

        Args:
            base_projection: Model's base projection (no HFA)
            position: Player position
            prop_type: Prop type
            game_id: Game ID
            team: Player's team
            is_home_team: Whether home team

        Returns:
            Dictionary with adjusted projection and details
        """
        hfa_adjustment = self.get_expected_hfa_adjustment(
            position, prop_type, game_id, team, is_home_team
        )

        adjusted_projection = base_projection + hfa_adjustment['adjusted_impact']

        return {
            'base_projection': round(base_projection, 2),
            'hfa_adjustment': hfa_adjustment['adjusted_impact'],
            'adjusted_projection': round(adjusted_projection, 2),
            'adjustment_pct': round(
                (hfa_adjustment['adjusted_impact'] / base_projection * 100) if base_projection else 0,
                2
            ),
            'factors': hfa_adjustment['adjustment_factors']
        }

    def compare_home_away_props(
        self,
        position: str,
        prop_type: str,
        base_projection: float,
        game_id: str,
        home_team: str,
        away_team: str
    ) -> Dict:
        """Compare same player's prop if playing home vs away.

        Useful for understanding HFA impact magnitude.
        """
        # Calculate as home team
        as_home = self.apply_hfa_to_projection(
            base_projection, position, prop_type, game_id, home_team, True
        )

        # Calculate as away team (using same game_id but swapped perspective)
        as_away = self.apply_hfa_to_projection(
            base_projection, position, prop_type, game_id, away_team, False
        )

        total_swing = as_home['adjusted_projection'] - as_away['adjusted_projection']

        return {
            'position': position,
            'prop_type': prop_type,
            'base_projection': base_projection,
            'as_home_team': as_home,
            'as_away_team': as_away,
            'total_hfa_swing': round(total_swing, 2),
            'swing_pct': round((total_swing / base_projection * 100) if base_projection else 0, 2)
        }

    def calculate_hfa_impacts_from_data(
        self,
        player_stats_path: Path,
        min_games: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """Calculate actual HFA impacts from historical player stats.

        Args:
            player_stats_path: Path to player stats CSV with home/away indicator
            min_games: Minimum games required for analysis

        Returns:
            Dictionary of position -> prop_type -> HFA impact
        """
        print(f"\n{'='*80}")
        print("CALCULATING HFA IMPACTS FROM ACTUAL DATA")
        print(f"{'='*80}\n")

        if not player_stats_path.exists():
            print(f"Stats file not found: {player_stats_path}")
            return self.EXPECTED_IMPACTS

        df = pd.read_csv(player_stats_path, low_memory=False)

        # Determine home/away if not present
        if 'is_home' not in df.columns:
            # Try to infer from recent_team vs opponent columns
            if 'recent_team' in df.columns and 'opponent_team' in df.columns:
                # This is a simplification - would need game context
                print("Warning: is_home column not found, using position-only analysis")
                return self.EXPECTED_IMPACTS

        calculated_impacts = {}

        # Position mapping
        prop_mappings = {
            'QB': ['passing_yards', 'passing_tds', 'completions', 'interceptions'],
            'RB': ['rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards'],
            'WR': ['receptions', 'receiving_yards', 'receiving_tds', 'targets'],
            'TE': ['receptions', 'receiving_yards', 'receiving_tds', 'targets']
        }

        for position, props in prop_mappings.items():
            pos_df = df[df['position'] == position] if 'position' in df.columns else df

            if len(pos_df) < min_games * 2:
                print(f"  {position}: Insufficient data, using defaults")
                calculated_impacts[position] = self.EXPECTED_IMPACTS.get(position, {})
                continue

            calculated_impacts[position] = {}

            for prop in props:
                if prop not in pos_df.columns:
                    calculated_impacts[position][prop] = self.EXPECTED_IMPACTS.get(position, {}).get(prop, 0)
                    continue

                # Calculate home vs away averages
                home_avg = pos_df[pos_df['is_home'] == 1][prop].mean() if 'is_home' in pos_df.columns else pos_df[prop].mean()
                away_avg = pos_df[pos_df['is_home'] == 0][prop].mean() if 'is_home' in pos_df.columns else pos_df[prop].mean()

                hfa_impact = home_avg - away_avg

                # Use calculated if we have enough data, else use expected
                if not pd.isna(hfa_impact):
                    calculated_impacts[position][prop] = round(hfa_impact, 2)
                    print(f"  {position} {prop}: Home={home_avg:.1f}, Away={away_avg:.1f}, HFA={hfa_impact:+.2f}")
                else:
                    calculated_impacts[position][prop] = self.EXPECTED_IMPACTS.get(position, {}).get(prop, 0)

        print(f"\n✓ Calculated HFA impacts for {len(calculated_impacts)} positions")
        return calculated_impacts

    def update_impacts_from_data(self, player_stats_path: Path) -> None:
        """Update EXPECTED_IMPACTS with calculated values from data.

        Args:
            player_stats_path: Path to player stats CSV
        """
        calculated = self.calculate_hfa_impacts_from_data(player_stats_path)
        self.EXPECTED_IMPACTS.update(calculated)
        print("✓ Updated HFA impacts with data-driven values")

    def export_hfa_tables(self, output_dir: Path) -> None:
        """Export HFA impact tables for documentation.

        Creates CSV files with expected HFA impacts by position.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create table for each position
        for position, impacts in self.EXPECTED_IMPACTS.items():
            output_file = output_dir / f"hfa_impacts_{position.lower()}.csv"

            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['prop_type', 'hfa_impact', 'unit'])

                for prop_type, impact in impacts.items():
                    unit = 'yards' if 'yards' in prop_type else 'count'
                    writer.writerow([prop_type, impact, unit])

            print(f"Exported HFA impacts for {position}: {output_file}")

        # Create sensitivities table
        sens_file = output_dir / "hfa_sensitivities.csv"
        with open(sens_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['position', 'dome_advantage', 'crowd_noise_penalty', 'travel_penalty_weight'])

            for position, sens in self.HFA_SENSITIVITIES.items():
                writer.writerow([
                    position,
                    sens.get('dome_advantage', 0),
                    sens.get('crowd_noise_penalty', 0),
                    sens.get('travel_penalty_weight', 0)
                ])

        print(f"Exported HFA sensitivities: {sens_file}")


# Singleton instance
hfa_impact_analyzer = HFAImpactAnalyzer()
