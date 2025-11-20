"""Fetch NFL injury reports from Sleeper API.

Downloads official injury reports (Out, Doubtful, Questionable) for all teams.
Critical for adjusting predictions based on player availability.

Data Sources:
1. Sleeper API (primary) - All NFL players with injury status
2. ESPN API (deprecated) - Was blocked, kept as reference

Injury Statuses:
- OUT: Player will not play
- DOUBTFUL: ~25% chance of playing
- QUESTIONABLE: ~50% chance of playing
- IR: Player on Injured Reserve

Output: JSON files with injury reports by team and week
"""

from pathlib import Path
import argparse
import requests
import json
from typing import Dict, List, Optional
from datetime import datetime
import time


class InjuryFetcher:
    """Fetches NFL injury reports from Sleeper API."""

    def __init__(self):
        """Initialize injury fetcher."""
        self.sleeper_api_base = "https://api.sleeper.app/v1"

        # NFL team abbreviations
        self.nfl_teams = [
            'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
            'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
            'LAC', 'LAR', 'LV', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
            'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS'
        ]

        # Cache for all players data
        self._players_cache = None

    def fetch_all_players(self) -> Dict:
        """Fetch all NFL players from Sleeper API.

        Returns:
            Dictionary of player_id -> player data
        """
        if self._players_cache is not None:
            return self._players_cache

        url = f"{self.sleeper_api_base}/players/nfl"

        try:
            print("Fetching all players from Sleeper API...")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            self._players_cache = response.json()
            print(f"✓ Retrieved {len(self._players_cache)} players")
            return self._players_cache

        except requests.exceptions.RequestException as e:
            print(f"✗ Error fetching players from Sleeper: {e}")
            print("  Note: If running in a restricted environment, try running locally.")
            return {}

    def fetch_team_injuries(self, team_abbr: str, all_players: Optional[Dict] = None) -> List[Dict]:
        """Get injury report for a specific team from cached data.

        Args:
            team_abbr: Team abbreviation (e.g., 'KC', 'BUF')
            all_players: Optional pre-fetched players dict

        Returns:
            List of injury dictionaries
        """
        if all_players is None:
            all_players = self.fetch_all_players()

        if not all_players:
            return []

        injuries = []

        for player_id, player in all_players.items():
            # Skip if not on this team
            if player.get('team') != team_abbr:
                continue

            # Check for injury status
            injury_status = player.get('injury_status')
            status = player.get('status', '')

            # Include players with injury designation or on IR
            has_injury = injury_status and injury_status.lower() not in ['', 'null', 'none']
            on_ir = status == 'Injured Reserve'

            if has_injury or on_ir:
                # Normalize injury status
                if on_ir and not has_injury:
                    normalized_status = 'IR'
                else:
                    # Normalize to uppercase
                    normalized_status = injury_status.upper() if injury_status else 'IR'
                    # Handle variations
                    if normalized_status in ['OUT', 'O']:
                        normalized_status = 'OUT'
                    elif normalized_status in ['DOUBTFUL', 'D']:
                        normalized_status = 'DOUBTFUL'
                    elif normalized_status in ['QUESTIONABLE', 'Q']:
                        normalized_status = 'QUESTIONABLE'
                    elif normalized_status in ['PROBABLE', 'P']:
                        normalized_status = 'PROBABLE'

                # Get player info
                first_name = player.get('first_name', '')
                last_name = player.get('last_name', '')
                player_name = f"{first_name} {last_name}".strip()

                injuries.append({
                    'player_id': player_id,
                    'player_name': player_name,
                    'team': team_abbr,
                    'position': player.get('position', ''),
                    'injury_status': normalized_status,
                    'injury_type': player.get('injury_body_part', ''),
                    'description': player.get('injury_notes', ''),
                    'practice_status': player.get('practice_participation', ''),
                    'injury_start_date': player.get('injury_start_date'),
                    'roster_status': status,
                    'timestamp': datetime.now().isoformat()
                })

        return injuries

    def fetch_all_injuries(self, output_dir: Path, week: Optional[int] = None) -> Dict:
        """Fetch injury reports for all NFL teams.

        Args:
            output_dir: Output directory
            week: Week number (for filename)

        Returns:
            Summary dict
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Fetching NFL Injury Reports from Sleeper API")
        print(f"{'='*60}\n")

        # Fetch all players once
        all_players = self.fetch_all_players()

        if not all_players:
            print("✗ Failed to fetch player data")
            return {'error': 'Failed to fetch player data'}

        all_injuries = {}
        total_injuries = 0

        for team in self.nfl_teams:
            print(f"Processing {team}...", end=' ')

            injuries = self.fetch_team_injuries(team, all_players)

            if injuries:
                all_injuries[team] = injuries
                total_injuries += len(injuries)

                # Count by status
                out_count = sum(1 for inj in injuries if inj['injury_status'] == 'OUT')
                doubtful_count = sum(1 for inj in injuries if inj['injury_status'] == 'DOUBTFUL')
                questionable_count = sum(1 for inj in injuries if inj['injury_status'] == 'QUESTIONABLE')
                ir_count = sum(1 for inj in injuries if inj['injury_status'] == 'IR')

                status_parts = []
                if out_count:
                    status_parts.append(f"OUT: {out_count}")
                if doubtful_count:
                    status_parts.append(f"D: {doubtful_count}")
                if questionable_count:
                    status_parts.append(f"Q: {questionable_count}")
                if ir_count:
                    status_parts.append(f"IR: {ir_count}")

                status_str = ', '.join(status_parts) if status_parts else ''
                print(f"✓ {len(injuries)} injuries ({status_str})")
            else:
                print("✓ No injuries")

        # Save to file
        week_str = f"week_{week}" if week else "current"
        output_file = output_dir / f"injuries_{week_str}.json"

        with open(output_file, 'w') as f:
            json.dump(all_injuries, f, indent=2)

        # Generate summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'source': 'Sleeper API',
            'total_teams': len(self.nfl_teams),
            'teams_with_injuries': len(all_injuries),
            'total_injuries': total_injuries,
            'by_status': {
                'OUT': sum(
                    1 for injuries in all_injuries.values()
                    for inj in injuries if inj['injury_status'] == 'OUT'
                ),
                'DOUBTFUL': sum(
                    1 for injuries in all_injuries.values()
                    for inj in injuries if inj['injury_status'] == 'DOUBTFUL'
                ),
                'QUESTIONABLE': sum(
                    1 for injuries in all_injuries.values()
                    for inj in injuries if inj['injury_status'] == 'QUESTIONABLE'
                ),
                'IR': sum(
                    1 for injuries in all_injuries.values()
                    for inj in injuries if inj['injury_status'] == 'IR'
                )
            },
            'output_file': str(output_file)
        }

        # Save summary
        summary_file = output_dir / f"injuries_{week_str}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"✓ Fetched injuries from {len(all_injuries)} teams")
        print(f"✓ Total injuries: {total_injuries}")
        print(f"  - OUT: {summary['by_status']['OUT']}")
        print(f"  - DOUBTFUL: {summary['by_status']['DOUBTFUL']}")
        print(f"  - QUESTIONABLE: {summary['by_status']['QUESTIONABLE']}")
        print(f"  - IR: {summary['by_status']['IR']}")
        print(f"✓ Saved to: {output_file}")
        print(f"{'='*60}\n")

        return summary

    def get_player_injury_status(
        self,
        player_name: str,
        team: str,
        injury_file: Path
    ) -> Optional[Dict]:
        """Get injury status for a specific player.

        Args:
            player_name: Player name
            team: Team abbreviation
            injury_file: Path to injury JSON file

        Returns:
            Injury dict if player is injured, None otherwise
        """
        if not injury_file.exists():
            return None

        with open(injury_file, 'r') as f:
            data = json.load(f)

        team_injuries = data.get(team, [])

        for injury in team_injuries:
            if injury.get('player_name', '').lower() == player_name.lower():
                return injury

        return None

    def get_team_key_injuries(
        self,
        team: str,
        injury_file: Path,
        positions: List[str] = ['QB', 'RB', 'WR', 'TE']
    ) -> List[Dict]:
        """Get key injuries for a team (skill positions).

        Args:
            team: Team abbreviation
            injury_file: Path to injury JSON file
            positions: Positions to include

        Returns:
            List of injury dicts for key positions
        """
        if not injury_file.exists():
            return []

        with open(injury_file, 'r') as f:
            data = json.load(f)

        team_injuries = data.get(team, [])

        # Filter to key positions and OUT/DOUBTFUL status
        key_injuries = [
            inj for inj in team_injuries
            if inj.get('position') in positions
            and inj.get('injury_status') in ['OUT', 'DOUBTFUL', 'IR']
        ]

        return key_injuries

    def get_all_active_injuries(self) -> Dict[str, List[Dict]]:
        """Get current injuries for all teams (live from API).

        Returns:
            Dict of team -> list of injuries
        """
        all_players = self.fetch_all_players()

        if not all_players:
            return {}

        all_injuries = {}

        for team in self.nfl_teams:
            injuries = self.fetch_team_injuries(team, all_players)
            if injuries:
                # Filter to only active injuries (not IR for game-day decisions)
                active_injuries = [
                    inj for inj in injuries
                    if inj['injury_status'] in ['OUT', 'DOUBTFUL', 'QUESTIONABLE']
                ]
                if active_injuries:
                    all_injuries[team] = active_injuries

        return all_injuries


def fetch_injuries(output_dir: Path, week: Optional[int] = None) -> Dict:
    """Main function to fetch injury reports.

    Args:
        output_dir: Output directory
        week: Week number (optional)

    Returns:
        Summary dict
    """
    fetcher = InjuryFetcher()
    return fetcher.fetch_all_injuries(output_dir, week)


def get_current_injuries() -> Dict[str, List[Dict]]:
    """Get current injuries without saving to file.

    Returns:
        Dict of team -> list of injuries
    """
    fetcher = InjuryFetcher()
    return fetcher.get_all_active_injuries()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fetch NFL injury reports from Sleeper API'
    )
    parser.add_argument('--output', type=Path, default=Path('inputs/injuries'),
                       help='Output directory (default: inputs/injuries/)')
    parser.add_argument('--week', type=int, default=None,
                       help='Week number (for filename)')
    args = parser.parse_args()

    fetch_injuries(args.output, args.week)
