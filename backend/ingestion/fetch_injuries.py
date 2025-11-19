"""Fetch NFL injury reports from ESPN API.

Downloads official injury reports (Out, Doubtful, Questionable) for all teams.
Critical for adjusting predictions based on player availability.

Data Sources:
1. ESPN API (primary) - Official injury designations
2. nflverse injuries (backup) - Historical injury data

Injury Statuses:
- OUT: Player will not play
- DOUBTFUL: ~25% chance of playing
- QUESTIONABLE: ~50% chance of playing
- PROBABLE: Deprecated (no longer used)

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
    """Fetches NFL injury reports from ESPN API."""

    def __init__(self):
        """Initialize injury fetcher."""
        self.espn_api_base = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

        # NFL team abbreviations (ESPN format)
        self.nfl_teams = [
            'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
            'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
            'LAC', 'LAR', 'LV', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
            'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS'
        ]

    def fetch_team_injuries(self, team_abbr: str) -> List[Dict]:
        """Fetch injury report for a specific team.

        Args:
            team_abbr: Team abbreviation (e.g., 'KC', 'BUF')

        Returns:
            List of injury dictionaries
        """
        # ESPN team endpoint
        url = f"{self.espn_api_base}/teams/{team_abbr}/injuries"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            injuries = []

            # Parse injuries from response
            for category in data.get('injuries', []):
                for athlete in category.get('athletes', []):
                    player_name = athlete.get('athlete', {}).get('displayName', '')
                    player_id = athlete.get('athlete', {}).get('id', '')
                    position = athlete.get('position', {}).get('abbreviation', '')

                    # Injury details
                    for injury in athlete.get('injuries', []):
                        status = injury.get('status', '')  # OUT, DOUBTFUL, QUESTIONABLE
                        injury_type = injury.get('type', '')
                        details = injury.get('details', {})
                        description = details.get('detail', '')
                        practice_status = details.get('fantasyStatus', '')

                        injuries.append({
                            'player_id': player_id,
                            'player_name': player_name,
                            'team': team_abbr,
                            'position': position,
                            'injury_status': status,
                            'injury_type': injury_type,
                            'description': description,
                            'practice_status': practice_status,
                            'timestamp': datetime.now().isoformat()
                        })

            return injuries

        except requests.exceptions.RequestException as e:
            print(f"  ✗ Error fetching injuries for {team_abbr}: {e}")
            return []

        except (KeyError, ValueError) as e:
            print(f"  ⚠️  Error parsing injury data for {team_abbr}: {e}")
            return []

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
        print(f"Fetching NFL Injury Reports")
        print(f"{'='*60}\n")

        all_injuries = {}
        total_injuries = 0

        for team in self.nfl_teams:
            print(f"Fetching {team}...", end=' ')

            injuries = self.fetch_team_injuries(team)

            if injuries:
                all_injuries[team] = injuries
                total_injuries += len(injuries)

                # Count by status
                out_count = sum(1 for inj in injuries if inj['injury_status'] == 'OUT')
                doubtful_count = sum(1 for inj in injuries if inj['injury_status'] == 'DOUBTFUL')
                questionable_count = sum(1 for inj in injuries if inj['injury_status'] == 'QUESTIONABLE')

                print(f"✓ {len(injuries)} injuries (OUT: {out_count}, D: {doubtful_count}, Q: {questionable_count})")
            else:
                print("✓ No injuries")

            # Rate limit
            time.sleep(0.2)

        # Save to file
        week_str = f"week_{week}" if week else "current"
        output_file = output_dir / f"injuries_{week_str}.json"

        with open(output_file, 'w') as f:
            json.dump(all_injuries, f, indent=2)

        # Generate summary
        summary = {
            'timestamp': datetime.now().isoformat(),
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
            and inj.get('injury_status') in ['OUT', 'DOUBTFUL']
        ]

        return key_injuries


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fetch NFL injury reports from ESPN API'
    )
    parser.add_argument('--output', type=Path, default=Path('inputs/injuries'),
                       help='Output directory (default: inputs/injuries/)')
    parser.add_argument('--week', type=int, default=None,
                       help='Week number (for filename)')
    args = parser.parse_args()

    fetch_injuries(args.output, args.week)
