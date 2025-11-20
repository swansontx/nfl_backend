"""Fetch NFL injury reports from ESPN Core API.

Downloads official injury reports (Out, Doubtful, Questionable) for all teams.
Critical for adjusting predictions based on player availability.

Data Sources:
1. ESPN Core API (primary) - Official injury designations
2. Sleeper API (backup) - All NFL players with injury status

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
from concurrent.futures import ThreadPoolExecutor, as_completed


class InjuryFetcher:
    """Fetches NFL injury reports from ESPN Core API."""

    def __init__(self):
        """Initialize injury fetcher."""
        self.espn_core_api = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
        self.sleeper_api_base = "https://api.sleeper.app/v1"

        # NFL team abbreviations to ESPN team IDs
        self.team_id_map = {
            'ARI': 22, 'ATL': 1, 'BAL': 33, 'BUF': 2, 'CAR': 29, 'CHI': 3,
            'CIN': 4, 'CLE': 5, 'DAL': 6, 'DEN': 7, 'DET': 8, 'GB': 9,
            'HOU': 34, 'IND': 11, 'JAX': 30, 'KC': 12, 'LAC': 24, 'LAR': 14,
            'LV': 13, 'MIA': 15, 'MIN': 16, 'NE': 17, 'NO': 18, 'NYG': 19,
            'NYJ': 20, 'PHI': 21, 'PIT': 23, 'SEA': 26, 'SF': 25, 'TB': 27,
            'TEN': 10, 'WAS': 28
        }

        # NFL team abbreviations
        self.nfl_teams = list(self.team_id_map.keys())

        # Request headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        }

    def _fetch_single_injury(self, ref_url: str, team_abbr: str) -> Optional[Dict]:
        """Fetch a single injury record from ESPN Core API.

        Args:
            ref_url: The $ref URL for the injury
            team_abbr: Team abbreviation

        Returns:
            Injury dict or None
        """
        try:
            # Fetch injury data
            inj_response = requests.get(ref_url, headers=self.headers, timeout=10)
            inj_response.raise_for_status()
            item = inj_response.json()

            # Extract athlete info - need to follow $ref
            athlete = item.get('athlete', {})
            player_name = ''
            player_id = ''
            position = ''

            if '$ref' in athlete:
                try:
                    ath_response = requests.get(athlete['$ref'], headers=self.headers, timeout=10)
                    ath_response.raise_for_status()
                    athlete_data = ath_response.json()
                    player_name = athlete_data.get('displayName', athlete_data.get('fullName', ''))
                    player_id = str(athlete_data.get('id', ''))
                    pos = athlete_data.get('position', {})
                    if isinstance(pos, dict):
                        position = pos.get('abbreviation', '')
                    else:
                        position = str(pos)
                except:
                    pass

            # Get status from type object (more reliable)
            type_obj = item.get('type', {})
            status = ''
            if isinstance(type_obj, dict):
                status = type_obj.get('description', type_obj.get('name', ''))
            if not status:
                status = item.get('status', '')

            # Normalize status
            status_upper = str(status).upper()
            if 'OUT' in status_upper and 'DOUBT' not in status_upper:
                normalized_status = 'OUT'
            elif 'DOUBT' in status_upper:
                normalized_status = 'DOUBTFUL'
            elif 'QUESTION' in status_upper:
                normalized_status = 'QUESTIONABLE'
            elif 'PROBABLE' in status_upper:
                normalized_status = 'PROBABLE'
            elif 'IR' in status_upper or 'INJURED RESERVE' in status_upper:
                normalized_status = 'IR'
            else:
                normalized_status = status_upper if status_upper else 'UNKNOWN'

            # Get injury details
            details = item.get('details', {})
            injury_type = details.get('type', '') if isinstance(details, dict) else ''
            description = item.get('shortComment', '') or (details.get('detail', '') if isinstance(details, dict) else '')

            if player_name:
                return {
                    'player_id': player_id,
                    'player_name': player_name,
                    'team': team_abbr,
                    'position': position,
                    'injury_status': normalized_status,
                    'injury_type': injury_type,
                    'description': description,
                    'practice_status': '',
                    'timestamp': datetime.now().isoformat()
                }
        except Exception:
            pass

        return None

    def fetch_team_injuries_espn(self, team_abbr: str) -> List[Dict]:
        """Fetch injury report for a specific team from ESPN Core API.

        Args:
            team_abbr: Team abbreviation (e.g., 'KC', 'BUF')

        Returns:
            List of injury dictionaries
        """
        team_id = self.team_id_map.get(team_abbr)
        if not team_id:
            print(f"  ⚠️  Unknown team: {team_abbr}")
            return []

        url = f"{self.espn_core_api}/teams/{team_id}/injuries?limit=100"

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            data = response.json()
            injuries = []

            # Get all $ref URLs
            items = data.get('items', [])
            ref_urls = [item.get('$ref') for item in items if '$ref' in item]

            if not ref_urls:
                return []

            # Fetch all injuries in parallel for better performance
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {
                    executor.submit(self._fetch_single_injury, url, team_abbr): url
                    for url in ref_urls
                }

                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        injuries.append(result)

            return injuries

        except requests.exceptions.RequestException as e:
            print(f"  ✗ Error fetching injuries for {team_abbr}: {e}")
            return []

        except (KeyError, ValueError, TypeError) as e:
            print(f"  ⚠️  Error parsing injury data for {team_abbr}: {e}")
            return []

    def fetch_team_injuries_sleeper(self, team_abbr: str, all_players: Dict) -> List[Dict]:
        """Get injury report for a specific team from Sleeper data.

        Args:
            team_abbr: Team abbreviation (e.g., 'KC', 'BUF')
            all_players: Pre-fetched players dict from Sleeper

        Returns:
            List of injury dictionaries
        """
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
            has_injury = injury_status and str(injury_status).lower() not in ['', 'null', 'none']
            on_ir = status == 'Injured Reserve'

            if has_injury or on_ir:
                # Normalize injury status
                if on_ir and not has_injury:
                    normalized_status = 'IR'
                else:
                    normalized_status = str(injury_status).upper() if injury_status else 'IR'
                    if normalized_status in ['OUT', 'O']:
                        normalized_status = 'OUT'
                    elif normalized_status in ['DOUBTFUL', 'D']:
                        normalized_status = 'DOUBTFUL'
                    elif normalized_status in ['QUESTIONABLE', 'Q']:
                        normalized_status = 'QUESTIONABLE'

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
                    'timestamp': datetime.now().isoformat()
                })

        return injuries

    def fetch_sleeper_players(self) -> Dict:
        """Fetch all NFL players from Sleeper API as backup.

        Returns:
            Dictionary of player_id -> player data
        """
        url = f"{self.sleeper_api_base}/players/nfl"

        try:
            print("Fetching players from Sleeper API (backup)...")
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            players = response.json()
            print(f"✓ Retrieved {len(players)} players from Sleeper")
            return players
        except requests.exceptions.RequestException as e:
            print(f"✗ Sleeper API also failed: {e}")
            return {}

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
        print(f"Fetching NFL Injury Reports from ESPN Core API")
        print(f"{'='*60}\n")

        all_injuries = {}
        total_injuries = 0
        espn_failed_teams = []

        # Try ESPN Core API first
        for team in self.nfl_teams:
            print(f"Fetching {team}...", end=' ')

            injuries = self.fetch_team_injuries_espn(team)

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
                espn_failed_teams.append(team)
                print("✓ No injuries (or ESPN failed)")

            # Rate limit
            time.sleep(0.1)

        # If ESPN failed for many teams, try Sleeper as backup
        if len(espn_failed_teams) > 20:
            print(f"\n⚠️  ESPN failed for {len(espn_failed_teams)} teams, trying Sleeper backup...")
            sleeper_players = self.fetch_sleeper_players()

            if sleeper_players:
                for team in espn_failed_teams:
                    injuries = self.fetch_team_injuries_sleeper(team, sleeper_players)
                    if injuries:
                        all_injuries[team] = injuries
                        total_injuries += len(injuries)
                        print(f"  {team}: {len(injuries)} injuries from Sleeper")

        # Save to file
        week_str = f"week_{week}" if week else "current"
        output_file = output_dir / f"injuries_{week_str}.json"

        with open(output_file, 'w') as f:
            json.dump(all_injuries, f, indent=2)

        # Generate summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'source': 'ESPN Core API',
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
    all_injuries = {}

    for team in fetcher.nfl_teams:
        injuries = fetcher.fetch_team_injuries_espn(team)
        if injuries:
            # Filter to active injuries only
            active = [i for i in injuries if i['injury_status'] in ['OUT', 'DOUBTFUL', 'QUESTIONABLE']]
            if active:
                all_injuries[team] = active
        time.sleep(0.05)

    return all_injuries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fetch NFL injury reports from ESPN Core API'
    )
    parser.add_argument('--output', type=Path, default=Path('inputs/injuries'),
                       help='Output directory (default: inputs/injuries/)')
    parser.add_argument('--week', type=int, default=None,
                       help='Week number (for filename)')
    args = parser.parse_args()

    fetch_injuries(args.output, args.week)
