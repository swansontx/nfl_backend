"""Ingestion: fetch nflverse releases and export CSVs

Downloads nflverse data from GitHub releases including:
- Play-by-play data (for feature engineering)
- Player stats by week (for training data)
- Weekly rosters (for roster/injury tracking)

Data sources:
- https://github.com/nflverse/nflverse-data/releases
- Player stats, rosters, play-by-play all available as CSV
"""

from pathlib import Path
import argparse
import requests
import time
from typing import Optional
import json


def fetch_nflverse(year: int, out_dir: Path, cache_dir: Optional[Path] = None):
    """Fetch nflverse data for a given season.

    Downloads and processes nflverse CSV files including:
    - Play-by-play data
    - Player stats by week
    - Weekly rosters

    Args:
        year: NFL season year (e.g., 2023, 2024, 2025)
        out_dir: Output directory for processed data
        cache_dir: Optional cache directory to avoid re-downloading

    Data will be saved to:
        - {out_dir}/player_stats_{year}.csv - Weekly player statistics
        - {out_dir}/play_by_play_{year}.csv - Play-by-play data
        - {out_dir}/weekly_rosters_{year}.csv - Weekly roster data
        - {out_dir}/player_lookup_{year}.json - Player metadata
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    # nflverse data URLs
    base_url = "https://github.com/nflverse/nflverse-data/releases/download"

    datasets = {
        'player_stats': {
            'url': f'{base_url}/player_stats/player_stats_{year}.csv',
            'output': f'player_stats_{year}.csv',
            'description': 'Weekly player statistics'
        },
        'play_by_play': {
            'url': f'{base_url}/pbp/play_by_play_{year}.csv.gz',
            'output': f'play_by_play_{year}.csv',
            'description': 'Play-by-play data',
            'compressed': True
        },
        'weekly_rosters': {
            'url': f'{base_url}/weekly_rosters/weekly_rosters_{year}.csv',
            'output': f'weekly_rosters_{year}.csv',
            'description': 'Weekly roster data'
        }
    }

    print(f"\n{'='*60}")
    print(f"Fetching nflverse data for {year} season")
    print(f"{'='*60}\n")

    for dataset_name, dataset_info in datasets.items():
        output_file = out_dir / dataset_info['output']

        # Check if already exists (skip download)
        if output_file.exists():
            print(f"✓ {dataset_info['description']} already exists: {output_file}")
            continue

        # Check cache
        if cache_dir:
            cache_file = cache_dir / dataset_info['output']
            if cache_file.exists():
                print(f"✓ Using cached {dataset_info['description']}: {cache_file}")
                # Copy from cache to output
                output_file.write_bytes(cache_file.read_bytes())
                continue

        # Download the file
        print(f"⬇ Downloading {dataset_info['description']}...")
        print(f"  URL: {dataset_info['url']}")

        try:
            response = _download_with_retry(dataset_info['url'], max_retries=3)

            if response.status_code == 200:
                # Handle compressed files
                if dataset_info.get('compressed'):
                    import gzip
                    # Decompress gzip content
                    decompressed = gzip.decompress(response.content)
                    output_file.write_bytes(decompressed)
                else:
                    output_file.write_bytes(response.content)

                # Also save to cache if enabled
                if cache_dir:
                    cache_file = cache_dir / dataset_info['output']
                    cache_file.write_bytes(output_file.read_bytes())

                # Get file size
                size_mb = output_file.stat().st_size / (1024 * 1024)
                print(f"✓ Downloaded {dataset_info['description']}: {output_file} ({size_mb:.1f} MB)")
            else:
                print(f"✗ Failed to download {dataset_info['description']}: HTTP {response.status_code}")
                print(f"  This data may not be available yet for {year}")
                # Create empty placeholder
                output_file.write_text('')

        except Exception as e:
            print(f"✗ Error downloading {dataset_info['description']}: {e}")
            # Create empty placeholder
            output_file.write_text('')

    # Build player lookup from player_stats
    _build_player_lookup(out_dir, year)

    print(f"\n{'='*60}")
    print(f"nflverse data fetch complete for {year}")
    print(f"{'='*60}\n")


def _download_with_retry(url: str, max_retries: int = 3, timeout: int = 30) -> requests.Response:
    """Download file with exponential backoff retry.

    Args:
        url: URL to download
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds

    Returns:
        requests.Response object

    Raises:
        Exception if all retries fail
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            return response

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"  Retry {attempt + 1}/{max_retries} after {wait_time}s... ({e})")
                time.sleep(wait_time)
            else:
                raise Exception(f"Failed after {max_retries} attempts: {e}")


def _build_player_lookup(out_dir: Path, year: int) -> None:
    """Build player lookup JSON from player_stats CSV.

    Extracts unique players and their metadata for quick lookups.

    Args:
        out_dir: Directory containing player_stats CSV
        year: Season year
    """
    player_stats_file = out_dir / f'player_stats_{year}.csv'
    lookup_file = out_dir / f'player_lookup_{year}.json'

    if not player_stats_file.exists() or player_stats_file.stat().st_size == 0:
        print(f"⚠ No player_stats file found, creating empty lookup")
        lookup_file.write_text('{}')
        return

    try:
        import csv

        players = {}

        with open(player_stats_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                player_id = row.get('player_id')
                if not player_id or player_id in players:
                    continue

                # Extract player metadata
                players[player_id] = {
                    'player_id': player_id,
                    'player_name': row.get('player_name', row.get('player_display_name', '')),
                    'position': row.get('position', ''),
                    'team': row.get('recent_team', row.get('team', '')),
                    'position_group': row.get('position_group', '')
                }

        # Write lookup JSON
        with open(lookup_file, 'w') as f:
            json.dump(players, f, indent=2)

        print(f"✓ Built player lookup: {lookup_file} ({len(players)} players)")

    except Exception as e:
        print(f"⚠ Error building player lookup: {e}")
        lookup_file.write_text('{}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Fetch nflverse data for a season')
    p.add_argument('--year', type=int, default=2024,
                   help='NFL season year (default: 2024)')
    p.add_argument('--out', type=Path, default=Path('inputs'),
                   help='Output directory (default: inputs/)')
    p.add_argument('--cache', type=Path, default=None,
                   help='Optional cache directory to avoid re-downloading')
    args = p.parse_args()

    fetch_nflverse(args.year, args.out, args.cache)
