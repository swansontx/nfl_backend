"""Ingestion: fetch nflverse releases and export CSVs

Downloads ALL available nflverse data from GitHub releases:
- Play-by-play data (300+ columns with advanced metrics)
- Player stats by week (aggregated offensive/defensive stats)
- Weekly rosters (for roster/injury tracking)
- Next Gen Stats (player tracking data)
- Snap counts (participation percentages)
- PFR advanced stats (Pro Football Reference)
- Depth charts (positional rankings)

Advanced Metrics in Play-by-Play:
- EPA (Expected Points Added): epa, qb_epa, total_home_epa, total_away_epa
- WPA (Win Probability Added): wpa, home_wp, away_wp, wp
- CPOE (Completion % Over Expected): cpoe
- Air Yards: air_yards, yards_after_catch, complete_pass
- Success Rate: success (EPA > 0)
- xYAC: xyac_epa, xyac_mean_yardage, xyac_median_yardage
- QB Pressure: qb_hit, qb_hurry, qb_knockdown
- Personnel: offense_personnel, defense_personnel, defenders_in_box

Data sources:
- https://github.com/nflverse/nflverse-data/releases
- All datasets available as CSV (some compressed with gzip)
"""

from pathlib import Path
import argparse
import requests
import time
from typing import Optional
import json
import os
from datetime import datetime, timedelta


def _check_remote_modified(url: str, local_file: Path) -> bool:
    """Check if remote file has been modified since local file was downloaded.

    Uses HTTP HEAD request to compare Last-Modified header with local mtime.
    This is the smart way to check for updates - only downloads when remote
    actually has new data (e.g., new week added).

    Args:
        url: Remote URL
        local_file: Local file path

    Returns:
        True if remote is newer or local doesn't exist, False if local is current
    """
    if not local_file.exists():
        return True

    try:
        # Get remote Last-Modified header
        response = requests.head(url, timeout=10, allow_redirects=True)
        if response.status_code != 200:
            # Can't check, assume we need to download
            return True

        last_modified = response.headers.get('Last-Modified')
        if not last_modified:
            # No header, can't compare - check file size instead
            content_length = response.headers.get('Content-Length')
            if content_length:
                local_size = local_file.stat().st_size
                remote_size = int(content_length)
                # If sizes differ significantly, remote has been updated
                if abs(remote_size - local_size) > 1000:  # 1KB tolerance
                    return True
            return False

        # Parse Last-Modified header (format: "Wed, 21 Oct 2015 07:28:00 GMT")
        from email.utils import parsedate_to_datetime
        remote_mtime = parsedate_to_datetime(last_modified)
        local_mtime = datetime.fromtimestamp(local_file.stat().st_mtime, tz=remote_mtime.tzinfo)

        # Remote is newer if its mtime is after local
        return remote_mtime > local_mtime

    except Exception as e:
        print(f"  ⚠ Could not check remote: {e}")
        return False  # Don't download if we can't verify


def fetch_nflverse(year: int, out_dir: Path, cache_dir: Optional[Path] = None,
                  include_all: bool = True, force: bool = False):
    """Fetch nflverse data for a given season with smart incremental updates.

    Downloads ALL available nflverse datasets by default.
    Uses HTTP conditional requests to check if remote files have been updated:
    - Only downloads when remote file has new data (e.g., new week added)
    - Compares Last-Modified header with local file timestamp
    - No unnecessary re-downloads of 100+ MB files
    - Use force=True to always re-download everything

    Datasets downloaded:
    - Play-by-play data (with EPA, WPA, CPOE, air yards, etc.)
    - Player stats by week
    - Weekly rosters
    - Next Gen Stats (player tracking data)
    - Snap counts (participation percentages)
    - PFR advanced stats (Pro Football Reference)
    - Depth charts (positional rankings)

    Args:
        year: NFL season year (e.g., 2023, 2024, 2025)
        out_dir: Output directory for processed data
        cache_dir: Optional cache directory to avoid re-downloading
        include_all: Download all available datasets (default: True)
        force: Force re-download even if files are current (default: False)

    Data will be saved to:
        - {out_dir}/player_stats_{year}.csv - Weekly player statistics
        - {out_dir}/play_by_play_{year}.csv - Play-by-play with 300+ metrics
        - {out_dir}/weekly_rosters_{year}.csv - Weekly roster data
        - {out_dir}/player_lookup_{year}.json - Player metadata
        - {out_dir}/nextgen_stats_{year}.csv - Next Gen Stats
        - {out_dir}/snap_counts_{year}.csv - Snap counts
        - {out_dir}/pfr_advstats_{year}.csv - PFR advanced stats
        - {out_dir}/depth_charts_{year}.csv - Weekly depth charts
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    # nflverse data URLs
    base_url = "https://github.com/nflverse/nflverse-data/releases/download"

    # Core datasets (always downloaded)
    datasets = {
        'player_stats': {
            'url': f'{base_url}/player_stats/player_stats_{year}.csv',
            'output': f'player_stats_{year}.csv',
            'description': 'Weekly player statistics',
            'core': True
        },
        'play_by_play': {
            'url': f'{base_url}/pbp/play_by_play_{year}.csv.gz',
            'output': f'play_by_play_{year}.csv',
            'description': 'Play-by-play data (EPA, WPA, CPOE, air yards, etc.)',
            'compressed': True,
            'core': True
        },
        'weekly_rosters': {
            'url': f'{base_url}/weekly_rosters/weekly_rosters_{year}.csv',
            'output': f'weekly_rosters_{year}.csv',
            'description': 'Weekly roster data',
            'core': True
        }
    }

    # Advanced datasets (included when include_all=True)
    if include_all:
        advanced_datasets = {
            'nextgen_stats': {
                'url': f'{base_url}/nextgen_stats/nextgen_stats_{year}.csv',
                'output': f'nextgen_stats_{year}.csv',
                'description': 'Next Gen Stats (player tracking data)'
            },
            'snap_counts': {
                'url': f'{base_url}/snap_counts/snap_counts_{year}.csv',
                'output': f'snap_counts_{year}.csv',
                'description': 'Snap count data (participation percentages)'
            },
            'pfr_advstats_pass': {
                'url': f'{base_url}/pfr_advstats/advstats_season_pass_{year}.csv',
                'output': f'pfr_advstats_pass_{year}.csv',
                'description': 'PFR advanced passing stats'
            },
            'pfr_advstats_rush': {
                'url': f'{base_url}/pfr_advstats/advstats_season_rush_{year}.csv',
                'output': f'pfr_advstats_rush_{year}.csv',
                'description': 'PFR advanced rushing stats'
            },
            'pfr_advstats_rec': {
                'url': f'{base_url}/pfr_advstats/advstats_season_rec_{year}.csv',
                'output': f'pfr_advstats_rec_{year}.csv',
                'description': 'PFR advanced receiving stats'
            },
            'pfr_advstats_def': {
                'url': f'{base_url}/pfr_advstats/advstats_season_def_{year}.csv',
                'output': f'pfr_advstats_def_{year}.csv',
                'description': 'PFR advanced defensive stats'
            },
            'depth_charts': {
                'url': f'{base_url}/depth_charts/depth_charts_{year}.csv',
                'output': f'depth_charts_{year}.csv',
                'description': 'Weekly depth charts'
            },
            'ftn_charting': {
                'url': f'{base_url}/ftn_charting/ftn_charting_{year}.csv',
                'output': f'ftn_charting_{year}.csv',
                'description': 'FTN charting data (coverage schemes, formations)'
            },
            'participations': {
                'url': f'{base_url}/participations/participations_{year}.csv',
                'output': f'participations_{year}.csv',
                'description': 'Play-level participation data'
            },
            'combine': {
                'url': f'{base_url}/combine/combine.csv',
                'output': f'combine_{year}.csv',
                'description': 'NFL combine results'
            }
        }
        datasets.update(advanced_datasets)

    print(f"\n{'='*60}")
    print(f"Fetching nflverse data for {year} season")
    if include_all:
        print(f"Mode: FULL (all available datasets)")
    else:
        print(f"Mode: CORE ONLY (play-by-play, stats, rosters)")
    if force:
        print(f"Update: FORCE (re-downloading all files)")
    else:
        print(f"Update: SMART (only download if remote has new data)")
    print(f"{'='*60}\n")

    downloaded_count = 0
    cached_count = 0
    failed_count = 0

    for dataset_name, dataset_info in datasets.items():
        output_file = out_dir / dataset_info['output']

        # Check if we need to download (smart check using HTTP conditional request)
        if output_file.exists() and not force:
            # Check if remote has been updated since we downloaded
            if not _check_remote_modified(dataset_info['url'], output_file):
                size_mb = output_file.stat().st_size / (1024 * 1024)
                print(f"✓ {dataset_info['description']} is current ({size_mb:.1f} MB)")
                cached_count += 1
                continue
            else:
                print(f"↻ {dataset_info['description']} has updates, re-downloading...")
                # Continue to download section below

        # Check cache
        if cache_dir and not force:
            cache_file = cache_dir / dataset_info['output']
            if cache_file.exists():
                # Use cache if remote hasn't been updated
                if not _check_remote_modified(dataset_info['url'], cache_file):
                    print(f"✓ Using cached {dataset_info['description']}: {cache_file}")
                    output_file.write_bytes(cache_file.read_bytes())
                    cached_count += 1
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
                downloaded_count += 1
            else:
                print(f"⚠ {dataset_info['description']} not available (HTTP {response.status_code})")
                print(f"  This data may not be published yet for {year}")
                # Create empty placeholder for core datasets only
                if dataset_info.get('core'):
                    output_file.write_text('')
                failed_count += 1

        except Exception as e:
            print(f"⚠ Error downloading {dataset_info['description']}: {e}")
            # Create empty placeholder for core datasets only
            if dataset_info.get('core'):
                output_file.write_text('')
            failed_count += 1

    # Build player lookup from player_stats
    _build_player_lookup(out_dir, year)

    print(f"\n{'='*60}")
    print(f"nflverse data fetch complete for {year}")
    print(f"  Downloaded: {downloaded_count} files")
    print(f"  Current/Cached: {cached_count} files")
    if failed_count > 0:
        print(f"  ⚠ Unavailable: {failed_count} files (may not exist for {year})")
    print(f"  Note: Run again anytime - only downloads when nflverse releases new data")
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
    p = argparse.ArgumentParser(
        description='Fetch nflverse data for a season (smart incremental updates)'
    )
    p.add_argument('--year', type=int, default=2025,
                   help='NFL season year (default: 2024)')
    p.add_argument('--out', type=Path, default=Path('inputs'),
                   help='Output directory (default: inputs/)')
    p.add_argument('--cache', type=Path, default=None,
                   help='Optional cache directory')
    p.add_argument('--core-only', action='store_true',
                   help='Download only core datasets (PBP, stats, rosters)')
    p.add_argument('--force', action='store_true',
                   help='Force re-download all files even if current')
    args = p.parse_args()

    fetch_nflverse(
        args.year,
        args.out,
        args.cache,
        include_all=not args.core_only,
        force=args.force
    )
