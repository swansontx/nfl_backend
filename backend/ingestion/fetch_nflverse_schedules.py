"""Fetch NFL schedules from nflverse.

Downloads schedule data for specific seasons.
Schedule data includes game times, locations, and results.

Usage:
    python -m backend.ingestion.fetch_nflverse_schedules --year 2024
"""

from pathlib import Path
import argparse
import requests
import time
from typing import Optional
from datetime import datetime


def _get_file_age_days(file_path: Path) -> float:
    """Get age of file in days."""
    if not file_path.exists():
        return float('inf')
    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
    age = datetime.now() - mtime
    return age.total_seconds() / 86400


def fetch_schedule(year: int, out_dir: Path, cache_dir: Optional[Path] = None,
                   max_age_days: int = 7, force: bool = False):
    """Fetch NFL schedule for a season with incremental updates.

    Schedules are updated weekly with new scores, so default refresh is 7 days.

    Args:
        year: NFL season year (e.g., 2023, 2024, 2025)
        out_dir: Output directory for schedule data
        cache_dir: Optional cache directory to avoid re-downloading
        max_age_days: Re-download if file is older than N days (default: 7)
        force: Force re-download even if file is fresh (default: False)

    Downloads:
        - inputs/schedule_{year}.csv - Full season schedule with results
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    # nflverse schedule URL
    schedule_url = f'https://github.com/nflverse/nflverse-data/releases/download/schedules/schedules.csv'

    print(f"\n{'='*60}")
    print(f"Fetching NFL schedule for {year}")
    if force:
        print(f"Update: FORCE (re-downloading)")
    else:
        print(f"Update: INCREMENTAL (refresh if older than {max_age_days} days)")
    print(f"{'='*60}\n")

    output_file = out_dir / f'schedule_{year}.csv'

    # Check if file exists and is fresh enough
    if output_file.exists() and not force:
        file_age = _get_file_age_days(output_file)
        if file_age < max_age_days:
            size_kb = output_file.stat().st_size / 1024
            print(f"✓ Schedule is fresh ({file_age:.1f} days old, {size_kb:.1f} KB)")
            return
        else:
            print(f"↻ Schedule is stale ({file_age:.1f} days old), re-downloading...")

    # Check cache
    if cache_dir and not force:
        cache_file = cache_dir / f'schedule_{year}.csv'
        if cache_file.exists():
            cache_age = _get_file_age_days(cache_file)
            if cache_age < max_age_days:
                print(f"✓ Using cached schedule: {cache_file}")
                output_file.write_bytes(cache_file.read_bytes())
                return

    # Download the full schedules file
    print(f"⬇ Downloading NFL schedules...")
    print(f"  URL: {schedule_url}")

    try:
        response = _download_with_retry(schedule_url, max_retries=3)

        if response.status_code == 200:
            # Save full schedules temporarily
            all_schedules_file = out_dir / 'schedules_all.csv'
            all_schedules_file.write_bytes(response.content)

            # Filter to specific year
            _filter_schedule_by_year(all_schedules_file, output_file, year)

            # Clean up temporary file
            all_schedules_file.unlink()

            # Save to cache if enabled
            if cache_dir:
                cache_file = cache_dir / f'schedule_{year}.csv'
                cache_file.write_bytes(output_file.read_bytes())

            # Get file size
            size_kb = output_file.stat().st_size / 1024
            print(f"✓ Downloaded schedule for {year}: {output_file} ({size_kb:.1f} KB)")
        else:
            print(f"✗ Failed to download schedules: HTTP {response.status_code}")
            # Create empty placeholder
            output_file.write_text('')

    except Exception as e:
        print(f"✗ Error downloading schedules: {e}")
        # Create empty placeholder
        output_file.write_text('')

    print(f"\n{'='*60}")
    print(f"Schedule download complete")
    print(f"{'='*60}\n")


def _download_with_retry(url: str, max_retries: int = 3, timeout: int = 30) -> requests.Response:
    """Download file with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            return response

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"  Retry {attempt + 1}/{max_retries} after {wait_time}s... ({e})")
                time.sleep(wait_time)
            else:
                raise Exception(f"Failed after {max_retries} attempts: {e}")


def _filter_schedule_by_year(input_file: Path, output_file: Path, year: int):
    """Filter schedule CSV to only include specific year."""
    import csv

    with open(input_file, 'r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames

        # Filter rows for this year
        rows = [row for row in reader if row.get('season') == str(year)]

    # Write filtered rows
    with open(output_file, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Filtered {len(rows)} games for {year} season")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Fetch NFL schedules from nflverse with incremental updates')
    p.add_argument('--year', type=int, default=2024,
                   help='NFL season year (default: 2024)')
    p.add_argument('--out', type=Path, default=Path('inputs'),
                   help='Output directory (default: inputs/)')
    p.add_argument('--cache', type=Path, default=None,
                   help='Optional cache directory to avoid re-downloading')
    p.add_argument('--max-age', type=int, default=7,
                   help='Re-download files older than N days (default: 7)')
    p.add_argument('--force', action='store_true',
                   help='Force re-download regardless of file age')
    args = p.parse_args()

    fetch_schedule(args.year, args.out, args.cache, args.max_age, args.force)
