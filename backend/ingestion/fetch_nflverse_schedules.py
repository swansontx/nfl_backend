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


def _check_remote_modified(url: str, local_file: Path) -> bool:
    """Check if remote file has been modified since local file was downloaded.

    Uses HTTP HEAD request to compare Last-Modified header with local mtime.
    Only downloads when remote actually has new data.
    """
    if not local_file.exists():
        return True

    try:
        response = requests.head(url, timeout=10, allow_redirects=True)
        if response.status_code != 200:
            return True

        last_modified = response.headers.get('Last-Modified')
        if not last_modified:
            # Check file size instead
            content_length = response.headers.get('Content-Length')
            if content_length:
                local_size = local_file.stat().st_size
                remote_size = int(content_length)
                if abs(remote_size - local_size) > 1000:
                    return True
            return False

        from email.utils import parsedate_to_datetime
        remote_mtime = parsedate_to_datetime(last_modified)
        local_mtime = datetime.fromtimestamp(local_file.stat().st_mtime, tz=remote_mtime.tzinfo)
        return remote_mtime > local_mtime

    except Exception as e:
        print(f"  ⚠ Could not check remote: {e}")
        return False


def fetch_schedule(year: int, out_dir: Path, cache_dir: Optional[Path] = None,
                   force: bool = False):
    """Fetch NFL schedule for a season with smart incremental updates.

    Uses HTTP conditional requests to check if nflverse has updated the schedule.
    Only downloads when remote file has new data (e.g., new week scores).

    Args:
        year: NFL season year (e.g., 2023, 2024, 2025)
        out_dir: Output directory for schedule data
        cache_dir: Optional cache directory to avoid re-downloading
        force: Force re-download even if file is current (default: False)

    Downloads:
        - inputs/schedule_{year}.csv - Full season schedule with results
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    # nflverse schedule URL
    schedule_url = 'https://github.com/nflverse/nflverse-data/releases/download/schedules/schedules.csv'

    print(f"\n{'='*60}")
    print(f"Fetching NFL schedule for {year}")
    if force:
        print(f"Update: FORCE (re-downloading)")
    else:
        print(f"Update: SMART (only download if nflverse has new data)")
    print(f"{'='*60}\n")

    output_file = out_dir / f'schedule_{year}.csv'

    # Check if we need to download
    if output_file.exists() and not force:
        if not _check_remote_modified(schedule_url, output_file):
            size_kb = output_file.stat().st_size / 1024
            print(f"✓ Schedule is current ({size_kb:.1f} KB)")
            return
        else:
            print(f"↻ Schedule has updates, re-downloading...")

    # Check cache
    if cache_dir and not force:
        cache_file = cache_dir / f'schedule_{year}.csv'
        if cache_file.exists():
            if not _check_remote_modified(schedule_url, cache_file):
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
    p = argparse.ArgumentParser(description='Fetch NFL schedules from nflverse (smart incremental updates)')
    p.add_argument('--year', type=int, default=2024,
                   help='NFL season year (default: 2024)')
    p.add_argument('--out', type=Path, default=Path('inputs'),
                   help='Output directory (default: inputs/)')
    p.add_argument('--cache', type=Path, default=None,
                   help='Optional cache directory')
    p.add_argument('--force', action='store_true',
                   help='Force re-download even if current')
    args = p.parse_args()

    fetch_schedule(args.year, args.out, args.cache, args.force)
