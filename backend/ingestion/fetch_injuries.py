"""Ingestion: fetch and parse NFL injury reports

This script fetches injury data from various sources and outputs
parsed JSON to outputs/injuries_YYYYMMDD_parsed.json.

Sources could include:
- nflverse injury data
- ESPN injury reports
- Official NFL injury reports
- News APIs

TODOs:
- Determine primary injury data source
- Add CLI args (date range, output path)
- Add retry/backoff for API/web scraping
- Add data validation/schema checking
- Handle multiple sources and deduplication
- Wire into orchestration/orchestrator
"""

from pathlib import Path
import argparse
import json
from datetime import datetime
from typing import List, Dict


def fetch_injuries(date: str, out_dir: Path) -> List[Dict]:
    """Fetch and parse injury data for a given date.

    Args:
        date: Date string in YYYYMMDD format
        out_dir: Output directory for parsed injury data

    Returns:
        List of injury records

    Output format:
        {
            "player_id": "nflverse_player_id",
            "player_name": "Player Name",
            "team": "BUF",
            "status": "Questionable|Doubtful|Out|IR",
            "injury": "Knee",
            "date": "20251119",
            "week": 12,
            "season": 2025
        }

    TODO: Implement actual data fetching from injury source(s)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Replace with actual injury data fetching
    # Options:
    # 1. nflverse injury data (if available)
    # 2. Web scraping from ESPN/NFL.com
    # 3. Sports data APIs

    # Placeholder: create sample injury data
    sample_injuries = [
        {
            'player_id': 'sample_player_001',
            'player_name': 'Sample Player',
            'team': 'BUF',
            'status': 'Questionable',
            'injury': 'Knee',
            'date': date,
            'week': 12,
            'season': 2025
        }
    ]

    # Write parsed output
    out_file = out_dir / f"injuries_{date}_parsed.json"
    out_file.write_text(json.dumps(sample_injuries, indent=2))
    print(f"Wrote {len(sample_injuries)} injury records to {out_file}")

    return sample_injuries


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Fetch and parse NFL injury reports')
    p.add_argument('--date', type=str,
                   default=datetime.now().strftime('%Y%m%d'),
                   help='Date in YYYYMMDD format (default: today)')
    p.add_argument('--out', type=Path, default=Path('outputs'),
                   help='Output directory')
    args = p.parse_args()

    # Validate date format
    try:
        datetime.strptime(args.date, '%Y%m%d')
    except ValueError:
        print(f"Error: Invalid date format '{args.date}'. Expected YYYYMMDD.")
        exit(1)

    injuries = fetch_injuries(args.date, args.out)
    print(f"Fetched {len(injuries)} injury records for {args.date}")
