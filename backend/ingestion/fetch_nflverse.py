"""Ingestion: fetch nflverse releases and export CSVs

This is a lightweight scaffold for the nflverse ingestion script.
It should download release CSVs (or read from local cache) and write to
inputs/stats_player_week_YYYY.csv and inputs/player_lookup_YYYY.json.

TODOs:
- Add CLI args (years, cache-dir)
- Add retry/backoff
- Validate checksums
- Wire into orchestration/orchestrator
"""

from pathlib import Path
import argparse


def fetch_nflverse(year: int, out_dir: Path):
    """Fetch nflverse data for a given season.

    Downloads and processes nflverse CSV files including:
    - Play-by-play data
    - Player stats by week
    - Player lookup/roster data

    TODO: Implement actual download from nflverse repository
    Example source: https://github.com/nflverse/nflverse-data/releases
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # TODO: download and unpack from nflverse
    # import requests
    # response = requests.get(f'https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{year}.csv')
    # with open(out_dir / f'stats_player_week_{year}.csv', 'wb') as f:
    #     f.write(response.content)

    # Create sample CSV with proper schema
    sample_stats = out_dir / f"stats_player_week_{year}.csv"
    sample_stats.write_text(
        'player_id,season,season_type,week,game_id,team,opponent,'
        'completions,attempts,passing_yards,passing_tds,interceptions,'
        'sacks,sack_yards,rushing_attempts,rushing_yards,rushing_tds,'
        'receptions,targets,receiving_yards,receiving_tds\n'
    )
    print(f'Wrote sample stats CSV: {sample_stats}')

    # Create sample player lookup
    sample_lookup = out_dir / f"player_lookup_{year}.json"
    sample_lookup.write_text('{}')
    print(f'Wrote sample player lookup: {sample_lookup}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--year', type=int, default=2025)
    p.add_argument('--out', type=Path, default=Path('inputs'))
    args = p.parse_args()
    fetch_nflverse(args.year, args.out)
