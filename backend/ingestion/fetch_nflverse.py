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
    # placeholder: implement download logic or local copy
    out_dir.mkdir(parents=True, exist_ok=True)
    # TODO: download and unpack
    sample = out_dir / f"stats_player_week_{year}.csv"
    sample.write_text('player_id,week,team,stat\n')
    print(f'Wrote sample {sample}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--year', type=int, default=2025)
    p.add_argument('--out', type=Path, default=Path('inputs'))
    args = p.parse_args()
    fetch_nflverse(args.year, args.out)
