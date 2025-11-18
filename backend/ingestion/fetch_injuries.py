"""Ingestion: fetch and parse injury reports (ESPN / news sources)

This scaffold should fetch injury HTML/json sources or read local cached files,
parse athlete name, team, status (OUT/QUESTIONABLE/Doubtful), and write to
outputs/injuries_<date>_parsed.json with structure keyed by game/team/player.

TODOs:
- Implement actual HTTP fetching or local cache reading
- Provide robust parsing for ESPN payloads and other sources
- Normalize injury statuses to canonical enums
"""

from pathlib import Path
import json
from datetime import date


def parse_injuries(input_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"injuries_{date.today().isoformat()}_parsed.json"
    # placeholder: produce empty dict
    data = {}
    out_file.write_text(json.dumps(data, indent=2))
    print(f'Wrote parsed injuries to {out_file}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='in_path', default=Path('inputs/injuries_raw.json'))
    p.add_argument('--out', dest='out_dir', default=Path('outputs'))
    args = p.parse_args()
    parse_injuries(Path(args.in_path), Path(args.out_dir))
