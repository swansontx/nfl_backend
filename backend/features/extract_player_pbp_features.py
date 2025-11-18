"""Features extraction scaffold

Placeholder script to show expected output and interface.
"""

from pathlib import Path


def extract_features(pbp_csv: Path, out_json: Path):
    out_json.write_text('{}')
    print(f'Wrote features to {out_json}')

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--pbp', type=Path, default=Path('inputs/pbp.csv'))
    p.add_argument('--out', type=Path, default=Path('outputs/player_pbp_features_by_id.json'))
    args = p.parse_args()
    extract_features(args.pbp, args.out)
