"""Modeling scaffold: model runner interface

- load features
- run count/td/yard models
- write props CSV to outputs/
"""

from pathlib import Path


def run_models(features_json: Path, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text('market,player_id,mu\n')
    print(f'Wrote sample props to {out_csv}')

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--features', type=Path, default=Path('outputs/player_pbp_features_by_id.json'))
    p.add_argument('--out', type=Path, default=Path('outputs/props_modelled.csv'))
    args = p.parse_args()
    run_models(args.features, args.out)
