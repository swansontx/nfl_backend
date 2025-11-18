"""Simple smoke test that runs the ingestion + model runner and verifies outputs exist
"""
from pathlib import Path
import subprocess

ROOT=Path('.')

def run_cmd(cmd):
    print('RUN:', ' '.join(cmd))
    r = subprocess.run(cmd)
    if r.returncode!=0:
        raise SystemExit(r.returncode)


def main():
    run_cmd(['python','backend/ingestion/fetch_nflverse.py','--year','2025','--out','inputs'])
    run_cmd(['python','backend/features/extract_player_pbp_features.py','--pbp','inputs/pbp.csv','--out','outputs/player_pbp_features_by_id.json'])
    run_cmd(['python','backend/modeling/model_runner.py','--features','outputs/player_pbp_features_by_id.json','--out','outputs/props_modelled.csv'])
    print('Smoke test completed')

if __name__=='__main__':
    main()
