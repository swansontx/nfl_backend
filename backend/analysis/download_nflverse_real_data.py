"""Download REAL NFLverse data for training."""

import pandas as pd
import requests
from pathlib import Path

def download_nflverse_file(dataset, year, format='parquet'):
    """Download a file from NFLverse releases."""
    url = f"https://github.com/nflverse/nflverse-data/releases/download/{dataset}/{dataset}_{year}.{format}"
    output_dir = Path('/home/user/nfl_backend/inputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{dataset}_{year}.{format}"

    print(f"Downloading {dataset}_{year}.{format}...")
    print(f"URL: {url}")

    try:
        response = requests.get(url, timeout=180)
        response.raise_for_status()
        output_file.write_bytes(response.content)
        size_mb = len(response.content) / 1024 / 1024
        print(f"✅ Downloaded {output_file.name} ({size_mb:.1f} MB)")

        # Convert to CSV if parquet
        if format == 'parquet':
            df = pd.read_parquet(output_file)
            csv_file = output_file.with_suffix('.csv')
            df.to_csv(csv_file, index=False)
            print(f"✅ Converted to CSV: {csv_file.name} ({len(df):,} rows)")
            return df
        return None
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None

print("="*80)
print("DOWNLOADING REAL NFLVERSE DATA")
print("="*80 + "\n")

# Download weekly player stats (this is aggregated stats per player per week/game)
datasets_to_download = [
    ('player_stats', 2024),  # 2024 has full season
    ('player_stats', 2023),  # Get more historical data
    ('rosters', 2024),       # Player positions/teams
    ('injuries', 2024),      # Real injury data
]

for dataset, year in datasets_to_download:
    print(f"\n{dataset.upper()} {year}:")
    df = download_nflverse_file(dataset, year)
    if df is not None:
        print(f"   Columns: {list(df.columns)[:20]}")
        print()

print("\n" + "="*80)
print("DOWNLOAD COMPLETE")
print("="*80)
