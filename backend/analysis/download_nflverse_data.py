"""Download necessary NFLverse data for training."""

import pandas as pd
import requests
from pathlib import Path

def download_file(url, output_path):
    """Download a file with progress."""
    print(f"Downloading {output_path.name}...")
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)
        print(f"✅ Downloaded {output_path.name} ({len(response.content) / 1024 / 1024:.1f} MB)")
        return True
    except Exception as e:
        print(f"❌ Failed to download {output_path.name}: {e}")
        return False

def main():
    base_url = "https://github.com/nflverse/nflverse-data/releases/download"
    inputs_dir = Path("/home/user/nfl_backend/inputs")

    files_to_download = [
        (f"{base_url}/player_stats/player_stats_2025.parquet", inputs_dir / "player_stats_2025.parquet"),
        (f"{base_url}/injuries/injuries_2025.parquet", inputs_dir / "injuries_2025.parquet"),
    ]

    print("="*80)
    print("DOWNLOADING NFLVERSE DATA")
    print("="*80 + "\n")

    for url, output_path in files_to_download:
        download_file(url, output_path)

    # Convert parquet to CSV if needed
    print("\nConverting parquet files to CSV...")
    for parquet_file in inputs_dir.glob("*.parquet"):
        csv_file = parquet_file.with_suffix('.csv')
        try:
            df = pd.read_parquet(parquet_file)
            df.to_csv(csv_file, index=False)
            print(f"✅ Converted {parquet_file.name} to CSV ({len(df)} rows)")
        except Exception as e:
            print(f"❌ Failed to convert {parquet_file.name}: {e}")

    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
