"""Fetch play-by-play data from NFLverse.

Downloads 2025 NFL play-by-play data from nflverse-data releases.
This data enables training of 25+ additional prop markets:
- First/Last TD Scorer
- Longest plays (completion, rush, reception)
- Drive props
- Scoring sequence props
"""

import pandas as pd
import requests
from pathlib import Path
import time


def download_pbp_data(year=2025, file_format='csv'):
    """Download play-by-play data from NFLverse.

    Args:
        year: Season year
        file_format: 'csv' or 'parquet'

    Returns:
        DataFrame with play-by-play data
    """

    print(f"\n{'='*80}")
    print(f"DOWNLOADING {year} PLAY-BY-PLAY DATA")
    print(f"{'='*80}\n")

    # NFLverse release URL
    base_url = "https://github.com/nflverse/nflverse-data/releases/download/pbp"
    file_name = f"play_by_play_{year}.{file_format}"
    url = f"{base_url}/{file_name}"

    print(f"URL: {url}")
    print()

    # Output file
    output_dir = Path('inputs')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / file_name

    # Check if already downloaded
    if output_file.exists():
        print(f"✅ File already exists: {output_file}")
        print(f"   Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
        print()

        # Load and return
        if file_format == 'csv':
            df = pd.read_csv(output_file, low_memory=False)
        else:
            df = pd.read_parquet(output_file)

        print(f"✅ Loaded {len(df)} plays")
        return df

    # Download
    print(f"📥 Downloading {file_name}...")
    print()

    try:
        # Use requests with redirects
        response = requests.get(url, allow_redirects=True, timeout=120)

        if response.status_code == 200:
            # Save to file
            with open(output_file, 'wb') as f:
                f.write(response.content)

            file_size = output_file.stat().st_size / 1024 / 1024
            print(f"✅ Downloaded successfully: {file_size:.2f} MB")
            print(f"💾 Saved to: {output_file}")
            print()

            # Load into DataFrame
            if file_format == 'csv':
                df = pd.read_csv(output_file, low_memory=False)
            else:
                df = pd.read_parquet(output_file)

            print(f"✅ Loaded {len(df)} plays")
            return df

        else:
            print(f"❌ Download failed: HTTP {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return pd.DataFrame()

    except Exception as e:
        print(f"❌ Error downloading: {str(e)}")
        return pd.DataFrame()


def inspect_pbp_columns(pbp_df):
    """Inspect PBP data columns and show relevant fields."""

    print(f"\n{'='*80}")
    print(f"INSPECTING PLAY-BY-PLAY DATA")
    print(f"{'='*80}\n")

    print(f"Total plays: {len(pbp_df)}")
    print(f"Total columns: {len(pbp_df.columns)}")
    print()

    # Show all columns
    print("ALL COLUMNS:")
    print("-" * 80)
    for i, col in enumerate(pbp_df.columns, 1):
        print(f"{i:3d}. {col}")
    print()

    # Key fields for props
    print("\n" + "="*80)
    print("KEY FIELDS FOR PROP BETTING")
    print("="*80 + "\n")

    # TD Scorer fields
    td_fields = [col for col in pbp_df.columns if 'td' in col.lower() or 'touchdown' in col.lower()]
    print(f"TD/TOUCHDOWN FIELDS ({len(td_fields)}):")
    for field in td_fields:
        print(f"  - {field}")
    print()

    # Player fields
    player_fields = [col for col in pbp_df.columns if 'player' in col.lower() or 'passer' in col.lower() or 'rusher' in col.lower() or 'receiver' in col.lower()]
    print(f"PLAYER FIELDS ({len(player_fields)}):")
    for field in player_fields[:20]:  # First 20
        print(f"  - {field}")
    if len(player_fields) > 20:
        print(f"  ... and {len(player_fields) - 20} more")
    print()

    # Yards/distance fields
    yards_fields = [col for col in pbp_df.columns if 'yards' in col.lower() or 'distance' in col.lower()]
    print(f"YARDS/DISTANCE FIELDS ({len(yards_fields)}):")
    for field in yards_fields:
        print(f"  - {field}")
    print()

    # Drive fields
    drive_fields = [col for col in pbp_df.columns if 'drive' in col.lower()]
    print(f"DRIVE FIELDS ({len(drive_fields)}):")
    for field in drive_fields:
        print(f"  - {field}")
    print()

    # Scoring fields
    scoring_fields = [col for col in pbp_df.columns if 'score' in col.lower() or 'field_goal' in col.lower() or 'extra_point' in col.lower()]
    print(f"SCORING FIELDS ({len(scoring_fields)}):")
    for field in scoring_fields:
        print(f"  - {field}")
    print()

    # Game state fields
    game_fields = [col for col in pbp_df.columns if any(x in col.lower() for x in ['game', 'qtr', 'quarter', 'time', 'week'])]
    print(f"GAME STATE FIELDS ({len(game_fields)}):")
    for field in game_fields[:15]:
        print(f"  - {field}")
    if len(game_fields) > 15:
        print(f"  ... and {len(game_fields) - 15} more")
    print()


def analyze_touchdown_data(pbp_df):
    """Analyze TD scorer data."""

    print(f"\n{'='*80}")
    print(f"TOUCHDOWN SCORER ANALYSIS")
    print(f"{'='*80}\n")

    # Filter to TD plays
    td_plays = pbp_df[pbp_df['touchdown'] == 1].copy()

    print(f"Total TD plays: {len(td_plays)}")
    print()

    # Check for td_player fields
    if 'td_player_name' in td_plays.columns:
        print("✅ td_player_name field exists")
        print(f"   Unique TD scorers: {td_plays['td_player_name'].nunique()}")
        print(f"   Non-null TD scorers: {td_plays['td_player_name'].notna().sum()}")
        print()

        # Top TD scorers
        print("Top 10 TD Scorers:")
        top_scorers = td_plays['td_player_name'].value_counts().head(10)
        for player, tds in top_scorers.items():
            print(f"  {player:30s}: {tds} TDs")
        print()
    else:
        print("❌ td_player_name field NOT found")
        print()

    if 'td_player_id' in td_plays.columns:
        print("✅ td_player_id field exists")
    else:
        print("❌ td_player_id field NOT found")

    print()

    # TD types
    if 'td_team' in td_plays.columns:
        print("✅ td_team field exists")
        print(f"   Teams with TDs: {td_plays['td_team'].nunique()}")
        print()

    # Show sample TD play
    print("SAMPLE TD PLAY:")
    print("-" * 80)
    if len(td_plays) > 0:
        sample_cols = ['game_id', 'week', 'posteam', 'td_player_name', 'td_team', 'yards_gained', 'play_type', 'desc']
        sample_cols = [col for col in sample_cols if col in td_plays.columns]
        sample = td_plays[sample_cols].iloc[0]
        for col in sample_cols:
            print(f"{col:20s}: {sample[col]}")
    print()


def analyze_longest_plays(pbp_df):
    """Analyze longest play data."""

    print(f"\n{'='*80}")
    print(f"LONGEST PLAY ANALYSIS")
    print(f"{'='*80}\n")

    # Check for yards_gained
    if 'yards_gained' not in pbp_df.columns:
        print("❌ yards_gained field NOT found")
        return

    print("✅ yards_gained field exists")
    print()

    # Overall stats
    print("YARDS GAINED STATS:")
    print(f"  Mean: {pbp_df['yards_gained'].mean():.2f}")
    print(f"  Median: {pbp_df['yards_gained'].median():.2f}")
    print(f"  Max: {pbp_df['yards_gained'].max():.0f}")
    print(f"  Min: {pbp_df['yards_gained'].min():.0f}")
    print()

    # By play type
    if 'play_type' in pbp_df.columns:
        print("LONGEST PLAYS BY TYPE:")

        for play_type in ['pass', 'run']:
            type_plays = pbp_df[pbp_df['play_type'] == play_type]
            if len(type_plays) > 0:
                max_yards = type_plays['yards_gained'].max()
                print(f"  {play_type.capitalize():10s}: {max_yards:.0f} yards")

        print()

    # Top 5 longest plays
    print("TOP 5 LONGEST PLAYS:")
    longest = pbp_df.nlargest(5, 'yards_gained')
    sample_cols = ['week', 'posteam', 'play_type', 'yards_gained', 'desc']
    sample_cols = [col for col in sample_cols if col in longest.columns]

    for idx, play in longest[sample_cols].iterrows():
        print(f"\n  Week {play['week']}: {play['yards_gained']:.0f} yards ({play['play_type']})")
        if 'desc' in play:
            desc = str(play['desc'])[:80]
            print(f"  {desc}...")

    print()


def analyze_drive_data(pbp_df):
    """Analyze drive data."""

    print(f"\n{'='*80}")
    print(f"DRIVE DATA ANALYSIS")
    print(f"{'='*80}\n")

    # Check for drive field
    if 'fixed_drive' not in pbp_df.columns and 'drive' not in pbp_df.columns:
        print("❌ No drive field found")
        return

    drive_field = 'fixed_drive' if 'fixed_drive' in pbp_df.columns else 'drive'
    print(f"✅ Using drive field: {drive_field}")
    print()

    # Group by game and drive
    if 'game_id' in pbp_df.columns:
        drive_stats = pbp_df.groupby(['game_id', drive_field]).agg({
            'play_id': 'count',  # Number of plays
            'yards_gained': 'sum'  # Total yards
        }).reset_index()

        drive_stats.columns = ['game_id', 'drive', 'plays', 'yards']

        print("DRIVE STATS:")
        print(f"  Total drives: {len(drive_stats)}")
        print(f"  Avg plays per drive: {drive_stats['plays'].mean():.2f}")
        print(f"  Avg yards per drive: {drive_stats['yards'].mean():.2f}")
        print(f"  Longest drive (plays): {drive_stats['plays'].max():.0f}")
        print(f"  Longest drive (yards): {drive_stats['yards'].max():.0f}")
        print()


def main():
    """Main function to fetch and inspect PBP data."""

    print(f"\n{'='*80}")
    print(f"NFLVERSE PLAY-BY-PLAY DATA FETCHER")
    print(f"{'='*80}\n")

    # Download 2025 data (try CSV first, can switch to parquet if too large)
    pbp_df = download_pbp_data(year=2025, file_format='csv')

    if len(pbp_df) == 0:
        print("\n❌ Failed to download PBP data")
        print("\nTROUBLESHOOTING:")
        print("1. Check internet connection")
        print("2. Verify NFLverse releases are accessible:")
        print("   https://github.com/nflverse/nflverse-data/releases/tag/pbp")
        print("3. Try downloading manually and placing in inputs/ folder")
        print("4. Try parquet format: download_pbp_data(year=2025, file_format='parquet')")
        return

    # Inspect columns
    inspect_pbp_columns(pbp_df)

    # Analyze TD data
    analyze_touchdown_data(pbp_df)

    # Analyze longest plays
    analyze_longest_plays(pbp_df)

    # Analyze drive data
    analyze_drive_data(pbp_df)

    # Final summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}\n")

    print(f"✅ Successfully downloaded and inspected 2025 PBP data")
    print(f"   Total plays: {len(pbp_df)}")
    print(f"   Total columns: {len(pbp_df.columns)}")
    print()

    print("NEXT STEPS:")
    print("1. Review column list above")
    print("2. Identify fields needed for each prop type")
    print("3. Create feature engineering scripts:")
    print("   - backend/analysis/process_pbp_features.py")
    print("4. Train first models:")
    print("   - backend/analysis/train_first_td_scorer.py")
    print("   - backend/analysis/train_longest_plays.py")
    print()


if __name__ == '__main__':
    main()
