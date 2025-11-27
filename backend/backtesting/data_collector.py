"""Historical Data Collector.

Fetches historical NFL data from various sources for backtesting.
Supports nfl-data-py, ESPN API, and Pro Football Reference.
"""

from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
import json


class HistoricalDataCollector:
    """Collects and caches historical NFL data."""

    def __init__(self, output_dir: str = 'inputs/historical'):
        """Initialize data collector.

        Args:
            output_dir: Directory to save collected data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.injuries_dir = self.output_dir / 'injuries'
        self.injuries_dir.mkdir(exist_ok=True)

    def collect_season_data(
        self,
        season: int,
        source: str = 'nfl_data_py'
    ) -> Dict[str, pd.DataFrame]:
        """Collect all data for a season.

        Args:
            season: Season year
            source: Data source ('nfl_data_py', 'espn', 'pfr')

        Returns:
            Dictionary with DataFrames for games, stats, injuries
        """
        print(f"Collecting data for {season} season from {source}...")

        data = {}

        if source == 'nfl_data_py':
            data = self._collect_from_nfl_data_py(season)
        elif source == 'espn':
            data = self._collect_from_espn(season)
        else:
            print(f"Source {source} not yet implemented")
            return {}

        # Save collected data
        self._save_season_data(season, data)

        return data

    def _collect_from_nfl_data_py(self, season: int) -> Dict[str, pd.DataFrame]:
        """Collect data using nfl_data_py package.

        Args:
            season: Season year

        Returns:
            Dictionary with DataFrames
        """
        data = {}

        try:
            import nfl_data_py as nfl

            print(f"  Fetching games...")
            # Get schedule/games
            games = nfl.import_schedules([season])
            data['games'] = self._process_games(games)
            print(f"    ✓ {len(data['games'])} games")

            print(f"  Fetching player stats...")
            # Try weekly data first, fallback to pbp aggregation
            try:
                weekly_stats = nfl.import_weekly_data([season])
                data['player_stats'] = self._process_player_stats(weekly_stats)
                print(f"    ✓ {len(data['player_stats'])} player-week records (from weekly_data)")
            except Exception as e:
                print(f"    ! Weekly data not available: {e}")
                print(f"    Aggregating from play-by-play data...")
                pbp = nfl.import_pbp_data([season])
                data['player_stats'] = self._aggregate_stats_from_pbp(pbp)
                print(f"    ✓ {len(data['player_stats'])} player-week records (from pbp)")

            print(f"  Fetching rosters...")
            # Get rosters for position classification
            rosters = nfl.import_weekly_rosters([season])
            data['rosters'] = rosters
            print(f"    ✓ {len(rosters)} players")

            print(f"  Fetching injuries (if available)...")
            # Injuries - may need separate source
            try:
                injuries = nfl.import_injuries([season])
                data['injuries'] = self._process_injuries(injuries)
                print(f"    ✓ {len(data['injuries'])} injury reports")
            except Exception as e:
                print(f"    ! Injury data not available: {e}")
                data['injuries'] = pd.DataFrame()

        except ImportError:
            print("  ! nfl_data_py not installed")
            print("    Install with: pip install nfl-data-py")
            return {}

        except Exception as e:
            print(f"  ! Error collecting data: {e}")
            return {}

        return data

    def _collect_from_espn(self, season: int) -> Dict[str, pd.DataFrame]:
        """Collect data from ESPN API.

        Args:
            season: Season year

        Returns:
            Dictionary with DataFrames
        """
        # ESPN API integration
        # This would use ESPN's public API
        print("  ESPN API collection not yet implemented")
        print("  Alternative: Use nfl_data_py which pulls from ESPN")
        return {}

    def _process_games(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Process games DataFrame for backtesting.

        Args:
            games_df: Raw games DataFrame

        Returns:
            Processed DataFrame
        """
        processed = pd.DataFrame()

        # Standard fields
        field_mapping = {
            'game_id': 'game_id',
            'season': 'season',
            'week': 'week',
            'gameday': 'game_date',
            'home_team': 'home_team',
            'away_team': 'away_team',
            'home_score': 'home_score',
            'away_score': 'away_score',
            'temp': 'temperature',
            'wind': 'wind_speed',
        }

        for source_col, target_col in field_mapping.items():
            if source_col in games_df.columns:
                processed[target_col] = games_df[source_col]

        # Add derived fields
        if 'roof' in games_df.columns:
            processed['is_dome'] = games_df['roof'].isin(['dome', 'closed'])

        # Determine primetime
        if 'gametime' in games_df.columns:
            processed['is_primetime'] = games_df['gametime'].apply(
                lambda x: '20:' in str(x) or '21:' in str(x) if pd.notna(x) else False
            )

        # Precipitation (would need weather data)
        processed['precipitation'] = 'none'  # Default

        # Division games (would need division mapping)
        processed['is_division_game'] = False  # Default

        processed['is_playoff'] = games_df.get('game_type', '') == 'REG'

        return processed

    def _aggregate_stats_from_pbp(self, pbp_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate player stats from play-by-play data.

        Used when weekly aggregated data isn't available (e.g., current season).

        Args:
            pbp_df: Play-by-play DataFrame

        Returns:
            Aggregated player stats DataFrame
        """
        all_stats = []
        season = pbp_df['season'].iloc[0] if len(pbp_df) > 0 else 2025

        # Aggregate passing stats
        passing = pbp_df[pbp_df['pass'] == 1].groupby(['passer_player_id', 'passer_player_name', 'posteam', 'week']).agg({
            'complete_pass': 'sum',
            'pass_attempt': 'sum',
            'passing_yards': 'sum',
            'pass_touchdown': 'sum',
            'interception': 'sum'
        }).reset_index()

        passing = passing.rename(columns={
            'passer_player_id': 'player_id',
            'passer_player_name': 'player',
            'posteam': 'recent_team',
            'complete_pass': 'completions',
            'pass_attempt': 'attempts',
            'pass_touchdown': 'passing_tds'
        })
        passing['position'] = 'QB'
        passing['season'] = season

        # Aggregate rushing stats
        rushing = pbp_df[pbp_df['rush'] == 1].groupby(['rusher_player_id', 'rusher_player_name', 'posteam', 'week']).agg({
            'rush_attempt': 'sum',
            'rushing_yards': 'sum',
            'rush_touchdown': 'sum'
        }).reset_index()

        rushing = rushing.rename(columns={
            'rusher_player_id': 'player_id',
            'rusher_player_name': 'player',
            'posteam': 'recent_team',
            'rush_attempt': 'carries',
            'rush_touchdown': 'rushing_tds'
        })
        rushing['position'] = 'RB'  # Simplified
        rushing['season'] = season

        # Aggregate receiving stats
        receiving = pbp_df[pbp_df['pass'] == 1].groupby(['receiver_player_id', 'receiver_player_name', 'posteam', 'week']).agg({
            'complete_pass': 'sum',
            'pass_attempt': 'sum',
            'receiving_yards': 'sum',
            'pass_touchdown': 'sum'
        }).reset_index()

        receiving = receiving.rename(columns={
            'receiver_player_id': 'player_id',
            'receiver_player_name': 'player',
            'posteam': 'recent_team',
            'complete_pass': 'receptions',
            'pass_attempt': 'targets',
            'pass_touchdown': 'receiving_tds'
        })
        receiving['position'] = 'WR'  # Simplified
        receiving['season'] = season

        # Combine all stats
        all_stats = pd.concat([passing, rushing, receiving], ignore_index=True)

        # Add missing columns with defaults
        for col in ['completions', 'attempts', 'passing_yards', 'passing_tds', 'carries', 'rushing_yards', 'rushing_tds', 'receptions', 'targets', 'receiving_yards', 'receiving_tds']:
            if col not in all_stats.columns:
                all_stats[col] = 0

        all_stats = all_stats.fillna(0)

        # Add team column and other required fields
        all_stats['team'] = all_stats['recent_team']
        all_stats['player_name'] = all_stats['player']
        all_stats['player_display_name'] = all_stats['player']

        # Calculate fantasy points (PPR)
        all_stats['fantasy_points_ppr'] = (
            all_stats.get('passing_yards', 0) * 0.04 +
            all_stats.get('passing_tds', 0) * 4 +
            all_stats.get('rushing_yards', 0) * 0.1 +
            all_stats.get('rushing_tds', 0) * 6 +
            all_stats.get('receptions', 0) * 1 +
            all_stats.get('receiving_yards', 0) * 0.1 +
            all_stats.get('receiving_tds', 0) * 6
        )
        all_stats['fantasy_points'] = all_stats['fantasy_points_ppr'] - all_stats.get('receptions', 0)
        all_stats['points'] = all_stats['fantasy_points']

        return all_stats

    def _process_player_stats(self, stats_df: pd.DataFrame) -> pd.DataFrame:
        """Process player stats DataFrame.

        Args:
            stats_df: Raw player stats

        Returns:
            Processed DataFrame
        """
        # Keep relevant columns
        keep_cols = [
            'player_id', 'player_name', 'player_display_name',
            'position', 'recent_team', 'opponent_team', 'week', 'season',
            'completions', 'attempts', 'passing_yards', 'passing_tds',
            'carries', 'rushing_yards', 'rushing_tds',
            'receptions', 'targets', 'receiving_yards', 'receiving_tds',
            'fantasy_points', 'fantasy_points_ppr'
        ]

        processed = stats_df[[col for col in keep_cols if col in stats_df.columns]].copy()

        # Rename for consistency
        if 'recent_team' in processed.columns:
            processed['team'] = processed['recent_team']

        if 'player_display_name' in processed.columns:
            processed['player'] = processed['player_display_name']
        elif 'player_name' in processed.columns:
            processed['player'] = processed['player_name']

        # Fill NaN with 0 for numeric columns
        numeric_cols = processed.select_dtypes(include=['float64', 'int64']).columns
        processed[numeric_cols] = processed[numeric_cols].fillna(0)

        # Add points scored (approximate from fantasy points)
        if 'fantasy_points' in processed.columns:
            processed['points'] = processed['fantasy_points']

        return processed

    def _process_injuries(self, injuries_df: pd.DataFrame) -> pd.DataFrame:
        """Process injuries DataFrame.

        Args:
            injuries_df: Raw injuries

        Returns:
            Processed DataFrame
        """
        if injuries_df.empty:
            return pd.DataFrame()

        # Keep relevant columns
        keep_cols = [
            'season', 'week', 'team', 'position',
            'full_name', 'gsis_id',
            'report_status', 'report_primary_injury', 'practice_status'
        ]

        processed = injuries_df[[col for col in keep_cols if col in injuries_df.columns]].copy()

        # Rename full_name to player for consistency
        if 'full_name' in processed.columns:
            processed['player'] = processed['full_name']

        # Map injury status
        status_mapping = {
            'Out': 'OUT',
            'Doubtful': 'DOUBTFUL',
            'Questionable': 'QUESTIONABLE',
            'Probable': 'QUESTIONABLE'
        }

        if 'report_status' in processed.columns:
            processed['injury_status'] = processed['report_status'].map(status_mapping)
        else:
            processed['injury_status'] = 'QUESTIONABLE'

        return processed

    def _save_season_data(self, season: int, data: Dict[str, pd.DataFrame]):
        """Save season data to files.

        Args:
            season: Season year
            data: Dictionary with DataFrames
        """
        print(f"  Saving data...")

        # Save games
        if 'games' in data and not data['games'].empty:
            games_file = self.output_dir / f'games_{season}.csv'
            data['games'].to_csv(games_file, index=False)
            print(f"    ✓ Saved {games_file}")

        # Save player stats
        if 'player_stats' in data and not data['player_stats'].empty:
            stats_file = self.output_dir / f'player_stats_{season}_all.csv'
            data['player_stats'].to_csv(stats_file, index=False)
            print(f"    ✓ Saved {stats_file}")

        # Save injuries
        if 'injuries' in data and not data['injuries'].empty:
            injuries_file = self.injuries_dir / f'injuries_{season}.csv'
            data['injuries'].to_csv(injuries_file, index=False)
            print(f"    ✓ Saved {injuries_file}")

        # Save metadata
        metadata = {
            'season': season,
            'collected_at': datetime.now().isoformat(),
            'source': 'nfl_data_py',
            'record_counts': {
                'games': len(data.get('games', [])),
                'player_stats': len(data.get('player_stats', [])),
                'injuries': len(data.get('injuries', []))
            }
        }

        metadata_file = self.output_dir / f'metadata_{season}.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"    ✓ Saved {metadata_file}")

    def collect_multiple_seasons(
        self,
        seasons: List[int],
        source: str = 'nfl_data_py'
    ):
        """Collect data for multiple seasons.

        Args:
            seasons: List of season years
            source: Data source
        """
        print(f"Collecting data for {len(seasons)} seasons...")

        for season in seasons:
            print(f"\n{'='*60}")
            print(f"Season {season}")
            print(f"{'='*60}")

            try:
                self.collect_season_data(season, source)
            except Exception as e:
                print(f"  ! Error collecting {season}: {e}")
                continue

        print(f"\n{'='*60}")
        print("Data collection complete!")
        print(f"{'='*60}")

    def verify_data_availability(self, seasons: List[int]) -> Dict[int, Dict]:
        """Verify what data is available for each season.

        Args:
            seasons: List of season years

        Returns:
            Dictionary with availability status for each season
        """
        availability = {}

        for season in seasons:
            games_file = self.output_dir / f'games_{season}.csv'
            stats_file = self.output_dir / f'player_stats_{season}_all.csv'
            injuries_file = self.injuries_dir / f'injuries_{season}.csv'

            availability[season] = {
                'games': games_file.exists(),
                'player_stats': stats_file.exists(),
                'injuries': injuries_file.exists(),
                'games_count': len(pd.read_csv(games_file)) if games_file.exists() else 0,
                'stats_count': len(pd.read_csv(stats_file)) if stats_file.exists() else 0,
                'injuries_count': len(pd.read_csv(injuries_file)) if injuries_file.exists() else 0
            }

        return availability

    def generate_data_report(self, seasons: List[int]) -> str:
        """Generate a report on available data.

        Args:
            seasons: List of season years

        Returns:
            Markdown-formatted report
        """
        availability = self.verify_data_availability(seasons)

        report = ["# Historical Data Availability Report\n"]
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append("\n## Summary\n")

        report.append("| Season | Games | Player Stats | Injuries | Status |")
        report.append("|--------|------:|--------------:|---------:|--------|")

        for season, avail in availability.items():
            status = "✅ Complete" if all([avail['games'], avail['player_stats']]) else "⚠️ Incomplete"
            report.append(
                f"| {season} | {avail['games_count']} | {avail['stats_count']} | "
                f"{avail['injuries_count']} | {status} |"
            )

        report.append("\n## Missing Data\n")

        for season, avail in availability.items():
            missing = []
            if not avail['games']:
                missing.append('games')
            if not avail['player_stats']:
                missing.append('player_stats')
            if not avail['injuries']:
                missing.append('injuries')

            if missing:
                report.append(f"- **{season}**: Missing {', '.join(missing)}")

        if not any(any([not v['games'], not v['player_stats']]) for v in availability.values()):
            report.append("\nAll required data is available!")

        return "\n".join(report)


if __name__ == "__main__":
    # Initialize collector
    collector = HistoricalDataCollector()

    # Collect data for recent seasons (2020-2025)
    # Note: 2025 uses play-by-play aggregation (weekly data not yet published)
    seasons_to_collect = [2020, 2021, 2022, 2023, 2024, 2025]

    print("Historical Data Collector")
    print("=" * 60)

    # Check what we already have
    print("\nChecking existing data...")
    availability = collector.verify_data_availability(seasons_to_collect)

    for season, avail in availability.items():
        status = "✓" if all([avail['games'], avail['player_stats']]) else "✗"
        print(f"  {season}: {status} (Games: {avail['games_count']}, Stats: {avail['stats_count']})")

    # Collect missing data
    missing_seasons = [s for s, a in availability.items() if not all([a['games'], a['player_stats']])]

    if missing_seasons:
        print(f"\nCollecting data for {len(missing_seasons)} seasons...")
        response = input("Proceed with data collection? (y/n): ")

        if response.lower() == 'y':
            collector.collect_multiple_seasons(missing_seasons)
    else:
        print("\nAll data already collected!")

    # Generate report
    print("\n" + collector.generate_data_report(seasons_to_collect))
