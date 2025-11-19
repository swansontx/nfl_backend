"""Fetch prop betting lines from The Odds API with historical snapshot tracking.

Downloads player prop lines (passing yards, rushing yards, receiving yards, TDs, etc.)
from major sportsbooks and tracks line movement over time for trending analysis.

API Documentation: https://the-odds-api.com/

Prop Markets Available:
- player_pass_yds (QB passing yards)
- player_pass_tds (QB passing TDs)
- player_pass_completions (QB completions)
- player_rush_yds (RB/QB rushing yards)
- player_rush_tds (Rushing TDs)
- player_receptions (Receptions)
- player_reception_yds (Receiving yards)
- player_reception_tds (Receiving TDs)
- player_anytime_td (Anytime TD scorer)

Line Movement Tracking:
- Captures historical snapshots with timestamps
- Tracks opening line vs current line
- Calculates line movement (e.g., +2.5 yards over 7 days)
- Stores vig/juice for both OVER and UNDER sides
- Identifies "hot movers" (2+ point moves)

Output: JSON files with timestamped snapshots for trending analysis
"""

from pathlib import Path
import argparse
import requests
import json
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os


class PropLineFetcher:
    """Fetches prop betting lines from The Odds API with movement tracking."""

    def __init__(self, api_key: str):
        """Initialize with API key.

        Args:
            api_key: The Odds API key
        """
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4"
        self.sport = "americanfootball_nfl"

        # Prop markets to fetch
        self.prop_markets = [
            'player_pass_yds',
            'player_pass_tds',
            'player_pass_completions',
            'player_rush_yds',
            'player_rush_tds',
            'player_receptions',
            'player_reception_yds',
            'player_anytime_td'
        ]

        # Sportsbooks to include (major US books)
        self.bookmakers = [
            'draftkings',
            'fanduel',
            'williamhill_us',  # Caesars
            'betmgm',
            'pointsbetus',
            'unibet_us'
        ]

    def fetch_upcoming_games(self) -> List[Dict]:
        """Fetch list of upcoming NFL games.

        Returns:
            List of game dictionaries with game_id, commence_time, teams
        """
        url = f"{self.base_url}/sports/{self.sport}/events"
        params = {
            'apiKey': self.api_key,
            'dateFormat': 'iso'
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            games = response.json()
            print(f"✓ Found {len(games)} upcoming NFL games")

            return games

        except requests.exceptions.RequestException as e:
            print(f"✗ Error fetching games: {e}")
            return []

    def fetch_prop_odds_for_game(
        self,
        game_id: str,
        market: str,
        bookmakers: Optional[List[str]] = None
    ) -> Dict:
        """Fetch prop odds for a specific game and market.

        Args:
            game_id: The Odds API game ID
            market: Prop market (e.g., 'player_pass_yds')
            bookmakers: List of bookmaker keys (optional)

        Returns:
            Dict with prop odds data
        """
        url = f"{self.base_url}/sports/{self.sport}/events/{game_id}/odds"

        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': market,
            'oddsFormat': 'american',
            'dateFormat': 'iso'
        }

        if bookmakers:
            params['bookmakers'] = ','.join(bookmakers)

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Check remaining requests
            remaining = response.headers.get('x-requests-remaining', 'unknown')
            print(f"  {market}: {remaining} requests remaining")

            return data

        except requests.exceptions.RequestException as e:
            print(f"  ✗ Error fetching {market} for {game_id}: {e}")
            return {}

    def process_prop_data_with_movement(
        self,
        raw_data: Dict,
        market: str,
        snapshot_time: datetime
    ) -> List[Dict]:
        """Process raw API response into structured prop data with both sides.

        Args:
            raw_data: Raw API response
            market: Market type
            snapshot_time: Timestamp of this snapshot

        Returns:
            List of processed prop dictionaries with OVER/UNDER sides
        """
        if not raw_data or 'bookmakers' not in raw_data:
            return []

        processed_props = []

        for bookmaker in raw_data.get('bookmakers', []):
            bookmaker_name = bookmaker.get('key', '')

            for market_data in bookmaker.get('markets', []):
                # Group outcomes by player (OVER and UNDER for same line)
                player_props = {}

                for outcome in market_data.get('outcomes', []):
                    player_name = outcome.get('description', '')
                    point = outcome.get('point')
                    price = outcome.get('price')  # American odds
                    name = outcome.get('name', '')  # 'Over' or 'Under'

                    if player_name and point is not None:
                        if player_name not in player_props:
                            player_props[player_name] = {
                                'player_name': player_name,
                                'market': market,
                                'bookmaker': bookmaker_name,
                                'line': point,
                                'timestamp': snapshot_time.isoformat(),
                                'over_odds': None,
                                'under_odds': None
                            }

                        if name.lower() == 'over':
                            player_props[player_name]['over_odds'] = price
                        elif name.lower() == 'under':
                            player_props[player_name]['under_odds'] = price

                # Add to processed props
                processed_props.extend(player_props.values())

        return processed_props

    def calculate_line_movement(
        self,
        current_snapshot: Dict,
        previous_snapshot: Optional[Dict]
    ) -> Dict:
        """Calculate line movement between snapshots.

        Args:
            current_snapshot: Current snapshot data
            previous_snapshot: Previous snapshot data (opening line or earlier)

        Returns:
            Dict with movement metrics
        """
        if not previous_snapshot:
            return {
                'line_movement': 0.0,
                'movement_direction': 'neutral',
                'is_hot_mover': False,
                'opening_line': current_snapshot.get('line'),
                'current_line': current_snapshot.get('line')
            }

        opening_line = previous_snapshot.get('line', 0)
        current_line = current_snapshot.get('line', 0)

        movement = current_line - opening_line

        # Determine if "hot mover" (2+ point move)
        is_hot_mover = abs(movement) >= 2.0

        # Direction
        if movement > 0.5:
            direction = 'up'
        elif movement < -0.5:
            direction = 'down'
        else:
            direction = 'neutral'

        return {
            'line_movement': round(movement, 1),
            'movement_direction': direction,
            'is_hot_mover': is_hot_mover,
            'opening_line': opening_line,
            'current_line': current_line,
            'opening_timestamp': previous_snapshot.get('timestamp'),
            'days_tracked': self._calculate_days_between(
                previous_snapshot.get('timestamp'),
                current_snapshot.get('timestamp')
            )
        }

    def _calculate_days_between(self, start_iso: str, end_iso: str) -> int:
        """Calculate days between two ISO timestamps."""
        if not start_iso or not end_iso:
            return 0

        try:
            start = datetime.fromisoformat(start_iso)
            end = datetime.fromisoformat(end_iso)
            return (end - start).days
        except:
            return 0

    def fetch_snapshot_with_movement(
        self,
        output_dir: Path,
        week: Optional[int] = None,
        load_previous: bool = True
    ) -> Dict:
        """Fetch current snapshot and calculate movement from previous snapshot.

        Args:
            output_dir: Directory to save/load snapshots
            week: Week number (for filename)
            load_previous: Whether to load previous snapshot for movement calc

        Returns:
            Dict with snapshot data and movement metrics
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        snapshot_time = datetime.now()

        print(f"\n{'='*60}")
        print(f"Fetching NFL Player Props Snapshot")
        print(f"Timestamp: {snapshot_time.isoformat()}")
        print(f"{'='*60}\n")

        # Get upcoming games
        games = self.fetch_upcoming_games()

        if not games:
            print("⚠️  No upcoming games found")
            return {'games': 0, 'props': 0}

        # Load previous snapshots for movement calculation
        previous_snapshots = {}
        if load_previous:
            previous_snapshots = self._load_latest_snapshot(output_dir, week)

        all_props_by_game = {}
        total_props = 0
        hot_movers = []

        for game in games:
            game_id = game.get('id')
            home_team = game.get('home_team', 'Unknown')
            away_team = game.get('away_team', 'Unknown')
            commence_time = game.get('commence_time', '')

            print(f"\n{away_team} @ {home_team}")
            print(f"  Game ID: {game_id}")
            print(f"  Kickoff: {commence_time}")

            game_props = {
                'game_id': game_id,
                'home_team': home_team,
                'away_team': away_team,
                'commence_time': commence_time,
                'snapshot_timestamp': snapshot_time.isoformat(),
                'props': {}
            }

            # Fetch each prop market
            for market in self.prop_markets:
                raw_data = self.fetch_prop_odds_for_game(
                    game_id,
                    market,
                    self.bookmakers
                )

                if raw_data:
                    props = self.process_prop_data_with_movement(
                        raw_data,
                        market,
                        snapshot_time
                    )

                    if props:
                        # Calculate movement for each prop
                        props_with_movement = []
                        for prop in props:
                            # Find previous snapshot for this player/bookmaker
                            prev_prop = self._find_previous_prop(
                                previous_snapshots,
                                game_id,
                                market,
                                prop['player_name'],
                                prop['bookmaker']
                            )

                            # Calculate movement
                            movement = self.calculate_line_movement(prop, prev_prop)
                            prop['movement'] = movement

                            # Track hot movers
                            if movement['is_hot_mover']:
                                hot_movers.append({
                                    'player': prop['player_name'],
                                    'market': market,
                                    'movement': movement['line_movement'],
                                    'opening_line': movement['opening_line'],
                                    'current_line': movement['current_line'],
                                    'bookmaker': prop['bookmaker']
                                })

                            props_with_movement.append(prop)

                        game_props['props'][market] = props_with_movement
                        total_props += len(props_with_movement)
                        print(f"    ✓ {market}: {len(props_with_movement)} props")

                # Rate limit: sleep between requests
                time.sleep(0.5)

            all_props_by_game[game_id] = game_props

        # Save snapshot with timestamp
        week_str = f"week_{week}" if week else "current"
        timestamp_str = snapshot_time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"snapshot_{week_str}_{timestamp_str}.json"

        with open(output_file, 'w') as f:
            json.dump(all_props_by_game, f, indent=2)

        # Also save as "latest" for easy access
        latest_file = output_dir / f"snapshot_{week_str}_latest.json"
        with open(latest_file, 'w') as f:
            json.dump(all_props_by_game, f, indent=2)

        # Save hot movers summary
        if hot_movers:
            hot_movers_file = output_dir / f"hot_movers_{week_str}_{timestamp_str}.json"
            with open(hot_movers_file, 'w') as f:
                json.dump(hot_movers, f, indent=2)

        print(f"\n{'='*60}")
        print(f"✓ Fetched props for {len(games)} games")
        print(f"✓ Total props: {total_props}")
        print(f"✓ Hot movers (2+ pts): {len(hot_movers)}")
        print(f"✓ Saved to: {output_file}")
        if hot_movers:
            print(f"✓ Hot movers: {hot_movers_file}")
        print(f"{'='*60}\n")

        return {
            'games': len(games),
            'props': total_props,
            'hot_movers': len(hot_movers),
            'snapshot_time': snapshot_time.isoformat(),
            'output_file': str(output_file)
        }

    def _load_latest_snapshot(self, output_dir: Path, week: Optional[int]) -> Dict:
        """Load latest snapshot for movement calculation."""
        week_str = f"week_{week}" if week else "current"
        latest_file = output_dir / f"snapshot_{week_str}_latest.json"

        if not latest_file.exists():
            return {}

        try:
            with open(latest_file, 'r') as f:
                return json.load(f)
        except:
            return {}

    def _find_previous_prop(
        self,
        previous_snapshots: Dict,
        game_id: str,
        market: str,
        player_name: str,
        bookmaker: str
    ) -> Optional[Dict]:
        """Find previous prop snapshot for comparison."""
        game_data = previous_snapshots.get(game_id, {})
        props = game_data.get('props', {}).get(market, [])

        for prop in props:
            if (prop.get('player_name') == player_name and
                prop.get('bookmaker') == bookmaker):
                return prop

        return None

    def get_trending_props(
        self,
        snapshots_dir: Path,
        week: Optional[int] = None,
        lookback_days: int = 7
    ) -> Dict:
        """Analyze prop line trends over time.

        Args:
            snapshots_dir: Directory with historical snapshots
            week: Week number
            lookback_days: Days to look back for trending

        Returns:
            Dict with trending analysis (hot movers, sustained trends)
        """
        week_str = f"week_{week}" if week else "current"

        # Find all snapshots for this week
        snapshot_files = sorted(
            snapshots_dir.glob(f"snapshot_{week_str}_*.json"),
            reverse=True  # Most recent first
        )

        if len(snapshot_files) < 2:
            return {
                'error': 'Insufficient snapshots for trending analysis',
                'snapshots_found': len(snapshot_files)
            }

        # Load current and opening snapshots
        with open(snapshot_files[0], 'r') as f:
            current = json.load(f)

        with open(snapshot_files[-1], 'r') as f:
            opening = json.load(f)

        # Analyze trends
        hot_movers = []
        sustained_trends = []

        for game_id, game_data in current.items():
            for market, props in game_data.get('props', {}).items():
                for prop in props:
                    movement = prop.get('movement', {})

                    # Hot movers (2+ point moves)
                    if movement.get('is_hot_mover'):
                        hot_movers.append({
                            'player': prop['player_name'],
                            'market': market,
                            'movement': movement['line_movement'],
                            'direction': movement['movement_direction'],
                            'opening_line': movement['opening_line'],
                            'current_line': movement['current_line'],
                            'bookmaker': prop['bookmaker'],
                            'badge': f"{movement['line_movement']:+.1f} pts ({movement.get('days_tracked', 0)} days)"
                        })

                    # Sustained trends (check multiple snapshots)
                    if len(snapshot_files) >= 3:
                        # TODO: Check middle snapshots for sustained direction
                        pass

        return {
            'hot_movers': sorted(hot_movers, key=lambda x: abs(x['movement']), reverse=True),
            'sustained_trends': sustained_trends,
            'snapshots_analyzed': len(snapshot_files),
            'current_timestamp': current.get(list(current.keys())[0], {}).get('snapshot_timestamp') if current else None
        }


def fetch_prop_lines(
    api_key: str,
    output_dir: Path,
    week: Optional[int] = None
) -> Dict:
    """Main function to fetch prop betting lines with movement tracking.

    Args:
        api_key: The Odds API key
        output_dir: Output directory
        week: Week number (optional)

    Returns:
        Summary dict
    """
    if not api_key:
        print("⚠️  No API key provided")
        print("Set ODDS_API_KEY environment variable or pass --api-key")
        return {'error': 'No API key'}

    fetcher = PropLineFetcher(api_key)
    return fetcher.fetch_snapshot_with_movement(output_dir, week)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fetch NFL player prop betting lines with movement tracking'
    )
    parser.add_argument('--api-key', type=str, default=None,
                       help='The Odds API key (or use ODDS_API_KEY env var)')
    parser.add_argument('--output', type=Path, default=Path('outputs/prop_lines'),
                       help='Output directory (default: outputs/prop_lines/)')
    parser.add_argument('--week', type=int, default=None,
                       help='Week number (for filename)')
    parser.add_argument('--analyze-trends', action='store_true',
                       help='Analyze trending props from historical snapshots')
    args = parser.parse_args()

    # Get API key from args or environment
    api_key = args.api_key or os.getenv('ODDS_API_KEY')

    if not api_key and not args.analyze_trends:
        print("\n⚠️  ERROR: No API key provided")
        print("\nTo use The Odds API:")
        print("1. Get a free API key at https://the-odds-api.com/")
        print("2. Set environment variable: export ODDS_API_KEY='your_key'")
        print("3. Or pass as argument: --api-key your_key")
        print("\nNote: Free tier includes 500 requests/month")
        exit(1)

    if args.analyze_trends:
        # Analyze existing snapshots
        fetcher = PropLineFetcher(api_key or 'dummy')
        trends = fetcher.get_trending_props(args.output, args.week)

        print("\n" + "="*60)
        print("PROP LINE TRENDING ANALYSIS")
        print("="*60)

        print(f"\nHot Movers (2+ point moves): {len(trends.get('hot_movers', []))}")
        for mover in trends.get('hot_movers', [])[:10]:  # Top 10
            direction_icon = "⬆️" if mover['direction'] == 'up' else "⬇️"
            print(f"  {direction_icon} {mover['player']} ({mover['market']})")
            print(f"     {mover['opening_line']} → {mover['current_line']} ({mover['badge']})")

    else:
        # Fetch new snapshot
        fetch_prop_lines(api_key, args.output, args.week)
