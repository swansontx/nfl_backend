"""Fetch prop betting lines from The Odds API with historical snapshot tracking.

Downloads comprehensive player prop lines from major sportsbooks and tracks line movement
over time for trending analysis and line shopping.

API Documentation: https://the-odds-api.com/sports-odds-data/betting-markets.html

Prop Markets Tracked (60+ markets):

FULL GAME PROPS:
- Passing: yards, TDs, completions, attempts, interceptions, longest
- Rushing: yards, TDs, attempts, longest
- Receiving: receptions, yards, TDs, longest
- Kicking: points, field goals, field goals made
- Touchdowns: anytime, first, last
- Defense: tackles+assists, sacks, interceptions
- Combos: pass+rush yards, pass+rush TDs, rush+rec yards

FIRST HALF (1H) PROPS:
- Passing: yards, TDs, completions
- Rushing: yards, TDs
- Receiving: receptions, yards
- Touchdowns: anytime TD

FIRST QUARTER (1Q) PROPS:
- Passing: yards, TDs, completions
- Rushing: yards, TDs
- Receiving: receptions, yards
- Touchdowns: anytime TD

SECOND HALF (2H), THIRD QUARTER (3Q), FOURTH QUARTER (4Q) PROPS:
- Passing yards, rushing yards, receiving yards

Line Movement Tracking:
- Captures historical snapshots with timestamps
- Tracks opening line vs current line
- Calculates line movement (e.g., +2.5 yards over 7 days)
- Stores vig/juice for both OVER and UNDER sides
- Identifies "hot movers" (2+ point moves)
- Detects sharp action (books moving in different directions)

Sportsbooks (DraftKings-only betting):
- DraftKings (PRIMARY - only book user can bet on)
- FanDuel, Caesars, BetMGM, PointsBet, Unibet (tracked for sharp action detection)

Sharp Action Detection:
- If DK moves alone = public money
- If other books move but DK stays flat = sharp money elsewhere
- If all books move together = consensus/steam move
- Track where smart money is going even if you can't bet there

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

        # Prop markets to fetch (comprehensive list from Odds API)
        # All prop markets we want to fetch
        # We'll dynamically check which are available per event
        self.prop_markets = [
            # FULL GAME - Passing props
            'player_pass_yds',
            'player_pass_tds',
            'player_pass_completions',
            'player_pass_attempts',
            'player_pass_interceptions',
            'player_pass_longest_completion',

            # FULL GAME - Rushing props
            'player_rush_yds',
            'player_rush_tds',
            'player_rush_attempts',
            'player_rush_longest',

            # FULL GAME - Receiving props
            'player_receptions',
            'player_reception_yds',
            'player_reception_tds',
            'player_reception_longest',

            # FULL GAME - Kicking props
            'player_kicking_points',
            'player_field_goals',
            'player_field_goals_made',

            # FULL GAME - Touchdown props
            'player_anytime_td',
            'player_first_td',
            'player_last_td',
            '1st_td_scorer',
            'last_td_scorer',

            # FULL GAME - Defensive props
            'player_tackles_assists',
            'player_sacks',
            'player_interceptions',

            # FULL GAME - Combo props
            'player_pass_rush_yds',
            'player_pass_tds_rush_tds',
            'player_receptions_rush_yds',
            'player_rush_reception_yds',

            # FIRST HALF (1H) props
            'player_1h_pass_yds',
            'player_1h_pass_tds',
            'player_1h_pass_completions',
            'player_1h_rush_yds',
            'player_1h_rush_tds',
            'player_1h_receptions',
            'player_1h_reception_yds',
            'player_1h_anytime_td',

            # FIRST QUARTER (1Q) props
            'player_1q_pass_yds',
            'player_1q_pass_tds',
            'player_1q_pass_completions',
            'player_1q_rush_yds',
            'player_1q_rush_tds',
            'player_1q_receptions',
            'player_1q_reception_yds',
            'player_1q_anytime_td',

            # SECOND HALF (2H) props
            'player_2h_pass_yds',
            'player_2h_pass_tds',
            'player_2h_rush_yds',
            'player_2h_receptions',
            'player_2h_reception_yds',

            # THIRD QUARTER (3Q) props
            'player_3q_pass_yds',
            'player_3q_rush_yds',
            'player_3q_reception_yds',

            # FOURTH QUARTER (4Q) props
            'player_4q_pass_yds',
            'player_4q_rush_yds',
            'player_4q_reception_yds'
        ]

        # Sportsbooks to track
        # User can ONLY bet on DraftKings, but we track others for:
        # - Sharp action detection (where is smart money moving?)
        # - Validation (if all books move together = consensus, if DK alone = public money)
        # - Steam detection (sudden moves on other books signal sharp action)
        self.bookmakers = [
            'draftkings',      # PRIMARY (only book user can bet on)
            'fanduel',         # Track for sharp action detection
            'williamhill_us',  # Caesars - track for movement validation
            'betmgm',          # Track for sharp money signals
            'pointsbetus',     # Track for consensus
            'unibet_us'        # Track for consensus
        ]

        # User can only bet on DraftKings
        self.primary_book = 'draftkings'

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

            # Handle 422 silently - market not available for this event
            if response.status_code == 422:
                return {}

            response.raise_for_status()

            data = response.json()

            # Check remaining requests
            remaining = response.headers.get('x-requests-remaining', 'unknown')
            print(f"  {market}: {remaining} requests remaining")

            return data

        except requests.exceptions.RequestException as e:
            # Only print real errors, not 422s
            if '422' not in str(e):
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

    def get_best_line(self, props: List[Dict], side: str = 'over') -> Optional[Dict]:
        """Find best line across all sportsbooks.

        Args:
            props: List of prop dicts for same player/market
            side: 'over' or 'under'

        Returns:
            Prop dict with best line (lowest for over, highest for under)
        """
        if not props:
            return None

        if side == 'over':
            # For OVER: want lowest line (easier to hit)
            return min(props, key=lambda p: p.get('line', float('inf')))
        else:
            # For UNDER: want highest line (easier to hit)
            return max(props, key=lambda p: p.get('line', float('-inf')))

    def get_draftkings_line(self, props: List[Dict]) -> Optional[Dict]:
        """Get DraftKings line (user's primary book).

        Args:
            props: List of prop dicts for same player/market

        Returns:
            DraftKings prop dict if available
        """
        for prop in props:
            if prop.get('bookmaker') == 'draftkings':
                return prop
        return None

    def analyze_sharp_action(self, props: List[Dict]) -> Dict:
        """Analyze sharp action across sportsbooks (DraftKings-only betting).

        Since user can only bet on DraftKings, this analyzes where sharp money
        is moving across other books to inform DK betting decisions.

        Args:
            props: List of prop dicts for same player/market (different books)

        Returns:
            Dict with DraftKings line and sharp action analysis
        """
        dk_line = self.get_draftkings_line(props)

        # Get lines from sharp books (FanDuel, Pinnacle are typically sharp)
        sharp_lines = [p for p in props if p.get('bookmaker') in ['fanduel', 'pinnacle', 'betmgm']]

        # Calculate sharp action signals
        sharp_action = {
            'dk_isolated': False,        # DK moving alone (public money)
            'dk_with_sharps': False,     # DK moving with sharp books (consensus)
            'sharps_opposite_dk': False, # Sharp books moving opposite DK (fade DK)
            'steam_move': False          # All books moving together (steam)
        }

        if dk_line and sharp_lines:
            dk_movement = dk_line.get('movement', {}).get('line_movement', 0)
            sharp_movements = [p.get('movement', {}).get('line_movement', 0) for p in sharp_lines]

            # Detect patterns
            if abs(dk_movement) >= 2.0:
                # DK moved significantly
                if all(abs(m) < 1.0 for m in sharp_movements):
                    sharp_action['dk_isolated'] = True  # Public money on DK
                elif all(abs(m) >= 2.0 for m in sharp_movements):
                    sharp_action['steam_move'] = True   # Everyone moving
                    sharp_action['dk_with_sharps'] = True

            elif any(abs(m) >= 2.0 for m in sharp_movements):
                # Sharp books moved but DK didn't
                sharp_action['sharps_opposite_dk'] = True

        return {
            'draftkings': dk_line,
            'sharp_books': sharp_lines,
            'sharp_action': sharp_action,
            'interpretation': self._interpret_sharp_action(sharp_action)
        }

    def _interpret_sharp_action(self, signals: Dict) -> str:
        """Interpret sharp action signals into betting guidance."""
        if signals['steam_move']:
            return "STEAM MOVE - All books moving together, strong consensus"
        elif signals['dk_isolated']:
            return "PUBLIC MONEY - DK moving alone, sharps staying away, FADE DK"
        elif signals['sharps_opposite_dk']:
            return "SHARP DISAGREEMENT - Sharp books moving opposite DK, follow sharps"
        elif signals['dk_with_sharps']:
            return "SHARP CONSENSUS - DK aligned with sharp books, good signal"
        else:
            return "NO CLEAR SIGNAL - Mixed or minimal movement"

    def get_trending_props(
        self,
        snapshots_dir: Path,
        week: Optional[int] = None,
        lookback_days: int = 7
    ) -> Dict:
        """Analyze prop line trends over time with 3-week sustained trend tracking.

        Args:
            snapshots_dir: Directory with historical snapshots
            week: Week number
            lookback_days: Days to look back for trending

        Returns:
            Dict with:
            - hot_movers: Week-over-week big changes (2+ pts)
            - sustained_trends: 3-week consistent direction patterns
            - timestamps, juice/vig, movement badges, visual indicators
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

        # Load snapshots (current + all historical)
        snapshots = []
        for snapshot_file in snapshot_files:
            with open(snapshot_file, 'r') as f:
                data = json.load(f)
                snapshots.append({
                    'file': snapshot_file.name,
                    'data': data
                })

        current = snapshots[0]['data']
        opening = snapshots[-1]['data']

        # Analyze trends
        hot_movers = []
        sustained_trends = []

        # Track each player/market/bookmaker combo
        player_trends = {}

        for game_id, game_data in current.items():
            for market, props in game_data.get('props', {}).items():
                for prop in props:
                    player_name = prop['player_name']
                    bookmaker = prop['bookmaker']
                    key = f"{player_name}_{market}_{bookmaker}"

                    # Only track DraftKings (user's book)
                    if bookmaker != 'draftkings':
                        continue

                    movement = prop.get('movement', {})
                    current_line = prop.get('line', 0)
                    over_odds = prop.get('over_odds', -110)
                    under_odds = prop.get('under_odds', -110)

                    # Week-over-week: Hot movers (2+ point moves)
                    if movement.get('is_hot_mover'):
                        opening_line = movement.get('opening_line', current_line)
                        line_change = current_line - opening_line
                        pct_change = (line_change / opening_line * 100) if opening_line != 0 else 0
                        days_tracked = movement.get('days_tracked', 0)

                        hot_movers.append({
                            'player': player_name,
                            'market': market,
                            'bookmaker': bookmaker,

                            # Line data
                            'opening_line': opening_line,
                            'current_line': current_line,
                            'line_movement': line_change,
                            'line_movement_pct': round(pct_change, 1),

                            # Juice/vig
                            'over_odds': over_odds,
                            'under_odds': under_odds,

                            # Direction
                            'direction': movement['movement_direction'],

                            # Timestamps
                            'opening_timestamp': movement.get('opening_timestamp'),
                            'current_timestamp': prop.get('timestamp'),
                            'days_tracked': days_tracked,

                            # Badge for frontend
                            'badge': f"{line_change:+.1f} pts ({days_tracked} days)" if days_tracked > 0 else f"{line_change:+.1f} pts",
                            'badge_pct': f"{pct_change:+.0f}% ({days_tracked} days)" if days_tracked > 0 else f"{pct_change:+.0f}%",

                            # Visual indicators
                            'icon': '⬆️' if line_change > 0 else '⬇️',
                            'color': 'green' if line_change > 0 else 'red',
                            'strength': '🔥' if abs(line_change) >= 5.0 else '⚡' if abs(line_change) >= 3.0 else '📊'
                        })

                    # 3-week sustained trends: Check for consistent direction
                    if len(snapshots) >= 3:
                        trend_history = self._analyze_sustained_trend(
                            player_name,
                            market,
                            bookmaker,
                            snapshots
                        )

                        if trend_history and trend_history['is_sustained']:
                            sustained_trends.append({
                                'player': player_name,
                                'market': market,
                                'bookmaker': bookmaker,

                                # Trend data
                                'week_1_line': trend_history['week_1_line'],
                                'week_2_line': trend_history.get('week_2_line'),
                                'week_3_line': trend_history.get('week_3_line'),
                                'current_line': current_line,

                                # Total movement
                                'total_movement': trend_history['total_movement'],
                                'total_movement_pct': trend_history['total_movement_pct'],

                                # Direction consistency
                                'direction': trend_history['direction'],
                                'consistency': trend_history['consistency'],  # e.g., "3/3 weeks up"

                                # Juice/vig
                                'over_odds': over_odds,
                                'under_odds': under_odds,

                                # Badge
                                'badge': f"{trend_history['total_movement']:+.1f} pts (3 weeks)",
                                'badge_pct': f"{trend_history['total_movement_pct']:+.0f}% (3 weeks)",

                                # Visual
                                'icon': '⬆️' if trend_history['direction'] == 'up' else '⬇️',
                                'color': 'green' if trend_history['direction'] == 'up' else 'red',
                                'strength': '🔥🔥' if abs(trend_history['total_movement']) >= 8.0 else '🔥'
                            })

        return {
            'hot_movers': sorted(hot_movers, key=lambda x: abs(x['line_movement']), reverse=True),
            'sustained_trends': sorted(sustained_trends, key=lambda x: abs(x['total_movement']), reverse=True),
            'snapshots_analyzed': len(snapshots),
            'current_timestamp': current.get(list(current.keys())[0], {}).get('snapshot_timestamp') if current else None
        }

    def _analyze_sustained_trend(
        self,
        player_name: str,
        market: str,
        bookmaker: str,
        snapshots: List[Dict],
        min_weeks: int = 3
    ) -> Optional[Dict]:
        """Analyze if a prop has sustained trend over multiple weeks.

        Args:
            player_name: Player name
            market: Market type
            bookmaker: Bookmaker
            snapshots: List of snapshot dicts (newest first)
            min_weeks: Minimum weeks for sustained trend (default 3)

        Returns:
            Dict with trend analysis or None if no sustained trend
        """
        if len(snapshots) < min_weeks:
            return None

        # Extract lines from each snapshot (up to 3 weeks)
        lines = []
        for i, snapshot in enumerate(snapshots[:min_weeks]):
            snapshot_data = snapshot['data']

            # Find this player's line in this snapshot
            for game_id, game_data in snapshot_data.items():
                props = game_data.get('props', {}).get(market, [])
                for prop in props:
                    if (prop.get('player_name') == player_name and
                        prop.get('bookmaker') == bookmaker):
                        lines.append({
                            'week': min_weeks - i,  # Week 1 (oldest), 2, 3 (newest)
                            'line': prop.get('line', 0),
                            'timestamp': prop.get('timestamp')
                        })
                        break

        if len(lines) < min_weeks:
            return None

        # Sort by week
        lines = sorted(lines, key=lambda x: x['week'])

        # Check for consistent direction
        movements = []
        for i in range(len(lines) - 1):
            movement = lines[i + 1]['line'] - lines[i]['line']
            movements.append(movement)

        # Sustained if all movements in same direction (all positive or all negative)
        if not movements:
            return None

        all_up = all(m > 0 for m in movements)
        all_down = all(m < 0 for m in movements)
        is_sustained = all_up or all_down

        if not is_sustained:
            return None

        # Calculate total movement
        total_movement = lines[-1]['line'] - lines[0]['line']
        total_movement_pct = (total_movement / lines[0]['line'] * 100) if lines[0]['line'] != 0 else 0

        # Direction and consistency
        direction = 'up' if all_up else 'down'
        consistency = f"{len(movements)}/{len(movements)} weeks {direction}"

        return {
            'is_sustained': True,
            'week_1_line': lines[0]['line'],
            'week_2_line': lines[1]['line'] if len(lines) > 1 else None,
            'week_3_line': lines[2]['line'] if len(lines) > 2 else None,
            'total_movement': round(total_movement, 1),
            'total_movement_pct': round(total_movement_pct, 1),
            'direction': direction,
            'consistency': consistency
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
