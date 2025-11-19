"""Extract market context features: spread, total, implied points.

Market context is EXTREMELY predictive for prop betting because:
1. Spread indicates expected game script
   - Favorites run more late (RB props UP)
   - Underdogs pass more (QB/WR props UP)
2. Total indicates expected pace and scoring
   - High total = more plays = more volume
3. Implied team totals predict offensive opportunity

These features capture information embedded in betting markets that often
reflects inside information (injuries, weather, coaching) before it's public.
"""

from pathlib import Path
import json
import requests
from typing import Dict, Optional
from datetime import datetime


def fetch_spreads_and_totals(
    season: int,
    week: int,
    odds_api_key: Optional[str] = None,
    output_file: Optional[Path] = None
) -> Dict:
    """Fetch spreads and totals from The Odds API.

    Args:
        season: Season year
        week: Week number
        odds_api_key: The Odds API key (optional, can use cached data)
        output_file: Optional path to save market data

    Returns:
        Dict mapping game_id -> market context
    """
    print(f"\n{'='*80}")
    print(f"FETCHING MARKET CONTEXT - {season} Week {week}")
    print(f"{'='*80}\n")

    market_context = {}

    if odds_api_key:
        # Fetch from The Odds API
        print("📊 Fetching from The Odds API...")

        url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"
        params = {
            'apiKey': odds_api_key,
            'regions': 'us',
            'markets': 'spreads,totals',
            'oddsFormat': 'american'
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            games = response.json()

            for game in games:
                home_team = game.get('home_team', '')
                away_team = game.get('away_team', '')

                # Extract spread and total from bookmakers
                spread = None
                total = None

                for bookmaker in game.get('bookmakers', []):
                    for market in bookmaker.get('markets', []):
                        if market['key'] == 'spreads':
                            for outcome in market['outcomes']:
                                if outcome['name'] == home_team:
                                    spread = outcome['point']
                        elif market['key'] == 'totals':
                            for outcome in market['outcomes']:
                                if outcome['name'] == 'Over':
                                    total = outcome['point']

                    if spread is not None and total is not None:
                        break

                # Calculate implied team totals
                if spread is not None and total is not None:
                    home_implied = (total - spread) / 2
                    away_implied = (total + spread) / 2

                    game_key = f"{season}_{week}_{home_team}_{away_team}"

                    market_context[game_key] = {
                        'home_team': home_team,
                        'away_team': away_team,
                        'spread': spread,  # Positive = home underdog
                        'total': total,
                        'home_implied_total': round(home_implied, 1),
                        'away_implied_total': round(away_implied, 1),
                        'timestamp': datetime.now().isoformat()
                    }

            print(f"✓ Fetched market context for {len(market_context)} games")

        except Exception as e:
            print(f"⚠️  Failed to fetch from Odds API: {e}")
            print("⚠️  Will use fallback/manual data")

    else:
        print("⚠️  No Odds API key provided, using fallback data")

    # Save if output file specified
    if output_file and market_context:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(market_context, f, indent=2)
        print(f"✓ Saved market context to: {output_file}")

    return market_context


def add_market_features_to_player_games(
    player_features: Dict,
    market_context: Dict
) -> Dict:
    """Add market context features to player game records.

    Args:
        player_features: Player features dict
        market_context: Market context dict

    Returns:
        Enhanced player features
    """
    print(f"\n{'='*80}")
    print("ADDING MARKET FEATURES TO PLAYER GAMES")
    print(f"{'='*80}\n")

    enhanced_count = 0

    for player_id, games in player_features.items():
        for game in games:
            season = game.get('season', '')
            week = game.get('week', '')
            team = game.get('team', '')
            opponent = game.get('opponent', '')
            is_home = game.get('is_home', None)

            # Try to find matching market data
            # Try both home/away combinations
            game_key_1 = f"{season}_{week}_{team}_{opponent}"
            game_key_2 = f"{season}_{week}_{opponent}_{team}"

            market_data = market_context.get(game_key_1) or market_context.get(game_key_2)

            if market_data:
                if is_home is not None:
                    # Determine team's perspective
                    if is_home is True:
                        game['spread'] = market_data['spread']
                        game['implied_team_total'] = market_data['home_implied_total']
                        game['opponent_implied_total'] = market_data['away_implied_total']
                    else:
                        game['spread'] = -market_data['spread']  # Flip spread for away team
                        game['implied_team_total'] = market_data['away_implied_total']
                        game['opponent_implied_total'] = market_data['home_implied_total']

                    game['total'] = market_data['total']

                    # Derived features
                    game['is_favorite'] = 1 if game['spread'] < 0 else 0
                    game['is_underdog'] = 1 if game['spread'] > 0 else 0
                    game['expected_to_trail'] = 1 if game['spread'] > 3 else 0  # 3+ point dog
                    game['expected_to_lead'] = 1 if game['spread'] < -3 else 0  # 3+ point favorite

                    # Game script indicator (positive = likely high-scoring shootout)
                    game['expected_script'] = game['spread'] * game['total'] / 100

                    enhanced_count += 1
                # else: Skip games where is_home is missing (can't determine perspective)

    print(f"✓ Added market features to {enhanced_count} player-games")

    return player_features


def load_manual_market_data(
    season: int,
    week: int,
    manual_file: Path
) -> Dict:
    """Load manually entered market data.

    For when Odds API is unavailable, load from a JSON file with format:
    {
      "2024_12_KC_BUF": {
        "spread": -3.0,
        "total": 49.5
      }
    }

    Args:
        season: Season year
        week: Week number
        manual_file: Path to manual market data JSON

    Returns:
        Market context dict
    """
    if not manual_file.exists():
        print(f"⚠️  Manual market file not found: {manual_file}")
        return {}

    with open(manual_file, 'r') as f:
        manual_data = json.load(f)

    # Convert to full format
    market_context = {}

    for game_key, data in manual_data.items():
        parts = game_key.split('_')
        if len(parts) == 4:
            season, week, home_team, away_team = parts

            spread = data.get('spread', 0)
            total = data.get('total', 45)

            home_implied = (total - spread) / 2
            away_implied = (total + spread) / 2

            market_context[game_key] = {
                'home_team': home_team,
                'away_team': away_team,
                'spread': spread,
                'total': total,
                'home_implied_total': round(home_implied, 1),
                'away_implied_total': round(away_implied, 1),
                'timestamp': datetime.now().isoformat()
            }

    print(f"✓ Loaded {len(market_context)} games from manual data")

    return market_context


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract market context (spreads, totals)'
    )
    parser.add_argument('--season', type=int, required=True)
    parser.add_argument('--week', type=int, required=True)
    parser.add_argument('--odds-api-key', type=str,
                       help='The Odds API key')
    parser.add_argument('--manual-file', type=Path,
                       help='Manual market data JSON file')
    parser.add_argument('--player-features', type=Path, required=True,
                       help='Player features JSON')
    parser.add_argument('--output-file', type=Path,
                       default=Path('outputs/features/market_context.json'),
                       help='Output file for market data')

    args = parser.parse_args()

    # Fetch or load market data
    if args.odds_api_key:
        market_context = fetch_spreads_and_totals(
            season=args.season,
            week=args.week,
            odds_api_key=args.odds_api_key,
            output_file=args.output_file
        )
    elif args.manual_file:
        market_context = load_manual_market_data(
            season=args.season,
            week=args.week,
            manual_file=args.manual_file
        )
    else:
        print("⚠️  Must provide either --odds-api-key or --manual-file")
        exit(1)

    # Load player features
    with open(args.player_features, 'r') as f:
        player_features = json.load(f)

    # Add market features
    enhanced_features = add_market_features_to_player_games(
        player_features=player_features,
        market_context=market_context
    )

    # Save enhanced features
    output_path = args.player_features.parent / f"{args.season}_player_features_with_market.json"
    with open(output_path, 'w') as f:
        json.dump(enhanced_features, f, indent=2)

    print(f"\n✓ Saved enhanced features to: {output_path}")
