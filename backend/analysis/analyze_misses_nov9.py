"""Analysis of betting misses from Nov 9, 2025.

Identifies patterns and red flags we could have caught.
"""

import pandas as pd
from pathlib import Path


def analyze_misses():
    """Deep dive into our misses to find learnings."""

    print(f"\n{'='*80}")
    print(f"POST-MORTEM: What Could We Have Done Better?")
    print(f"{'='*80}\n")

    # Load games
    games_file = Path('inputs/games_2025.csv')
    all_games = pd.read_csv(games_file)

    week10 = all_games[
        (all_games['season'] == 2025) &
        (all_games['week'] == 10) &
        (all_games['weekday'] == 'Sunday') &
        (all_games['game_type'] == 'REG')
    ].copy()

    # Analyze each miss
    print("🔍 ANALYZING PROP MISSES:\n")

    # Miss 1: Josh Allen
    print("❌ MISS #1: Josh Allen OVER 265.5 pass yards (actual: 189)")
    print("   Game: BUF @ MIA (final: 13-30)")
    print()

    buf_mia = week10[week10['game_id'] == '2025_10_BUF_MIA'].iloc[0]

    print(f"   🚩 RED FLAGS WE MISSED:")
    print(f"   - BUF was -8.5 favorite but had SHORT week? NO - both 7+ days rest")
    print(f"   - BUF moneyline: {buf_mia['away_moneyline']} (HUGE favorite)")
    print(f"   - MIA moneyline: {buf_mia['home_moneyline']} (massive underdog)")
    print(f"   - MIA had MORE rest: {buf_mia['home_rest']} days vs BUF {buf_mia['away_rest']} days")
    print()

    print(f"   💡 LEARNINGS:")
    print(f"   - When a team gets blown out (down 17), QB stops throwing")
    print(f"   - BUF scored only 13 points (way under their average)")
    print(f"   - Division game (AFC East) = unpredictable")
    print(f"   - MIA had extra rest (10 days vs 7)")
    print(f"   - Should have checked: Is MIA particularly good at home?")
    print(f"   - Should have checked: BUF's recent road performance")
    print()

    # Miss 2: Bijan Robinson
    print("❌ MISS #2: Bijan Robinson OVER 75.5 rush yards (actual: 72)")
    print("   Game: ATL @ IND (final: 25-31)")
    print()

    atl_ind = week10[week10['game_id'] == '2025_10_ATL_IND'].iloc[0]

    print(f"   🚩 RED FLAGS WE MISSED:")
    print(f"   - ATL was +6.5 underdog")
    print(f"   - Total was 48.5 (high-scoring game expected)")
    print(f"   - Game went to OT: {atl_ind.get('overtime', 0)}")
    print()

    print(f"   💡 LEARNINGS:")
    print(f"   - VERY CLOSE: 72 vs 75.5 (only 3.5 yards off)")
    print(f"   - High-scoring game (56 total) = more passing, less rushing")
    print(f"   - ATL playing from behind = less run-heavy script")
    print(f"   - Should have adjusted: High total games favor passing props")
    print()

    print(f"\n{'='*80}")
    print(f"🔍 ANALYZING GAME PICK MISSES:\n")
    print(f"{'='*80}\n")

    # Game miss 1: NO @ CAR
    print("❌ GAME MISS #1: Picked CAR, but NO won 17-7")
    print()

    no_car = week10[week10['game_id'] == '2025_10_NO_CAR'].iloc[0]

    print(f"   Game details:")
    print(f"   - Spread: CAR {no_car['spread_line']:+.1f} (CAR favored by 5.5)")
    print(f"   - Total: {no_car['total_line']} (very low - defensive game)")
    print(f"   - Division game: {no_car['div_game']} (NFC South)")
    print()

    print(f"   💡 LEARNINGS:")
    print(f"   - Low total (38.5) games are UNPREDICTABLE")
    print(f"   - Division games are coin flips")
    print(f"   - NO covered as underdog (+5.5)")
    print()

    # Game miss 2: BUF @ MIA
    print("❌ GAME MISS #2: Picked BUF (-8.5), but MIA won 30-13")
    print()
    print(f"   💡 LEARNINGS:")
    print(f"   - MASSIVE upset - BUF was -455 moneyline favorite!")
    print(f"   - This is why we don't bet the house on favorites")
    print(f"   - MIA at home with extra rest = dangerous")
    print(f"   - Division rivalry game (AFC East)")
    print()

    # Game miss 3: JAX @ HOU
    print("❌ GAME MISS #3: Picked JAX (-1.5), but HOU won 36-29")
    print()

    jax_hou = week10[week10['game_id'] == '2025_10_JAX_HOU'].iloc[0]

    print(f"   Game details:")
    print(f"   - Spread: {jax_hou['spread_line']} (basically a pick'em)")
    print(f"   - Total: {jax_hou['total_line']}")
    print(f"   - Actual total: {jax_hou['total']} (MASSIVE over!)")
    print(f"   - Division game: {jax_hou['div_game']} (AFC South)")
    print()

    print(f"   💡 LEARNINGS:")
    print(f"   - Pick'em games (< 2 point spread) are 50/50")
    print(f"   - Should have avoided betting on this")
    print(f"   - 65 total points! Way over the 37.5 line")
    print()

    print(f"\n{'='*80}")
    print(f"📊 PATTERN ANALYSIS")
    print(f"{'='*80}\n")

    # Analyze all division games
    div_games = week10[week10['div_game'] == 1]

    print(f"Division Games This Week: {len(div_games)}")
    print()

    for idx, game in div_games.iterrows():
        away = game['away_team']
        home = game['home_team']
        spread = game['spread_line']
        away_score = game['away_score']
        home_score = game['home_score']

        favorite = away if spread > 0 else home
        favorite_score = away_score if spread > 0 else home_score
        underdog_score = home_score if spread > 0 else away_score

        covered = (favorite_score - underdog_score) > abs(spread)

        print(f"  {away} @ {home} (spread: {spread:+.1f})")
        print(f"    Final: {away} {away_score}, {home} {home_score}")
        print(f"    Favorite covered: {'❌ NO' if not covered else '✅ YES'}")
        print()

    # Count upsets
    upsets = 0
    for idx, game in week10.iterrows():
        spread = game['spread_line']
        actual_margin = game['home_score'] - game['away_score']

        # If spread < 0, home favored. If actual_margin < spread, upset
        # If spread > 0, away favored. If actual_margin > spread, upset
        if (spread < 0 and actual_margin < spread) or (spread > 0 and actual_margin > spread):
            upsets += 1

    print(f"\n📈 WEEK 10 STATISTICS:")
    print(f"   Total games: {len(week10)}")
    print(f"   Division games: {len(div_games)}")
    print(f"   Upsets (underdog won or favorite didn't cover): {upsets}")
    print(f"   Upset rate: {upsets / len(week10):.1%}")
    print()

    print(f"\n{'='*80}")
    print(f"✅ ACTIONABLE IMPROVEMENTS")
    print(f"{'='*80}\n")

    print("1️⃣  AVOID DIVISION GAMES")
    print("   - They're unpredictable (NO beat CAR, MIA beat BUF)")
    print("   - Or bet smaller on division matchups")
    print()

    print("2️⃣  CHECK REST DAYS")
    print("   - MIA had 10 days rest vs BUF's 7")
    print("   - Should be a feature in our model")
    print()

    print("3️⃣  GAME SCRIPT MATTERS")
    print("   - High totals (>48) favor passing props")
    print("   - Low totals (<40) are unpredictable")
    print("   - Blowouts kill QB props for losing team")
    print()

    print("4️⃣  AVOID PICK'EM GAMES (<2 pt spread)")
    print("   - JAX/HOU was -1.5 (coin flip)")
    print("   - Better opportunities elsewhere")
    print()

    print("5️⃣  HOME UNDERDOGS ARE DANGEROUS")
    print("   - MIA +350 at home beat BUF")
    print("   - Home field + extra rest = upset potential")
    print()

    print("6️⃣  DON'T CHASE BIG FAVORITES")
    print("   - BUF was -455 and lost")
    print("   - Value is usually on underdogs")
    print()

    print("7️⃣  BIJAN WAS CLOSE (72 vs 75.5)")
    print("   - Sometimes you just get unlucky")
    print("   - 3.5 yards is variance, not a mistake")
    print()


if __name__ == '__main__':
    analyze_misses()
