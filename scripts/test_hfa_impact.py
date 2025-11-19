#!/usr/bin/env python3
"""Quick script to test HFA impact on prop projections.

Usage:
    python scripts/test_hfa_impact.py
"""

from backend.features.hfa_impact_analysis import hfa_impact_analyzer

# Test case: Patrick Mahomes passing yards
# Base projection: 285 yards (neutral site)
# Game: KC @ BUF (Week 10, 2025)

print("=" * 60)
print("HFA IMPACT TEST: Patrick Mahomes Passing Yards")
print("=" * 60)

base_projection = 285.0
game_id = "2025_10_KC_BUF"

# Test 1: Mahomes at home (KC hosting BUF)
home_game_id = "2025_10_BUF_KC"  # BUF @ KC (Mahomes at home)
result_home = hfa_impact_analyzer.apply_hfa_to_projection(
    base_projection=base_projection,
    position='QB',
    prop_type='passing_yards',
    game_id=home_game_id,
    team='KC',
    is_home_team=True
)

print(f"\n🏠 MAHOMES AT HOME (KC hosting):")
print(f"   Base Projection: {result_home['base_projection']} yards")
print(f"   HFA Adjustment:  +{result_home['hfa_adjustment']} yards")
print(f"   Final Projection: {result_home['adjusted_projection']} yards")
print(f"   Adjustment %:     +{result_home['adjustment_pct']}%")
print(f"\n   Factors:")
for key, val in result_home['factors'].items():
    print(f"   - {key}: {val}")

# Test 2: Mahomes on the road (KC @ BUF)
result_away = hfa_impact_analyzer.apply_hfa_to_projection(
    base_projection=base_projection,
    position='QB',
    prop_type='passing_yards',
    game_id=game_id,
    team='KC',
    is_home_team=False
)

print(f"\n✈️  MAHOMES ON ROAD (KC @ BUF):")
print(f"   Base Projection: {result_away['base_projection']} yards")
print(f"   HFA Adjustment:  {result_away['hfa_adjustment']} yards")
print(f"   Final Projection: {result_away['adjusted_projection']} yards")
print(f"   Adjustment %:     {result_away['adjustment_pct']}%")
print(f"\n   Factors:")
for key, val in result_away['factors'].items():
    print(f"   - {key}: {val}")

# Test 3: Compare home vs away
comparison = hfa_impact_analyzer.compare_home_away_props(
    position='QB',
    prop_type='passing_yards',
    base_projection=base_projection,
    game_id=game_id,
    home_team='BUF',
    away_team='KC'
)

print(f"\n📊 HOME VS AWAY COMPARISON:")
print(f"   Base Projection: {comparison['base_projection']} yards")
print(f"   As Home Team:    {comparison['as_home_team']['adjusted_projection']} yards")
print(f"   As Away Team:    {comparison['as_away_team']['adjusted_projection']} yards")
print(f"   Total HFA Swing: {comparison['total_hfa_swing']} yards ({comparison['swing_pct']}%)")

print("\n" + "=" * 60)
print("TAKEAWAY:")
print("=" * 60)
swing = comparison['total_hfa_swing']
print(f"Playing at home vs away creates a {abs(swing):.1f} yard difference")
print(f"for QB passing yards. This is substantial for prop betting!")
print(f"\nArrowhead Stadium (KC) HFA multiplier: 1.20 (top 3 in NFL)")
print(f"Highmark Stadium (BUF) in outdoor weather also impacts passing.")
print("\n" + "=" * 60)

# Test 4: Different positions
print("\n📈 HFA IMPACT BY POSITION:")
print("=" * 60)

positions_tests = [
    ('QB', 'passing_yards', 285),
    ('RB', 'rushing_yards', 75),
    ('WR', 'receiving_yards', 68),
    ('TE', 'receiving_yards', 52)
]

for pos, prop, base in positions_tests:
    result = hfa_impact_analyzer.apply_hfa_to_projection(
        base_projection=base,
        position=pos,
        prop_type=prop,
        game_id=home_game_id,  # At home
        team='KC',
        is_home_team=True
    )

    print(f"\n{pos} {prop}:")
    print(f"  Base: {base} → Adjusted: {result['adjusted_projection']}")
    print(f"  HFA Impact: +{result['hfa_adjustment']} ({result['adjustment_pct']:+.1f}%)")

print("\n" + "=" * 60)
