#!/usr/bin/env python3
"""
Demo: Full Game Analysis

Analyzes a game and provides comprehensive prop recommendations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("NFL PROPS - FULL GAME ANALYSIS")
print("=" * 80)
print()

try:
    from backend.database.session import get_db
    from backend.database.models import Game, Player
    from datetime import datetime, timedelta

    print("✅ Database connection established")

    # Find a recent game
    with get_db() as session:
        game = session.query(Game).filter(
            Game.season == 2024
        ).order_by(Game.week.desc()).first()

        if not game:
            print("❌ No games found in database")
            print("\nTo analyze games, you need:")
            print("1. Game data loaded in database")
            print("2. Player features generated")
            print("3. Models trained")
            sys.exit(1)

        print(f"\n📊 Analyzing Game:")
        print(f"   {game.away_team} @ {game.home_team}")
        print(f"   Week {game.week}, {game.season}")
        print(f"   Game ID: {game.game_id}")
        print()

    # Try to generate recommendations
    try:
        from backend.recommendations import RecommendationScorer

        print("🔮 Generating Recommendations...")
        scorer = RecommendationScorer(min_confidence=0.6)

        recs = scorer.recommend_props(
            game_id=game.game_id,
            limit=10,
            include_reasoning=True
        )

        if not recs:
            print("⚠️  No recommendations generated (may need trained models)")
            print("\nTo get recommendations:")
            print("1. Ensure PlayerGameFeatures exist for this game")
            print("2. Train XGBoost models: python scripts/train_models.py")
            print("3. Run calibration: python scripts/calibrate.py")
        else:
            print(f"✅ Generated {len(recs)} recommendations\n")
            print("=" * 80)
            print("TOP PROP RECOMMENDATIONS")
            print("=" * 80)

            for i, rec in enumerate(recs[:5], 1):
                print(f"\n#{i}. {rec.player_name} ({rec.position}) - {rec.market}")
                print(f"   Team: {rec.team}")
                print(f"   Line: {rec.line:.1f}")
                print(f"   Probability: {rec.calibrated_prob*100:.1f}%")
                print(f"   Overall Score: {rec.overall_score:.3f}")
                print(f"   Strength: {rec.recommendation_strength.value.upper()}")
                print(f"   Confidence: {rec.confidence*100:.1f}%")

                # Signal breakdown
                print(f"\n   Signal Breakdown:")
                print(f"      Base Model:  {rec.base_signal:.3f}")
                print(f"      Matchup:     {rec.matchup_signal:.3f}")
                print(f"      Trend:       {rec.trend_signal:.3f}")
                print(f"      News:        {rec.news_signal:.3f}")
                print(f"      Roster:      {rec.roster_signal:.3f}")

                if rec.reasoning:
                    print(f"\n   Reasoning:")
                    for reason in rec.reasoning:
                        print(f"      • {reason}")

                if rec.flags:
                    print(f"\n   ⚠️  Flags: {', '.join(rec.flags)}")

                print("-" * 80)

        # Try parlays
        print("\n" + "=" * 80)
        print("PARLAY RECOMMENDATIONS")
        print("=" * 80)

        parlays = scorer.recommend_parlays(
            game_id=game.game_id,
            parlay_size=3,
            limit=3
        )

        if parlays:
            for i, parlay in enumerate(parlays, 1):
                print(f"\n#{i}. {len(parlay.props)}-Leg Parlay")
                print(f"   Overall Score: {parlay.overall_score:.3f}")
                print(f"   Confidence: {parlay.confidence*100:.1f}%")
                print(f"\n   Props:")
                for prop in parlay.props:
                    print(f"      • {prop.player_name} {prop.market} {prop.line}")
                print(f"\n   Probability:")
                print(f"      Raw (independent): {parlay.raw_probability*100:.1f}%")
                print(f"      Adjusted (correlated): {parlay.adjusted_probability*100:.1f}%")
                print(f"      Correlation Impact: {parlay.correlation_impact}")

                if parlay.reasoning:
                    print(f"\n   Reasoning:")
                    for reason in parlay.reasoning:
                        print(f"      • {reason}")
                print("-" * 80)
        else:
            print("\n⚠️  No parlays generated")

    except Exception as e:
        print(f"❌ Recommendation generation failed: {e}")
        print("\nThis likely means:")
        print("1. Models not trained yet")
        print("2. Missing player features")
        print("3. Database schema issues")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"❌ Error: {e}")
    print("\nMake sure:")
    print("1. PostgreSQL is running")
    print("2. Database is configured in backend/config/.env")
    print("3. Dependencies are installed: pip install -r requirements.txt")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)
