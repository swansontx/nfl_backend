"""
Unified Picks Pipeline: The main "make picks" entry point.

This module ties together:
1. Model projections from trained models
2. Current odds from The Odds API
3. Prop analysis for edge detection
4. Portfolio optimization for parlay construction
5. CLV tracking for feedback loop

Usage:
    from backend.orchestration.picks_pipeline import PicksPipeline

    pipeline = PicksPipeline()
    picks = pipeline.generate_picks(week=11)
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import pandas as pd
import joblib


@dataclass
class PickRecommendation:
    """A single pick recommendation with full context."""
    # Identity
    player_id: str
    player_name: str
    game_id: str
    team: str
    opponent: str

    # Prop details
    prop_type: str
    line: float
    side: str  # "OVER" or "UNDER"
    odds: int

    # Model outputs
    projection: float
    hit_probability: float
    edge: float  # percentage points
    ev: float    # expected value as decimal

    # Quality metrics
    grade: str   # A+, A, B+, B, C, F
    trust_score: float
    confidence: str  # HIGH, MEDIUM, LOW

    # Sizing
    recommended_stake_pct: float
    recommended_stake_dollars: float

    # Context
    notes: List[str]


@dataclass
class ParlayRecommendation:
    """A parlay recommendation with correlation awareness."""
    legs: List[PickRecommendation]
    combined_odds: int
    combined_probability: float
    expected_value: float
    recommended_stake_pct: float
    recommended_stake_dollars: float
    correlation_adjustment: float
    confidence: str
    notes: List[str]


@dataclass
class PicksReport:
    """Complete picks report for a slate."""
    generated_at: str
    week: int
    games_analyzed: int
    props_screened: int

    # Recommendations
    single_picks: List[PickRecommendation]
    parlays: List[ParlayRecommendation]

    # Summary
    total_edge_dollars: float
    total_stake_dollars: float
    best_single: Optional[PickRecommendation]
    best_parlay: Optional[ParlayRecommendation]

    # Risk metrics
    by_game_exposure: Dict[str, float]
    by_prop_type: Dict[str, int]


class PicksPipeline:
    """Main pipeline for generating betting picks."""

    def __init__(
        self,
        models_dir: str = "outputs/models",
        inputs_dir: str = "inputs",
        bankroll: float = 1000.0,
        min_edge: float = 3.0,
        min_trust_score: float = 0.4,
        max_risk_per_game: float = 0.05,
        max_total_risk: float = 0.20,
        odds_api_key: Optional[str] = None,
    ):
        self.models_dir = Path(models_dir)
        self.inputs_dir = Path(inputs_dir)
        self.bankroll = bankroll
        self.min_edge = min_edge
        self.min_trust_score = min_trust_score
        self.max_risk_per_game = max_risk_per_game
        self.max_total_risk = max_total_risk
        self.odds_api_key = odds_api_key or os.getenv("ODDS_API_KEY")

        # Load models
        self.models = self._load_models()

    def _load_models(self) -> Dict:
        """Load all trained models."""
        models = {}

        for model_file in self.models_dir.rglob("*.pkl"):
            try:
                model_name = model_file.stem
                models[model_name] = joblib.load(model_file)
            except Exception as e:
                print(f"Warning: Could not load {model_file}: {e}")

        print(f"Loaded {len(models)} models")
        return models

    def _load_player_features(self, week: int) -> pd.DataFrame:
        """Load player features for predictions."""
        # Try enhanced stats first
        enhanced_file = self.inputs_dir / "player_stats_enhanced_2025.csv"
        if enhanced_file.exists():
            df = pd.read_csv(enhanced_file)
            # Filter to weeks before current for features
            df = df[df['week'] < week].copy()
            return df

        # Fallback to basic stats
        basic_file = self.inputs_dir / "player_stats_2025.csv"
        if basic_file.exists():
            df = pd.read_csv(basic_file)
            df = df[df['week'] < week].copy()
            return df

        return pd.DataFrame()

    def _get_latest_player_features(self, player_id: str, df: pd.DataFrame) -> Dict:
        """Get most recent features for a player."""
        player_df = df[df['player_id'] == player_id].sort_values('week', ascending=False)

        if len(player_df) == 0:
            return {}

        row = player_df.iloc[0]
        return row.to_dict()

    def _predict_prop(
        self,
        player_id: str,
        prop_type: str,
        features: Dict
    ) -> Tuple[float, float]:
        """
        Generate prediction for a prop.

        Returns:
            (projection, std_dev)
        """
        # Map prop types to model names
        model_map = {
            "pass_yds": "pass_yards_models",
            "pass_tds": "pass_tds_models",
            "rush_yds": "rush_yards_models",
            "rush_tds": "rush_tds_models",
            "rec_yds": "rec_yards_models",
            "receptions": "receptions_models",
            "rec_tds": "rec_tds_models",
            "completions": "completions_models",
            "attempts": "attempts_models",
            "interceptions": "interceptions_models",
            "carries": "carries_models",
            "targets": "targets_models",
        }

        model_name = model_map.get(prop_type)
        if not model_name or model_name not in self.models:
            # Fallback: use rolling average
            avg_col = f"{prop_type.replace('_', '')}_season_avg"
            if avg_col in features:
                return features[avg_col], features[avg_col] * 0.25
            return 0.0, 10.0

        # Get model
        model = self.models[model_name]

        # Prepare features (simplified - in production use proper feature engineering)
        feature_cols = [
            'games_played', 'is_home', 'spread_line', 'total_line',
            f"{prop_type.replace('_', '')}_season_avg",
            f"{prop_type.replace('_', '')}_l3_avg",
        ]

        X = []
        for col in feature_cols:
            val = features.get(col, 0)
            X.append(val if pd.notna(val) else 0)

        try:
            projection = model.predict([X])[0]
            # Estimate std_dev as fraction of projection
            std_dev = max(projection * 0.20, 5.0)
            return projection, std_dev
        except Exception as e:
            print(f"Prediction error for {player_id} {prop_type}: {e}")
            return 0.0, 10.0

    def _calculate_hit_probability(
        self,
        projection: float,
        std_dev: float,
        line: float,
        side: str
    ) -> float:
        """Calculate probability of hitting the prop."""
        from math import erf, sqrt

        if std_dev <= 0:
            return 0.5

        # Z-score
        z = (line - projection) / std_dev

        # CDF using error function
        prob_under = 0.5 * (1 + erf(z / sqrt(2)))

        if side == "OVER":
            return 1 - prob_under
        else:
            return prob_under

    def _odds_to_implied_prob(self, odds: int) -> float:
        """Convert American odds to implied probability."""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)

    def _calculate_edge(
        self,
        model_prob: float,
        odds: int
    ) -> float:
        """Calculate edge in percentage points."""
        implied_prob = self._odds_to_implied_prob(odds)
        return (model_prob - implied_prob) * 100

    def _calculate_ev(
        self,
        model_prob: float,
        odds: int
    ) -> float:
        """Calculate expected value as decimal."""
        if odds > 0:
            payout = odds / 100
        else:
            payout = 100 / abs(odds)

        return (model_prob * payout) - ((1 - model_prob) * 1)

    def _grade_pick(self, edge: float) -> str:
        """Assign grade based on edge."""
        if edge >= 15:
            return "A+"
        elif edge >= 10:
            return "A"
        elif edge >= 7:
            return "B+"
        elif edge >= 5:
            return "B"
        elif edge >= 3:
            return "C"
        else:
            return "F"

    def _calculate_kelly_stake(
        self,
        model_prob: float,
        odds: int,
        kelly_fraction: float = 0.25
    ) -> float:
        """Calculate Kelly-optimal stake."""
        if odds > 0:
            decimal_odds = 1 + (odds / 100)
        else:
            decimal_odds = 1 + (100 / abs(odds))

        b = decimal_odds - 1
        p = model_prob
        q = 1 - model_prob

        if b <= 0:
            return 0.0

        kelly = (b * p - q) / b
        stake = kelly * kelly_fraction

        return max(0, min(stake, 0.05))

    def generate_picks(
        self,
        week: int,
        odds_data: Optional[List[Dict]] = None
    ) -> PicksReport:
        """
        Generate picks for a given week.

        Args:
            week: NFL week number
            odds_data: Optional pre-loaded odds data (for testing without API)

        Returns:
            PicksReport with all recommendations
        """
        print(f"\n{'='*60}")
        print(f"GENERATING PICKS - WEEK {week}")
        print(f"{'='*60}\n")

        # Load player features
        player_features = self._load_player_features(week)
        print(f"Loaded features for {player_features['player_id'].nunique()} players")

        # Load or fetch odds
        if odds_data is None:
            odds_data = self._fetch_odds()

        if not odds_data:
            print("ERROR: No odds data available.")
            print("Please either:")
            print("  1. Set ODDS_API_KEY environment variable and ensure network access")
            print("  2. Pass pre-loaded odds_data to generate_picks()")
            print("  3. Load odds from: inputs/odds/player_props_latest.json")
            return PicksReport(
                generated_at=datetime.now().isoformat(),
                week=week,
                games_analyzed=0,
                props_screened=0,
                single_picks=[],
                parlays=[],
                total_edge_dollars=0,
                total_stake_dollars=0,
                best_single=None,
                best_parlay=None,
                by_game_exposure={},
                by_prop_type={},
            )

        # Screen all props
        all_picks = []
        games_analyzed = set()

        for game in odds_data:
            game_id = game.get('id', 'unknown')
            games_analyzed.add(game_id)

            for bookmaker in game.get('bookmakers', []):
                for market in bookmaker.get('markets', []):
                    market_key = market.get('key', '')

                    # Skip non-player props
                    if not market_key.startswith('player_'):
                        continue

                    for outcome in market.get('outcomes', []):
                        pick = self._analyze_prop_outcome(
                            game, market_key, outcome, player_features
                        )
                        if pick:
                            all_picks.append(pick)

        print(f"Screened {len(all_picks)} props from {len(games_analyzed)} games")

        # Filter to value picks
        value_picks = [
            p for p in all_picks
            if p.edge >= self.min_edge and p.trust_score >= self.min_trust_score
        ]

        print(f"Found {len(value_picks)} value picks (edge >= {self.min_edge}%)")

        # Sort by edge
        value_picks.sort(key=lambda p: p.edge, reverse=True)

        # Apply risk management (limit per game)
        single_picks = self._apply_risk_limits(value_picks)

        # Build parlays
        parlays = self._build_parlays(value_picks)

        # Calculate summary stats
        total_stake = sum(p.recommended_stake_dollars for p in single_picks)
        total_ev = sum(p.ev * p.recommended_stake_dollars for p in single_picks)

        # Game exposure
        by_game = {}
        for pick in single_picks:
            by_game[pick.game_id] = by_game.get(pick.game_id, 0) + pick.recommended_stake_dollars

        # Prop type breakdown
        by_prop = {}
        for pick in single_picks:
            by_prop[pick.prop_type] = by_prop.get(pick.prop_type, 0) + 1

        report = PicksReport(
            generated_at=datetime.now().isoformat(),
            week=week,
            games_analyzed=len(games_analyzed),
            props_screened=len(all_picks),
            single_picks=single_picks,
            parlays=parlays,
            total_edge_dollars=total_ev,
            total_stake_dollars=total_stake,
            best_single=single_picks[0] if single_picks else None,
            best_parlay=parlays[0] if parlays else None,
            by_game_exposure=by_game,
            by_prop_type=by_prop,
        )

        return report

    def _analyze_prop_outcome(
        self,
        game: Dict,
        market_key: str,
        outcome: Dict,
        player_features: pd.DataFrame
    ) -> Optional[PickRecommendation]:
        """Analyze a single prop outcome for value."""
        # Extract prop details
        player_name = outcome.get('description', outcome.get('name', ''))
        side = outcome.get('name', 'over').upper()
        line = outcome.get('point', 0)
        odds = outcome.get('price', -110)

        # Map market key to prop type
        prop_type_map = {
            'player_pass_yds': 'pass_yds',
            'player_pass_tds': 'pass_tds',
            'player_rush_yds': 'rush_yds',
            'player_receptions': 'receptions',
            'player_reception_yds': 'rec_yds',
        }

        prop_type = prop_type_map.get(market_key, market_key.replace('player_', ''))

        # Validate player name
        if not player_name or not player_name.strip():
            return None

        # Find player in features
        first_name = player_name.split()[0] if player_name.split() else player_name
        player_df = player_features[
            player_features['player_display_name'].str.contains(first_name, case=False, na=False)
        ]

        if len(player_df) == 0:
            return None

        # Get latest features
        player_id = player_df.iloc[0]['player_id']
        features = self._get_latest_player_features(player_id, player_features)

        if not features:
            return None

        # Generate prediction
        projection, std_dev = self._predict_prop(player_id, prop_type, features)

        if projection <= 0:
            return None

        # Calculate probabilities and edge
        hit_prob = self._calculate_hit_probability(projection, std_dev, line, side)
        edge = self._calculate_edge(hit_prob, odds)
        ev = self._calculate_ev(hit_prob, odds)

        # Skip if no edge
        if edge < 0:
            return None

        # Grade and sizing
        grade = self._grade_pick(edge)
        stake_pct = self._calculate_kelly_stake(hit_prob, odds)

        # Trust score (simplified - in production use meta-trust model)
        games_played = features.get('games_played', 1)
        trust_score = min(1.0, games_played / 10)  # More games = more trust

        # Confidence
        if edge >= 10 and hit_prob >= 0.55:
            confidence = "HIGH"
        elif edge >= 5 and hit_prob >= 0.50:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # Notes
        notes = []
        if projection > line * 1.1 and side == "OVER":
            notes.append(f"Projection {projection:.1f} is {((projection/line)-1)*100:.0f}% above line")
        if edge >= 10:
            notes.append("Strong edge - prioritize this pick")

        return PickRecommendation(
            player_id=player_id,
            player_name=player_name,
            game_id=game.get('id', 'unknown'),
            team=features.get('team', 'UNK'),
            opponent=features.get('opponent_team', 'UNK'),
            prop_type=prop_type,
            line=line,
            side=side,
            odds=odds,
            projection=round(projection, 1),
            hit_probability=round(hit_prob, 3),
            edge=round(edge, 1),
            ev=round(ev, 3),
            grade=grade,
            trust_score=round(trust_score, 2),
            confidence=confidence,
            recommended_stake_pct=round(stake_pct * 100, 2),
            recommended_stake_dollars=round(stake_pct * self.bankroll, 2),
            notes=notes,
        )

    def _apply_risk_limits(
        self,
        picks: List[PickRecommendation]
    ) -> List[PickRecommendation]:
        """Apply risk management limits."""
        filtered = []
        by_game = {}
        total_risk = 0.0

        for pick in picks:
            # Check game limit
            game_risk = by_game.get(pick.game_id, 0)
            if game_risk >= self.max_risk_per_game * self.bankroll:
                continue

            # Check total limit
            if total_risk >= self.max_total_risk * self.bankroll:
                break

            # Adjust stake if needed
            remaining_game = self.max_risk_per_game * self.bankroll - game_risk
            remaining_total = self.max_total_risk * self.bankroll - total_risk

            actual_stake = min(
                pick.recommended_stake_dollars,
                remaining_game,
                remaining_total
            )

            if actual_stake < 5:  # Minimum $5 bet
                continue

            # Update pick with adjusted stake
            pick.recommended_stake_dollars = round(actual_stake, 2)
            pick.recommended_stake_pct = round(actual_stake / self.bankroll * 100, 2)

            filtered.append(pick)
            by_game[pick.game_id] = game_risk + actual_stake
            total_risk += actual_stake

        return filtered

    def _build_parlays(
        self,
        picks: List[PickRecommendation]
    ) -> List[ParlayRecommendation]:
        """Build parlay suggestions from value picks."""
        parlays = []

        if len(picks) < 2:
            return parlays

        # Strategy 1: 2-leg cross-game parlay
        by_game = {}
        for pick in picks:
            if pick.game_id not in by_game:
                by_game[pick.game_id] = pick

        if len(by_game) >= 2:
            games = list(by_game.keys())[:2]
            legs = [by_game[g] for g in games]

            parlay = self._create_parlay(legs)
            if parlay:
                parlays.append(parlay)

        # Strategy 2: 3-leg if enough games
        if len(by_game) >= 3:
            games = list(by_game.keys())[:3]
            legs = [by_game[g] for g in games]

            parlay = self._create_parlay(legs)
            if parlay:
                parlays.append(parlay)

        return parlays

    def _create_parlay(
        self,
        legs: List[PickRecommendation]
    ) -> Optional[ParlayRecommendation]:
        """Create a parlay from legs."""
        if not legs:
            return None

        # Combined odds
        decimal_odds = 1.0
        for leg in legs:
            if leg.odds > 0:
                dec = 1 + (leg.odds / 100)
            else:
                dec = 1 + (100 / abs(leg.odds))
            decimal_odds *= dec

        # Guard against division by zero
        if decimal_odds <= 1.0:
            return None  # Invalid odds

        if decimal_odds >= 2.0:
            combined_odds = int((decimal_odds - 1) * 100)
        else:
            combined_odds = int(-100 / (decimal_odds - 1))

        # Combined probability (naive - assume independence)
        combined_prob = np.prod([leg.hit_probability for leg in legs])

        # EV
        ev = self._calculate_ev(combined_prob, combined_odds)

        # Stake
        stake_pct = self._calculate_kelly_stake(combined_prob, combined_odds, kelly_fraction=0.15)

        return ParlayRecommendation(
            legs=legs,
            combined_odds=combined_odds,
            combined_probability=round(combined_prob, 4),
            expected_value=round(ev, 3),
            recommended_stake_pct=round(stake_pct * 100, 2),
            recommended_stake_dollars=round(stake_pct * self.bankroll, 2),
            correlation_adjustment=1.0,
            confidence="MEDIUM" if ev > 0.10 else "LOW",
            notes=[f"{len(legs)}-leg parlay", "Cross-game (uncorrelated)"],
        )

    def _fetch_odds(self) -> List[Dict]:
        """Fetch current odds from The Odds API."""
        if not self.odds_api_key:
            return []

        import requests

        url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"
        params = {
            'apiKey': self.odds_api_key,
            'regions': 'us',
            'markets': 'player_pass_yds,player_rush_yds,player_receptions,player_reception_yds',
            'oddsFormat': 'american',
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error fetching odds: {e}")

        return []

    def _load_saved_odds(self) -> List[Dict]:
        """Load saved odds from disk if available."""
        odds_file = Path("inputs/odds/player_props_latest.json")
        if not odds_file.exists():
            return []

        try:
            with open(odds_file) as f:
                data = json.load(f)
            return data.get('events', [])
        except Exception as e:
            print(f"Error loading saved odds: {e}")
            return []

    def to_json(self, report: PicksReport) -> str:
        """Convert report to JSON string."""
        def serialize(obj):
            if hasattr(obj, '__dict__'):
                return asdict(obj) if hasattr(obj, '__dataclass_fields__') else obj.__dict__
            return str(obj)

        return json.dumps(asdict(report), default=serialize, indent=2)


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate NFL betting picks")
    parser.add_argument("--week", type=int, required=True, help="NFL week")
    parser.add_argument("--bankroll", type=float, default=1000, help="Bankroll size")
    parser.add_argument("--min-edge", type=float, default=3.0, help="Minimum edge %")

    args = parser.parse_args()

    pipeline = PicksPipeline(
        bankroll=args.bankroll,
        min_edge=args.min_edge,
    )

    report = pipeline.generate_picks(week=args.week)

    print(f"\n{'='*60}")
    print("PICKS SUMMARY")
    print(f"{'='*60}")
    print(f"Single picks: {len(report.single_picks)}")
    print(f"Parlays: {len(report.parlays)}")
    print(f"Total stake: ${report.total_stake_dollars:.2f}")
    print(f"Expected edge: ${report.total_edge_dollars:.2f}")

    if report.best_single:
        print(f"\nBest single: {report.best_single.player_name} {report.best_single.prop_type} {report.best_single.side} {report.best_single.line}")
        print(f"  Edge: {report.best_single.edge}%, Grade: {report.best_single.grade}")
