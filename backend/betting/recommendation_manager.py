"""Recommendation Manager - CLV Feedback Loop Integration

This module wires together:
1. Prop recommendations (from prop_analyzer or portfolio_optimizer)
2. CLV tracking (from clv_tracker)
3. Meta trust model training data (from meta_trust_model)

The feedback loop:
  Recommendation → Log to CLV Tracker → Update Closing Line → Update Result
  → Feed into Meta Trust Model → Improve Future Recommendations

This is the "learning" component that makes the system self-improving.
"""

from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import json
from dataclasses import dataclass, asdict

from backend.betting.clv_tracker import CLVTracker


@dataclass
class PropRecommendation:
    """A prop recommendation to be logged."""
    # Identity
    player_id: str
    player_name: str
    game_id: str
    prop_type: str

    # Line details
    side: str  # 'over' or 'under'
    line: float
    odds: int  # American odds

    # Model outputs
    projection: float
    hit_probability: float
    edge: float
    ev: float

    # Quality signals
    trust_score: Optional[float] = None
    games_sampled: Optional[int] = None
    model_r2: Optional[float] = None

    # Recommendation metadata
    recommendation: str = "BET"  # BET, CONSIDER, PASS
    confidence: str = "MEDIUM"  # HIGH, MEDIUM, LOW
    stake_pct: Optional[float] = None  # Recommended stake as % of bankroll

    # Timestamps
    timestamp: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dict."""
        return asdict(self)


@dataclass
class ParlayRecommendation:
    """A parlay recommendation to be logged."""
    # Legs
    legs: List[PropRecommendation]

    # Combined metrics
    combined_odds: int
    combined_probability: float
    raw_probability: float  # If independent
    ev: float

    # Risk
    stake_pct: float
    confidence: str

    # Correlation
    correlation_adjustment: float
    correlation_theme: str

    # Timestamps
    timestamp: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dict."""
        return {
            'legs': [leg.to_dict() for leg in self.legs],
            'combined_odds': self.combined_odds,
            'combined_probability': self.combined_probability,
            'raw_probability': self.raw_probability,
            'ev': self.ev,
            'stake_pct': self.stake_pct,
            'confidence': self.confidence,
            'correlation_adjustment': self.correlation_adjustment,
            'correlation_theme': self.correlation_theme,
            'timestamp': self.timestamp or datetime.now().isoformat()
        }


class RecommendationManager:
    """Manages recommendations and CLV feedback loop."""

    def __init__(
        self,
        clv_storage_file: Path = Path("outputs/betting/clv_bets.json"),
        recommendations_file: Path = Path("outputs/betting/recommendations.json")
    ):
        """Initialize recommendation manager.

        Args:
            clv_storage_file: Path to CLV tracker storage
            recommendations_file: Path to recommendations log
        """
        self.clv_tracker = CLVTracker(storage_file=clv_storage_file)
        self.recommendations_file = recommendations_file
        self.recommendations_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing recommendations
        if self.recommendations_file.exists():
            with open(self.recommendations_file, 'r') as f:
                self.recommendations = json.load(f)
        else:
            self.recommendations = {
                'single_props': [],
                'parlays': []
            }

    def log_prop_recommendation(
        self,
        recommendation: PropRecommendation,
        actually_bet: bool = False
    ) -> str:
        """Log a single prop recommendation.

        Args:
            recommendation: Prop recommendation
            actually_bet: Whether this was actually bet (for CLV tracking)

        Returns:
            Bet ID for tracking
        """
        # Generate bet ID
        bet_id = f"{recommendation.game_id}_{recommendation.player_id}_{recommendation.prop_type}_{recommendation.side}"

        # Add to recommendations log
        rec_dict = recommendation.to_dict()
        rec_dict['bet_id'] = bet_id
        rec_dict['actually_bet'] = actually_bet
        rec_dict['timestamp'] = rec_dict.get('timestamp') or datetime.now().isoformat()

        self.recommendations['single_props'].append(rec_dict)
        self._save_recommendations()

        # If actually bet, log to CLV tracker
        if actually_bet:
            self.clv_tracker.log_bet(
                bet_id=bet_id,
                player_name=recommendation.player_name,
                prop_type=recommendation.prop_type,
                side=recommendation.side,
                opening_line=recommendation.line,
                opening_odds=recommendation.odds,
                model_projection=recommendation.projection,
                model_edge=recommendation.edge,
                game_id=recommendation.game_id,
                timestamp=rec_dict['timestamp']
            )

        return bet_id

    def log_parlay_recommendation(
        self,
        parlay: ParlayRecommendation,
        actually_bet: bool = False
    ) -> str:
        """Log a parlay recommendation.

        Args:
            parlay: Parlay recommendation
            actually_bet: Whether this was actually bet

        Returns:
            Parlay ID for tracking
        """
        # Generate parlay ID
        game_ids = '_'.join(sorted(set(leg.game_id for leg in parlay.legs)))
        parlay_id = f"parlay_{game_ids}_{len(parlay.legs)}legs"

        # Add to recommendations log
        parlay_dict = parlay.to_dict()
        parlay_dict['parlay_id'] = parlay_id
        parlay_dict['actually_bet'] = actually_bet
        parlay_dict['timestamp'] = parlay_dict.get('timestamp') or datetime.now().isoformat()

        self.recommendations['parlays'].append(parlay_dict)
        self._save_recommendations()

        # If actually bet, log each leg to CLV tracker
        if actually_bet:
            for i, leg in enumerate(parlay.legs):
                leg_bet_id = f"{parlay_id}_leg{i}"
                self.clv_tracker.log_bet(
                    bet_id=leg_bet_id,
                    player_name=leg.player_name,
                    prop_type=leg.prop_type,
                    side=leg.side,
                    opening_line=leg.line,
                    opening_odds=leg.odds,
                    model_projection=leg.projection,
                    model_edge=leg.edge,
                    game_id=leg.game_id,
                    timestamp=parlay_dict['timestamp']
                )

        return parlay_id

    def update_closing_lines_from_odds_snapshot(
        self,
        odds_snapshot: Dict[str, Dict]
    ) -> Dict[str, float]:
        """Update closing lines for all pending bets from odds snapshot.

        Args:
            odds_snapshot: Dict mapping game_id -> odds data
                Format: {
                    'game_id': {
                        'props': [
                            {
                                'player_name': 'Patrick Mahomes',
                                'prop_type': 'player_pass_yds',
                                'over_line': 280.5,
                                'over_odds': -110,
                                ...
                            }
                        ]
                    }
                }

        Returns:
            Dict mapping bet_id -> CLV
        """
        clv_results = {}

        # Get all pending/closed bets from CLV tracker
        pending_bets = [b for b in self.clv_tracker.bets if b['status'] in ['pending', 'closed']]

        for bet in pending_bets:
            game_id = bet['game_id']

            if game_id not in odds_snapshot:
                continue

            # Find matching prop in odds snapshot
            for prop in odds_snapshot[game_id].get('props', []):
                if (prop.get('player_name') == bet['player_name'] and
                    prop.get('prop_type') == bet['prop_type']):

                    # Extract closing line based on side
                    if bet['side'] == 'over':
                        closing_line = prop.get('over_line')
                        closing_odds = prop.get('over_odds')
                    else:
                        closing_line = prop.get('under_line')
                        closing_odds = prop.get('under_odds')

                    if closing_line is not None and closing_odds is not None:
                        # Update closing line
                        updated_bet = self.clv_tracker.update_closing_line(
                            bet_id=bet['bet_id'],
                            closing_line=closing_line,
                            closing_odds=closing_odds
                        )

                        clv_results[bet['bet_id']] = updated_bet['clv']

                    break

        return clv_results

    def update_results_from_boxscores(
        self,
        boxscores: Dict[str, Dict]
    ) -> Dict[str, bool]:
        """Update actual results for bets from boxscores.

        Args:
            boxscores: Dict mapping game_id -> boxscore data
                Format: {
                    'game_id': {
                        'players': [
                            {
                                'player_name': 'Patrick Mahomes',
                                'passing_yards': 318,
                                'rushing_yards': 12,
                                ...
                            }
                        ]
                    }
                }

        Returns:
            Dict mapping bet_id -> won (bool)
        """
        results = {}

        # Get all closed bets (have closing line but no result yet)
        closed_bets = [b for b in self.clv_tracker.bets if b['status'] == 'closed']

        for bet in closed_bets:
            game_id = bet['game_id']

            if game_id not in boxscores:
                continue

            # Find matching player in boxscore
            for player in boxscores[game_id].get('players', []):
                if player.get('player_name') == bet['player_name']:
                    # Extract actual stat based on prop type
                    actual_result = self._extract_stat_from_boxscore(
                        player, bet['prop_type']
                    )

                    if actual_result is not None:
                        # Update result
                        updated_bet = self.clv_tracker.update_result(
                            bet_id=bet['bet_id'],
                            actual_result=actual_result
                        )

                        results[bet['bet_id']] = updated_bet['won']

                    break

        return results

    def _extract_stat_from_boxscore(
        self,
        player_stats: Dict,
        prop_type: str
    ) -> Optional[float]:
        """Extract relevant stat from player boxscore.

        Args:
            player_stats: Player stats dict
            prop_type: Prop type (e.g., 'player_pass_yds')

        Returns:
            Stat value or None if not found
        """
        # Map prop types to stat keys
        stat_map = {
            'player_pass_yds': 'passing_yards',
            'player_pass_tds': 'passing_tds',
            'player_pass_completions': 'completions',
            'player_pass_attempts': 'attempts',
            'player_rush_yds': 'rushing_yards',
            'player_rush_tds': 'rushing_tds',
            'player_rush_attempts': 'rushing_attempts',
            'player_receptions': 'receptions',
            'player_reception_yds': 'receiving_yards',
            'player_reception_tds': 'receiving_tds',
        }

        stat_key = stat_map.get(prop_type)
        if stat_key:
            return player_stats.get(stat_key)

        return None

    def generate_meta_training_data(self) -> List[Dict]:
        """Generate training data for meta trust model from bet history.

        Returns:
            List of training samples with features and labels
        """
        training_data = []

        # Get all resulted bets
        resulted_bets = [b for b in self.clv_tracker.bets if b['status'] == 'resulted']

        for bet in resulted_bets:
            # Features for meta model
            features = {
                'prop_type': bet['prop_type'],
                'edge_size': bet['model_edge'],
                'edge_bucket': self._bucket_edge(bet['model_edge']),
                'side': bet['side'],

                # CLV signals
                'clv': bet.get('clv', 0),
                'clv_positive': 1 if bet.get('clv', 0) > 0 else 0,

                # Can add more features from recommendation metadata
                # (trust_score, games_sampled, etc.)
            }

            # Label
            label = 1 if bet['won'] else 0

            training_data.append({
                'features': features,
                'label': label,
                'bet_id': bet['bet_id']
            })

        return training_data

    def _bucket_edge(self, edge: float) -> str:
        """Bucket edge into categories."""
        if edge >= 0.10:
            return 'high'
        elif edge >= 0.05:
            return 'medium'
        else:
            return 'low'

    def _save_recommendations(self):
        """Save recommendations to file."""
        with open(self.recommendations_file, 'w') as f:
            json.dump(self.recommendations, f, indent=2)

    def get_recent_clv_summary(self, last_n: int = 50) -> Dict:
        """Get CLV summary for recent bets.

        Args:
            last_n: Number of recent bets to analyze

        Returns:
            Summary dict with CLV metrics
        """
        # Get recent closed bets
        closed_bets = [b for b in self.clv_tracker.bets if b.get('clv') is not None][-last_n:]

        if not closed_bets:
            return {'error': 'No CLV data available'}

        clvs = [b['clv'] for b in closed_bets]

        return {
            'total_bets': len(closed_bets),
            'avg_clv': round(sum(clvs) / len(clvs), 2),
            'positive_clv_rate': round(sum(1 for c in clvs if c > 0) / len(clvs), 3),
            'median_clv': round(sorted(clvs)[len(clvs) // 2], 2),
            'max_clv': round(max(clvs), 2),
            'min_clv': round(min(clvs), 2),
        }


# Global instance
recommendation_manager = RecommendationManager()
