"""Tier assignment for projections (Core/Mid/Lotto)"""
from typing import List, Dict, Tuple
from dataclasses import dataclass

from backend.config import settings
from backend.config.logging_config import get_logger
from .scorer import ScoredProjection

logger = get_logger(__name__)


@dataclass
class TierAssignment:
    """Tier assignment result"""
    core: List[ScoredProjection]
    mid: List[ScoredProjection]
    lotto: List[ScoredProjection]

    def get_by_tier(self, tier: str) -> List[ScoredProjection]:
        """Get projections by tier name"""
        tier = tier.lower()
        if tier == 'core':
            return self.core
        elif tier == 'mid':
            return self.mid
        elif tier == 'lotto':
            return self.lotto
        else:
            return []

    def to_dict(self) -> Dict[str, List[ScoredProjection]]:
        """Convert to dictionary"""
        return {
            'core': self.core,
            'mid': self.mid,
            'lotto': self.lotto
        }


class TierAssigner:
    """
    Assign projections to tiers based on score and characteristics

    Tiers:
    - Core: Highest scores, best opportunities, high confidence (N=8)
    - Mid: Medium scores, good value but higher variance (N=12)
    - Lotto: High variance, longshot opportunities with upside (N=5)
    """

    def __init__(self):
        self.core_count = settings.core_picks_count
        self.mid_count = settings.mid_picks_count
        self.lotto_count = settings.lotto_picks_count

    def assign_tiers(
        self,
        scored_projections: List[ScoredProjection],
        strategy: str = "score_based"
    ) -> TierAssignment:
        """
        Assign projections to tiers

        Args:
            scored_projections: List of scored projections
            strategy: Assignment strategy ("score_based", "variance_based", "hybrid")

        Returns:
            TierAssignment with categorized projections
        """
        if strategy == "score_based":
            return self._assign_score_based(scored_projections)
        elif strategy == "variance_based":
            return self._assign_variance_based(scored_projections)
        elif strategy == "hybrid":
            return self._assign_hybrid(scored_projections)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _assign_score_based(self, scored_projections: List[ScoredProjection]) -> TierAssignment:
        """
        Simple score-based assignment

        Top N by score -> Core
        Next M by score -> Mid
        High variance outliers -> Lotto
        """
        # Filter to positive EV only
        positive_ev = [sp for sp in scored_projections if sp.edge >= settings.min_edge_threshold]

        if not positive_ev:
            logger.warning("no_positive_ev_projections")
            return TierAssignment(core=[], mid=[], lotto=[])

        # Sort by score
        sorted_projs = sorted(positive_ev, key=lambda x: x.score, reverse=True)

        # Core: Top N highest scores
        core = sorted_projs[:self.core_count]

        # Mid: Next M highest scores
        remaining = sorted_projs[self.core_count:]
        mid = remaining[:self.mid_count]

        # Lotto: High volatility from remaining
        lotto_candidates = remaining[self.mid_count:]
        lotto = self._select_lotto(lotto_candidates)

        logger.info(
            "tiers_assigned_score_based",
            core=len(core),
            mid=len(mid),
            lotto=len(lotto)
        )

        return TierAssignment(core=core, mid=mid, lotto=lotto)

    def _assign_variance_based(self, scored_projections: List[ScoredProjection]) -> TierAssignment:
        """
        Variance-aware assignment

        Core: High score, low variance
        Mid: Medium score, medium variance
        Lotto: Lower score but high upside (high variance, good edge)
        """
        positive_ev = [sp for sp in scored_projections if sp.edge >= settings.min_edge_threshold]

        if not positive_ev:
            return TierAssignment(core=[], mid=[], lotto=[])

        # Separate by volatility
        low_vol = [sp for sp in positive_ev if sp.volatility < 0.5]
        med_vol = [sp for sp in positive_ev if 0.5 <= sp.volatility < 1.0]
        high_vol = [sp for sp in positive_ev if sp.volatility >= 1.0]

        # Core: Best low-volatility plays
        core_candidates = sorted(low_vol, key=lambda x: x.score, reverse=True)
        core = core_candidates[:self.core_count]

        # Mid: Best medium-volatility plays
        mid_candidates = sorted(med_vol, key=lambda x: x.score, reverse=True)
        mid = mid_candidates[:self.mid_count]

        # Lotto: Best high-volatility plays
        lotto_candidates = sorted(high_vol, key=lambda x: x.edge, reverse=True)
        lotto = lotto_candidates[:self.lotto_count]

        # Fill gaps if needed
        if len(core) < self.core_count:
            gap = self.core_count - len(core)
            core.extend(mid_candidates[:gap])
            mid = mid_candidates[gap:]

        logger.info(
            "tiers_assigned_variance_based",
            core=len(core),
            mid=len(mid),
            lotto=len(lotto)
        )

        return TierAssignment(core=core, mid=mid, lotto=lotto)

    def _assign_hybrid(self, scored_projections: List[ScoredProjection]) -> TierAssignment:
        """
        Hybrid approach combining score and variance

        Core: Top scores with confidence >= 0.7 and volatility < 0.6
        Mid: Good scores with moderate risk
        Lotto: High edge, high variance plays
        """
        positive_ev = [sp for sp in scored_projections if sp.edge >= settings.min_edge_threshold]

        if not positive_ev:
            return TierAssignment(core=[], mid=[], lotto=[])

        # Core candidates: High confidence, low-medium volatility
        core_candidates = [
            sp for sp in positive_ev
            if sp.projection.confidence >= 0.7 and sp.volatility < 0.6
        ]
        core_candidates = sorted(core_candidates, key=lambda x: x.score, reverse=True)
        core = core_candidates[:self.core_count]

        # Remove core picks from pool
        remaining = [sp for sp in positive_ev if sp not in core]

        # Mid candidates: Decent confidence, reasonable variance
        mid_candidates = [
            sp for sp in remaining
            if sp.projection.confidence >= 0.5 and sp.volatility < 1.0
        ]
        mid_candidates = sorted(mid_candidates, key=lambda x: x.score, reverse=True)
        mid = mid_candidates[:self.mid_count]

        # Lotto candidates: High variance, good edge
        lotto_candidates = [sp for sp in remaining if sp not in mid]
        lotto = self._select_lotto(lotto_candidates)

        # Backfill if needed
        if len(core) < self.core_count:
            gap = self.core_count - len(core)
            additional = sorted(remaining, key=lambda x: x.score, reverse=True)[:gap]
            core.extend([sp for sp in additional if sp not in mid and sp not in lotto])

        logger.info(
            "tiers_assigned_hybrid",
            core=len(core),
            mid=len(mid),
            lotto=len(lotto)
        )

        return TierAssignment(core=core, mid=mid, lotto=lotto)

    def _select_lotto(self, candidates: List[ScoredProjection]) -> List[ScoredProjection]:
        """
        Select lotto plays from candidates

        Prioritize:
        1. High edge (even if lower confidence)
        2. High volatility (upside potential)
        3. Reasonable model probability (not complete longshots)
        """
        if not candidates:
            return []

        # Calculate lotto score: edge * volatility * model_prob
        lotto_scored = [
            (sp, sp.edge * sp.volatility * sp.model_prob)
            for sp in candidates
        ]

        # Sort by lotto score
        lotto_scored.sort(key=lambda x: x[1], reverse=True)

        # Take top N
        lotto = [sp for sp, _ in lotto_scored[:self.lotto_count]]

        return lotto

    def get_tier_summary(self, assignment: TierAssignment) -> Dict:
        """Get summary statistics by tier"""
        def summarize_tier(tier_name: str, projections: List[ScoredProjection]) -> Dict:
            if not projections:
                return {
                    'count': 0,
                    'avg_score': 0,
                    'avg_edge': 0,
                    'avg_volatility': 0
                }

            return {
                'count': len(projections),
                'avg_score': float(sum(sp.score for sp in projections) / len(projections)),
                'avg_edge': float(sum(sp.edge for sp in projections) / len(projections)),
                'avg_volatility': float(sum(sp.volatility for sp in projections) / len(projections)),
                'avg_model_prob': float(sum(sp.model_prob for sp in projections) / len(projections)),
                'top_players': [
                    sp.projection.player_name for sp in projections[:3]
                ]
            }

        return {
            'core': summarize_tier('core', assignment.core),
            'mid': summarize_tier('mid', assignment.mid),
            'lotto': summarize_tier('lotto', assignment.lotto),
            'total_opportunities': len(assignment.core) + len(assignment.mid) + len(assignment.lotto)
        }

    def apply_tiers_to_projections(
        self,
        assignment: TierAssignment,
        update_db: bool = False
    ) -> None:
        """
        Apply tier assignments back to projection objects

        Args:
            assignment: TierAssignment result
            update_db: Whether to update database records
        """
        # Update projection objects
        for sp in assignment.core:
            sp.projection.tier = 'core'

        for sp in assignment.mid:
            sp.projection.tier = 'mid'

        for sp in assignment.lotto:
            sp.projection.tier = 'lotto'

        # Update database if requested
        if update_db:
            from backend.database.session import get_db
            from backend.database.models import Projection

            with get_db() as session:
                for sp in assignment.core + assignment.mid + assignment.lotto:
                    db_proj = (
                        session.query(Projection)
                        .filter(
                            Projection.player_id == sp.projection.player_id,
                            Projection.game_id == sp.projection.game_id,
                            Projection.market == sp.projection.market
                        )
                        .first()
                    )

                    if db_proj:
                        db_proj.tier = sp.projection.tier
                        db_proj.score = sp.score

            logger.info("tiers_updated_in_db", count=len(assignment.core) + len(assignment.mid) + len(assignment.lotto))
