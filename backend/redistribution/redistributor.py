"""Volume redistribution when players are inactive"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from backend.config.logging_config import get_logger
from backend.models.prop_models import PropProjection
from backend.roster_injury import RosterInjuryService, PlayerGameStatus
from backend.database.session import get_db
from backend.database.models import Player

logger = get_logger(__name__)


@dataclass
class RedistributionResult:
    """Result of volume redistribution"""
    game_id: str
    team: str
    market: str

    # Redistribution details
    inactive_players: List[str]
    total_inactive_mu: float
    active_players: List[str]
    redistributed_amounts: Dict[str, float]  # player_id -> added mu

    # Metadata
    method: str  # proportional, redzone_weighted, qb_backup


class VolumeRedistributor:
    """
    Redistribute projected volume when players are inactive

    Algorithm:
    1. Identify inactive players (OUT, IR, DEV, CUT, RET)
    2. Calculate total inactive mu for each market/team
    3. Redistribute to active players proportionally
    4. Special handling for redzone and QB scenarios
    """

    def __init__(self):
        self.roster_service = RosterInjuryService(use_cache=True)

    def redistribute_game_projections(
        self,
        game_id: str,
        projections: List[PropProjection],
        save_updates: bool = True
    ) -> Tuple[List[PropProjection], List[RedistributionResult]]:
        """
        Redistribute volume for all projections in a game

        Args:
            game_id: Game ID
            projections: Original projections
            save_updates: Whether to save updated projections

        Returns:
            (updated_projections, redistribution_results)
        """
        logger.info("starting_redistribution", game_id=game_id, projections=len(projections))

        # Get all player statuses
        player_ids = list(set(p.player_id for p in projections))
        player_statuses = self.roster_service.get_batch_status(
            [(pid, game_id) for pid in player_ids]
        )

        # Separate active and inactive projections
        active_projections = []
        inactive_projections = []

        for proj in projections:
            status = player_statuses.get((proj.player_id, game_id))
            if status and not status.is_active:
                inactive_projections.append(proj)
            else:
                active_projections.append(proj)

        logger.info(
            "projections_classified",
            active=len(active_projections),
            inactive=len(inactive_projections)
        )

        if not inactive_projections:
            logger.info("no_inactive_players")
            return projections, []

        # Group by team and market
        active_by_team_market = self._group_by_team_market(active_projections)
        inactive_by_team_market = self._group_by_team_market(inactive_projections)

        redistribution_results = []
        updated_projections = list(active_projections)

        # Redistribute for each team/market combination
        for (team, market), inactive_projs in inactive_by_team_market.items():
            active_projs = active_by_team_market.get((team, market), [])

            if not active_projs:
                logger.warning(
                    "no_active_players_for_redistribution",
                    team=team,
                    market=market
                )
                continue

            # Choose redistribution method
            if self._is_redzone_market(market):
                result = self._redistribute_redzone(
                    game_id, team, market, inactive_projs, active_projs
                )
            elif self._is_qb_market(market):
                result = self._redistribute_qb(
                    game_id, team, market, inactive_projs, active_projs
                )
            else:
                result = self._redistribute_proportional(
                    game_id, team, market, inactive_projs, active_projs
                )

            redistribution_results.append(result)

        logger.info(
            "redistribution_complete",
            game_id=game_id,
            redistributions=len(redistribution_results)
        )

        return updated_projections, redistribution_results

    def _redistribute_proportional(
        self,
        game_id: str,
        team: str,
        market: str,
        inactive_projs: List[PropProjection],
        active_projs: List[PropProjection]
    ) -> RedistributionResult:
        """
        Proportional redistribution based on existing mu

        Formula: new_mu = mu + inactive_mu * (mu / active_total_mu)
        """
        # Calculate inactive total
        inactive_mu = sum(p.mu for p in inactive_projs)
        inactive_player_ids = [p.player_id for p in inactive_projs]

        # Calculate active total
        active_mu = sum(p.mu for p in active_projs)
        active_player_ids = [p.player_id for p in active_projs]

        if active_mu == 0:
            # Equal distribution if all active players have 0
            redistributed_amounts = {
                p.player_id: inactive_mu / len(active_projs)
                for p in active_projs
            }
        else:
            # Proportional distribution
            redistributed_amounts = {
                p.player_id: inactive_mu * (p.mu / active_mu)
                for p in active_projs
            }

        # Update projections
        for proj in active_projs:
            added = redistributed_amounts[proj.player_id]
            proj.mu += added

            # Update distribution params if available
            if proj.dist_params:
                if 'lambda' in proj.dist_params:
                    proj.dist_params['lambda'] = proj.mu
                elif 'mu' in proj.dist_params:
                    # For lognormal, adjust mu parameter (this is approximate)
                    proj.dist_params['mu'] = proj.mu

        logger.info(
            "proportional_redistribution",
            game_id=game_id,
            team=team,
            market=market,
            inactive_mu=inactive_mu,
            active_count=len(active_projs)
        )

        return RedistributionResult(
            game_id=game_id,
            team=team,
            market=market,
            inactive_players=inactive_player_ids,
            total_inactive_mu=inactive_mu,
            active_players=active_player_ids,
            redistributed_amounts=redistributed_amounts,
            method="proportional"
        )

    def _redistribute_redzone(
        self,
        game_id: str,
        team: str,
        market: str,
        inactive_projs: List[PropProjection],
        active_projs: List[PropProjection]
    ) -> RedistributionResult:
        """
        Redzone redistribution weighted by redzone share

        For TD props, we want to weight by redzone usage, not overall usage
        """
        inactive_mu = sum(p.mu for p in inactive_projs)
        inactive_player_ids = [p.player_id for p in inactive_projs]
        active_player_ids = [p.player_id for p in active_projs]

        # Get redzone shares from database or estimate
        redzone_shares = self._get_redzone_shares(
            game_id, [p.player_id for p in active_projs]
        )

        total_redzone_share = sum(redzone_shares.values())

        if total_redzone_share == 0:
            # Fallback to proportional if no redzone data
            return self._redistribute_proportional(
                game_id, team, market, inactive_projs, active_projs
            )

        # Redistribute based on redzone share
        redistributed_amounts = {}
        for proj in active_projs:
            share = redzone_shares.get(proj.player_id, 0)
            added = inactive_mu * (share / total_redzone_share)
            redistributed_amounts[proj.player_id] = added
            proj.mu += added

            # Update prob for TD markets
            if 'p' in proj.dist_params:
                # Recompute TD probability
                # P(TD >= 1) = 1 - exp(-lambda)
                proj.dist_params['p'] = 1 - (1 - proj.dist_params['p']) * (1 - added)
                proj.mu = proj.dist_params['p']

        logger.info(
            "redzone_redistribution",
            game_id=game_id,
            team=team,
            market=market,
            inactive_mu=inactive_mu,
            total_redzone_share=total_redzone_share
        )

        return RedistributionResult(
            game_id=game_id,
            team=team,
            market=market,
            inactive_players=inactive_player_ids,
            total_inactive_mu=inactive_mu,
            active_players=active_player_ids,
            redistributed_amounts=redistributed_amounts,
            method="redzone_weighted"
        )

    def _redistribute_qb(
        self,
        game_id: str,
        team: str,
        market: str,
        inactive_projs: List[PropProjection],
        active_projs: List[PropProjection]
    ) -> RedistributionResult:
        """
        QB redistribution to backup QB

        When starting QB is out, backup gets majority of volume
        """
        inactive_mu = sum(p.mu for p in inactive_projs)
        inactive_player_ids = [p.player_id for p in inactive_projs]
        active_player_ids = [p.player_id for p in active_projs]

        if len(active_projs) == 0:
            logger.warning("no_backup_qb", team=team, game_id=game_id)
            return RedistributionResult(
                game_id=game_id,
                team=team,
                market=market,
                inactive_players=inactive_player_ids,
                total_inactive_mu=inactive_mu,
                active_players=[],
                redistributed_amounts={},
                method="qb_backup"
            )

        # Get depth chart positions
        depth_positions = self._get_depth_positions(
            game_id, [p.player_id for p in active_projs]
        )

        # Find highest on depth chart (lowest number)
        backup_qb = min(active_projs, key=lambda p: depth_positions.get(p.player_id, 99))

        # Backup gets most of the volume (90%), others share 10%
        redistributed_amounts = {}

        # Backup QB gets 90%
        backup_amount = inactive_mu * 0.9
        redistributed_amounts[backup_qb.player_id] = backup_amount
        backup_qb.mu += backup_amount

        # Others share remaining 10%
        other_qbs = [p for p in active_projs if p.player_id != backup_qb.player_id]
        if other_qbs:
            per_qb = (inactive_mu * 0.1) / len(other_qbs)
            for qb in other_qbs:
                redistributed_amounts[qb.player_id] = per_qb
                qb.mu += per_qb

        logger.info(
            "qb_redistribution",
            game_id=game_id,
            team=team,
            backup_qb=backup_qb.player_id,
            backup_gets=backup_amount
        )

        return RedistributionResult(
            game_id=game_id,
            team=team,
            market=market,
            inactive_players=inactive_player_ids,
            total_inactive_mu=inactive_mu,
            active_players=active_player_ids,
            redistributed_amounts=redistributed_amounts,
            method="qb_backup"
        )

    def _group_by_team_market(
        self,
        projections: List[PropProjection]
    ) -> Dict[Tuple[str, str], List[PropProjection]]:
        """Group projections by (team, market)"""
        grouped = defaultdict(list)
        for proj in projections:
            key = (proj.team, proj.market)
            grouped[key].append(proj)
        return dict(grouped)

    def _is_redzone_market(self, market: str) -> bool:
        """Check if market is redzone-related"""
        return 'td' in market.lower() or 'touchdown' in market.lower()

    def _is_qb_market(self, market: str) -> bool:
        """Check if market is QB-specific"""
        return 'pass' in market.lower() and 'rec' not in market.lower()

    def _get_redzone_shares(
        self,
        game_id: str,
        player_ids: List[str]
    ) -> Dict[str, float]:
        """
        Get redzone shares for players

        In production, this would come from historical feature data
        For now, estimate based on position
        """
        with get_db() as session:
            players = (
                session.query(Player)
                .filter(Player.player_id.in_(player_ids))
                .all()
            )

            # Simple heuristic: RBs get higher redzone share
            redzone_shares = {}
            for player in players:
                if player.position == 'RB':
                    redzone_shares[player.player_id] = 0.4
                elif player.position == 'TE':
                    redzone_shares[player.player_id] = 0.3
                elif player.position == 'WR':
                    redzone_shares[player.player_id] = 0.2
                elif player.position == 'QB':
                    redzone_shares[player.player_id] = 0.1
                else:
                    redzone_shares[player.player_id] = 0.1

        return redzone_shares

    def _get_depth_positions(
        self,
        game_id: str,
        player_ids: List[str]
    ) -> Dict[str, int]:
        """
        Get depth chart positions

        Lower number = higher on depth chart
        """
        # Get from roster status
        statuses = self.roster_service.get_batch_status(
            [(pid, game_id) for pid in player_ids]
        )

        depth_positions = {}
        for player_id in player_ids:
            status = statuses.get((player_id, game_id))
            if status and status.depth_chart_position:
                depth_positions[player_id] = status.depth_chart_position
            else:
                # Default to low priority if unknown
                depth_positions[player_id] = 99

        return depth_positions

    def get_redistribution_summary(
        self,
        results: List[RedistributionResult]
    ) -> Dict:
        """Get summary statistics of redistributions"""
        if not results:
            return {
                'total_redistributions': 0,
                'total_volume_redistributed': 0,
                'methods': {},
                'markets': {}
            }

        total_volume = sum(r.total_inactive_mu for r in results)

        methods = defaultdict(int)
        markets = defaultdict(float)

        for result in results:
            methods[result.method] += 1
            markets[result.market] += result.total_inactive_mu

        return {
            'total_redistributions': len(results),
            'total_volume_redistributed': total_volume,
            'methods': dict(methods),
            'markets': dict(markets),
            'avg_volume_per_redistribution': total_volume / len(results) if results else 0
        }
