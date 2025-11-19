"""Closing Line Value (CLV) Tracker - The GOLD STANDARD for Model Evaluation

CLV (Closing Line Value) is the PRIMARY metric for evaluating sports betting models.

WHY CLV MATTERS MORE THAN WIN/LOSS RECORD:
  - Short-term results are noisy (variance dominates over small samples)
  - You can beat closing line and still lose money short-term (bad luck)
  - You can beat closing line and win money long-term (good process)

  Research: "Bettors who beat closing lines by 1-2% have positive ROI over 10,000+ bets"
  Source: Pinnacle Sports analysis, academic papers on market efficiency

WHAT IS CLV:
  CLV = Closing Line - Opening Line (where you bet)

  Example:
    - You bet Patrick Mahomes OVER 275.5 yards at -110
    - Line closes at 280.5 (-110)
    - CLV = +5 yards (you got 5 extra yards of value!)

IF CLV > 0: You beat the closing line (GOOD!)
IF CLV < 0: Closing line moved against you (BAD)

INTERPRETATION:
  - Consistent positive CLV = Your model finds value
  - Negative CLV = You're betting on the wrong side of information
  - CLV matters MORE than short-term W/L record

THIS MODULE:
  1. Logs opening lines when bets are made
  2. Records closing lines before game starts
  3. Calculates CLV for each bet
  4. Generates reports on CLV performance by prop type, player, etc.
"""

from pathlib import Path
import json
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict
import numpy as np


class CLVTracker:
    """Track closing line value for all bets."""

    def __init__(self, storage_file: Path):
        """Initialize CLV tracker.

        Args:
            storage_file: Path to JSON file for storing bet records
        """
        self.storage_file = storage_file
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing records
        if self.storage_file.exists():
            with open(self.storage_file, 'r') as f:
                self.bets = json.load(f)
        else:
            self.bets = []

    def log_bet(
        self,
        bet_id: str,
        player_name: str,
        prop_type: str,
        side: str,  # 'over' or 'under'
        opening_line: float,
        opening_odds: int,
        model_projection: float,
        model_edge: float,
        game_id: str,
        timestamp: Optional[str] = None
    ) -> Dict:
        """Log a new bet with opening line.

        Args:
            bet_id: Unique bet identifier
            player_name: Player name
            prop_type: Type of prop (e.g., 'player_pass_yds')
            side: 'over' or 'under'
            opening_line: Line when bet was made
            opening_odds: Odds when bet was made (American format)
            model_projection: Model's projection
            model_edge: Model's calculated edge
            game_id: Game identifier
            timestamp: Timestamp of bet (defaults to now)

        Returns:
            Bet record dict
        """
        bet_record = {
            'bet_id': bet_id,
            'player_name': player_name,
            'prop_type': prop_type,
            'side': side,
            'opening_line': opening_line,
            'opening_odds': opening_odds,
            'model_projection': model_projection,
            'model_edge': model_edge,
            'game_id': game_id,
            'timestamp': timestamp or datetime.now().isoformat(),
            'status': 'pending',  # pending, closed, resulted
            'closing_line': None,
            'closing_odds': None,
            'clv': None,
            'clv_pct': None,
            'actual_result': None,
            'won': None
        }

        self.bets.append(bet_record)
        self._save()

        return bet_record

    def update_closing_line(
        self,
        bet_id: str,
        closing_line: float,
        closing_odds: int
    ) -> Dict:
        """Update bet record with closing line.

        Call this right before game starts to record closing line.

        Args:
            bet_id: Bet identifier
            closing_line: Line at close
            closing_odds: Odds at close

        Returns:
            Updated bet record with CLV calculated
        """
        bet = self._find_bet(bet_id)

        if not bet:
            raise ValueError(f"Bet {bet_id} not found")

        bet['closing_line'] = closing_line
        bet['closing_odds'] = closing_odds
        bet['status'] = 'closed'

        # Calculate CLV
        bet['clv'] = self._calculate_clv(
            side=bet['side'],
            opening_line=bet['opening_line'],
            closing_line=closing_line
        )

        bet['clv_pct'] = round((bet['clv'] / bet['opening_line']) * 100, 2) if bet['opening_line'] > 0 else 0

        self._save()

        return bet

    def update_result(
        self,
        bet_id: str,
        actual_result: float
    ) -> Dict:
        """Update bet with actual result after game.

        Args:
            bet_id: Bet identifier
            actual_result: Actual stat value (e.g., actual passing yards)

        Returns:
            Updated bet record with win/loss
        """
        bet = self._find_bet(bet_id)

        if not bet:
            raise ValueError(f"Bet {bet_id} not found")

        bet['actual_result'] = actual_result
        bet['status'] = 'resulted'

        # Determine if bet won
        if bet['side'] == 'over':
            bet['won'] = actual_result > bet['opening_line']
        else:  # under
            bet['won'] = actual_result < bet['opening_line']

        self._save()

        return bet

    def _calculate_clv(
        self,
        side: str,
        opening_line: float,
        closing_line: float
    ) -> float:
        """Calculate closing line value.

        Args:
            side: 'over' or 'under'
            opening_line: Line when bet was made
            closing_line: Line at close

        Returns:
            CLV (positive = beat closing line)
        """
        if side == 'over':
            # For OVER: Lower line at close = better for us
            # Example: Bet OVER 275.5, closes at 280.5 → CLV = +5
            clv = closing_line - opening_line
        else:  # under
            # For UNDER: Higher line at close = better for us
            # Example: Bet UNDER 275.5, closes at 270.5 → CLV = +5
            clv = opening_line - closing_line

        return round(clv, 2)

    def _find_bet(self, bet_id: str) -> Optional[Dict]:
        """Find bet by ID."""
        for bet in self.bets:
            if bet['bet_id'] == bet_id:
                return bet
        return None

    def _save(self):
        """Save bet records to file."""
        with open(self.storage_file, 'w') as f:
            json.dump(self.bets, f, indent=2)

    def generate_clv_report(self, output_file: Optional[Path] = None) -> Dict:
        """Generate comprehensive CLV report.

        Returns:
            CLV metrics dict
        """
        print(f"\n{'='*80}")
        print("CLV PERFORMANCE REPORT")
        print(f"{'='*80}\n")

        # Filter to bets with CLV calculated
        closed_bets = [b for b in self.bets if b['status'] in ['closed', 'resulted']]
        resulted_bets = [b for b in self.bets if b['status'] == 'resulted']

        if not closed_bets:
            print("⚠️  No closed bets to analyze")
            return {}

        # Overall CLV metrics
        clvs = [b['clv'] for b in closed_bets]
        clv_pcts = [b['clv_pct'] for b in closed_bets]

        overall = {
            'total_bets': len(closed_bets),
            'avg_clv': round(np.mean(clvs), 2),
            'median_clv': round(np.median(clvs), 2),
            'avg_clv_pct': round(np.mean(clv_pcts), 2),
            'positive_clv_rate': round(len([c for c in clvs if c > 0]) / len(clvs), 3),
            'total_clv': round(sum(clvs), 2)
        }

        print(f"Overall CLV Metrics:")
        print(f"  Total Bets: {overall['total_bets']}")
        print(f"  Avg CLV: {overall['avg_clv']:+.2f}")
        print(f"  Avg CLV %: {overall['avg_clv_pct']:+.2f}%")
        print(f"  Positive CLV Rate: {overall['positive_clv_rate']*100:.1f}%")

        # CLV by prop type
        clv_by_prop = defaultdict(list)
        for bet in closed_bets:
            clv_by_prop[bet['prop_type']].append(bet['clv'])

        prop_summary = {}
        for prop_type, clvs in clv_by_prop.items():
            prop_summary[prop_type] = {
                'count': len(clvs),
                'avg_clv': round(np.mean(clvs), 2),
                'positive_rate': round(len([c for c in clvs if c > 0]) / len(clvs), 3)
            }

        print(f"\nCLV by Prop Type:")
        for prop_type, metrics in sorted(prop_summary.items(), key=lambda x: x[1]['avg_clv'], reverse=True):
            print(f"  {prop_type:30s} | Avg CLV: {metrics['avg_clv']:+.2f} | Pos Rate: {metrics['positive_rate']*100:.0f}% | N={metrics['count']}")

        # Win rate vs CLV correlation
        if resulted_bets:
            win_rate_overall = len([b for b in resulted_bets if b['won']]) / len(resulted_bets)

            # Split by CLV
            positive_clv_bets = [b for b in resulted_bets if b['clv'] > 0]
            negative_clv_bets = [b for b in resulted_bets if b['clv'] <= 0]

            win_rate_positive_clv = len([b for b in positive_clv_bets if b['won']]) / len(positive_clv_bets) if positive_clv_bets else 0
            win_rate_negative_clv = len([b for b in negative_clv_bets if b['won']]) / len(negative_clv_bets) if negative_clv_bets else 0

            print(f"\nWin Rate Analysis:")
            print(f"  Overall Win Rate: {win_rate_overall*100:.1f}%")
            print(f"  Win Rate (Positive CLV): {win_rate_positive_clv*100:.1f}% (N={len(positive_clv_bets)})")
            print(f"  Win Rate (Negative CLV): {win_rate_negative_clv*100:.1f}% (N={len(negative_clv_bets)})")

            overall['win_rate_overall'] = round(win_rate_overall, 3)
            overall['win_rate_positive_clv'] = round(win_rate_positive_clv, 3)
            overall['win_rate_negative_clv'] = round(win_rate_negative_clv, 3)

        # Top CLV wins (biggest edges captured)
        top_clv = sorted(closed_bets, key=lambda b: b['clv'], reverse=True)[:10]

        print(f"\nTop 10 CLV Wins:")
        for i, bet in enumerate(top_clv, 1):
            print(f"  {i:2d}. {bet['player_name']:20s} {bet['prop_type']:20s} "
                  f"{bet['side'].upper():5s} | CLV: +{bet['clv']:.1f} ({bet['clv_pct']:+.1f}%)")

        # Worst CLV (moved against us)
        worst_clv = sorted(closed_bets, key=lambda b: b['clv'])[:10]

        print(f"\nWorst 10 CLV (Line moved against us):")
        for i, bet in enumerate(worst_clv, 1):
            print(f"  {i:2d}. {bet['player_name']:20s} {bet['prop_type']:20s} "
                  f"{bet['side'].upper():5s} | CLV: {bet['clv']:.1f} ({bet['clv_pct']:+.1f}%)")

        # Compile report
        report = {
            'overall': overall,
            'by_prop_type': prop_summary,
            'top_clv_wins': [
                {
                    'player': b['player_name'],
                    'prop_type': b['prop_type'],
                    'side': b['side'],
                    'clv': b['clv'],
                    'clv_pct': b['clv_pct']
                }
                for b in top_clv
            ],
            'worst_clv': [
                {
                    'player': b['player_name'],
                    'prop_type': b['prop_type'],
                    'side': b['side'],
                    'clv': b['clv'],
                    'clv_pct': b['clv_pct']
                }
                for b in worst_clv
            ]
        }

        # Save report if output file specified
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n✓ CLV report saved to: {output_file}")

        print(f"\n{'='*80}\n")

        return report


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='CLV Tracker - Measure Closing Line Value'
    )
    parser.add_argument('--action', required=True,
                       choices=['log', 'update_close', 'update_result', 'report'],
                       help='Action to perform')
    parser.add_argument('--storage-file', type=Path,
                       default=Path('outputs/betting/clv_bets.json'),
                       help='Storage file for bet records')
    parser.add_argument('--bet-id', type=str, help='Bet ID')
    parser.add_argument('--player', type=str, help='Player name')
    parser.add_argument('--prop-type', type=str, help='Prop type')
    parser.add_argument('--side', type=str, choices=['over', 'under'], help='Bet side')
    parser.add_argument('--opening-line', type=float, help='Opening line')
    parser.add_argument('--opening-odds', type=int, help='Opening odds')
    parser.add_argument('--closing-line', type=float, help='Closing line')
    parser.add_argument('--closing-odds', type=int, help='Closing odds')
    parser.add_argument('--projection', type=float, help='Model projection')
    parser.add_argument('--edge', type=float, help='Model edge')
    parser.add_argument('--game-id', type=str, help='Game ID')
    parser.add_argument('--actual-result', type=float, help='Actual result')
    parser.add_argument('--output-report', type=Path,
                       help='Output path for CLV report')

    args = parser.parse_args()

    tracker = CLVTracker(storage_file=args.storage_file)

    if args.action == 'log':
        # Log new bet
        bet = tracker.log_bet(
            bet_id=args.bet_id,
            player_name=args.player,
            prop_type=args.prop_type,
            side=args.side,
            opening_line=args.opening_line,
            opening_odds=args.opening_odds,
            model_projection=args.projection,
            model_edge=args.edge,
            game_id=args.game_id
        )
        print(f"✓ Logged bet: {bet['bet_id']}")

    elif args.action == 'update_close':
        # Update with closing line
        bet = tracker.update_closing_line(
            bet_id=args.bet_id,
            closing_line=args.closing_line,
            closing_odds=args.closing_odds
        )
        print(f"✓ Updated closing line: CLV = {bet['clv']:+.2f} ({bet['clv_pct']:+.1f}%)")

    elif args.action == 'update_result':
        # Update with actual result
        bet = tracker.update_result(
            bet_id=args.bet_id,
            actual_result=args.actual_result
        )
        print(f"✓ Updated result: {'WON' if bet['won'] else 'LOST'}")

    elif args.action == 'report':
        # Generate CLV report
        tracker.generate_clv_report(output_file=args.output_report)
