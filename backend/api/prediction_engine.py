"""Prediction Engine - Combines all signals with proper weighting for final predictions.

Signal Hierarchy (from strongest to weakest):
1. Contextual Performance (30%) - How player performs under specific conditions
2. EPA Matchup Edge (25%) - Offensive EPA vs Defensive EPA allowed
3. Success Rate Edge (15%) - Efficiency matchup
4. CPOE Edge (10%) - QB accuracy vs secondary quality (QB only)
5. Pressure Matchup (10%) - OL protection vs DL pass rush (QB only)
6. Trend Momentum (5%) - Recent performance trajectory
7. Game Script (5%) - Expected pace, score, volume

When signals align → High confidence
When signals conflict → Weight by reliability (contextual > EPA > others)
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import statistics


@dataclass
class PredictionSignals:
    """All signals that feed into final prediction."""
    # Base projection
    baseline_projection: float  # Player's season average

    # Contextual adjustment (strongest signal - 30% weight)
    contextual_adjustment: float = 0.0  # From performance splits
    contextual_confidence: float = 0.0  # Based on sample size

    # EPA matchup (25% weight)
    epa_edge: float = 0.0  # Offensive EPA - Defensive EPA allowed
    epa_confidence: float = 0.0

    # Success rate matchup (15% weight)
    success_rate_edge: float = 0.0  # Percentage points
    success_rate_confidence: float = 0.0

    # CPOE matchup (10% weight, QB only)
    cpoe_edge: float = 0.0  # Percentage points
    cpoe_confidence: float = 0.0

    # Pressure matchup (10% weight, QB only)
    pressure_edge: float = 0.0  # Percentage points (negative = OL advantage)
    pressure_confidence: float = 0.0

    # Trend momentum (5% weight)
    trend_adjustment: float = 0.0  # From recent EPA trend
    trend_confidence: float = 0.0

    # Game script (5% weight)
    game_script_adjustment: float = 0.0  # Expected volume adjustment
    game_script_confidence: float = 0.0


@dataclass
class Prediction:
    """Final prediction with confidence and breakdown."""
    player_id: str
    player_name: str
    position: str
    stat_type: str  # 'passing_yards', 'rushing_yards', 'receiving_yards'

    # Prediction
    baseline: float
    final_projection: float
    total_adjustment: float
    confidence: float  # 0.0 to 1.0

    # Signal breakdown
    signal_contributions: Dict[str, float]  # How much each signal contributed
    signal_alignments: str  # "All signals positive", "Mixed signals", etc.

    # Recommendation
    recommendation: str  # "STRONG OVER", "VALUE OVER", "NEUTRAL", "FADE UNDER", etc.
    line_comparison: Optional[float] = None  # Sportsbook line (if available)
    edge_vs_line: Optional[float] = None  # projection - line


class PredictionEngine:
    """Combines all signals to generate final predictions."""

    def __init__(self):
        # Signal weights (must sum to 1.0)
        self.weights = {
            'contextual': 0.30,
            'epa_matchup': 0.25,
            'success_rate': 0.15,
            'cpoe': 0.10,
            'pressure': 0.10,
            'trend': 0.05,
            'game_script': 0.05
        }

        # Conversion factors (how to convert each signal to yards)
        self.conversions = {
            'epa_to_yards': 200,  # 0.1 EPA ≈ 20 yards
            'success_rate_to_yards': 2,  # 1% success rate ≈ 2 yards
            'cpoe_to_yards': 3,  # 1% CPOE ≈ 3 yards
            'pressure_to_yards': -1.5,  # 1% more pressure ≈ -1.5 yards
        }

    def generate_prediction(
        self,
        signals: PredictionSignals,
        sportsbook_line: Optional[float] = None
    ) -> Prediction:
        """Generate final prediction from all signals.

        Args:
            signals: All input signals
            sportsbook_line: Optional sportsbook line for comparison

        Returns:
            Prediction object with final projection and breakdown
        """
        # Convert each signal to yards adjustment
        adjustments = {}

        # 1. Contextual (already in yards)
        adjustments['contextual'] = signals.contextual_adjustment * self.weights['contextual']

        # 2. EPA Matchup
        epa_yards = signals.epa_edge * self.conversions['epa_to_yards']
        adjustments['epa_matchup'] = epa_yards * self.weights['epa_matchup']

        # 3. Success Rate
        sr_yards = signals.success_rate_edge * self.conversions['success_rate_to_yards']
        adjustments['success_rate'] = sr_yards * self.weights['success_rate']

        # 4. CPOE (QB only)
        cpoe_yards = signals.cpoe_edge * self.conversions['cpoe_to_yards']
        adjustments['cpoe'] = cpoe_yards * self.weights['cpoe']

        # 5. Pressure (QB only, negative edge = good for QB)
        pressure_yards = signals.pressure_edge * self.conversions['pressure_to_yards']
        adjustments['pressure'] = pressure_yards * self.weights['pressure']

        # 6. Trend (already in yards)
        adjustments['trend'] = signals.trend_adjustment * self.weights['trend']

        # 7. Game Script (already in yards)
        adjustments['game_script'] = signals.game_script_adjustment * self.weights['game_script']

        # Calculate total adjustment
        total_adjustment = sum(adjustments.values())

        # Final projection
        final_projection = signals.baseline_projection + total_adjustment

        # Calculate overall confidence (weighted average of signal confidences)
        confidences = [
            signals.contextual_confidence * self.weights['contextual'],
            signals.epa_confidence * self.weights['epa_matchup'],
            signals.success_rate_confidence * self.weights['success_rate'],
            signals.cpoe_confidence * self.weights['cpoe'],
            signals.pressure_confidence * self.weights['pressure'],
            signals.trend_confidence * self.weights['trend'],
            signals.game_script_confidence * self.weights['game_script']
        ]
        overall_confidence = sum(confidences)

        # Check signal alignment
        positive_signals = sum(1 for adj in adjustments.values() if adj > 2)
        negative_signals = sum(1 for adj in adjustments.values() if adj < -2)
        neutral_signals = sum(1 for adj in adjustments.values() if abs(adj) <= 2)

        if positive_signals >= 5:
            alignment = "Strong consensus OVER"
            overall_confidence *= 1.1  # Boost confidence when signals align
        elif negative_signals >= 5:
            alignment = "Strong consensus UNDER"
            overall_confidence *= 1.1
        elif positive_signals > negative_signals:
            alignment = "Lean OVER (mixed signals)"
        elif negative_signals > positive_signals:
            alignment = "Lean UNDER (mixed signals)"
        else:
            alignment = "Neutral (conflicting signals)"
            overall_confidence *= 0.9  # Reduce confidence when signals conflict

        # Cap confidence at 0.95
        overall_confidence = min(overall_confidence, 0.95)

        # Generate recommendation
        if sportsbook_line:
            edge = final_projection - sportsbook_line

            if edge >= 15 and overall_confidence >= 0.75:
                recommendation = "STRONG OVER"
            elif edge >= 8 and overall_confidence >= 0.65:
                recommendation = "VALUE OVER"
            elif edge >= 3:
                recommendation = "SLIGHT OVER"
            elif edge <= -15 and overall_confidence >= 0.75:
                recommendation = "STRONG UNDER"
            elif edge <= -8 and overall_confidence >= 0.65:
                recommendation = "VALUE UNDER"
            elif edge <= -3:
                recommendation = "SLIGHT UNDER"
            else:
                recommendation = "NEUTRAL / PASS"
        else:
            recommendation = "No line available"

        return Prediction(
            player_id=signals.baseline_projection,  # Placeholder, pass actual player_id
            player_name="Player",  # Placeholder
            position="QB",  # Placeholder
            stat_type="passing_yards",  # Placeholder
            baseline=signals.baseline_projection,
            final_projection=round(final_projection, 1),
            total_adjustment=round(total_adjustment, 1),
            confidence=round(overall_confidence, 2),
            signal_contributions={k: round(v, 1) for k, v in adjustments.items()},
            signal_alignments=alignment,
            recommendation=recommendation,
            line_comparison=sportsbook_line,
            edge_vs_line=round(edge, 1) if sportsbook_line else None
        )

    def explain_prediction(self, prediction: Prediction) -> str:
        """Generate human-readable explanation of prediction.

        Args:
            prediction: Prediction object

        Returns:
            Multi-line explanation string
        """
        lines = []
        lines.append(f"{'='*60}")
        lines.append(f"Prediction: {prediction.player_name} {prediction.stat_type}")
        lines.append(f"{'='*60}")
        lines.append(f"")
        lines.append(f"Baseline (season avg): {prediction.baseline:.1f}")
        lines.append(f"Final Projection: {prediction.final_projection:.1f}")
        lines.append(f"Total Adjustment: {prediction.total_adjustment:+.1f}")
        lines.append(f"Confidence: {prediction.confidence:.0%}")
        lines.append(f"")
        lines.append(f"Signal Breakdown:")
        lines.append(f"-" * 60)

        for signal, contribution in sorted(
            prediction.signal_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ):
            if abs(contribution) >= 0.5:  # Only show meaningful contributions
                direction = "↑" if contribution > 0 else "↓"
                lines.append(f"  {signal:20s} {direction} {contribution:+6.1f} yards")

        lines.append(f"")
        lines.append(f"Signal Alignment: {prediction.signal_alignments}")
        lines.append(f"")

        if prediction.line_comparison:
            lines.append(f"Sportsbook Line: {prediction.line_comparison:.1f}")
            lines.append(f"Edge: {prediction.edge_vs_line:+.1f} yards")
            lines.append(f"")

        lines.append(f"Recommendation: {prediction.recommendation}")
        lines.append(f"{'='*60}")

        return "\n".join(lines)


# Example usage
if __name__ == '__main__':
    engine = PredictionEngine()

    # Example 1: Josh Allen vs Ravens (all signals negative)
    print("Example 1: Josh Allen @ Ravens (Elite Defense)")
    print("")

    signals_allen = PredictionSignals(
        baseline_projection=270.0,
        contextual_adjustment=-22.0,  # Poor vs high pressure
        contextual_confidence=0.80,  # Good sample size
        epa_edge=-0.08,  # Ravens strong pass defense
        epa_confidence=0.75,
        success_rate_edge=-4.0,  # Ravens limit success rate
        success_rate_confidence=0.70,
        cpoe_edge=-1.5,  # Ravens good secondary
        cpoe_confidence=0.65,
        pressure_edge=+6.0,  # Ravens generate 6% more pressure
        pressure_confidence=0.75,
        trend_adjustment=+5.0,  # Allen trending up recently
        trend_confidence=0.60,
        game_script_adjustment=+3.0,  # Slight volume boost
        game_script_confidence=0.50
    )

    prediction_allen = engine.generate_prediction(signals_allen, sportsbook_line=265.5)
    print(engine.explain_prediction(prediction_allen))

    print("\n\n")

    # Example 2: Patrick Mahomes vs weak defense (all signals positive)
    print("Example 2: Patrick Mahomes vs Cardinals (Weak Defense)")
    print("")

    signals_mahomes = PredictionSignals(
        baseline_projection=275.0,
        contextual_adjustment=+18.0,  # Excels vs weak defenses
        contextual_confidence=0.85,
        epa_edge=+0.12,  # Cardinals poor pass defense
        epa_confidence=0.80,
        success_rate_edge=+5.0,  # Cardinals allow high success rate
        success_rate_confidence=0.75,
        cpoe_edge=+2.5,  # Cardinals weak secondary
        cpoe_confidence=0.70,
        pressure_edge=-5.0,  # Cardinals weak pass rush
        pressure_confidence=0.80,
        trend_adjustment=+8.0,  # Mahomes hot recently
        trend_confidence=0.75,
        game_script_adjustment=+5.0,  # Expected shootout
        game_script_confidence=0.65
    )

    prediction_mahomes = engine.generate_prediction(signals_mahomes, sportsbook_line=285.5)
    print(engine.explain_prediction(prediction_mahomes))
