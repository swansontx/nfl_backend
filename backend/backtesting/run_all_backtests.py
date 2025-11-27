"""Master Backtesting Orchestrator.

Runs all backtests, compares results, and generates comprehensive reports.
Validates all deep analysis features against historical data.
"""

from pathlib import Path
from typing import List, Dict
from datetime import datetime
import json

from backend.backtesting.framework import BacktestingFramework, BacktestResult
from backend.backtesting.injury_impact_backtest import InjuryImpactBacktester
from backend.backtesting.defense_matchup_backtest import DefenseMatchupBacktester
from backend.backtesting.weather_impact_backtest import WeatherImpactBacktester
from backend.backtesting.situational_factors_backtest import SituationalFactorsBacktester
from backend.backtesting.overall_accuracy_backtest import OverallAccuracyBacktester
from backend.backtesting.data_collector import HistoricalDataCollector


class BacktestingOrchestrator:
    """Orchestrates all backtesting activities."""

    def __init__(self, seasons: List[int] = None):
        """Initialize orchestrator.

        Args:
            seasons: Seasons to test (defaults to [2020, 2021, 2022, 2023, 2024])
        """
        self.seasons = seasons or [2020, 2021, 2022, 2023, 2024]
        self.framework = BacktestingFramework(seasons=self.seasons)
        self.data_collector = HistoricalDataCollector()

        # Results
        self.results: Dict[str, BacktestResult] = {}

        # Output
        self.output_dir = Path('outputs/backtesting')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def verify_data_availability(self) -> bool:
        """Verify that historical data is available.

        Returns:
            True if all required data is available
        """
        print("Verifying data availability...")

        availability = self.data_collector.verify_data_availability(self.seasons)

        all_available = True
        for season, avail in availability.items():
            has_required = avail['games'] and avail['player_stats']
            status = "✓" if has_required else "✗"
            print(f"  {season}: {status} (Games: {avail['games_count']}, Stats: {avail['stats_count']}, Injuries: {avail['injuries_count']})")

            if not has_required:
                all_available = False

        if not all_available:
            print("\n⚠️  Missing required data!")
            print("Run data_collector.py to fetch historical data")
            print("  Example: python -m backend.backtesting.data_collector")

        return all_available

    def run_all_backtests(self, skip_data_check: bool = False) -> Dict[str, BacktestResult]:
        """Run all backtests.

        Args:
            skip_data_check: Skip data availability check (for testing)

        Returns:
            Dictionary of BacktestResult objects
        """
        print("\n" + "=" * 80)
        print("HISTORICAL BACKTESTING - NFL BACKEND DEEP ANALYSIS")
        print("=" * 80)
        print(f"Seasons: {self.seasons}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80 + "\n")

        # Verify data
        if not skip_data_check:
            if not self.verify_data_availability():
                return {}

        # Run each backtest
        results = {}

        print("\n" + "-" * 80)
        print("1. INJURY IMPACT REDISTRIBUTION")
        print("-" * 80)
        try:
            injury_backtester = InjuryImpactBacktester(self.framework)
            results['injury_impact'] = injury_backtester.run_backtest()
            print("✓ Injury impact backtest complete")
        except Exception as e:
            print(f"✗ Injury impact backtest failed: {e}")
            results['injury_impact'] = BacktestResult(
                feature_name="Injury Impact",
                seasons_tested=self.seasons,
                sample_size=0,
                notes=[f"Error: {str(e)}"]
            )

        print("\n" + "-" * 80)
        print("2. DEFENSE MATCHUP ADJUSTMENTS")
        print("-" * 80)
        try:
            defense_backtester = DefenseMatchupBacktester(self.framework)
            results['defense_matchup'] = defense_backtester.run_backtest()
            print("✓ Defense matchup backtest complete")
        except Exception as e:
            print(f"✗ Defense matchup backtest failed: {e}")
            results['defense_matchup'] = BacktestResult(
                feature_name="Defense Matchup",
                seasons_tested=self.seasons,
                sample_size=0,
                notes=[f"Error: {str(e)}"]
            )

        print("\n" + "-" * 80)
        print("3. WEATHER IMPACT COEFFICIENTS")
        print("-" * 80)
        try:
            weather_backtester = WeatherImpactBacktester(self.framework)
            results['weather_impact'] = weather_backtester.run_backtest()
            print("✓ Weather impact backtest complete")
        except Exception as e:
            print(f"✗ Weather impact backtest failed: {e}")
            results['weather_impact'] = BacktestResult(
                feature_name="Weather Impact",
                seasons_tested=self.seasons,
                sample_size=0,
                notes=[f"Error: {str(e)}"]
            )

        print("\n" + "-" * 80)
        print("4. SITUATIONAL FACTORS ADJUSTMENTS")
        print("-" * 80)
        try:
            situational_backtester = SituationalFactorsBacktester(self.framework)
            results['situational_factors'] = situational_backtester.run_backtest()
            print("✓ Situational factors backtest complete")
        except Exception as e:
            print(f"✗ Situational factors backtest failed: {e}")
            results['situational_factors'] = BacktestResult(
                feature_name="Situational Factors",
                seasons_tested=self.seasons,
                sample_size=0,
                notes=[f"Error: {str(e)}"]
            )

        print("\n" + "-" * 80)
        print("5. OVERALL PREDICTION ACCURACY")
        print("-" * 80)
        try:
            accuracy_backtester = OverallAccuracyBacktester(self.framework)
            results['overall_accuracy'] = accuracy_backtester.run_backtest()
            print("✓ Overall accuracy backtest complete")
        except Exception as e:
            print(f"✗ Overall accuracy backtest failed: {e}")
            results['overall_accuracy'] = BacktestResult(
                feature_name="Overall Accuracy",
                seasons_tested=self.seasons,
                sample_size=0,
                notes=[f"Error: {str(e)}"]
            )

        self.results = results
        return results

    def generate_master_report(self) -> str:
        """Generate comprehensive backtesting report.

        Returns:
            Markdown-formatted report
        """
        report = ["# Historical Backtesting Report\n"]
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Seasons Analyzed:** {', '.join(map(str, self.seasons))}\n")
        report.append("\n---\n")

        # Executive Summary
        report.append("\n## Executive Summary\n")

        total_sample = sum(r.sample_size for r in self.results.values())
        updates_recommended = sum(1 for r in self.results.values() if r.should_update)

        report.append(f"- **Total Observations:** {total_sample:,}")
        report.append(f"- **Features Tested:** {len(self.results)}")
        report.append(f"- **Updates Recommended:** {updates_recommended}/{len(self.results)}")

        avg_improvement = sum(r.improvement_pct for r in self.results.values() if r.should_update) / max(1, updates_recommended)
        report.append(f"- **Average Improvement:** {avg_improvement:.1f}%\n")

        # Summary Table
        report.append("\n## Results Summary\n")
        report.append("| Feature | Sample Size | RMSE | MAE | R² | Improvement | Update? |")
        report.append("|---------|------------:|-----:|----:|---:|------------:|:-------:|")

        for feature_name, result in self.results.items():
            update_icon = "✅" if result.should_update else "❌"
            report.append(
                f"| {result.feature_name} | {result.sample_size:,} | "
                f"{result.rmse:.2f} | {result.mae:.2f} | {result.r_squared:.3f} | "
                f"{result.improvement_pct:+.1f}% | {update_icon} |"
            )

        # Detailed Results
        report.append("\n---\n")
        report.append("\n## Detailed Results\n")

        for feature_name, result in self.results.items():
            report.append(f"\n### {result.feature_name}\n")

            if result.sample_size > 0:
                report.append(f"**Sample Size:** {result.sample_size:,} observations\n")

                if result.rmse > 0:
                    report.append("\n**Accuracy Metrics:**")
                    report.append(f"- RMSE: {result.rmse:.2f}")
                    report.append(f"- MAE: {result.mae:.2f}")
                    report.append(f"- Correlation: {result.correlation:.3f}")
                    report.append(f"- R²: {result.r_squared:.3f}\n")

                if result.should_update:
                    report.append(f"\n**✅ RECOMMENDATION: Update factors ({result.improvement_pct:+.1f}% improvement)**\n")
                else:
                    report.append(f"\n**❌ Current factors are adequate**\n")

                if result.notes:
                    report.append("\n**Findings:**")
                    for note in result.notes:
                        report.append(f"{note}")

                if result.calculated_factors:
                    report.append("\n<details>")
                    report.append("<summary><b>Calculated Factors (Click to Expand)</b></summary>\n")
                    report.append("```json")
                    report.append(json.dumps(result.calculated_factors, indent=2))
                    report.append("```")
                    report.append("</details>\n")
            else:
                report.append("⚠️ Insufficient data for analysis\n")
                if result.notes:
                    for note in result.notes:
                        report.append(f"- {note}")

        # Recommendations
        report.append("\n---\n")
        report.append("\n## Implementation Recommendations\n")

        features_to_update = [r for r in self.results.values() if r.should_update]

        if features_to_update:
            report.append("\n### Priority Updates\n")

            sorted_features = sorted(features_to_update, key=lambda x: x.improvement_pct, reverse=True)

            for i, result in enumerate(sorted_features, 1):
                report.append(f"\n**{i}. {result.feature_name}** ({result.improvement_pct:+.1f}% improvement)")
                report.append(f"- Update configuration files with calculated factors")
                report.append(f"- Expected accuracy improvement: {result.improvement_pct:.1f}%")
                report.append(f"- Sample size: {result.sample_size:,} observations\n")

        else:
            report.append("\nNo updates recommended at this time. Current factors are performing well.\n")

        # Next Steps
        report.append("\n## Next Steps\n")
        report.append("1. Review calculated factors for each feature")
        report.append("2. Update configuration files with data-driven coefficients")
        report.append("3. Re-run validation tests to confirm improvements")
        report.append("4. Monitor performance in production")
        report.append("5. Schedule quarterly backtesting to refine factors\n")

        # Appendix
        report.append("\n---\n")
        report.append("\n## Appendix: Methodology\n")
        report.append("\n### Data Sources")
        report.append("- Historical game data: NFL official stats")
        report.append("- Player statistics: nfl-data-py package")
        report.append("- Injury reports: NFL injury reports")
        report.append("- Weather data: Historical weather APIs\n")

        report.append("\n### Validation Metrics")
        report.append("- **RMSE (Root Mean Square Error):** Lower is better")
        report.append("- **MAE (Mean Absolute Error):** Lower is better")
        report.append("- **Correlation:** Higher is better (closer to 1.0)")
        report.append("- **R² (Coefficient of Determination):** Higher is better (closer to 1.0)\n")

        report.append("\n### Statistical Significance")
        report.append("- Minimum sample size: 30 observations")
        report.append("- Confidence level: 95%")
        report.append("- Improvement threshold: 5% reduction in prediction error\n")

        return "\n".join(report)

    def save_results(self):
        """Save all backtest results and reports."""
        print("\nSaving results...")

        # Save individual results
        for feature_name, result in self.results.items():
            self.framework.save_results(result, f"{feature_name}_backtest.json")

        # Save master report
        report = self.generate_master_report()
        report_file = self.output_dir / 'BACKTESTING_REPORT.md'
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"  ✓ Master report: {report_file}")

        # Save summary JSON
        summary = {
            'generated_at': datetime.now().isoformat(),
            'seasons_tested': self.seasons,
            'features_tested': len(self.results),
            'total_observations': sum(r.sample_size for r in self.results.values()),
            'updates_recommended': sum(1 for r in self.results.values() if r.should_update),
            'results': {
                name: {
                    'sample_size': result.sample_size,
                    'rmse': result.rmse,
                    'mae': result.mae,
                    'improvement_pct': result.improvement_pct,
                    'should_update': result.should_update
                } for name, result in self.results.items()
            }
        }

        summary_file = self.output_dir / 'backtest_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  ✓ Summary JSON: {summary_file}")

    def run(self, skip_data_check: bool = False):
        """Run full backtesting workflow.

        Args:
            skip_data_check: Skip data availability check (for testing)
        """
        # Run all backtests
        results = self.run_all_backtests(skip_data_check=skip_data_check)

        if not results:
            print("\n✗ Backtesting failed - no results generated")
            return

        # Save results
        self.save_results()

        # Print summary
        print("\n" + "=" * 80)
        print("BACKTESTING COMPLETE")
        print("=" * 80)

        total_observations = sum(r.sample_size for r in results.values())
        updates_recommended = sum(1 for r in results.values() if r.should_update)

        print(f"\nTotal observations: {total_observations:,}")
        print(f"Features tested: {len(results)}")
        print(f"Updates recommended: {updates_recommended}/{len(results)}")

        if updates_recommended > 0:
            avg_improvement = sum(r.improvement_pct for r in results.values() if r.should_update) / updates_recommended
            print(f"Average improvement: {avg_improvement:.1f}%")

        print(f"\nReports saved to: {self.output_dir}")
        print("\nNext steps:")
        print("1. Review BACKTESTING_REPORT.md")
        print("2. Update system configurations with calculated factors")
        print("3. Validate improvements in production\n")


if __name__ == "__main__":
    # Run backtesting on 2020-2024 seasons
    orchestrator = BacktestingOrchestrator(seasons=[2020, 2021, 2022, 2023, 2024])
    orchestrator.run(skip_data_check=False)
