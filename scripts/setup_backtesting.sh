#!/bin/bash

# Setup script for Historical Backtesting System

echo "=================================="
echo "Historical Backtesting Setup"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python version: $python_version"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install nfl-data-py pandas numpy scipy --quiet

if [ $? -eq 0 ]; then
    echo "  ✓ Dependencies installed"
else
    echo "  ✗ Failed to install dependencies"
    exit 1
fi

# Create required directories
echo ""
echo "Creating directories..."
mkdir -p inputs/historical/injuries
mkdir -p outputs/backtesting
mkdir -p .cache/weather
echo "  ✓ Directories created"

# Verify backtesting modules
echo ""
echo "Verifying backtesting modules..."

modules=(
    "backend/backtesting/framework.py"
    "backend/backtesting/data_collector.py"
    "backend/backtesting/injury_impact_backtest.py"
    "backend/backtesting/defense_matchup_backtest.py"
    "backend/backtesting/weather_impact_backtest.py"
    "backend/backtesting/situational_factors_backtest.py"
    "backend/backtesting/overall_accuracy_backtest.py"
    "backend/backtesting/run_all_backtests.py"
)

all_exist=true
for module in "${modules[@]}"; do
    if [ -f "$module" ]; then
        echo "  ✓ $module"
    else
        echo "  ✗ $module (missing)"
        all_exist=false
    fi
done

if [ "$all_exist" = false ]; then
    echo ""
    echo "✗ Some modules are missing!"
    exit 1
fi

# Compile Python modules to check for syntax errors
echo ""
echo "Compiling Python modules..."
python3 -m py_compile backend/backtesting/framework.py
python3 -m py_compile backend/backtesting/data_collector.py
python3 -m py_compile backend/backtesting/injury_impact_backtest.py
python3 -m py_compile backend/backtesting/defense_matchup_backtest.py
python3 -m py_compile backend/backtesting/weather_impact_backtest.py
python3 -m py_compile backend/backtesting/situational_factors_backtest.py
python3 -m py_compile backend/backtesting/overall_accuracy_backtest.py
python3 -m py_compile backend/backtesting/run_all_backtests.py

if [ $? -eq 0 ]; then
    echo "  ✓ All modules compiled successfully"
else
    echo "  ✗ Compilation errors found"
    exit 1
fi

# Print next steps
echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next Steps:"
echo ""
echo "1. Collect historical data:"
echo "   python -m backend.backtesting.data_collector"
echo ""
echo "2. Run backtests:"
echo "   python -m backend.backtesting.run_all_backtests"
echo ""
echo "3. Review results:"
echo "   cat outputs/backtesting/BACKTESTING_REPORT.md"
echo ""
echo "See BACKTESTING_SYSTEM.md for full documentation"
echo ""
