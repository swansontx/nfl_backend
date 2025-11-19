#!/usr/bin/env python3
"""
Update NFL data with latest games and stats

Run this weekly (or daily during season) to keep data fresh.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import subprocess
from datetime import datetime

print("=" * 80)
print("NFL DATA UPDATE")
print(f"Started: {datetime.now()}")
print("=" * 80)
print()

# Step 1: Load latest NFL data
print("1️⃣ Loading latest NFL schedules, rosters, and game results...")
result = subprocess.run(
    ["python", "scripts/load_nfl_data.py"],
    capture_output=False
)

if result.returncode != 0:
    print("❌ Failed to load NFL data")
    sys.exit(1)

print()

# Step 2: Generate features from new games
print("2️⃣ Generating player features from play-by-play data...")
result = subprocess.run(
    ["python", "scripts/generate_features.py"],
    capture_output=False
)

if result.returncode != 0:
    print("❌ Failed to generate features")
    sys.exit(1)

print()

# Step 3: Retrain models with new data
print("3️⃣ Retraining models with updated data...")
result = subprocess.run(
    ["python", "scripts/train_models.py"],
    capture_output=False
)

if result.returncode != 0:
    print("❌ Failed to train models")
    sys.exit(1)

print()
print("=" * 80)
print("✅ DATA UPDATE COMPLETE")
print(f"Finished: {datetime.now()}")
print("=" * 80)
print()
print("Your API now has the latest NFL data!")
print("Frontend will automatically use updated recommendations.")
