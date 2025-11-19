#!/usr/bin/env python3
"""
Train XGBoost models for prop predictions

This script trains separate models for each prop market:
- Receiving yards
- Rushing yards
- Passing yards
- Touchdowns
- Receptions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from backend.database.session import get_db
from backend.database.models import PlayerGameFeature
from backend.config.logging_config import get_logger

logger = get_logger(__name__)

print("=" * 80)
print("MODEL TRAINING")
print("=" * 80)
print()

# Check dependencies
try:
    import xgboost as xgb
    print("✅ xgboost is installed")
except ImportError:
    print("❌ xgboost not installed")
    print("Install it with: pip install xgboost")
    sys.exit(1)

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    print("✅ scikit-learn is installed")
except ImportError:
    print("❌ scikit-learn not installed")
    print("Install it with: pip install scikit-learn")
    sys.exit(1)

print()
print("📊 Loading features from database...")

# Load all features
with get_db() as session:
    features = session.query(PlayerGameFeature).all()

    data = []
    for f in features:
        data.append({
            'player_id': f.player_id,
            'game_id': f.game_id,
            'targets': f.targets or 0,
            'receptions': f.receptions or 0,
            'receiving_yards': f.receiving_yards or 0,
            'receiving_tds': f.receiving_tds or 0,
            'rush_attempts': f.rush_attempts or 0,
            'rushing_yards': f.rushing_yards or 0,
            'rushing_tds': f.rushing_tds or 0,
            'pass_attempts': f.pass_attempts or 0,
            'passing_yards': f.passing_yards or 0,
            'passing_tds': f.passing_tds or 0,
            'snaps': f.snaps or 0,
        })

df = pd.DataFrame(data)
print(f"   Loaded {len(df):,} feature rows")

if len(df) < 100:
    print()
    print("⚠️  Not enough data to train models reliably")
    print(f"   Found: {len(df)} samples")
    print("   Recommended: 1000+ samples")
    print()
    print("Options:")
    print("  1. Load more historical seasons with load_nfl_data.py")
    print("  2. Generate more features with generate_features.py")
    print("  3. Use the quick_setup_for_testing.py for demo purposes")
    print()
    sys.exit(1)

print()
print("🏋️  Training models...")
print()

# Create models directory
models_dir = Path(__file__).parent.parent / 'models'
models_dir.mkdir(exist_ok=True)

# Train receiving yards model
print("1️⃣ Training receiving yards model...")
rec_df = df[df['targets'] > 0].copy()
if len(rec_df) > 50:
    X = rec_df[['targets', 'receptions', 'snaps']].fillna(0)
    y = rec_df['receiving_yards']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    model.save_model(str(models_dir / 'receiving_yards.json'))
    print(f"   ✅ MAE: {mae:.2f}, RMSE: {rmse:.2f}")
else:
    print(f"   ⚠️  Insufficient data ({len(rec_df)} samples)")

# Train rushing yards model
print("2️⃣ Training rushing yards model...")
rush_df = df[df['rush_attempts'] > 0].copy()
if len(rush_df) > 50:
    X = rush_df[['rush_attempts', 'snaps']].fillna(0)
    y = rush_df['rushing_yards']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    model.save_model(str(models_dir / 'rushing_yards.json'))
    print(f"   ✅ MAE: {mae:.2f}, RMSE: {rmse:.2f}")
else:
    print(f"   ⚠️  Insufficient data ({len(rush_df)} samples)")

# Train passing yards model
print("3️⃣ Training passing yards model...")
pass_df = df[df['pass_attempts'] > 0].copy()
if len(pass_df) > 50:
    X = pass_df[['pass_attempts', 'snaps']].fillna(0)
    y = pass_df['passing_yards']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    model.save_model(str(models_dir / 'passing_yards.json'))
    print(f"   ✅ MAE: {mae:.2f}, RMSE: {rmse:.2f}")
else:
    print(f"   ⚠️  Insufficient data ({len(pass_df)} samples)")

print()
print("=" * 80)
print("✅ Model Training Complete!")
print("=" * 80)
print()
print(f"Models saved to: {models_dir}")
print()
print("Next step:")
print("  Test recommendations: python scripts/demo_analysis.py")
print()
