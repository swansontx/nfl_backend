"""
Train snap count and usage-based markets.
Uses enhanced player stats with snap count data.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
import joblib
from pathlib import Path

print("="*80)
print("TRAINING SNAP COUNT / USAGE MARKETS")
print("="*80 + "\n")

# Load enhanced data
inputs_dir = Path('/home/user/nfl_backend/inputs')
outputs_dir = Path('/home/user/nfl_backend/outputs/models/usage')
outputs_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(inputs_dir / 'player_stats_enhanced_2025.csv', low_memory=False)
print(f"Loaded {len(df)} records")

# Filter to players with snap data
df_snaps = df[df['offense_pct'].notna()].copy()
print(f"Records with snap data: {len(df_snaps)}")

# Train/test split by week
train = df_snaps[df_snaps['week'] <= 9].copy()
test = df_snaps[df_snaps['week'] > 9].copy()
print(f"Training: {len(train)} records (weeks 1-9)")
print(f"Testing: {len(test)} records (weeks 10-11)")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

# Base features
base_features = [
    'games_played',
    'is_home',
    'spread_line',
    'total_line',
]

# Rolling averages for snap usage
snap_features = [
    'offense_pct_season_avg',
    'offense_pct_l3_avg',
    'offense_snaps_season_avg',
    'offense_snaps_l3_avg',
]

# Position-specific features
stat_features = [
    'completions_season_avg',
    'attempts_season_avg',
    'passing_yards_season_avg',
    'rushing_yards_season_avg',
    'receptions_season_avg',
    'targets_season_avg',
]

all_features = base_features + snap_features + stat_features

# Drop rows with missing features
train_clean = train.dropna(subset=all_features + ['offense_pct'])
test_clean = test.dropna(subset=all_features + ['offense_pct'])
print(f"Training (clean): {len(train_clean)} records")
print(f"Testing (clean): {len(test_clean)} records")

# ============================================================================
# MARKET 1: Offensive Snap Percentage O/U
# ============================================================================
print("\n" + "-"*80)
print("MARKET 1: Offensive Snap Percentage O/U")
print("-"*80)

X_train = train_clean[all_features].fillna(0)
y_train = train_clean['offense_pct']
X_test = test_clean[all_features].fillna(0)
y_test = test_clean['offense_pct']

model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mae = np.mean(np.abs(preds - y_test))
rmse = np.sqrt(np.mean((preds - y_test)**2))

print(f"MAE: {mae:.2f}%")
print(f"RMSE: {rmse:.2f}%")

# Save
joblib.dump(model, outputs_dir / 'offense_snap_pct_model.pkl')
print("Saved: offense_snap_pct_model.pkl")

# ============================================================================
# MARKET 2: Offensive Snaps O/U
# ============================================================================
print("\n" + "-"*80)
print("MARKET 2: Offensive Snaps O/U")
print("-"*80)

y_train = train_clean['offense_snaps']
y_test = test_clean['offense_snaps']

model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = np.mean(np.abs(preds - y_test))
print(f"MAE: {mae:.2f} snaps")

joblib.dump(model, outputs_dir / 'offense_snaps_model.pkl')
print("Saved: offense_snaps_model.pkl")

# ============================================================================
# MARKET 3: Starter Prediction (>50% snaps)
# ============================================================================
print("\n" + "-"*80)
print("MARKET 3: Starter Prediction (>50% snaps)")
print("-"*80)

train_clean['is_starter'] = (train_clean['offense_pct'] >= 50).astype(int)
test_clean['is_starter'] = (test_clean['offense_pct'] >= 50).astype(int)

y_train = train_clean['is_starter']
y_test = test_clean['is_starter']

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

preds = model.predict(X_test)
accuracy = (preds == y_test).mean()
print(f"Accuracy: {accuracy:.1%}")

joblib.dump(model, outputs_dir / 'starter_prediction_model.pkl')
print("Saved: starter_prediction_model.pkl")

# ============================================================================
# MARKET 4: Target Share % (for WR/TE/RB)
# ============================================================================
print("\n" + "-"*80)
print("MARKET 4: Target Share % O/U")
print("-"*80)

# Calculate target share (targets / team total targets)
# For simplicity, use targets as proxy
skill_positions = df_snaps[df_snaps['position_group'].isin(['WR', 'TE', 'RB'])].copy()

train_skill = skill_positions[skill_positions['week'] <= 9].copy()
test_skill = skill_positions[skill_positions['week'] > 9].copy()

skill_features = base_features + snap_features + ['targets_season_avg', 'targets_l3_avg',
                                                   'receptions_season_avg', 'receptions_l3_avg']

train_skill_clean = train_skill.dropna(subset=skill_features + ['targets'])
test_skill_clean = test_skill.dropna(subset=skill_features + ['targets'])

if len(train_skill_clean) > 0 and len(test_skill_clean) > 0:
    X_train = train_skill_clean[skill_features].fillna(0)
    y_train = train_skill_clean['targets']
    X_test = test_skill_clean[skill_features].fillna(0)
    y_test = test_skill_clean['targets']

    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = np.mean(np.abs(preds - y_test))
    print(f"MAE: {mae:.2f} targets")

    joblib.dump(model, outputs_dir / 'targets_model.pkl')
    print("Saved: targets_model.pkl")
else:
    print("Insufficient data for target share model")

# ============================================================================
# MARKET 5: Red Zone Opportunities
# ============================================================================
print("\n" + "-"*80)
print("MARKET 5: Red Zone Opportunities O/U")
print("-"*80)

# Use TD opportunities as proxy for red zone
if 'receiving_tds' in df_snaps.columns and 'rushing_tds' in df_snaps.columns:
    df_snaps['total_tds'] = df_snaps['receiving_tds'].fillna(0) + df_snaps['rushing_tds'].fillna(0)

    train_rz = df_snaps[df_snaps['week'] <= 9].copy()
    test_rz = df_snaps[df_snaps['week'] > 9].copy()

    rz_features = base_features + snap_features + ['targets_season_avg', 'carries_season_avg']
    train_rz_clean = train_rz.dropna(subset=[f for f in rz_features if f in train_rz.columns])
    test_rz_clean = test_rz.dropna(subset=[f for f in rz_features if f in test_rz.columns])

    if len(train_rz_clean) > 0 and len(test_rz_clean) > 0:
        available_features = [f for f in rz_features if f in train_rz_clean.columns]
        X_train = train_rz_clean[available_features].fillna(0)
        y_train = train_rz_clean['total_tds']
        X_test = test_rz_clean[available_features].fillna(0)
        y_test = test_rz_clean['total_tds']

        model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = np.mean(np.abs(preds - y_test))
        print(f"MAE: {mae:.2f} TDs")

        joblib.dump(model, outputs_dir / 'red_zone_tds_model.pkl')
        print("Saved: red_zone_tds_model.pkl")

# ============================================================================
# MARKET 6: Special Teams Snaps
# ============================================================================
print("\n" + "-"*80)
print("MARKET 6: Special Teams Snap % O/U")
print("-"*80)

st_data = df_snaps[df_snaps['st_pct'].notna() & (df_snaps['st_pct'] > 0)].copy()
train_st = st_data[st_data['week'] <= 9].copy()
test_st = st_data[st_data['week'] > 9].copy()

if len(train_st) > 50 and len(test_st) > 10:
    st_features = ['games_played', 'is_home', 'offense_pct_season_avg']
    st_features = [f for f in st_features if f in train_st.columns]

    train_st_clean = train_st.dropna(subset=st_features + ['st_pct'])
    test_st_clean = test_st.dropna(subset=st_features + ['st_pct'])

    X_train = train_st_clean[st_features].fillna(0)
    y_train = train_st_clean['st_pct']
    X_test = test_st_clean[st_features].fillna(0)
    y_test = test_st_clean['st_pct']

    model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = np.mean(np.abs(preds - y_test))
    print(f"MAE: {mae:.2f}%")

    joblib.dump(model, outputs_dir / 'st_snap_pct_model.pkl')
    print("Saved: st_snap_pct_model.pkl")
else:
    print("Insufficient special teams data")

# ============================================================================
# MARKET 7: Touches per Game O/U
# ============================================================================
print("\n" + "-"*80)
print("MARKET 7: Touches per Game O/U")
print("-"*80)

# Touches = carries + receptions
df_snaps['touches'] = df_snaps['carries'].fillna(0) + df_snaps['receptions'].fillna(0)

train_touches = df_snaps[df_snaps['week'] <= 9].copy()
test_touches = df_snaps[df_snaps['week'] > 9].copy()

touch_features = base_features + snap_features + ['carries_season_avg', 'receptions_season_avg']
touch_features = [f for f in touch_features if f in train_touches.columns]

train_touch_clean = train_touches.dropna(subset=touch_features + ['touches'])
test_touch_clean = test_touches.dropna(subset=touch_features + ['touches'])

if len(train_touch_clean) > 0 and len(test_touch_clean) > 0:
    X_train = train_touch_clean[touch_features].fillna(0)
    y_train = train_touch_clean['touches']
    X_test = test_touch_clean[touch_features].fillna(0)
    y_test = test_touch_clean['touches']

    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = np.mean(np.abs(preds - y_test))
    print(f"MAE: {mae:.2f} touches")

    joblib.dump(model, outputs_dir / 'touches_model.pkl')
    print("Saved: touches_model.pkl")

# ============================================================================
# MARKET 8: High Usage Game (>75% snaps)
# ============================================================================
print("\n" + "-"*80)
print("MARKET 8: High Usage Game (>75% snaps)")
print("-"*80)

train_clean['high_usage'] = (train_clean['offense_pct'] >= 75).astype(int)
test_clean['high_usage'] = (test_clean['offense_pct'] >= 75).astype(int)

X_train = train_clean[all_features].fillna(0)
y_train = train_clean['high_usage']
X_test = test_clean[all_features].fillna(0)
y_test = test_clean['high_usage']

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

preds = model.predict(X_test)
accuracy = (preds == y_test).mean()
print(f"Accuracy: {accuracy:.1%}")

joblib.dump(model, outputs_dir / 'high_usage_model.pkl')
print("Saved: high_usage_model.pkl")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SNAP/USAGE MARKETS TRAINING COMPLETE")
print("="*80)

models_saved = list(outputs_dir.glob('*.pkl'))
print(f"\nModels saved: {len(models_saved)}")
for model_file in sorted(models_saved):
    print(f"  - {model_file.name}")
