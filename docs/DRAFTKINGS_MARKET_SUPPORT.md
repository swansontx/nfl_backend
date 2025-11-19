# DraftKings Market Support - Training Data & Models

## Overview
This document maps all DraftKings prop markets to our training data requirements and model architecture. We build the prediction models, and lines come from external sources (manual entry, scraping, etc.).

## DraftKings Prop Markets

### 1. Passing Props

#### Pass Yards (PASS_YARDS)
**Market:** Player to throw OVER/UNDER X yards

**Training Data Needed:**
```python
# From NFLverse play-by-play
player_stats = {
    'player_id': '00-0034857',  # Josh Allen
    'player_name': 'J.Allen',
    'week': 10,
    'pass_yards': 189,
    'pass_attempts': 34,
    'completions': 22,
    'pass_tds': 1,
    'interceptions': 1,
}
```

**Features to Extract:**
- Player season average (through previous week)
- Player L3 game average
- Opponent pass defense EPA/play
- Opponent pass defense yards allowed/game
- Vegas game total
- Team implied total
- Is home game
- Weather (wind, dome)
- Rest days
- Division game

**Model:**
```python
GradientBoostingRegressor(
    loss='quantile',  # Quantile regression
    alpha=[0.10, 0.25, 0.50, 0.75, 0.90],  # Multiple quantiles
    n_estimators=200,
    max_depth=5
)
```

**Target Variable:** `pass_yards`

---

#### Pass TDs (PASS_TDS)
**Market:** Player to throw OVER/UNDER X touchdowns

**Training Data:** Same as Pass Yards

**Additional Features:**
- Red zone attempts/game
- TD rate (season)
- Opponent red zone defense

**Model:** Poisson regression or quantile regression

**Target Variable:** `pass_tds`

---

#### Pass Completions (COMPLETIONS)
**Market:** Player to complete OVER/UNDER X passes

**Training Data:** Same as Pass Yards

**Additional Features:**
- Completion percentage (season)
- Average target depth
- Pressure rate faced

**Model:** Quantile regression

**Target Variable:** `completions`

---

#### Pass Attempts (PASS_ATTEMPTS)
**Market:** Player to attempt OVER/UNDER X passes

**Training Data:** Same as Pass Yards

**Additional Features:**
- Team pass rate
- Game script (expected to be ahead/behind)
- Pace of play

**Model:** Quantile regression

**Target Variable:** `pass_attempts`

---

#### Interceptions (INTERCEPTIONS)
**Market:** Player to throw OVER/UNDER X INTs (usually 0.5)

**Training Data:** Same as Pass Yards

**Additional Features:**
- INT rate (season)
- Opponent INT rate
- Pressure rate

**Model:** Bernoulli (0 vs 1+) or Poisson

**Target Variable:** `interceptions`

---

### 2. Rushing Props

#### Rush Yards (RUSH_YARDS)
**Market:** Player to rush for OVER/UNDER X yards

**Training Data:**
```python
player_stats = {
    'player_id': '00-0036945',  # Derrick Henry
    'player_name': 'D.Henry',
    'week': 10,
    'rush_yards': 118,
    'rush_attempts': 24,
    'rush_tds': 2,
}
```

**Features:**
- Player season rush yards/game
- Player L3 game average
- Opponent run defense EPA/play
- Opponent rush yards allowed/game
- Vegas game total
- Expected game script (ahead = more rushing)
- Is high total game (>48 = less rushing)
- Carries per game
- Yards per carry

**Model:** Quantile regression

**Target Variable:** `rush_yards`

---

#### Rush Attempts (RUSH_ATTEMPTS)
**Market:** Player to rush OVER/UNDER X times

**Training Data:** Same as Rush Yards

**Features:**
- Team run rate
- Game script expectation
- Player snap share
- Is goal-line back

**Model:** Quantile regression

**Target Variable:** `rush_attempts`

---

#### Rush TDs (RUSH_TDS)
**Market:** Player to score OVER/UNDER X rushing TDs

**Training Data:** Same as Rush Yards

**Features:**
- Red zone carries/game
- Goal-line carry share
- Opponent goal-line defense

**Model:** Bernoulli or Poisson

**Target Variable:** `rush_tds`

---

### 3. Receiving Props

#### Receiving Yards (REC_YARDS)
**Market:** Player to have OVER/UNDER X receiving yards

**Training Data:**
```python
player_stats = {
    'player_id': '00-0036945',  # Amon-Ra St. Brown
    'player_name': 'A.St. Brown',
    'week': 10,
    'rec_yards': 135,
    'receptions': 11,
    'targets': 14,
    'rec_tds': 2,
}
```

**Features:**
- Player season rec yards/game
- Player L3 game average
- Target share
- Air yards share
- Opponent pass defense vs WRs
- QB rating when targeting player
- Slot vs outside rate
- Vegas game total

**Model:** Quantile regression

**Target Variable:** `rec_yards`

---

#### Receptions (RECEPTIONS)
**Market:** Player to have OVER/UNDER X receptions

**Training Data:** Same as Rec Yards

**Features:**
- Target share
- Receptions per target (catch rate)
- Team pass rate
- PPR scoring expectation

**Model:** Quantile regression

**Target Variable:** `receptions`

---

#### Receiving TDs (REC_TDS)
**Market:** Player to score OVER/UNDER X receiving TDs

**Training Data:** Same as Rec Yards

**Features:**
- Red zone target share
- End zone targets/game
- TD rate

**Model:** Bernoulli or Poisson

**Target Variable:** `rec_tds`

---

### 4. Anytime Touchdown (ANYTIME_TD)
**Market:** Player to score a TD (any type)

**Training Data:** All rushing + receiving TDs

**Features:**
- Total TDs (season)
- Red zone touches
- Goal-line role
- Opponent red zone defense

**Model:** Bernoulli classification

**Target Variable:** `scored_td` (binary: 0 or 1)

---

### 5. First Touchdown (FIRST_TD)
**Market:** Player to score the first TD of the game

**Training Data:** First TD scorer per game

**Features:**
- Team scoring first rate
- Player TD rate
- Opening drive usage

**Model:** Multinomial classification (all players)

**Target Variable:** `is_first_td_scorer` (binary)

---

### 6. Kicker Props

#### FG Made (FG_MADE)
**Market:** Kicker to make OVER/UNDER X field goals

**Training Data:**
```python
kicker_stats = {
    'player_name': 'J.Tucker',
    'week': 10,
    'fg_made': 3,
    'fg_attempts': 3,
    'fg_pct': 100.0,
    'long_fg': 52,
}
```

**Features:**
- Team offensive efficiency (stalls in red zone = more FGs)
- Kicker FG%
- Weather (wind, temp)
- Dome vs outdoors

**Model:** Poisson regression

**Target Variable:** `fg_made`

---

#### Extra Points (XP_MADE)
**Market:** Kicker to make OVER/UNDER X extra points

**Training Data:** Same as FG Made

**Features:**
- Team TDs/game
- Red zone efficiency

**Model:** Poisson regression

**Target Variable:** `xp_made`

---

### 7. Combo Props

#### Pass + Rush Yards (PASS_RUSH_YARDS)
**Market:** QB to have OVER/UNDER X combined pass + rush yards

**Training Data:** Sum of pass_yards + rush_yards

**Features:** Combine pass and rush features

**Model:** Quantile regression on combined total

**Target Variable:** `pass_yards + rush_yards`

---

#### Rec + Rush Yards (REC_RUSH_YARDS)
**Market:** RB to have OVER/UNDER X combined rec + rush yards

**Training Data:** Sum of rec_yards + rush_yards

**Features:** Combine receiving and rushing features

**Model:** Quantile regression on combined total

**Target Variable:** `rec_yards + rush_yards`

---

## Data Requirements Summary

### Required NFLverse Data
| Dataset | Purpose | Granularity |
|---------|---------|-------------|
| Play-by-play | All stats | Play-level |
| Player stats | Aggregated stats | Weekly |
| Rosters | Player positions | Weekly |
| Snap counts | Usage rates | Weekly |
| Depth charts | Starting roles | Weekly |

### Feature Categories
| Category | Examples | Source |
|----------|----------|--------|
| Player history | Season avg, L3 avg, trend | NFLverse |
| Opponent defense | EPA allowed, yards allowed | NFLverse |
| Market context | Spread, total, implied total | Games table |
| Game environment | Dome, wind, temp | Games table |
| Usage | Snap%, target%, carry% | Snap counts |
| Team context | Pace, pass rate, run rate | NFLverse |

## Model Architecture

### For Each Prop Type:
```python
class PropModel:
    def __init__(self, prop_type):
        self.prop_type = prop_type

        # Quantile models for distribution
        self.models = {
            'q10': GradientBoostingRegressor(loss='quantile', alpha=0.10),
            'q25': GradientBoostingRegressor(loss='quantile', alpha=0.25),
            'q50': GradientBoostingRegressor(loss='quantile', alpha=0.50),
            'q75': GradientBoostingRegressor(loss='quantile', alpha=0.75),
            'q90': GradientBoostingRegressor(loss='quantile', alpha=0.90),
        }

    def fit(self, X_train, y_train):
        """Train all quantile models."""
        for name, model in self.models.items():
            model.fit(X_train, y_train)

    def predict_distribution(self, X):
        """Predict full distribution."""
        return {
            name: model.predict(X)
            for name, model in self.models.items()
        }

    def calculate_prob_over(self, X, line):
        """Calculate P(X > line) from distribution."""
        dist = self.predict_distribution(X)

        # Interpolate from quantiles
        prob = interpolate_prob_from_quantiles(
            q10=dist['q10'],
            q25=dist['q25'],
            q50=dist['q50'],
            q75=dist['q75'],
            q90=dist['q90'],
            line=line
        )

        return prob
```

## Training Pipeline

### Step 1: Data Collection (Weeks 1-9)
```python
# Collect all player stats
player_stats = fetch_player_stats(weeks=range(1, 10), season=2025)

# For each prop type
for prop_type in ['pass_yards', 'rush_yards', 'rec_yards', ...]:
    train_data = extract_features_for_prop_type(
        player_stats=player_stats,
        prop_type=prop_type,
        weeks=range(1, 10)
    )

    save_training_data(train_data, f'prop_{prop_type}_weeks_1_9.csv')
```

### Step 2: Model Training
```python
for prop_type in PROP_TYPES:
    # Load training data
    train_df = pd.read_csv(f'prop_{prop_type}_weeks_1_9.csv')

    # Prepare features
    X = train_df[FEATURES]
    y = train_df[prop_type]

    # Train model
    model = PropModel(prop_type)
    model.fit(X, y)

    # Save model
    joblib.dump(model, f'models/prop_{prop_type}.pkl')
```

### Step 3: Prediction (Week 10)
```python
# User provides DK lines manually
dk_lines = [
    {'player': 'Josh Allen', 'prop_type': 'pass_yards', 'line': 265.5},
    {'player': 'Derrick Henry', 'prop_type': 'rush_yards', 'line': 85.5},
    # ...
]

# Generate features for Week 10
features_week10 = extract_features_for_week(
    week=10,
    historical_data_through_week=9
)

# Make predictions
for prop in dk_lines:
    model = load_model(f'models/prop_{prop["prop_type"]}.pkl')

    prob_over = model.calculate_prob_over(
        X=features_week10[prop['player']],
        line=prop['line']
    )

    # Calculate edge (assuming -110 odds = 52.4% implied)
    edge = prob_over - 0.524

    if edge > 0.05:  # 5% edge threshold
        print(f"BET: {prop['player']} {prop['prop_type']} OVER {prop['line']}")
        print(f"  Our prob: {prob_over:.1%} | Edge: {edge:+.1%}")
```

## Implementation Checklist

### Data Pipeline
- [ ] Fetch NFLverse play-by-play for 2025 (Weeks 1-11)
- [ ] Extract player stats per game
- [ ] Calculate rolling averages (season, L3)
- [ ] Extract opponent defensive stats
- [ ] Merge with game context (spread, total, weather)

### Feature Engineering
- [ ] Player historical features
- [ ] Opponent defensive features
- [ ] Market context features
- [ ] Usage/opportunity features
- [ ] Game script features

### Model Training
- [ ] Train pass_yards model
- [ ] Train rush_yards model
- [ ] Train rec_yards model
- [ ] Train TD models (pass, rush, rec)
- [ ] Train attempts/targets models
- [ ] Train anytime TD model

### Validation
- [ ] Backtest on Week 10
- [ ] Calculate hit rate by prop type
- [ ] Calculate ROI by prop type
- [ ] Calibration curves (are probabilities accurate?)
- [ ] Feature importance analysis

### Production
- [ ] API endpoint to accept DK lines
- [ ] Generate predictions with edge calculations
- [ ] Apply quality filters
- [ ] Output recommended bets

## Expected Files Structure
```
data/
  player_stats/
    2025_weeks_1_9.csv          # All player stats
  training/
    prop_pass_yards_train.csv   # Features + targets
    prop_rush_yards_train.csv
    prop_rec_yards_train.csv
    ...
models/
  prop_pass_yards.pkl            # Trained models
  prop_rush_yards.pkl
  prop_rec_yards.pkl
  ...
predictions/
  week_10_predictions.json       # Model outputs
  week_10_recommendations.json   # Filtered bets
```

## Next Implementation Steps

1. **Fetch NFLverse data for 2025 Weeks 1-11**
2. **Build feature extraction pipeline**
3. **Train models for top 5 prop types:**
   - pass_yards
   - rush_yards
   - rec_yards
   - pass_tds
   - rush_tds
4. **Validate on Week 10 with manual DK lines**
5. **Compare our predictions to actual results**
6. **Calculate real win rate and ROI**
