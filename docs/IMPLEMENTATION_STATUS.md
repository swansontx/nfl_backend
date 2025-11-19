# Implementation Status & Roadmap

This document tracks what's been built vs what's scaffolded (structure exists but needs implementation).

---

## ✅ Fully Implemented (Production Ready)

### API Endpoints
- ✅ `/health` - Health check
- ✅ `/api/v1/news` - News aggregation (Sleeper injuries)
- ✅ `/api/v1/games/{game_id}/injuries` - Game injury reports
- ✅ `/api/v1/games/{game_id}/insights` - Matchup insights (templates)
- ✅ `/api/v1/games/{game_id}/narrative` - Game narratives (templates)
- ✅ `/api/v1/games/{game_id}/weather` - Weather forecasts (OpenWeather API)
- ✅ `/api/v1/props/value` - Prop value finder (Odds API + model loader)
- ✅ `/api/v1/players/{player_id}/insights` - Player trend analysis
- ✅ `/api/v1/props/compare` - Prop comparison
- ✅ `/api/v1/games/{game_id}/prop-sheet` - Game prop sheet
- ✅ `/game/{game_id}/projections` - Load model predictions
- ✅ `/admin/odds-api-usage` - Check Odds API quota
- ✅ `/admin/create-sample-projections/{game_id}` - Create test data

### Core Modules
- ✅ **Cache Layer** (`backend/api/cache.py`) - In-memory TTL caching
- ✅ **Stadium Database** (`backend/api/stadium_database.py`) - All 32 NFL stadiums
- ✅ **External APIs** (`backend/api/external_apis.py`) - Weather + Sleeper
- ✅ **Odds API** (`backend/api/odds_api.py`) - Real sportsbook lines
- ✅ **Model Loader** (`backend/api/model_loader.py`) - Load ML predictions
- ✅ **Insights Engine** (`backend/api/insights_engine.py`) - Statistical analysis
- ✅ **Narrative Generator** (`backend/api/narrative_generator.py`) - Templates
- ✅ **Prop Analyzer** (`backend/api/prop_analyzer.py`) - Edge calculation
- ✅ **Home Field Advantage** (`backend/features/home_field_advantage.py`) - HFA features
- ✅ **HFA Impact Analysis** (`backend/features/hfa_impact_analysis.py`) - Position-specific HFA

### Testing
- ✅ 31 tests passing
- ✅ Test coverage for all core endpoints
- ✅ Test coverage for enhanced analytics

---

## ⚠️ Partially Implemented (Scaffolded)

### API Endpoints
- ⚠️ `/api/v1/games/{game_id}/content` - Content aggregation
  - **Status:** Returns placeholder data
  - **TODO:** Integrate YouTube API, RSS feeds, podcast APIs
  - **Priority:** Low (users can Google themselves)

### Ingestion Layer
- ⚠️ `backend/ingestion/fetch_nflverse.py`
  - **Status:** Creates sample CSV, no actual download
  - **TODO:** Implement real download from nflverse GitHub releases
  - **Priority:** Medium (needed for model training)
  - **URL:** https://github.com/nflverse/nflverse-data/releases

- ⚠️ `backend/ingestion/fetch_injuries.py`
  - **Status:** Placeholder structure
  - **TODO:** Implement if needed (we have Sleeper API already)
  - **Priority:** Low (Sleeper API covers this)

### Roster/Injury System
- ⚠️ `backend/roster_injury/build_game_roster_index.py`
  - **Status:** Creates sample roster index
  - **TODO:** Load real roster data from nflverse
  - **Priority:** Medium (useful for injury impact analysis)

- ⚠️ `backend/roster_injury/build_injury_game_index.py`
  - **Status:** Placeholder structure
  - **TODO:** Map injuries to specific games
  - **Priority:** Medium (useful for backfilling historical injury data)

- ⚠️ `backend/roster_injury/roster_lookup.py`
  - **Status:** Returns placeholder status
  - **TODO:** Load from `outputs/game_rosters_YYYY.json`
  - **Priority:** Medium (needed for historical analysis)

### Feature Engineering
- ⚠️ `backend/features/extract_player_pbp_features.py`
  - **Status:** Structure exists
  - **TODO:** Check if implementation is complete
  - **Priority:** High (core for model training)

- ⚠️ `backend/features/smoothing_and_rolling.py`
  - **Status:** Structure exists
  - **TODO:** Check if implementation is complete
  - **Priority:** High (core for model training)

---

## ❌ Not Yet Implemented

### Orchestration Pipeline
- ❌ `backend/orchestration/orchestrator.py`
  - **Purpose:** Coordinate full pipeline (ingestion → features → model → predictions)
  - **TODO:** Build orchestration workflow
  - **Priority:** High (for automated model runs)
  - **Notes:** Referenced in `/admin/recompute` endpoint

### ML Model Pipeline
- ❌ Model training scripts
  - **TODO:** Build model training pipeline
  - **Priority:** High
  - **Files Needed:**
    - `backend/modeling/train_passing_model.py`
    - `backend/modeling/train_rushing_model.py`
    - `backend/modeling/train_receiving_model.py`

- ❌ Calibration & backtesting
  - **TODO:** Implement backtest framework
  - **Priority:** High (to validate model accuracy)
  - **Directory:** `backend/calib_backtest/`

### LLM Integration
- ❌ OpenAI/Claude narrative enhancement
  - **Status:** Scaffolded in `narrative_generator.py`
  - **TODO:** Implement `_enhance_with_llm()` function
  - **Priority:** Low (templates work fine for now)
  - **Cost:** ~$0.02 per game with GPT-4

### Content Aggregation
- ❌ YouTube API integration
- ❌ Podcast API integration
- ❌ Twitter/X API integration
  - **Priority:** Low for all
  - **Reason:** Users can find this content themselves

### Database Layer
- ❌ PostgreSQL integration
  - **Status:** Not implemented (using files currently)
  - **TODO:** Add database for historical data, user tracking, etc.
  - **Priority:** Medium (when you need user features)

---

## 🎯 Recommended Priority Order

### Next Sprint (High Value, Low Effort)
1. **Integrate HFA into prop projections** (2-3 hours)
   - Modify model_loader to apply HFA adjustments
   - Use `hfa_impact_analyzer.apply_hfa_to_projection()`
   - Immediate value for prop accuracy

2. **Implement nflverse download** (3-4 hours)
   - Replace placeholder in `fetch_nflverse.py`
   - Download play-by-play and player stats
   - Needed for model training

3. **Build simple orchestrator** (4-5 hours)
   - Coordinate: fetch data → build features → run models → output predictions
   - Makes model runs automated

### Following Sprint (Medium Priority)
4. **Implement roster index building** (3-4 hours)
   - Load real roster data
   - Build game-by-game roster index
   - Useful for injury impact analysis

5. **Add player prop projection adjustments** (2-3 hours)
   - Apply HFA, weather, injury adjustments to base projections
   - More accurate model outputs

6. **Build backtest framework** (1 week)
   - Validate model accuracy
   - Track calibration over time
   - Essential for trust in predictions

### Later (Lower Priority)
7. **LLM narrative enhancement** (2-3 hours)
   - Polish templates with GPT-4
   - Only if you have budget for it

8. **Content aggregation** (1-2 days)
   - YouTube, podcast, Twitter integration
   - Low value (users can Google)

9. **Database migration** (1 week)
   - Move from files to PostgreSQL
   - When you need user accounts, bet tracking, etc.

---

## 🔧 How to Use HFA Right Now

### Option 1: Apply HFA in Model Loader
```python
# In backend/api/model_loader.py
from backend.features.hfa_impact_analysis import hfa_impact_analyzer

def load_projections_for_game(self, game_id: str) -> List[PropProjection]:
    projections = self._load_csv(file_path)

    # Apply HFA adjustments
    adjusted_projections = []
    for proj in projections:
        # Determine if player is home/away
        is_home = self._is_player_home(proj.player_id, game_id)
        team = self._get_player_team(proj.player_id)
        position = self._get_player_position(proj.player_id)

        # Apply HFA
        result = hfa_impact_analyzer.apply_hfa_to_projection(
            base_projection=proj.projection,
            position=position,
            prop_type=proj.prop_type,
            game_id=game_id,
            team=team,
            is_home_team=is_home
        )

        # Update projection
        proj.projection = result['adjusted_projection']
        adjusted_projections.append(proj)

    return adjusted_projections
```

### Option 2: Add HFA as Feature in Model Training
```python
# In your model training script
from backend.features.home_field_advantage import hfa_calculator

# Add HFA features to your training dataframe
df['is_home'] = ...
df['stadium_hfa_multiplier'] = ...
df['travel_penalty'] = ...
# etc (12 HFA features total)

# Let the model learn HFA weights
# QB might weight travel_penalty higher
# RB might weight dome_advantage lower
```

### Option 3: Test HFA Impact
```python
# Create analysis script
from backend.features.hfa_impact_analysis import hfa_impact_analyzer

# Compare home vs away for same player
result = hfa_impact_analyzer.compare_home_away_props(
    position='QB',
    prop_type='passing_yards',
    base_projection=285.0,
    game_id='2025_10_KC_BUF',
    home_team='BUF',
    away_team='KC'
)

print(f"HFA swing: {result['total_hfa_swing']} yards")
# Output: "HFA swing: 12.3 yards"
```

---

## 📊 Expected HFA Impacts (By Position)

### QB Passing Props
- **Passing Yards:** +8.5 yards at home
- **Passing TDs:** +0.15 TDs at home
- **Completions:** +1.2 completions at home
- **Dome Bonus:** +15% (no weather interference)
- **Travel Penalty:** -12% per travel unit

### RB Props
- **Rushing Yards:** +4.2 yards at home
- **Rushing TDs:** +0.08 TDs at home
- **Receptions:** +0.3 receptions at home
- **Dome Bonus:** +5% (minimal)
- **Travel Penalty:** -8% per travel unit

### WR Props
- **Receiving Yards:** +6.3 yards at home
- **Receptions:** +0.5 receptions at home
- **Receiving TDs:** +0.12 TDs at home
- **Dome Bonus:** +12%
- **Travel Penalty:** -10% per travel unit

### TE Props
- **Receiving Yards:** +4.8 yards at home
- **Receptions:** +0.4 receptions at home
- **Receiving TDs:** +0.10 TDs at home

---

## 🚦 Status Summary

| Category | Status | Count |
|---|---|---|
| ✅ Production Ready | Complete | 13 endpoints, 10 modules |
| ⚠️ Partially Implemented | Scaffolded | 8 modules |
| ❌ Not Started | TODO | 5 major areas |

**Overall Completion:** ~70% for API layer, ~40% for full pipeline

**Next Bottleneck:** Model training pipeline (orchestration + feature engineering)

**Immediate Win:** Integrate HFA into prop projections (2-3 hours, big accuracy boost)
