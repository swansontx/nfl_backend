# Data Quality Analysis Report
Generated: 2025-11-19

## Executive Summary

This report documents data quality issues identified in the NFL Backend codebase. The analysis revealed **13 critical and high-priority issues** affecting data consistency, missing components, and potential runtime errors.

## Critical Issues (Fix Immediately)

### 1. ✅ FIXED: Incorrect game_id Format
**Location:** `backend/canonical/map_event_to_game.py:7-9`
**Status:** FIXED
**Issue:** game_id was using format `week_10_BUF` instead of nflverse standard `{season}_{week}_{away}_{home}`

**Impact:**
- Cannot distinguish between seasons (2024 vs 2025)
- Missing away team information
- Would break joins with nflverse data

**Fix Applied:**
- Updated to correct format: `2025_10_KC_BUF`
- Added comprehensive documentation
- Added `season` parameter with default 2025

### 2. Missing Python Package Structure
**Location:** All `backend/` subdirectories
**Severity:** HIGH
**Issue:** No `__init__.py` files in any backend directories

**Impact:**
- Python cannot treat directories as packages
- Imports like `from backend.api import app` will fail
- Docker container may fail to start
- Tests cannot import modules

**Recommended Fix:**
```bash
touch backend/__init__.py
touch backend/api/__init__.py
touch backend/canonical/__init__.py
touch backend/features/__init__.py
touch backend/ingestion/__init__.py
touch backend/roster_injury/__init__.py
```

### 3. Missing Core Components
**Location:** Entire backend structure
**Severity:** HIGH
**Issue:** Documentation references components that don't exist

**Missing Directories/Files:**
- `backend/modeling/` - entire directory missing
  - `model_runner.py` (referenced in COMPONENTS.md:17)
- `backend/calib_backtest/` - entire directory missing
  - `calibrate.py` (referenced in COMPONENTS.md:20)
  - `backtest.py` (referenced in COMPONENTS.md:21)
- `backend/orchestration/` - entire directory missing
  - Runner scripts (referenced in ARCHITECTURE.md:11)

**Impact:**
- Core functionality for modeling and calibration is absent
- Cannot run the full pipeline
- Documentation is misleading

### 4. Inconsistent File Path Conventions
**Location:** Multiple files
**Severity:** MEDIUM
**Issue:** Inconsistent use of `inputs/`, `outputs/`, `data/`, and `cache/` directories

**Inconsistencies Found:**

| Component | Document Says | Code Says | Match? |
|-----------|--------------|-----------|--------|
| fetch_nflverse | `inputs/stats_player_week_YYYY.csv` | `inputs/stats_player_week_{year}.csv` | ✅ |
| fetch_nflverse | `inputs/player_lookup.json` | `inputs/player_lookup_YYYY.json` | ❌ |
| fetch_injuries | `outputs/injuries_YYYYMMDD_parsed.json` | Not implemented | ❌ |
| game rosters | `outputs/game_rosters_YYYY.json` | `outputs/game_rosters_YYYY.json` | ✅ |
| features | `outputs/player_pbp_features_by_id.json` | `outputs/player_pbp_features_by_id.json` | ✅ |
| fetch_odds | `cache/web_event_<id>.json` | Not implemented | ❌ |

**Issue Details:**

1. **player_lookup filename inconsistency:**
   - COMPONENTS.md:4 says: `inputs/player_lookup.json` (no year)
   - fetch_nflverse.py:5 says: `inputs/player_lookup_YYYY.json` (with year)

2. **Architecture mentions `data/` directory:**
   - ARCHITECTURE.md:13 mentions `data/` or `outputs/` layout
   - No actual use of `data/` directory in code

### 5. Missing Ingestion Scripts
**Location:** `backend/ingestion/`
**Severity:** HIGH
**Issue:** Only 1 of 3 expected ingestion scripts exists

**Expected vs Actual:**
- ✅ `fetch_nflverse.py` - EXISTS (scaffold)
- ❌ `fetch_odds.py` - MISSING
- ❌ `fetch_injuries.py` - MISSING

**Impact:**
- Cannot fetch odds API data
- Cannot fetch injury data
- Incomplete ingestion pipeline

### 6. Missing Canonical Mapping Utilities
**Location:** `backend/canonical/`
**Severity:** MEDIUM
**Issue:** Only 1 of 2 expected utilities exists

**Expected vs Actual:**
- ✅ `map_event_to_game.py` - EXISTS (now fixed)
- ❌ `player_map.py` - MISSING (referenced in COMPONENTS.md:10)

**Impact:**
- Cannot map odds API player names to nflverse player_ids
- Manual name matching will be required

### 7. Missing Features Scripts
**Location:** `backend/features/`
**Severity:** MEDIUM
**Issue:** Only 1 of 2 expected scripts exists

**Expected vs Actual:**
- ✅ `extract_player_pbp_features.py` - EXISTS (scaffold)
- ❌ `smoothing_and_rolling.py` - MISSING (referenced in COMPONENTS.md:14)

### 8. Missing Roster/Injury Builders
**Location:** `backend/roster_injury/`
**Severity:** MEDIUM
**Issue:** Only lookup service exists, missing data builders

**Expected vs Actual:**
- ✅ `roster_lookup.py` - EXISTS (returns 'ACT' hardcoded)
- ❌ `build_game_roster_index.py` - MISSING (referenced in COMPONENTS.md:24)
- ❌ `build_injury_game_index.py` - MISSING (referenced in COMPONENTS.md:25)

**Impact:**
- Roster lookup has no data to read from
- Will always return default 'ACT' status
- Cannot track injuries or roster changes

### 9. Incomplete player_lookup File Naming
**Location:** `backend/ingestion/fetch_nflverse.py:5`
**Severity:** MEDIUM
**Issue:** Inconsistent naming - stats file has year suffix, player_lookup doesn't

**Current:**
- `stats_player_week_YYYY.csv` - includes year ✅
- `player_lookup_YYYY.json` - doc says this, but COMPONENTS.md:4 says `player_lookup.json`

**Recommended:** Decide on standard - either both with year or neither

### 10. Missing Test Infrastructure
**Location:** Entire project
**Severity:** MEDIUM
**Issue:** No test files exist

**Missing:**
- No `tests/` directory
- No test files (test_*.py)
- CI runs `pytest -q || true` but will find nothing
- `|| true` masks test failures

**Impact:**
- No validation of components
- Cannot verify fixes
- Poor code quality assurance

## Medium Priority Issues

### 11. Incomplete CSV Schema Definition
**Location:** `backend/ingestion/fetch_nflverse.py:23`
**Severity:** MEDIUM
**Issue:** Sample CSV has minimal schema

**Current Schema:**
```csv
player_id,week,team,stat
```

**Missing Fields (typical nflverse data):**
- `season` - Which season is this? (Critical for 2025 labeling!)
- `season_type` - REG, POST, PRE?
- `opponent` - Who did they play against?
- `game_id` - Link to specific game
- Specific stat columns (rush_yards, pass_yards, etc.)

**Impact:**
- Sample data won't match real nflverse schema
- Testing with sample will not catch schema issues

### 12. Missing Season Context in Roster Lookup
**Location:** `backend/roster_injury/roster_lookup.py:13-15`
**Severity:** MEDIUM
**Issue:** Function signature doesn't consider that game_id contains season

**Current:**
```python
def get_player_status(game_id: str, player_id: str) -> str:
```

With new game_id format `2025_10_KC_BUF`, the season is embedded. The TODO comment mentions loading from `outputs/game_rosters_YYYY.json` but doesn't specify how to extract YYYY from game_id.

**Recommended:** Add helper to parse game_id
```python
def parse_game_id(game_id: str) -> dict:
    """Parse game_id into components.

    Args:
        game_id: Format {season}_{week}_{away}_{home}

    Returns:
        dict with keys: season, week, away_team, home_team
    """
    parts = game_id.split('_')
    return {
        'season': int(parts[0]),
        'week': int(parts[1]),
        'away_team': parts[2],
        'home_team': parts[3]
    }
```

### 13. API Endpoint Placeholder Returns
**Location:** `backend/api/app.py:18-21`
**Severity:** LOW
**Issue:** Endpoints return empty/placeholder data

**Current Behavior:**
- `GET /game/{game_id}/projections` - returns empty projections array
- `POST /admin/recompute` - doesn't actually trigger anything

**Impact:**
- API is non-functional
- Cannot test integration
- Frontend will receive no data

## Data Directory Structure Issues

### No Standard Directory Layout
**Issue:** Code references multiple directory patterns without consistency

**Current References:**
- `inputs/` - for ingested data
- `outputs/` - for processed data
- `cache/` - for cached API responses
- `data/` - mentioned in architecture but unused

**Recommended Structure:**
```
/app/
  data/
    inputs/           # Raw ingested data
      stats/          # nflverse stats
      odds/           # Odds API cache
      injuries/       # Injury reports
    outputs/          # Processed/generated data
      features/       # Feature extractions
      rosters/        # Roster indices
      projections/    # Model outputs
    models/           # Saved model files
```

## Missing Dependencies

### Potential Missing Requirements
**Location:** `requirements.txt`
**Issue:** Only 4 packages listed for a data science project

**Current:**
- fastapi
- uvicorn[standard]
- pydantic
- pytest

**Likely Missing:**
- `pandas` - for CSV processing
- `numpy` - for numerical operations
- `requests` - for API calls (odds, injuries)
- `scikit-learn` - for modeling (if using sklearn)
- Data validation libraries (e.g., `pandera`)
- Testing libraries (e.g., `pytest-asyncio` for API tests)

## 2025 Season Labeling - Verification Needed

### Current Status: Likely Correct
The default year is set to 2025 in `fetch_nflverse.py:29` and `map_event_to_game.py:17`.

**Verification Checklist:**
- [ ] Confirm nflverse 2025 data is available
- [ ] Verify season transition logic (when to switch to 2026)
- [ ] Document playoff handling (Jan/Feb 2026 games = 2025 season)
- [ ] Add season detection logic for automated updates

## Recommendations

### Immediate Actions (This Week)
1. ✅ Fix game_id format - COMPLETED
2. Add __init__.py files to all backend directories
3. Create missing ingestion scripts (fetch_odds.py, fetch_injuries.py)
4. Update player_lookup filename to be consistent
5. Add game_id parsing utility function

### Short Term (Next 2 Weeks)
1. Create modeling directory and scaffold model_runner.py
2. Create calib_backtest directory with calibrate.py and backtest.py
3. Add player_map.py utility for name matching
4. Create build_game_roster_index.py and build_injury_game_index.py
5. Establish standard directory structure (data/inputs, data/outputs)
6. Add test directory with basic smoke tests

### Medium Term (This Month)
1. Add comprehensive tests for each component
2. Update CI to fail on test failures (remove || true)
3. Add data validation schemas
4. Document nflverse schema mappings
5. Add season transition detection logic
6. Update requirements.txt with all needed dependencies

### Documentation Updates Needed
1. Update COMPONENTS.md to match actual file structure
2. Add DATA_SCHEMA.md documenting all file formats
3. Add DEVELOPMENT.md with setup instructions
4. Update README.md with current status and roadmap

## File Naming Standards (Proposed)

### Established Patterns
Based on current code, standardize on:

**Year Labeling:**
- `{description}_YYYY.{ext}` for annual data
- `{description}_YYYYMMDD.{ext}` for daily snapshots
- Always include year for historical data

**Examples:**
- ✅ `stats_player_week_2025.csv`
- ✅ `player_lookup_2025.json`
- ✅ `game_rosters_2025.json`
- ✅ `injuries_20251119_parsed.json`

**Directory Structure:**
- `inputs/` - external data sources
- `outputs/` - generated/processed data
- `cache/` - temporary API responses

## Summary

**Total Issues Found:** 13

**Breakdown by Severity:**
- Critical: 1 (fixed)
- High: 5
- Medium: 7

**Components with Issues:**
- canonical/ - 2 issues (1 fixed, 1 missing file)
- ingestion/ - 3 issues
- features/ - 1 issue
- roster_injury/ - 2 issues
- modeling/ - 1 issue (entire component missing)
- calib_backtest/ - 1 issue (entire component missing)
- Project structure - 3 issues

**Next Steps:**
Focus on the 5 HIGH severity issues first, particularly:
1. Adding __init__.py files (5 min fix)
2. Creating missing ingestion scripts
3. Adding missing modeling/calibration components
