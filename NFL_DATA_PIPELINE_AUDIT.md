# NFL Backend Data Pipeline Audit Report
**Generated:** 2025-11-20

---

## EXECUTIVE SUMMARY

The NFL backend data pipeline has **3 CRITICAL GAPS**:
1. **10 data sources are downloaded but NOT stored in database**
2. **6 tables exist but have NO refresh methods**
3. **Critical analyzers are using tables that may not be populated**

---

## PART 1: DATA SOURCES DOWNLOADED (fetch_nflverse.py)

### Core Datasets (Always Downloaded)
✓ **player_stats_{year}.csv** - Weekly player statistics
✓ **play_by_play_{year}.csv.gz** - Play-by-play with EPA, WPA, CPOE
✓ **weekly_rosters_{year}.csv** - Weekly roster data

### Advanced Datasets (include_all=True)
⚠ **nextgen_stats_{year}.csv** - Next Gen Stats (player tracking)
⚠ **snap_counts_{year}.csv** - Snap count participation %
⚠ **pfr_advstats_pass_{year}.csv** - PFR advanced passing
⚠ **pfr_advstats_rush_{year}.csv** - PFR advanced rushing  
⚠ **pfr_advstats_rec_{year}.csv** - PFR advanced receiving
⚠ **pfr_advstats_def_{year}.csv** - PFR advanced defense
⚠ **depth_charts_{year}.csv** - Weekly depth charts
⚠ **ftn_charting_{year}.csv** - Coverage schemes & formations
⚠ **participations_{year}.csv** - Play-level participation data
⚠ **combine.csv** - NFL combine results

---

## PART 2: DATABASE SCHEMA (local_db.py)

### IMPLEMENTED TABLES

**Betting/Projections Tables:**
- odds_snapshots (append-only) - Line movement tracking
- projections (append-only) - Model projection history
- value_props - Identified betting opportunities
- results - Actual game outcomes for backtesting
- model_runs - Training run logging

**Core Stat Tables:**
- player_stats - Player season/weekly stats (32 columns)
- team_stats - Team aggregates
- rosters - Current team rosters
- schedules - Full season schedule
- games - Schedule & results (APPEARS DEPRECATED)

**Injuries:**
- injuries (append-only) - Injury report tracking

**Advanced Tables (Created in data_refresh.py):**
- play_by_play - Game logs with EPA/WPA/CPOE
- snap_counts - Participation percentages
- ftn_charting - Coverage schemes & formations
- pfr_advanced - Advanced stats by type

**System Tables:**
- refresh_timestamps - Last refresh tracking

---

## PART 3: REFRESH METHODS (data_refresh.py)

### IMPLEMENTED REFRESH METHODS ✓

1. **refresh_injuries()** - Fetches from ESPN API, stores in SQLite
2. **refresh_schedules()** - Loads from CSV/Parquet files
3. **refresh_player_stats()** - Loads from CSV files
4. **refresh_team_stats()** - Loads from CSV files
5. **refresh_rosters()** - Loads rosters + depth_charts
6. **refresh_play_by_play()** - Loads from Parquet/CSV files
7. **refresh_odds()** - Fetches from The Odds API
8. **refresh_snap_counts()** - Loads from CSV files
9. **refresh_ftn_charting()** - Loads from CSV files
10. **refresh_pfr_advanced()** - Loads from CSV files

---

## CRITICAL GAPS FOUND

### GAP 1: Downloaded But Never Stored in Database

| Data Source | Downloaded | Table | Refresh | Status |
|---|---|---|---|---|
| nextgen_stats | ✓ | ✗ MISSING | ✗ | **BROKEN** |
| participations | ✓ | ✗ MISSING | ✗ | **BROKEN** |
| combine | ✓ | ✗ MISSING | ✗ | **BROKEN** |
| depth_charts | ✓ | ⚠ (merged with rosters) | ⚠ (in refresh_rosters) | **PARTIAL** |

**Impact:** These files are downloaded to disk but never loaded into SQLite. Any analyzer needing this data must read raw CSV files.

---

### GAP 2: Tables Exist But NO Refresh Methods

| Table | Created | Refresh Method | Status |
|---|---|---|---|
| projections | ✓ local_db.py | ✗ MISSING | **BROKEN** |
| results | ✓ local_db.py | ✗ MISSING | **BROKEN** |
| value_props | ✓ local_db.py | ✗ MISSING | **BROKEN** |
| model_runs | ✓ local_db.py | ✗ (logging only) | **INCOMPLETE** |
| games | ✓ local_db.py | ✗ MISSING | **DEPRECATED?** |

**Impact:** Table structures exist but data is never populated from files or APIs.

---

### GAP 3: Data Pipeline Completeness

```
Scenario 1: nextgen_stats (Next Gen Stats - Player Tracking)
┌──────────────────────────────────────────────────────────┐
│ Downloaded: ngs_passing_2024_2025.csv (278 KB)           │
│ Downloaded: ngs_receiving_2024_2025.csv (446 KB)         │
│ Downloaded: ngs_rushing_2024_2025.csv (195 KB)           │
│ Table Definition: ✗ NONE                                  │
│ Refresh Method: ✗ NONE                                    │
│ Current State: Raw files only, not in database            │
│ Analyzer Access: Direct CSV file read or data wasted      │
└──────────────────────────────────────────────────────────┘

Scenario 2: participations (Play-Level Participation)
┌──────────────────────────────────────────────────────────┐
│ Downloaded: participations_{year}.csv                     │
│ Table Definition: ✗ NONE                                  │
│ Refresh Method: ✗ NONE                                    │
│ Current State: Downloaded but completely unused           │
│ Analyzer Access: ✗ NOT ACCESSIBLE via API                │
└──────────────────────────────────────────────────────────┘

Scenario 3: projections (Model Projections)
┌──────────────────────────────────────────────────────────┐
│ Downloaded: ✗ NOT DOWNLOADED (needs model generation)    │
│ Table Definition: ✓ EXISTS (13 columns)                   │
│ Refresh Method: ✗ MISSING                                 │
│ Current State: Table exists but always empty              │
│ Analyzer Access: ✓ Can query but no data                  │
└──────────────────────────────────────────────────────────┘
```

---

## CURRENT FILE INVENTORY vs DATABASE

### Files in /home/user/nfl_backend/inputs/

**STORED IN DATABASE (Verified):**
- ✓ player_stats_2024_2025.csv → player_stats table
- ✓ player_stats_2025.csv → player_stats table
- ✓ play_by_play_2025.parquet → play_by_play table
- ✓ schedules_2024_2025.csv → schedules table
- ✓ snap_counts_2024_2025.csv → snap_counts table
- ✓ team_stats_2024_2025.csv → team_stats table
- ✓ rosters_weekly_2024_2025.csv → rosters table
- ✓ depth_charts_2024_2025.csv → rosters table (merged)

**NOT IN DATABASE (Data Loss):**
- ✗ ngs_passing_2024_2025.csv (278 KB) → No table, No refresh
- ✗ ngs_receiving_2024_2025.csv (446 KB) → No table, No refresh
- ✗ ngs_rushing_2024_2025.csv (195 KB) → No table, No refresh
- ✗ players.csv (8.3 MB) → Player metadata lost
- ✗ teams.csv → Team metadata lost
- ✗ ftn_charting missing from inputs (though table exists)

**EXTRA FILES (Non-nflverse):**
- red_zone_stats_2025.csv → No refresh method
- defensive_stats_from_pbp_2025.csv → Loaded into team_stats?
- kicker_stats_2025.csv → No table or refresh
- officials_2024_2025.csv → No table or refresh
- player_stats_enhanced_2025.csv → Likely duplicate

---

## ANALYZER DATA DEPENDENCIES

### defense_analyzer.py
**Requires:**
- play_by_play table (EPA, WPA, coverage, personnel) ✓ **Available**
- team_stats → defense rankings ✓ **Available**

### situational_analyzer.py
**Requires:**
- team rankings → Uses team_stats ✓ **Available**
- play_by_play → For metrics ✓ **Available**
- schedules → Weather, rest ✓ **Available**

### prediction_engine.py
**Likely Requires:**
- player_stats → Historical performance ✓ **Available**
- play_by_play → Historical plays ✓ **Available**
- snap_counts → Usage % ⚠ **May be incomplete**

### injury_impact_analyzer.py
**Requires:**
- injuries table ✓ **Available**
- rosters table ✓ **Available**
- player_stats ✓ **Available**

---

## MISSING IN CURRENT PIPELINE

### Never Downloaded (Should Be Added)
- ⚠ **Team rosters metadata** - players.csv, teams.csv not used
- ⚠ **Kicker stats** - kicker_stats_2025.csv not in database
- ⚠ **Official data** - officials_2024_2025.csv not used

### Downloaded But Ignored
- ⚠ **Next Gen Stats (player tracking)** - ngs_*.csv files
- ⚠ **Participations** - Play-level participation data
- ⚠ **Combine results** - combine.csv historical data

### Table Exists But Never Populated
- ⚠ **projections** - No generation logic
- ⚠ **results** - No game result loader
- ⚠ **value_props** - No value detection logic
- ⚠ **games** - Replaced by schedules?

---

## RECOMMENDATIONS

### CRITICAL (Do First)
1. **Add nextgen_stats table** to local_db.py
   - Columns: season, week, player_id, player_name, team, position, max_speed, max_accel, avg_speed, routes, targets, rec_yards
   - Add refresh_nextgen_stats() method

2. **Add results table population**
   - Implement refresh_results() to extract from play_by_play + player_stats
   - Required for backtesting/evaluation

3. **Implement projections generator**
   - Add method to generate and store model projections
   - Enable projection history tracking

### HIGH (Do Soon)
4. **Add FTN charting to nflverse fetch** (currently not in fetch list)
5. **Add participations table** if play-level data needed
6. **Remove/consolidate games table** (appears duplicate with schedules)
7. **Add nextgen_stats refresh** to startup_refresh()

### MEDIUM (Nice to Have)
8. **Create combine table** for draft history lookups
9. **Add kicker_stats table** for special teams analysis
10. **Add player_metadata table** from players.csv
11. **Consolidate duplicate player_stats files** in inputs/

---

## VERIFICATION CHECKLIST

- [ ] Does fetch_nflverse.py download all needed data?
- [ ] Does local_db.py have tables for all downloaded files?
- [ ] Does data_refresh.py have refresh methods for all tables?
- [ ] Does startup_refresh() call all refresh methods?
- [ ] Are all tables queryable via SQL from analyzers?
- [ ] Is database initialization idempotent?
- [ ] Do file formats match between fetch and refresh (CSV vs Parquet)?
- [ ] Are staleness thresholds appropriate?

---

## DATA FLOW MAP

```
SOURCES
  ↓
┌─────────────────────────────────────────┐
│ nflverse GitHub Releases                │
│ - player_stats                          │
│ - play_by_play                          │
│ - weekly_rosters                        │
│ - snap_counts                           │
│ - nextgen_stats    ← DOWNLOADED         │
│ - depth_charts                          │
│ - ftn_charting                          │
│ - participations   ← DOWNLOADED         │
│ - combine          ← DOWNLOADED         │
└──────────────┬──────────────────────────┘
               │ fetch_nflverse.py
               ↓
        /home/user/nfl_backend/inputs/
               │
  ┌────────────┴────────────┐
  │                         │
  ↓ data_refresh.py         ↓ Not Loaded
  
┌──────────────────────┐   ├─ ngs_*.csv
│ SQLite Database      │   ├─ participations_*.csv
│ ─────────────────   │   ├─ combine.csv
│ player_stats     ✓   │   └─ (Other custom files)
│ play_by_play     ✓   │
│ snap_counts      ✓   │
│ ftn_charting     ✓   │
│ pfr_advanced     ✓   │
│ schedules        ✓   │
│ rosters          ✓   │
│ team_stats       ✓   │
│ injuries         ✓   │
│ games            ⚠   │
│ odds_snapshots   ✓   │
│ projections      ✗   │
│ results          ✗   │
│ nextgen_stats    ✗   │
│ value_props      ✗   │
└──────────────────────┘
        ↓ API/Analyzers
```

