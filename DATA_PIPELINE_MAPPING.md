# NFL Backend Data Pipeline: Complete Mapping

## SOURCE → TABLE → REFRESH → STATUS

### CORE DATA PIPELINE

| Source File | Downloaded | Database Table | Refresh Method | Current Status | In Startup |
|---|---|---|---|---|---|
| **player_stats_{year}.csv** | ✓ fetch_nflverse.py:L75 | player_stats | refresh_player_stats() | ✓ WORKING | ✓ Yes |
| **play_by_play_{year}.csv.gz** | ✓ fetch_nflverse.py:L81 | play_by_play | refresh_play_by_play() | ✓ WORKING | ✓ Yes |
| **weekly_rosters_{year}.csv** | ✓ fetch_nflverse.py:L88 | rosters | refresh_rosters() | ✓ WORKING | ✓ Yes |
| **schedules_{year}.csv** | ⚠ Not in fetch_nflverse.py | schedules | refresh_schedules() | ⚠ PARTIAL | ✓ Yes |
| **snap_counts_{year}.csv** | ✓ fetch_nflverse.py:L104 | snap_counts | refresh_snap_counts() | ✓ WORKING | ✓ Yes |
| **depth_charts_{year}.csv** | ✓ fetch_nflverse.py:L129 | rosters (merged) | refresh_rosters() | ✓ WORKING | ✓ Yes |
| **ftn_charting_{year}.csv** | ✓ fetch_nflverse.py:L133 | ftn_charting | refresh_ftn_charting() | ✓ WORKING | ✓ Yes |
| **pfr_advstats_pass_{year}.csv** | ✓ fetch_nflverse.py:L109 | pfr_advanced | refresh_pfr_advanced() | ✓ WORKING | ✓ Yes |
| **pfr_advstats_rush_{year}.csv** | ✓ fetch_nflverse.py:L113 | pfr_advanced | refresh_pfr_advanced() | ✓ WORKING | ✓ Yes |
| **pfr_advstats_rec_{year}.csv** | ✓ fetch_nflverse.py:L118 | pfr_advanced | refresh_pfr_advanced() | ✓ WORKING | ✓ Yes |
| **pfr_advstats_def_{year}.csv** | ✓ fetch_nflverse.py:L123 | pfr_advanced | refresh_pfr_advanced() | ✓ WORKING | ✓ Yes |

---

### BROKEN PIPELINE ITEMS

| Source File | Downloaded | Database Table | Refresh Method | Current Status | Impact |
|---|---|---|---|---|---|
| **nextgen_stats_{year}.csv** | ✓ fetch_nflverse.py:L98 | ✗ MISSING | ✗ MISSING | **BROKEN** | Player tracking data lost |
| **participations_{year}.csv** | ✓ fetch_nflverse.py:L138 | ✗ MISSING | ✗ MISSING | **BROKEN** | Play-level participation lost |
| **combine.csv** | ✓ fetch_nflverse.py:L143 | ✗ MISSING | ✗ MISSING | **BROKEN** | Historical draft data lost |

---

### TABLES WITHOUT DATA SOURCES

| Database Table | Location | Refresh Method | Has Data | Issue |
|---|---|---|---|---|
| **projections** | local_db.py:L82 | ✗ MISSING | ✗ Always empty | Model projections never generated |
| **results** | local_db.py:L164 | ✗ MISSING | ✗ Always empty | Game results never extracted |
| **value_props** | local_db.py:L183 | ✗ MISSING | ✗ Always empty | Betting opportunities never found |
| **odds_snapshots** | local_db.py:L47 | ✓ refresh_odds() | ⚠ API-dependent | Requires The Odds API key |
| **injuries** | local_db.py:L115 | ✓ refresh_injuries() | ⚠ API-dependent | Requires ESPN API access |
| **games** | local_db.py:L141 | ✗ MISSING | ⚠ Unused | Appears DEPRECATED (use schedules) |
| **model_runs** | local_db.py:L203 | ✗ (logging only) | ✓ Logging works | Not for data loading |

---

## DETAILED TABLE STATUS

### COMPLETE PIPELINE ✓

```
1. player_stats
   Downloaded:  ✓ inputs/player_stats_2024_2025.csv (12 MB)
   Table:       ✓ local_db.py:L219 (32 columns)
   Refresh:     ✓ data_refresh.py:L737 (finds *.csv files)
   Startup:     ✓ Called in refresh_all() L478
   Status:      FULLY OPERATIONAL

2. play_by_play
   Downloaded:  ✓ inputs/play_by_play_2025.parquet (12 MB)
   Table:       ✓ Created in data_refresh.py:L45 (45 columns)
   Refresh:     ✓ data_refresh.py:L928 (finds *.parquet & *.csv)
   Startup:     ✓ Called in refresh_all() L493
   Status:      FULLY OPERATIONAL

3. schedules
   Downloaded:  ⚠ inputs/schedules_2024_2025.csv (185 KB)
   Table:       ✓ local_db.py:L356 (39 columns)
   Refresh:     ✓ data_refresh.py:L669 (finds *schedule*.csv)
   Startup:     ✓ Called in refresh_all() L473
   Status:      OPERATIONAL (not in fetch_nflverse!)

4. snap_counts
   Downloaded:  ✓ inputs/snap_counts_2024_2025.csv (4.3 MB)
   Table:       ✓ Created in data_refresh.py:L122 (13 columns)
   Refresh:     ✓ data_refresh.py:L1082 (finds snap_counts*.csv)
   Startup:     ✓ Called in refresh_all() L504
   Status:      FULLY OPERATIONAL

5. ftn_charting
   Downloaded:  ⚠ Not found in inputs/ (though in fetch_nflverse.py:L133)
   Table:       ✓ Created in data_refresh.py:L153 (11 columns)
   Refresh:     ✓ data_refresh.py:L1164 (finds ftn_charting*.csv)
   Startup:     ✓ Called in refresh_all() L509
   Status:      PARTIAL (no input file found)

6. pfr_advanced
   Downloaded:  ✓ inputs/pfr_advstats_*.csv files
   Table:       ✓ Created in data_refresh.py:L183 (21 columns)
   Refresh:     ✓ data_refresh.py:L1244 (finds pfr_advstats*.csv)
   Startup:     ✓ Called in refresh_all() L514
   Status:      FULLY OPERATIONAL

7. rosters
   Downloaded:  ✓ inputs/rosters_weekly_2024_2025.csv (27 MB)
               ✓ inputs/depth_charts_2024_2025.csv (45 MB)
   Table:       ✓ local_db.py:L323 (14 columns)
   Refresh:     ✓ data_refresh.py:L864 (finds rosters + depth_charts)
   Startup:     ✓ Called in refresh_all() L488
   Status:      FULLY OPERATIONAL

8. team_stats
   Downloaded:  ✓ inputs/team_stats_2024_2025.csv (283 KB)
   Table:       ✓ local_db.py:L282 (22 columns)
   Refresh:     ✓ data_refresh.py:L800 (finds team_stats + defensive_stats)
   Startup:     ✓ Called in refresh_all() L483
   Status:      FULLY OPERATIONAL

9. injuries
   Downloaded:  ✓ (via ESPN API, not nflverse)
   Table:       ✓ local_db.py:L115 (8 columns, append-only)
   Refresh:     ✓ data_refresh.py:L604 (fetches from ESPN API)
   Startup:     ✓ Called in refresh_all() L468
   Status:      API-DEPENDENT (requires network)

10. odds_snapshots
    Downloaded:  ✓ (via The Odds API, not nflverse)
    Table:       ✓ local_db.py:L47 (13 columns, append-only)
    Refresh:     ✓ data_refresh.py:L523 (fetches from The Odds API)
    Startup:     ✓ Called in refresh_all() L498
    Status:      API-DEPENDENT (requires API key)
```

---

### BROKEN PIPELINE ✗

```
1. nextgen_stats (Next Gen Stats)
   Downloaded:  ✓ fetch_nflverse.py:L98 - URL pattern configured
   Files:       ✓ inputs/ngs_passing_2024_2025.csv (278 KB)
                ✓ inputs/ngs_receiving_2024_2025.csv (446 KB)
                ✓ inputs/ngs_rushing_2024_2025.csv (195 KB)
   Table:       ✗ NONE - Not defined in local_db.py
   Refresh:     ✗ NONE - No refresh method
   Startup:     ✗ Not called
   Status:      DATA LOSS - Downloaded but unused

2. participations (Play-Level Participation)
   Downloaded:  ✓ fetch_nflverse.py:L138 - URL pattern configured
   Files:       ✗ Not found in inputs/ (may not exist for 2025)
   Table:       ✗ NONE - Not defined
   Refresh:     ✗ NONE - No refresh method
   Startup:     ✗ Not called
   Status:      UNUSED - Downloaded but not stored

3. combine (NFL Combine Results)
   Downloaded:  ✓ fetch_nflverse.py:L143 - URL pattern configured
   Files:       ✗ Not found in inputs/ (single year file)
   Table:       ✗ NONE - Not defined
   Refresh:     ✗ NONE - No refresh method
   Startup:     ✗ Not called
   Status:      UNUSED - Downloaded but not stored

4. projections (Model Projections)
   Downloaded:  ✗ NONE - Requires model generation
   Table:       ✓ local_db.py:L82 - Defined (13 columns, append-only)
   Refresh:     ✗ NONE - No generation method
   Startup:     ✗ Not called
   Status:      EMPTY - Table exists but never populated
   Notes:       Needs model to generate projections

5. results (Game Results for Backtesting)
   Downloaded:  ✗ NONE - Should extract from play_by_play
   Table:       ✓ local_db.py:L164 - Defined (5 columns, unique constraint)
   Refresh:     ✗ NONE - No extraction method
   Startup:     ✗ Not called
   Status:      EMPTY - Table exists but never populated
   Notes:       Could be populated from play_by_play + player_stats

6. value_props (Identified Betting Opportunities)
   Downloaded:  ✗ NONE - Requires analysis
   Table:       ✓ local_db.py:L183 - Defined (11 columns, append-only)
   Refresh:     ✗ NONE - No analysis method
   Startup:     ✗ Not called
   Status:      EMPTY - Table exists but never populated
   Notes:       Needs prop analysis engine to populate

7. games (Schedule & Results)
   Downloaded:  ✗ NONE - Replaced by schedules
   Table:       ✓ local_db.py:L141 - Defined (11 columns)
   Refresh:     ✗ NONE - No refresh method
   Startup:     ✗ Not called
   Status:      DEPRECATED - Appears redundant with schedules table
   Notes:       schedules table has 39 columns, games has 11
```

---

## ANALYZER DATA REQUIREMENTS VERIFICATION

### defense_analyzer.py (Defensive Analysis)
**Line: Searches play_by_play queries**
```sql
FROM play_by_play p
  - Accesses: game_id, season, week, posteam, defteam, play_type, epa, wpa, 
              yards_gained, touchdown, pass_length, pass_location, run_location,
              receivers, rushers, coverage_type, offense_personnel, defense_personnel
  - Status: ✓ Available in play_by_play table
```

### situational_analyzer.py
**Accesses: team_rankings (from team_stats)**
```sql
  - Accesses: team, wins, losses, points_scored, offense_rank, defense_rank
  - Status: ✓ Available in team_stats table
```

### prediction_engine.py (Likely Usage)
```python
  - player_stats: ✓ Available
  - play_by_play: ✓ Available
  - snap_counts: ✓ Available (but may be incomplete)
  - pfr_advanced: ✓ Available
  - nextgen_stats: ✗ MISSING (needs ngs_*.csv)
```

### injury_impact_analyzer.py
```python
  - injuries: ✓ Available
  - rosters: ✓ Available
  - player_stats: ✓ Available
```

---

## INPUT FILE STATUS

### Properly Utilized Files ✓
- `player_stats_2024_2025.csv` → player_stats table
- `play_by_play_2025.parquet` → play_by_play table
- `schedules_2024_2025.csv` → schedules table
- `snap_counts_2024_2025.csv` → snap_counts table
- `rosters_weekly_2024_2025.csv` → rosters table
- `depth_charts_2024_2025.csv` → rosters table
- `team_stats_2024_2025.csv` → team_stats table

### Downloaded But Unused ⚠
- `ngs_passing_2024_2025.csv` (278 KB) - No table, no refresh
- `ngs_receiving_2024_2025.csv` (446 KB) - No table, no refresh
- `ngs_rushing_2024_2025.csv` (195 KB) - No table, no refresh
- `players.csv` (8.3 MB) - No table, no refresh
- `teams.csv` - No table, no refresh
- `officials_2024_2025.csv` - No table, no refresh

### Extra/Custom Files ⚠
- `player_stats_2025.csv` (816 KB) - Duplicate?
- `player_stats_enhanced_2025.csv` (6.7 MB) - Where from?
- `red_zone_stats_2025.csv` (151 KB) - No table defined
- `kicker_stats_2025.csv` (36 KB) - No table defined
- `games_2025_with_quarters.csv` - For games table?
- `player_stats_2025_from_pbp.csv` - Generated from PBP?
- `defensive_stats_from_pbp_2025.csv` - Generated from PBP?

---

## RECOMMENDED FIXES (Priority Order)

### P0: CRITICAL - Data Completeness
1. Create `nextgen_stats` table (ngs_*.csv files)
2. Implement `refresh_results()` method
3. Implement `projections` generator method
4. Fix `schedules` download (add to fetch_nflverse.py)

### P1: HIGH - Data Integrity
5. Create `participations` table (optional, verify need)
6. Clarify `games` vs `schedules` duplication
7. Add `ftn_charting` verification (in fetch but not in inputs/)
8. Document all non-nflverse files in inputs/

### P2: MEDIUM - Enhancement
9. Create `combine` table for draft history
10. Create `kicker_stats` table
11. Create player metadata table from players.csv
12. Consolidate duplicate player_stats files

### P3: LOW - Cleanup
13. Remove unused input files
14. Add data lineage documentation
15. Add validation/integrity checks

