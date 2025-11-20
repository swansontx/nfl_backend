# Critical Data Pipeline Fixes Required

## FIX #1: Missing nextgen_stats Table

### Current Status
- **Downloaded files exist:** ngs_passing_2024_2025.csv, ngs_receiving_2024_2025.csv, ngs_rushing_2024_2025.csv
- **Database table:** MISSING
- **Refresh method:** MISSING
- **Impact:** Player tracking data is wasted

### What Needs to Be Done

#### Step 1: Add Table Definition to local_db.py

Location: `/home/user/nfl_backend/backend/database/local_db.py`

After line 409 (end of schedules table), add:

```python
# Next Gen Stats - Player tracking metrics (speed, acceleration, routes)
cursor.execute("""
    CREATE TABLE IF NOT EXISTS nextgen_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        season INTEGER,
        week INTEGER,
        nflverse_game_id TEXT,
        player_id TEXT,
        player_name TEXT,
        team TEXT,
        position TEXT,
        stat_type TEXT,  -- 'passing', 'receiving', 'rushing'
        max_speed REAL,
        avg_speed REAL,
        max_acceleration REAL,
        avg_acceleration REAL,
        routes INTEGER,
        targets INTEGER,
        rec_yards INTEGER,
        rec_tds INTEGER,
        attempts INTEGER,
        avg_depth_of_target REAL,
        avg_air_yards REAL,
        avg_yards_after_catch REAL,
        UNIQUE(season, player_id, stat_type)
    )
""")

cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_nextgen_player
    ON nextgen_stats(player_id, season)
""")

cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_nextgen_team
    ON nextgen_stats(team, season, position)
""")
```

#### Step 2: Add Repository Class to local_db.py

After line 1350 (after RostersRepository), add:

```python
class NextGenStatsRepository:
    """Repository for Next Gen Stats operations."""

    @staticmethod
    def upsert_nextgen_stats(stats: List[Dict[str, Any]]) -> int:
        """Insert or update next gen stats."""
        with get_db() as conn:
            cursor = conn.cursor()
            count = 0

            for stat in stats:
                cursor.execute("""
                    INSERT INTO nextgen_stats
                    (season, week, nflverse_game_id, player_id, player_name, team,
                     position, stat_type, max_speed, avg_speed, max_acceleration,
                     avg_acceleration, routes, targets, rec_yards, rec_tds, attempts,
                     avg_depth_of_target, avg_air_yards, avg_yards_after_catch)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(season, player_id, stat_type) DO UPDATE SET
                        max_speed = excluded.max_speed,
                        avg_speed = excluded.avg_speed,
                        max_acceleration = excluded.max_acceleration,
                        avg_acceleration = excluded.avg_acceleration,
                        routes = excluded.routes,
                        targets = excluded.targets,
                        rec_yards = excluded.rec_yards,
                        rec_tds = excluded.rec_tds,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    stat.get('season'),
                    stat.get('week'),
                    stat.get('nflverse_game_id'),
                    stat.get('player_id'),
                    stat.get('player_name'),
                    stat.get('team'),
                    stat.get('position'),
                    stat.get('stat_type'),
                    stat.get('max_speed'),
                    stat.get('avg_speed'),
                    stat.get('max_acceleration'),
                    stat.get('avg_acceleration'),
                    stat.get('routes'),
                    stat.get('targets'),
                    stat.get('rec_yards'),
                    stat.get('rec_tds'),
                    stat.get('attempts'),
                    stat.get('avg_depth_of_target'),
                    stat.get('avg_air_yards'),
                    stat.get('avg_yards_after_catch')
                ))
                count += 1

            return count

    @staticmethod
    def get_player_nextgen_stats(player_name: str, season: int = 2025) -> List[Dict]:
        """Get next gen stats for a player."""
        with get_db() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM nextgen_stats
                WHERE player_name LIKE ?
                AND season = ?
                ORDER BY stat_type
            """, (f"%{player_name}%", season))

            return [dict(row) for row in cursor.fetchall()]
```

#### Step 3: Add Refresh Method to data_refresh.py

Location: `/home/user/nfl_backend/backend/api/data_refresh.py`

After line 1163 (end of refresh_snap_counts), add:

```python
async def refresh_nextgen_stats(self, force: bool = False) -> Dict:
    """Refresh Next Gen Stats (player tracking data).

    Args:
        force: Force refresh

    Returns:
        Dict with refresh results
    """
    if self.refresh_in_progress.get('nextgen_stats', False):
        return {'status': 'in_progress', 'message': 'NGS refresh already running'}

    self.refresh_in_progress['nextgen_stats'] = True

    try:
        logger.info("Refreshing Next Gen Stats...")

        # Find NGS files (ngs_*.csv)
        ngs_files = list(self.inputs_dir.glob('ngs_*.csv'))

        if not ngs_files:
            return {
                'status': 'warning',
                'message': 'No Next Gen Stats files found',
                'count': 0
            }

        from backend.database.local_db import get_db, NextGenStatsRepository

        total_records = 0
        for file in ngs_files:
            try:
                df = pd.read_csv(file)
                
                # Determine stat type from filename
                stat_type = 'unknown'
                if 'pass' in file.name:
                    stat_type = 'passing'
                elif 'rush' in file.name:
                    stat_type = 'rushing'
                elif 'rec' in file.name:
                    stat_type = 'receiving'

                with get_db() as conn:
                    cursor = conn.cursor()
                    for _, row in df.iterrows():
                        try:
                            # Map column names to our schema
                            cursor.execute("""
                                INSERT OR REPLACE INTO nextgen_stats
                                (season, week, nflverse_game_id, player_id, player_name,
                                 team, position, stat_type, max_speed, avg_speed,
                                 max_acceleration, avg_acceleration, routes, targets,
                                 rec_yards, rec_tds, attempts, avg_depth_of_target,
                                 avg_air_yards, avg_yards_after_catch)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                row.get('season'),
                                row.get('week'),
                                row.get('game_id', row.get('nflverse_game_id')),
                                row.get('player_id'),
                                row.get('player_name', row.get('player')),
                                row.get('team'),
                                row.get('position'),
                                stat_type,
                                row.get('max_speed'),
                                row.get('avg_speed'),
                                row.get('max_accel', row.get('max_acceleration')),
                                row.get('avg_accel', row.get('avg_acceleration')),
                                row.get('routes'),
                                row.get('targets'),
                                row.get('rec_yds', row.get('rec_yards')),
                                row.get('rec_td', row.get('rec_tds')),
                                row.get('att', row.get('attempts')),
                                row.get('avg_depth_of_target', row.get('adot')),
                                row.get('avg_air_yds', row.get('avg_air_yards')),
                                row.get('avg_yac', row.get('avg_yards_after_catch'))
                            ))
                            total_records += 1
                        except:
                            pass  # Skip invalid rows

            except Exception as e:
                logger.warning(f"Error reading NGS file {file}: {e}")

        self.last_refresh['nextgen_stats'] = datetime.now()
        self._save_timestamp('nextgen_stats', total_records)

        return {
            'status': 'success',
            'count': total_records,
            'files_processed': len(ngs_files),
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error refreshing next gen stats: {e}")
        return {'status': 'error', 'message': str(e)}

    finally:
        self.refresh_in_progress['nextgen_stats'] = False
```

#### Step 4: Add to refresh_all() Method

Location: `/home/user/nfl_backend/backend/api/data_refresh.py:L448`

After line 517 (after pfr_advanced), add:

```python
if force or self.is_stale('nextgen_stats'):
    results['nextgen_stats'] = await self.refresh_nextgen_stats(force)
else:
    results['nextgen_stats'] = {'status': 'skipped', 'reason': 'not stale'}
```

#### Step 5: Add Staleness Threshold

Location: `/home/user/nfl_backend/backend/api/data_refresh.py:L224`

Add to STALENESS_THRESHOLDS dict:

```python
'nextgen_stats': 168  # Once per week
```

#### Step 6: Add to _has_data Check

Location: `/home/user/nfl_backend/backend/api/data_refresh.py:L315`

Add to table_map dict:

```python
'nextgen_stats': ('nextgen_stats', 'season'),
```

---

## FIX #2: Missing refresh_results() Method

### Current Status
- **Table exists:** results table (local_db.py:L164)
- **Data source:** Should extract from play_by_play + player_stats
- **Refresh method:** MISSING
- **Impact:** Backtesting/evaluation can't access actual results

### What Needs to Be Done

Add to data_refresh.py after refresh_nextgen_stats():

```python
async def refresh_results(self, force: bool = False) -> Dict:
    """Extract actual game results from play-by-play data.

    Args:
        force: Force refresh

    Returns:
        Dict with refresh results
    """
    if self.refresh_in_progress.get('results', False):
        return {'status': 'in_progress', 'message': 'Results refresh already running'}

    self.refresh_in_progress['results'] = True

    try:
        logger.info("Extracting game results from play-by-play...")

        from backend.database.local_db import get_db, ResultsRepository

        with get_db() as conn:
            cursor = conn.cursor()
            
            # Extract final passing/rushing/receiving stats by game
            cursor.execute("""
                SELECT DISTINCT
                    game_id,
                    passer_player_id as player_id,
                    passer_player_name as player_name,
                    SUM(CASE WHEN play_type = 'pass' THEN yards_gained ELSE 0 END) as pass_yards,
                    SUM(CASE WHEN play_type = 'pass' AND touchdown = 1 THEN 1 ELSE 0 END) as pass_tds
                FROM play_by_play
                WHERE passer_player_id IS NOT NULL
                GROUP BY game_id, passer_player_id
            """)
            pass_results = cursor.fetchall()

            results_list = []
            for row in pass_results:
                if row[1]:  # Has player_id
                    results_list.append({
                        'game_id': row[0],
                        'player_id': row[1],
                        'player_name': row[2],
                        'prop_type': 'passing_yards',
                        'actual_value': row[3] or 0
                    })

            # Insert results
            count = ResultsRepository.insert_results(results_list)

            self.last_refresh['results'] = datetime.now()
            self._save_timestamp('results', count)

            return {
                'status': 'success',
                'count': count,
                'timestamp': datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"Error refreshing results: {e}")
        return {'status': 'error', 'message': str(e)}

    finally:
        self.refresh_in_progress['results'] = False
```

Then add to refresh_all() and STALENESS_THRESHOLDS like above.

---

## FIX #3: Add schedules to fetch_nflverse.py

### Current Status
- **Downloaded:** YES (found in inputs/)
- **In fetch_nflverse.py:** NO
- **Problem:** Source of schedules_2024_2025.csv is unclear

### What Needs to Be Done

Add to fetch_nflverse.py after line 92 (in core datasets):

```python
'schedules': {
    'url': f'{base_url}/schedules/schedules_{year}.csv',
    'output': f'schedules_{year}.csv',
    'description': 'NFL season schedule',
    'core': True
},
```

---

## Summary of Code Changes

| File | Change | Lines | Impact |
|------|--------|-------|--------|
| local_db.py | Add nextgen_stats table + repository | ~80 | Critical |
| data_refresh.py | Add refresh_nextgen_stats() method | ~120 | Critical |
| data_refresh.py | Add refresh_results() method | ~60 | Critical |
| data_refresh.py | Update refresh_all() | ~5 | Critical |
| data_refresh.py | Update STALENESS_THRESHOLDS | ~2 | Critical |
| data_refresh.py | Update _has_data() | ~2 | Critical |
| fetch_nflverse.py | Add schedules download | ~5 | Minor |

**Total Lines of Code:** ~275 lines

**Estimated Effort:** 2-3 hours including testing

