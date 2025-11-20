"""Local SQLite database for NFL betting data.

Append-only tables for historical tracking:
- odds_snapshots - line movement over time
- projections - model projection history
- injuries - injury report tracking
- games - schedule and results
- value_props - betting opportunities found
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

# Database file location
DB_PATH = Path(__file__).parent.parent.parent / "data" / "nfl_betting.db"


def get_connection():
    """Get database connection."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_database():
    """Initialize database with all tables."""
    with get_db() as conn:
        cursor = conn.cursor()

        # Odds snapshots - append-only for line movement tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS odds_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                game_id TEXT,
                player_id TEXT,
                player_name TEXT,
                team TEXT,
                prop_type TEXT,
                line REAL,
                over_odds INTEGER,
                under_odds INTEGER,
                book TEXT DEFAULT 'draftkings',
                week INTEGER,
                season INTEGER
            )
        """)

        # Create indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_odds_player_prop
            ON odds_snapshots(player_name, prop_type, snapshot_time)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_odds_game
            ON odds_snapshots(game_id, snapshot_time)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_odds_snapshot_time
            ON odds_snapshots(snapshot_time)
        """)

        # Projections - append-only for tracking model evolution
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                game_id TEXT,
                player_id TEXT,
                player_name TEXT,
                team TEXT,
                prop_type TEXT,
                projection REAL,
                std_dev REAL,
                confidence_lower REAL,
                confidence_upper REAL,
                hit_prob_over REAL,
                hit_prob_under REAL,
                games_sampled INTEGER,
                model_quality REAL,
                week INTEGER,
                season INTEGER
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_proj_player_prop
            ON projections(player_name, prop_type, generated_at)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_proj_game
            ON projections(game_id, generated_at)
        """)

        # Injuries - append-only for tracking injury status over time
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS injuries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                player_name TEXT,
                player_id TEXT,
                team TEXT,
                position TEXT,
                status TEXT,
                injury_type TEXT,
                week INTEGER,
                season INTEGER
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_injuries_player
            ON injuries(player_name, reported_at)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_injuries_team_status
            ON injuries(team, status, reported_at)
        """)

        # Games/Schedule
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT UNIQUE,
                season INTEGER,
                week INTEGER,
                game_time TIMESTAMP,
                away_team TEXT,
                home_team TEXT,
                away_score INTEGER,
                home_score INTEGER,
                spread REAL,
                total REAL,
                completed BOOLEAN DEFAULT FALSE
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_games_week
            ON games(season, week)
        """)

        # Results - for backtesting projections
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT,
                player_id TEXT,
                player_name TEXT,
                prop_type TEXT,
                actual_value REAL,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(game_id, player_id, prop_type)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_results_player
            ON results(player_name, prop_type)
        """)

        # Value props found - track betting opportunities
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS value_props (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                found_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                game_id TEXT,
                player_name TEXT,
                prop_type TEXT,
                line REAL,
                projection REAL,
                edge_over REAL,
                edge_under REAL,
                recommendation TEXT,
                confidence REAL,
                grade TEXT,
                week INTEGER,
                season INTEGER
            )
        """)

        # Model training runs - track when models were trained
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                season INTEGER,
                model_type TEXT,
                models_trained INTEGER,
                prop_types TEXT,
                training_time_seconds REAL,
                notes TEXT
            )
        """)

        # ============ PLAYER/TEAM STATS TABLES (for general knowledge queries) ============

        # Player season stats - comprehensive stats like ESPN page
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                player_id TEXT,
                player_name TEXT,
                team TEXT,
                position TEXT,
                season INTEGER,
                week INTEGER,
                -- General
                games_played INTEGER,
                -- Passing
                pass_attempts INTEGER,
                pass_completions INTEGER,
                pass_yards INTEGER,
                pass_tds INTEGER,
                interceptions INTEGER,
                sacks INTEGER,
                sack_yards INTEGER,
                pass_rating REAL,
                -- Rushing
                rush_attempts INTEGER,
                rush_yards INTEGER,
                rush_tds INTEGER,
                rush_yards_per_attempt REAL,
                -- Receiving
                targets INTEGER,
                receptions INTEGER,
                rec_yards INTEGER,
                rec_tds INTEGER,
                yards_per_reception REAL,
                -- Fantasy
                fantasy_points REAL,
                fantasy_points_ppr REAL,
                -- Advanced
                air_yards REAL,
                yards_after_catch REAL,
                epa REAL,
                cpoe REAL,
                -- Snap data
                snap_pct REAL,
                route_participation REAL,
                UNIQUE(player_id, season, week)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_player_stats_player
            ON player_stats(player_id, season, week)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_player_stats_name
            ON player_stats(player_name, season)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_player_stats_team
            ON player_stats(team, season, position)
        """)

        # Team stats - team-level aggregates
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS team_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                team TEXT,
                season INTEGER,
                week INTEGER,
                -- Record
                wins INTEGER,
                losses INTEGER,
                ties INTEGER,
                -- Offense
                points_scored INTEGER,
                total_yards INTEGER,
                pass_yards INTEGER,
                rush_yards INTEGER,
                turnovers INTEGER,
                -- Defense
                points_allowed INTEGER,
                yards_allowed INTEGER,
                pass_yards_allowed INTEGER,
                rush_yards_allowed INTEGER,
                takeaways INTEGER,
                sacks INTEGER,
                -- Efficiency
                third_down_pct REAL,
                red_zone_pct REAL,
                time_of_possession TEXT,
                -- Rankings (within league)
                offense_rank INTEGER,
                defense_rank INTEGER,
                UNIQUE(team, season, week)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_team_stats_team
            ON team_stats(team, season)
        """)

        # Rosters - current team rosters
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rosters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                player_id TEXT,
                player_name TEXT,
                team TEXT,
                position TEXT,
                jersey_number INTEGER,
                status TEXT,
                height TEXT,
                weight INTEGER,
                birth_date TEXT,
                college TEXT,
                years_exp INTEGER,
                season INTEGER,
                week INTEGER,
                depth_chart_position INTEGER,
                UNIQUE(player_id, season, week)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rosters_team
            ON rosters(team, season, week)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rosters_player
            ON rosters(player_name, season)
        """)

        # Season schedules - full season schedule
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schedules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT UNIQUE,
                season INTEGER,
                week INTEGER,
                game_type TEXT,
                gameday TEXT,
                weekday TEXT,
                gametime TEXT,
                away_team TEXT,
                away_score INTEGER,
                home_team TEXT,
                home_score INTEGER,
                location TEXT,
                roof TEXT,
                surface TEXT,
                temp INTEGER,
                wind INTEGER,
                away_rest INTEGER,
                home_rest INTEGER,
                away_moneyline INTEGER,
                home_moneyline INTEGER,
                spread_line REAL,
                away_spread_odds INTEGER,
                home_spread_odds INTEGER,
                total_line REAL,
                under_odds INTEGER,
                over_odds INTEGER,
                div_game BOOLEAN,
                overtime BOOLEAN,
                old_game_id TEXT,
                gsis INTEGER,
                nfl_detail_id TEXT,
                pfr TEXT,
                pff INTEGER,
                espn INTEGER,
                ftn INTEGER,
                away_coach TEXT,
                home_coach TEXT,
                stadium_id TEXT,
                stadium TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_schedules_season_week
            ON schedules(season, week)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_schedules_teams
            ON schedules(away_team, home_team, season)
        """)

        print(f"Database initialized at {DB_PATH}")
        return str(DB_PATH)


# ============ REPOSITORY CLASSES ============

class OddsRepository:
    """Repository for odds data operations."""

    @staticmethod
    def insert_snapshot(odds_data: List[Dict[str, Any]], week: int, season: int) -> int:
        """Insert odds snapshot (appends, doesn't overwrite)."""
        with get_db() as conn:
            cursor = conn.cursor()
            inserted = 0

            for odds in odds_data:
                cursor.execute("""
                    INSERT INTO odds_snapshots
                    (game_id, player_id, player_name, team, prop_type,
                     line, over_odds, under_odds, book, week, season)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    odds.get('game_id'),
                    odds.get('player_id'),
                    odds.get('player_name'),
                    odds.get('team'),
                    odds.get('prop_type'),
                    odds.get('line'),
                    odds.get('over_odds'),
                    odds.get('under_odds'),
                    odds.get('book', 'draftkings'),
                    week,
                    season
                ))
                inserted += 1

            return inserted

    @staticmethod
    def get_latest_odds(game_id: Optional[str] = None,
                        player_name: Optional[str] = None,
                        prop_type: Optional[str] = None) -> List[Dict]:
        """Get latest odds snapshot."""
        with get_db() as conn:
            cursor = conn.cursor()

            # Get the most recent snapshot time
            cursor.execute("SELECT MAX(snapshot_time) FROM odds_snapshots")
            latest_time = cursor.fetchone()[0]

            if not latest_time:
                return []

            query = "SELECT * FROM odds_snapshots WHERE snapshot_time = ?"
            params = [latest_time]

            if game_id:
                query += " AND game_id = ?"
                params.append(game_id)
            if player_name:
                query += " AND player_name LIKE ?"
                params.append(f"%{player_name}%")
            if prop_type:
                query += " AND prop_type = ?"
                params.append(prop_type)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def get_line_movement(player_name: str, prop_type: str,
                          days: int = 7) -> List[Dict]:
        """Get line movement for a player/prop over time."""
        with get_db() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT snapshot_time, line, over_odds, under_odds
                FROM odds_snapshots
                WHERE player_name LIKE ?
                AND prop_type = ?
                AND snapshot_time >= datetime('now', ?)
                ORDER BY snapshot_time ASC
            """, (f"%{player_name}%", prop_type, f"-{days} days"))

            return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def get_hot_movers(min_movement: float = 1.5, hours: int = 48) -> List[Dict]:
        """Find props with significant line movement."""
        with get_db() as conn:
            cursor = conn.cursor()

            # Get opening and current lines for recent props
            cursor.execute("""
                WITH ranked AS (
                    SELECT
                        player_name,
                        prop_type,
                        game_id,
                        line,
                        snapshot_time,
                        ROW_NUMBER() OVER (
                            PARTITION BY player_name, prop_type
                            ORDER BY snapshot_time ASC
                        ) as first_rank,
                        ROW_NUMBER() OVER (
                            PARTITION BY player_name, prop_type
                            ORDER BY snapshot_time DESC
                        ) as last_rank
                    FROM odds_snapshots
                    WHERE snapshot_time >= datetime('now', ?)
                )
                SELECT
                    f.player_name,
                    f.prop_type,
                    f.game_id,
                    f.line as opening_line,
                    l.line as current_line,
                    (l.line - f.line) as movement
                FROM ranked f
                JOIN ranked l ON f.player_name = l.player_name
                    AND f.prop_type = l.prop_type
                WHERE f.first_rank = 1 AND l.last_rank = 1
                AND ABS(l.line - f.line) >= ?
                ORDER BY ABS(l.line - f.line) DESC
            """, (f"-{hours} hours", min_movement))

            return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def get_snapshot_count() -> int:
        """Get total number of snapshots."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT snapshot_time) FROM odds_snapshots")
            return cursor.fetchone()[0]


class ProjectionsRepository:
    """Repository for projections data operations."""

    @staticmethod
    def insert_projections(projections: List[Dict[str, Any]],
                           week: int, season: int) -> int:
        """Insert projections (appends for history)."""
        with get_db() as conn:
            cursor = conn.cursor()
            inserted = 0

            for proj in projections:
                cursor.execute("""
                    INSERT INTO projections
                    (game_id, player_id, player_name, team, prop_type,
                     projection, std_dev, confidence_lower, confidence_upper,
                     hit_prob_over, hit_prob_under, games_sampled,
                     model_quality, week, season)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    proj.get('game_id'),
                    proj.get('player_id'),
                    proj.get('player_name'),
                    proj.get('team'),
                    proj.get('prop_type'),
                    proj.get('projection'),
                    proj.get('std_dev'),
                    proj.get('confidence_lower'),
                    proj.get('confidence_upper'),
                    proj.get('hit_prob_over'),
                    proj.get('hit_prob_under'),
                    proj.get('games_sampled'),
                    proj.get('model_quality'),
                    week,
                    season
                ))
                inserted += 1

            return inserted

    @staticmethod
    def get_latest_projections(game_id: Optional[str] = None,
                                player_name: Optional[str] = None,
                                prop_type: Optional[str] = None) -> List[Dict]:
        """Get most recent projections."""
        with get_db() as conn:
            cursor = conn.cursor()

            # Get latest generation time
            cursor.execute("SELECT MAX(generated_at) FROM projections")
            latest = cursor.fetchone()[0]

            if not latest:
                return []

            query = "SELECT * FROM projections WHERE generated_at = ?"
            params = [latest]

            if game_id:
                query += " AND game_id = ?"
                params.append(game_id)
            if player_name:
                query += " AND player_name LIKE ?"
                params.append(f"%{player_name}%")
            if prop_type:
                query += " AND prop_type = ?"
                params.append(prop_type)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def get_projection_history(player_name: str, prop_type: str,
                                limit: int = 10) -> List[Dict]:
        """Get projection history for a player/prop."""
        with get_db() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT generated_at, projection, std_dev,
                       hit_prob_over, hit_prob_under, game_id
                FROM projections
                WHERE player_name LIKE ?
                AND prop_type = ?
                ORDER BY generated_at DESC
                LIMIT ?
            """, (f"%{player_name}%", prop_type, limit))

            return [dict(row) for row in cursor.fetchall()]


class InjuriesRepository:
    """Repository for injury data operations."""

    @staticmethod
    def insert_injuries(injuries: List[Dict[str, Any]],
                        week: int, season: int) -> int:
        """Insert injury reports (appends for tracking)."""
        with get_db() as conn:
            cursor = conn.cursor()
            inserted = 0

            for injury in injuries:
                cursor.execute("""
                    INSERT INTO injuries
                    (player_name, player_id, team, position, status,
                     injury_type, week, season)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    injury.get('player_name'),
                    injury.get('player_id'),
                    injury.get('team'),
                    injury.get('position'),
                    injury.get('status'),
                    injury.get('injury_type'),
                    week,
                    season
                ))
                inserted += 1

            return inserted

    @staticmethod
    def get_latest_injuries(team: Optional[str] = None,
                            status: Optional[str] = None) -> List[Dict]:
        """Get most recent injury reports."""
        with get_db() as conn:
            cursor = conn.cursor()

            # Get latest report time
            cursor.execute("SELECT MAX(reported_at) FROM injuries")
            latest = cursor.fetchone()[0]

            if not latest:
                return []

            query = "SELECT * FROM injuries WHERE reported_at = ?"
            params = [latest]

            if team:
                query += " AND team = ?"
                params.append(team)
            if status:
                query += " AND status = ?"
                params.append(status)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def get_injury_history(player_name: str, weeks: int = 4) -> List[Dict]:
        """Get injury history for a player."""
        with get_db() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT reported_at, status, injury_type, week
                FROM injuries
                WHERE player_name LIKE ?
                AND reported_at >= datetime('now', ?)
                ORDER BY reported_at DESC
            """, (f"%{player_name}%", f"-{weeks * 7} days"))

            return [dict(row) for row in cursor.fetchall()]


class GamesRepository:
    """Repository for games/schedule operations."""

    @staticmethod
    def upsert_games(games: List[Dict[str, Any]]) -> int:
        """Insert or update games."""
        with get_db() as conn:
            cursor = conn.cursor()
            count = 0

            for game in games:
                cursor.execute("""
                    INSERT INTO games
                    (game_id, season, week, game_time, away_team, home_team,
                     away_score, home_score, spread, total, completed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(game_id) DO UPDATE SET
                        away_score = excluded.away_score,
                        home_score = excluded.home_score,
                        completed = excluded.completed
                """, (
                    game.get('game_id'),
                    game.get('season'),
                    game.get('week'),
                    game.get('game_time'),
                    game.get('away_team'),
                    game.get('home_team'),
                    game.get('away_score'),
                    game.get('home_score'),
                    game.get('spread'),
                    game.get('total'),
                    game.get('completed', False)
                ))
                count += 1

            return count

    @staticmethod
    def get_games(week: Optional[int] = None, season: int = 2024) -> List[Dict]:
        """Get games for a week/season."""
        with get_db() as conn:
            cursor = conn.cursor()

            if week:
                cursor.execute("""
                    SELECT * FROM games
                    WHERE season = ? AND week = ?
                    ORDER BY game_time
                """, (season, week))
            else:
                cursor.execute("""
                    SELECT * FROM games
                    WHERE season = ?
                    ORDER BY week, game_time
                """, (season,))

            return [dict(row) for row in cursor.fetchall()]


class ResultsRepository:
    """Repository for actual results (backtesting)."""

    @staticmethod
    def insert_results(results: List[Dict[str, Any]]) -> int:
        """Insert actual results."""
        with get_db() as conn:
            cursor = conn.cursor()
            count = 0

            for result in results:
                cursor.execute("""
                    INSERT INTO results
                    (game_id, player_id, player_name, prop_type, actual_value)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(game_id, player_id, prop_type) DO UPDATE SET
                        actual_value = excluded.actual_value
                """, (
                    result.get('game_id'),
                    result.get('player_id'),
                    result.get('player_name'),
                    result.get('prop_type'),
                    result.get('actual_value')
                ))
                count += 1

            return count

    @staticmethod
    def get_results(game_id: Optional[str] = None,
                    player_name: Optional[str] = None) -> List[Dict]:
        """Get results for backtesting."""
        with get_db() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM results WHERE 1=1"
            params = []

            if game_id:
                query += " AND game_id = ?"
                params.append(game_id)
            if player_name:
                query += " AND player_name LIKE ?"
                params.append(f"%{player_name}%")

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]


class ValuePropsRepository:
    """Repository for tracking value props found."""

    @staticmethod
    def insert_value_props(props: List[Dict[str, Any]],
                           week: int, season: int) -> int:
        """Insert value props found."""
        with get_db() as conn:
            cursor = conn.cursor()
            count = 0

            for prop in props:
                cursor.execute("""
                    INSERT INTO value_props
                    (game_id, player_name, prop_type, line, projection,
                     edge_over, edge_under, recommendation, confidence,
                     grade, week, season)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    prop.get('game_id'),
                    prop.get('player_name'),
                    prop.get('prop_type'),
                    prop.get('line'),
                    prop.get('projection'),
                    prop.get('edge_over'),
                    prop.get('edge_under'),
                    prop.get('recommendation'),
                    prop.get('confidence'),
                    prop.get('grade'),
                    week,
                    season
                ))
                count += 1

            return count

    @staticmethod
    def get_value_props_history(days: int = 7,
                                min_edge: float = 0) -> List[Dict]:
        """Get historical value props."""
        with get_db() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM value_props
                WHERE found_at >= datetime('now', ?)
                AND (edge_over >= ? OR edge_under >= ?)
                ORDER BY found_at DESC
            """, (f"-{days} days", min_edge, min_edge))

            return [dict(row) for row in cursor.fetchall()]


class ModelRunsRepository:
    """Repository for tracking model training runs."""

    @staticmethod
    def log_run(season: int, model_type: str, models_trained: int,
                prop_types: List[str], training_time: float,
                notes: str = "") -> int:
        """Log a model training run."""
        with get_db() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO model_runs
                (season, model_type, models_trained, prop_types,
                 training_time_seconds, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                season,
                model_type,
                models_trained,
                ",".join(prop_types),
                training_time,
                notes
            ))

            return cursor.lastrowid

    @staticmethod
    def get_latest_run() -> Optional[Dict]:
        """Get most recent training run."""
        with get_db() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM model_runs
                ORDER BY run_time DESC
                LIMIT 1
            """)

            row = cursor.fetchone()
            return dict(row) if row else None

    @staticmethod
    def get_all_runs(limit: int = 20) -> List[Dict]:
        """Get recent training runs."""
        with get_db() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM model_runs
                ORDER BY run_time DESC
                LIMIT ?
            """, (limit,))

            return [dict(row) for row in cursor.fetchall()]


# ============ STATS REPOSITORY CLASSES (for general knowledge queries) ============

class PlayerStatsRepository:
    """Repository for player stats operations."""

    @staticmethod
    def upsert_player_stats(stats: List[Dict[str, Any]]) -> int:
        """Insert or update player stats."""
        with get_db() as conn:
            cursor = conn.cursor()
            count = 0

            for stat in stats:
                cursor.execute("""
                    INSERT INTO player_stats
                    (player_id, player_name, team, position, season, week,
                     games_played, pass_attempts, pass_completions, pass_yards,
                     pass_tds, interceptions, sacks, sack_yards, pass_rating,
                     rush_attempts, rush_yards, rush_tds, rush_yards_per_attempt,
                     targets, receptions, rec_yards, rec_tds, yards_per_reception,
                     fantasy_points, fantasy_points_ppr, air_yards, yards_after_catch,
                     epa, cpoe, snap_pct, route_participation)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(player_id, season, week) DO UPDATE SET
                        player_name = excluded.player_name,
                        team = excluded.team,
                        position = excluded.position,
                        games_played = excluded.games_played,
                        pass_attempts = excluded.pass_attempts,
                        pass_completions = excluded.pass_completions,
                        pass_yards = excluded.pass_yards,
                        pass_tds = excluded.pass_tds,
                        interceptions = excluded.interceptions,
                        sacks = excluded.sacks,
                        sack_yards = excluded.sack_yards,
                        pass_rating = excluded.pass_rating,
                        rush_attempts = excluded.rush_attempts,
                        rush_yards = excluded.rush_yards,
                        rush_tds = excluded.rush_tds,
                        rush_yards_per_attempt = excluded.rush_yards_per_attempt,
                        targets = excluded.targets,
                        receptions = excluded.receptions,
                        rec_yards = excluded.rec_yards,
                        rec_tds = excluded.rec_tds,
                        yards_per_reception = excluded.yards_per_reception,
                        fantasy_points = excluded.fantasy_points,
                        fantasy_points_ppr = excluded.fantasy_points_ppr,
                        air_yards = excluded.air_yards,
                        yards_after_catch = excluded.yards_after_catch,
                        epa = excluded.epa,
                        cpoe = excluded.cpoe,
                        snap_pct = excluded.snap_pct,
                        route_participation = excluded.route_participation,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    stat.get('player_id'),
                    stat.get('player_name'),
                    stat.get('team'),
                    stat.get('position'),
                    stat.get('season'),
                    stat.get('week'),
                    stat.get('games_played'),
                    stat.get('pass_attempts'),
                    stat.get('pass_completions'),
                    stat.get('pass_yards'),
                    stat.get('pass_tds'),
                    stat.get('interceptions'),
                    stat.get('sacks'),
                    stat.get('sack_yards'),
                    stat.get('pass_rating'),
                    stat.get('rush_attempts'),
                    stat.get('rush_yards'),
                    stat.get('rush_tds'),
                    stat.get('rush_yards_per_attempt'),
                    stat.get('targets'),
                    stat.get('receptions'),
                    stat.get('rec_yards'),
                    stat.get('rec_tds'),
                    stat.get('yards_per_reception'),
                    stat.get('fantasy_points'),
                    stat.get('fantasy_points_ppr'),
                    stat.get('air_yards'),
                    stat.get('yards_after_catch'),
                    stat.get('epa'),
                    stat.get('cpoe'),
                    stat.get('snap_pct'),
                    stat.get('route_participation')
                ))
                count += 1

            return count

    @staticmethod
    def get_player_stats(player_name: str, season: int = 2025) -> List[Dict]:
        """Get all weekly stats for a player (like ESPN player page)."""
        with get_db() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM player_stats
                WHERE player_name LIKE ?
                AND season = ?
                ORDER BY week DESC
            """, (f"%{player_name}%", season))

            return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def get_player_season_totals(player_name: str, season: int = 2025) -> Optional[Dict]:
        """Get season totals for a player."""
        with get_db() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    player_name,
                    team,
                    position,
                    season,
                    COUNT(DISTINCT week) as games_played,
                    SUM(pass_attempts) as pass_attempts,
                    SUM(pass_completions) as pass_completions,
                    SUM(pass_yards) as pass_yards,
                    SUM(pass_tds) as pass_tds,
                    SUM(interceptions) as interceptions,
                    SUM(sacks) as sacks,
                    SUM(rush_attempts) as rush_attempts,
                    SUM(rush_yards) as rush_yards,
                    SUM(rush_tds) as rush_tds,
                    SUM(targets) as targets,
                    SUM(receptions) as receptions,
                    SUM(rec_yards) as rec_yards,
                    SUM(rec_tds) as rec_tds,
                    SUM(fantasy_points) as fantasy_points,
                    SUM(fantasy_points_ppr) as fantasy_points_ppr,
                    AVG(epa) as avg_epa,
                    AVG(cpoe) as avg_cpoe,
                    AVG(snap_pct) as avg_snap_pct
                FROM player_stats
                WHERE player_name LIKE ?
                AND season = ?
                GROUP BY player_name, team, position, season
            """, (f"%{player_name}%", season))

            row = cursor.fetchone()
            return dict(row) if row else None

    @staticmethod
    def get_team_players(team: str, season: int = 2025,
                         position: Optional[str] = None) -> List[Dict]:
        """Get all players on a team with their season totals."""
        with get_db() as conn:
            cursor = conn.cursor()

            query = """
                SELECT
                    player_name,
                    team,
                    position,
                    COUNT(DISTINCT week) as games,
                    SUM(pass_yards) as pass_yards,
                    SUM(pass_tds) as pass_tds,
                    SUM(rush_yards) as rush_yards,
                    SUM(rush_tds) as rush_tds,
                    SUM(rec_yards) as rec_yards,
                    SUM(rec_tds) as rec_tds,
                    SUM(fantasy_points_ppr) as fantasy_ppr
                FROM player_stats
                WHERE team = ?
                AND season = ?
            """
            params = [team, season]

            if position:
                query += " AND position = ?"
                params.append(position)

            query += " GROUP BY player_name ORDER BY fantasy_ppr DESC"

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def get_league_leaders(stat_type: str, season: int = 2025, limit: int = 20) -> List[Dict]:
        """Get league leaders for a specific stat."""
        stat_columns = {
            'passing_yards': 'SUM(pass_yards)',
            'passing_tds': 'SUM(pass_tds)',
            'rushing_yards': 'SUM(rush_yards)',
            'rushing_tds': 'SUM(rush_tds)',
            'receiving_yards': 'SUM(rec_yards)',
            'receiving_tds': 'SUM(rec_tds)',
            'receptions': 'SUM(receptions)',
            'fantasy_ppr': 'SUM(fantasy_points_ppr)',
            'fantasy': 'SUM(fantasy_points)'
        }

        if stat_type not in stat_columns:
            return []

        with get_db() as conn:
            cursor = conn.cursor()

            cursor.execute(f"""
                SELECT
                    player_name,
                    team,
                    position,
                    {stat_columns[stat_type]} as value
                FROM player_stats
                WHERE season = ?
                GROUP BY player_name
                HAVING value > 0
                ORDER BY value DESC
                LIMIT ?
            """, (season, limit))

            return [dict(row) for row in cursor.fetchall()]


class TeamStatsRepository:
    """Repository for team stats operations."""

    @staticmethod
    def upsert_team_stats(stats: List[Dict[str, Any]]) -> int:
        """Insert or update team stats."""
        with get_db() as conn:
            cursor = conn.cursor()
            count = 0

            for stat in stats:
                cursor.execute("""
                    INSERT INTO team_stats
                    (team, season, week, wins, losses, ties,
                     points_scored, total_yards, pass_yards, rush_yards, turnovers,
                     points_allowed, yards_allowed, pass_yards_allowed, rush_yards_allowed,
                     takeaways, sacks, third_down_pct, red_zone_pct, time_of_possession,
                     offense_rank, defense_rank)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(team, season, week) DO UPDATE SET
                        wins = excluded.wins,
                        losses = excluded.losses,
                        ties = excluded.ties,
                        points_scored = excluded.points_scored,
                        total_yards = excluded.total_yards,
                        pass_yards = excluded.pass_yards,
                        rush_yards = excluded.rush_yards,
                        turnovers = excluded.turnovers,
                        points_allowed = excluded.points_allowed,
                        yards_allowed = excluded.yards_allowed,
                        pass_yards_allowed = excluded.pass_yards_allowed,
                        rush_yards_allowed = excluded.rush_yards_allowed,
                        takeaways = excluded.takeaways,
                        sacks = excluded.sacks,
                        third_down_pct = excluded.third_down_pct,
                        red_zone_pct = excluded.red_zone_pct,
                        time_of_possession = excluded.time_of_possession,
                        offense_rank = excluded.offense_rank,
                        defense_rank = excluded.defense_rank,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    stat.get('team'),
                    stat.get('season'),
                    stat.get('week'),
                    stat.get('wins'),
                    stat.get('losses'),
                    stat.get('ties'),
                    stat.get('points_scored'),
                    stat.get('total_yards'),
                    stat.get('pass_yards'),
                    stat.get('rush_yards'),
                    stat.get('turnovers'),
                    stat.get('points_allowed'),
                    stat.get('yards_allowed'),
                    stat.get('pass_yards_allowed'),
                    stat.get('rush_yards_allowed'),
                    stat.get('takeaways'),
                    stat.get('sacks'),
                    stat.get('third_down_pct'),
                    stat.get('red_zone_pct'),
                    stat.get('time_of_possession'),
                    stat.get('offense_rank'),
                    stat.get('defense_rank')
                ))
                count += 1

            return count

    @staticmethod
    def get_team_stats(team: str, season: int = 2025) -> Optional[Dict]:
        """Get latest team stats."""
        with get_db() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM team_stats
                WHERE team = ?
                AND season = ?
                ORDER BY week DESC
                LIMIT 1
            """, (team, season))

            row = cursor.fetchone()
            return dict(row) if row else None

    @staticmethod
    def get_all_teams(season: int = 2025) -> List[Dict]:
        """Get all team stats for rankings."""
        with get_db() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT t1.*
                FROM team_stats t1
                INNER JOIN (
                    SELECT team, MAX(week) as max_week
                    FROM team_stats
                    WHERE season = ?
                    GROUP BY team
                ) t2 ON t1.team = t2.team AND t1.week = t2.max_week
                WHERE t1.season = ?
                ORDER BY t1.wins DESC, t1.points_scored DESC
            """, (season, season))

            return [dict(row) for row in cursor.fetchall()]


class RostersRepository:
    """Repository for roster operations."""

    @staticmethod
    def upsert_rosters(rosters: List[Dict[str, Any]]) -> int:
        """Insert or update rosters."""
        with get_db() as conn:
            cursor = conn.cursor()
            count = 0

            for player in rosters:
                cursor.execute("""
                    INSERT INTO rosters
                    (player_id, player_name, team, position, jersey_number, status,
                     height, weight, birth_date, college, years_exp,
                     season, week, depth_chart_position)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(player_id, season, week) DO UPDATE SET
                        player_name = excluded.player_name,
                        team = excluded.team,
                        position = excluded.position,
                        jersey_number = excluded.jersey_number,
                        status = excluded.status,
                        height = excluded.height,
                        weight = excluded.weight,
                        depth_chart_position = excluded.depth_chart_position,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    player.get('player_id'),
                    player.get('player_name'),
                    player.get('team'),
                    player.get('position'),
                    player.get('jersey_number'),
                    player.get('status'),
                    player.get('height'),
                    player.get('weight'),
                    player.get('birth_date'),
                    player.get('college'),
                    player.get('years_exp'),
                    player.get('season'),
                    player.get('week'),
                    player.get('depth_chart_position')
                ))
                count += 1

            return count

    @staticmethod
    def get_team_roster(team: str, season: int = 2025, week: Optional[int] = None) -> List[Dict]:
        """Get team roster."""
        with get_db() as conn:
            cursor = conn.cursor()

            if week:
                cursor.execute("""
                    SELECT * FROM rosters
                    WHERE team = ?
                    AND season = ?
                    AND week = ?
                    ORDER BY position, depth_chart_position
                """, (team, season, week))
            else:
                # Get most recent week
                cursor.execute("""
                    SELECT * FROM rosters
                    WHERE team = ?
                    AND season = ?
                    AND week = (SELECT MAX(week) FROM rosters WHERE team = ? AND season = ?)
                    ORDER BY position, depth_chart_position
                """, (team, season, team, season))

            return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def get_player_info(player_name: str) -> Optional[Dict]:
        """Get player info (bio, team, position)."""
        with get_db() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM rosters
                WHERE player_name LIKE ?
                ORDER BY season DESC, week DESC
                LIMIT 1
            """, (f"%{player_name}%",))

            row = cursor.fetchone()
            return dict(row) if row else None


class SchedulesRepository:
    """Repository for schedule operations."""

    @staticmethod
    def upsert_schedules(schedules: List[Dict[str, Any]]) -> int:
        """Insert or update schedules from nflverse."""
        with get_db() as conn:
            cursor = conn.cursor()
            count = 0

            for game in schedules:
                cursor.execute("""
                    INSERT INTO schedules
                    (game_id, season, week, game_type, gameday, weekday, gametime,
                     away_team, away_score, home_team, home_score, location, roof, surface,
                     temp, wind, away_rest, home_rest, away_moneyline, home_moneyline,
                     spread_line, away_spread_odds, home_spread_odds, total_line,
                     under_odds, over_odds, div_game, overtime, old_game_id, gsis,
                     nfl_detail_id, pfr, pff, espn, ftn, away_coach, home_coach,
                     stadium_id, stadium)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(game_id) DO UPDATE SET
                        away_score = excluded.away_score,
                        home_score = excluded.home_score,
                        temp = excluded.temp,
                        wind = excluded.wind,
                        overtime = excluded.overtime
                """, (
                    game.get('game_id'),
                    game.get('season'),
                    game.get('week'),
                    game.get('game_type'),
                    game.get('gameday'),
                    game.get('weekday'),
                    game.get('gametime'),
                    game.get('away_team'),
                    game.get('away_score'),
                    game.get('home_team'),
                    game.get('home_score'),
                    game.get('location'),
                    game.get('roof'),
                    game.get('surface'),
                    game.get('temp'),
                    game.get('wind'),
                    game.get('away_rest'),
                    game.get('home_rest'),
                    game.get('away_moneyline'),
                    game.get('home_moneyline'),
                    game.get('spread_line'),
                    game.get('away_spread_odds'),
                    game.get('home_spread_odds'),
                    game.get('total_line'),
                    game.get('under_odds'),
                    game.get('over_odds'),
                    game.get('div_game'),
                    game.get('overtime'),
                    game.get('old_game_id'),
                    game.get('gsis'),
                    game.get('nfl_detail_id'),
                    game.get('pfr'),
                    game.get('pff'),
                    game.get('espn'),
                    game.get('ftn'),
                    game.get('away_coach'),
                    game.get('home_coach'),
                    game.get('stadium_id'),
                    game.get('stadium')
                ))
                count += 1

            return count

    @staticmethod
    def get_schedule(season: int = 2025, week: Optional[int] = None) -> List[Dict]:
        """Get schedule for season/week."""
        with get_db() as conn:
            cursor = conn.cursor()

            if week:
                cursor.execute("""
                    SELECT * FROM schedules
                    WHERE season = ? AND week = ?
                    ORDER BY gameday, gametime
                """, (season, week))
            else:
                cursor.execute("""
                    SELECT * FROM schedules
                    WHERE season = ?
                    ORDER BY week, gameday, gametime
                """, (season,))

            return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def get_team_schedule(team: str, season: int = 2025) -> List[Dict]:
        """Get schedule for a specific team."""
        with get_db() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM schedules
                WHERE season = ?
                AND (home_team = ? OR away_team = ?)
                ORDER BY week
            """, (season, team, team))

            return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def get_upcoming_games(season: int = 2025, week: int = 12) -> List[Dict]:
        """Get upcoming games (not yet played)."""
        with get_db() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM schedules
                WHERE season = ?
                AND week >= ?
                AND (away_score IS NULL OR home_score IS NULL)
                ORDER BY week, gameday, gametime
            """, (season, week))

            return [dict(row) for row in cursor.fetchall()]


# Database status helper
def get_database_status() -> Dict[str, Any]:
    """Get overview of database contents."""
    with get_db() as conn:
        cursor = conn.cursor()

        stats = {}

        # Odds snapshots
        cursor.execute("SELECT COUNT(*) FROM odds_snapshots")
        stats['odds_records'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT snapshot_time) FROM odds_snapshots")
        stats['odds_snapshots'] = cursor.fetchone()[0]

        cursor.execute("SELECT MAX(snapshot_time) FROM odds_snapshots")
        stats['odds_last_update'] = cursor.fetchone()[0]

        # Projections
        cursor.execute("SELECT COUNT(*) FROM projections")
        stats['projection_records'] = cursor.fetchone()[0]

        cursor.execute("SELECT MAX(generated_at) FROM projections")
        stats['projections_last_update'] = cursor.fetchone()[0]

        # Injuries
        cursor.execute("SELECT COUNT(*) FROM injuries")
        stats['injury_records'] = cursor.fetchone()[0]

        cursor.execute("SELECT MAX(reported_at) FROM injuries")
        stats['injuries_last_update'] = cursor.fetchone()[0]

        # Games
        cursor.execute("SELECT COUNT(*) FROM games")
        stats['games'] = cursor.fetchone()[0]

        # Results
        cursor.execute("SELECT COUNT(*) FROM results")
        stats['results'] = cursor.fetchone()[0]

        # Value props
        cursor.execute("SELECT COUNT(*) FROM value_props")
        stats['value_props_found'] = cursor.fetchone()[0]

        # Model runs
        cursor.execute("SELECT COUNT(*) FROM model_runs")
        stats['training_runs'] = cursor.fetchone()[0]

        # Player stats
        cursor.execute("SELECT COUNT(*) FROM player_stats")
        stats['player_stats_records'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT player_name) FROM player_stats")
        stats['players_with_stats'] = cursor.fetchone()[0]

        cursor.execute("SELECT MAX(updated_at) FROM player_stats")
        stats['player_stats_last_update'] = cursor.fetchone()[0]

        # Team stats
        cursor.execute("SELECT COUNT(*) FROM team_stats")
        stats['team_stats_records'] = cursor.fetchone()[0]

        cursor.execute("SELECT MAX(updated_at) FROM team_stats")
        stats['team_stats_last_update'] = cursor.fetchone()[0]

        # Rosters
        cursor.execute("SELECT COUNT(*) FROM rosters")
        stats['roster_records'] = cursor.fetchone()[0]

        cursor.execute("SELECT MAX(updated_at) FROM rosters")
        stats['rosters_last_update'] = cursor.fetchone()[0]

        # Schedules
        cursor.execute("SELECT COUNT(*) FROM schedules")
        stats['schedule_games'] = cursor.fetchone()[0]

        return stats


# Initialize database on import
if __name__ == "__main__":
    init_database()
