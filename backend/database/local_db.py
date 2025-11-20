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

        return stats


# Initialize database on import
if __name__ == "__main__":
    init_database()
