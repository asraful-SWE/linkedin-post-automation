"""
Database Migrations - Safely adds new columns and tables to existing DB.
Always backward compatible - uses IF NOT EXISTS and ALTER TABLE IF NOT EXISTS pattern.

Usage
-----
Call ``run_migrations(db_path)`` once on application startup **after**
``DatabaseManager`` has already initialised the base schema::

    from database.models import DatabaseManager
    from database.migrations import run_migrations

    db = DatabaseManager()
    run_migrations(db.db_path)

Every migration is idempotent: running this function multiple times on the
same database is completely safe.

Migration log format
--------------------
Each migration emits one structured log line::

    migration=N|status=applied/skipped|description=<human-readable text>
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Callable, List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal type alias
# ---------------------------------------------------------------------------

# A migration function receives an *open* connection and returns a 2-tuple:
#   (status, description)
# where status ∈ {"applied", "skipped"} and description is a short sentence
# that will be embedded in the structured log line.
_MigrationFn = Callable[[sqlite3.Connection], Tuple[str, str]]


# ---------------------------------------------------------------------------
# Migration 1 – posts.post_goal
# ---------------------------------------------------------------------------


def _m01_add_post_goal(conn: sqlite3.Connection) -> Tuple[str, str]:
    """Add post_goal TEXT column to posts (default: 'educational')."""
    description = "add post_goal TEXT DEFAULT 'educational' to posts"
    try:
        conn.execute(
            "ALTER TABLE posts ADD COLUMN post_goal TEXT DEFAULT 'educational'"
        )
        conn.commit()
        return "applied", description
    except sqlite3.OperationalError as exc:
        if "duplicate column name" in str(exc).lower():
            return "skipped", description + " — column already exists"
        raise


# ---------------------------------------------------------------------------
# Migration 2 – posts.content_score
# ---------------------------------------------------------------------------


def _m02_add_content_score(conn: sqlite3.Connection) -> Tuple[str, str]:
    """Add content_score REAL column to posts (default: 0.0)."""
    description = "add content_score REAL DEFAULT 0.0 to posts"
    try:
        conn.execute("ALTER TABLE posts ADD COLUMN content_score REAL DEFAULT 0.0")
        conn.commit()
        return "applied", description
    except sqlite3.OperationalError as exc:
        if "duplicate column name" in str(exc).lower():
            return "skipped", description + " — column already exists"
        raise


# ---------------------------------------------------------------------------
# Migration 3 – posts.ab_test_id
# ---------------------------------------------------------------------------


def _m03_add_ab_test_id(conn: sqlite3.Connection) -> Tuple[str, str]:
    """Add ab_test_id TEXT column to posts (nullable, links to ab_tests)."""
    description = "add ab_test_id TEXT DEFAULT NULL to posts"
    try:
        conn.execute("ALTER TABLE posts ADD COLUMN ab_test_id TEXT DEFAULT NULL")
        conn.commit()
        return "applied", description
    except sqlite3.OperationalError as exc:
        if "duplicate column name" in str(exc).lower():
            return "skipped", description + " — column already exists"
        raise


# ---------------------------------------------------------------------------
# Migration 4 – posts.retry_count
# ---------------------------------------------------------------------------


def _m04_add_retry_count(conn: sqlite3.Connection) -> Tuple[str, str]:
    """Add retry_count INTEGER column to posts (default: 0)."""
    description = "add retry_count INTEGER DEFAULT 0 to posts"
    try:
        conn.execute("ALTER TABLE posts ADD COLUMN retry_count INTEGER DEFAULT 0")
        conn.commit()
        return "applied", description
    except sqlite3.OperationalError as exc:
        if "duplicate column name" in str(exc).lower():
            return "skipped", description + " — column already exists"
        raise


# ---------------------------------------------------------------------------
# Migration 5 – ab_tests table
# ---------------------------------------------------------------------------


def _m05_create_ab_tests(conn: sqlite3.Connection) -> Tuple[str, str]:
    """Create ab_tests table for A/B experiment management."""
    description = "create ab_tests table"
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ab_tests (
            test_id          TEXT      PRIMARY KEY,
            topic            TEXT      NOT NULL,
            goal             TEXT      NOT NULL,
            variants         TEXT      NOT NULL,   -- JSON array of variant objects
            status           TEXT      DEFAULT 'active',
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            winner_variant_id TEXT,
            winning_pattern  TEXT
        )
        """
    )
    conn.commit()
    return "applied", description


# ---------------------------------------------------------------------------
# Migration 6 – pattern_learnings table
# ---------------------------------------------------------------------------


def _m06_create_pattern_learnings(conn: sqlite3.Connection) -> Tuple[str, str]:
    """Create pattern_learnings table for storing extracted content patterns."""
    description = "create pattern_learnings table"
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pattern_learnings (
            id         INTEGER   PRIMARY KEY AUTOINCREMENT,
            pattern    TEXT      NOT NULL,
            context    TEXT,                          -- JSON blob for extra metadata
            frequency  INTEGER   DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    return "applied", description


# ---------------------------------------------------------------------------
# Migration 7 – task_queue table
# ---------------------------------------------------------------------------


def _m07_create_task_queue(conn: sqlite3.Connection) -> Tuple[str, str]:
    """Create task_queue table for async / deferred task tracking."""
    description = "create task_queue table"
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS task_queue (
            id            INTEGER   PRIMARY KEY AUTOINCREMENT,
            task_name     TEXT      NOT NULL,
            task_args     TEXT,                       -- JSON blob of task arguments
            status        TEXT      DEFAULT 'pending',-- pending|running|completed|failed
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at  TIMESTAMP,
            error_message TEXT,
            retry_count   INTEGER   DEFAULT 0
        )
        """
    )
    conn.commit()
    return "applied", description


# ---------------------------------------------------------------------------
# Migration 8 – index on posts.status
# ---------------------------------------------------------------------------


def _m08_index_posts_status(conn: sqlite3.Connection) -> Tuple[str, str]:
    """Create index idx_posts_status on posts(status) for fast status filters."""
    description = "create index idx_posts_status on posts(status)"
    conn.execute("CREATE INDEX IF NOT EXISTS idx_posts_status ON posts(status)")
    conn.commit()
    return "applied", description


# ---------------------------------------------------------------------------
# Migration 9 – index on posts.created_at
# ---------------------------------------------------------------------------


def _m09_index_posts_created_at(conn: sqlite3.Connection) -> Tuple[str, str]:
    """Create index idx_posts_created_at on posts(created_at) for time-range queries."""
    description = "create index idx_posts_created_at on posts(created_at)"
    conn.execute("CREATE INDEX IF NOT EXISTS idx_posts_created_at ON posts(created_at)")
    conn.commit()
    return "applied", description


# ---------------------------------------------------------------------------
# Migration registry
# ---------------------------------------------------------------------------

# Ordered list of (migration_number, migration_function) pairs.
# NEVER re-order or remove entries — only append new ones.
_MIGRATIONS: List[Tuple[int, _MigrationFn]] = [
    (1, _m01_add_post_goal),
    (2, _m02_add_content_score),
    (3, _m03_add_ab_test_id),
    (4, _m04_add_retry_count),
    (5, _m05_create_ab_tests),
    (6, _m06_create_pattern_learnings),
    (7, _m07_create_task_queue),
    (8, _m08_index_posts_status),
    (9, _m09_index_posts_created_at),
]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_migrations(db_path: str) -> None:
    """Run all schema migrations against the SQLite database at *db_path*.

    Design principles
    -----------------
    * **Idempotent** – calling this function more than once on the same
      database is completely safe.  Every migration checks for the pre-
      existence of its artefact before touching the schema.
    * **Sequential** – migrations execute in ascending numeric order so the
      dependency graph is always satisfied.
    * **Fail-fast** – if any migration raises an unexpected exception the
      runner stops immediately and re-raises a :class:`RuntimeError` so the
      application can refuse to start with a broken schema.
    * **Backward-compatible** – ``ALTER TABLE`` operations are wrapped in
      ``try/except`` blocks because SQLite raises
      ``OperationalError: duplicate column name`` instead of supporting
      ``ALTER TABLE … ADD COLUMN IF NOT EXISTS``.  ``CREATE TABLE/INDEX``
      statements use the native ``IF NOT EXISTS`` clause.

    Args:
        db_path: Filesystem path to the SQLite ``.db`` file.  The file must
                 already exist and contain the base schema created by
                 :meth:`DatabaseManager.init_database`.

    Raises:
        RuntimeError: Wraps any unexpected error from a migration function,
                      preserving the original traceback via ``__cause__``.
    """
    logger.info(
        "migration_runner|db=%s|total_migrations=%d",
        db_path,
        len(_MIGRATIONS),
    )

    try:
        with sqlite3.connect(db_path) as conn:
            # WAL mode gives better read/write concurrency during migrations.
            conn.execute("PRAGMA journal_mode=WAL")
            # Enforce referential integrity for any FK constraints we add.
            conn.execute("PRAGMA foreign_keys=ON")

            for migration_number, migration_fn in _MIGRATIONS:
                try:
                    status, description = migration_fn(conn)
                    logger.info(
                        "migration=%d|status=%s|description=%s",
                        migration_number,
                        status,
                        description,
                    )
                except Exception as exc:
                    logger.error(
                        "migration=%d|status=error|description=%s|error=%s",
                        migration_number,
                        migration_fn.__doc__ or "no description",
                        exc,
                    )
                    raise RuntimeError(
                        f"Migration {migration_number} "
                        f"({migration_fn.__name__}) failed: {exc}"
                    ) from exc

    except RuntimeError:
        # Re-raise migration failures without wrapping them a second time.
        raise
    except Exception as exc:
        logger.error(
            "migration_runner|status=failed|error=%s",
            exc,
        )
        raise RuntimeError(f"Migration runner could not open database: {exc}") from exc

    logger.info(
        "migration_runner|status=complete|migrations_processed=%d",
        len(_MIGRATIONS),
    )
