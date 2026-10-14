"""Database engine/session management (SQLAlchemy 2.0, SQLite by default)."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone

from sqlalchemy import DateTime, create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.types import TypeDecorator


class Base(DeclarativeBase):
    pass


class UTCDateTime(TypeDecorator):
    """Stores naive UTC in SQLite and always returns timezone-aware UTC datetimes."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:  # type: ignore[override]
        if value is None:
            return None
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:  # type: ignore[override]
        return value.replace(tzinfo=timezone.utc) if value is not None else None


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _refuse_placement_sqlite(database_url: str) -> None:
    """Open an existing SQLite file read-only and refuse it if the PlacementPredict schema created it.

    This runs before the engine connects, because the engine's WAL pragma would otherwise rewrite
    the old file's header.
    """
    import sqlite3
    from pathlib import Path

    if not database_url.startswith("sqlite:///"):
        return
    path = Path(database_url.removeprefix("sqlite:///"))
    if not path.exists() or path.stat().st_size == 0:
        return
    con = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if "predictions" in tables and "app_meta" not in tables:
            cols = {r[1] for r in con.execute("PRAGMA table_info(predictions)")}
            if "hybrid_score" not in cols:
                raise IncompatibleDatabase(
                    f"{path.name} was created by the PlacementPredict schema. Point FG_DATABASE_URL at a new file "
                    "(the default is backend/fraudguard.db); the old database is left untouched."
                )
    finally:
        con.close()


_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def configure(database_url: str) -> Engine:
    """(Re)create the engine. Called on app start-up and by tests with an isolated database."""
    global _engine, _SessionLocal
    _refuse_placement_sqlite(database_url)
    if _engine is not None:
        _engine.dispose()
    connect_args = {"check_same_thread": False, "timeout": 30} if database_url.startswith("sqlite") else {}
    _engine = create_engine(database_url, connect_args=connect_args, pool_pre_ping=True)
    if database_url.startswith("sqlite"):

        @event.listens_for(_engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):  # type: ignore[no-untyped-def]
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA busy_timeout=30000")
            cur.close()

    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


SCHEMA_VERSION = "fraud-2"


class IncompatibleDatabase(RuntimeError):
    """The configured database was created by a different schema (e.g. the PlacementPredict app)."""


def init_db() -> None:
    """Create tables and stamp the schema version.

    ``create_all`` never alters existing tables, so pointing the app at a database created by the
    previous PlacementPredict schema would fail later in confusing ways. That case is detected here
    and reported; the old file is never modified (see docs/MIGRATION.md for the reset strategy).
    """
    from sqlalchemy import inspect, text

    from . import orm  # noqa: F401  (register tables)

    assert _engine is not None, "configure() must be called first"
    insp = inspect(_engine)
    tables = set(insp.get_table_names())
    if "predictions" in tables and "app_meta" not in tables:
        cols = {c["name"] for c in insp.get_columns("predictions")}
        if "hybrid_score" not in cols:
            raise IncompatibleDatabase(
                f"{_engine.url.render_as_string(hide_password=True)} was created by the PlacementPredict schema. "
                "Point FG_DATABASE_URL at a new file (the default is backend/fraudguard.db); the old database is left untouched."
            )
    Base.metadata.create_all(_engine)
    with _engine.begin() as conn:
        row = conn.execute(text("SELECT value FROM app_meta WHERE key = 'schema_version'")).first()
        if row is None:
            conn.execute(text("INSERT INTO app_meta (key, value) VALUES ('schema_version', :v)"), {"v": SCHEMA_VERSION})
        elif row[0] != SCHEMA_VERSION:
            raise IncompatibleDatabase(f"Database schema version {row[0]} does not match the application ({SCHEMA_VERSION})")


def engine() -> Engine:
    assert _engine is not None, "configure() must be called first"
    return _engine


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on any error."""
    assert _SessionLocal is not None, "configure() must be called first"
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Iterator[Session]:
    """FastAPI dependency. Always declare it with ``scope="function"`` so the transaction commits
    *before* the response is sent; otherwise a pending SQLite write lock would be held while the
    request-logging middleware tries to write."""
    with session_scope() as s:
        yield s
