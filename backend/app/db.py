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


_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def configure(database_url: str) -> Engine:
    """(Re)create the engine. Called on app start-up and by tests with an isolated database."""
    global _engine, _SessionLocal
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


def init_db() -> None:
    from . import orm  # noqa: F401  (register tables)

    assert _engine is not None, "configure() must be called first"
    Base.metadata.create_all(_engine)


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
