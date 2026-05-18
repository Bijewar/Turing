"""
database/database.py — Async Database Engine & Session Factory

Configures an async SQLAlchemy engine backed by SQLite (with WAL mode) or PostgreSQL.
Exposes a robust session dependency for use in FastAPI routes with transactional safety.
"""
import os
from typing import AsyncGenerator

from fastapi import HTTPException
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

# Import Base and all models so create_all() can discover every table
from database.models import Base  # noqa: F401 — side effect: registers RunModel


# ---------------------------------------------------------------------------
# Engine Configuration — defaults to async SQLite, overridable via DATABASE_URL
# ---------------------------------------------------------------------------
_DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./causal_nexus.db")
_is_sqlite: bool = _DATABASE_URL.startswith("sqlite")
_echo: bool = os.getenv("DB_ECHO", "false").lower() in ("true", "1", "t")

_connect_args = {"check_same_thread": False, "timeout": 30} if _is_sqlite else {}

if not _is_sqlite:
    engine = create_async_engine(
        _DATABASE_URL,
        echo=_echo,
        connect_args=_connect_args,
        pool_size=10,
        max_overflow=20,
        pool_recycle=1800,
        pool_pre_ping=True,
        pool_timeout=30,
    )
else:
    engine = create_async_engine(
        _DATABASE_URL,
        echo=_echo,
        connect_args=_connect_args,
    )

if _is_sqlite:
    @event.listens_for(engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.execute("PRAGMA busy_timeout=30000;")
        cursor.close()

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


# ---------------------------------------------------------------------------
# DB Initialisation — called on application startup (via lifespan in main.py)
# ---------------------------------------------------------------------------
async def init_db() -> None:
    """Creates all tables that do not yet exist. Safe to call on every startup."""
    async with engine.begin() as conn:
        if _is_sqlite:
            try:
                await conn.execute(text("PRAGMA journal_mode=WAL;"))
            except Exception:
                pass
        await conn.run_sync(Base.metadata.create_all)


# ---------------------------------------------------------------------------
# FastAPI dependency — yields a session per HTTP request
# ---------------------------------------------------------------------------
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency injection helper.

    Usage in a router:
        @router.post("/")
        async def create_run(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            if session.in_transaction() and session.is_active:
                sync_s = getattr(session, "sync_session", session)
                tx = getattr(sync_s, "_transaction", None)
                tx_dirty = bool(tx and (getattr(tx, "_new", None) or getattr(tx, "_dirty", None) or getattr(tx, "_deleted", None)))
                dirty = bool(session.dirty or session.new or session.deleted or tx_dirty)
                if dirty:
                    await session.commit()
        except HTTPException:
            if session.in_transaction() and session.is_active:
                await session.rollback()
            raise
        except Exception:
            if session.in_transaction() and session.is_active:
                await session.rollback()
            raise
