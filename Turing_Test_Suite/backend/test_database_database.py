"""test_database_database.py — Engine + session factory + lifespan tests.

Each test names the exact SWARM-6 bug ID it catches. Several tests use
source inspection of `database/database.py` and `main.py` because the bug is
in the engine-construction kwargs (which are applied at import time and not
exposed as easily-inspectable runtime attributes in the async API).
"""
import asyncio
import inspect
import os
import re

import pytest
from sqlalchemy import inspect as sa_inspect, text

import database.database as db_mod
from database.models import Base


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_engine_path(relative: str) -> str:
    """Resolve a path relative to the python-engine source dir."""
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in [
        os.path.abspath(os.path.join(here, "..", "..", "python-engine", relative)),
        os.path.abspath(os.path.join(here, "..", "..", "upload", "Turing-main", "python-engine", relative)),
    ]:
        if os.path.exists(candidate):
            return candidate
    return os.path.abspath(os.path.join(here, "..", "..", "python-engine", relative))


# ---------------------------------------------------------------------------
# Engine pool configuration (SWARM-6 BUG #41)
# ---------------------------------------------------------------------------


def _engine_kwargs_in_source() -> dict:
    """Extract the kwargs passed to `create_async_engine(...)` from the
    module source — useful because some kwargs (pool_pre_ping, pool_recycle)
    are applied at engine construction time and aren't reliably exposed on
    the live engine across SQLAlchemy versions."""
    src = inspect.getsource(db_mod)
    # Find `engine = create_async_engine(` ... `)` as a single block.
    match = re.search(r"create_async_engine\(\s*(.*?)\)\s*\n", src, re.DOTALL)
    if not match:
        return {}
    body = match.group(1)
    # Crude kwargs extraction: matches `name=value` or `name = value`.
    kwargs = {}
    for line in body.splitlines():
        m = re.match(r"\s*([a-zA-Z_][a-zA-Z_0-9]*)\s*=\s*(.+?),?\s*$", line)
        if m:
            kwargs[m.group(1)] = m.group(2).strip()
    return kwargs


def test_engine_has_pool_pre_ping():
    """SWARM-6 BUG #41 — engine should have `pool_pre_ping=True` so stale
    pooled connections (killed by server idle timeout) are detected before
    use. Without pre_ping, the next request fails with "server closed the
    connection unexpectedly"."""
    src = inspect.getsource(db_mod)
    assert "pool_pre_ping" in src, (
        "engine is constructed without pool_pre_ping — stale connections "
        "from the pool are not detected before use, causing intermittent "
        "'server closed the connection' 500s in production (Postgres)."
    )


def test_engine_has_pool_recycle():
    """SWARM-6 BUG #41 — engine should have `pool_recycle` set (e.g. 1800s)
    so connections are recycled before the DB server's idle timeout kills
    them."""
    src = inspect.getsource(db_mod)
    assert "pool_recycle" in src, (
        "engine is constructed without pool_recycle — long-lived connections "
        "are killed by the DB server's idle timeout and the next request 500s."
    )


def test_engine_has_pool_size_and_max_overflow():
    """SWARM-6 BUG #41 — engine should have `pool_size` and `max_overflow`
    set so the pool can serve high-traffic APIs without exhausting
    connections. The default pool (5 connections, 0 overflow) is too small."""
    src = inspect.getsource(db_mod)
    assert "pool_size" in src, (
        "engine is constructed without pool_size — the default of 5 is too "
        "small for a production API; concurrent requests will queue/block."
    )
    assert "max_overflow" in src, (
        "engine is constructed without max_overflow — no headroom for "
        "traffic spikes; connections will be refused when the pool is full."
    )


# ---------------------------------------------------------------------------
# get_db transaction lifecycle (SWARM-6 BUG #40)
# ---------------------------------------------------------------------------


def test_get_db_commits_only_when_dirty():
    """SWARM-6 BUG #40 — `get_db` should only commit when the session has
    dirty state. The buggy code unconditionally calls `await
    session.commit()` even for read-only endpoints, wasting WAL/IO on
    Postgres."""
    src = inspect.getsource(db_mod.get_db)
    # Look for any dirty-state check.
    has_dirty_check = any(
        marker in src
        for marker in ("is_modified()", "session.dirty", "session.new", "session.deleted", "in_transaction()")
    )
    assert has_dirty_check, (
        "get_db commits unconditionally — should only commit when the "
        "session has dirty state (session.is_modified() or "
        "session.dirty/new/deleted). Wasted COMMITs on read-only endpoints."
    )


def test_get_db_rolls_back_on_exception():
    """`get_db` should roll back on exception. Regression test — the
    current code already does this correctly."""
    src = inspect.getsource(db_mod.get_db)
    assert "rollback" in src, "get_db does not roll back on exception."


def test_get_db_rolls_back_on_http_exception():
    """SWARM-6 BUG #40 — `get_db` should explicitly handle `HTTPException`
    (a business-logic error like a 404 should not be treated like a 500
    server error: the rollback is wasted when there's no dirty state, and
    HTTPException should be propagated with proper status)."""
    src = inspect.getsource(db_mod.get_db)
    assert "HTTPException" in src, (
        "get_db does not explicitly mention HTTPException — it catches "
        "every exception as if it were a 500, rolling back unconditionally "
        "even for read-only 404s. Add `except HTTPException:` with a "
        "dirty-aware rollback."
    )


# ---------------------------------------------------------------------------
# _DATABASE_URL check (SWARM-6 BUG #42)
# ---------------------------------------------------------------------------


def test_database_url_sqlite_check_uses_startswith():
    """SWARM-6 BUG #42 — the `"sqlite" in _DATABASE_URL` check is too
    loose: a Postgres URL like `postgresql+psycopg://user:sqlite@host/db`
    contains "sqlite" in the password and false-positives, causing
    `check_same_thread=False` to be passed to psycopg (which rejects it).
    The fix: use `_DATABASE_URL.startswith("sqlite")`."""
    src = inspect.getsource(db_mod)
    # Look for the buggy substring-match pattern.
    buggy_in = re.search(r'["\']sqlite["\']\s+in\s+_DATABASE_URL', src)
    # Look for the fixed startswith pattern.
    fixed_startswith = re.search(r'_DATABASE_URL\.startswith\(\s*["\']sqlite', src)
    assert (buggy_in is None) and (fixed_startswith is not None), (
        f"_DATABASE_URL sqlite check should use startswith() not `in`. "
        f"Found buggy `in` pattern: {bool(buggy_in)}; "
        f"found fixed `startswith` pattern: {bool(fixed_startswith)}."
    )


# ---------------------------------------------------------------------------
# init_db + lifespan (SWARM-6 BUG #5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_init_db_creates_tables():
    """`init_db()` should create all registered tables. We use a fresh
    in-memory engine monkeypatched onto the module so we don't interfere
    with the shared module-level engine."""
    from sqlalchemy.ext.asyncio import create_async_engine
    from database.models import Base as OrigBase  # noqa: F401

    fresh = create_async_engine("sqlite+aiosqlite:///:memory:")
    original = db_mod.engine
    db_mod.engine = fresh
    try:
        await db_mod.init_db()
        async with fresh.connect() as conn:
            tables = await conn.run_sync(
                lambda sync: sa_inspect(sync).get_table_names()
            )
        assert "runs" in tables, (
            f"init_db() did not create the 'runs' table. Tables found: {tables}"
        )
    finally:
        db_mod.engine = original
        await fresh.dispose()


@pytest.mark.asyncio
async def test_lifespan_calls_init_db_on_startup():
    """`lifespan` should call `init_db()` on startup. Regression test —
    the current code already does this.

    We read the `main.py` source text directly (instead of importing
    `main`, which triggers `litellm` import and hangs in the sandbox)."""
    main_path = _resolve_engine_path("main.py")
    with open(main_path, "r") as f:
        src = f.read()
    # Extract the lifespan function body.
    assert "def lifespan" in src, "main.py has no `lifespan` function."
    # Find the body of lifespan.
    import re
    m = re.search(r"async def lifespan\([^)]*\):.*?(?=\n@|\ndef |\nclass |\Z)", src, re.DOTALL)
    assert m, "Could not extract lifespan function body from main.py"
    body = m.group(0)
    assert "init_db" in body, "lifespan does not call init_db() on startup."


@pytest.mark.asyncio
async def test_lifespan_disposes_engine_on_shutdown():
    """SWARM-6 BUG #5 — `lifespan` should call `await engine.dispose()` on
    shutdown so the connection pool is released (otherwise hot-reloads leave
    orphaned SQLite file handles and Postgres connections).

    We read the `main.py` source text directly (instead of importing
    `main`, which triggers `litellm` import and hangs in the sandbox)."""
    main_path = _resolve_engine_path("main.py")
    with open(main_path, "r") as f:
        src = f.read()
    import re
    m = re.search(r"async def lifespan\([^)]*\):.*?(?=\n@|\ndef |\nclass |\Z)", src, re.DOTALL)
    assert m, "Could not extract lifespan function body from main.py"
    body = m.group(0)
    assert "engine.dispose" in body and "await" in body, (
        "lifespan does not call `await engine.dispose()` on shutdown. "
        "Add `await engine.dispose()` after the `yield`."
    )


# ---------------------------------------------------------------------------
# Behaviour tests: get_db commit-on-dirty behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_db_does_not_commit_read_only_session(async_engine):
    """SWARM-6 BUG #40 — When a get_db consumer only does SELECTs, the
    session should NOT issue a COMMIT. We verify by inspecting whether the
    session's `commit` was actually called. The buggy code always commits
    (even for read-only) — this test fails on buggy code."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
    from unittest.mock import AsyncMock, patch

    Session = async_sessionmaker(
        bind=async_engine, class_=AsyncSession, expire_on_commit=False
    )

    # Build a real session but mock `.commit` to detect calls.
    real_session = Session()
    real_session.commit = AsyncMock(name="commit")
    real_session.rollback = AsyncMock(name="rollback")

    # Monkeypatch AsyncSessionLocal to return our instrumented session.
    original_factory = db_mod.AsyncSessionLocal
    db_mod.AsyncSessionLocal = lambda: _ctx(real_session)
    try:
        # Drive the get_db async generator: yield, then resume (no exception).
        gen = db_mod.get_db()
        sess = await gen.__anext__()
        assert sess is real_session
        # Read-only operation:
        from sqlalchemy import select
        from database.models import RunModel
        await sess.execute(select(RunModel))
        # End the request normally.
        try:
            await gen.__anext__()
        except StopAsyncIteration:
            pass

        # Buggy code: commit was called. Fixed code: commit was NOT called.
        assert real_session.commit.call_count == 0, (
            f"get_db called commit() {real_session.commit.call_count} times on a "
            "read-only session — should skip commit when session has no dirty state."
        )
    finally:
        db_mod.AsyncSessionLocal = original_factory
        await real_session.close()


@pytest.mark.asyncio
async def test_get_db_commits_dirty_session(async_engine):
    """Counter-test for BUG #40 — When a get_db consumer adds/modifies a row,
    the session SHOULD commit on exit. Ensures the fix doesn't accidentally
    skip commits for real writes."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
    from unittest.mock import AsyncMock

    Session = async_sessionmaker(
        bind=async_engine, class_=AsyncSession, expire_on_commit=False
    )

    real_session = Session()
    committed = {"count": 0}
    original_commit = real_session.commit

    async def tracked_commit():
        committed["count"] += 1
        await original_commit()

    real_session.commit = tracked_commit

    original_factory = db_mod.AsyncSessionLocal
    db_mod.AsyncSessionLocal = lambda: _ctx(real_session)
    try:
        gen = db_mod.get_db()
        sess = await gen.__anext__()
        # Modify state (use explicit id since the buggy schema has no default)
        import uuid
        from database.models import RunModel
        run = RunModel(id=str(uuid.uuid4()), input_file="x.csv", input_type="csv")
        sess.add(run)
        await sess.flush()
        try:
            await gen.__anext__()
        except StopAsyncIteration:
            pass
        assert committed["count"] >= 1, (
            "get_db should commit when session has dirty state, but commit() "
            "was never called."
        )
    finally:
        db_mod.AsyncSessionLocal = original_factory
        await real_session.close()


# ---------------------------------------------------------------------------
# Tiny helper to fake an async context manager session.
# ---------------------------------------------------------------------------


class _ctx:
    """Wraps a session so it can be used as `async with AsyncSessionLocal() as s:`."""

    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, exc_type, exc, tb):
        # Don't close — the test manages lifecycle.
        return False