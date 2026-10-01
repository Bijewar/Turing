"""Shared pytest fixtures for the Turing-main test suite.

This conftest is shared across all test agents (TEST-1 through TEST-10).
Other agents should add new fixtures below the "Agent-specific fixtures"
section to avoid conflicts with existing fixtures.

Layer 1 algorithm tests (TEST-1) primarily use:
  - rng_seed
  - simple_causal_df, v_structure_df, cond_independent_df, collinear_df
  - empty_df, single_col_df, constant_col_df
  - mock_llm_client
"""
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Force in-memory SQLite BEFORE any `database.*` import — `database.database`
# reads DATABASE_URL at import time and constructs the module-level engine.
# The sandbox default `DATABASE_URL=file:/...` is NOT a valid SQLAlchemy URL,
# so we MUST override it here.
# ---------------------------------------------------------------------------
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
for _k in (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "NVIDIA_API_KEY",
    "NVIDIA_NIM_API_KEY",
    "GROQ_API_KEY",
):
    os.environ.pop(_k, None)
os.environ.setdefault("DEFAULT_LLM_MODEL", "gpt-4o-test")

# ---------------------------------------------------------------------------
# sys.path setup: make `services.*` and `schemas.*` importable.
# ---------------------------------------------------------------------------
_PYTHON_ENGINE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "python-engine")
)
if _PYTHON_ENGINE_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_ENGINE_DIR)


# ---------------------------------------------------------------------------
# Random seed
# ---------------------------------------------------------------------------
@pytest.fixture
def rng_seed() -> int:
    """Default random seed for reproducible tests."""
    return 42


# ---------------------------------------------------------------------------
# Synthetic datasets with known causal structure
# ---------------------------------------------------------------------------
@pytest.fixture
def simple_causal_df(rng_seed) -> pd.DataFrame:
    """DataFrame with a known X → Y → Z chain causal structure.

    Columns:
      X ~ N(0, 1)
      Y = 2.0 * X + N(0, 0.5)
      Z = -1.5 * Y + N(0, 0.5)
    """
    rng = np.random.default_rng(rng_seed)
    n = 500
    X = rng.normal(0, 1, n)
    Y = 2.0 * X + rng.normal(0, 0.5, n)
    Z = -1.5 * Y + rng.normal(0, 0.5, n)
    return pd.DataFrame({"X": X, "Y": Y, "Z": Z})


@pytest.fixture
def v_structure_df(rng_seed) -> pd.DataFrame:
    """DataFrame with a v-structure / collider: X → Z ← Y, where X ⊥ Y.

    Columns:
      X ~ N(0, 1)
      Y ~ N(0, 1)            (independent of X)
      Z = 1.5*X + 2.0*Y + N(0, 0.5)
    """
    rng = np.random.default_rng(rng_seed)
    n = 500
    X = rng.normal(0, 1, n)
    Y = rng.normal(0, 1, n)
    Z = 1.5 * X + 2.0 * Y + rng.normal(0, 0.5, n)
    return pd.DataFrame({"X": X, "Y": Y, "Z": Z})


@pytest.fixture
def cond_independent_df(rng_seed) -> pd.DataFrame:
    """DataFrame where X ⊥ Y | {Z1, Z2} but not marginally.

    Both X and Y depend on Z1 and Z2; given {Z1, Z2} they are independent.
    Removing the X-Y edge requires a conditioning set of size 2 — depth=1
    PC algorithms cannot do it.

    Columns:
      Z1 ~ N(0, 1)
      Z2 ~ N(0, 1)
      X  = 2.0*Z1 + 3.0*Z2 + N(0, 0.3)
      Y  = 4.0*Z1 - 5.0*Z2 + N(0, 0.3)
    """
    rng = np.random.default_rng(rng_seed)
    n = 1000
    Z1 = rng.normal(0, 1, n)
    Z2 = rng.normal(0, 1, n)
    X = 2.0 * Z1 + 3.0 * Z2 + rng.normal(0, 0.3, n)
    Y = 4.0 * Z1 - 5.0 * Z2 + rng.normal(0, 0.3, n)
    return pd.DataFrame({"X": X, "Y": Y, "Z1": Z1, "Z2": Z2})


@pytest.fixture
def collinear_df(rng_seed) -> pd.DataFrame:
    """DataFrame with perfectly collinear columns: Y = X (exact copy)."""
    rng = np.random.default_rng(rng_seed)
    n = 200
    X = rng.normal(0, 1, n)
    Y = X.copy()  # perfectly collinear with X
    Z = rng.normal(0, 1, n)
    return pd.DataFrame({"X": X, "Y": Y, "Z": Z})


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """Empty DataFrame (0 rows, columns declared)."""
    return pd.DataFrame(columns=["A", "B", "C"])


@pytest.fixture
def single_col_df(rng_seed) -> pd.DataFrame:
    """DataFrame with a single column (causal-learn PC/GES need >=2 cols)."""
    rng = np.random.default_rng(rng_seed)
    return pd.DataFrame({"A": rng.normal(0, 1, 100)})


@pytest.fixture
def constant_col_df(rng_seed) -> pd.DataFrame:
    """DataFrame with one constant column alongside a varying column."""
    rng = np.random.default_rng(rng_seed)
    return pd.DataFrame({
        "const": np.full(100, 7.0),
        "varying": rng.normal(0, 1, 100),
    })


# ---------------------------------------------------------------------------
# Mock LLM client (litellm.completion)
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_llm_client(monkeypatch):
    """Mocks litellm.completion to avoid making real LLM API calls.

    Returns the underlying MagicMock so tests can configure responses.
    Works whether or not `litellm` is installed (injects a fake module
    into sys.modules if not, tracked by monkeypatch for rollback).
    """
    try:
        import litellm  # noqa: F401
    except ImportError:
        fake_litellm = MagicMock(spec=["completion", "acompletion"])
        monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
        litellm = fake_litellm

    mock_completion = MagicMock(name="litellm.completion")
    monkeypatch.setattr(litellm, "completion", mock_completion)
    return mock_completion


# ===========================================================================
# Agent-specific fixtures — add new fixtures below this line.
# Other agents: append your fixtures here to avoid conflicts.
# ===========================================================================


# ---------------------------------------------------------------------------
# TEST-2 (Layer 1 Validator / Classifier / Ontology / Extractor / Parser /
#        File Detector) — appended fixtures.
# ---------------------------------------------------------------------------

class _MockMessage:
    """Minimal stand-in for litellm's `response.choices[0].message`."""

    def __init__(self, content):
        self.content = content


class _MockChoice:
    def __init__(self, content):
        self.message = _MockMessage(content)


class _MockResponse:
    """Minimal stand-in for a litellm completion response object."""

    def __init__(self, content):
        self.choices = [_MockChoice(content)]


def make_mock_completion_response(content):
    """Build an object shaped like a litellm completion response."""
    return _MockResponse(content)


class FakeCompletion:
    """
    Configurable stand-in for `litellm.completion` and `litellm.acompletion`.

    Records every call (sync vs async, prompt contents) so tests can assert
    which path was taken. Used as a singleton via the `mock_litellm` fixture.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.sync_calls = []
        self.async_calls = []
        # A queue of canned responses, popped FIFO.
        # Each entry is either a string (content) or an Exception instance
        # (which will be raised when the next call is made).
        self.responses = []
        # Default response used when queue is empty.
        self.default_response = "[]"

    def queue_response(self, content):
        """Append a canned response. Use an Exception instance to make the
        next call raise."""
        self.responses.append(content)

    def queue_responses(self, *contents):
        for c in contents:
            self.responses.append(c)

    def _next_response(self):
        if self.responses:
            return self.responses.pop(0)
        return self.default_response

    @staticmethod
    def _build_response(canned):
        if isinstance(canned, BaseException):
            raise canned
        if isinstance(canned, _MockResponse):
            return canned
        return make_mock_completion_response(str(canned))

    def completion(self, *args, **kwargs):
        self.sync_calls.append({"args": args, "kwargs": kwargs})
        return self._build_response(self._next_response())

    async def acompletion(self, *args, **kwargs):
        self.async_calls.append({"args": args, "kwargs": kwargs})
        return self._build_response(self._next_response())


@pytest.fixture
def fake_completion():
    """Provide a fresh FakeCompletion instance for the test."""
    return FakeCompletion()


@pytest.fixture
def mock_litellm(monkeypatch, fake_completion):
    """
    Patch `litellm.completion` AND `litellm.acompletion` with the same
    `FakeCompletion` instance, so tests can:
      * queue canned responses,
      * assert whether sync vs async paths were used.

    Also rebinds module-level `from litellm import completion` references
    captured at import-time in the modules under test.

    Returns the FakeCompletion instance.
    """
    import litellm

    monkeypatch.setattr(litellm, "completion", fake_completion.completion,
                        raising=False)
    monkeypatch.setattr(litellm, "acompletion", fake_completion.acompletion,
                        raising=False)

    # Re-bind module-level `completion` references captured at import time
    # in modules under test. Some modules did `from litellm import completion`
    # at the top — patch the symbol on those modules too, otherwise they keep
    # calling the original (real) `litellm.completion`.
    modules_to_rebind = [
        "services.layer1.validator",
        "services.layer1.classifier",
        "services.layer1.ontology_builder",
        "services.anthropic_client",
    ]
    for mod_name in modules_to_rebind:
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
            if hasattr(mod, "completion"):
                monkeypatch.setattr(mod, "completion",
                                    fake_completion.completion, raising=False)
            if hasattr(mod, "acompletion"):
                monkeypatch.setattr(mod, "acompletion",
                                    fake_completion.acompletion, raising=False)

    return fake_completion


@pytest.fixture
def make_graph_data():
    """
    Returns a helper that builds a JSON-style graph dict matching the format
    produced by pc_algorithm/notears/ges/ontology_builder and consumed by
    GraphValidator / AmbiguityDetector / ConfidenceChecker.
    """

    def _make(nodes=None, edges=None):
        node_list = []
        for n in nodes or []:
            if isinstance(n, str):
                node_list.append({"id": n, "label": n, "confidence": 95.0})
            elif isinstance(n, tuple):
                nid, attrs = n
                base = {"id": nid, "label": nid, "confidence": 95.0}
                base.update(attrs)
                node_list.append(base)
            elif isinstance(n, dict):
                base = {"id": n["id"],
                        "label": n.get("label", n["id"]),
                        "confidence": n.get("confidence", 95.0)}
                base.update({k: v for k, v in n.items() if k not in base})
                node_list.append(base)
        return {"nodes": node_list, "edges": list(edges or [])}

    return _make


@pytest.fixture
def clean_llm_env(monkeypatch):
    """Clear all LLM-related env vars so tests don't pick up a real key."""
    for var in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "NVIDIA_API_KEY",
        "NVIDIA_NIM_API_KEY",
        "GROQ_API_KEY",
        "DEFAULT_LLM_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)


import importlib as _importlib


def _have(modname):
    try:
        _importlib.import_module(modname)
        return True
    except ImportError:
        return False


@pytest.fixture
def requires_docx():
    if not _have("docx"):
        pytest.skip("python-docx not installed")
    return True


@pytest.fixture
def requires_pdfplumber():
    if not _have("pdfplumber"):
        pytest.skip("pdfplumber not installed")
    return True


@pytest.fixture
def requires_openpyxl():
    if not _have("openpyxl"):
        pytest.skip("openpyxl not installed")
    return True


@pytest.fixture
def requires_xlrd():
    if not _have("xlrd"):
        pytest.skip("xlrd not installed")
    return True


# ===========================================================================
# TEST-6 (Routers + SSE + Endpoint validation) — appended fixtures.
# All fixtures below are scoped to TEST-6 and use names prefixed with `t6_`
# to avoid collisions with other agents.
# ===========================================================================
import asyncio as _asyncio

from sqlalchemy.ext.asyncio import (  # noqa: E402
    create_async_engine as _create_async_engine,
    async_sessionmaker as _async_sessionmaker,
    AsyncSession as _AsyncSession,
)
from sqlalchemy.pool import StaticPool as _StaticPool

from database.models import Base as _Base  # noqa: E402


# Shared in-memory engine — single connection reused across all sessions so
# the schema persists. Tables are NOT wiped per-test (we rely on run_id
# uniqueness + the reset_runs_store fixture for cache isolation).
_t6_engine = _create_async_engine(
    "sqlite+aiosqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=_StaticPool,
    echo=False,
)
_t6_session_factory = _async_sessionmaker(
    bind=_t6_engine,
    class_=_AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def _t6_create_tables():
    async with _t6_engine.begin() as conn:
        await conn.run_sync(_Base.metadata.create_all)


async def _t6_dispose():
    await _t6_engine.dispose()


@pytest.fixture(scope="session", autouse=True)
def _t6_session_init_db():
    """Create all tables once for the test session (TEST-6)."""
    _asyncio.run(_t6_create_tables())
    yield
    _asyncio.run(_t6_dispose())


@pytest.fixture(autouse=True)
async def _t6_auto_tables():
    """Ensure tables exist in the active event loop before each test."""
    async with _t6_engine.begin() as conn:
        await conn.run_sync(_Base.metadata.create_all)


@pytest.fixture(autouse=True)
def _t6_patch_db(monkeypatch):
    """Patch database.database.engine + AsyncSessionLocal to the shared
    in-memory engine so every request goes to our test DB."""
    import database.database as _db

    monkeypatch.setattr(_db, "engine", _t6_engine, raising=True)
    monkeypatch.setattr(_db, "AsyncSessionLocal", _t6_session_factory, raising=True)


@pytest.fixture(autouse=True)
def _t6_clear_runs_store():
    """Clear routers.runs.RUNS_STORE before and after each test (TEST-6)."""
    try:
        from routers import runs as _runs
        _runs.RUNS_STORE.clear()
        yield _runs
        _runs.RUNS_STORE.clear()
    except ImportError:
        yield


# ---------------------------------------------------------------------------
# App + HTTP client fixtures (TEST-6)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def app():
    """The FastAPI app instance, imported once for the whole test session."""
    import main as main_module
    return main_module.app


@pytest.fixture
def sync_client(app):
    """FastAPI TestClient — for sync endpoint tests."""
    from starlette.testclient import TestClient
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def async_client(app):
    """httpx.AsyncClient backed by ASGITransport — for async endpoint tests."""
    from httpx import AsyncClient, ASGITransport
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# ---------------------------------------------------------------------------
# Run-seeding helpers (TEST-6)
# ---------------------------------------------------------------------------
@pytest.fixture
async def seed_run(async_client):
    """Create a Run via POST /api/runs/ and return the JSON response."""
    resp = await async_client.post(
        "/api/runs/",
        json={"input_file": "test.csv", "input_type": "csv", "domain": "Biology"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


@pytest.fixture
async def seed_completed_run(async_client):
    """Create a Run whose DB status is COMPLETE — simulates a post-restart view."""
    resp = await async_client.post(
        "/api/runs/",
        json={"input_file": "test.csv", "input_type": "csv", "domain": "Biology"},
    )
    assert resp.status_code == 201, resp.text
    run = resp.json()

    from database import crud
    from schemas.run import RunStatus

    async with _t6_session_factory() as db:
        await crud.update_run_status(db, run["id"], RunStatus.COMPLETE)
        await db.commit()
    run["status"] = "COMPLETE"
    return run


@pytest.fixture
async def seed_failed_run(async_client):
    """Create a Run whose DB status is FAILED."""
    resp = await async_client.post(
        "/api/runs/",
        json={"input_file": "test.csv", "input_type": "csv", "domain": "Biology"},
    )
    assert resp.status_code == 201, resp.text
    run = resp.json()

    from database import crud
    from schemas.run import RunStatus

    async with _t6_session_factory() as db:
        await crud.update_run_status(db, run["id"], RunStatus.FAILED, "boom")
        await db.commit()
    run["status"] = "FAILED"
    return run


# ===========================================================================
# TEST-7 (Database + CRUD + Race + Persistence) — appended fixtures.
# All fixtures below are scoped to TEST-7 and use names without a `t7_` prefix
# (because no other agent has claimed `async_engine`, `db_session`, `sync_engine`,
# `reset_runs_store`). These create a FRESH in-memory engine per test, fully
# isolated from the shared `_t6_engine` used by the routers/SSE tests.
# ===========================================================================
import asyncio as _t7_asyncio
from sqlalchemy import create_engine as _t7_create_engine
from sqlalchemy.ext.asyncio import (
    create_async_engine as _t7_create_async_engine,
    async_sessionmaker as _t7_async_sessionmaker,
    AsyncSession as _t7_AsyncSession,
)
from sqlalchemy.pool import StaticPool as _t7_StaticPool

from database.models import Base as _t7_Base  # noqa: E402


@pytest.fixture
def sync_engine():
    """Fresh in-memory SQLite sync engine for schema inspection (one per test)."""
    eng = _t7_create_engine("sqlite:///:memory:")
    _t7_Base.metadata.create_all(eng)
    try:
        yield eng
    finally:
        eng.dispose()


@pytest.fixture
async def async_engine():
    """Fresh in-memory async engine (one per test) — isolated from `_t6_engine`."""
    eng = _t7_create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=_t7_StaticPool,
        future=True,
    )
    async with eng.begin() as conn:
        await conn.run_sync(_t7_Base.metadata.create_all)
    try:
        yield eng
    finally:
        await eng.dispose()


@pytest.fixture
async def db_session(async_engine):
    """Fresh async session bound to the per-test async_engine."""
    Session = _t7_async_sessionmaker(
        bind=async_engine, class_=_t7_AsyncSession, expire_on_commit=False
    )
    async with Session() as s:
        yield s


@pytest.fixture
def reset_runs_store():
    """Clear the module-level RUNS_STORE before (and after) the test."""
    try:
        from routers import runs as runs_router
    except ImportError:
        yield None
        return
    runs_router.RUNS_STORE.clear()
    yield runs_router
    runs_router.RUNS_STORE.clear()