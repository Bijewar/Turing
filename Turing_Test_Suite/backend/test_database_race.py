"""test_database_race.py — Concurrency / race-condition tests.

Each test names the exact SWARM-6 bug ID it catches. Race-condition tests
are inherently hard to make deterministic on SQLite (which serialises writes
at the DB layer). For deterministic bug-detection we use BOTH:

  1. Source-inspection of `database/crud.py` and `routers/runs.py` to verify
     that atomic UPDATE statements (or `with_for_update()`) are used and
     that an `asyncio.Lock` per `run_id` exists — these FAIL with the
     current buggy read-modify-write code.

  2. Behaviour tests using `asyncio.gather` to run many concurrent operations
     and assert the resulting state is consistent (no crashes, no torn
     JSON, all IDs unique). These pass with both buggy and fixed code on
     SQLite but serve as regression tests and would catch torn writes /
     crash regressions.
"""
import asyncio
import inspect
import re
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

import database.crud as crud_mod
import database.models as models_mod
from database.crud import (
    create_run,
    get_run,
    update_run_status,
    update_run_graph,
    update_run_bridges,
)
from database.models import RunModel
from schemas.run import RunCreate, RunStatus
from schemas.graph import CausalGraph, Node, Edge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session_factory(async_engine):
    return async_sessionmaker(
        bind=async_engine, class_=AsyncSession, expire_on_commit=False
    )


def _make_graph(label: str = "g") -> CausalGraph:
    return CausalGraph(
        nodes=[Node(id="a", label=label, confidence=90.0)],
        edges=[],
    )


# ===========================================================================
# BUG #49 — concurrent update_run_status uses read-modify-write, loses writes
# ===========================================================================


def test_concurrent_status_updates_use_atomic_or_locking():
    """SWARM-6 BUG #49 — `update_run_status` MUST use an atomic UPDATE
    statement (or `with_for_update()`) so that N concurrent calls don't
    lose updates. This is the source-inspection half of the test.

    The behavioural half (`test_concurrent_status_updates_dont_lose_writes`)
    below may pass even with buggy code on SQLite (writes serialise), but the
    source-inspection test FAILS on buggy code, so the bug IS detected.
    """
    src = inspect.getsource(crud_mod.update_run_status)
    uses_atomic = ("update(" in src) or ("with_for_update" in src)
    uses_read_modify_write = (
        re.search(r"=\s*await\s+get_run\(", src)
        and "run.status" in src
        and "await db.flush" in src
    )
    assert uses_atomic, (
        "update_run_status uses read-modify-write — concurrent calls will "
        "race. Use `update(RunModel).where(...).values(...)` (atomic UPDATE) "
        "or `select(...).with_for_update()` to lock the row. (BUG #49)"
    )
    if uses_read_modify_write:
        assert "with_for_update" in src, (
            "update_run_status still uses read-modify-write; if you keep the "
            "pattern, you MUST call with_for_update() to lock the row."
        )


@pytest.mark.asyncio
async def test_concurrent_status_updates_dont_lose_writes(async_engine):
    """SWARM-6 BUG #49 — 10 concurrent `update_run_status` calls on the
    same run with different statuses; the final status is one of the 10
    (not a torn write that loses all updates, and not an exception that
    aborts the run).

    NOTE: On SQLite, writes serialise at the DB layer, so this test passes
    even with the buggy read-modify-write code. The bug is detected by the
    companion source-inspection test (`test_concurrent_status_updates_use_atomic_or_locking`).
    This test serves as a regression / behaviour-correctness test.
    """
    Session = _session_factory(async_engine)
    run_id = str(uuid.uuid4())
    async with Session() as s:
        run = RunModel(id=run_id, input_file="x.csv", input_type="csv")
        s.add(run)
        await s.commit()

    statuses = [RunStatus.RUNNING, RunStatus.PENDING, RunStatus.RUNNING, RunStatus.COMPLETE] * 3

    async def one_update(status):
        async with Session() as s:
            await update_run_status(s, run_id, status)
            await s.commit()

    # Run all concurrently.
    await asyncio.gather(*(one_update(s) for s in statuses))

    async with Session() as s:
        row = await get_run(s, run_id)
    expected_values = {s.value for s in statuses}
    assert row.status in expected_values, (
        f"Final status {row.status!r} is not one of the {len(statuses)} "
        f"concurrent writes {expected_values} — a torn write lost all updates. (BUG #49)"
    )


# ===========================================================================
# BUG #49 — concurrent update_run_graph: no torn / corrupted JSON
# ===========================================================================


@pytest.mark.asyncio
async def test_concurrent_graph_updates_dont_corrupt_json(async_engine):
    """SWARM-6 BUG #49 — 5 concurrent `update_run_graph` calls with different
    graphs; the final graph is valid JSON (not corrupted by a torn write).

    NOTE: On SQLite, writes serialise at the DB layer, so JSON corruption
    from concurrent writes is unlikely. This test is a regression test for
    the case where the atomic UPDATE fix is reverted (or where a future
    refactor re-introduces a torn-write bug).
    """
    import json

    Session = _session_factory(async_engine)
    run_id = str(uuid.uuid4())
    async with Session() as s:
        run = RunModel(id=run_id, input_file="x.csv", input_type="csv")
        s.add(run)
        await s.commit()

    graphs = [_make_graph(label=f"g{i}") for i in range(5)]

    async def one_update(graph):
        async with Session() as s:
            await update_run_graph(s, run_id, graph)
            await s.commit()

    await asyncio.gather(*(one_update(g) for g in graphs))

    # Read the raw column and verify it parses as JSON.
    async with Session() as s:
        result = await s.execute(
            text("SELECT causal_graph FROM runs WHERE id = :id"), {"id": run_id}
        )
        raw = result.scalar()
    assert raw is not None, "No graph was persisted"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        pytest.fail(
            f"Final causal_graph is corrupted JSON: {exc!r}. "
            "Concurrent graph updates produced a torn write. (BUG #49)"
        )
    assert isinstance(parsed, dict), f"Expected dict, got {type(parsed).__name__}"
    # Should be one of the 5 graphs (no torn mix). The label is on the node.
    labels = {g.nodes[0].label for g in graphs}
    actual_label = parsed.get("nodes", [{}])[0].get("label")
    assert actual_label in labels, (
        f"Final graph node label {actual_label!r} is not one of the 5 "
        f"submitted graphs {labels} — torn write. (BUG #49)"
    )


# ===========================================================================
# Unique IDs from concurrent create_run
# ===========================================================================


@pytest.mark.asyncio
async def test_concurrent_create_run_unique_ids(async_engine):
    """100 concurrent `create_run` calls produce 100 unique IDs (UUIDs are
    inherently unique with negligible collision probability, but this test
    also catches any accidental shared-state / counter-based ID generator
    that might collide under concurrency)."""
    Session = _session_factory(async_engine)
    payloads = [
        RunCreate(input_file=f"f{i}.csv", input_type="DATA", domain="Biology")
        for i in range(100)
    ]

    async def one_create(p):
        async with Session() as s:
            run = await create_run(s, p)
            await s.commit()
            return run.id

    ids = await asyncio.gather(*(one_create(p) for p in payloads))
    assert len(ids) == 100
    assert len(set(ids)) == 100, (
        f"Only {len(set(ids))} unique IDs out of {len(ids)} create_run calls "
        "— UUID collisions under concurrency (or a counter-based ID generator)."
    )


# ===========================================================================
# BUG #49 — concurrent graph + bridges updates don't clobber each other
# ===========================================================================


@pytest.mark.asyncio
async def test_update_run_bridges_does_not_overwrite_concurrent_graph_update(async_engine):
    """SWARM-6 BUG #49 — A concurrent `update_run_graph` and `update_run_bridges`
    should both persist: after both commit, the row has both the new graph
    AND the new bridges (neither overwrites the other).

    With the buggy read-modify-write code, both coroutines SELECT the same
    original row, then each UPDATEs only its assigned columns. On SQLite the
    UPDATEs serialise and both succeed, so this test passes even with the
    buggy code. The fix is verified by the source-inspection tests in
    test_database_crud.py.
    """
    import json

    Session = _session_factory(async_engine)
    run_id = str(uuid.uuid4())
    async with Session() as s:
        run = RunModel(id=run_id, input_file="x.csv", input_type="csv")
        s.add(run)
        await s.commit()

    graph = _make_graph(label="final_graph")
    bridges = [{"title": "Bridge A", "score": 0.95}]

    async def update_graph():
        async with Session() as s:
            await update_run_graph(s, run_id, graph)
            await s.commit()

    async def update_bridges():
        # Bridges are passed as RankedBridge objects, but crud.update_run_bridges
        # accepts List[RankedBridge]. For this test, bypass the schema by
        # using a raw dict-persisting path: directly set on the model.
        async with Session() as s:
            row = await get_run(s, run_id)
            assert row is not None
            row.set_top_bridges(bridges)
            row.status = "COMPLETE"
            row.updated_at = datetime.utcnow()
            await s.commit()

    # Run concurrently.
    await asyncio.gather(update_graph(), update_bridges())

    async with Session() as s:
        row = await get_run(s, run_id)
    g = row.get_causal_graph()
    b = row.get_top_bridges()
    assert g is not None, (
        "Graph was overwritten to None by the concurrent bridges update — "
        "lost update (BUG #49)."
    )
    # The graph's first node should have our label.
    assert g.get("nodes") and g["nodes"][0].get("label") == "final_graph", (
        f"Graph was not persisted correctly: {g!r}"
    )
    assert b is not None and len(b) == 1, (
        f"Bridges were not persisted (got {b!r}) — lost update (BUG #49)."
    )
    assert b[0]["title"] == "Bridge A"


# ===========================================================================
# BUG #16 — RUNS_STORE concurrent access needs an asyncio.Lock per run_id
# ===========================================================================


def test_run_store_concurrent_access_with_lock():
    """SWARM-6 BUG #16 — `RUNS_STORE` is a process-local dict mutated by both
    the background task and the SSE request handlers with NO lock. Multi-step
    read-modify-write sequences (e.g. `RUNS_STORE[run_id]["layer3Data"]
    ["topBridges"] = [...]` interleaved with `RUNS_STORE.pop(run_id)`) race.
    The fix is to add an `asyncio.Lock` per `run_id` (or move state to Redis).

    We use source inspection of `routers/runs.py` to verify that an
    `asyncio.Lock` (or equivalent) is referenced."""
    try:
        from routers import runs as runs_router
    except ImportError as exc:
        pytest.skip(f"Could not import routers.runs: {exc}")
    src = inspect.getsource(runs_router)
    # Look for asyncio.Lock (any of: asyncio.Lock, AsyncIOSessionLock, etc.)
    has_lock = "asyncio.Lock" in src or "Lock()" in src or "RLock" in src
    # Also acceptable: per-run_id dict of locks
    has_locks_dict = re.search(r"_locks\b|locks\[|RUN_LOCKS|RUN_LOCK|run_lock", src)
    assert has_lock or has_locks_dict, (
        "routers.runs uses RUNS_STORE without any asyncio.Lock per run_id — "
        "concurrent background task + request handlers race. (BUG #16) "
        "Add `RUN_LOCKS: Dict[str, asyncio.Lock] = {}` and acquire before any "
        "multi-step mutation."
    )


# ===========================================================================
# Delete during update — no crash
# ===========================================================================


@pytest.mark.asyncio
async def test_run_store_concurrent_delete_during_update(async_engine):
    """SWARM-6 BUG #15/#49 — A concurrent delete while an update is in
    flight should not crash either side. The delete returns True/False, the
    update returns None (because the row is gone). Neither raises."""
    from database.crud import delete_run

    Session = _session_factory(async_engine)
    run_id = str(uuid.uuid4())
    async with Session() as s:
        run = RunModel(id=run_id, input_file="x.csv", input_type="csv")
        s.add(run)
        await s.commit()

    async def do_update():
        async with Session() as s:
            await update_run_status(s, run_id, RunStatus.RUNNING)
            await s.commit()

    async def do_delete():
        async with Session() as s:
            return await delete_run(s, run_id)
            await s.commit()

    # Run concurrently — neither should raise.
    results = await asyncio.gather(do_update(), do_delete(), return_exceptions=True)
    for r in results:
        assert not isinstance(r, Exception), (
            f"Concurrent update+delete raised: {r!r} — should not crash. (BUG #15/#49)"
        )


# ===========================================================================
# Behaviour test: 100 concurrent status reads + 10 writes — readers see
# only valid RunStatus values, no transient None / torn state.
# ===========================================================================


@pytest.mark.asyncio
async def test_concurrent_reads_during_writes_see_only_valid_states(async_engine):
    """SWARM-6 BUG #49 — Readers concurrent with writers should see only
    valid `RunStatus` values (PENDING/RUNNING/COMPLETE/FAILED), never None
    or a torn/empty string. The atomic UPDATE fix guarantees this; the
    read-modify-write buggy code can briefly expose `None` if the read
    happens between a `get_run()` and the subsequent attribute-set/flush
    (when the ORM hasn't yet populated `status`)."""
    Session = _session_factory(async_engine)
    run_id = str(uuid.uuid4())
    async with Session() as s:
        run = RunModel(id=run_id, input_file="x.csv", input_type="csv")
        s.add(run)
        await s.commit()

    valid_statuses = {"PENDING", "RUNNING", "COMPLETE", "FAILED"}

    async def reader():
        for _ in range(50):
            async with Session() as s:
                row = await get_run(s, run_id)
                if row is not None:
                    assert row.status in valid_statuses, (
                        f"Reader saw transient/torn status {row.status!r} — "
                        "concurrent writer exposed an invalid intermediate state. (BUG #49)"
                    )
            await asyncio.sleep(0)

    async def writer():
        for s_enum in (RunStatus.RUNNING, RunStatus.COMPLETE, RunStatus.RUNNING, RunStatus.COMPLETE):
            async with Session() as s:
                await update_run_status(s, run_id, s_enum)
                await s.commit()

    await asyncio.gather(reader(), writer())