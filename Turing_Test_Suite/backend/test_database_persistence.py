"""test_database_persistence.py — Verifies that data produced by the
pipeline survives a server restart (i.e., it was actually persisted to the
DB, not just kept in the in-memory RUNS_STORE).

Each test names the exact SWARM-6 bug ID it catches. Several tests use
source inspection of `routers/runs.py` because the bug is "the pipeline
never calls `crud.update_run_*`" — only runnable end-to-end with mocked
LLM/ML services, which would couple these tests to many unrelated services.

For `test_get_state_after_restart_reconstructs_all_layers` we drive the
`routers.runs.get_state` handler directly with a fresh DB session so we
don't need to spin up a full HTTP server.
"""
import inspect
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

import database.crud as crud_mod
import database.models as models_mod
from database.models import RunModel


# ===========================================================================
# BUG #25 — Layer 2 data not persisted to DB
# ===========================================================================


def test_layer2_data_persisted_to_db(sync_engine):
    """SWARM-6 BUG #25 — After Layer 2 (agent simulation) completes, the
    results should be persisted to the DB so they survive a server restart.
    The buggy code keeps them only in `RUNS_STORE[run_id]["layer2Data"]` and
    never writes a DB column.

    We verify two things:
      1. `RunModel` has a `layer2_data` (or similar) column.
      2. `routers/runs.py` actually persists to that column (or calls a
         crud function that does) during `process_full_pipeline`.
    """
    insp = sa_inspect(sync_engine)
    cols = {c["name"] for c in insp.get_columns("runs")}
    has_layer2_col = any(
        "layer2" in c.lower() for c in cols
    )
    assert has_layer2_col, (
        f"RunModel has no `layer2_data` column. Found: {sorted(cols)}. "
        "Layer 2 agent results are only kept in RUNS_STORE and lost on restart (BUG #25)."
    )

    # Also verify the pipeline persists Layer 2 results to the DB.
    try:
        from routers import runs as runs_router
    except ImportError as exc:
        pytest.skip(f"Could not import routers.runs: {exc}")
    src = inspect.getsource(runs_router)
    # Look for any persistence of layer2 — either a column assignment or a
    # crud call.
    persists_layer2 = (
        ("layer2_data" in src) or
        ("layer2Data" in src and "crud" in src) or
        ("update_run_layer2" in src) or
        ("Layer2Response" in src and "crud" in src) or
        ("run.layer2_data" in src) or
        ("run\\.layer2" in src)
    )
    assert persists_layer2, (
        "routers/runs.py never persists Layer 2 data to the DB. After "
        "process_full_pipeline completes Layer 2, the in-memory `layer2Data` "
        "is never written to a column. (BUG #25)"
    )


# ===========================================================================
# BUG #11 — Layer 3 top_bridges not persisted to DB
# ===========================================================================


def test_layer3_bridges_persisted_to_db():
    """SWARM-6 BUG #11 — After Layer 3 ranking completes, the orchestrator
    stores `topBridges` in `RUNS_STORE[run_id]["layer3Data"]["topBridges"]`
    (in-memory only) and never calls `crud.update_run_bridges`. After a
    server restart, `get_state` rebuilds from the DB and `top_bridges` is
    None — the frontend shows "no bridges found" for a successful run.

    We verify `routers/runs.py` calls `crud.update_run_bridges` (or
    directly assigns `row.top_bridges`).
    """
    try:
        from routers import runs as runs_router
    except ImportError as exc:
        pytest.skip(f"Could not import routers.runs: {exc}")
    src = inspect.getsource(runs_router)
    # The buggy code never calls crud.update_run_bridges.
    persists_bridges = (
        "crud.update_run_bridges" in src
        or "update_run_bridges" in src
        or ("row.set_top_bridges" in src and "session" in src.lower())
        or ("run.set_top_bridges" in src and "crud" in src.lower())
    )
    assert persists_bridges, (
        "routers/runs.py never calls crud.update_run_bridges — top_bridges "
        "from Layer 3 are only kept in RUNS_STORE and lost on restart. (BUG #11)"
    )


# ===========================================================================
# Layer 4 report not persisted to DB
# ===========================================================================


def test_layer4_report_persisted_to_db(sync_engine):
    """After Layer 4 report generation, the report should be persisted to
    the DB so it survives a server restart. The buggy code keeps the report
    only in `RUNS_STORE[run_id]["layer4Data"]` (in-memory).

    We verify:
      1. `RunModel` has a column for the Layer 4 report (`layer4_report`,
         `final_report`, `report`, etc.).
      2. `routers/runs.py` actually persists to that column (or calls a
         crud function that does) after `build_report` returns.
    """
    insp = sa_inspect(sync_engine)
    cols = {c["name"] for c in insp.get_columns("runs")}
    has_layer4_col = any(
        ("layer4" in c.lower()) or ("report" in c.lower()) or ("final_report" in c.lower())
        for c in cols
    )
    assert has_layer4_col, (
        f"RunModel has no layer4_report / report / final_report column. "
        f"Found: {sorted(cols)}. Layer 4 report is lost on restart."
    )

    try:
        from routers import runs as runs_router
    except ImportError as exc:
        pytest.skip(f"Could not import routers.runs: {exc}")
    src = inspect.getsource(runs_router)
    persists_report = (
        ("layer4_report" in src) or
        ("final_report" in src) or
        ("update_run_report" in src) or
        ("build_report" in src and "crud" in src) or
        ("row.set_layer4" in src)
    )
    assert persists_report, (
        "routers/runs.py never persists the Layer 4 report to the DB. "
        "After build_report returns, the report is only stored in "
        "RUNS_STORE[run_id]['layer4Data'] and lost on restart."
    )


# ===========================================================================
# BUG #24 — get_state cold-start rebuild returns stale "pending" for layers 2-4
# ===========================================================================


@pytest.mark.asyncio
async def test_get_state_after_restart_reconstructs_all_layers(async_engine):
    """SWARM-6 BUG #24 — When a `run_id` is NOT in `RUNS_STORE` (cold-start
    after server restart), `get_state` rebuilds the state from the DB row.
    The buggy code only sets `layer1Status` based on the DB status; layers
    2, 3, 4 are hardcoded to `"pending"` even when the DB status is
    `COMPLETE`. After a restart, a completed run shows up as "only Layer 1
    finished".

    We simulate the restart by clearing `RUNS_STORE`, creating a COMPLETE
    run in the DB, and calling `routers.runs.get_state(run_id, db=s)`
    directly.
    """
    from routers import runs as runs_router

    Session = async_sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)
    run_id = str(uuid.uuid4())
    async with Session() as s:
        run = RunModel(
            id=run_id,
            status="COMPLETE",
            input_file="data.csv",
            input_type="csv",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        run.set_causal_graph({"nodes": [], "edges": []})
        run.set_top_bridges([])
        s.add(run)
        await s.commit()

    # Simulate restart — clear the in-memory cache.
    runs_router.RUNS_STORE.clear()

    # Call the handler directly with our test DB session.
    async with Session() as s:
        state = await runs_router.get_state(run_id, db=s)

    # The handler should set ALL layer statuses to "completed" when DB
    # status is COMPLETE, not just layer1.
    assert state.get("layer1Status") == "completed", (
        f"layer1Status should be 'completed' for a COMPLETE run, got "
        f"{state.get('layer1Status')!r}"
    )
    assert state.get("layer2Status") == "completed", (
        f"layer2Status should be 'completed' for a COMPLETE run, got "
        f"{state.get('layer2Status')!r} — cold-start rebuild hardcodes "
        "'pending' for layers 2-4 even when DB status is COMPLETE (BUG #24)."
    )
    assert state.get("layer3Status") == "completed", (
        f"layer3Status should be 'completed' for a COMPLETE run, got "
        f"{state.get('layer3Status')!r} (BUG #24)."
    )
    assert state.get("layer4Status") == "completed", (
        f"layer4Status should be 'completed' for a COMPLETE run, got "
        f"{state.get('layer4Status')!r} (BUG #24)."
    )


@pytest.mark.asyncio
async def test_get_state_after_restart_failed_run_shows_failed_layer1(async_engine):
    """Companion test for BUG #24 — When the DB status is FAILED, the
    cold-start rebuild should mark layer1Status as 'failed' (the buggy
    code already does this correctly, but the layers 2-4 are still hardcoded
    to 'pending'). Verify that at least layer1 is correctly marked."""
    from routers import runs as runs_router

    Session = async_sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)
    run_id = str(uuid.uuid4())
    async with Session() as s:
        run = RunModel(
            id=run_id,
            status="FAILED",
            input_file="data.csv",
            input_type="csv",
            error_message="something went wrong",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        s.add(run)
        await s.commit()

    runs_router.RUNS_STORE.clear()

    async with Session() as s:
        state = await runs_router.get_state(run_id, db=s)
    assert state.get("layer1Status") == "failed", (
        f"layer1Status should be 'failed' for a FAILED run, got "
        f"{state.get('layer1Status')!r}"
    )


# ===========================================================================
# Bonus: verify _persist_graph is called for Layer 1 graph (currently
# works in the buggy code; regression test)
# ===========================================================================


def test_layer1_graph_persisted_to_db():
    """Regression test: `process_full_pipeline` should persist the Layer 1
    causal graph to the DB via `_persist_graph` (which calls
    `crud.update_run_graph`). The buggy code already does this correctly;
    this test ensures a refactor doesn't break it."""
    try:
        from routers import runs as runs_router
    except ImportError as exc:
        pytest.skip(f"Could not import routers.runs: {exc}")
    src = inspect.getsource(runs_router)
    assert "crud.update_run_graph" in src or "update_run_graph" in src, (
        "routers/runs.py does not call crud.update_run_graph — Layer 1 "
        "graph is not persisted to the DB."
    )


# ===========================================================================
# Bonus: verify _persist_status is called for status changes (regression)
# ===========================================================================


def test_status_changes_persisted_to_db():
    """Regression test: `process_full_pipeline` should call `_persist_status`
    (which calls `crud.update_run_status`) for RUNNING / COMPLETE / FAILED
    status transitions. The buggy code already does this."""
    try:
        from routers import runs as runs_router
    except ImportError as exc:
        pytest.skip(f"Could not import routers.runs: {exc}")
    src = inspect.getsource(runs_router)
    assert "crud.update_run_status" in src or "update_run_status" in src, (
        "routers/runs.py does not call crud.update_run_status — status "
        "changes are not persisted to the DB."
    )