"""test_database_crud.py — Tests for database/crud.py functions.

Each test names the exact SWARM-6 bug ID it catches in its docstring.
Tests use a fresh in-memory async engine per test (see conftest.py).
"""
import inspect
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

import pytest
from sqlalchemy import inspect as sa_inspect, text
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

import database.crud as crud_mod
from database.crud import (
    create_run,
    get_run,
    get_all_runs,
    update_run_status,
    update_run_graph,
    update_run_bridges,
    delete_run,
)
from database.models import RunModel
from schemas.run import RunCreate, RunStatus
from schemas.graph import CausalGraph, Node, Edge
from schemas.layer3 import (
    RankedBridge,
    IsomorphismMatch,
    ExtractedMechanism,
    ValidityScores,
    MatchType,
    SearchResult,
    StructuralQuery,
    SearchSource,
    MergedResult,
)


# ---------------------------------------------------------------------------
# create_run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_run_generates_uuid_id(async_engine):
    """`create_run` should return a row whose `id` is a UUID-format string
    (so it can be safely used as a URL path component and joined across
    systems)."""
    payload = RunCreate(input_file="data.csv", input_type="DATA", domain="Biology")
    Session = async_sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        run = await create_run(s, payload)
        await s.commit()
    assert run.id is not None
    parsed = uuid.UUID(str(run.id))  # raises ValueError on non-UUID
    assert str(parsed) == str(run.id).lower() or str(parsed).upper() == str(run.id)


@pytest.mark.asyncio
async def test_create_run_persists_domain(async_engine):
    """SWARM-6 BUG #54 — `RunCreate.domain` is accepted by the API but the
    buggy `create_run` does not persist it (no `domain` column, no
    assignment). After a `create_run` call, the row should have the
    submitted `domain` value."""
    payload = RunCreate(input_file="data.csv", input_type="DATA", domain="Chemistry")
    Session = async_sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        run = await create_run(s, payload)
        await s.commit()

    # Verify the persisted row has the domain (after a fresh session).
    async with Session() as s:
        result = await s.execute(text("SELECT * FROM runs WHERE id = :id"), {"id": run.id})
        row = result.first()
    # If the column doesn't exist, row won't have a `domain` attribute at all.
    has_domain_col = "domain" in row._fields if row._fields else hasattr(row, "domain")
    assert has_domain_col, (
        "RunModel has no `domain` column — domain is accepted by the API but "
        "cannot be persisted (BUG #54)."
    )
    assert row.domain == "Chemistry", (
        f"Expected domain='Chemistry', got {row.domain!r}. The `domain` field is "
        "accepted by RunCreate but not persisted in create_run (BUG #54)."
    )


# ---------------------------------------------------------------------------
# get_run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_run_returns_none_for_missing(async_engine):
    """`get_run(db, "nonexistent")` should return None, not raise."""
    Session = async_sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        result = await get_run(s, "nonexistent-id")
    assert result is None, (
        f"Expected None for missing run, got {result!r}"
    )


@pytest.mark.asyncio
async def test_get_run_filters_soft_deleted(async_engine):
    """SWARM-6 BUG #52 — If a soft-delete column (`deleted_at`) exists,
    `get_run` should NOT return soft-deleted runs. The buggy code has no
    soft-delete column AND no filter; we test the filter behavior
    conditionally on the column existing."""
    if not hasattr(RunModel, "deleted_at"):
        pytest.skip(
            "RunModel has no `deleted_at` column — soft-delete filter test is "
            "N/A (the column is missing entirely, which is itself BUG #52)."
        )

    Session = async_sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)
    run_id = str(uuid.uuid4())
    async with Session() as s:
        run = RunModel(id=run_id, input_file="x.csv", input_type="csv")
        s.add(run)
        await s.commit()

    # Soft-delete the run via raw UPDATE (so we don't depend on crud.delete_run).
    async with Session() as s:
        await s.execute(
            text("UPDATE runs SET deleted_at = :ts WHERE id = :id"),
            {"ts": datetime.now(timezone.utc).isoformat(), "id": run_id},
        )
        await s.commit()

    # Now get_run should NOT return it.
    async with Session() as s:
        fetched = await get_run(s, run_id)
    assert fetched is None, (
        "get_run returned a soft-deleted run — should filter "
        "WHERE deleted_at IS NULL (BUG #52)."
    )


# ---------------------------------------------------------------------------
# update_run_status — atomic update (BUG #49)
# ---------------------------------------------------------------------------


def test_update_run_status_uses_atomic_update():
    """SWARM-6 BUG #49 — `update_run_status` should use a single atomic
    `UPDATE` statement (or `with_for_update()` on the SELECT), NOT a
    read-modify-write cycle that loses updates on concurrent calls."""
    src = inspect.getsource(crud_mod.update_run_status)
    # Either uses SQLAlchemy `update(...)` or `with_for_update()`.
    uses_atomic_update = (
        "update(" in src
        or "from sqlalchemy import update" in inspect.getsource(crud_mod)
        or "with_for_update" in src
    )
    uses_read_modify_write = (
        "= await get_run(" in src and "run.status" in src and "await db.flush" in src
    )
    assert uses_atomic_update, (
        "update_run_status uses read-modify-write (get_run + attribute set + flush) "
        "instead of an atomic UPDATE statement or with_for_update(). "
        "Two concurrent calls will both read the same row and the second "
        "commit silently overwrites the first (BUG #49)."
    )
    # If still uses read-modify-write, must have with_for_update
    if uses_read_modify_write:
        assert "with_for_update" in src, (
            "update_run_status still uses read-modify-write; "
            "if you keep that pattern, you MUST use with_for_update() to lock the row."
        )


@pytest.mark.asyncio
async def test_update_run_status_handles_missing_run(async_engine):
    """`update_run_status(db, "nonexistent", ...)` should return None, not
    raise. Regression test."""
    Session = async_sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        result = await update_run_status(s, "nonexistent-id", RunStatus.RUNNING)
    assert result is None, (
        f"Expected None for missing run, got {result!r}"
    )


# ---------------------------------------------------------------------------
# update_run_graph — atomic update (BUG #49)
# ---------------------------------------------------------------------------


def test_update_run_graph_uses_atomic_update():
    """SWARM-6 BUG #49 — `update_run_graph` should use a single atomic
    UPDATE statement (or `with_for_update()`), not read-modify-write."""
    src = inspect.getsource(crud_mod.update_run_graph)
    uses_atomic_update = "update(" in src or "with_for_update" in src
    uses_read_modify_write = (
        "= await get_run(" in src and "set_causal_graph" in src and "await db.flush" in src
    )
    assert uses_atomic_update, (
        "update_run_graph uses read-modify-write — concurrent calls lose "
        "updates (BUG #49). Use UPDATE statement or with_for_update()."
    )
    if uses_read_modify_write:
        assert "with_for_update" in src


# ---------------------------------------------------------------------------
# update_run_bridges — atomic update + RunStatus.COMPLETE.value (BUG #49, #50)
# ---------------------------------------------------------------------------


def test_update_run_bridges_uses_atomic_update():
    """SWARM-6 BUG #49 — `update_run_bridges` should use a single atomic
    UPDATE statement (or `with_for_update()`), not read-modify-write."""
    src = inspect.getsource(crud_mod.update_run_bridges)
    uses_atomic_update = "update(" in src or "with_for_update" in src
    uses_read_modify_write = (
        "= await get_run(" in src and "set_top_bridges" in src and "await db.flush" in src
    )
    assert uses_atomic_update, (
        "update_run_bridges uses read-modify-write — concurrent calls lose "
        "updates (BUG #49). Use UPDATE statement or with_for_update()."
    )
    if uses_read_modify_write:
        assert "with_for_update" in src


def test_update_run_bridges_uses_run_status_complete_value():
    """SWARM-6 BUG #50 — `update_run_bridges` should set `run.status =
    RunStatus.COMPLETE.value` (not the raw string `"COMPLETE"`) so that if
    the enum's string value ever changes, this code doesn't silently break."""
    src = inspect.getsource(crud_mod.update_run_bridges)
    # Buggy: a raw "COMPLETE" string literal.
    has_raw_complete_string = bool(
        re.search(r'\.status\s*=\s*["\']COMPLETE["\']', src)
    )
    # Fixed: uses RunStatus.COMPLETE.value or RunStatus.COMPLETE
    uses_enum = (
        "RunStatus.COMPLETE" in src
        or 'RunStatus.COMPLETE.value' in src
    )
    assert (not has_raw_complete_string) and uses_enum, (
        "update_run_bridges sets status via raw string 'COMPLETE' instead of "
        "RunStatus.COMPLETE.value — if the enum value ever changes this breaks "
        "silently (BUG #50)."
    )


# ---------------------------------------------------------------------------
# get_all_runs — offset + status filter (BUG #51)
# ---------------------------------------------------------------------------


def test_get_all_runs_supports_offset_signature():
    """SWARM-6 BUG #51 — `get_all_runs` should accept an `offset` parameter
    so the router can paginate to older runs. The buggy function signature
    is `get_all_runs(db, limit=50)` with no offset."""
    sig = inspect.signature(get_all_runs)
    assert "offset" in sig.parameters, (
        "get_all_runs does not accept an `offset` parameter — no pagination "
        "is possible; users can only ever see the latest `limit` runs (BUG #51)."
    )


@pytest.mark.asyncio
async def test_get_all_runs_supports_offset(async_engine):
    """SWARM-6 BUG #51 — `get_all_runs(db, limit=10, offset=20)` should
    return the 21st–30th most-recent runs (i.e., the next page), not the
    same first 10 every time."""
    Session = async_sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)
    # Insert 30 runs with deterministic created_at so ordering is stable.
    async with Session() as s:
        for i in range(30):
            run = RunModel(
                id=f"run-{i:03d}",
                status="PENDING",
                input_file=f"f{i}.csv",
                input_type="csv",
                created_at=datetime(2024, 1, 1, 0, 0, i),
                updated_at=datetime(2024, 1, 1, 0, 0, i),
            )
            s.add(run)
        await s.commit()

    if "offset" not in inspect.signature(get_all_runs).parameters:
        pytest.skip("get_all_runs has no `offset` parameter (BUG #51 still present).")

    async with Session() as s:
        page1 = await get_all_runs(s, limit=10, offset=0)
        page2 = await get_all_runs(s, limit=10, offset=10)
    page1_ids = [r.id for r in page1]
    page2_ids = [r.id for r in page2]
    assert len(page1) == 10
    assert len(page2) == 10, (
        f"Expected page2 to have 10 runs, got {len(page2)} — offset is broken (BUG #51)."
    )
    assert set(page1_ids).isdisjoint(set(page2_ids)), (
        f"page1 and page2 overlap: {set(page1_ids) & set(page2_ids)} — "
        "offset is ignored (BUG #51)."
    )


def test_get_all_runs_supports_status_filter_signature():
    """SWARM-6 BUG #51 — `get_all_runs` should accept a `status` filter so
    callers can fetch only COMPLETED runs (etc.) without filtering in
    Python."""
    sig = inspect.signature(get_all_runs)
    assert "status" in sig.parameters, (
        "get_all_runs does not accept a `status` parameter — callers cannot "
        "filter by status at the DB layer (BUG #51)."
    )


@pytest.mark.asyncio
async def test_get_all_runs_supports_status_filter(async_engine):
    """SWARM-6 BUG #51 — `get_all_runs(db, status="COMPLETE")` should only
    return runs whose status is COMPLETE."""
    if "status" not in inspect.signature(get_all_runs).parameters:
        pytest.skip("get_all_runs has no `status` parameter (BUG #51 still present).")

    Session = async_sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        for i in range(5):
            s.add(RunModel(
                id=f"p-{i}", status="PENDING", input_file="x.csv", input_type="csv",
                created_at=datetime(2024, 1, 1, 0, 0, i),
                updated_at=datetime(2024, 1, 1, 0, 0, i),
            ))
        for i in range(3):
            s.add(RunModel(
                id=f"c-{i}", status="COMPLETE", input_file="x.csv", input_type="csv",
                created_at=datetime(2024, 1, 1, 0, 1, i),
                updated_at=datetime(2024, 1, 1, 0, 1, i),
            ))
        await s.commit()

    async with Session() as s:
        result = await get_all_runs(s, limit=50, status="COMPLETE")
    statuses = {r.status for r in result}
    assert statuses.issubset({"COMPLETE"}), (
        f"get_all_runs(status='COMPLETE') returned runs with statuses {statuses} "
        "— should only return COMPLETE runs (BUG #51)."
    )
    assert len(result) == 3, f"Expected 3 COMPLETE runs, got {len(result)}"


# ---------------------------------------------------------------------------
# delete_run — soft delete (BUG #52)
# ---------------------------------------------------------------------------


def test_delete_run_soft_deletes():
    """SWARM-6 BUG #52 — `delete_run` should set `deleted_at` (soft delete),
    not `db.delete(run)` (hard delete). The buggy code hard-deletes — no
    audit trail, no recovery. We check the implementation either sets
    `deleted_at` OR uses a soft-delete marker."""
    src = inspect.getsource(crud_mod.delete_run)
    hard_delete = "db.delete(run)" in src or "session.delete(run)" in src or "await db.delete" in src
    soft_delete = "deleted_at" in src or "is_deleted" in src
    assert (not hard_delete) and soft_delete, (
        "delete_run hard-deletes the row (`await db.delete(run)`) instead of "
        "setting `deleted_at = ...` (soft delete). Hard delete loses the audit "
        "trail and prevents recovery from accidental deletes (BUG #52)."
    )


@pytest.mark.asyncio
async def test_delete_run_returns_false_for_missing(async_engine):
    """`delete_run(db, "nonexistent")` should return False, not raise."""
    Session = async_sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        result = await delete_run(s, "nonexistent-id")
    assert result is False, f"Expected False, got {result!r}"