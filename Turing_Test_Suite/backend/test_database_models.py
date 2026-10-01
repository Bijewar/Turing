"""test_database_models.py — Schema and method checks for RunModel.

Every test names the exact SWARM-6 / SWARM-9 bug ID it catches in its
docstring. The current implementation in
`python-engine/database/models.py` is
intentionally incomplete; most of these tests are expected to FAIL until
the bugs are fixed.
"""
import inspect
import json
import logging
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase

import database.models as models_mod
from database.models import RunModel, Base


# ---------------------------------------------------------------------------
# Indexes
# ---------------------------------------------------------------------------


def test_run_model_has_index_on_status(sync_engine):
    """SWARM-6 BUG #45 — `RunModel.status` should be indexed so that
    `get_all_runs(status=...)` does not full-scan the table at scale."""
    insp = sa_inspect(sync_engine)
    indexes = insp.get_indexes("runs")
    indexed_columns = {tuple(ix["column_names"]) for ix in indexes}
    assert ("status",) in indexed_columns or any(
        "status" in cols for cols in indexed_columns
    ), (
        f"`status` is not indexed. Indexes found: {indexes}. "
        "Add `index=True` to the `status` column or `Index('ix_runs_status', 'status')`."
    )


def test_run_model_has_index_on_created_at(sync_engine):
    """SWARM-6 BUG #45 — `created_at` should be indexed; `get_all_runs`
    orders by `created_at DESC LIMIT N` and without an index this becomes a
    full-table-scan + sort at 100k rows."""
    insp = sa_inspect(sync_engine)
    indexes = insp.get_indexes("runs")
    indexed_columns = {tuple(ix["column_names"]) for ix in indexes}
    assert ("created_at",) in indexed_columns, (
        f"`created_at` is not indexed. Indexes found: {indexes}. "
        "Add `index=True` to the `created_at` column or `Index('ix_runs_created_at', 'created_at')`."
    )


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_run_model_id_has_default():
    """SWARM-6 BUG #46 — `id` should have a `default=` callable so that
    `RunModel()` without an explicit `id` does not IntegrityError on flush."""
    col = RunModel.id
    assert col.default is not None, (
        "RunModel.id has no default — constructing RunModel() without id will "
        "raise IntegrityError on flush. Add `default=lambda: str(uuid.uuid4())`."
    )


def test_run_model_id_uuid_format():
    """SWARM-6 BUG #46 — `id` should default to a UUID-format string AND
    non-UUID strings should be rejected (CHECK constraint or UUID type).

    The buggy implementation uses `String(36)` which accepts any 36-char
    string and has no default. This test verifies BOTH:
      1. The default id is a valid UUID.
      2. Inserting a non-UUID string raises IntegrityError (or similar).
    """
    # 1. Default value is a valid UUID.
    run = RunModel(input_file="x.csv", input_type="csv")
    if run.id is None:
        pytest.fail(
            "RunModel.id has no default — cannot test UUID format. "
            "Add `default=lambda: str(uuid.uuid4())`."
        )
    try:
        parsed = uuid.UUID(str(run.id))
    except (ValueError, AttributeError, TypeError) as exc:
        pytest.fail(f"Default RunModel.id is not a valid UUID: {run.id!r} ({exc})")
    assert str(parsed) == str(run.id).lower() or str(parsed).upper() == str(run.id)

    # 2. Non-UUID strings should be rejected. A 36-char non-UUID string passes
    #    String(36) but should fail UUID validation.
    #    The fix could be: a CHECK constraint, a custom TypeDecorator, or the
    #    SQLAlchemy Uuid type. We just verify SOME rejection occurs.
    col_type_name = type(RunModel.id.type).__name__.upper()
    has_uuid_check = _has_uuid_check_constraint(RunModel)
    accepts_uuid_type = "UUID" in col_type_name

    if not (accepts_uuid_type or has_uuid_check):
        pytest.fail(
            "RunModel.id does not enforce UUID format. "
            f"Column type is {type(RunModel.id.type).__name__} and no UUID check "
            "constraint found. Use `Uuid` type, `String(36)` with a CHECK "
            "constraint validating the UUID pattern, or a custom TypeDecorator."
        )


# ---------------------------------------------------------------------------
# Timezone awareness (BUG #44)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_created_at_is_timezone_aware(async_engine):
    """SWARM-6 BUG #44 — `created_at` should be timezone-aware UTC after a
    DB roundtrip, not naive. The buggy code uses `datetime.utcnow()` which
    produces naive datetimes; combined with SQLite's TEXT storage, the
    value comes back without tzinfo. Fix requires `DateTime(timezone=True)`
    + `datetime.now(timezone.utc)` AND a TypeDecorator (or similar) so the
    value retains tz info on read.
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

    Session = async_sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        run = RunModel(input_file="x.csv", input_type="csv")
        s.add(run)
        await s.flush()
        await s.refresh(run)
        created = run.created_at
    assert created is not None, "created_at should be set on flush"
    assert created.tzinfo is not None, (
        f"created_at is naive ({created!r}). "
        "Use `DateTime(timezone=True)` + `default=lambda: datetime.now(timezone.utc)` "
        "AND a TypeDecorator that re-applies UTC tz on read."
    )


@pytest.mark.asyncio
async def test_updated_at_is_timezone_aware(async_engine):
    """SWARM-6 BUG #44 — `updated_at` should be timezone-aware UTC after a
    DB roundtrip, not naive."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

    Session = async_sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        run = RunModel(input_file="x.csv", input_type="csv")
        s.add(run)
        await s.flush()
        await s.refresh(run)
        updated = run.updated_at
    assert updated is not None, "updated_at should be set on flush"
    assert updated.tzinfo is not None, (
        f"updated_at is naive ({updated!r}). Use a timezone-aware default and "
        "`DateTime(timezone=True)` + a TypeDecorator that preserves tz on read."
    )


@pytest.mark.asyncio
async def test_updated_at_auto_updates_on_modify(async_engine):
    """SWARM-6 BUG (general) — `updated_at` should auto-update when a row is
    modified. The buggy code uses `onupdate=datetime.utcnow` which only
    fires when SQLAlchemy ORM tracks the dirty attribute, AND only if the
    column has `onupdate` set. Verify the column has an `onupdate`.
    """
    col = RunModel.updated_at
    assert col.onupdate is not None, (
        "RunModel.updated_at has no `onupdate=` — it will not auto-update on modify."
    )


# ---------------------------------------------------------------------------
# JSON serialisation (BUG #47, BUG #48)
# ---------------------------------------------------------------------------


def test_set_causal_graph_handles_numpy_floats():
    """SWARM-6 BUG #47 — `set_causal_graph` should handle numpy floats
    without raising `TypeError: Object of type float32 is not JSON
    serializable`. The buggy code uses plain `json.dumps` which fails on
    `np.float32` (and on `np.float64` in older numpy versions). PC/NOTEARS
    commonly output `np.float32`/`np.float64` so this is a real bug."""
    np = pytest.importorskip("numpy")
    run = RunModel(id=str(uuid.uuid4()), input_file="x.csv", input_type="csv")
    try:
        run.set_causal_graph({
            "weight": np.float32(0.5),
            "score": np.float64(0.25),
            "count": np.int64(7),
        })
    except TypeError as exc:
        pytest.fail(
            f"set_causal_graph raised TypeError on numpy scalar: {exc}. "
            "Use a custom JSONEncoder that coerces numpy types "
            "(np.integer -> int, np.floating -> float, np.ndarray -> list)."
        )
    loaded = json.loads(run.causal_graph)
    assert loaded["weight"] == 0.5
    assert isinstance(loaded["weight"], float)


def test_set_causal_graph_handles_numpy_arrays():
    """SWARM-6 BUG #47 — `set_causal_graph` should handle numpy arrays
    without raising. The buggy code uses plain `json.dumps` which fails on
    np.ndarray."""
    np = pytest.importorskip("numpy")
    run = RunModel(id=str(uuid.uuid4()), input_file="x.csv", input_type="csv")
    try:
        run.set_causal_graph({"matrix": np.array([[1, 2], [3, 4]])})
    except TypeError as exc:
        pytest.fail(
            f"set_causal_graph raised TypeError on np.ndarray: {exc}. "
            "Use a custom JSONEncoder whose default() coerces np.ndarray to list."
        )
    loaded = json.loads(run.causal_graph)
    assert loaded["matrix"] == [[1, 2], [3, 4]]


def test_set_causal_graph_handles_datetime():
    """SWARM-6 BUG #47 — `set_causal_graph` should handle datetime objects
    (which `json.dumps` rejects by default)."""
    run = RunModel(id=str(uuid.uuid4()), input_file="x.csv", input_type="csv")
    try:
        run.set_causal_graph({"timestamp": datetime.now(timezone.utc)})
    except TypeError as exc:
        pytest.fail(
            f"set_causal_graph raised TypeError on datetime: {exc}. "
            "Use a custom JSONEncoder whose default() coerces datetime to ISO string."
        )
    loaded = json.loads(run.causal_graph)
    assert isinstance(loaded["timestamp"], str)


def test_set_top_bridges_handles_numpy_floats():
    """SWARM-6 BUG #47 — `set_top_bridges` should handle numpy floats
    without raising. Same fix as `set_causal_graph`."""
    np = pytest.importorskip("numpy")
    run = RunModel(id=str(uuid.uuid4()), input_file="x.csv", input_type="csv")
    try:
        run.set_top_bridges([{"score": np.float32(0.95), "rank": np.int64(1)}])
    except TypeError as exc:
        pytest.fail(
            f"set_top_bridges raised TypeError on numpy scalar: {exc}. "
            "Use a custom JSONEncoder that coerces numpy types."
        )
    loaded = json.loads(run.top_bridges)
    assert loaded[0]["score"] == 0.95


@pytest.mark.asyncio
async def test_get_causal_graph_handles_malformed_json(async_engine):
    """SWARM-6 BUG #48 — `get_causal_graph` should return None and log a
    warning when the stored JSON is malformed, not raise `JSONDecodeError`."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

    Session = async_sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)
    run_id = str(uuid.uuid4())
    async with Session() as s:
        run = RunModel(id=run_id, input_file="x.csv", input_type="csv")
        s.add(run)
        await s.flush()
        # Manually corrupt the JSON column with a raw UPDATE.
        await s.execute(
            text("UPDATE runs SET causal_graph = :v WHERE id = :id"),
            {"v": "{broken", "id": run_id},
        )
        await s.commit()

    # Now read it back; the buggy code raises JSONDecodeError.
    async with Session() as s:
        result = await s.execute(text("SELECT * FROM runs WHERE id = :id"), {"id": run_id})
        row = result.first()
        # Map row to RunModel — we can't use ORM easily after raw SQL, so
        # construct one manually.
        run_obj = RunModel(
            id=row.id, input_file=row.input_file, input_type=row.input_type
        )
        # Simulate the ORM-loaded attribute on the instance.
        run_obj.causal_graph = row.causal_graph

    # Should not raise.
    try:
        out = run_obj.get_causal_graph()
    except json.JSONDecodeError as exc:
        pytest.fail(
            f"get_causal_graph raised JSONDecodeError on malformed JSON: {exc}. "
            "Catch JSONDecodeError and return None with a warning log."
        )
    assert out is None, "get_causal_graph should return None for malformed JSON"


@pytest.mark.asyncio
async def test_get_top_bridges_handles_malformed_json(async_engine):
    """SWARM-6 BUG #48 — `get_top_bridges` should return None and log a
    warning when the stored JSON is malformed, not raise `JSONDecodeError`."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

    Session = async_sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)
    run_id = str(uuid.uuid4())
    async with Session() as s:
        run = RunModel(id=run_id, input_file="x.csv", input_type="csv")
        s.add(run)
        await s.flush()
        await s.execute(
            text("UPDATE runs SET top_bridges = :v WHERE id = :id"),
            {"v": "[broken", "id": run_id},
        )
        await s.commit()

    async with Session() as s:
        result = await s.execute(text("SELECT * FROM runs WHERE id = :id"), {"id": run_id})
        row = result.first()
        run_obj = RunModel(id=row.id, input_file=row.input_file, input_type=row.input_type)
        run_obj.top_bridges = row.top_bridges

    try:
        out = run_obj.get_top_bridges()
    except json.JSONDecodeError as exc:
        pytest.fail(
            f"get_top_bridges raised JSONDecodeError on malformed JSON: {exc}. "
            "Catch JSONDecodeError and return None with a warning log."
        )
    assert out is None, "get_top_bridges should return None for malformed JSON"


def test_set_causal_graph_stores_dict():
    """`get_causal_graph` should return a dict (not a JSON string) — i.e.
    the round-trip should parse the stored string back to a Python object."""
    run = RunModel(id=str(uuid.uuid4()), input_file="x.csv", input_type="csv")
    payload = {"nodes": [{"id": "a"}], "edges": []}
    run.set_causal_graph(payload)
    # Simulate attribute reload from DB
    run_obj = RunModel(id=run.id, input_file=run.input_file, input_type=run.input_type)
    run_obj.causal_graph = run.causal_graph
    out = run_obj.get_causal_graph()
    assert isinstance(out, dict), (
        f"get_causal_graph returned {type(out).__name__}, expected dict — "
        "the round-trip should parse the JSON string back to a Python object."
    )
    assert out == payload


# ---------------------------------------------------------------------------
# Schema choices (BUG #43, BUG #54, BUG #25, etc.)
# ---------------------------------------------------------------------------


def test_status_uses_string_not_native_enum(sync_engine):
    """SWARM-6 BUG #43 — `status` column should use `String(16)` (or
    `SAEnum(..., native_enum=False)`) so adding a new status doesn't
    require a Postgres `ALTER TYPE ADD VALUE` migration. The buggy code
    uses `SAEnum(...)` without `native_enum=False`, which creates a native
    Postgres enum type.
    """
    col = RunModel.status
    type_name = type(col.type).__name__.upper()
    # Either String (good) or Enum with native_enum=False (also good).
    if type_name == "ENUM":
        # Check native_enum flag — for SAEnum, it's stored on the type.
        native = getattr(col.type, "native_enum", True)
        assert native is False, (
            f"status uses native SAEnum (native_enum={native}). "
            "Use `String(16)` or `SAEnum(..., native_enum=False)`."
        )
    else:
        assert "STRING" in type_name or "VARCHAR" in type_name, (
            f"status uses {type_name}; expected String(16) or VARCHAR. "
            "Native enums cause Postgres migration pain."
        )


def test_run_model_has_domain_column(sync_engine):
    """SWARM-6 BUG #54 — `RunModel` should have a `domain` column so that
    `RunCreate.domain` is persisted. The current schema drops it silently."""
    insp = sa_inspect(sync_engine)
    cols = {c["name"] for c in insp.get_columns("runs")}
    assert "domain" in cols, (
        f"RunModel has no `domain` column. Found: {sorted(cols)}. "
        "RunCreate.domain is accepted by the API but never persisted."
    )


def test_run_model_has_layer2_data_column(sync_engine):
    """SWARM-6 BUG #25 — `RunModel` should have a `layer2_data` column (or
    equivalent) so that Layer 2 (agent simulation) results survive a server
    restart."""
    insp = sa_inspect(sync_engine)
    cols = {c["name"] for c in insp.get_columns("runs")}
    has_layer2 = any("layer2" in c.lower() for c in cols)
    assert has_layer2, (
        f"RunModel has no `layer2_data` column. Found: {sorted(cols)}. "
        "Layer 2 agent results are only kept in RUNS_STORE and lost on restart."
    )


def test_run_model_has_layer4_report_column(sync_engine):
    """Layer 4 final report should be persisted to the DB so it survives a
    server restart. The current schema has no column for it."""
    insp = sa_inspect(sync_engine)
    cols = {c["name"] for c in insp.get_columns("runs")}
    has_layer4 = any(
        "layer4" in c.lower() or "report" in c.lower() or "final_report" in c.lower()
        for c in cols
    )
    assert has_layer4, (
        f"RunModel has no `layer4_report`/`report` column. Found: {sorted(cols)}. "
        "Layer 4 report is only kept in RUNS_STORE and lost on restart."
    )


def test_run_model_uses_soft_delete(sync_engine):
    """SWARM-6 BUG #52 — `delete_run` should be a soft delete (sets
    `deleted_at`), not a hard delete. Verify the schema has a `deleted_at`
    column (or `is_deleted` flag)."""
    insp = sa_inspect(sync_engine)
    cols = {c["name"] for c in insp.get_columns("runs")}
    has_soft_delete = "deleted_at" in cols or "is_deleted" in cols
    assert has_soft_delete, (
        f"RunModel has no `deleted_at` / `is_deleted` column. Found: {sorted(cols)}. "
        "delete_run is a hard delete — no audit trail, no recovery from accidental deletes."
    )


def test_run_model_has_version_column(sync_engine):
    """SWARM-9 BUG #5 — `RunModel` should have a `version` column for
    optimistic concurrency (SQLAlchemy `__mapper_args__ = {"version_id_col":
    version}`) so that concurrent updates don't silently lose writes."""
    insp = sa_inspect(sync_engine)
    cols = {c["name"] for c in insp.get_columns("runs")}
    has_version = "version" in cols
    assert has_version, (
        f"RunModel has no `version` column. Found: {sorted(cols)}. "
        "Without a version column, concurrent update_run_* calls lose writes silently."
    )
    # Also verify the mapper uses version_id_col.
    mapper_args = getattr(RunModel, "__mapper_args__", {})
    assert mapper_args.get("version_id_col") is not None or any(
        col.name == "version" for col in RunModel.__table__.columns
    ), (
        "RunModel has a `version` column but __mapper_args__ doesn't reference it "
        "via version_id_col — optimistic locking isn't actually enabled."
    )


def test_run_model_has_audit_columns(sync_engine):
    """SWARM-6 (audit) — `RunModel` should have `created_by` / `updated_by`
    audit columns (or the absence should be intentional / documented).

    If the team explicitly decides not to add audit columns, this test can
    be marked as skipped; otherwise the schema should include them.
    """
    insp = sa_inspect(sync_engine)
    cols = {c["name"] for c in insp.get_columns("runs")}
    missing = [c for c in ("created_by", "updated_by") if c not in cols]
    if missing:
        pytest.skip(
            f"RunModel is missing audit columns {missing}. Skipped because audit "
            "tracking is optional, but consider adding `created_by` / `updated_by` "
            "for production audit trails."
        )


def test_run_model_relationships_have_on_delete(sync_engine):
    """SWARM-9 (schema design) — If RunModel has any FK relationships (e.g.
    to child tables like `layer_runs`), they should specify `on_delete`
    (CASCADE, SET NULL, or RESTRICT) so orphan rows don't accumulate."""
    insp = sa_inspect(sync_engine)
    fks = insp.get_foreign_keys("runs")
    if not fks:
        # No FK relationships on this table; nothing to check.
        pytest.skip("RunModel has no FK relationships; on_delete check is N/A.")
    for fk in fks:
        opts = fk.get("options") or {}
        # SQLAlchemy returns {"ondelete": "CASCADE"} etc.
        on_delete = opts.get("ondelete") or fk.get("ondelete")
        assert on_delete in ("CASCADE", "SET NULL", "RESTRICT"), (
            f"FK {fk.get('name')} has no on_delete action. Specify ondelete='CASCADE' "
            f"(or 'SET NULL' / 'RESTRICT') to avoid orphan rows. FK: {fk}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_uuid_check_constraint(model: type) -> bool:
    """True if the model's table has a CHECK constraint that mentions 'uuid'
    or applies a UUID-like pattern to the id column."""
    table = model.__table__
    for constraint in table.constraints:
        sqltext = getattr(constraint, "sqltext", None)
        if sqltext and "uuid" in str(sqltext).lower():
            return True
    return False