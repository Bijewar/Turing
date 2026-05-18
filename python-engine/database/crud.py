"""
database/crud.py — CRUD Operations for RunModel

Provides async create/read/update/delete functions used by the FastAPI routers.
All operations are atomic using single UPDATE statements and with_for_update() row locking,
wrapped with explicit rollback on error, use timezone-aware UTC datetimes,
and support soft-delete and artifact cleanup.
"""
import json
import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Any, Dict

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import RunModel
from schemas.run import Run, RunCreate, RunStatus
from schemas.graph import CausalGraph
from schemas.layer3 import RankedBridge


def _dump_json(val: Any) -> Optional[str]:
    """Helper to safely serialize dicts, lists, and Pydantic models to JSON strings."""
    if val is None:
        return None
    if isinstance(val, str):
        return val
    if hasattr(val, "model_dump"):
        return json.dumps(val.model_dump(mode="json"), default=str)
    if hasattr(val, "dict"):
        return json.dumps(val.dict(), default=str)
    if isinstance(val, list):
        items = [
            x.model_dump(mode="json") if hasattr(x, "model_dump")
            else (x.dict() if hasattr(x, "dict") else x)
            for x in val
        ]
        return json.dumps(items, default=str)
    return json.dumps(val, default=str)


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------

async def create_run(db: AsyncSession, payload: RunCreate) -> RunModel:
    """
    Persists a new Run with PENDING status and returns the ORM row.
    The caller is responsible for committing the session (handled by get_db).
    """
    try:
        now = datetime.now(timezone.utc)
        status_val = RunStatus.PENDING.value if hasattr(RunStatus.PENDING, "value") else "PENDING"
        input_type_val = payload.input_type.value if hasattr(payload.input_type, "value") else str(payload.input_type or "DATA_PATH")
        input_file_val = payload.input_file or "unknown.csv"

        kwargs: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "status": status_val,
            "input_file": input_file_val,
            "input_type": input_type_val,
            "domain": getattr(payload, "domain", "Biology") or "Biology",
            "created_at": now,
            "updated_at": now,
        }

        run = RunModel(**kwargs)
        db.add(run)
        await db.flush()   # Flush to get the id without committing
        return run
    except Exception:
        await db.rollback()
        raise


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

async def get_run(db: AsyncSession, run_id: str) -> Optional[RunModel]:
    """Returns the RunModel for the given run_id, or None if not found."""
    try:
        stmt = select(RunModel).where(RunModel.id == run_id)
        if hasattr(RunModel, "deleted_at"):
            stmt = stmt.where(RunModel.deleted_at.is_(None))
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    except Exception:
        await db.rollback()
        raise


async def get_run_for_update(db: AsyncSession, run_id: str) -> Optional[RunModel]:
    """
    Returns the RunModel for the given run_id with row-level locking (FOR UPDATE)
    to prevent read-modify-write race conditions in concurrent workloads.
    """
    try:
        stmt = select(RunModel).where(RunModel.id == run_id).with_for_update()
        if hasattr(RunModel, "deleted_at"):
            stmt = stmt.where(RunModel.deleted_at.is_(None))
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    except Exception:
        await db.rollback()
        raise


async def get_all_runs(
    db: AsyncSession,
    limit: int = 50,
    offset: int = 0,
    status: Optional[str | RunStatus] = None,
) -> List[RunModel]:
    """
    Returns runs ordered by creation time (newest first) with support for
    pagination (offset/limit) and optional status filtering.
    """
    try:
        stmt = select(RunModel).order_by(RunModel.created_at.desc())
        if hasattr(RunModel, "deleted_at"):
            stmt = stmt.where(RunModel.deleted_at.is_(None))
        if status is not None:
            status_val = status.value if hasattr(status, "value") else str(status)
            stmt = stmt.where(RunModel.status == status_val)
        stmt = stmt.offset(offset).limit(limit)
        result = await db.execute(stmt)
        return list(result.scalars().all())
    except Exception:
        await db.rollback()
        raise


async def list_runs(
    db: AsyncSession,
    limit: int = 50,
    offset: int = 0,
    status: Optional[str | RunStatus] = None,
) -> List[RunModel]:
    """Alias for get_all_runs supporting pagination and status filtering."""
    return await get_all_runs(db, limit=limit, offset=offset, status=status)


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------

async def update_run_status(
    db: AsyncSession,
    run_id: str,
    status: RunStatus | str,
    error_message: Optional[str] = None,
) -> Optional[RunModel]:
    """Updates only the status (and optional error message) of a run atomically."""
    try:
        now = datetime.now(timezone.utc)
        status_val = status.value if hasattr(status, "value") else str(status)
        values: Dict[str, Any] = {
            "status": status_val,
            "updated_at": now,
        }
        if error_message is not None:
            values["error_message"] = error_message
        if hasattr(RunModel, "version"):
            values["version"] = RunModel.version + 1

        stmt = update(RunModel).where(RunModel.id == run_id)
        if hasattr(RunModel, "deleted_at"):
            stmt = stmt.where(RunModel.deleted_at.is_(None))
        result = await db.execute(stmt.values(**values))
        await db.flush()
        if result.rowcount == 0:
            return None
        return await get_run(db, run_id)
    except Exception:
        await db.rollback()
        raise


async def update_run_graph(
    db: AsyncSession,
    run_id: str,
    graph: CausalGraph | dict,
) -> Optional[RunModel]:
    """Persists the causal graph produced by Layer 1/2 for a given run atomically."""
    try:
        now = datetime.now(timezone.utc)
        graph_json = _dump_json(graph)
        values: Dict[str, Any] = {
            "causal_graph": graph_json,
            "updated_at": now,
        }
        if hasattr(RunModel, "version"):
            values["version"] = RunModel.version + 1

        stmt = update(RunModel).where(RunModel.id == run_id)
        if hasattr(RunModel, "deleted_at"):
            stmt = stmt.where(RunModel.deleted_at.is_(None))
        result = await db.execute(stmt.values(**values))
        await db.flush()
        if result.rowcount == 0:
            return None
        return await get_run(db, run_id)
    except Exception:
        await db.rollback()
        raise


async def update_run_simulations(
    db: AsyncSession,
    run_id: str,
    simulations_data: dict,
) -> Optional[RunModel]:
    """Persists the Layer 2 multi-agent simulation data for a given run atomically."""
    try:
        now = datetime.now(timezone.utc)
        sim_json = _dump_json(simulations_data)
        values: Dict[str, Any] = {
            "layer2_data": sim_json,
            "updated_at": now,
        }
        if hasattr(RunModel, "version"):
            values["version"] = RunModel.version + 1

        stmt = update(RunModel).where(RunModel.id == run_id)
        if hasattr(RunModel, "deleted_at"):
            stmt = stmt.where(RunModel.deleted_at.is_(None))
        result = await db.execute(stmt.values(**values))
        await db.flush()
        if result.rowcount == 0:
            return None
        return await get_run(db, run_id)
    except Exception:
        await db.rollback()
        raise


async def update_run_layer2(
    db: AsyncSession,
    run_id: str,
    layer2_data: dict,
) -> Optional[RunModel]:
    """Alias for update_run_simulations used by router handlers."""
    return await update_run_simulations(db, run_id, layer2_data)


async def update_run_bridges(
    db: AsyncSession,
    run_id: str,
    bridges: List[RankedBridge] | list,
) -> Optional[RunModel]:
    """Persists the Top 3 bridges produced by Layer 3 for a given run atomically."""
    try:
        now = datetime.now(timezone.utc)
        bridges_json = _dump_json(bridges)
        values: Dict[str, Any] = {
            "top_bridges": bridges_json,
            "status": RunStatus.COMPLETE.value if hasattr(RunStatus.COMPLETE, "value") else "COMPLETE",
            "updated_at": now,
        }
        if hasattr(RunModel, "version"):
            values["version"] = RunModel.version + 1

        stmt = update(RunModel).where(RunModel.id == run_id)
        if hasattr(RunModel, "deleted_at"):
            stmt = stmt.where(RunModel.deleted_at.is_(None))
        result = await db.execute(stmt.values(**values))
        await db.flush()
        if result.rowcount == 0:
            return None
        return await get_run(db, run_id)
    except Exception:
        await db.rollback()
        raise


async def update_run_report(
    db: AsyncSession,
    run_id: str,
    report: dict,
) -> Optional[RunModel]:
    """Persists the final human-readable report produced by Layer 4 for a given run atomically."""
    try:
        now = datetime.now(timezone.utc)
        report_json = _dump_json(report)
        values: Dict[str, Any] = {
            "final_report": report_json,
            "status": RunStatus.COMPLETE.value if hasattr(RunStatus.COMPLETE, "value") else "COMPLETE",
            "updated_at": now,
        }
        if hasattr(RunModel, "version"):
            values["version"] = RunModel.version + 1

        stmt = update(RunModel).where(RunModel.id == run_id)
        if hasattr(RunModel, "deleted_at"):
            stmt = stmt.where(RunModel.deleted_at.is_(None))
        result = await db.execute(stmt.values(**values))
        await db.flush()
        if result.rowcount == 0:
            return None
        return await get_run(db, run_id)
    except Exception:
        await db.rollback()
        raise


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

async def delete_run(db: AsyncSession, run_id: str) -> bool:
    """
    Soft-deletes a run by setting deleted_at (and is_deleted) and cleans up associated on-disk artifacts.
    Returns True if deleted, False if not found.
    """
    try:
        now = datetime.now(timezone.utc)
        stmt = update(RunModel).where(RunModel.id == run_id)
        if hasattr(RunModel, "deleted_at"):
            stmt = stmt.where(RunModel.deleted_at.is_(None))
        values: Dict[str, Any] = {"updated_at": now}
        if hasattr(RunModel, "deleted_at"):
            values["deleted_at"] = now
        if hasattr(RunModel, "is_deleted"):
            values["is_deleted"] = True
        if hasattr(RunModel, "version"):
            values["version"] = RunModel.version + 1

        result = await db.execute(stmt.values(**values))
        await db.flush()

        if result.rowcount == 0:
            return False

        # Clean up on-disk model artifacts if present
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "storage", "models", f"{run_id}.joblib")
            if os.path.exists(model_path):
                os.unlink(model_path)
        except Exception:
            pass

        return True
    except Exception:
        await db.rollback()
        raise
