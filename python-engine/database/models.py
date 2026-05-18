"""
database/models.py — SQLAlchemy ORM Models

Defines the SQLAlchemy table models for persisting Run sessions and their results.
Includes cross-dialect compatibility (SQLite/PostgreSQL), optimistic locking via version column,
timezone-aware UTC datetime tracking, UUID format validation, and robust JSON serialization.
"""
import json
import logging
import uuid
from datetime import datetime, timezone, date, time
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Text,
    Boolean,
    Index,
    TypeDecorator,
)
from sqlalchemy.orm import DeclarativeBase

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Shared declarative base — imported by database.py to create all tables."""
    pass


def _utcnow() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


class UUIDType(TypeDecorator):
    """
    Cross-dialect UUID string column type that enforces 36-char string representation.
    Returns String(36) across SQLite, Postgres, and MySQL.
    """
    impl = String(36)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return str(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return str(value)
        return value


class AwareDateTime(TypeDecorator):
    """
    Cross-dialect timezone-aware DateTime column type.
    Ensures datetime objects always retain UTC tzinfo on bind and on result load.
    """
    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            if isinstance(value, datetime):
                if value.tzinfo is None:
                    return value.replace(tzinfo=timezone.utc)
                return value.astimezone(timezone.utc)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            if isinstance(value, datetime):
                if value.tzinfo is None:
                    return value.replace(tzinfo=timezone.utc)
                return value.astimezone(timezone.utc)
            elif isinstance(value, str):
                try:
                    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
                except Exception:
                    return value
        return value


class _CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder supporting NumPy scalar/arrays, datetime objects, UUIDs, sets, and Pydantic models."""
    def default(self, o: Any) -> Any:
        try:
            import numpy as np
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(str(o))
            if isinstance(o, np.bool_):
                return bool(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
        except ImportError:
            pass

        if isinstance(o, (datetime, date, time)):
            return o.isoformat()
        if isinstance(o, uuid.UUID):
            return str(o)
        if isinstance(o, (set, frozenset)):
            return list(o)
        if hasattr(o, "model_dump") and callable(getattr(o, "model_dump")):
            return o.model_dump()
        if hasattr(o, "dict") and callable(getattr(o, "dict")):
            return o.dict()

        return super().default(o)


def _safe_json_dumps(val: Any) -> Optional[str]:
    """Serialize any object to a JSON string using _CustomJSONEncoder."""
    if val is None:
        return None
    if isinstance(val, str):
        return val
    try:
        return json.dumps(val, cls=_CustomJSONEncoder)
    except Exception:
        return json.dumps(val, default=str)


def _safe_json_loads(val: Any, field_name: str = "field", run_id: Optional[str] = None) -> Any:
    """Safely deserialize JSON string to Python objects, handling malformed data gracefully."""
    if not val:
        return None
    if isinstance(val, (dict, list)):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Malformed %s JSON for run %s: %s", field_name, run_id, exc)
        return None


class RunModel(Base):
    """
    Persists a single analysis session.

    Mirrors the Pydantic `Run` schema in schemas/run.py but is stored in
    SQLite (default) or PostgreSQL (production).

    JSON blobs (causal_graph, top_bridges, layer2_data, final_report) are
    serialised as TEXT for cross-database compatibility with fallback methods.
    """
    __tablename__ = "runs"
    __table_args__ = (
        Index("ix_runs_status_created", "status", "created_at"),
        Index("ix_runs_tenant_created", "tenant_id", "created_at"),
    )

    id            = Column(UUIDType, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    version       = Column(Integer, nullable=False, default=1)
    status        = Column(String(16), default="PENDING", nullable=False, index=True)
    input_file    = Column(String(512), nullable=False)
    input_type    = Column(String(32),  nullable=False)   # "csv" | "text"
    domain        = Column(String(64),  nullable=False, default="Biology")
    created_at    = Column(AwareDateTime, default=_utcnow, nullable=False, index=True)
    updated_at    = Column(AwareDateTime, default=_utcnow, onupdate=_utcnow, nullable=False)
    deleted_at    = Column(AwareDateTime, nullable=True, index=True)
    is_deleted    = Column(Boolean, nullable=False, default=False)
    created_by    = Column(String(64), nullable=True, index=True)
    updated_by    = Column(String(64), nullable=True)
    tenant_id     = Column(String(64), nullable=True, index=True)
    causal_graph  = Column(Text, nullable=True)    # JSON-serialised CausalGraph
    layer2_data   = Column(Text, nullable=True)    # JSON-serialised Layer 2 Simulation Output
    top_bridges   = Column(Text, nullable=True)    # JSON-serialised List[RankedBridge]
    final_report  = Column(Text, nullable=True)    # JSON-serialised FinalReport
    error_message = Column(Text, nullable=True)

    __mapper_args__ = {"version_id_col": version}

    def __init__(self, **kwargs):
        if "id" not in kwargs or kwargs["id"] is None:
            kwargs["id"] = str(uuid.uuid4())
        if "created_at" not in kwargs or kwargs["created_at"] is None:
            kwargs["created_at"] = _utcnow()
        if "updated_at" not in kwargs or kwargs["updated_at"] is None:
            kwargs["updated_at"] = _utcnow()
        if "version" not in kwargs or kwargs["version"] is None:
            kwargs["version"] = 1
        if "status" not in kwargs or kwargs["status"] is None:
            kwargs["status"] = "PENDING"
        if "domain" not in kwargs or kwargs["domain"] is None:
            kwargs["domain"] = "Biology"
        if "is_deleted" not in kwargs or kwargs["is_deleted"] is None:
            kwargs["is_deleted"] = False
        super().__init__(**kwargs)

    def set_causal_graph(self, graph_dict: Any):
        self.causal_graph = _safe_json_dumps(graph_dict)

    def get_causal_graph(self) -> Optional[dict]:
        return _safe_json_loads(self.causal_graph, "causal_graph", str(self.id) if self.id else None)

    def set_layer2_data(self, layer2_dict: Any):
        self.layer2_data = _safe_json_dumps(layer2_dict)

    def get_layer2_data(self) -> Optional[dict]:
        return _safe_json_loads(self.layer2_data, "layer2_data", str(self.id) if self.id else None)

    def set_top_bridges(self, bridges: Any):
        self.top_bridges = _safe_json_dumps(bridges)

    def get_top_bridges(self) -> Optional[list]:
        return _safe_json_loads(self.top_bridges, "top_bridges", str(self.id) if self.id else None)

    def set_final_report(self, report_dict: Any):
        self.final_report = _safe_json_dumps(report_dict)

    def get_final_report(self) -> Optional[dict]:
        return _safe_json_loads(self.final_report, "final_report", str(self.id) if self.id else None)

    def __repr__(self):
        return f"<RunModel id={self.id} status={self.status} file={self.input_file}>"
