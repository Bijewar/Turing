"""
schemas/run.py

Data models for analysis runs and sessions.
"""
import json
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timezone
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from .graph import CausalGraph
from .layer3 import RankedBridge


class RunStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class RunInputType(str, Enum):
    DATA = "DATA"
    TEXT = "TEXT"


class RunCreate(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    input_file: str = Field(..., min_length=1, max_length=512, description="Filename of the uploaded CSV, PDF, or TXT")
    input_type: RunInputType = Field(default=RunInputType.DATA, description="'DATA' for Data Path, 'TEXT' for Text Path")
    domain: Optional[str] = Field(default="Biology", max_length=64, description="Scientific domain or topic")


class Run(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(..., description="Unique run UUID identifier")
    status: RunStatus = Field(default=RunStatus.PENDING, description="Current run status")
    input_file: str = Field(..., min_length=1, max_length=512)
    input_type: str = Field(..., description="Input data type (e.g. DATA, TEXT, csv, text)")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    causal_graph: Optional[Union[Dict[str, Any], CausalGraph]] = None
    layer2_data: Optional[Dict[str, Any]] = None
    top_bridges: Optional[List[RankedBridge]] = None
    final_report: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _from_orm_obj(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return v
        if hasattr(v, "__dict__") or not isinstance(v, (str, int, float, list, tuple)):
            data = {}
            for attr in ("id", "status", "input_file", "input_type", "created_at", "updated_at", "error_message"):
                if hasattr(v, attr):
                    data[attr] = getattr(v, attr)

            if hasattr(v, "get_causal_graph") and callable(v.get_causal_graph):
                data["causal_graph"] = v.get_causal_graph()
            elif hasattr(v, "causal_graph"):
                data["causal_graph"] = getattr(v, "causal_graph")

            if hasattr(v, "get_layer2_data") and callable(v.get_layer2_data):
                data["layer2_data"] = v.get_layer2_data()
            elif hasattr(v, "layer2_data"):
                data["layer2_data"] = getattr(v, "layer2_data")

            if hasattr(v, "get_top_bridges") and callable(v.get_top_bridges):
                data["top_bridges"] = v.get_top_bridges()
            elif hasattr(v, "top_bridges"):
                data["top_bridges"] = getattr(v, "top_bridges")

            if hasattr(v, "get_final_report") and callable(v.get_final_report):
                data["final_report"] = v.get_final_report()
            elif hasattr(v, "final_report"):
                data["final_report"] = getattr(v, "final_report")
            return data
        return v

    @field_validator("causal_graph", "layer2_data", "top_bridges", "final_report", mode="before")
    @classmethod
    def _json_decode(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_str = v.strip()
            if not v_str:
                return None
            try:
                return json.loads(v_str)
            except Exception:
                return v
        return v

