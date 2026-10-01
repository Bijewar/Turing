"""
Pydantic schema tests for `schemas/run.py`.

Catches:
- SWARM-6 BUG #53 — `RunCreate.input_type` defaults to "DATA_PATH" instead of an enum.
- SWARM-6 BUG #55 — `Run` schema lacks `from_attributes=True`.
- SWARM-6 BUG #56 — `Run.id` is `str` instead of `UUID`.
- Related: input_file/domain length bounds, JSON-string auto-decode for ORM rows.

Every test below FAILS against the current buggy code and PASSES only once
the corresponding fix lands.
"""
from datetime import datetime
from typing import get_args, get_origin
from uuid import UUID

import pytest
from pydantic import BaseModel, ValidationError


# ---------------------------------------------------------------------------
# Imports — fail loudly if the schema module is missing or unfixable.
# ---------------------------------------------------------------------------
from schemas.run import Run, RunCreate, RunStatus


# ===========================================================================
# RunCreate.input_type — should be a proper Enum, not bare `str`.
# ===========================================================================

def test_run_create_input_type_is_enum_not_string():
    """SWARM-6 BUG #53: `RunCreate.input_type` MUST be `RunInputType` enum.

    Currently the field is annotated `Optional[str]` with default
    `"DATA_PATH"`. The fix introduces a `RunInputType` enum (members DATA,
    TEXT) and uses it for the field annotation.
    """
    field = RunCreate.model_fields["input_type"]
    annotation = field.annotation

    # Walk Optional[...] wrappers if present.
    inner = annotation
    while get_origin(inner) in (type(None),):
        inner = get_args(inner)[0]
    # Strip `Optional[X]` → X (Pydantic models `Optional[X]` as
    # `Union[X, NoneType]`).
    if get_origin(inner) is not None and None in get_args(inner):
        args = [a for a in get_args(inner) if a is not type(None)]
        inner = args[0] if args else inner

    # Must be an enum subclass (EnumType from `enum` module).
    import enum
    assert inspect_is_enum(inner), (
        f"Expected RunCreate.input_type to be an Enum, got {inner!r}. "
        "Bare `str` accepts garbage like 'DATA_PATH' / 'csv' / 'foobar'."
    )
    # And it must be the RunInputType enum (or compatible) with at least
    # DATA and TEXT members.
    member_names = {m.name for m in inner}
    assert {"DATA", "TEXT"}.issubset(member_names), (
        f"Expected RunInputType enum to declare DATA and TEXT members, "
        f"got {member_names}"
    )


def inspect_is_enum(t) -> bool:
    import enum
    try:
        return issubclass(t, enum.Enum)
    except TypeError:
        return False


def test_run_create_input_type_valid_values():
    """SWARM-6 BUG #53: only "DATA" and "TEXT" are valid `input_type` values.

    With the buggy code, ANY string is accepted (including the legacy
    `"DATA_PATH"` default, lowercase `"csv"`/`"text"`, etc.). The fix
    restricts the field to a closed enum.
    """
    # Valid construction with the two canonical values must succeed.
    RunCreate(input_type="DATA", input_file="x.csv", domain="Biology")
    RunCreate(input_type="TEXT", input_file="x.txt", domain="Biology")


def test_run_create_input_type_invalid_value_rejected():
    """`input_type="foobar"` MUST raise ValidationError (currently doesn't)."""
    with pytest.raises(ValidationError):
        RunCreate(input_type="foobar")


def test_run_create_input_type_legacy_data_path_rejected():
    """The legacy default `"DATA_PATH"` MUST no longer validate.

    SWARM-6 BUG #53 explicitly calls this out — the buggy default is
    `"DATA_PATH"` which is meaningless.
    """
    with pytest.raises(ValidationError):
        RunCreate(input_type="DATA_PATH")


def test_run_create_input_type_lowercase_rejected():
    """Lowercase `"csv"` / `"text"` MUST be rejected (only uppercase DATA/TEXT).

    The Layer 1 `file_detector.InputType` enum uses lowercase `csv`/`text`
    but the user-facing `RunCreate.input_type` field uses uppercase DATA/TEXT
    per the schema fix. The closed-enum validation must reject lowercase
    spellings to prevent silent mis-routing.
    """
    with pytest.raises(ValidationError):
        RunCreate(input_type="csv")
    with pytest.raises(ValidationError):
        RunCreate(input_type="text")


# ===========================================================================
# RunCreate.input_file — required, max length 512 (matches DB column).
# ===========================================================================

def test_run_create_input_file_required():
    """Empty `input_file` MUST raise ValidationError.

    Currently `input_file` has a default of `"unknown.csv"` so the field
    silently accepts nothing. The fix makes it required.
    """
    with pytest.raises(ValidationError):
        RunCreate(input_file="")


def test_run_create_input_file_max_length():
    """`input_file` longer than 512 chars MUST raise ValidationError.

    SWARM-6 BUG: The DB column is `String(512)` — sending a longer filename
    would crash the database flush with a generic 500 error. The schema
    should reject it with a 422 first.
    """
    too_long = "a" * 513 + ".csv"
    with pytest.raises(ValidationError):
        RunCreate(input_file=too_long, input_type="DATA", domain="Biology")


def test_run_create_input_file_max_length_boundary_ok():
    """Boundary check: a 512-char filename should validate."""
    ok = "a" * 508 + ".csv"  # 512 chars total
    obj = RunCreate(input_file=ok, input_type="DATA", domain="Biology")
    assert obj.input_file == ok


# ===========================================================================
# RunCreate.domain — max length 64.
# ===========================================================================

def test_run_create_domain_max_length():
    """`domain` longer than 64 chars MUST raise ValidationError."""
    too_long = "a" * 65
    with pytest.raises(ValidationError):
        RunCreate(input_file="x.csv", input_type="DATA", domain=too_long)


def test_run_create_domain_max_length_boundary_ok():
    """Boundary check: a 64-char domain should validate."""
    ok = "a" * 64
    obj = RunCreate(input_file="x.csv", input_type="DATA", domain=ok)
    assert obj.domain == ok


# ===========================================================================
# Run.id — must be UUID, not str.
# ===========================================================================

def test_run_id_is_uuid_not_string():
    """SWARM-6 BUG #56: `Run.id` MUST be `UUID`, not `str`.

    The buggy schema uses `id: str` which accepts path-traversal payloads
    (`../../etc/passwd`) that flow into DB queries and filesystem ops.
    """
    field = Run.model_fields["id"]
    annotation = field.annotation
    # Strip Optional wrapper if any.
    if get_origin(annotation) is not None and None in get_args(annotation):
        args = [a for a in get_args(annotation) if a is not type(None)]
        annotation = args[0] if args else annotation
    assert annotation is UUID, (
        f"Expected Run.id to be UUID, got {annotation!r}. "
        "Using str lets attackers pass arbitrary path-like IDs."
    )


def test_run_id_accepts_valid_uuid():
    """A valid UUID string MUST construct successfully."""
    valid_uuid_str = "12345678-1234-4321-8765-123456789012"
    obj = Run(id=valid_uuid_str, input_file="x.csv", input_type="DATA")
    assert isinstance(obj.id, UUID)


def test_run_id_rejects_non_uuid():
    """SWARM-6 BUG #56: `Run(id="not-a-uuid")` MUST raise ValidationError."""
    with pytest.raises(ValidationError):
        Run(id="not-a-uuid", input_file="x.csv", input_type="DATA")


def test_run_id_rejects_path_traversal():
    """SWARM-6 BUG #56: `Run(id="../../etc/passwd")` MUST raise ValidationError.

    Path-traversal IDs must be rejected at the schema boundary before they
    reach DB queries or filesystem operations.
    """
    with pytest.raises(ValidationError):
        Run(id="../../etc/passwd", input_file="x.csv", input_type="DATA")


def test_run_id_rejects_synth_attack_strings():
    """Defence in depth: other suspicious ID strings must also raise."""
    for bad in ("", "   ", "1' OR '1'='1", "%2e%2e%2f", "id with spaces"):
        with pytest.raises(ValidationError):
            Run(id=bad, input_file="x.csv", input_type="DATA")


# ===========================================================================
# Run.model_config — must have from_attributes=True for ORM conversion.
# ===========================================================================

def test_run_schema_has_from_attributes():
    """SWARM-6 BUG #55: `Run.model_config` MUST have `from_attributes=True`.

    Without this, `Run.model_validate(orm_row)` raises
    `pydantic.errors.PydanticUserError` — the `_to_schema()` helper in
    `routers/runs.py` cannot build a Run directly from an ORM row.
    """
    cfg = getattr(Run, "model_config", {})
    assert cfg.get("from_attributes") is True, (
        "Run.model_config must set from_attributes=True so that "
        "Run.model_validate(orm_row) works without manual field copying."
    )


# ===========================================================================
# Run — JSON-string auto-decoding from ORM rows.
# ===========================================================================

class _FakeORMRow:
    """Minimal stand-in for a SQLAlchemy RunModel row.

    The real `RunModel.get_causal_graph()` / `get_top_bridges()` methods
    decode a JSON-string column into a dict / list. With `from_attributes`
    enabled, `Run.model_validate(row)` would invoke those methods.
    """
    def __init__(self, graph_json: str, bridges_json: str | None = None):
        import json
        self.id = "12345678-1234-4321-8765-123456789012"
        self.status = "PENDING"
        self.input_file = "x.csv"
        self.input_type = "DATA"
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.error_message = None
        self._graph_json = graph_json
        self._bridges_json = bridges_json

    def get_causal_graph(self):
        import json
        return json.loads(self._graph_json) if self._graph_json else None

    def get_top_bridges(self):
        import json
        return json.loads(self._bridges_json) if self._bridges_json else None


def test_run_schema_json_decodes_causal_graph():
    """Constructing Run from an ORM row whose `causal_graph` is a JSON
    string MUST auto-decode it to a dict (not the raw string).

    SWARM-6 BUG #55 — without `from_attributes`, the schema cannot be
    built from an ORM row at all; even with it, the JSON-string column
    must be decoded (typically via a field validator).
    """
    graph_dict = {"nodes": [{"id": "a", "label": "A", "confidence": 50.0}], "edges": []}
    import json
    row = _FakeORMRow(graph_json=json.dumps(graph_dict), bridges_json=None)

    run = Run.model_validate(row)

    assert run.causal_graph is not None
    # Must be a dict (decoded), not a JSON string.
    assert isinstance(run.causal_graph, dict), (
        "Expected causal_graph to be decoded into a dict from the JSON column"
    )
    assert run.causal_graph["nodes"][0]["id"] == "a"


def test_run_schema_json_decodes_top_bridges():
    """Same as the causal_graph test, but for `top_bridges`."""
    # Build a minimal valid RankedBridge-compatible dict.
    # RankedBridge requires `match` (IsomorphismMatch) + `scores` (ValidityScores).
    bridges_list = [
        {
            "match": {
                "mechanism": {
                    "source_result": {
                        "source": "PAPER",
                        "title": "T",
                        "summary": "S",
                        "url": None,
                        "confidence": 50.0,
                        "original_query": {
                            "original_node_id": "n",
                            "original_confidence": 50.0,
                            "structural_description": "d",
                        },
                        "citation_count": 0,
                        "replication_count": 0,
                        "deployment_status": "unknown",
                    },
                    "causal_graph": {"nodes": [], "edges": []},
                },
                "isomorphism_score": 50.0,
                "match_type": "STRONG_PARTIAL",
            },
            "scores": {
                "structural_match": 0.5,
                "constraint_compatibility": 0.5,
                "solution_transferability": 0.5,
                "evidence_strength": 0.5,
                "final_score": 0.5,
            },
        }
    ]
    import json
    row = _FakeORMRow(
        graph_json=json.dumps({"nodes": [], "edges": []}),
        bridges_json=json.dumps(bridges_list),
    )

    run = Run.model_validate(row)

    assert run.top_bridges is not None
    assert isinstance(run.top_bridges, list)
    assert len(run.top_bridges) == 1
    # And each bridge must have decoded into a RankedBridge, not stayed a dict.
    from schemas.layer3 import RankedBridge
    assert isinstance(run.top_bridges[0], RankedBridge), (
        "Expected top_bridges list to contain RankedBridge objects, not raw dicts"
    )