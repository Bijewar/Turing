"""
Pydantic schema tests for `schemas/report.py`.

Catches:
- SWARM-6 BUG: `FinalReport.run_id` is `str` instead of `UUID` (matches
  `Run.id` inconsistency).
- SWARM-6 BUG: missing length bounds on user-facing list fields
  (`warnings_and_conflicts`, `next_3_actions`).
- SWARM-6 BUG: `top_bridges` should validate each `RankedBridge` (Pydantic
  already does, but the test locks the behaviour in).
- Regression: required-string fields must reject empty / missing.
"""
import pytest
from pydantic import ValidationError
from uuid import UUID
from typing import get_args, get_origin

from schemas.layer3 import (
    IsomorphismMatch,
    RankedBridge,
    ValidityScores,
)
from schemas.report import FinalReport


def _valid_ranked_bridge() -> RankedBridge:
    """Helper — returns a minimal valid RankedBridge."""
    from schemas.layer3 import ExtractedMechanism
    from schemas.graph import CausalGraph
    from schemas.layer3 import MergedResult, SearchResult, SearchSource, StructuralQuery
    return RankedBridge(
        match=IsomorphismMatch(
            mechanism=ExtractedMechanism(
                source_result=MergedResult(
                    title="T", merged_summary="S", underlying_mechanism="M",
                    sources=[SearchSource.PAPER],
                    source_results=[
                        SearchResult(
                            source=SearchSource.PAPER,
                            title="T2", summary="S2",
                            confidence=50.0,
                            original_query=StructuralQuery(
                                original_node_id="n",
                                original_confidence=50.0,
                                structural_description="d",
                            ),
                        )
                    ],
                    confidence=50.0,
                ),
                causal_graph=CausalGraph(nodes=[], edges=[]),
            ),
            isomorphism_score=50.0,
            match_type="STRONG_PARTIAL",
        ),
        scores=ValidityScores(
            structural_match=0.5,
            constraint_compatibility=0.5,
            solution_transferability=0.5,
            evidence_strength=0.5,
            final_score=0.5,
        ),
    )


# ===========================================================================
# Required field validation.
# ===========================================================================

def test_final_report_validates_required_fields():
    """Missing any of the 3 required string sections MUST raise ValidationError."""
    with pytest.raises(ValidationError):
        FinalReport(run_id="x")  # missing the_mechanism, the_experiment, who_solved_this


def test_final_report_requires_run_id():
    """Missing run_id MUST raise ValidationError."""
    with pytest.raises(ValidationError):
        FinalReport(
            the_mechanism="m",
            the_experiment="e",
            who_solved_this="w",
        )


def test_final_report_requires_nonempty_sections():
    """Empty strings for required sections MUST raise ValidationError."""
    with pytest.raises(ValidationError):
        FinalReport(
            run_id="x",
            the_mechanism="",
            the_experiment="e",
            who_solved_this="w",
        )
    with pytest.raises(ValidationError):
        FinalReport(
            run_id="x",
            the_mechanism="m",
            the_experiment="",
            who_solved_this="w",
        )
    with pytest.raises(ValidationError):
        FinalReport(
            run_id="x",
            the_mechanism="m",
            the_experiment="e",
            who_solved_this="",
        )


def test_final_report_valid_construction_passes():
    """Regression check: a fully-populated FinalReport validates."""
    report = FinalReport(
        run_id="12345678-1234-4321-8765-123456789012",
        the_mechanism="Cause X causes Y",
        the_experiment="Run experiment Z",
        who_solved_this="Field A solved this via analogy B",
    )
    assert report.the_mechanism.startswith("Cause")
    assert report.warnings_and_conflicts == []
    assert report.next_3_actions == []


# ===========================================================================
# run_id — must be UUID.
# ===========================================================================

def test_final_report_run_id_is_uuid():
    """SWARM-6 BUG: `FinalReport.run_id` MUST be `UUID`, not `str`.

    The `Run.id` schema is being fixed to UUID; `FinalReport.run_id`
    must match for consistency. Without UUID validation, an attacker
    can supply a path-traversal `run_id` like `../../etc/passwd` that
    gets serialized into the report.
    """
    field = FinalReport.model_fields["run_id"]
    annotation = field.annotation
    # Strip Optional[X] wrapper if any.
    if get_origin(annotation) is not None and None in get_args(annotation):
        args = [a for a in get_args(annotation) if a is not type(None)]
        annotation = args[0] if args else annotation
    assert annotation is UUID, (
        f"Expected FinalReport.run_id to be UUID, got {annotation!r}"
    )


def test_final_report_run_id_accepts_uuid_string():
    """A valid UUID string MUST construct successfully."""
    report = FinalReport(
        run_id="12345678-1234-4321-8765-123456789012",
        the_mechanism="m", the_experiment="e", who_solved_this="w",
    )
    assert isinstance(report.run_id, UUID)


def test_final_report_run_id_rejects_non_uuid():
    """Non-UUID run_id MUST raise ValidationError."""
    with pytest.raises(ValidationError):
        FinalReport(
            run_id="not-a-uuid",
            the_mechanism="m", the_experiment="e", who_solved_this="w",
        )


def test_final_report_run_id_rejects_path_traversal():
    """Path-traversal run_id MUST raise ValidationError."""
    with pytest.raises(ValidationError):
        FinalReport(
            run_id="../../etc/passwd",
            the_mechanism="m", the_experiment="e", who_solved_this="w",
        )


# ===========================================================================
# top_bridges — each entry must validate as a RankedBridge.
# ===========================================================================

def test_final_report_top_bridges_validates_each_bridge():
    """Passing a malformed bridge entry MUST raise ValidationError.

    SWARM-6 BUG: the schema's `top_bridges: List[RankedBridge]` already
    enforces this in Pydantic v2, but the test locks the behaviour in so
    a future "loose" refactor (e.g. List[Dict]) doesn't slip through.
    """
    bad_bridge = {"not": "a valid RankedBridge"}
    with pytest.raises(ValidationError):
        FinalReport(
            run_id="12345678-1234-4321-8765-123456789012",
            the_mechanism="m", the_experiment="e", who_solved_this="w",
            top_bridges=[bad_bridge],
        )


def test_final_report_top_bridges_accepts_valid_ranked_bridge():
    """Regression check: a valid RankedBridge validates."""
    report = FinalReport(
        run_id="12345678-1234-4321-8765-123456789012",
        the_mechanism="m", the_experiment="e", who_solved_this="w",
        top_bridges=[_valid_ranked_bridge()],
    )
    assert len(report.top_bridges) == 1
    assert isinstance(report.top_bridges[0], RankedBridge)


# ===========================================================================
# warnings_and_conflicts — max length.
# ===========================================================================

def test_final_report_warnings_max_length():
    """A single warning string longer than 500 chars MUST raise.

    SWARM-6 BUG: the field is `List[str]` with no length limit. An
    attacker (or buggy LLM) can emit a 10MB warning string that bloats
    the report JSON and the LLM context in any retry. The fix caps each
    warning string at a reasonable length (e.g., 500 chars).
    """
    long_warning = "x" * 501
    with pytest.raises(ValidationError):
        FinalReport(
            run_id="12345678-1234-4321-8765-123456789012",
            the_mechanism="m", the_experiment="e", who_solved_this="w",
            warnings_and_conflicts=[long_warning],
        )


def test_final_report_warnings_list_max_length():
    """The warnings list itself MUST have a max length (e.g., 100)."""
    many_warnings = ["w"] * 101
    with pytest.raises(ValidationError):
        FinalReport(
            run_id="12345678-1234-4321-8765-123456789012",
            the_mechanism="m", the_experiment="e", who_solved_this="w",
            warnings_and_conflicts=many_warnings,
        )


# ===========================================================================
# next_3_actions — max length (and the README's "exactly 3" spec).
# ===========================================================================

def test_final_report_actions_max_length():
    """SWARM-6 BUG: `next_3_actions` MUST have a max length.

    The README spec says "Exactly 3 ranked, specific, actionable next
    steps". The current schema is `List[str]` with no max — an
    attacker (or buggy LLM) can emit 1000 actions and bloat the report.
    The fix caps it at a small number (e.g., 5 to give a small buffer).
    """
    many_actions = ["action"] * 100
    with pytest.raises(ValidationError):
        FinalReport(
            run_id="12345678-1234-4321-8765-123456789012",
            the_mechanism="m", the_experiment="e", who_solved_this="w",
            next_3_actions=many_actions,
        )


def test_final_report_actions_string_max_length():
    """Each action string MUST have a max length (e.g., 500 chars)."""
    long_action = "x" * 501
    with pytest.raises(ValidationError):
        FinalReport(
            run_id="12345678-1234-4321-8765-123456789012",
            the_mechanism="m", the_experiment="e", who_solved_this="w",
            next_3_actions=[long_action],
        )