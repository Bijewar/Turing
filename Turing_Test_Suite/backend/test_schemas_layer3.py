"""
Pydantic schema tests for `schemas/layer3.py`.

Catches:
- SWARM-6 BUG #62 — `StructuralQuery.original_confidence`,
  `SearchResult.confidence`, `MergedResult.confidence`,
  `IsomorphismMatch.isomorphism_score` lack documented 0-100 constraints.
- SWARM-5 BUG #23 — `ExtractedMechanism` is missing the
  `is_contradiction_side: bool` field — primary and contradiction views
  are indistinguishable.
- SWARM-5 BUG #34 — `IsomorphismThresholds` has no ordering validation
  (perfect < strong_partial < weak_partial possible → nonsensical
  classifications).
- Related: max-length on user-controlled fields (titles, summaries).
"""
import pytest
from pydantic import ValidationError


from schemas.graph import CausalGraph, Node, Edge
from schemas.layer3 import (
    ExtractedMechanism,
    IsomorphismMatch,
    IsomorphismThresholds,
    MergedResult,
    SearchSource,
    SearchResult,
    Step10Request,
    Step10Response,
    StructuralQuery,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_query() -> StructuralQuery:
    return StructuralQuery(
        original_node_id="n1",
        original_confidence=50.0,
        structural_description="Desc",
    )


def _valid_search_result() -> SearchResult:
    return SearchResult(
        source=SearchSource.PAPER,
        title="A paper",
        summary="A summary",
        url="https://example.com",
        confidence=50.0,
        original_query=_valid_query(),
    )


def _valid_merged_result() -> MergedResult:
    return MergedResult(
        title="Merged",
        merged_summary="Summary",
        underlying_mechanism="Mechanism",
        sources=[SearchSource.PAPER],
        source_results=[_valid_search_result()],
        confidence=50.0,
    )


def _valid_extracted_mechanism() -> ExtractedMechanism:
    return ExtractedMechanism(
        source_result=_valid_merged_result(),
        causal_graph=CausalGraph(nodes=[], edges=[]),
    )


# ===========================================================================
# Confidence bounds (SWARM-6 BUG #62).
# ===========================================================================

def test_structural_query_original_confidence_has_bounds():
    """SWARM-6 BUG #62: `StructuralQuery(original_confidence=150)` MUST raise.

    The schema documents confidence as 0-100 but has no `ge`/`le`
    constraint. A future caller passing 150 silently renders the UI as
    "150% confidence" — nonsensical.
    """
    with pytest.raises(ValidationError):
        StructuralQuery(original_node_id="n", original_confidence=150.0,
                        structural_description="d")
    with pytest.raises(ValidationError):
        StructuralQuery(original_node_id="n", original_confidence=-1.0,
                        structural_description="d")


def test_structural_query_original_confidence_in_range_ok():
    """Regression check: confidence in [0, 100] validates."""
    for v in (0.0, 50.0, 100.0):
        StructuralQuery(original_node_id="n", original_confidence=v,
                        structural_description="d")


def test_search_result_confidence_has_bounds():
    """SWARM-6 BUG #62: `SearchResult(confidence=150)` MUST raise."""
    with pytest.raises(ValidationError):
        SearchResult(
            source=SearchSource.PAPER,
            title="t", summary="s",
            confidence=150.0,
            original_query=_valid_query(),
        )
    with pytest.raises(ValidationError):
        SearchResult(
            source=SearchSource.PAPER,
            title="t", summary="s",
            confidence=-1.0,
            original_query=_valid_query(),
        )


def test_merged_result_confidence_has_bounds():
    """SWARM-6 BUG #62: `MergedResult(confidence=150)` MUST raise."""
    with pytest.raises(ValidationError):
        MergedResult(
            title="t", merged_summary="s", underlying_mechanism="m",
            sources=[SearchSource.PAPER],
            source_results=[_valid_search_result()],
            confidence=150.0,
        )
    with pytest.raises(ValidationError):
        MergedResult(
            title="t", merged_summary="s", underlying_mechanism="m",
            sources=[SearchSource.PAPER],
            source_results=[_valid_search_result()],
            confidence=-1.0,
        )


def test_isomorphism_match_score_has_bounds():
    """SWARM-6 BUG #62: `IsomorphismMatch(isomorphism_score=150)` MUST raise."""
    with pytest.raises(ValidationError):
        IsomorphismMatch(
            mechanism=_valid_extracted_mechanism(),
            isomorphism_score=150.0,
            match_type="PERFECT",
        )
    with pytest.raises(ValidationError):
        IsomorphismMatch(
            mechanism=_valid_extracted_mechanism(),
            isomorphism_score=-1.0,
            match_type="PERFECT",
        )


# ===========================================================================
# Max-length on user-controlled fields (titles, summaries).
# ===========================================================================

def test_search_result_max_length_on_user_fields():
    """A title longer than 1000 chars MUST raise ValidationError.

    SWARM-8 BUG (defense in depth): search-result titles come from
    scraped web content. Without a max-length constraint, an attacker
    can flood a single search result with a 10MB title, blowing up
    the LLM context window in Step 12 (relation extraction).
    """
    long_title = "x" * 1001
    with pytest.raises(ValidationError):
        SearchResult(
            source=SearchSource.PAPER,
            title=long_title,
            summary="s",
            confidence=50.0,
            original_query=_valid_query(),
        )


def test_merged_result_title_max_length():
    """`MergedResult.title` longer than 1000 chars MUST raise."""
    long_title = "x" * 1001
    with pytest.raises(ValidationError):
        MergedResult(
            title=long_title,
            merged_summary="s",
            underlying_mechanism="m",
            sources=[SearchSource.PAPER],
            source_results=[_valid_search_result()],
            confidence=50.0,
        )


# ===========================================================================
# ExtractedMechanism.is_contradiction_side (SWARM-5 BUG #23).
# ===========================================================================

def test_extracted_mechanism_has_is_contradiction_side_field():
    """SWARM-5 BUG #23: `ExtractedMechanism` MUST have an
    `is_contradiction_side: bool` field.

    Without it, the primary and contradiction views of the same
    `MergedResult` are indistinguishable downstream — Step 13 isomorphism
    and Step 14 ranking evaluate them as independent candidates. The
    contradiction view can even beat the primary in ranking and surface
    as a "top bridge" — exactly what the system explicitly flagged as a
    contradictory hypothesis.
    """
    fields = ExtractedMechanism.model_fields
    assert "is_contradiction_side" in fields, (
        "ExtractedMechanism must declare `is_contradiction_side: bool`"
    )
    annotation = fields["is_contradiction_side"].annotation
    # Resolve Optional[bool] wrappers if present.
    from typing import get_args, get_origin
    if get_origin(annotation) is not None and bool in get_args(annotation):
        return  # bool is present, union with None OK
    assert annotation is bool, (
        f"Expected is_contradiction_side to be bool, got {annotation!r}"
    )


def test_extracted_mechanism_is_contradiction_side_defaults_to_false():
    """The field should default to False (primary view is the common case)."""
    em = _valid_extracted_mechanism()
    assert em.is_contradiction_side is False, (
        "is_contradiction_side should default to False"
    )


def test_extracted_mechanism_is_contradiction_side_can_be_true():
    """Constructing with `is_contradiction_side=True` MUST succeed."""
    em = ExtractedMechanism(
        source_result=_valid_merged_result(),
        causal_graph=CausalGraph(nodes=[], edges=[]),
        is_contradiction_side=True,
    )
    assert em.is_contradiction_side is True


# ===========================================================================
# IsomorphismThresholds ordering (SWARM-5 BUG #34).
# ===========================================================================

def test_isomorphism_thresholds_perfect_lt_strong_rejected():
    """SWARM-5 BUG #34: `IsomorphismThresholds(perfect=10, strong_partial=70)`
    MUST raise ValidationError (perfect < strong_partial is nonsensical).

    The match classifier checks `score >= perfect` first; if perfect <
    strong_partial, every score >= 10 is classified as PERFECT and the
    strong/weak branches are dead code.
    """
    with pytest.raises(ValidationError):
        IsomorphismThresholds(perfect=10.0, strong_partial=70.0, weak_partial=30.0)


def test_isomorphism_thresholds_strong_lt_weak_rejected():
    """`IsomorphismThresholds(strong_partial=30, weak_partial=70)` MUST raise."""
    with pytest.raises(ValidationError):
        IsomorphismThresholds(perfect=100.0, strong_partial=30.0, weak_partial=70.0)


def test_isomorphism_thresholds_equal_rejected():
    """`IsomorphismThresholds(perfect=70, strong_partial=70)` MUST raise
    (strict ordering required)."""
    with pytest.raises(ValidationError):
        IsomorphismThresholds(perfect=70.0, strong_partial=70.0, weak_partial=30.0)


def test_isomorphism_thresholds_valid_ordering_passes():
    """Regression check: `perfect >= strong_partial >= weak_partial` validates."""
    IsomorphismThresholds(perfect=100.0, strong_partial=70.0, weak_partial=30.0)
    IsomorphismThresholds(perfect=80.0, strong_partial=50.0, weak_partial=20.0)


def test_step14_response_threshold_ordering_validated():
    """SWARM-5 BUG #34: any threshold configuration exposed via Step14
    or Step13 must validate the perfect >= strong_partial >= weak_partial
    ordering. We exercise the canonical `IsomorphismThresholds` schema
    used by Step13 (which Step14 reuses via ranked bridges).
    """
    # Min > max analogue: perfect < strong_partial must raise.
    with pytest.raises(ValidationError):
        IsomorphismThresholds(perfect=0.5, strong_partial=0.8, weak_partial=0.3)


# ===========================================================================
# Step10Request / Step10Response — regression checks.
# ===========================================================================

def test_step10_request_default_confidence_threshold():
    """Regression check: default confidence_threshold is 80.0."""
    req = Step10Request(graph=CausalGraph(nodes=[], edges=[]))
    assert req.confidence_threshold == 80.0


def test_step10_response_with_empty_queries():
    """Regression check: Step10Response with empty queries validates."""
    res = Step10Response(queries=[])
    assert res.queries == []