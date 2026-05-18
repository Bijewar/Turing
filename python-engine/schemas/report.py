"""
schemas/report.py

FinalReport schema — 5 sections matching the README specification exactly.
"""
from typing import List, Optional, Annotated
from uuid import UUID
from pydantic import BaseModel, Field
from .layer3 import RankedBridge

# Test compatibility: legacy string id alias
# run_id: str
# the_mechanism: str
# the_experiment: str
# who_solved_this: str
# warnings_and_conflicts: str
# next_3_actions: str
# confidence_disclaimer: str
# top_bridges: str


class FinalReport(BaseModel):
    """
    The structured output of Layer 4.
    Five flat-string sections that map directly to what the README calls for.
    All LLM fields MUST be plain strings — never nested dicts.
    """
    run_id: UUID

    # Section 1 — THE MECHANISM
    # Plain English cause chain + inline confidence badges (✅ ⚠️ ❌ ⚡)
    the_mechanism: str = Field(
        ...,
        min_length=1,
        description=(
            "Plain-English explanation of what is actually causing what. "
            "Must include inline confidence markers: ✅ High / ⚠️ Medium / ❌ Low / ⚡ Contradicted. "
            "Never mention algorithm names."
        )
    )

    # Section 2 — THE EXPERIMENT
    # Exact lab values to test next. Always present (data or text path).
    the_experiment: str = Field(
        ...,
        min_length=1,
        description=(
            "The single most important next experiment the researcher should run. "
            "For data paths: include specific parameter values, expected outcome, and which agent/optimizer identified this region. "
            "For text paths: describe a conceptual experiment or literature search to resolve the key uncertainty. "
            "Be specific and actionable."
        )
    )

    # Section 3 — WHO ALREADY SOLVED THIS
    # Narrative analogies for each bridge, not just titles
    who_solved_this: str = Field(
        ...,
        min_length=1,
        description=(
            "Top 3 cross-domain bridges written as human analogies. "
            "For each: Source domain, field, match %, evidence level, and a vivid explanation of how "
            "their solution maps to the researcher's problem. Explain the structural analogy clearly."
        )
    )

    # Section 4 — WARNINGS AND CONFLICTS (raw list, rendered individually in UI)
    warnings_and_conflicts: List[Annotated[str, Field(max_length=500)]] = Field(
        default_factory=list,
        max_length=100,
        description=(
            "List of plain-string warnings. Each entry is one of: "
            "'⚡ CONTRADICTION: ...' or '⚠️ LOW DATA: ...' or '❌ LOW CONFIDENCE: ...'. "
            "Empty list if no issues found."
        )
    )

    # Section 5 — NEXT 3 ACTIONS
    next_3_actions: List[Annotated[str, Field(max_length=500)]] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "Exactly 3 ranked, specific, actionable next steps for the researcher. "
            "Action 1 = most important. Action 2 = if Action 1 confirms. Action 3 = parallel track. "
            "Each is a plain string, e.g. 'Run experiment at temp=68, compound=0.047 — resolves 71% ambiguity'."
        )
    )

    # Metadata fields — not rendered as sections
    top_bridges: List[RankedBridge] = Field(default_factory=list)
    confidence_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Overall confidence score between 0.0 and 1.0"
    )
    confidence_disclaimer: str = Field(default="")
