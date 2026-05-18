"""
schemas/layer3.py — Layer 3 Data Models and Pydantic Schemas

Defines request/response schemas, validation rules, and domain models
for cross-domain search, mechanism extraction, isomorphism matching,
and bridge ranking.
"""
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator
from .graph import CausalGraph


class StructuralQuery(BaseModel):
    original_node_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Target node identifier"
    )
    original_confidence: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Original node confidence score (0.0 to 100.0)"
    )
    structural_description: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The domain-blind structural description of the problem"
    )


class Step10Request(BaseModel):
    graph: CausalGraph
    confidence_threshold: float = Field(
        default=80.0,
        ge=0.0,
        le=100.0,
        description="Nodes with confidence below this threshold are extracted"
    )


class Step10Response(BaseModel):
    queries: List[StructuralQuery] = Field(
        default_factory=list,
        description="Extracted domain-blind structural queries"
    )


class SearchSource(str, Enum):
    PAPER = "PAPER"
    WIKIPEDIA = "WIKIPEDIA"
    WEB = "WEB"
    PATENT = "PATENT"


class SearchResult(BaseModel):
    source: SearchSource
    title: str = Field(
        default="Untitled",
        max_length=500,
        description="Title of the paper, article, or patent"
    )
    summary: str = Field(
        default="",
        max_length=10000,
        description="Summary or abstract of the finding"
    )
    url: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="URL link to the source document"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Initial confidence score from this single source (0.0 to 100.0)"
    )
    original_query: StructuralQuery
    citation_count: int = Field(
        default=0,
        ge=0,
        description="Number of times this paper/patent has been cited"
    )
    replication_count: int = Field(
        default=0,
        ge=0,
        description="Number of times this study has been independently replicated"
    )
    deployment_status: str = Field(
        default="unknown",
        max_length=100,
        description="Production deployment evidence: 'deployed', 'replicated', 'single_study', 'blog', 'unknown'"
    )


class DebateTranscript(BaseModel):
    proposer_argument: str = Field(
        default="",
        description="Proposer agent's structured hypothesis argument"
    )
    skeptic_rebuttal: str = Field(
        default="",
        description="Skeptic agent's counter-evidence or acceptance rebuttal"
    )
    synthesizer_consensus: str = Field(
        default="",
        description="Synthesizer agent's final consensus resolution"
    )


class ContradictionDetails(BaseModel):
    conflict_detected: bool = Field(
        default=False,
        description="Whether a directional or mechanistic contradiction was detected between sources"
    )
    nature_of_conflict: str = Field(
        default="None",
        max_length=2000,
        description="Description of the disagreement between sources"
    )
    debate_log: DebateTranscript = Field(
        default_factory=lambda: DebateTranscript(
            proposer_argument="",
            skeptic_rebuttal="",
            synthesizer_consensus=""
        ),
        description="Transcript of the 3-agent adversarial debate"
    )
    recommended_experiment: str = Field(
        default="Proceed with standard implementation.",
        max_length=2000,
        description="Suggested experiment to empirically resolve the conflict"
    )
    contradiction_risk: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Estimated contradiction risk score bounded between 0.0 and 1.0"
    )


class MergedResult(BaseModel):
    title: str = Field(
        default="Untitled",
        max_length=500,
        description="Consolidated title representing the clustered mechanism"
    )
    merged_summary: str = Field(
        default="",
        max_length=20000,
        description="Synthesized multi-source summary of the mechanism"
    )
    underlying_mechanism: str = Field(
        default="",
        max_length=5000,
        description="Underlying domain-blind causal mechanism description"
    )
    sources: List[SearchSource] = Field(
        default_factory=list,
        description="List of distinct source types contributing to this finding"
    )
    source_results: List[SearchResult] = Field(
        default_factory=list,
        description="Raw search results merged into this finding"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Boosted confidence score due to combined evidence (0.0 to 100.0)"
    )
    contradiction_analysis: Optional[ContradictionDetails] = Field(
        default=None,
        description="Output of the Step 11.5 3-Agent Debate"
    )


class Step11Response(BaseModel):
    query: StructuralQuery
    results: List[MergedResult] = Field(
        default_factory=list,
        description="Deduplicated and merged search results"
    )


class Step12Request(BaseModel):
    merged_results: List[MergedResult] = Field(
        default_factory=list,
        description="Merged search results to extract causal mechanisms from"
    )


class ExtractedMechanism(BaseModel):
    source_result: MergedResult
    causal_graph: CausalGraph
    is_contradiction_side: bool = Field(
        default=False,
        description="True if this mechanism is the contradiction/negative side of a debate"
    )


class Step12Response(BaseModel):
    extracted_mechanisms: List[ExtractedMechanism] = Field(
        default_factory=list,
        description="List of extracted causal mechanisms with graphs"
    )


class MatchType(str, Enum):
    PERFECT = "PERFECT"
    STRONG_PARTIAL = "STRONG_PARTIAL"
    WEAK_PARTIAL = "WEAK_PARTIAL"
    DISCARDED = "DISCARDED"


class IsomorphismMatch(BaseModel):
    mechanism: ExtractedMechanism
    isomorphism_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Structural similarity score (0.0 to 100.0)"
    )
    match_type: MatchType


class IsomorphismThresholds(BaseModel):
    perfect: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Score threshold for PERFECT match"
    )
    strong_partial: float = Field(
        default=70.0,
        ge=0.0,
        le=100.0,
        description="Score threshold for STRONG_PARTIAL match"
    )
    weak_partial: float = Field(
        default=30.0,
        ge=0.0,
        le=100.0,
        description="Score threshold for WEAK_PARTIAL match"
    )

    @model_validator(mode="after")
    def _check_ordering(self):
        if not (self.perfect > self.strong_partial > self.weak_partial):
            raise ValueError(
                "Thresholds must satisfy perfect > strong_partial > weak_partial; "
                f"got perfect={self.perfect}, strong_partial={self.strong_partial}, weak_partial={self.weak_partial}"
            )
        return self


class Step13Request(BaseModel):
    target_graph: CausalGraph
    candidates: List[ExtractedMechanism] = Field(
        default_factory=list,
        description="Candidate mechanisms to match against the target graph"
    )
    thresholds: IsomorphismThresholds = Field(
        default_factory=IsomorphismThresholds,
        description="Configurable match classification thresholds"
    )


class Step13Response(BaseModel):
    matches: List[IsomorphismMatch] = Field(
        default_factory=list,
        description="List of classified isomorphism matches"
    )


class ValidityScores(BaseModel):
    structural_match: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Structural isomorphism score normalized to 0.0-1.0"
    )
    constraint_compatibility: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Domain constraint compatibility score (0.0-1.0)"
    )
    solution_transferability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Solution transferability score (0.0-1.0)"
    )
    evidence_strength: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Empirical evidence strength score (0.0-1.0)"
    )
    final_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Composite bridge validity score (0.0-1.0)"
    )


class RankedBridge(BaseModel):
    match: IsomorphismMatch
    scores: ValidityScores


class Step14Request(BaseModel):
    matches: List[IsomorphismMatch] = Field(
        default_factory=list,
        description="Candidate isomorphism matches to rank"
    )
    user_domain_context: str = Field(
        default="",
        max_length=5000,
        description="The domain-blind structural description from Step 10, used as LLM evaluation context"
    )


class Step14Response(BaseModel):
    top_bridges: List[RankedBridge] = Field(
        default_factory=list,
        description="Top ranked cross-domain candidate bridges"
    )
