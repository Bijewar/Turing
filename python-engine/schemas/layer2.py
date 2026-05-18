"""
schemas/layer2.py

Data models for Layer 2 Agent Simulation and Bayesian Optimization.
Combines granular step-by-step models (Sub_Manas) with orchestration models (main).
"""
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator
from .graph import CausalGraph


# ==============================================================================
# 0. SHARED & ENUM MODELS
# ==============================================================================

class InputType(str, Enum):
    DATA = "data"
    CSV = "csv"
    TEXT = "text"


class PipelineMode(str, Enum):
    SIMULATION = "SIMULATION_MODE"
    LITERATURE = "LITERATURE_MODE"


class ModeDetectorInput(BaseModel):
    graph_data: Dict[str, Any] = Field(description="The causal graph topology from Layer 1")
    unknown_nodes: List[Dict[str, Any]] = Field(default_factory=list, description="List of unknown nodes with rankings")
    input_type: InputType = Field(description="Type of input data processed in Layer 1 (data or text)")


class ModeDetectorOutput(BaseModel):
    mode: PipelineMode = Field(description="The detected pipeline mode based on input type")
    message: str = Field(description="Explanation of the routing decision")


class AgentType(str, Enum):
    EXPLORER = "explorer"
    EXPLOITER = "exploiter"
    CONTRARIAN = "contrarian"
    SKEPTIC = "skeptic"


class AgentEntry(BaseModel):
    id: str
    type: AgentType
    agentId: str
    y: float
    content: str
    delayMs: int = 0


class OptimizerState(BaseModel):
    targetMaximizationPercentage: float = Field(..., ge=0.0, le=100.0)
    basePoint: Optional[Dict[str, float]] = None
    iteration: Optional[int] = None
    bestIntervention: Optional[str] = None


# ==============================================================================
# 1. ORCHESTRATION MODELS (main)
# ==============================================================================

class AgentAction(BaseModel):
    agent_id: str
    agent_role: str = Field(..., description="'explorer', 'exploiter', or 'contrarian'")
    think: str = Field(..., description="ReAct THINK step — the agent's reasoning")
    decide: str = Field(..., description="ReAct DECIDE step — the chosen intervention")
    act: str = Field(..., description="ReAct ACT step — the executed action and result")
    expected_improvement: float = Field(..., description="Bayesian Expected Improvement score")


class IterationResult(BaseModel):
    """(Formerly SimulationResult in main branch) Result of one full orchestration iteration."""
    iteration: int
    agent_actions: List[AgentAction]
    best_intervention: str
    predicted_outcome: float = Field(..., description="Predicted value after intervention")
    confidence: float = Field(..., ge=0.0, le=100.0)


class SearchSpace(BaseModel):
    """Defines the actionable bounds (min/max) for a controllable parameter."""
    min: float
    max: float

    @model_validator(mode='after')
    def validate_min_max(self) -> 'SearchSpace':
        if self.min >= self.max:
            raise ValueError(f"SearchSpace min ({self.min}) must be strictly less than max ({self.max})")
        return self


class Layer2Request(BaseModel):
    graph: CausalGraph
    target_node_id: str = Field(..., min_length=1, description="The node we are trying to optimise")
    max_iterations: int = Field(default=10, ge=1, le=50, description="Maximum Bayesian optimisation rounds")
    domain_config: Optional[Dict[str, SearchSpace]] = None


class Layer2Response(BaseModel):
    final_graph: CausalGraph
    simulation_results: List[IterationResult]
    best_intervention: str
    confidence: float


# ==============================================================================
# 2. GRANULAR STEP MODELS (Sub_Manas)
# ==============================================================================

class GraphEdge(BaseModel):
    """Represents a directed causal link between two nodes."""
    source: str
    target: str
    weight: Optional[float] = None
    confidence: Optional[float] = None


class VariableIdentifierInput(BaseModel):
    nodes: List[str] = Field(default_factory=list, description="List of node names")
    edges: List[GraphEdge] = Field(default_factory=list, description="List of directed edges")
    domain_config: Optional[Dict[str, SearchSpace]] = Field(default=None, description="Optional min/max config for source nodes")


class IdentifiedNode(BaseModel):
    name: str
    search_space: Optional[SearchSpace] = None


class VariableIdentifierOutput(BaseModel):
    source_nodes: List[IdentifiedNode] = Field(description="Controllable variables (no incoming edges)")
    sink_nodes: List[str] = Field(description="Outcomes to maximize (no outgoing edges)")
    intermediate_nodes: List[str] = Field(description="Nodes with both incoming and outgoing edges")


class GaussianPrediction(BaseModel):
    """Represents a normal distribution for a predicted node outcome."""
    mean: float
    std_dev: float = Field(ge=0.0)


class SimulationStepInput(BaseModel):
    nodes: List[str]
    edges: List[GraphEdge]
    source_values: Dict[str, float] = Field(description="Intervention values for source nodes (e.g. temperature=68)")


class SimulationStepOutput(BaseModel):
    predictions: Dict[str, GaussianPrediction] = Field(description="Mean and std_dev for every node in the graph")


class AgentProposal(BaseModel):
    """A proposed intervention strategy from a specialized agent (Explorer/Exploiter/Contrarian)."""
    agent_name: str
    proposed_values: Dict[str, float]
    justification: str


class SimulationResult(BaseModel):
    """The evaluated outcome of an agent's proposed intervention."""
    agent_name: str
    yield_prediction: GaussianPrediction
    ambiguity_reduction: float
    is_winner: bool = False


class RoundInput(BaseModel):
    round_number: int
    nodes: List[str]
    edges: List[GraphEdge]
    historical_data: List[Dict[str, Any]] = Field(default_factory=list, description="List of past successful simulation results to fit GP")
    domain_config: Optional[Dict[str, SearchSpace]] = None
    sink_node: Optional[str] = Field(default=None, description="Name of output node to maximize. Auto-detected if None.")


class RoundHistory(BaseModel):
    round_number: int
    winner_agent: Optional[str] = None  # None if all proposals returned 0 yield
    tested_values: Dict[str, float]
    yield_result: float
    ambiguity_score: float
    all_results: List[SimulationResult]


class HeatmapPoint(BaseModel):
    """A single coordinate point on the generated causal heatmap."""
    x_val: float
    y_val: float
    z_val: float = Field(description="The predicted yield")
    is_cliff: bool = Field(default=False, description="Flagged true if yield drops suddenly")
    value: Optional[float] = None
    color: Optional[float] = None

    def model_post_init(self, __context: Any) -> None:
        if self.value is None:
            self.value = self.z_val
        if self.color is None:
            self.color = max(0.0, min(1.0, self.z_val / 100.0 if self.z_val > 1.0 else self.z_val))


class HeatmapInput(BaseModel):
    nodes: List[str] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    domain_config: Dict[str, SearchSpace] = Field(default_factory=dict)
    historical_data: List[Dict[str, Any]] = Field(default_factory=list)
    resolution: int = Field(default=10, ge=2, le=200, description="Number of points per axis (min 2, max 200, default 10 = 10x10 grid)")
    sink_node: Optional[str] = Field(default=None, description="Name of output node to maximize. Auto-detected if None.")
    cliff_sigma: float = Field(default=1.5, gt=0.0, description="Cliff threshold: points below (mean - cliff_sigma * std) are flagged")


class HeatmapOutput(BaseModel):
    x_label: str
    y_label: str
    is_1d: bool = Field(default=False, description="True when only one source node exists (axes are the same variable)")
    data: List[HeatmapPoint]

    @property
    def points(self) -> List[HeatmapPoint]:
        return self.data

    @property
    def cells(self) -> List[HeatmapPoint]:
        return self.data

    @property
    def matrix(self) -> List[List[float]]:
        # Reshape data into 2D grid matrix if possible
        n = int(len(self.data) ** 0.5)
        if n * n == len(self.data) and n > 0:
            return [[self.data[r * n + c].z_val for c in range(n)] for r in range(n)]
        return [[p.z_val for p in self.data]]


class UnexploredZone(BaseModel):
    """A mathematically uncertain but highly promising contiguous gap in the parameter space."""
    parameter_name: str
    range_min: float
    range_max: float
    predicted_yield: float
    uncertainty: float
    message: str


class ZoneFinderInput(BaseModel):
    nodes: List[str] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    domain_config: Dict[str, SearchSpace] = Field(default_factory=dict)
    historical_data: List[Dict[str, Any]] = Field(default_factory=list, description="Past simulation results; defaults to empty list")
    sink_node: Optional[str] = Field(default=None, description="Name of output node to maximize. Auto-detected if None.")
    gap_threshold: float = Field(default=0.2, ge=0.0, le=1.0, description="Minimum percentage of space (0.0 to 1.0) to be considered a gap")


class ZoneFinderOutput(BaseModel):
    unexplored_zones: List[UnexploredZone]