from typing import List, Optional
from pydantic import BaseModel, Field, model_validator

class Node(BaseModel):
    id: str
    label: str
    domain: Optional[str] = None
    confidence: float = Field(default=80.0, ge=0.0, le=100.0, description="Confidence percentage from 0 to 100")
    is_unknown: bool = Field(default=False, description="Explicitly marked as an unknown node")
    description: Optional[str] = None

class Edge(BaseModel):
    source: str
    target: str
    relation: str
    confidence: float = Field(default=80.0, ge=0.0, le=100.0, description="Confidence percentage from 0 to 100")
    weight: float = Field(default=1.0, description="Edge weight (e.g., correlation coefficient)")

class CausalGraph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

    @model_validator(mode='after')
    def validate_graph_structure(self) -> 'CausalGraph':
        node_ids = [n.id for n in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Duplicate node IDs found in graph.")
        return self
