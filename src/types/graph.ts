// src/types/graph.ts
// Matches python-engine/schemas/graph.py

export interface CausalNode {
  id: string;
  label: string;
  domain?: string | null;
  confidence: number;
  is_unknown?: boolean;
  description?: string | null;
}

export interface CausalEdge {
  source: string;
  target: string;
  relation: string;
  confidence: number;
  weight?: number;
  curvature?: number;
  label?: string;
  crossDomain?: boolean;
}

export interface CausalGraph {
  nodes: CausalNode[];
  edges: CausalEdge[];
}

export interface CausalGraphRuntime extends CausalGraph {
  overall_graph_confidence?: number;
  urgent_nodes?: { node_id: string; urgency_score: number; label: string }[];
  is_fitted?: boolean;
  routing?: string;
}

export interface UrgentNode {
  node_id: string;
  urgency_score: number;
  label: string;
}
