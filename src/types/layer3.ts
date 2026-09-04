// src/types/layer3.ts
// Matches python-engine/schemas/layer3.py & routers/runs.py

// Canonical Python enum kind
export type SearchSourceKind = "PAPER" | "WIKIPEDIA" | "WEB" | "PATENT";

// Stream source preview item from SSE stream
export interface SearchSource {
  src: string;
  title: string;
  match: number;
}

export type StreamSourcePreview = SearchSource;

// SSE runtime shape (matches routers/runs.py layer3Data.topBridges)
export interface BridgeResult {
  sourceDomain: string;
  targetDomain: string;
  isomorphismScore: number;
  description: string;
  title: string;
  evidenceTier: "HIGH" | "MEDIUM" | "LOW" | string;
}

export type BridgeResultStream = BridgeResult;

// Canonical Python schema models (schemas/layer3.py)
export interface ValidityScores {
  structural_match: number;
  constraint_compatibility: number;
  solution_transferability: number;
  evidence_strength: number;
  final_score: number;
}

export interface ExtractedMechanism {
  id: string;
  domain: string;
  name: string;
  graph: any;
  confidence: number;
  source_kind: SearchSourceKind;
}

export interface IsomorphismMatch {
  mechanism: ExtractedMechanism;
  isomorphism_score: number;
  match_type: "PERFECT" | "STRONG_PARTIAL" | "WEAK_PARTIAL" | "DISCARDED";
}

export interface RankedBridge {
  match: IsomorphismMatch;
  scores: ValidityScores;
}

export type Layer3Status =
  | "Waiting..."
  | "Searching Literature..."
  | "Extracting Graphs..."
  | "Matching Isomorphisms..."
  | "Ranking Bridges..."
  | "Completed Bridge Discovery"
  | `Failed: ${string}`
  | string;

export interface Layer3State {
  status: Layer3Status;
  query: string;
  streamSources: SearchSource[];
  topBridges: BridgeResult[];
  isLocked: boolean;
}
