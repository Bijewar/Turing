// src/types/layer2.ts
// Matches python-engine/schemas/layer2.py

export type AgentRole = 'explorer' | 'exploiter' | 'contrarian' | 'skeptic';

export interface AgentEntry {
  id: string;
  type: AgentRole;
  agentId: string;
  y: number;
  content: string;
  delayMs: number;
}

export type HypothesisNodeVariant = 'normal' | 'falsified' | 'confirmed' | 'synthesized' | 'exploiter';

export interface HypothesisNode {
  id: string;
  variant: HypothesisNodeVariant;
  label: string;
  title: string;
  description: string;
  x: number;
  y: number;
  delayMs?: number;
  mechanism?: string;
}

export type ConnectorLineVariant = 'explorer' | 'contrarian' | 'exploiter' | 'skeptic';

export interface ConnectorLine {
  id: string;
  variant: ConnectorLineVariant;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  delayMs?: number;
}

export interface Layer2State {
  status: string;
  nodesComputed: number;
  agents: AgentEntry[];
  heatmapNodes: HypothesisNode[];
  heatmapLines: ConnectorLine[];
}
