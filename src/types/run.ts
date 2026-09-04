// src/types/run.ts
// Matches python-engine/schemas/run.py and routers/runs.py RUNS_STORE

export type LayerStatus = 'pending' | 'processing' | 'completed' | 'failed';

export interface RunState {
  runId: string;
  currentLayer: number; // 1 to 4
  layer1Status: LayerStatus;
  layer2Status: LayerStatus;
  layer3Status: LayerStatus;
  layer4Status: LayerStatus;
  layer1Progress?: number;
  progressPercentage?: number;
  layer2Progress?: number;
  layer3Progress?: number;
  layer4Progress?: number;
}
