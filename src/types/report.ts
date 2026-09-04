// src/types/report.ts
// SSE runtime shape (camelCase) returned by GET /api/runs/{runId}/layer4/report/stream
// Matches python-engine/routers/runs.py layer4Data

export interface Layer4Report {
  isComplete: boolean;

  // Section 1 — The Mechanism (plain-English cause chain + confidence badges)
  theMechanism: string;

  // Section 2 — The Experiment (specific next experiment to run)
  theExperiment: string;

  // Section 3 — Who Already Solved This (top 3 bridge analogies)
  whoSolvedThis: string;

  // Section 4 — Warnings & Conflicts (⚡ contradictions, ⚠️ low data)
  warningsAndConflicts: string[];

  // Section 5 — Next 3 Actions (ranked, specific, actionable)
  next3Actions: string[];

  // Metadata
  confidenceDisclaimer: string;
  // Legacy fields also sent by backend
  mechanismExplanation?: string;
  bridgeSummary?: string;
  experimentResults?: string;
  warnings?: string[];
}

// Canonical Python schema (snake_case) returned by POST /api/layer4/report
// Matches python-engine/schemas/report.py
export interface FinalReport {
  run_id: string;
  the_mechanism: string;
  the_experiment: string;
  who_solved_this: string;
  warnings_and_conflicts: string[];
  next_3_actions: string[];
  top_bridges: unknown[];
  confidence_disclaimer: string;
  confidence_score?: number | null;
}
