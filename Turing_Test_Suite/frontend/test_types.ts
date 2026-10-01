import { describe, test, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SRC_ROOT = path.resolve(HERE, '../../src');

const runTsSrc = fs.readFileSync(path.join(SRC_ROOT, 'types/run.ts'), 'utf8');
const graphTsSrc = fs.readFileSync(path.join(SRC_ROOT, 'types/graph.ts'), 'utf8');
const reportTsSrc = fs.readFileSync(path.join(SRC_ROOT, 'types/report.ts'), 'utf8');

// Also read the Python schemas for cross-language drift checks.
const PY_ROOT = path.resolve(HERE, '../../python-engine/schemas');
const pyReportSrc = fs.readFileSync(path.join(PY_ROOT, 'report.py'), 'utf8');
const pyGraphSrc = fs.readFileSync(path.join(PY_ROOT, 'graph.py'), 'utf8');

/**
 * test_types.ts — Vitest tests for src/types/*.ts
 *
 * Most tests are STATIC source checks because TypeScript types are erased at
 * runtime — a `CausalNode` interface has no runtime representation. We
 * inspect the source of the type files and the Python schemas to verify the
 * TS types match the backend contract.
 */

describe('types/run.ts', () => {
  /**
   * SWARM-7 BUG #1 — `LayerStatus` type is `'idle' | 'running' | 'completed'
   * | 'error'` but the backend uses `'pending' | 'processing' | 'completed'
   * | 'failed'`. Every layer status field in the frontend type is wrong.
   *
   * Static-source check: the union must be the backend's actual values.
   */
  test('test_layer_status_matches_backend_values', () => {
    // Buggy:
    //   export type LayerStatus = 'idle' | 'running' | 'completed' | 'error';
    // Fixed:
    //   export type LayerStatus = 'pending' | 'processing' | 'completed' | 'failed';
    const match = runTsSrc.match(
      /export\s+type\s+LayerStatus\s*=\s*([^;]+);/,
    );
    expect(match).toBeTruthy();
    const union = match![1];

    // Should NOT contain the buggy values.
    expect(union).not.toContain("'idle'");
    expect(union).not.toContain("'running'");
    expect(union).not.toContain("'error'");

    // Should contain the backend values.
    expect(union).toContain("'pending'");
    expect(union).toContain("'processing'");
    expect(union).toContain("'completed'");
    expect(union).toContain("'failed'");
  });
});

describe('types/graph.ts', () => {
  /**
   * SWARM-7 BUG #4 — `CausalNode` type has `type`, `value`, `beta` fields
   * that don't exist in the Python `Node` schema, and is MISSING `confidence`,
   * `is_unknown`, `domain`, `description` from the Python schema. The most
   * critical missing field is `confidence` (used by ConfidencePanel).
   *
   * Static-source check: the `CausalNode` interface must declare a
   * `confidence: number` (required, not optional).
   */
  test('test_causal_node_has_confidence_field', () => {
    // Find the CausalNode interface block.
    const idx = graphTsSrc.indexOf('export interface CausalNode');
    expect(idx).toBeGreaterThan(-1);
    const block = graphTsSrc.slice(idx, idx + 500);
    // Extract just the interface body (up to the closing `}`).
    const body = block.slice(0, block.indexOf('}') + 1);
    // Required field (no `?`).
    expect(body).toMatch(/confidence\s*:\s*number/);
    // Optional `confidence?` is NOT enough — it must be required.
    expect(body).not.toMatch(/confidence\s*\?\s*:\s*number/);
  });

  /**
   * SWARM-7 BUG #4 (continued) — `CausalNode` declares `type`, `value`,
   * `beta` which the Python `Node` schema does NOT have. These are bogus
   * fields that mislead consumers.
   *
   * Static-source check: the `CausalNode` interface must NOT declare
   * `type`, `value`, or `beta` fields (the Python Node schema doesn't
   * have them).
   */
  test('test_causal_node_does_not_have_bogus_fields', () => {
    const idx = graphTsSrc.indexOf('export interface CausalNode');
    expect(idx).toBeGreaterThan(-1);
    const block = graphTsSrc.slice(idx, idx + 500);
    const body = block.slice(0, block.indexOf('}') + 1);
    expect(body).not.toMatch(/^\s*type\s*:/m);
    expect(body).not.toMatch(/^\s*value\s*:/m);
    expect(body).not.toMatch(/^\s*beta\s*:/m);
  });

  /**
   * SWARM-7 BUG #5 — `CausalEdge.relation` is optional in TS (`relation?: string`)
   * but REQUIRED in the Python `Edge` schema (`relation: str`).
   *
   * Static-source check: the `CausalEdge` interface must declare `relation`
   * as required (no `?`).
   */
  test('test_causal_edge_relation_required', () => {
    const idx = graphTsSrc.indexOf('export interface CausalEdge');
    expect(idx).toBeGreaterThan(-1);
    const block = graphTsSrc.slice(idx, idx + 500);
    const body = block.slice(0, block.indexOf('}') + 1);
    // Required field (no `?`).
    expect(body).toMatch(/relation\s*:\s*string/);
    // Not optional.
    expect(body).not.toMatch(/relation\s*\?\s*:\s*string/);
  });
});

describe('types/report.ts', () => {
  /**
   * SWARM-7 BUG #3 — `Layer4Report` type claims to match
   * python-engine/schemas/report.py but uses camelCase while the Python
   * `FinalReport` uses snake_case. Two parallel "FinalReport" shapes exist:
   *   - SSE runtime shape (camelCase) returned by
   *     `GET /api/runs/{runId}/layer4/report/stream`
   *   - REST shape (snake_case) returned by `POST /api/layer4/report`
   *
   * The TS type only matches the SSE shape; REST consumers silently get
   * `undefined` for every field.
   *
   * Strategy: inspect the Python FinalReport schema for snake_case field
   * names, then assert the TS type (Layer4Report) has matching snake_case
   * fields OR that there's a separate FinalReport type for the REST shape.
   *
   * Static-source checks:
   *   1. The Python schema defines fields like `run_id`, `the_mechanism`, etc.
   *   2. The TS file must define a `FinalReport` (REST shape) with snake_case
   *      fields, OR `Layer4Report` must use snake_case fields.
   */
  test('test_layer4_report_matches_python_final_report', () => {
    // First, verify the Python schema uses snake_case.
    expect(pyReportSrc).toMatch(/run_id\s*:\s*str/);
    expect(pyReportSrc).toMatch(/the_mechanism\s*:/);
    expect(pyReportSrc).toMatch(/the_experiment\s*:/);
    expect(pyReportSrc).toMatch(/who_solved_this\s*:/);
    expect(pyReportSrc).toMatch(/warnings_and_conflicts\s*:/);
    expect(pyReportSrc).toMatch(/next_3_actions\s*:/);
    expect(pyReportSrc).toMatch(/confidence_disclaimer\s*:/);
    expect(pyReportSrc).toMatch(/top_bridges\s*:/);

    // Now check that the TS report.ts either:
    //   (a) has a FinalReport interface with snake_case fields, OR
    //   (b) Layer4Report uses snake_case fields.
    // Buggy: only camelCase Layer4Report, no snake_case FinalReport.
    // Fixed: either add a FinalReport (snake_case) OR change Layer4Report
    //   to snake_case.
    const hasSnakeCaseFinalReport = /export\s+interface\s+FinalReport\b/.test(
      reportTsSrc,
    );
    const hasSnakeCaseFields =
      /\brun_id\s*:/.test(reportTsSrc) ||
      /\bthe_mechanism\s*:/.test(reportTsSrc) ||
      /\bwarnings_and_conflicts\s*:/.test(reportTsSrc) ||
      /\bnext_3_actions\s*:/.test(reportTsSrc) ||
      /\bconfidence_disclaimer\s*:/.test(reportTsSrc) ||
      /\btop_bridges\s*:/.test(reportTsSrc);

    expect(hasSnakeCaseFinalReport || hasSnakeCaseFields).toBe(true);

    // The buggy file's comment claims it matches the Python schema but it
    // only matches the SSE runtime shape. The fix should either remove the
    // misleading comment or add the missing FinalReport (REST) type.
    // We assert the misleading comment is updated to NOT claim match with
    // the Python schema if it doesn't actually match.
    // (We only enforce that the snake_case fields exist — see above.)
  });
});