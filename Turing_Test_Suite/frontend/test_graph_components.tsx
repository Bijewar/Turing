import { render, screen, fireEvent } from '@testing-library/react';
import React from 'react';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SRC_ROOT = path.resolve(HERE, '../../src');

const graphPaneSrc = fs.readFileSync(
  path.join(SRC_ROOT, 'components/graph/GraphPane.tsx'),
  'utf8',
);
const graphControlsSrc = fs.readFileSync(
  path.join(SRC_ROOT, 'components/graph/GraphControls.tsx'),
  'utf8',
);
const crossDomainBridgeSrc = fs.readFileSync(
  path.join(SRC_ROOT, 'components/graph/CrossDomainBridge.tsx'),
  'utf8',
);

// We import the components via the `@/` alias.
async function importGraphControls() {
  return (await import('@/components/graph/GraphControls')).GraphControls;
}
async function importCrossDomainBridge() {
  return (await import('@/components/graph/CrossDomainBridge')).CrossDomainBridge;
}

describe('GraphControls', () => {
  beforeEach(() => vi.resetModules());

  /**
   * SWARM-2 BUG #29 — "Causal Discovery" item must show a visually distinct
   * "completed" state when the step has advanced past 'idle' (e.g. step='simulated',
   * 'bottleneck', or 'crossdomain'). The buggy code styles it the same as
   * 'idle' (no completed color/checkmark).
   *
   * Strategy: render with step='simulated' and inspect the "1. Causal Discovery"
   * element's className — it should contain a class indicating completion
   * (e.g. 'green', 'completed', '✓', 'checkmark') rather than the idle
   * styling (text-[#666]).
   */
  test('test_completed_state_visually_distinguished', async () => {
    const GraphControls = await importGraphControls();
    const { container } = render(<GraphControls step="simulated" />);
    // Find the "1. Causal Discovery" row.
    const row = Array.from(container.querySelectorAll('div')).find(d =>
      (d.textContent || '').includes('Causal Discovery'),
    );
    expect(row).toBeTruthy();
    const cls = row!.className;

    // The buggy code applies `border-[#E0E0E0] bg-white text-[#666]` for both
    // `step === 'idle'` and `step === 'simulated'`. After fix, the step='simulated'
    // (or any past-idle step) should produce a distinct "completed" style.
    //
    // We compare: render with step='idle' and verify the className is the same
    // as step='simulated' in the buggy code (and different in the fixed code).
    const idleRender = render(<GraphControls step="idle" />);
    const idleRow = Array.from(idleRender.container.querySelectorAll('div')).find(d =>
      (d.textContent || '').includes('Causal Discovery'),
    );
    const idleCls = idleRow!.className;

    // Buggy: identical classes for idle and past-idle.
    // Fixed: classes differ — the fixed code introduces a 'completed' visual.
    expect(cls).not.toBe(idleCls);
  });
});

describe('CrossDomainBridge', () => {
  beforeEach(() => vi.resetModules());

  /**
   * SWARM-2 BUG #30 — `isomorphismScore` is assumed to be 0-1 (fraction).
   * If the backend returns a percentage (0-100), the display multiplies
   * by 100 → "7500.0%". The fix should detect values > 1 and use as-is.
   *
   * We render with score=75 (already a percentage) and assert the displayed
   * text is "75.0%" — NOT "7500.0%".
   */
  test('test_isomorphism_score_handles_percentage_input', async () => {
    const CrossDomainBridge = await importCrossDomainBridge();
    render(
      <CrossDomainBridge
        insightVisible
        bridge={{
          sourceDomain: 'Biology',
          targetDomain: 'Chemistry',
          isomorphismScore: 75,
          description: 'A test bridge',
          title: 'Test',
          evidenceTier: 'A',
        }}
      />,
    );
    // The buggy code computes (75 * 100).toFixed(1) => "7500.0%".
    // The fix should display "75.0%".
    expect(screen.getByText(/75\.0%/)).toBeTruthy();
    // And NOT "7500.0%"
    expect(screen.queryByText(/7500\.0%/)).toBeNull();
  });

  /** Same bug — fraction input should display as percentage. */
  test('test_isomorphism_score_handles_fraction_input', async () => {
    const CrossDomainBridge = await importCrossDomainBridge();
    render(
      <CrossDomainBridge
        insightVisible
        bridge={{
          sourceDomain: 'Biology',
          targetDomain: 'Chemistry',
          isomorphismScore: 0.75,
          description: 'A test bridge',
          title: 'Test',
          evidenceTier: 'A',
        }}
      />,
    );
    expect(screen.getByText(/75\.0%/)).toBeTruthy();
  });
});

describe('GraphPane (static-source checks)', () => {
  /**
   * SWARM-2 BUG #27 — GraphPane defines handleRunDiscovery, handleIdentifyBottleneck,
   * handleSearchCrossDomain but never passes them to <D3GraphEngine>.
   *
   * Static check: the <D3GraphEngine .../> JSX in GraphPane should include
   * the three onRunDiscovery/onIdentifyBottleneck/onSearchCrossDomain props.
   */
  test('test_layer_trigger_handlers_passed_to_d3graphengine', () => {
    // Find the JSX block where <D3GraphEngine .../> is rendered.
    const idx = graphPaneSrc.indexOf('<D3GraphEngine');
    expect(idx).toBeGreaterThan(-1);
    // Read 400 chars after the opening tag (long enough to capture the props).
    const block = graphPaneSrc.slice(idx, idx + 600);
    expect(block).toContain('onRunDiscovery');
    expect(block).toContain('onIdentifyBottleneck');
    expect(block).toContain('onSearchCrossDomain');
  });

  /**
   * SWARM-2 BUG #60 — progress overlay must show when Layer 1 is running
   * (status 'running' or backend's 'processing'), even when layer1Progress === 0.
   *
   * The buggy code requires `layer1Progress > 0` for the overlay:
   *   {(runState?.layer1Status === 'running' || runState?.layer1Status === 'idle')
   *    && runState?.layer1Progress !== undefined && runState.layer1Progress > 0 && ...}
   *
   * Static check: the source should NOT have the `layer1Progress > 0` guard.
   * (Or it should accept `>= 0`.)
   */
  test('test_progress_overlay_shows_when_layer1_running', () => {
    // Find the progress overlay condition in GraphPane.
    const overlayIdx = graphPaneSrc.indexOf('Causal Graph Building');
    expect(overlayIdx).toBeGreaterThan(-1);
    // Read the condition before this overlay text — backwards up to 600 chars.
    const before = graphPaneSrc.slice(Math.max(0, overlayIdx - 800), overlayIdx);
    // The buggy code has `layer1Progress > 0` somewhere in the condition.
    // The fix removes this guard (or replaces with >= 0).
    const hasProgressGuard = /layer1Progress\s*>\s*0/.test(before);
    expect(hasProgressGuard).toBe(false);
  });
});