import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SRC_ROOT = path.resolve(HERE, '../../src');

const heatmapSrc = fs.readFileSync(
  path.join(SRC_ROOT, 'components/layer2/Heatmap.tsx'),
  'utf8',
);
const bestFoundSrc = fs.readFileSync(
  path.join(SRC_ROOT, 'components/layer2/BestFoundPanel.tsx'),
  'utf8',
);
const agentStatusSrc = fs.readFileSync(
  path.join(SRC_ROOT, 'components/layer2/AgentStatusPanel.tsx'),
  'utf8',
);

async function importHeatmap() {
  return (await import('@/components/layer2/Heatmap')).Heatmap;
}
async function importBestFoundPanel() {
  return (await import('@/components/layer2/BestFoundPanel')).BestFoundPanel;
}
async function importAgentStatusPanel() {
  return (await import('@/components/layer2/AgentStatusPanel')).AgentStatusPanel;
}

describe('Heatmap', () => {
  beforeEach(() => vi.resetModules());

  /**
   * SWARM-2 BUG #36 — the SVG must have a `viewBox` attribute to scale
   * responsively. The buggy code has `<svg className="absolute inset-0
   * w-full h-[100%] min-h-screen pointer-events-none z-10">` with NO viewBox.
   */
  test('test_svg_has_viewbox', async () => {
    const Heatmap = await importHeatmap();
    const { container } = render(<Heatmap nodes={[]} lines={[]} />);
    const svg = container.querySelector('svg');
    expect(svg).toBeTruthy();
    expect(svg!.getAttribute('viewBox')).toBeTruthy();
  });

  /**
   * SWARM-2 BUG #37 — fixed-width nodes (w-[500px], w-[600px]) overflow
   * horizontally on a 375px mobile viewport. The fix should use
   * `max-w-[Npx] w-full` or similar so content doesn't overflow.
   *
   * We render the Heatmap with a few nodes and assert that no element has
   * a fixed width that would overflow at 375px (i.e., no `w-[500px]` or
   * `w-[600px]` classes that aren't paired with `max-w-`).
   *
   * Static-source check: the buggy code has `w-[500px]` or `w-[600px]` and
   * no `max-w-` on the same element. The fix uses `max-w-[Npx] w-full`.
   */
  test('test_responsive_on_mobile', () => {
    // Find any `w-[500px]` or `w-[600px]` that aren't paired with `max-w-`.
    // Buggy source has these in lines 18, 33, 43, 53, 67.
    // The fix should either replace with `max-w-[500px] w-full` or omit.
    const lines = heatmapSrc.split('\n');
    const buggyFixedWidths = lines.filter(l =>
      /w-\[(500|600)px\]/.test(l) && !/max-w-/.test(l),
    );
    expect(buggyFixedWidths.length).toBe(0);
  });

  /**
   * SWARM-2 BUG #38 — the SVG uses `min-h-screen` which is `100vh`. On iOS
   * Safari, `100vh` includes the address bar and is wrong on mobile. The
   * fix should use `min-h-[100dvh]` (dynamic viewport height) instead.
   *
   * Static-source check: the source should NOT contain `min-h-screen` on
   * SVG elements (or anywhere that affects responsive height). Instead
   * it should use `dvh`.
   */
  test('test_uses_dvh_not_vh', () => {
    // The buggy source has `min-h-screen` on the SVG (line 79).
    // The fix uses `min-h-[100dvh]` or similar.
    expect(heatmapSrc).not.toContain('min-h-screen');
    // Look for a `dvh` reference somewhere — the fix should introduce it.
    expect(/dvh/.test(heatmapSrc)).toBe(true);
  });

  /**
   * SWARM-2 BUG #59 — `HypothesisNodeView` renders `id={id}` which can
   * produce duplicate DOM IDs if two nodes share an `id`. Duplicate IDs
   * are invalid HTML and break `getElementById` queries.
   *
   * The fix should use `data-node-id` (or suffix the id) to avoid DOM id
   * collisions while still exposing the id for selection/tests.
   */
  test('test_no_duplicate_dom_ids', async () => {
    const Heatmap = await importHeatmap();
    // Two nodes with the SAME id — buggy code would render duplicate DOM IDs.
    const nodes = [
      { id: 'dup', variant: 'normal' as const, label: 'A', title: 'A', description: 'd', x: 0, y: 0 },
      { id: 'dup', variant: 'normal' as const, label: 'B', title: 'B', description: 'd', x: 100, y: 100 },
    ];
    const { container } = render(<Heatmap nodes={nodes} lines={[]} />);
    // Count DOM elements with id="dup".
    const dupCount = container.querySelectorAll('[id="dup"]').length;
    // The buggy code renders TWO elements with id="dup". The fix should
    // either error or use data-node-id (so [id="dup"] count is 0 or 1).
    expect(dupCount).toBeLessThanOrEqual(1);
  });
});

describe('BestFoundPanel', () => {
  beforeEach(() => vi.resetModules());

  /**
   * SWARM-2 BUG #40 — the component destructures required fields
   * (`targetMaximizationPercentage`, `expectedImprovement`) without
   * validation. If the backend omits one of them, the destructure throws
   * a TypeError when calling `.toFixed()`.
   *
   * We render with a partial state (missing `targetMaximizationPercentage`
   * and `expectedImprovement`) and assert no crash — the panel should
   * display 0 or N/A.
   */
  test('test_handles_partial_optimizer_state', async () => {
    const BestFoundPanel = await importBestFoundPanel();
    const partial = {
      parameters: {},
      roundsComputed: 0,
      maxRounds: 10,
      // NOTE: targetMaximizationPercentage and expectedImprovement MISSING.
    } as any;
    // Render must not throw.
    const { container } = render(<BestFoundPanel optimizerState={partial} />);
    expect(container).toBeTruthy();
    // The panel should render some content (either 0%, N/A, or default 0).
    const text = container.textContent || '';
    // Should NOT throw an exception, and should show some sensible default.
    expect(text.length).toBeGreaterThan(0);
  });
});

describe('AgentStatusPanel', () => {
  beforeEach(() => vi.resetModules());

  /**
   * SWARM-2 BUG #41 — `Math.max(...nodes.map(n => n.delayMs || 0), 0)` uses
   * spread to pass an array to `Math.max`. With a huge array (> ~125k
   * elements), this overflows the call stack and throws `RangeError`.
   *
   * We render with 200k heatmap nodes and assert no RangeError is thrown.
   *
   * To make this fast and avoid React's reconciliation cost, we mock
   * useAgentLoop to return a state with a huge array but few agents.
   */
  test('test_math_max_does_not_stackoverflow', async () => {
    // Use a module-level state holder pattern (top-level const) so the mock
    // factory closure captures a stable reference.
    const mockState = {
      status: 'RUNNING',
      nodesComputed: 200000,
      agents: [],
      heatmapNodes: Array.from({ length: 200000 }, (_, i) => ({
        id: `n-${i}`,
        variant: 'normal',
        label: `Node ${i}`,
        title: `Node ${i}`,
        description: 'd',
        x: 0,
        y: 0,
        delayMs: i, // incrementing delays to defeat any empty-array shortcut
      })),
      heatmapLines: [],
    };
    vi.doMock('@/hooks/useAgentLoop', () => ({
      useAgentLoop: () => ({
        state: mockState,
        error: null,
        isLoading: false,
      }),
    }));
    vi.doMock('next/navigation', () => ({
      useParams: () => ({ runId: 'test-run-id' }),
      useRouter: () => ({ replace: () => {}, push: () => {} }),
      notFound: () => { throw new Error('NF'); },
    }));

    const AgentStatusPanel = await importAgentStatusPanel();
    // Render must NOT throw a RangeError.
    let renderError: unknown = null;
    let container: any = null;
    try {
      const r = render(<AgentStatusPanel />);
      container = r.container;
    } catch (e) {
      renderError = e;
    }
    expect(renderError).toBeNull();
    expect(container).toBeTruthy();
  });

  /**
   * SWARM-2 BUG #42 — the auto-scroll effect restarts on every state update.
   * If the user scrolls during the animation, the next state update re-triggers
   * the effect, which resets `userInterrupted = false` and starts scrolling
   * again from the top, overriding the user's position.
   *
   * Strategy: Render the panel, scroll the container to a position, then
   * advance the state (increment agent count), and assert the container's
   * scrollTop hasn't been reset to a low value (i.e., user position preserved).
   *
   * Since the auto-scroll uses requestAnimationFrame in a tight loop, we use
   * fake timers and step through manually.
   */
  test('test_auto_scroll_does_not_override_user_position', async () => {
    vi.useFakeTimers();
    const mockStateRef = { current: {
      status: 'RUNNING',
      nodesComputed: 1,
      agents: [{ id: 'a1', type: 'explorer', agentId: 'a1', y: 100, content: 'c', delayMs: 1000 }],
      heatmapNodes: [{ id: 'n1', variant: 'normal', label: 'N1', title: 'N1', description: 'd', x: 0, y: 0, delayMs: 1000 }],
      heatmapLines: [],
    } };
    vi.doMock('@/hooks/useAgentLoop', () => ({
      useAgentLoop: () => ({
        state: mockStateRef.current,
        error: null,
        isLoading: false,
      }),
    }));
    vi.doMock('next/navigation', () => ({
      useParams: () => ({ runId: 'test-run-id' }),
      useRouter: () => ({ replace: () => {}, push: () => {} }),
      notFound: () => { throw new Error('NF'); },
    }));

    const AgentStatusPanel = await importAgentStatusPanel();
    const { container, rerender } = render(<AgentStatusPanel />);
    const scroller = container.firstElementChild as HTMLElement;
    expect(scroller).toBeTruthy();

    let scrollTopValue = 0; // start at top so the buggy code's animation kicks in
    Object.defineProperty(scroller, 'scrollTop', {
      configurable: true,
      get: () => scrollTopValue,
      set: (v: number) => { scrollTopValue = v; },
    });
    Object.defineProperty(scroller, 'scrollHeight', {
      configurable: true, get: () => 2000,
    });
    Object.defineProperty(scroller, 'clientHeight', {
      configurable: true, get: () => 400,
    });

    // Run a few animation frames so the auto-scroll advances a bit.
    for (let i = 0; i < 5; i++) vi.advanceTimersByTime(16);
    // At this point scrollTop should be > 0 from the auto-scroll.
    const afterAutoScroll = scrollTopValue;
    expect(afterAutoScroll).toBeGreaterThan(0);

    // User manually scrolls down (wheel event) — sets userInterrupted=true.
    scrollTopValue = 800; // user scrolls to near the bottom
    fireEvent.wheel(scroller, { deltaY: 100 });

    // Advance the state — re-triggers the effect.
    mockStateRef.current = {
      ...mockStateRef.current,
      agents: [
        ...mockStateRef.current.agents,
        { id: 'a2', type: 'explorer', agentId: 'a2', y: 200, content: 'c2', delayMs: 200 },
      ],
    };
    rerender(<AgentStatusPanel />);

    // Step through several animation frames after the re-render.
    for (let i = 0; i < 50; i++) vi.advanceTimersByTime(16);

    // The buggy code resets scrollTop to a low value on effect re-run because
    // userInterrupted is reinitialized to false (it's a local variable).
    // The fix preserves user position via a ref that survives re-runs.
    expect(scrollTopValue).toBeGreaterThanOrEqual(700);
    vi.useRealTimers();
  });

  /**
   * SWARM-2 BUG #43 — the error fallback message says "WebSocket" but the
   * hook actually uses `fetch` polling. Misleading error message.
   *
   * Static-source check: the source should not mention "WebSocket" in the
   * error message. It should mention "fetch" or "polling".
   */
  test('test_error_message_says_fetch_not_websocket', () => {
    // The buggy source has "Failed to sync Layer 2 State via WebSocket".
    // The fix should remove "WebSocket" and use "fetch" or "polling".
    expect(agentStatusSrc).not.toMatch(/WebSocket/i);
    // And should mention "fetch" or "polling".
    expect(/(fetch|poll)/i.test(agentStatusSrc)).toBe(true);
  });
});