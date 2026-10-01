/**
 * test_useGraphAnimation.ts — Vitest tests for src/hooks/useGraphAnimation.ts
 *
 * Coverage:
 *   - SWARM-2 BUG #3  — JSON.stringify comparison is O(N) and crashes on
 *                       circular refs / false-negatives on float rounding
 *   - SWARM-2 BUG #4  — no AbortController on unmount / runId change
 *   - SWARM-2 BUG #14 — no exponential backoff on fetch errors
 *
 * Strategy:
 *   - fetchMock returns CausalGraph-shaped objects (see setup.ts makeResponse)
 *   - vi.useFakeTimers() for setInterval control
 *   - For "does not use JSON.stringify" test, we read the source file and
 *     check that the comparison logic does NOT call JSON.stringify.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useGraphAnimation } from '@/hooks/useGraphAnimation';
import { fetchMock, makeResponse, abortCalls, consoleErrors } from './setup';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// Helper: build a mock Response whose `.json()` resolves to the given body
// WITHOUT round-tripping through JSON.stringify (so the body can have
// circular references that would otherwise break the mock itself).
function makeResponseWithRawBody(body: unknown, status = 200) {
  const resp = makeResponse({}, { status });
  resp.json = async () => body;
  return resp;
}

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SRC_PATH = path.resolve(
  HERE,
  '../../src/hooks/useGraphAnimation.ts',
);
const SOURCE = fs.readFileSync(SRC_PATH, 'utf8');

const BASE_GRAPH = {
  nodes: [
    { id: 'A', label: 'A', type: 'controllable' as const, value: 0, beta: 0.5 },
    { id: 'B', label: 'B', type: 'outcome' as const, value: 0, beta: 0.5 },
  ],
  edges: [
    { source: 'A', target: 'B', curvature: 0.2, weight: 0.5 },
  ],
  is_fitted: false,
};

beforeEach(() => {
  vi.useFakeTimers();
});
afterEach(() => {
  vi.useRealTimers();
});

describe('useGraphAnimation', () => {
  // -----------------------------------------------------------------------
  // BUG #3 — JSON.stringify comparison
  // -----------------------------------------------------------------------

  /**
   * SWARM-2 BUG #3 — verify the source code does NOT use JSON.stringify
   * inside the graph comparison logic.
   *
   * The buggy code: `if (JSON.stringify(prev) === JSON.stringify(data))`
   * The fix: use a content hash, field-by-field comparison, or a fast
   * structural equality check (e.g., nodes.length + edges.length + sum of
   * weights) that is O(N) but doesn't allocate strings.
   *
   * We use a source-code audit (regex match) to catch this — same approach
   * as the D3GraphEngine tests use for impossible-to-runtime-verify
   * assertions.
   */
  it('test_does_not_use_json_stringify_for_comparison (BUG #3)', () => {
    // Look for JSON.stringify in the comparison branch of setGraph.
    const hasJsonStringify = /JSON\.stringify\s*\(\s*prev\s*\)/.test(SOURCE)
      || /JSON\.stringify\s*\(\s*data\s*\)/.test(SOURCE)
      || /JSON\.stringify\s*\(\s*graph\s*\)/.test(SOURCE);
    expect(hasJsonStringify).toBe(false);
  });

  /**
   * SWARM-2 BUG #3 — circular references must NOT crash the comparison.
   *
   * The buggy code's `JSON.stringify(prev)` throws TypeError:
   *   "Converting circular structure to JSON"
   * inside the setState updater. The error propagates up during the next
   * render (no error boundary in tests).
   *
   * The fixed code uses a structural hash / field-by-field check that
   * doesn't traverse circular references.
   *
   * Test: feed a non-circular graph on the first poll (so prev is a
   * well-formed graph), then a circular graph on the second poll. The
   * second poll triggers the comparison `JSON.stringify(prev) ===
   * JSON.stringify(data)`, and JSON.stringify throws on the circular
   * data. The hook MUST NOT crash.
   */
  it('test_comparison_handles_circular_references (BUG #3)', async () => {
    // First poll returns a non-circular graph (so prev is well-formed).
    // Second poll returns a graph with a circular ref — JSON.stringify(data)
    // will throw on the circular ref.
    let callCount = 0;
    fetchMock.mockImplementation(() => {
      callCount++;
      // Use makeResponseWithRawBody so the circular body doesn't break the
      // mock itself (it bypasses JSON.stringify in the mock).
      const body = JSON.parse(JSON.stringify(BASE_GRAPH));
      if (callCount >= 2) {
        // Add a circular reference for the second+ poll.
        // @ts-expect-error — intentionally create a circular ref
        body.nodes[0].self = body.nodes[0];
      }
      return Promise.resolve(makeResponseWithRawBody(body));
    });

    const { result } = renderHook(() => useGraphAnimation('run-1'));

    // Flush first poll (non-circular graph).
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    // Use a ref-based check to avoid expect() serializing the graph.
    const g = result.current.graph;
    expect(g === null).toBe(false);
    expect(typeof g).toBe('object');

    // Snapshot console.error state before the second poll.
    const errorsBefore = consoleErrors.length;

    // Advance 2s to trigger second poll — this calls setGraph(prev => ...)
    // where data has a circular ref. The buggy code's JSON.stringify(data)
    // would throw here. React captures the throw during the next render
    // and logs it via console.error (the test renderer has no error
    // boundary, so the error becomes an uncaught render error).
    let didThrow = false;
    try {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(2000);
      });
    } catch {
      didThrow = true;
    }

    // The fixed code does NOT throw on circular refs.
    // The buggy code throws (either as an act() rejection OR as a
    // console.error log from React's uncaught-render-error handler).
    const newErrors = consoleErrors.slice(errorsBefore);
    const circularErrorLogged = newErrors.some((e) =>
      /circular|json|stringify/i.test(e),
    );

    // The test passes if BOTH of these are false:
    //   1. No act() throw
    //   2. No circular/JSON error logged via console.error
    expect(didThrow).toBe(false);
    expect(circularErrorLogged).toBe(false);
  });

  /**
   * SWARM-2 BUG #3 — floating-point precision differences should not
   * trigger a graph rebuild.
   *
   * The buggy code's JSON.stringify treats `0.5` and `0.50000001` as
   * different strings, so the comparison returns false, and the hook
   * calls setGraph with a NEW reference (data), causing D3 to rebuild
   * the physics layout (visual flicker).
   *
   * The fix should treat values within a small epsilon as equal (or use
   * a coarser structural hash that ignores tiny float differences).
   *
   * Test: feed graph1 with beta=0.5, then graph2 with beta=0.50000001.
   * After both polls, the hook's `graph` reference should NOT have changed
   * (i.e., setState returned `prev`).
   *
   * Buggy code: graph reference changes (state set to graph2).
   * Fixed code: graph reference stays at graph1.
   */
  it('test_comparison_handles_floating_point_precision (BUG #3)', async () => {
    const graph1 = JSON.parse(JSON.stringify(BASE_GRAPH));
    graph1.nodes[0].beta = 0.5;

    const graph2 = JSON.parse(JSON.stringify(BASE_GRAPH));
    graph2.nodes[0].beta = 0.50000001; // tiny float difference

    let callCount = 0;
    fetchMock.mockImplementation(() => {
      callCount++;
      const body = callCount === 1 ? graph1 : graph2;
      return Promise.resolve(makeResponse(body));
    });

    const { result } = renderHook(() => useGraphAnimation('run-1'));

    // Flush first poll.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.graph).not.toBeNull();
    const refAfterFirst = result.current.graph;

    // Advance 2s to trigger second poll (with graph2 — tiny float diff).
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    // The fixed code returns prev (same reference) because the graphs are
    // semantically equal (only floating-point rounding differs).
    // The buggy code returns data (new reference) because JSON.stringify
    // treats 0.5 and 0.50000001 as different strings.
    expect(result.current.graph).toBe(refAfterFirst);
  });

  // -----------------------------------------------------------------------
  // BUG #4 — no AbortController
  // -----------------------------------------------------------------------

  /**
   * SWARM-2 BUG #4 — unmount mid-fetch must abort the in-flight fetch.
   */
  it('test_uses_abortcontroller_on_unmount (BUG #4)', async () => {
    fetchMock.mockImplementationOnce(() => new Promise(() => {}));

    const { unmount } = renderHook(() => useGraphAnimation('run-1'));

    unmount();

    expect(abortCalls.length).toBeGreaterThanOrEqual(1);
  });

  /**
   * SWARM-2 BUG #4 — runId change must abort the OLD in-flight fetch.
   */
  it('test_uses_abortcontroller_on_run_id_change (BUG #4)', async () => {
    fetchMock.mockImplementation(() => new Promise(() => {}));

    const { rerender } = renderHook(
      ({ runId }) => useGraphAnimation(runId),
      { initialProps: { runId: 'run-1' } },
    );

    act(() => {
      rerender({ runId: 'run-2' });
    });

    expect(abortCalls.length).toBeGreaterThanOrEqual(1);
  });

  // -----------------------------------------------------------------------
  // Stops polling when graph complete
  // -----------------------------------------------------------------------

  /**
   * SWARM-2 BUG #2/#12 (graph variant) — when `is_fitted === true`, the
   * polling must stop (the graph discovery has converged).
   *
   * The buggy code polls every 2s forever.
   */
  it('test_stops_polling_when_graph_complete', async () => {
    fetchMock.mockImplementation(() =>
      Promise.resolve(makeResponse({ ...BASE_GRAPH, is_fitted: true })),
    );

    const { result } = renderHook(() => useGraphAnimation('run-1'));

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.graph).not.toBeNull();

    const initialCallCount = fetchMock.mock.calls.length;
    expect(initialCallCount).toBe(1);

    // Advance 10s — buggy code would fire ~5 polls at 2s interval.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10_000);
    });

    expect(fetchMock.mock.calls.length).toBe(1);
  });

  // -----------------------------------------------------------------------
  // BUG #14 — no exponential backoff on fetch errors
  // -----------------------------------------------------------------------

  /**
   * SWARM-2 BUG #14 — when the backend is down, the buggy hook polls
   * every 2s forever. The fixed hook should increase the interval
   * exponentially (2s → 4s → 8s → 16s).
   */
  it('test_exponential_backoff_on_error (BUG #14)', async () => {
    const fetchTimes: number[] = [];
    fetchMock.mockImplementation(() => {
      fetchTimes.push(Date.now());
      return Promise.reject(new Error('network down'));
    });

    renderHook(() => useGraphAnimation('run-1'));

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000);
    });

    expect(fetchTimes.length).toBeGreaterThanOrEqual(4);

    const deltas: number[] = [];
    for (let i = 1; i < fetchTimes.length; i++) {
      deltas.push(fetchTimes[i] - fetchTimes[i - 1]);
    }

    const maxDelta = Math.max(...deltas);
    expect(maxDelta).toBeGreaterThan(3000);

    const allEqual = deltas.every((d) => Math.abs(d - deltas[0]) < 100);
    expect(allEqual).toBe(false);
  });
});