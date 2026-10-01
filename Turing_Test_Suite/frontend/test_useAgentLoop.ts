/**
 * test_useAgentLoop.ts — Vitest tests for src/hooks/useAgentLoop.ts
 *
 * Coverage:
 *   - SWARM-2 BUG #1  — no AbortController on unmount / runId change
 *   - SWARM-2 BUG #2  — polling never stops after agents complete
 *   - SWARM-2 BUG #14 — no exponential backoff on fetch errors
 *   - SWARM-1 BUG #9  — late responses overwrite newer state (race)
 *
 * Strategy mirrors test_useRunState.ts (fetch mock + fake timers).
 *
 * POLL_INTERVAL_MS = 1000 (1 second) for useAgentLoop.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useAgentLoop } from '@/hooks/useAgentLoop';
import { fetchMock, makeResponse, abortCalls } from './setup';

const RUNNING_STATE = {
  status: 'RUNNING',
  nodesComputed: 5,
  agents: [],
  heatmapNodes: [],
  heatmapLines: [],
};

const COMPLETE_STATE = {
  status: 'COMPLETE',
  nodesComputed: 42,
  agents: [],
  heatmapNodes: [],
  heatmapLines: [],
};

const FAILED_STATE = {
  status: 'FAILED',
  nodesComputed: 7,
  agents: [],
  heatmapNodes: [],
  heatmapLines: [],
};

beforeEach(() => {
  vi.useFakeTimers();
});
afterEach(() => {
  vi.useRealTimers();
});

describe('useAgentLoop', () => {
  // -----------------------------------------------------------------------
  // BUG #1 — no AbortController on unmount
  // -----------------------------------------------------------------------

  /**
   * SWARM-2 BUG #1 — unmount mid-fetch must abort the in-flight fetch.
   *
   * The hook does not currently pass an AbortController signal to fetch,
   * so unmounting mid-fetch leaves the request running (wasted bandwidth,
   * potential state pollution on remount).
   *
   * Fixed code: create AbortController in effect, pass signal to fetch,
   * call .abort() in cleanup.
   */
  it('test_uses_abortcontroller_on_unmount (BUG #1)', async () => {
    fetchMock.mockImplementationOnce(() => new Promise(() => {}));

    const { unmount } = renderHook(() => useAgentLoop('run-1'));

    unmount();

    expect(abortCalls.length).toBeGreaterThanOrEqual(1);
  });

  /**
   * SWARM-2 BUG #1 — runId change must abort the OLD in-flight fetch.
   */
  it('test_uses_abortcontroller_on_run_id_change (BUG #1)', async () => {
    fetchMock.mockImplementation(() => new Promise(() => {}));

    const { rerender } = renderHook(
      ({ runId }) => useAgentLoop(runId),
      { initialProps: { runId: 'run-1' } },
    );

    act(() => {
      rerender({ runId: 'run-2' });
    });

    expect(abortCalls.length).toBeGreaterThanOrEqual(1);
  });

  // -----------------------------------------------------------------------
  // BUG #2 — polling never stops after run completes
  // -----------------------------------------------------------------------

  /**
   * SWARM-2 BUG #2 — when `state.status === 'COMPLETE'` or `'FAILED'`,
   * the polling must stop (the Layer 2 agent state is terminal).
   *
   * The buggy code keeps polling every 1s forever.
   *
   * Test: mock fetch to return COMPLETE_STATE on the first call, advance
   * the clock by 10 seconds. Only ONE fetch call should have happened.
   */
  it('test_stops_polling_when_agents_complete (BUG #2)', async () => {
    fetchMock.mockImplementation(() =>
      Promise.resolve(makeResponse(COMPLETE_STATE)),
    );

    const { result } = renderHook(() => useAgentLoop('run-1'));

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.state).not.toBeNull();
    expect(result.current.state?.status).toBe('COMPLETE');

    const initialCallCount = fetchMock.mock.calls.length;
    expect(initialCallCount).toBe(1);

    // Advance 10s — buggy code would fire ~10 polls at 1s interval.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10_000);
    });

    // Buggy code: ~10 more fetches.
    // Fixed code: 0 more fetches (interval cleared).
    expect(fetchMock.mock.calls.length).toBe(1);
  });

  /**
   * SWARM-2 BUG #2 (FAILED variant)
   */
  it('test_stops_polling_when_agents_failed (BUG #2)', async () => {
    fetchMock.mockImplementation(() =>
      Promise.resolve(makeResponse(FAILED_STATE)),
    );

    const { result } = renderHook(() => useAgentLoop('run-1'));

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.state?.status).toBe('FAILED');

    const initialCallCount = fetchMock.mock.calls.length;

    await act(async () => {
      await vi.advanceTimersByTimeAsync(10_000);
    });

    expect(fetchMock.mock.calls.length).toBe(initialCallCount);
  });

  // -----------------------------------------------------------------------
  // BUG #14 — no exponential backoff on fetch errors
  // -----------------------------------------------------------------------

  /**
   * SWARM-2 BUG #14 — when the backend is down, the buggy hook keeps
   * polling every 1s forever. The fixed hook should increase the interval
   * on each consecutive error (1s → 2s → 4s → 8s).
   */
  it('test_exponential_backoff_on_error (BUG #14)', async () => {
    const fetchTimes: number[] = [];
    fetchMock.mockImplementation(() => {
      fetchTimes.push(Date.now());
      return Promise.reject(new Error('network down'));
    });

    renderHook(() => useAgentLoop('run-1'));

    // Settle the initial fetch (it rejects).
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    // Advance enough time to fire several polls under backoff.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000);
    });

    expect(fetchTimes.length).toBeGreaterThanOrEqual(4);

    const deltas: number[] = [];
    for (let i = 1; i < fetchTimes.length; i++) {
      deltas.push(fetchTimes[i] - fetchTimes[i - 1]);
    }

    // Buggy code: every delta is ~1000ms (constant interval).
    // Fixed code: deltas grow exponentially.
    const maxDelta = Math.max(...deltas);
    expect(maxDelta).toBeGreaterThan(2000);

    const allEqual = deltas.every((d) => Math.abs(d - deltas[0]) < 100);
    expect(allEqual).toBe(false);
  });

  // -----------------------------------------------------------------------
  // SWARM-1 BUG #9 — late response overwrites newer state
  // -----------------------------------------------------------------------

  /**
   * SWARM-1 BUG #9 — race in polling: a slow poll can resolve AFTER a
   * faster poll fired later, overwriting newer state with stale state.
   *
   * Test:
   *   1. Render the hook — initial fetch fires (P1, currentLayer=1).
   *   2. Advance 1s — first interval tick fires fetch (P2, currentLayer=4).
   *   3. Resolve P2 first → state.currentLayer = 4.
   *   4. Resolve P1 last → state.currentLayer = 1 (BUG: stale overwrites).
   *
   * Fixed code uses AbortController or a request counter to ignore the
   * late P1 response.
   */
  it('test_no_late_response_overwrites_newer_state (SWARM-1 BUG #9)', async () => {
    let resolveP1: (v: unknown) => void = () => {};
    let resolveP2: (v: unknown) => void = () => {};
    const P1 = new Promise((res) => { resolveP1 = res; });
    const P2 = new Promise((res) => { resolveP2 = res; });

    fetchMock.mockImplementationOnce(() => P1);
    fetchMock.mockImplementationOnce(() => P2);

    const { result } = renderHook(() => useAgentLoop('run-1'));

    // Advance 1s to trigger the first interval poll (P2).
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000);
    });
    expect(fetchMock.mock.calls.length).toBe(2);

    // Resolve P2 first (newer — status=COMPLETE).
    await act(async () => {
      resolveP2(makeResponse(COMPLETE_STATE));
      await vi.advanceTimersByTimeAsync(0);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.state?.status).toBe('COMPLETE');

    // Resolve P1 LAST (older — status=RUNNING).
    await act(async () => {
      resolveP1(makeResponse(RUNNING_STATE));
      await vi.advanceTimersByTimeAsync(0);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    // The state must reflect the newer (P2) response, not the older (P1).
    expect(result.current.state?.status).toBe('COMPLETE');
  });

  // -----------------------------------------------------------------------
  // Error handling
  // -----------------------------------------------------------------------

  it('test_handles_run_not_found', async () => {
    fetchMock.mockImplementation(() =>
      Promise.resolve(makeResponse({ detail: 'not found' }, { status: 404 })),
    );

    const { result } = renderHook(() => useAgentLoop('run-1'));

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.error).not.toBeNull();
    expect(result.current.isLoading).toBe(false);
  });

  it('test_handles_network_error', async () => {
    fetchMock.mockImplementation(() =>
      Promise.reject(new Error('network down')),
    );

    const { result } = renderHook(() => useAgentLoop('run-1'));

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.error).not.toBeNull();
    expect(result.current.isLoading).toBe(false);
  });
});