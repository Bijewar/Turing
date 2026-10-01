/**
 * test_useRunState.ts — Vitest tests for src/hooks/useRunState.ts
 *
 * Coverage:
 *   - SWARM-2 BUG #11 — no AbortController on unmount / runId change
 *   - SWARM-2 BUG #12 — polling never stops after run completes
 *   - SWARM-2 BUG #14 — no exponential backoff on fetch errors
 *   - SWARM-1 BUG #9  — late responses overwrite newer state (race)
 *
 * Strategy:
 *   - vi.useFakeTimers() for setInterval control
 *   - fetchMock returns Response-shaped objects (see setup.ts makeResponse)
 *   - abortCalls array tracks every AbortController.abort() call (see setup.ts)
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useRunState } from '@/hooks/useRunState';
import { fetchMock, makeResponse, abortCalls } from './setup';

const RUNNING_STATE = {
  runId: 'run-1',
  currentLayer: 2,
  layer1Status: 'completed',
  layer2Status: 'running',
  layer3Status: 'idle',
  layer4Status: 'idle',
  progressPercentage: 35,
};

const COMPLETE_STATE = {
  ...RUNNING_STATE,
  currentLayer: 4,
  layer1Status: 'completed',
  layer2Status: 'completed',
  layer3Status: 'completed',
  layer4Status: 'completed',
  progressPercentage: 100,
};

const FAILED_STATE = {
  ...RUNNING_STATE,
  layer2Status: 'error',
  layer3Status: 'idle',
  layer4Status: 'idle',
  progressPercentage: 50,
};

beforeEach(() => {
  vi.useFakeTimers();
});
afterEach(() => {
  vi.useRealTimers();
});

describe('useRunState', () => {
  // -----------------------------------------------------------------------
  // BUG #11 — no AbortController on unmount
  // -----------------------------------------------------------------------

  /**
   * SWARM-2 BUG #11 — unmount mid-fetch must abort the in-flight fetch.
   *
   * The hook does not currently pass an AbortController signal to fetch,
   * so unmounting mid-fetch leaves the request running (wasted bandwidth,
   * and a late response could clobber state on remount).
   *
   * The fixed code creates an AbortController in the effect body, passes
   * its signal to fetch, and calls .abort() in the cleanup.
   *
   * Test: render, immediately unmount BEFORE the initial fetch resolves,
   * verify AbortController.abort() was called at least once.
   */
  it('test_uses_abortcontroller_on_unmount (BUG #11)', async () => {
    // Initial fetch is pending — never resolves in this test.
    fetchMock.mockImplementationOnce(
      () => new Promise(() => {}),
    );

    const { unmount } = renderHook(() => useRunState('run-1'));

    // Unmount BEFORE the initial fetch resolves — the in-flight request
    // should be aborted via AbortController.abort().
    unmount();

    expect(abortCalls.length).toBeGreaterThanOrEqual(1);
  });

  /**
   * SWARM-2 BUG #11 — runId change must abort the OLD in-flight fetch.
   */
  it('test_uses_abortcontroller_on_run_id_change (BUG #11)', async () => {
    fetchMock.mockImplementation(() => new Promise(() => {}));

    const { rerender } = renderHook(
      ({ runId }) => useRunState(runId),
      { initialProps: { runId: 'run-1' } },
    );

    // Change runId mid-fetch — old fetch must be aborted.
    act(() => {
      rerender({ runId: 'run-2' });
    });

    expect(abortCalls.length).toBeGreaterThanOrEqual(1);
  });

  // -----------------------------------------------------------------------
  // BUG #12 — polling never stops on terminal state
  // -----------------------------------------------------------------------

  /**
   * SWARM-2 BUG #12 — polling must stop when the run reaches a terminal
   * state (`currentLayer === 4 && progressPercentage === 100`, or status
   * FAILED). The buggy code keeps polling every 1.5s forever.
   *
   * Test: mock fetch to return COMPLETE_STATE on the first call, then
   * advance the clock by 10 seconds. Only ONE fetch call should have
   * happened (the initial fetch). The fixed code clears the interval
   * after receiving COMPLETE.
   */
  it('test_stops_polling_when_run_complete (BUG #12)', async () => {
    fetchMock.mockImplementation(() =>
      Promise.resolve(makeResponse(COMPLETE_STATE)),
    );

    const { result } = renderHook(() => useRunState('run-1'));

    // Flush the initial fetch (microtasks + 0ms timers).
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    // Drain any remaining microtasks from response.json() etc.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.runState).not.toBeNull();
    expect(result.current.runState?.currentLayer).toBe(4);
    expect(result.current.runState?.progressPercentage).toBe(100);

    const initialCallCount = fetchMock.mock.calls.length;
    expect(initialCallCount).toBe(1); // initial fetch only

    // Advance 10s — would normally fire 6 polls at 1.5s interval.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10_000);
    });

    // Buggy code: ~6 more fetches.
    // Fixed code: 0 more fetches (interval cleared).
    expect(fetchMock.mock.calls.length).toBe(1);
  });

  /**
   * SWARM-2 BUG #12 (failed variant)
   *
   * When the run has FAILED status, polling must also stop (terminal state).
   */
  it('test_stops_polling_when_run_failed (BUG #12)', async () => {
    fetchMock.mockImplementation(() =>
      Promise.resolve(makeResponse(FAILED_STATE)),
    );

    const { result } = renderHook(() => useRunState('run-1'));

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.runState).not.toBeNull();

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
   * polling every 1.5s forever. The fixed hook should increase the
   * interval on each consecutive error (e.g., 1.5s → 3s → 6s → 12s).
   *
   * Test: mock fetch to always reject, then advance time and record the
   * wall-clock time of each fetch call. Verify the gaps between fetch
   * calls grow exponentially.
   */
  it('test_exponential_backoff_on_error (BUG #14)', async () => {
    const fetchTimes: number[] = [];
    fetchMock.mockImplementation(() => {
      fetchTimes.push(Date.now());
      return Promise.reject(new Error('network down'));
    });

    renderHook(() => useRunState('run-1'));

    // Allow the initial fetch to settle (it rejects).
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    // Advance time generously to trigger several polls.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000);
    });

    // We need at least 4 fetches to verify exponential gaps.
    expect(fetchTimes.length).toBeGreaterThanOrEqual(4);

    // Compute deltas between successive fetch calls (ms).
    const deltas: number[] = [];
    for (let i = 1; i < fetchTimes.length; i++) {
      deltas.push(fetchTimes[i] - fetchTimes[i - 1]);
    }

    // Buggy code: every delta is ~1500ms (constant interval, no backoff).
    // Fixed code: deltas should grow exponentially (e.g., 1500, 3000, 6000, 12000).
    //
    // A robust assertion: at least one delta must be > 2500ms (i.e., NOT
    // the constant 1500ms). This catches the "no backoff" bug without
    // requiring an exact exponential formula.
    const maxDelta = Math.max(...deltas);
    expect(maxDelta).toBeGreaterThan(2500);

    // Also verify the deltas are NOT all equal (which would mean no backoff).
    const allEqual = deltas.every((d) => Math.abs(d - deltas[0]) < 100);
    expect(allEqual).toBe(false);
  });

  // -----------------------------------------------------------------------
  // SWARM-1 BUG #9 — late response overwrites newer state
  // -----------------------------------------------------------------------

  /**
   * SWARM-1 BUG #9 — race in polling: a slow poll (3s) can resolve AFTER
   * a faster poll (later fired) has already updated state. The slow
   * poll's stale response overwrites the newer state.
   *
   * The fixed code uses either:
   *   (a) an AbortController to abort the previous in-flight fetch, OR
   *   (b) a monotonically-increasing request counter to ignore late
   *       responses.
   *
   * Test:
   *   1. Render the hook — initial fetch fires (promise P1).
   *   2. Advance 1.5s — first interval tick fires fetch (promise P2).
   *   3. Resolve P2 first (newer response, currentLayer=4).
   *   4. Resolve P1 last (older response, currentLayer=1).
   *   5. Assert runState.currentLayer === 4 (the newer response wins).
   *
   * Buggy code: state ends up at 1 (the late P1 overwrote P2).
   * Fixed code: state ends up at 4 (P1 is aborted or ignored).
   */
  it('test_no_late_response_overwrites_newer_state (SWARM-1 BUG #9)', async () => {
    let resolveP1: (v: unknown) => void = () => {};
    let resolveP2: (v: unknown) => void = () => {};
    const P1 = new Promise((res) => {
      resolveP1 = res;
    });
    const P2 = new Promise((res) => {
      resolveP2 = res;
    });

    fetchMock.mockImplementationOnce(() => P1);
    fetchMock.mockImplementationOnce(() => P2);

    const { result } = renderHook(() => useRunState('run-1'));

    // Advance 1.5s to trigger the first interval poll (P2).
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1500);
    });

    expect(fetchMock.mock.calls.length).toBe(2);

    // Resolve P2 first (the newer response — currentLayer=4).
    await act(async () => {
      resolveP2(makeResponse(COMPLETE_STATE));
      // Flush the microtask chain so setRunState fires.
      await vi.advanceTimersByTimeAsync(0);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    // At this point the newer (P2) response should have been applied.
    expect(result.current.runState?.currentLayer).toBe(4);

    // Resolve P1 LAST (older response — currentLayer=1).
    await act(async () => {
      resolveP1(
        makeResponse({
          ...RUNNING_STATE,
          currentLayer: 1,
          progressPercentage: 10,
        }),
      );
      await vi.advanceTimersByTimeAsync(0);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    // The state must reflect the newer (P2) response, not the older (P1).
    expect(result.current.runState?.currentLayer).toBe(4);
    expect(result.current.runState?.progressPercentage).toBe(100);
  });

  // -----------------------------------------------------------------------
  // Error handling
  // -----------------------------------------------------------------------

  /**
   * 404 — hook must set `error`, not crash.
   */
  it('test_handles_run_not_found', async () => {
    fetchMock.mockImplementation(() =>
      Promise.resolve(makeResponse({ detail: 'not found' }, { status: 404 })),
    );

    const { result } = renderHook(() => useRunState('run-1'));

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.error).not.toBeNull();
    expect(result.current.isLoading).toBe(false);
  });

  /**
   * Network failure — hook must set `error`, not crash.
   */
  it('test_handles_network_error', async () => {
    fetchMock.mockImplementation(() =>
      Promise.reject(new Error('network down')),
    );

    const { result } = renderHook(() => useRunState('run-1'));

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current.error).not.toBeNull();
    expect(result.current.isLoading).toBe(false);
  });

  /**
   * Before the first fetch resolves, runState must be null and isLoading
   * must be true.
   */
  it('test_returns_initial_state_before_first_fetch', async () => {
    // Initial fetch is pending.
    fetchMock.mockImplementation(() => new Promise(() => {}));

    const { result } = renderHook(() => useRunState('run-1'));

    expect(result.current.runState).toBeNull();
    expect(result.current.isLoading).toBe(true);
    expect(result.current.error).toBeNull();
  });
});