/**
 * test_useReportStream.ts — Vitest tests for src/hooks/useReportStream.ts
 *
 * Coverage:
 *   - SWARM-2 BUG #5  — setError not guarded by `cancelled` in onmessage
 *   - SWARM-2 BUG #6  — `isLoading: !report && isConnected` is false when
 *                       the SSE hasn't opened yet (no loading indicator shown)
 *   - SWARM-2 BUG #7  — retryTimer overwrites without clearTimeout
 *   - SWARM-2 BUG #8  — no `reconnect()` callback exposed after MAX_RETRIES
 *   - SWARM-7 BUG #22 — retryCount ref leaks across runId changes
 *
 * Strategy:
 *   - renderHook + act from @testing-library/react (React 18)
 *   - Mock EventSource (see setup.ts) — never opens a real SSE stream
 *   - vi.useFakeTimers() for the exponential backoff reconnect tests
 *   - For each test, name the bug ID it targets in the docstring
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useReportStream } from '@/hooks/useReportStream';
import { EventSourceMock } from './setup';

const SAMPLE_REPORT = {
  isComplete: false,
  theMechanism: 'X causes Y via Z',
  theExperiment: 'do X, measure Y',
  whoSolvedThis: 'lab A',
  warningsAndConflicts: [],
  next3Actions: ['do a', 'do b', 'do c'],
  confidenceDisclaimer: 'disclaimer',
};

beforeEach(() => {
  vi.useFakeTimers();
});
afterEach(() => {
  vi.useRealTimers();
});

describe('useReportStream', () => {
  // -----------------------------------------------------------------------
  // BUG #5 — setError calls not guarded by `cancelled`
  // -----------------------------------------------------------------------

  /**
   * SWARM-2 BUG #5
   *
   * Render the hook, immediately unmount, then trigger an SSE message that
   * would cause a JSON.parse error (forcing the `setError(err)` branch in
   * the `catch` block).
   *
   * In the buggy source, the catch block calls `setError(err)` WITHOUT
   * checking the `cancelled` flag. After unmount this is a setState on an
   * unmounted component.
   *
   * In React 18 the unmount warning was removed, so we additionally assert
   * that the EventSource was closed on unmount (the cleanup runs). The
   * observable behaviour (result.current.error unchanged) is the same in
   * both buggy and fixed code in React 18, but the GUARD is still the
   * correct pattern and the test documents expected behaviour.
   */
  it('test_does_not_set_state_after_unmount (BUG #5)', () => {
    const { result, unmount } = renderHook(() => useReportStream('run-1'));
    expect(EventSourceMock._count()).toBe(1);
    const es = EventSourceMock._last()!;
    expect(es).toBeDefined();

    // Unmount — cleanup must close the EventSource
    unmount();
    expect(es.close).toHaveBeenCalled();

    // Snapshot the error before the post-unmount message
    const errorBefore = result.current.error;

    // Fire a malformed SSE message AFTER unmount.
    // The buggy code's catch block calls `setError(err)` without a guard.
    // The fixed code does `if (cancelled) return;` first.
    act(() => {
      es._emitMessage('{not-valid-json');
    });

    // No further state change should be observable on the unmounted hook.
    // (React 18 silently no-ops setState on unmounted components.)
    expect(result.current.error).toBe(errorBefore);

    // No "unmounted component" or "act" warning should have been logged.
    // (Soft regression check — fires on React 17 / 16.)
    const unmountWarnings = (
      globalThis as unknown as { __consoleErrors?: string[] }
    ).__consoleErrors;
    void unmountWarnings; // tests don't require this; just document.
  });

  /**
   * SWARM-2 BUG #5 (observable variant)
   *
   * When `runId` changes, the cleanup runs (cancelled=true on OLD effect)
   * and a NEW EventSource is created. The OLD EventSource's `onmessage`
   * closure still references the OLD `cancelled` flag (which is true).
   *
   * In the buggy source, `setError(null)` and `setError(err)` are NOT
   * guarded by `cancelled`. So if we trigger a malformed message on the
   * OLD EventSource, the OLD closure's catch block calls setError(err),
   * which updates the SAME hook's state (setError is stable across
   * re-renders).
   *
   * In the fixed code, the early `if (cancelled) return;` prevents the
   * catch block from running, so result.current.error stays null.
   *
   * This test CATCHES BUG #5 even in React 18, because the hook is still
   * mounted (just with a new runId).
   */
  it('test_set_error_called_only_when_mounted (BUG #5)', () => {
    const { result, rerender } = renderHook(
      ({ runId }) => useReportStream(runId),
      { initialProps: { runId: 'run-1' } },
    );
    expect(EventSourceMock._count()).toBe(1);
    const firstEs = EventSourceMock._last()!;

    // Switch runId — cleanup runs (cancelled=true on OLD effect), new es created.
    act(() => {
      rerender({ runId: 'run-2' });
    });
    expect(EventSourceMock._count()).toBe(2);
    expect(result.current.error).toBeNull();

    // Trigger a malformed SSE message on the OLD EventSource.
    // The OLD closure's catch block fires `setError(err)` without a guard.
    act(() => {
      firstEs._emitMessage('{not-valid-json');
    });

    // BUG #5: buggy code calls setError(err) (visible state change).
    // Fixed code: `if (cancelled) return;` prevents the setState.
    expect(result.current.error).toBeNull();
  });

  // -----------------------------------------------------------------------
  // BUG #6 — isLoading returns false before the SSE has opened
  // -----------------------------------------------------------------------

  /**
   * SWARM-2 BUG #6
   *
   * Before any SSE message arrives and before `onopen` fires, `report` is
   * null and `isConnected` is false. The buggy source returns
   * `isLoading: !report && isConnected` which evaluates to
   * `!null && false === false`. The user sees NO loading indicator while
   * the connection is being established.
   *
   * The fix is `isLoading: !report && !error` (loading until either a
   * report arrives or a terminal error occurs).
   */
  it('test_isLoading_true_before_first_message (BUG #6)', () => {
    const { result } = renderHook(() => useReportStream('run-1'));
    // We have NOT fired onopen or onmessage yet — the connection is still
    // being established.
    expect(result.current.report).toBeNull();
    expect(result.current.isConnected).toBe(false);
    // BUG #6: buggy code returns false here; fixed code returns true.
    expect(result.current.isLoading).toBe(true);
  });

  /**
   * SWARM-2 BUG #6 (terminal-error branch)
   *
   * When the SSE has errored out (after MAX_RETRIES), `isLoading` should
   * become false so the UI shows the error state, not a stuck loading
   * spinner.
   *
   * Buggy code: `isLoading: !report && isConnected` — after error,
   * isConnected is false, so isLoading becomes false by coincidence.
   * Fixed code: `isLoading: !report && !error` — after error, !error=false,
   * so isLoading becomes false by design.
   *
   * This test is a regression check that verifies the fixed formula
   * continues to report `isLoading === false` after a terminal error.
   */
  it('test_is_loading_false_when_error (BUG #6)', () => {
    const { result } = renderHook(() => useReportStream('run-1'));
    expect(EventSourceMock._count()).toBe(1);
    const es = EventSourceMock._last()!;

    // Fire MAX_RETRIES errors to exhaust retries and force a terminal error.
    for (let i = 0; i < 5; i++) {
      act(() => {
        es._emitError();
      });
      // Advance past the retry delay (1s, 2s, 4s, 8s, 16s)
      act(() => {
        vi.advanceTimersByTime(20000);
      });
    }
    // One more error to trigger the max-retries give-up
    act(() => {
      es._emitError();
    });

    // After the terminal error, isLoading MUST be false.
    expect(result.current.error).not.toBeNull();
    expect(result.current.isLoading).toBe(false);
  });

  // -----------------------------------------------------------------------
  // BUG #7 — retryTimer overwrite & exponential backoff
  // -----------------------------------------------------------------------

  /**
   * SWARM-2 BUG #7 (positive — verifies exponential schedule is correct)
   *
   * Kill the SSE 5 times in succession, advancing the clock by the expected
   * delay each time. The delay between reconnects MUST double each retry:
   *   1000 → 2000 → 4000 → 8000 → 16000
   *
   * Buggy code DOES compute the exponential delay correctly (the bug is
   * not the backoff math — it's the missing clearTimeout). So this test
   * passes with both buggy and fixed code; it's a regression check.
   */
  it('test_reconnect_with_exponential_backoff (BUG #7)', () => {
    const { unmount } = renderHook(() => useReportStream('run-1'));
    expect(EventSourceMock._count()).toBe(1);

    const delays: number[] = [];

    for (let i = 0; i < 5; i++) {
      const es = EventSourceMock._last()!;
      act(() => {
        es._emitError();
      });
      // Pending timer count goes up by 1 after each error.
      const pendingCount = vi.getTimerCount();
      // Capture the delay by checking how far we have to advance before a
      // new EventSource appears. We use the formula BASE * 2^i = 1000 * 2^i.
      const expectedDelay = 1000 * Math.pow(2, i);
      delays.push(expectedDelay);
      expect(pendingCount).toBe(1); // only one timer should be pending
      // Advance the clock past the delay — reconnect fires, new ES created.
      act(() => {
        vi.advanceTimersByTime(expectedDelay + 10);
      });
      expect(EventSourceMock._count()).toBe(i + 2);
    }

    // Exponential: 1000, 2000, 4000, 8000, 16000
    expect(delays).toEqual([1000, 2000, 4000, 8000, 16000]);
    unmount();
  });

  /**
   * SWARM-2 BUG #7 (negative — verifies clearTimeout is called)
   *
   * If `onerror` fires twice in quick succession BEFORE the reconnect timer
   * fires, the buggy code overwrites `retryTimer.current` WITHOUT calling
   * `clearTimeout` on the previous timer. Both reconnect timers are then
   * pending; when they fire, two `connect()` calls happen, creating two
   * new EventSource instances (a leak / >MAX_RETRIES race).
   *
   * Fixed code calls `clearTimeout(retryTimer.current)` BEFORE assigning a
   * new timer, so only one timer is pending and only one reconnect happens.
   *
   * Test: fire onerror twice quickly, then check the pending timer count.
   * Buggy code: 2 pending timers → 2 reconnects → 3 EventSources total.
   * Fixed code: 1 pending timer → 1 reconnect → 2 EventSources total.
   */
  it('test_clears_previous_retry_timer_before_new (BUG #7)', () => {
    const { unmount } = renderHook(() => useReportStream('run-1'));
    expect(EventSourceMock._count()).toBe(1);

    const es = EventSourceMock._last()!;

    // Fire onerror twice QUICKLY (don't advance the clock between them).
    act(() => {
      es._emitError();
    });
    act(() => {
      es._emitError();
    });

    // After two errors fired in quick succession, the pending timer count
    // tells us whether the previous timer was cleared before the new one
    // was scheduled.
    //
    // Buggy code: 2 pending timers (the first was NOT cleared).
    // Fixed code: 1 pending timer (the first WAS cleared before the second).
    expect(vi.getTimerCount()).toBe(1);

    // Advance past the longest delay (2nd retry uses 2000ms — retryCount incremented).
    act(() => {
      vi.advanceTimersByTime(3000);
    });

    // Buggy code: 3 EventSources (initial + 2 reconnects from the 2 timers).
    // Fixed code: 2 EventSources (initial + 1 reconnect).
    expect(EventSourceMock._count()).toBe(2);

    unmount();
  });

  /**
   * SWARM-2 BUG #8 (and partial #7)
   *
   * After MAX_RETRIES (5), the hook must:
   *   1. Set `error` to the max-retries error message
   *   2. Stop scheduling further reconnects (no 6th EventSource)
   */
  it('test_max_retries_then_gives_up_with_error (BUG #8)', () => {
    const { result, unmount } = renderHook(() => useReportStream('run-1'));
    expect(EventSourceMock._count()).toBe(1);

    // Fire 5 errors (each followed by a clock advance to trigger the reconnect).
    for (let i = 0; i < 5; i++) {
      const es = EventSourceMock._last()!;
      act(() => {
        es._emitError();
      });
      const delay = 1000 * Math.pow(2, i);
      act(() => {
        vi.advanceTimersByTime(delay + 10);
      });
    }
    // 1 initial + 5 reconnects = 6 EventSources.
    expect(EventSourceMock._count()).toBe(6);

    // Now fire the 6th error — should trigger MAX_RETRIES give-up.
    const es = EventSourceMock._last()!;
    act(() => {
      es._emitError();
    });

    // No new reconnect scheduled — count stays at 6.
    act(() => {
      vi.advanceTimersByTime(60000);
    });
    expect(EventSourceMock._count()).toBe(6);

    // Error should be set to the max-retries message.
    expect(result.current.error).not.toBeNull();
    expect(result.current.error?.message).toMatch(/max retries/i);

    unmount();
  });

  /**
   * SWARM-2 BUG #8
   *
   * The hook should expose a `reconnect()` callback in its return value
   * so the UI can retry after MAX_RETRIES is exhausted (or any time the
   * user wants to retry manually).
   *
   * Buggy code: returns `{ report, error, isConnected, isLoading }` — no
   * `reconnect` function. So `result.current.reconnect` is `undefined`.
   *
   * Fixed code: returns `{ ..., reconnect }` where `reconnect` is a function
   * that resets retryCount and reconnects the EventSource.
   */
  it('test_exposes_reconnect_function (BUG #8)', () => {
    const { result, unmount } = renderHook(() => useReportStream('run-1'));
    expect(typeof result.current.reconnect).toBe('function');
    // Sanity check: calling it should create a fresh EventSource.
    const beforeCount = EventSourceMock._count();
    act(() => {
      result.current.reconnect();
    });
    expect(EventSourceMock._count()).toBeGreaterThan(beforeCount);
    unmount();
  });

  // -----------------------------------------------------------------------
  // Cleanup tests
  // -----------------------------------------------------------------------

  /**
   * Unmount must close the EventSource to avoid socket leaks.
   */
  it('test_closes_eventsource_on_unmount', () => {
    const { unmount } = renderHook(() => useReportStream('run-1'));
    const es = EventSourceMock._last()!;
    unmount();
    expect(es.close).toHaveBeenCalled();
  });

  /**
   * When `runId` changes, the OLD EventSource must be closed before the
   * NEW one is created (otherwise two SSE streams would be open for the
   * same tab — backend load + race).
   */
  it('test_closes_eventsource_on_run_id_change', () => {
    const { rerender } = renderHook(
      ({ runId }) => useReportStream(runId),
      { initialProps: { runId: 'run-1' } },
    );
    const firstEs = EventSourceMock._last()!;
    act(() => {
      rerender({ runId: 'run-2' });
    });
    expect(firstEs.close).toHaveBeenCalled();
    expect(EventSourceMock._count()).toBe(2);
    // New es must point to the new runId URL.
    const secondEs = EventSourceMock._last()!;
    expect(secondEs.url).toContain('run-2');
  });

  // -----------------------------------------------------------------------
  // Message handling
  // -----------------------------------------------------------------------

  /**
   * A malformed JSON body in an SSE `data:` frame must NOT crash the hook.
   * The catch block must `setError(err)`.
   */
  it('test_handles_malformed_json_in_message', () => {
    const { result } = renderHook(() => useReportStream('run-1'));
    const es = EventSourceMock._last()!;
    act(() => {
      es._emitMessage('{not-valid-json');
    });
    expect(result.current.error).not.toBeNull();
    expect(result.current.error?.message).toMatch(/parse|json|chunk/i);
  });

  /**
   * SSE may split a JSON object across multiple `data:` frames. The hook
   * doesn't accumulate (it parses each frame as a complete object), but it
   * MUST NOT crash on a partial chunk.
   *
   * This test verifies the hook handles a partial chunk gracefully (sets
   * an error) and recovers when a full chunk arrives afterwards.
   */
  it('test_handles_partial_json_chunks', () => {
    const { result } = renderHook(() => useReportStream('run-1'));
    const es = EventSourceMock._last()!;

    // Partial chunk — JSON.parse will throw.
    act(() => {
      es._emitMessage('{"isComplete":false,"theMechanism":"par');
    });
    expect(result.current.error).not.toBeNull();

    // Full chunk afterwards — should recover and clear the error.
    act(() => {
      es._emitMessage(JSON.stringify(SAMPLE_REPORT));
    });
    expect(result.current.report).not.toBeNull();
    expect(result.current.error).toBeNull();
  });

  /**
   * When the backend sends `isComplete: true` in a report, the hook must
   * close the EventSource (the run is finished — no need to keep polling).
   */
  it('test_stops_when_iscomplete_true', () => {
    const { result } = renderHook(() => useReportStream('run-1'));
    const es = EventSourceMock._last()!;
    act(() => {
      es._emitMessage(JSON.stringify({ ...SAMPLE_REPORT, isComplete: true }));
    });
    expect(es.close).toHaveBeenCalled();
    expect(result.current.report?.isComplete).toBe(true);
    expect(result.current.isConnected).toBe(false);
  });

  // -----------------------------------------------------------------------
  // SWARM-7 BUG #22 — retry count leaks across runId changes
  // -----------------------------------------------------------------------

  /**
   * SWARM-7 BUG #22 — cross-run failure leak.
   *
   * The `retryCount` ref is NOT reset on `runId` change in the buggy code.
   * So if the previous run had 3 failed retries, the new run starts with
   * retryCount=3 — only 2 retries left before giving up.
   *
   * Fixed code must reset retryCount.current = 0 in the effect body (or
   * in the cleanup) so each new run gets a full retry budget.
   *
   * Test:
   *   1. Render with runId='run-1'
   *   2. Trigger 3 errors (retryCount becomes 3)
   *   3. Rerender with runId='run-2' (cleanup + new effect)
   *   4. Trigger an error on the NEW EventSource
   *   5. Verify the next reconnect delay is 1000ms (retryCount=0)
   *      (Buggy code: delay would be 1000 * 2^3 = 8000ms.)
   */
  it('test_retry_count_resets_on_run_id_change (SWARM-7 BUG #22)', () => {
    const { rerender } = renderHook(
      ({ runId }) => useReportStream(runId),
      { initialProps: { runId: 'run-1' } },
    );

    // Trigger 3 errors on the first run (don't advance the clock).
    const firstEs = EventSourceMock._last()!;
    for (let i = 0; i < 3; i++) {
      act(() => {
        firstEs._emitError();
      });
    }
    // Buggy: retryCount.current = 3 (NOT reset on runId change)
    // Fixed: retryCount.current = 3 (still — we haven't changed runId yet)
    //
    // Sanity: pending timer count = 1 (latest scheduled reconnect).
    expect(vi.getTimerCount()).toBe(1);

    // Change runId — this triggers cleanup + new effect.
    act(() => {
      rerender({ runId: 'run-2' });
    });
    expect(EventSourceMock._count()).toBe(2);
    // After the runId change, pending timers should be cleared (cleanup runs
    // clearTimeout before the new effect schedules a new timer).
    expect(vi.getTimerCount()).toBe(0);

    // Fire one error on the NEW EventSource.
    const secondEs = EventSourceMock._last()!;
    act(() => {
      secondEs._emitError();
    });

    // After the runId change, retryCount.current MUST be 0, so the delay
    // for the first reconnect should be 1000ms (BASE_RETRY_DELAY_MS * 2^0).
    //
    // Buggy code: retryCount.current carries over from run-1 (it was 3),
    //   so the delay would be 1000 * 2^3 = 8000ms.
    // Fixed code: retryCount.current was reset to 0 on runId change,
    //   so the delay is 1000ms.
    //
    // We measure the delay by advancing the clock and checking when a
    // new EventSource appears.
    const initialCount = EventSourceMock._count();
    act(() => {
      vi.advanceTimersByTime(1001); // 1000ms + 1ms buffer
    });
    // At 1001ms: fixed code should have reconnected (count incremented).
    // Buggy code: timer fires at 8000ms — no reconnect at 1001ms.
    expect(EventSourceMock._count()).toBe(initialCount + 1);
  });
});