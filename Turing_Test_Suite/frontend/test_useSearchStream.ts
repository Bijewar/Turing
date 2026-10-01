/**
 * test_useSearchStream.ts — Vitest tests for src/hooks/useSearchStream.ts
 *
 * Coverage:
 *   - SWARM-2 BUG #9  — setError not guarded by `cancelled` in onmessage
 *                       (same as useReportStream BUG #5)
 *   - SWARM-2 BUG #10 — useSearchStream doesn't expose `isLoading`
 *   - SWARM-7 BUG #22 — retryCount ref leaks across runId changes
 *
 * Strategy mirrors test_useReportStream.ts (mock EventSource, fake timers).
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useSearchStream } from '@/hooks/useSearchStream';
import { EventSourceMock } from './setup';

const SAMPLE_LAYER3 = {
  status: 'searching',
  query: 'q',
  streamSources: [],
  topBridges: [],
  isLocked: false,
};

beforeEach(() => {
  vi.useFakeTimers();
});
afterEach(() => {
  vi.useRealTimers();
});

describe('useSearchStream', () => {
  // -----------------------------------------------------------------------
  // BUG #9 — setError not guarded by `cancelled` (same as BUG #5)
  // -----------------------------------------------------------------------

  /**
   * SWARM-2 BUG #9 (observable variant)
   *
   * Same scenario as useReportStream.test_set_error_called_only_when_mounted
   * but for useSearchStream. When `runId` changes, the OLD EventSource's
   * `onmessage` closure still references the OLD `cancelled` flag (true).
   *
   * Buggy code: catch block calls `setError(err)` WITHOUT a guard → updates
   * the SAME hook's state (visible in result.current.error).
   * Fixed code: `if (cancelled) return;` prevents the setState.
   */
  it('test_set_error_called_only_when_mounted (BUG #9)', () => {
    const { result, rerender } = renderHook(
      ({ runId }) => useSearchStream(runId),
      { initialProps: { runId: 'run-1' } },
    );
    expect(EventSourceMock._count()).toBe(1);
    const firstEs = EventSourceMock._last()!;

    act(() => {
      rerender({ runId: 'run-2' });
    });
    expect(EventSourceMock._count()).toBe(2);
    expect(result.current.error).toBeNull();

    act(() => {
      firstEs._emitMessage('{not-valid-json');
    });

    // BUG #9: buggy code calls setError(err) (visible state change).
    expect(result.current.error).toBeNull();
  });

  /**
   * SWARM-2 BUG #9 (unmount variant — soft in React 18)
   *
   * After unmount, triggering an SSE message that would cause an error
   * should not produce a React warning (or any observable state change).
   * In React 18, setState on an unmounted component is a silent no-op, so
   * this test passes with both buggy and fixed code; it documents expected
   * behaviour and serves as a regression check on React 17/16.
   */
  it('test_does_not_set_state_after_unmount (BUG #9)', () => {
    const { result, unmount } = renderHook(() => useSearchStream('run-1'));
    expect(EventSourceMock._count()).toBe(1);
    const es = EventSourceMock._last()!;

    unmount();
    expect(es.close).toHaveBeenCalled();

    const errorBefore = result.current.error;

    act(() => {
      es._emitMessage('{not-valid-json');
    });

    expect(result.current.error).toBe(errorBefore);
  });

  // -----------------------------------------------------------------------
  // BUG #10 — useSearchStream doesn't expose isLoading
  // -----------------------------------------------------------------------

  /**
   * SWARM-2 BUG #10
   *
   * Unlike useReportStream which returns `isLoading`, useSearchStream only
   * returns `{ state, error, isConnected }`. Consumers have to derive
   * loading state themselves.
   *
   * The fix adds `isLoading: !state && !error` to the return value.
   */
  it('test_exposes_is_loading (BUG #10)', () => {
    const { result } = renderHook(() => useSearchStream('run-1'));
    // Before any message: state=null, error=null → isLoading should be true.
    expect(result.current.state).toBeNull();
    expect(result.current.error).toBeNull();
    // BUG #10: buggy code returns undefined for isLoading.
    expect(result.current).toHaveProperty('isLoading');
    expect(result.current.isLoading).toBe(true);

    // After a message arrives: state set, isLoading should be false.
    const es = EventSourceMock._last()!;
    act(() => {
      es._emitMessage(JSON.stringify(SAMPLE_LAYER3));
    });
    expect(result.current.isLoading).toBe(false);
  });

  // -----------------------------------------------------------------------
  // Reconnect / backoff
  // -----------------------------------------------------------------------

  /**
   * Exponential backoff: delays should be 1000, 2000, 4000, 8000, 16000.
   *
   * Buggy code DOES compute the exponential delay correctly. This test
   * is a regression check.
   */
  it('test_reconnect_with_backoff', () => {
    const { unmount } = renderHook(() => useSearchStream('run-1'));
    expect(EventSourceMock._count()).toBe(1);

    const delays: number[] = [];
    for (let i = 0; i < 5; i++) {
      const es = EventSourceMock._last()!;
      act(() => {
        es._emitError();
      });
      const expectedDelay = 1000 * Math.pow(2, i);
      delays.push(expectedDelay);
      act(() => {
        vi.advanceTimersByTime(expectedDelay + 10);
      });
      expect(EventSourceMock._count()).toBe(i + 2);
    }
    expect(delays).toEqual([1000, 2000, 4000, 8000, 16000]);
    unmount();
  });

  /**
   * After MAX_RETRIES (5), the hook must set `error` and stop scheduling
   * further reconnects.
   */
  it('test_max_retries_gives_up', () => {
    const { result, unmount } = renderHook(() => useSearchStream('run-1'));
    expect(EventSourceMock._count()).toBe(1);

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
    expect(EventSourceMock._count()).toBe(6);

    const es = EventSourceMock._last()!;
    act(() => {
      es._emitError();
    });
    act(() => {
      vi.advanceTimersByTime(60000);
    });
    expect(EventSourceMock._count()).toBe(6);
    expect(result.current.error).not.toBeNull();
    expect(result.current.error?.message).toMatch(/max retries/i);

    unmount();
  });

  // -----------------------------------------------------------------------
  // Cleanup
  // -----------------------------------------------------------------------

  it('test_closes_eventsource_on_unmount', () => {
    const { unmount } = renderHook(() => useSearchStream('run-1'));
    const es = EventSourceMock._last()!;
    unmount();
    expect(es.close).toHaveBeenCalled();
  });

  it('test_closes_eventsource_on_run_id_change', () => {
    const { rerender } = renderHook(
      ({ runId }) => useSearchStream(runId),
      { initialProps: { runId: 'run-1' } },
    );
    const firstEs = EventSourceMock._last()!;
    act(() => {
      rerender({ runId: 'run-2' });
    });
    expect(firstEs.close).toHaveBeenCalled();
    expect(EventSourceMock._count()).toBe(2);
    const secondEs = EventSourceMock._last()!;
    expect(secondEs.url).toContain('run-2');
  });

  // -----------------------------------------------------------------------
  // Message handling
  // -----------------------------------------------------------------------

  it('test_handles_malformed_json', () => {
    const { result } = renderHook(() => useSearchStream('run-1'));
    const es = EventSourceMock._last()!;
    act(() => {
      es._emitMessage('{not-valid-json');
    });
    expect(result.current.error).not.toBeNull();
    expect(result.current.error?.message).toMatch(/parse|json|chunk/i);
  });

  /**
   * When the backend sends `isLocked: true`, the hook must close the
   * EventSource (the search has locked onto an isomorphism).
   */
  it('test_stops_when_islocked_true', () => {
    const { result } = renderHook(() => useSearchStream('run-1'));
    const es = EventSourceMock._last()!;
    act(() => {
      es._emitMessage(JSON.stringify({ ...SAMPLE_LAYER3, isLocked: true }));
    });
    expect(es.close).toHaveBeenCalled();
    expect(result.current.state?.isLocked).toBe(true);
    expect(result.current.isConnected).toBe(false);
  });

  // -----------------------------------------------------------------------
  // SWARM-7 BUG #22 — retry count leaks across runId changes
  // -----------------------------------------------------------------------

  /**
   * SWARM-7 BUG #22
   *
   * Same as useReportStream.test_retry_count_resets_on_run_id_change.
   * After 3 errors on run-1, switch to run-2 and trigger 1 error.
   * The first reconnect for run-2 should fire at 1000ms (retryCount=0).
   *
   * Buggy code: retryCount.current carries over (3) → first reconnect at
   * 1000 * 2^3 = 8000ms.
   * Fixed code: retryCount.current reset to 0 → first reconnect at 1000ms.
   */
  it('test_retry_count_resets_on_run_id_change (SWARM-7 BUG #22)', () => {
    const { rerender } = renderHook(
      ({ runId }) => useSearchStream(runId),
      { initialProps: { runId: 'run-1' } },
    );

    const firstEs = EventSourceMock._last()!;
    for (let i = 0; i < 3; i++) {
      act(() => {
        firstEs._emitError();
      });
    }

    act(() => {
      rerender({ runId: 'run-2' });
    });
    expect(EventSourceMock._count()).toBe(2);

    const secondEs = EventSourceMock._last()!;
    act(() => {
      secondEs._emitError();
    });

    const initialCount = EventSourceMock._count();
    act(() => {
      vi.advanceTimersByTime(1001);
    });
    // Fixed: 1 reconnect at 1001ms.
    // Buggy: 0 reconnects (delay is 8000ms).
    expect(EventSourceMock._count()).toBe(initialCount + 1);
  });
});