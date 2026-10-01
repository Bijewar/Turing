/**
 * test_useSystemStatus.ts — Vitest tests for src/hooks/useSystemStatus.ts
 *
 * Coverage:
 *   - SWARM-2 BUG #13 — getVersion() swallows errors silently; the hook
 *                       does NOT expose a `versionError` state.
 *   - The hook must distinguish "backend down" (500 / network error) from
 *     "backend has no version field" (200 with no version).
 *
 * Strategy:
 *   - Mock `@/lib/api`'s `fetchHealth` and `fetchEngineInfo` exports via
 *     vi.mock, so we don't have to drive the fetchMock through the
 *     higher-level helpers.
 *   - vi.useFakeTimers() for the 30-second polling interval.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

// We must declare the mock BEFORE importing the hook (so the hook gets
// the mocked version of @/lib/api).
const fetchHealthMock = vi.fn();
const fetchEngineInfoMock = vi.fn();

vi.mock('@/lib/api', () => ({
  fetchHealth: (...args: unknown[]) => fetchHealthMock(...args),
  fetchEngineInfo: (...args: unknown[]) => fetchEngineInfoMock(...args),
}));

import { useSystemStatus } from '@/hooks/useSystemStatus';
import { abortCalls } from './setup';

beforeEach(() => {
  vi.useFakeTimers();
  // Default: fetchHealth returns false (offline), fetchEngineInfo returns null.
  fetchHealthMock.mockReset();
  fetchEngineInfoMock.mockReset();
  fetchHealthMock.mockResolvedValue(false);
  fetchEngineInfoMock.mockResolvedValue(null);
});
afterEach(() => {
  vi.useRealTimers();
});

describe('useSystemStatus', () => {
  // -----------------------------------------------------------------------
  // BUG #13 — versionError not exposed
  // -----------------------------------------------------------------------

  /**
   * SWARM-2 BUG #13 — when the backend is unreachable (fetchEngineInfo
   * returns null), the hook keeps `version: null` but does NOT expose a
   * `versionError` state. The UI cannot distinguish "backend has no
   * version field" from "backend unreachable".
   *
   * Fixed code: expose a `versionError` state that's set when the
   * fetchEngineInfo call fails (network error, 500, etc.).
   */
  it('test_get_version_exposes_error_state (BUG #13)', async () => {
    fetchHealthMock.mockResolvedValue(true); // online
    fetchEngineInfoMock.mockResolvedValue(null); // unreachable / error

    const { result } = renderHook(() => useSystemStatus());

    // Allow initial promises to settle.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    // The hook MUST expose a versionError state.
    expect(result.current).toHaveProperty('versionError');
    // When fetchEngineInfo returns null (error / unreachable), versionError
    // must be set (not null/undefined).
    expect(result.current.versionError).not.toBe(undefined);
    expect(result.current.versionError).not.toBe(null);
  });

  /**
   * SWARM-2 BUG #13 (distinguish backend-down from no-version)
   *
   * When the backend returns 200 with no version field (e.g., the
   * response is `{ status: 'ok', service: 'engine' }` with no `version`),
   * the hook should set `version: null` AND `versionError: null` (or a
   * specific "no version" marker), NOT `versionError` set to an Error.
   *
   * When the backend is down (500 / network error), the hook should set
   * `versionError` to an Error.
   *
   * Buggy code: both cases silently set version=null, no versionError —
   * they're indistinguishable.
   *
   * Fixed code: distinguishes between the two cases.
   */
  it('test_distiguishes_backend_down_from_no_version (BUG #13)', async () => {
    // Case 1: Backend is online, returns 200, but the response has no
    // version field. The hook should expose version=null but NO error.
    fetchHealthMock.mockResolvedValue(true);
    fetchEngineInfoMock.mockResolvedValue({
      status: 'ok',
      service: 'engine',
      version: '',  // empty version — backend has no version field
    });

    let { result, unmount } = renderHook(() => useSystemStatus());
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    // Backend reachable but no version → versionError should NOT be an Error.
    // (Either null, or a "no version field" marker — not a network error.)
    expect(result.current).toHaveProperty('versionError');
    // When version field is missing/empty but backend is reachable,
    // versionError should not indicate a network error.
    if (result.current.versionError instanceof Error) {
      // If versionError IS set to an Error here, that's the BUG (can't
      // distinguish from "backend down").
      expect(result.current.versionError.message).not.toMatch(
        /network|unreachable|fetch|down/i,
      );
    }
    unmount();

    // Case 2: Backend is down (network error / 500). fetchEngineInfo
    // returns null. The hook should expose versionError as a real Error.
    fetchHealthMock.mockResolvedValue(false);
    fetchEngineInfoMock.mockResolvedValue(null);

    ({ result, unmount } = renderHook(() => useSystemStatus()));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(result.current).toHaveProperty('versionError');
    expect(result.current.versionError).not.toBe(null);
    expect(result.current.versionError).not.toBe(undefined);
    unmount();
  });

  // -----------------------------------------------------------------------
  // Polling
  // -----------------------------------------------------------------------

  /**
   * The hook should call fetchHealth on mount, and again every 30s.
   */
  it('test_polls_for_status_periodically', async () => {
    fetchHealthMock.mockResolvedValue(true);
    fetchEngineInfoMock.mockResolvedValue({ status: 'ok', service: 'engine', version: '1.0.0' });

    renderHook(() => useSystemStatus());

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    // Initial call on mount.
    expect(fetchHealthMock).toHaveBeenCalledTimes(1);

    // Advance 30s — should fire 1 more poll.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000);
    });
    expect(fetchHealthMock).toHaveBeenCalledTimes(2);

    // Advance another 60s — should fire 2 more polls.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000);
    });
    expect(fetchHealthMock).toHaveBeenCalledTimes(4);
  });

  /**
   * On unmount, the polling interval must be cleared so no more polls
   * fire. (This is already correctly handled by the buggy code's
   * `clearInterval(intervalId)` in cleanup.)
   */
  it('test_stops_polling_on_unmount', async () => {
    fetchHealthMock.mockResolvedValue(true);

    const { unmount } = renderHook(() => useSystemStatus());

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    const initialCount = fetchHealthMock.mock.calls.length;

    unmount();

    // Advance 90s — should NOT fire any more polls.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(90_000);
    });

    expect(fetchHealthMock.mock.calls.length).toBe(initialCount);
  });

  /**
   * The hook's in-flight fetchHealth / fetchEngineInfo should be aborted
   * via AbortController on unmount (so a dead backend's 3-second timeout
   * doesn't tie up a defunct component).
   *
   * The buggy code does not create an AbortController for the polling
   * fetches — `cancelled` guards the setState calls but doesn't actually
   * cancel the in-flight network request.
   */
  it('test_uses_abortcontroller', async () => {
    // Make fetchHealth hang (never resolves) so unmount mid-fetch
    // is observable via the AbortController.abort() call.
    fetchHealthMock.mockImplementation(
      () => new Promise<boolean>(() => {}),
    );
    fetchEngineInfoMock.mockResolvedValue(null);

    const { unmount } = renderHook(() => useSystemStatus());

    // Let the initial fetchHealth call start (returns a hanging promise).
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    unmount();

    expect(abortCalls.length).toBeGreaterThanOrEqual(1);
  });
});