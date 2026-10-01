import { vi, describe, test, expect, beforeEach, afterEach } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { fetchMock } from './setup';

/**
 * test_api.ts — Vitest tests for src/lib/api.ts
 *
 * We use the shared `fetchMock` from setup.ts so the per-test `beforeEach`
 * reset works correctly.  The shared mock defaults to returning a 404 so
 * unconfigured tests fail loudly.
 *
 * We import `apiUrl`, `createRun`, `uploadDataset` etc. via the `@/` alias.
 */

const BASE = 'http://127.0.0.1:8000';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SRC = path.resolve(HERE, '../../src');
const API_SRC = fs.readFileSync(path.join(SRC, 'lib/api.ts'), 'utf8');

beforeEach(() => {
  // fetchMock is reset by setup.ts's beforeEach; we don't need to do it here.
  // Set NEXT_PUBLIC_API_URL via process.env override (api.ts reads it at module
  // load via process.env.NEXT_PUBLIC_API_URL). Each test re-imports api.ts
  // fresh via vi.resetModules + dynamic import.
  delete process.env.NEXT_PUBLIC_API_URL;
});

afterEach(() => {
  vi.resetModules();
});

async function importApi() {
  return await import('@/lib/api');
}

describe('apiUrl', () => {
  /**
   * SWARM-7 BUG #10 — `apiUrl(path)` returns `${API_BASE}${path}`. If
   * API_BASE has a trailing slash and path starts with a slash, the result
   * has a double slash (e.g. `http://localhost:8000//api/runs`). Many
   * servers reject this with 404.
   *
   * The buggy code uses naive string concatenation. The fix should normalize
   * via `new URL(path, API_BASE)` or strip trailing slash.
   *
   * Strategy: set NEXT_PUBLIC_API_URL to a value with trailing slash, re-import
   * api.ts, and assert apiUrl('/api/runs') returns the joined URL without
   * a double slash.
   */
  test('test_apiUrl_handles_trailing_slash', async () => {
    process.env.NEXT_PUBLIC_API_URL = 'http://localhost:8000/';
    vi.resetModules();
    const { apiUrl } = await importApi();
    const url = apiUrl('/api/runs');
    expect(url).toBe('http://localhost:8000/api/runs');
    // Must NOT have a double slash.
    expect(url).not.toContain('//api/runs');
  });

  /** Same bug — path without leading slash should still produce a valid URL. */
  test('test_apiUrl_handles_missing_leading_slash', async () => {
    process.env.NEXT_PUBLIC_API_URL = 'http://localhost:8000';
    vi.resetModules();
    const { apiUrl } = await importApi();
    const url = apiUrl('api/runs');
    // Buggy: returns "http://localhost:8000api/runs" (no slash).
    expect(url).toBe('http://localhost:8000/api/runs');
  });
});

describe('createRun', () => {
  /**
   * SWARM-7 BUG #11 — createRun has no timeout. If the backend is dead, the
   * promise hangs forever. The fix should add an AbortController with a
   * timeout (e.g., 10s).
   *
   * Strategy: stub fetch to never resolve, start createRun, wait 50ms, and
   * assert the promise is rejected (because the AbortController's timeout
   * fires) — NOT still pending.
   *
   * Use fake timers and a small AbortSignal.timeout, or just check the
   * source for `AbortController` usage in createRun.
   */
  test('test_createRun_has_timeout', async () => {
    // Make fetch hang forever.
    fetchMock.mockImplementation(() => new Promise(() => {}));
    process.env.NEXT_PUBLIC_API_URL = BASE;
    vi.resetModules();
    const { createRun } = await importApi();

    let resolved = false;
    let rejected = false;
    // DO NOT await the promise — just start it and observe flags.
    createRun('data.csv', 'csv').then(
      () => { resolved = true; },
      () => { rejected = true; },
    );
    // Wait up to 200ms for the timeout to fire.
    await new Promise(r => setTimeout(r, 200));

    // The fix must reject (timeout fires); the buggy code stays pending.
    expect(rejected).toBe(true);
    expect(resolved).toBe(false);
  });

  /**
   * SWARM-7 BUG #14 — `res.json() as Promise<RunResponse>` is an unchecked
   * cast. If the backend returns a different shape, the frontend silently
   * uses garbage data.
   *
   * The fix should validate the response (e.g., check for `id` field,
   * throw if missing).
   *
   * Strategy: make fetch return a JSON response missing the `id` field, and
   * assert createRun throws (not silently returns garbage).
   */
  test('test_createRun_validates_response', async () => {
    fetchMock.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ not_id: 'foo', status: 'RUNNING' }), // missing `id`
      text: async () => JSON.stringify({ not_id: 'foo' }),
    } as any);
    process.env.NEXT_PUBLIC_API_URL = BASE;
    vi.resetModules();
    const { createRun } = await importApi();

    let threw = false;
    let result: any = null;
    try {
      result = await createRun('data.csv', 'csv');
    } catch (e) {
      threw = true;
    }
    // The buggy code returns the response as-is (no validation), so `result`
    // would be `{ not_id: 'foo' }`. The fix should throw.
    expect(threw).toBe(true);
    expect(result).toBeNull();
  });

  /**
   * SWARM-7 BUG #17 — `createRun(inputType: "csv" | "text")` restricts to
   * "csv" | "text" but the backend's RunCreate defaults to "DATA_PATH" and
   * accepts any string. The frontend should accept `"DATA"` (newer backend
   * value) and `"TEXT"` and reject `"DATA_PATH"` and `"csv"` (lowercase
   * mismatch).
   *
   * Wait — re-reading the bug: the backend uses 'csv'/'text' (lowercase),
   * and the default is "DATA_PATH". The TS type restricts to "csv" | "text".
   * The bug is that the backend has a third value ("DATA_PATH") that the TS
   * type doesn't allow.
   *
   * For the test, we test that the function accepts the documented valid
   * values ('csv', 'text') AND rejects clearly-invalid values ('DATA_PATH',
   * 'csv' is fine; 'DATA' might be a new value that should be accepted).
   *
   * Per the task description: `input_type="DATA"` and `"TEXT"` are valid;
   * `"DATA_PATH"` and `"csv"` are rejected. This suggests the fix should
   * switch to the canonical uppercase backend values.
   *
   * Static-source check: the createRun signature should accept the new
   * value space. We import the createRun function and assert the typescript
   * signature includes the new values.
   */
  test('test_input_type_valid_values', async () => {
    process.env.NEXT_PUBLIC_API_URL = BASE;
    vi.resetModules();
    await importApi();

    // Buggy: `createRun(inputFile: string, inputType: "csv" | "text")`.
    // Fixed: `createRun(inputFile: string, inputType: "DATA" | "TEXT")` (or includes them).
    // Check that the union includes "DATA" and "TEXT".
    const sigMatch = API_SRC.match(
      /export\s+async\s+function\s+createRun\s*\(([^)]*)\)/,
    );
    expect(sigMatch).toBeTruthy();
    const sig = sigMatch![1];
    expect(sig).toContain('"DATA"');
    expect(sig).toContain('"TEXT"');
    // And should NOT contain the old lowercase "csv" / "text" (or at least
    // the canonical new values must be there).
  });
});

describe('uploadDataset', () => {
  /**
   * SWARM-7 BUG #11/12 — uploadDataset has no timeout. The fix should add an
   * AbortController with a timeout.
   */
  test('test_uploadDataset_has_timeout', async () => {
    fetchMock.mockImplementation(() => new Promise(() => {}));
    process.env.NEXT_PUBLIC_API_URL = BASE;
    vi.resetModules();
    const { uploadDataset } = await importApi();

    const file = new File(['data'], 'data.csv', { type: 'text/csv' });
    let resolved = false;
    let rejected = false;
    // DO NOT await — observe flags.
    uploadDataset('run-id', file).then(
      () => { resolved = true; },
      () => { rejected = true; },
    );
    await new Promise(r => setTimeout(r, 200));

    // Buggy code hangs forever; fix rejects via timeout.
    expect(rejected).toBe(true);
    expect(resolved).toBe(false);
  });

  /**
   * SWARM-7 BUG #12 — uploadDataset should validate file size before sending.
   * Files > 50MB should throw an error before fetch is called.
   */
  test('test_uploadDataset_validates_file_size', async () => {
    process.env.NEXT_PUBLIC_API_URL = BASE;
    vi.resetModules();
    const { uploadDataset } = await importApi();

    // Create a File with size > 50MB (50 * 1024 * 1024 + 1 bytes).
    const hugeSize = 50 * 1024 * 1024 + 1;
    const file = new File(['x'.repeat(1024)], 'huge.csv', { type: 'text/csv' });
    // Override the size property (jsdom File.size is byte-accurate; we need a >50MB file).
    Object.defineProperty(file, 'size', { value: hugeSize, configurable: true });

    let threw = false;
    let fetchCalled = false;
    fetchMock.mockImplementation(() => {
      fetchCalled = true;
      return Promise.resolve({ ok: true } as any);
    });
    try {
      await uploadDataset('run-id', file);
    } catch (e) {
      threw = true;
    }
    // The buggy code sends the file regardless of size (fetchCalled=true,
    // threw=false). The fix should throw BEFORE calling fetch (threw=true,
    // fetchCalled=false).
    expect(threw).toBe(true);
    expect(fetchCalled).toBe(false);
  });

  /**
   * SWARM-7 BUG #26 — uploadDataset should accept an AbortController (or
   * AbortSignal) for cancellation. The fix should accept an optional `signal`
   * parameter and pass it to fetch.
   *
   * Static-source check: the uploadDataset signature should include an
   * AbortSignal parameter.
   */
  test('test_uploadDataset_supports_abort', async () => {
    const sigMatch = API_SRC.match(
      /export\s+async\s+function\s+uploadDataset\s*\(([^)]*)\)/,
    );
    expect(sigMatch).toBeTruthy();
    const sig = sigMatch![1];
    // Buggy: `uploadDataset(runId: string, file: File)` — no signal.
    // Fixed: `uploadDataset(runId: string, file: File, signal?: AbortSignal)`.
    expect(sig).toMatch(/AbortSignal|signal/);
  });
});