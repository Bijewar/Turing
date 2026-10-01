import '@testing-library/jest-dom/vitest';
import { afterEach, beforeEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';

// Auto-cleanup between tests so each test gets a fresh DOM.
afterEach(() => {
  cleanup();
  // Reset window body overflow changes from Modal tests etc.
  document.body.style.overflow = '';
  // Clear all mocks between tests so call-counts are deterministic.
  vi.clearAllMocks();
});

// jsdom doesn't implement layout — give getBoundingClientRect a sensible default.
if (!Element.prototype.getBoundingClientRect) {
  Element.prototype.getBoundingClientRect = () => ({
    x: 0, y: 0, top: 0, left: 0, right: 0, bottom: 0,
    width: 800, height: 600, toJSON: () => ({}),
  });
}
// SVG elements in jsdom don't have clientWidth/Height — give them defaults.
Object.defineProperty(HTMLElement.prototype, 'clientWidth', {
  configurable: true,
  get() { return 900; },
});
Object.defineProperty(HTMLElement.prototype, 'clientHeight', {
  configurable: true,
  get() { return 600; },
});
// SVGSVGElement clientWidth/Height
if (typeof SVGSVGElement !== 'undefined') {
  Object.defineProperty(SVGSVGElement.prototype, 'clientWidth', {
    configurable: true,
    get() { return 900; },
  });
  Object.defineProperty(SVGSVGElement.prototype, 'clientHeight', {
    configurable: true,
    get() { return 600; },
  });
}

// requestAnimationFrame shim — jsdom doesn't polyfill it.
if (typeof globalThis.requestAnimationFrame !== 'function') {
  globalThis.requestAnimationFrame = (cb: FrameRequestCallback) =>
    setTimeout(() => cb(performance.now()), 0) as unknown as number;
  globalThis.cancelAnimationFrame = (id: number) => clearTimeout(id);
}

// matchMedia is not in jsdom.
if (!window.matchMedia) {
  // @ts-expect-error - jsdom doesn't have matchMedia
  window.matchMedia = (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  });
}

// ---------------------------------------------------------------------------
// 0. Environment / API URL
// ---------------------------------------------------------------------------
process.env.NEXT_PUBLIC_API_URL = 'http://test-api.local';

// ---------------------------------------------------------------------------
// 1. Mock EventSource
// ---------------------------------------------------------------------------
//
// SSE hooks (useReportStream, useSearchStream) construct `new EventSource(url)`.
// We replace the global EventSource with a minimal stub that:
//   - records every instance created
//   - exposes onopen/onmessage/onerror as assignable properties
//   - exposes a `.close()` spy so tests can assert cleanup
//   - exposes a `.url` so tests can verify which endpoint was hit
//
// The mock does NOT auto-fire any events — tests schedule them explicitly
// via the `_emitOpen/_emitMessage/_emitError` helper methods.

export interface MockEventSource {
  url: string;
  onopen: ((ev: Event) => void) | null;
  onmessage: ((ev: MessageEvent) => void) | null;
  onerror: ((ev: Event) => void) | null;
  readyState: number;
  close: ReturnType<typeof vi.fn>;
  // Helpers used by tests to dispatch synthetic events without typing pain
  _emitOpen(): void;
  _emitMessage(data: string, lastEventId?: string): void;
  _emitError(): void;
}

const _instances: MockEventSource[] = [];

// The global EventSource constructor the hooks will see
class EventSourceMock {
  static _instances = _instances;
  static _last(): MockEventSource | undefined {
    return _instances[_instances.length - 1];
  }
  static _at(idx: number): MockEventSource | undefined {
    return _instances[idx];
  }
  static _count(): number {
    return _instances.length;
  }
  static _reset() {
    _instances.length = 0;
  }

  // instance fields mirror the spec'd EventSource surface area
  url: string;
  onopen: ((ev: Event) => void) | null = null;
  onmessage: ((ev: MessageEvent) => void) | null = null;
  onerror: ((ev: Event) => void) | null = null;
  readyState = 0;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  withCredentials = false;
  close: ReturnType<typeof vi.fn>;

  constructor(url: string) {
    this.url = url;
    this.close = vi.fn(() => {
      this.readyState = 2;
    });
    _instances.push(this as unknown as MockEventSource);
  }

  // Convenience helpers (also available via `EventSourceMock._last()`)
  _emitOpen() {
    this.readyState = 1;
    this.onopen?.(new Event('open'));
  }
  _emitMessage(data: string, lastEventId = '') {
    this.onmessage?.(
      new MessageEvent('message', { data, lastEventId, origin: this.url }),
    );
  }
  _emitError() {
    this.onerror?.(new Event('error'));
  }
}

// Make the mock globally visible to the hooks-under-test
// eslint-disable-next-line @typescript-eslint/no-explicit-any
(globalThis as any).EventSource = EventSourceMock;

export { EventSourceMock };

// ---------------------------------------------------------------------------
// 2. Mock fetch
// ---------------------------------------------------------------------------
//
// We install a vi.fn() so each test can call fetch.mockResolvedValueOnce(...)
// or fetch.mockRejectedValueOnce(...).  The mock returns a Response-shaped
// object with `.ok`, `.status`, `.json()` and `.text()` helpers.

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const fetchMock = vi.fn() as any;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function makeResponse(body: any, init: { status?: number; ok?: boolean } = {}): any {
  const status = init.status ?? 200;
  const ok = init.ok ?? (status >= 200 && status < 300);
  // NOTE: We keep a reference to the raw body and defer serialization until
  // `.json()` or `.text()` is called. This is important for tests that
  // want to pass bodies with circular references (which would otherwise
  // throw `TypeError: Converting circular structure to JSON` inside the
  // mock fetch itself, before the hook's setGraph updater can be tested).
  const serialize = (): string =>
    typeof body === 'string' ? body : JSON.stringify(body);
  return {
    ok,
    status,
    headers: new Map<string, string>(),
    json: async () => JSON.parse(serialize()),
    text: async () => serialize(),
    clone() {
      return makeResponse(body, init);
    },
  };
}

// Default implementation: return a 404 so unconfigured tests fail loudly
// rather than silently returning undefined.
fetchMock.mockImplementation(() =>
  Promise.resolve(makeResponse({ error: 'not mocked' }, { status: 404 })),
);

// eslint-disable-next-line @typescript-eslint/no-explicit-any
(globalThis as any).fetch = fetchMock;

// ---------------------------------------------------------------------------
// 3. AbortController tracking
// ---------------------------------------------------------------------------
//
// Some hooks (useRunState, useAgentLoop, useGraphAnimation) SHOULD use an
// AbortController to cancel in-flight fetches on unmount / runId change.
// We wrap AbortController so tests can assert on `.abort()` calls.
//
// We do NOT replace the class itself — we just track each instance's
// `.abort()` invocations on a shared list so tests can introspect them.
export const abortCalls: AbortController[] = [];

class TrackedAbortController extends AbortController {
  abort(reason?: unknown) {
    abortCalls.push(this);
    // @ts-ignore — abort signature varies across TS lib versions
    return super.abort(reason);
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
(globalThis as any).AbortController = TrackedAbortController;

// ---------------------------------------------------------------------------
// 4. console.error capture
// ---------------------------------------------------------------------------
//
// Tests that check for "setState on unmounted component" warnings will use
// the `consoleErrors` array exposed here.
export const consoleErrors: string[] = [];
const originalConsoleError = console.error;
beforeEach(() => {
  consoleErrors.length = 0;
  console.error = (...args: unknown[]) => {
    consoleErrors.push(
      args
        .map((a) => (a instanceof Error ? `${a.name}: ${a.message}` : String(a)))
        .join(' '),
    );
  };
});
afterEach(() => {
  console.error = originalConsoleError;
});

// ---------------------------------------------------------------------------
// 5. Per-test resets
// ---------------------------------------------------------------------------
beforeEach(() => {
  // Reset EventSource mock state
  EventSourceMock._reset();
  // Reset fetch mock calls/implementation but keep the default 404 fallback
  fetchMock.mockReset();
  fetchMock.mockImplementation(() =>
    Promise.resolve(makeResponse({ error: 'not mocked' }, { status: 404 })),
  );
  // Reset AbortController tracker
  abortCalls.length = 0;
  // Clear console errors
  consoleErrors.length = 0;
  // ensure the test runner always has a clean default body overflow.
  document.body.style.overflow = '';
});