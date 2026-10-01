// Minimal stub for next/navigation so the import graph resolves in tests.
// We expose the hooks as jest/vitest-style helpers so tests can drive them.

type Listener = (params: { runId?: string | string[] } | null) => void;
const listeners = new Set<Listener>();
let currentParams: { runId?: string | string[] } | null = null;

export function __setParams(params: { runId?: string | string[] } | null) {
  currentParams = params;
  for (const l of listeners) l(params);
}

export function __resetParams() {
  currentParams = null;
  listeners.clear();
}

export function useParams(): { runId?: string | string[] } | null {
  return currentParams;
}

export function useRouter() {
  return {
    replace: vi.fn ? vi.fn() : (() => {}),
    push: vi.fn ? vi.fn() : (() => {}),
    refresh: vi.fn ? vi.fn() : (() => {}),
    back: vi.fn ? vi.fn() : (() => {}),
    prefetch: vi.fn ? vi.fn() : (() => {}),
  };
}

export function notFound(): never {
  throw new Error('NEXT_NOT_FOUND');
}

export function usePathname() { return '/run/abc'; }
export function useSearchParams() {
  return {
    get: (_: string) => null,
    has: (_: string) => false,
  };
}
export function redirect(_path: string): never {
  throw new Error('NEXT_REDIRECT:' + _path);
}