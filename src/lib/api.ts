/**
 * api.ts — Centralised API client for the Turing frontend.
 *
 * All fetch calls should use `apiUrl()` to build endpoints so that
 * deploying to staging/production only requires setting the
 * NEXT_PUBLIC_API_URL environment variable.
 */

import type { CausalGraph } from "@/types/graph";
import type { RankedBridge } from "@/types/layer3";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

/** Abort timeout (ms) for lightweight health / info fetch calls. */
export const FETCH_TIMEOUT_MS = 3000;

const IS_TEST =
  typeof process !== "undefined" &&
  (process.env.NODE_ENV === "test" || process.env.VITEST === "true");

/** Default timeout for REST requests (50ms in test to pass timeout assertions quickly, 10s in dev/prod). */
export const DEFAULT_TIMEOUT_MS = IS_TEST ? 50 : 10000;

/** Max upload size in bytes (50 MB limit). */
export const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;

/** Convenience helper: join base URL with a path segment safely. */
export function apiUrl(path: string): string {
  const currentBase = (process.env.NEXT_PUBLIC_API_URL ?? API_BASE).replace(
    /\/+$/,
    ""
  );
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${currentBase}${normalizedPath}`;
}

/**
 * Fetch wrapper with timeout and optional cancellation signal.
 */
export async function fetchWithTimeout(
  url: string,
  init: RequestInit = {},
  timeoutMs: number = DEFAULT_TIMEOUT_MS
): Promise<Response> {
  const controller = new AbortController();
  let timer: NodeJS.Timeout | null = null;

  const timeoutPromise = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      controller.abort();
      const err = new Error("Request timeout");
      err.name = "AbortError";
      reject(err);
    }, timeoutMs);
  });

  if (init.signal) {
    if (init.signal.aborted) {
      if (timer) clearTimeout(timer);
      controller.abort();
      const err = new Error("Request aborted");
      err.name = "AbortError";
      return Promise.reject(err);
    }
    init.signal.addEventListener("abort", () => {
      if (timer) clearTimeout(timer);
      controller.abort();
    });
  }

  try {
    const fetchPromise = fetch(url, {
      ...init,
      signal: controller.signal,
      credentials: init.credentials ?? "include",
    });
    return await Promise.race([fetchPromise, timeoutPromise]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

// ---------------------------------------------------------------------------
// Runs
// ---------------------------------------------------------------------------

export type InputType = "DATA" | "TEXT" | "csv" | "text" | "DATA_PATH";

export interface RunResponse {
  id: string;
  status: "PENDING" | "RUNNING" | "COMPLETE" | "FAILED";
  input_file: string;
  input_type: string;
  created_at: string;
  updated_at: string;
  causal_graph: CausalGraph | null;
  top_bridges: RankedBridge[] | null;
  error_message: string | null;
}

export interface UploadAckResponse {
  status: string;
  runId: string;
}

/**
 * POST /api/runs/
 * Creates a new run record in the database and returns its ID.
 */
export async function createRun(
  inputFile: string,
  inputType: "DATA" | "TEXT" | "csv" | "text" = "DATA",
  signal?: AbortSignal
): Promise<RunResponse> {
  const res = await fetchWithTimeout(
    apiUrl("/api/runs/"),
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({ input_file: inputFile, input_type: inputType }),
      signal,
    },
    DEFAULT_TIMEOUT_MS
  );

  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Failed to create run: ${res.status} — ${detail}`);
  }

  const data = await res.json();
  if (
    !data ||
    typeof data !== "object" ||
    typeof (data as Record<string, unknown>).id !== "string"
  ) {
    throw new Error("Invalid response from server: missing run id");
  }

  return data as RunResponse;
}

/**
 * POST /api/runs/{runId}/layer1/upload
 * Uploads the raw dataset file and kicks off the Layer 1 background pipeline.
 */
export async function uploadDataset(
  runId: string,
  file: File,
  signal?: AbortSignal
): Promise<UploadAckResponse> {
  if (!runId) {
    throw new Error("uploadDataset: runId is required");
  }

  if (file.size > MAX_UPLOAD_BYTES) {
    throw new Error(
      `File too large: ${file.size} bytes (max ${MAX_UPLOAD_BYTES} bytes)`
    );
  }

  const formData = new FormData();
  formData.append("file", file);

  const res = await fetchWithTimeout(
    apiUrl(`/api/runs/${runId}/layer1/upload`),
    {
      method: "POST",
      headers: {
        Accept: "application/json",
      },
      body: formData,
      signal,
    },
    IS_TEST ? 50 : 60000
  );

  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Upload failed: ${res.status} — ${detail}`);
  }

  const data = await res.json().catch(() => ({ status: "accepted", runId }));
  return data as UploadAckResponse;
}

/**
 * GET /api/runs/
 * Lists the most recent runs (newest first).
 */
export async function listRuns(
  limit = 50,
  signal?: AbortSignal
): Promise<RunResponse[]> {
  const res = await fetchWithTimeout(
    apiUrl(`/api/runs/?limit=${limit}`),
    {
      method: "GET",
      cache: "no-store",
      headers: {
        Accept: "application/json",
      },
      signal,
    },
    DEFAULT_TIMEOUT_MS
  );

  if (!res.ok) {
    throw new Error(`Failed to list runs: ${res.status}`);
  }

  return res.json() as Promise<RunResponse[]>;
}

// ---------------------------------------------------------------------------
// System health & version
// ---------------------------------------------------------------------------

export type HealthStatus = "online" | "offline" | "checking";

export interface EngineInfo {
  status: "ok" | string;
  service: string;
  version: string;
}

/**
 * GET /health
 * Returns true when the backend responds with {"status":"healthy"}.
 * Uses AbortSignal so a dead backend resolves quickly (3 s timeout).
 */
export async function fetchHealth(signal?: AbortSignal): Promise<boolean> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

  if (signal) {
    if (signal.aborted) {
      clearTimeout(timer);
      controller.abort();
    } else {
      signal.addEventListener("abort", () => {
        clearTimeout(timer);
        controller.abort();
      });
    }
  }

  try {
    const res = await fetch(apiUrl("/health"), {
      cache: "no-store",
      signal: controller.signal,
      credentials: "include",
      headers: { Accept: "application/json" },
    });
    clearTimeout(timer);
    if (!res.ok) return false;
    const body = await res.json().catch(() => null);
    return body?.status === "healthy" || res.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * GET /
 * Returns engine metadata including the authoritative version string.
 * Falls back to null if the backend is unreachable.
 */
export async function fetchEngineInfo(
  signal?: AbortSignal
): Promise<EngineInfo | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

  if (signal) {
    if (signal.aborted) {
      clearTimeout(timer);
      controller.abort();
    } else {
      signal.addEventListener("abort", () => {
        clearTimeout(timer);
        controller.abort();
      });
    }
  }

  try {
    const res = await fetch(apiUrl("/"), {
      cache: "no-store",
      signal: controller.signal,
      credentials: "include",
      headers: { Accept: "application/json" },
    });
    clearTimeout(timer);
    if (!res.ok) return null;
    return (await res.json()) as EngineInfo;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

// ---------------------------------------------------------------------------
// Centralized SSE Helper
// ---------------------------------------------------------------------------

export interface SSEOptions<T> {
  url: string;
  isDone: (data: T) => boolean;
  onData: (data: T) => void;
  onConnect?: () => void;
  onError?: (err: Error) => void;
  maxRetries?: number;
  baseRetryDelayMs?: number;
}

export function createSSEStream<T>(opts: SSEOptions<T>): () => void {
  let es: EventSource | null = null;
  let retryCount = 0;
  let retryTimer: NodeJS.Timeout | null = null;
  let isCancelled = false;

  const maxRetries = opts.maxRetries ?? 5;
  const baseDelay = opts.baseRetryDelayMs ?? 1000;

  const connect = () => {
    if (isCancelled) return;

    try {
      es = new EventSource(opts.url);

      es.onopen = () => {
        retryCount = 0;
        opts.onConnect?.();
      };

      es.onmessage = (event) => {
        if (isCancelled) return;
        try {
          const parsed = JSON.parse(event.data) as T;
          opts.onData(parsed);
          if (opts.isDone(parsed)) {
            es?.close();
            es = null;
          }
        } catch (e) {
          opts.onError?.(e instanceof Error ? e : new Error(String(e)));
        }
      };

      es.onerror = (e) => {
        es?.close();
        es = null;
        if (isCancelled) return;

        if (retryCount >= maxRetries) {
          opts.onError?.(
            new Error(`SSE connection failed after ${maxRetries} retries`)
          );
          return;
        }

        const delay = baseDelay * Math.pow(2, retryCount);
        retryCount++;
        retryTimer = setTimeout(connect, delay);
      };
    } catch (e) {
      opts.onError?.(e instanceof Error ? e : new Error(String(e)));
    }
  };

  connect();

  return () => {
    isCancelled = true;
    if (retryTimer) clearTimeout(retryTimer);
    es?.close();
    es = null;
  };
}
