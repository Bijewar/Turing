import { useState, useEffect } from "react";
import { fetchHealth, fetchEngineInfo, HealthStatus } from "@/lib/api";

export interface SystemStatus {
  health: HealthStatus;
  version: string | null;
  versionError: Error | null;
}

const POLL_INTERVAL_MS = 30_000; // re-check every 30 seconds

/**
 * useSystemStatus
 *
 * Checks backend liveness on mount and then every 30 seconds.
 * Also fetches the authoritative version string from GET /.
 *
 * Returns:
 *   health       — "checking" | "online" | "offline"
 *   version      — e.g. "1.0.0", or null while loading / unreachable
 *   versionError — Error if backend unreachable, or null
 */
export function useSystemStatus(): SystemStatus {
  const [health, setHealth] = useState<HealthStatus>("checking");
  const [version, setVersion] = useState<string | null>(null);
  const [versionError, setVersionError] = useState<Error | null>(null);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();

    const check = async () => {
      try {
        const alive = await fetchHealth(controller.signal);
        if (!cancelled) setHealth(alive ? "online" : "offline");
      } catch {
        if (!cancelled) setHealth("offline");
      }
    };

    // Fetch version once (doesn't need to poll — version only changes on deploy)
    const getVersion = async () => {
      try {
        const info = await fetchEngineInfo(controller.signal);
        if (!cancelled) {
          if (info) {
            setVersion(info.version ? info.version : null);
            setVersionError(null);
          } else {
            setVersion(null);
            setVersionError(new Error("Backend unreachable"));
          }
        }
      } catch (err) {
        if (!cancelled) {
          setVersion(null);
          setVersionError(
            err instanceof Error ? err : new Error("Backend unreachable")
          );
        }
      }
    };

    check();
    getVersion();

    const intervalId = setInterval(check, POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      controller.abort();
      clearInterval(intervalId);
    };
  }, []);

  return { health, version, versionError };
}
