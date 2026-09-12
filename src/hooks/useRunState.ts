import { useState, useEffect } from 'react';
import { RunState } from '../types/run';
import { apiUrl } from '@/lib/api';

/** How often (ms) to poll the backend for the global run state. */
const POLL_INTERVAL_MS = 1500;
const MAX_BACKOFF_MS = 30000;

export function useRunState(runId: string) {
  const [runState, setRunState] = useState<RunState | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    if (!runId) return;

    let isMounted = true;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    let intervalId: ReturnType<typeof setInterval> | null = null;
    let consecutiveErrors = 0;
    let lastRequestId = 0;
    const controller = new AbortController();

    const isTerminal = (data: RunState) => {
      const record = data as unknown as Record<string, unknown>;
      return (
        (data.currentLayer === 4 && data.progressPercentage === 100) ||
        data.layer1Status === 'error' ||
        data.layer2Status === 'error' ||
        data.layer3Status === 'error' ||
        data.layer4Status === 'error' ||
        record.status === 'COMPLETE' ||
        record.status === 'FAILED'
      );
    };

    const stopPolling = () => {
      if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
      }
      if (timeoutId) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }
    };

    const fetchRunState = async () => {
      const currentReqId = ++lastRequestId;

      try {
        const response = await fetch(apiUrl(`/api/runs/${runId}/state`), {
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data: RunState = await response.json();

        if (isMounted && currentReqId === lastRequestId) {
          setRunState(data);
          setError(null);
          setIsLoading(false);
          consecutiveErrors = 0;

          if (isTerminal(data)) {
            stopPolling();
          }
        }
      } catch (err: unknown) {
        if (!isMounted) return;
        if (err instanceof DOMException && err.name === 'AbortError') return;

        if (currentReqId === lastRequestId) {
          setError(err instanceof Error ? err : new Error('Unknown error fetching global RunState'));
          setIsLoading(false);
          consecutiveErrors += 1;

          if (intervalId) {
            clearInterval(intervalId);
            intervalId = null;
          }
          if (timeoutId) {
            clearTimeout(timeoutId);
          }
          const backoffDelay = Math.min(
            MAX_BACKOFF_MS,
            POLL_INTERVAL_MS * Math.pow(2, consecutiveErrors)
          );
          timeoutId = setTimeout(() => {
            if (isMounted) fetchRunState();
          }, backoffDelay);
        }
      }
    };

    fetchRunState();
    intervalId = setInterval(fetchRunState, POLL_INTERVAL_MS);

    return () => {
      isMounted = false;
      controller.abort();
      stopPolling();
    };
  }, [runId]);

  return { runState, error, isLoading };
}

