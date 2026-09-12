import { useState, useEffect } from 'react';
import { Layer2State } from '../types/layer2';
import { apiUrl } from '@/lib/api';

/** How often (ms) to poll for live Layer 2 agent swarm debate updates. */
const POLL_INTERVAL_MS = 1000;
const MAX_BACKOFF_MS = 30000;

export function useAgentLoop(runId: string) {
  const [state, setState] = useState<Layer2State | null>(null);
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

    const isTerminal = (data: Layer2State) => {
      const status = (data.status || '').toUpperCase();
      return status === 'COMPLETE' || status === 'FAILED' || status === 'COMPLETED';
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

    const fetchAgentState = async () => {
      const currentReqId = ++lastRequestId;

      try {
        const response = await fetch(apiUrl(`/api/runs/${runId}/layer2/agents`), {
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data: Layer2State = await response.json();

        if (isMounted && currentReqId === lastRequestId) {
          setState(data);
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
          setError(err instanceof Error ? err : new Error('Unknown error fetching Layer 2 agent state'));
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
            if (isMounted) fetchAgentState();
          }, backoffDelay);
        }
      }
    };

    fetchAgentState();
    intervalId = setInterval(fetchAgentState, POLL_INTERVAL_MS);

    return () => {
      isMounted = false;
      controller.abort();
      stopPolling();
    };
  }, [runId]);

  return { state, error, isLoading };
}
