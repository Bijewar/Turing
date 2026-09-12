import { useEffect, useState } from 'react';
import { CausalGraph } from '../types/graph';
import { apiUrl } from '@/lib/api';

/** How often (ms) to poll for live causal graph topology updates. */
const POLL_INTERVAL_MS = 2000;
const MAX_BACKOFF_MS = 30000;

function computeGraphHash(g?: CausalGraph | null): string {
  if (!g) return '';
  const nodeCount = g.nodes?.length || 0;
  const edgeCount = g.edges?.length || 0;
  const nodeIds = (g.nodes || []).map((n) => n.id).join(',');
  const edgeTuples = (g.edges || []).map((e) => `${e.source}->${e.target}:${e.weight ?? 1}`).join(';');
  return `${nodeCount}|${edgeCount}|${nodeIds}|${edgeTuples}`;
}

export function useGraphAnimation(runId: string) {
  const [graph, setGraph] = useState<CausalGraph | null>(null);
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

    const fetchGraph = async () => {
      const currentReqId = ++lastRequestId;

      try {
        const response = await fetch(apiUrl(`/api/runs/${runId}/layer1/graph`), {
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data: CausalGraph = await response.json();

        if (isMounted && currentReqId === lastRequestId) {
          setGraph((prev) => {
            if (!prev) return data;
            // Structural hash comparison to prevent circular reference crashes and unneeded physics layout resets
            if (computeGraphHash(prev) === computeGraphHash(data)) {
              return prev;
            }
            return data;
          });
          setError(null);
          setIsLoading(false);
          consecutiveErrors = 0;

          if (data.is_fitted) {
            stopPolling();
          }
        }
      } catch (err: unknown) {
        if (!isMounted) return;
        if (err instanceof DOMException && err.name === 'AbortError') return;

        if (currentReqId === lastRequestId) {
          setError(err instanceof Error ? err : new Error('Unknown error fetching Layer 1 Causal Graph'));
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
            if (isMounted) fetchGraph();
          }, backoffDelay);
        }
      }
    };

    fetchGraph();
    intervalId = setInterval(fetchGraph, POLL_INTERVAL_MS);

    return () => {
      isMounted = false;
      controller.abort();
      stopPolling();
    };
  }, [runId]);

  return { graph, error, isLoading };
}

