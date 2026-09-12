import { useState, useEffect, useRef, useCallback } from 'react';
import { Layer3State } from '../types/layer3';
import { apiUrl } from '@/lib/api';

const MAX_RETRIES = 5;
const BASE_RETRY_DELAY_MS = 1000; // doubles each attempt (exponential backoff)

export function useSearchStream(runId: string) {
  const [state, setState] = useState<Layer3State | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const retryCount = useRef(0);
  const retryTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const connectFnRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (!runId) return;

    // Reset retry state when runId changes (SWARM-7 BUG #22)
    retryCount.current = 0;
    if (retryTimer.current) {
      clearTimeout(retryTimer.current);
      retryTimer.current = null;
    }

    let es: EventSource | null = null;
    let cancelled = false;

    const connect = () => {
      if (cancelled) return;

      // Establish Server-Sent Events (SSE) connection for live cross-domain search updates
      es = new EventSource(apiUrl(`/api/runs/${runId}/layer3/search/stream`));

      es.onopen = () => {
        if (!cancelled) {
          setIsConnected(true);
          setError(null);
          retryCount.current = 0; // reset backoff on successful connection
        }
      };

      es.onmessage = (event) => {
        if (cancelled) return;
        try {
          const data: Layer3State = JSON.parse(event.data);
          if (!cancelled) {
            setState(data);
            setError(null);
          }

          // If the backend radar natively locks onto a structural isomorphism,
          // close the search pipeline gracefully.
          if (data.isLocked) {
            es?.close();
            if (!cancelled) setIsConnected(false);
          }
        } catch (err) {
          if (!cancelled) {
            setError(
              err instanceof Error
                ? err
                : new Error('Failed to parse Layer 3 search stream chunk')
            );
          }
        }
      };

      es.onerror = () => {
        es?.close();
        if (cancelled) return;

        setIsConnected(false);

        if (retryCount.current >= MAX_RETRIES) {
          if (!cancelled) {
            setError(
              new Error('Connection to Layer 3 search stream lost after max retries')
            );
          }
          return;
        }

        // Exponential backoff reconnect
        const delay = BASE_RETRY_DELAY_MS * Math.pow(2, retryCount.current);
        retryCount.current += 1;
        if (retryTimer.current) {
          clearTimeout(retryTimer.current);
          retryTimer.current = null;
        }
        retryTimer.current = setTimeout(() => {
          if (!cancelled) connect();
        }, delay);
      };
    };

    connectFnRef.current = connect;
    connect();

    return () => {
      cancelled = true;
      es?.close();
      if (retryTimer.current) {
        clearTimeout(retryTimer.current);
        retryTimer.current = null;
      }
    };
  }, [runId]);

  const reconnect = useCallback(() => {
    if (retryTimer.current) {
      clearTimeout(retryTimer.current);
      retryTimer.current = null;
    }
    retryCount.current = 0;
    setError(null);
    connectFnRef.current?.();
  }, []);

  return {
    state,
    error,
    isConnected,
    isLoading: !state && !error,
    reconnect,
  };
}
