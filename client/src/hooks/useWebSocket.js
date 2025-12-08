import { useState, useEffect, useCallback, useRef } from 'react';

/**
 * Hook for WebSocket connection to receive real-time job progress updates.
 *
 * @param {string} jobId - Job ID to connect to
 * @returns {object} - { isConnected, progress, logs, reconnect }
 */
export function useWebSocket(jobId) {
  const [isConnected, setIsConnected] = useState(false);
  const [progress, setProgress] = useState(null);
  const [logs, setLogs] = useState([]);
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const isTerminalStateRef = useRef(false);
  const reconnectAttemptRef = useRef(0);
  const MAX_RECONNECT_ATTEMPTS = 10;

  const connect = useCallback(() => {
    if (!jobId) return;

    // Close existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/${jobId}`;

    console.log('Connecting to WebSocket:', wsUrl);

    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setIsConnected(true);
      console.log('WebSocket connected for job:', jobId);
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        console.log('WebSocket message:', message);

        // Update progress state
        if (message.data) {
          setProgress(message.data);

          // Check for terminal states - stop reconnecting if job is done
          const status = message.data.status;
          if (status === 'failed' || status === 'completed' || status === 'cancelled') {
            console.log(`Job reached terminal state: ${status}`);
            isTerminalStateRef.current = true;
          }
        }

        // Reset reconnect attempts on successful message
        reconnectAttemptRef.current = 0;

        // Add to logs
        setLogs((prev) => [
          ...prev,
          {
            timestamp: message.timestamp || new Date().toISOString(),
            event: message.event,
            data: message.data,
          },
        ]);
      } catch (err) {
        console.error('Failed to parse WebSocket message:', err);
      }
    };

    ws.onclose = (event) => {
      setIsConnected(false);
      console.log('WebSocket disconnected:', event.code, event.reason);

      // Don't reconnect if:
      // 1. Normal close (1000)
      // 2. Job reached terminal state (failed/completed/cancelled)
      // 3. Max reconnection attempts exceeded
      if (event.code === 1000) {
        console.log('Normal WebSocket close, not reconnecting');
        return;
      }

      if (isTerminalStateRef.current) {
        console.log('Job in terminal state, not reconnecting');
        return;
      }

      if (reconnectAttemptRef.current >= MAX_RECONNECT_ATTEMPTS) {
        console.log(`Max reconnection attempts (${MAX_RECONNECT_ATTEMPTS}) exceeded`);
        return;
      }

      // Exponential backoff: 1s, 2s, 4s, 8s, 16s, max 30s
      const backoffMs = Math.min(1000 * Math.pow(2, reconnectAttemptRef.current), 30000);
      reconnectAttemptRef.current += 1;

      console.log(`Reconnecting in ${backoffMs}ms (attempt ${reconnectAttemptRef.current}/${MAX_RECONNECT_ATTEMPTS})...`);
      reconnectTimeoutRef.current = setTimeout(() => {
        connect();
      }, backoffMs);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    wsRef.current = ws;
  }, [jobId]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    if (wsRef.current) {
      wsRef.current.close(1000, 'User disconnected');
      wsRef.current = null;
    }
  }, []);

  const clearLogs = useCallback(() => {
    setLogs([]);
    setProgress(null);
    isTerminalStateRef.current = false;
    reconnectAttemptRef.current = 0;
  }, []);

  useEffect(() => {
    if (jobId) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [jobId, connect, disconnect]);

  return {
    isConnected,
    progress,
    logs,
    reconnect: connect,
    disconnect,
    clearLogs,
  };
}

export default useWebSocket;
