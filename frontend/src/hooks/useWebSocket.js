import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * Hook for WebSocket connection with auto-reconnect
 */
export function useWebSocket(url, { onMessage, onOpen, onClose, autoConnect = true } = {}) {
  const ws = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const reconnectTimer = useRef(null);

  const connect = useCallback(() => {
    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}${url}`;
      ws.current = new WebSocket(wsUrl);

      ws.current.onopen = () => {
        setIsConnected(true);
        onOpen?.();
      };

      ws.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          onMessage?.(data);
        } catch {
          onMessage?.(event.data);
        }
      };

      ws.current.onclose = () => {
        setIsConnected(false);
        onClose?.();
        // Auto-reconnect after 3 seconds
        reconnectTimer.current = setTimeout(connect, 3000);
      };

      ws.current.onerror = () => {
        ws.current?.close();
      };
    } catch (err) {
      console.error('WebSocket connection error:', err);
    }
  }, [url, onMessage, onOpen, onClose]);

  const disconnect = useCallback(() => {
    clearTimeout(reconnectTimer.current);
    ws.current?.close();
  }, []);

  const send = useCallback((data) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(typeof data === 'string' ? data : JSON.stringify(data));
    }
  }, []);

  useEffect(() => {
    if (autoConnect) connect();
    return () => disconnect();
  }, [autoConnect, connect, disconnect]);

  return { isConnected, send, connect, disconnect };
}

/**
 * Hook for authentication state
 */
export function useAuth() {
  const [user, setUser] = useState(() => {
    const saved = localStorage.getItem('yaqiz_user');
    return saved ? JSON.parse(saved) : null;
  });

  const [token, setToken] = useState(() => localStorage.getItem('yaqiz_token'));

  const login = useCallback((tokenStr, userData) => {
    localStorage.setItem('yaqiz_token', tokenStr);
    localStorage.setItem('yaqiz_user', JSON.stringify(userData));
    setToken(tokenStr);
    setUser(userData);
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem('yaqiz_token');
    localStorage.removeItem('yaqiz_user');
    setToken(null);
    setUser(null);
  }, []);

  const isAuthenticated = !!token;

  return { user, token, isAuthenticated, login, logout };
}
