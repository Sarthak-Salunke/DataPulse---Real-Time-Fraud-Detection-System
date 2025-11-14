// src/hooks/useApiData.ts
// Custom hooks for fetching data from FastAPI backend

import { useState, useEffect, useCallback, useRef } from 'react';

// ============================================================================
// Types
// ============================================================================

export interface DashboardMetrics {
  totalTransactions: number;
  fraudDetected: number;
  fraudRate: number;
  accuracy: number;
}

export interface FraudAlert {
  timestamp: string;
  ccNum: string;
  amount: number;
  merchant: string;
  confidence: number;
  transNum: string;
  category?: string;
  distance?: number;
}

export interface Transaction {
  id: string;
  time: string;
  customer: string;
  merchant: string;
  category: string;
  amount: number;
  distance: number;
  status: 'Fraud' | 'Normal';
  confidence?: number;
}

// ============================================================================
// Configuration
// ============================================================================

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const WS_URL = import.meta.env.VITE_WEBSOCKET_URL || 'ws://localhost:8000/ws';

// ============================================================================
// Hook: Dashboard Metrics
// ============================================================================

export function useDashboardMetrics(refreshInterval: number = 5000) {
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchMetrics = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/dashboard/metrics`);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      setMetrics(data);
      setError(null);
    } catch (err) {
      console.error('Error fetching dashboard metrics:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch metrics');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    // Initial fetch
    fetchMetrics();

    // Set up polling
    const interval = setInterval(fetchMetrics, refreshInterval);

    return () => clearInterval(interval);
  }, [fetchMetrics, refreshInterval]);

  return { metrics, loading, error, refresh: fetchMetrics };
}

// ============================================================================
// Hook: Recent Fraud Alerts
// ============================================================================

export function useRecentAlerts(limit: number = 10, refreshInterval: number = 3000) {
  const [alerts, setAlerts] = useState<FraudAlert[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchAlerts = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/fraud/alerts?limit=${limit}`);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      setAlerts(data);
      setError(null);
    } catch (err) {
      console.error('Error fetching fraud alerts:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch alerts');
    } finally {
      setLoading(false);
    }
  }, [limit]);

  useEffect(() => {
    // Initial fetch
    fetchAlerts();

    // Set up polling
    const interval = setInterval(fetchAlerts, refreshInterval);

    return () => clearInterval(interval);
  }, [fetchAlerts, refreshInterval]);

  return { alerts, loading, error, refresh: fetchAlerts };
}

// ============================================================================
// Hook: All Transactions
// ============================================================================

export function useTransactions(limit: number = 50, refreshInterval: number = 5000) {
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchTransactions = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/transactions?limit=${limit}`);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      setTransactions(data);
      setError(null);
    } catch (err) {
      console.error('Error fetching transactions:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch transactions');
    } finally {
      setLoading(false);
    }
  }, [limit]);

  useEffect(() => {
    // Initial fetch
    fetchTransactions();

    // Set up polling
    const interval = setInterval(fetchTransactions, refreshInterval);

    return () => clearInterval(interval);
  }, [fetchTransactions, refreshInterval]);

  return { transactions, loading, error, refresh: fetchTransactions };
}

// ============================================================================
// Hook: WebSocket Connection for Real-time Updates
// ============================================================================

interface WebSocketMessage {
  type: string;
  data?: any;
  message?: string;
  timestamp?: string;
}

export function useWebSocket(onMessage?: (message: WebSocketMessage) => void) {
  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;

  const connect = useCallback(() => {
    try {
      console.log('🔌 Connecting to WebSocket:', WS_URL);
      
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('✅ WebSocket connected');
        setConnected(true);
        reconnectAttempts.current = 0;
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          console.log('📨 WebSocket message received:', message);
          
          setLastMessage(message);
          
          if (onMessage) {
            onMessage(message);
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };

      ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
      };

      ws.onclose = () => {
        console.log('🔌 WebSocket disconnected');
        setConnected(false);
        wsRef.current = null;

        // Attempt reconnection
        if (reconnectAttempts.current < maxReconnectAttempts) {
          reconnectAttempts.current++;
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
          
          console.log(`⏱️  Reconnecting in ${delay / 1000}s... (Attempt ${reconnectAttempts.current}/${maxReconnectAttempts})`);
          
          reconnectTimeoutRef.current = setTimeout(connect, delay);
        } else {
          console.error('❌ Max reconnection attempts reached');
        }
      };
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
    }
  }, [onMessage]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    setConnected(false);
  }, []);

  const sendMessage = useCallback((data: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    } else {
      console.warn('WebSocket is not connected');
    }
  }, []);

  useEffect(() => {
    connect();

    // Cleanup on unmount
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    connected,
    lastMessage,
    sendMessage,
    reconnect: connect,
    disconnect,
  };
}

// ============================================================================
// Hook: Health Check
// ============================================================================

export function useHealthCheck(checkInterval: number = 30000) {
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [lastCheck, setLastCheck] = useState<string | null>(null);

  const checkHealth = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/health`);
      
      if (!response.ok) {
        throw new Error('Health check failed');
      }
      
      const data = await response.json();
      setHealthy(data.status === 'healthy');
      setLastCheck(data.time);
    } catch (err) {
      console.error('Health check failed:', err);
      setHealthy(false);
    }
  }, []);

  useEffect(() => {
    // Initial check
    checkHealth();

    // Set up periodic checks
    const interval = setInterval(checkHealth, checkInterval);

    return () => clearInterval(interval);
  }, [checkHealth, checkInterval]);

  return { healthy, lastCheck, check: checkHealth };
}