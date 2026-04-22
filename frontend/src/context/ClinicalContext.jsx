import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

const ClinicalContext = createContext(null);

const DEFAULT_THERAPY_MODE = 'Supportive_Stabilization';
const DEFAULT_CONNECTION_STATUS = 'disconnected';

const toFiniteNumber = (value, fallback = 0) => {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
};

const asObject = (value) => (value && typeof value === 'object' ? value : {});

export const ClinicalProvider = ({ children }) => {
  const [activeRiskScore, setActiveRiskScore] = useState(0);
  const [isCrisisMode, setIsCrisisMode] = useState(false);
  const [currentTherapyMode, setCurrentTherapyMode] = useState(DEFAULT_THERAPY_MODE);
  const [connectionStatus, setConnectionStatus] = useState(DEFAULT_CONNECTION_STATUS);

  const wsRef = useRef(null);
  const sseRef = useRef(null);
  const reconnectTimerRef = useRef(null);

  const applyClinicalPayload = useCallback((payload) => {
    const event = asObject(payload);
    const eventType = String(event.type || event.event || '').trim().toUpperCase();
    const clinical = asObject(event.clinical);

    const nextRisk =
      event.activeRiskScore ??
      event.active_risk_score ??
      event.riskScore ??
      event.risk_score ??
      clinical.activeRiskScore ??
      clinical.riskScore ??
      clinical.risk_score;

    if (nextRisk !== undefined) {
      setActiveRiskScore(toFiniteNumber(nextRisk, 0));
    }

    const nextTherapyMode =
      event.currentTherapyMode ??
      event.current_therapy_mode ??
      event.framework ??
      clinical.currentTherapyMode ??
      clinical.current_therapy_mode ??
      clinical.framework;

    if (nextTherapyMode) {
      setCurrentTherapyMode(String(nextTherapyMode));
    }

    const crisisDetected =
      eventType === 'CRISIS_OVERRIDE' ||
      eventType === 'SAFETY_OVERRIDE' ||
      (eventType === 'SAFETY_MODE' && event.enabled !== false) ||
      event.isCrisisMode === true ||
      event.is_crisis_mode === true ||
      event.crisis === true ||
      event.safety_alert === true ||
      clinical.isCrisisMode === true ||
      clinical.requires_safety_review === true;

    if (crisisDetected) {
      setIsCrisisMode(true);
    }
  }, []);

  const clearReconnectTimer = () => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  };

  useEffect(() => {
    const wsUrl = String(import.meta.env.VITE_CLINICAL_WS_URL || '').trim();
    const sseUrl = String(import.meta.env.VITE_CLINICAL_SSE_URL || '').trim();
    const reconnectMs = Math.max(800, toFiniteNumber(import.meta.env.VITE_CLINICAL_RECONNECT_MS, 2500));

    let unmounted = false;

    const scheduleReconnect = (connectFn) => {
      clearReconnectTimer();
      reconnectTimerRef.current = setTimeout(() => {
        if (!unmounted) {
          connectFn();
        }
      }, reconnectMs);
    };

    const connectWebSocket = () => {
      if (!wsUrl || unmounted) {
        return;
      }

      setConnectionStatus('connecting');
      const socket = new WebSocket(wsUrl);
      wsRef.current = socket;

      socket.onopen = () => {
        if (!unmounted) {
          setConnectionStatus('connected');
        }
      };

      socket.onmessage = (messageEvent) => {
        if (unmounted) {
          return;
        }
        try {
          const parsed = JSON.parse(String(messageEvent.data || '{}'));
          applyClinicalPayload(parsed);
        } catch {
          // Ignore malformed messages from experimental feeds.
        }
      };

      socket.onerror = () => {
        if (!unmounted) {
          setConnectionStatus('disconnected');
        }
      };

      socket.onclose = () => {
        if (unmounted) {
          return;
        }
        setConnectionStatus('disconnected');
        scheduleReconnect(connectWebSocket);
      };
    };

    const connectSse = () => {
      if (!sseUrl || unmounted) {
        return;
      }

      setConnectionStatus('connecting');
      const source = new EventSource(sseUrl);
      sseRef.current = source;

      source.onopen = () => {
        if (!unmounted) {
          setConnectionStatus('connected');
        }
      };

      source.onmessage = (messageEvent) => {
        if (unmounted) {
          return;
        }
        try {
          const parsed = JSON.parse(String(messageEvent.data || '{}'));
          applyClinicalPayload(parsed);
        } catch {
          // Ignore malformed messages from experimental feeds.
        }
      };

      source.onerror = () => {
        if (unmounted) {
          return;
        }

        setConnectionStatus('disconnected');
        source.close();
        scheduleReconnect(connectSse);
      };
    };

    if (wsUrl) {
      connectWebSocket();
    } else if (sseUrl) {
      connectSse();
    } else {
      setConnectionStatus(DEFAULT_CONNECTION_STATUS);
    }

    return () => {
      unmounted = true;
      clearReconnectTimer();

      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }

      if (sseRef.current) {
        sseRef.current.close();
        sseRef.current = null;
      }
    };
  }, [applyClinicalPayload]);

  const value = useMemo(
    () => ({
      activeRiskScore,
      isCrisisMode,
      currentTherapyMode,
      connectionStatus,
      ingestBackendEvent: applyClinicalPayload,
      setCrisisMode: () => setIsCrisisMode(true),
      clearCrisisMode: () => setIsCrisisMode(false),
      setActiveRiskScore,
      setCurrentTherapyMode,
    }),
    [
      activeRiskScore,
      applyClinicalPayload,
      connectionStatus,
      currentTherapyMode,
      isCrisisMode,
    ]
  );

  return <ClinicalContext.Provider value={value}>{children}</ClinicalContext.Provider>;
};

export const useClinical = () => {
  const context = useContext(ClinicalContext);
  if (!context) {
    throw new Error('useClinical must be used within ClinicalProvider');
  }
  return context;
};
