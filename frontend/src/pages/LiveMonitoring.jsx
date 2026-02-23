import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Camera, CameraOff, Settings, Activity, Users, HardHat,
  ShieldCheck, ShieldAlert, Wifi, WifiOff, AlertTriangle,
  Gauge, UserCheck, UserX, ClipboardList
} from 'lucide-react';
import { detectionAPI } from '../services/api';

export default function LiveMonitoring() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [confidence, setConfidence] = useState(0.5);
  const [wsData, setWsData] = useState(null);
  const [tracking, setTracking] = useState(null);
  const [selectedWorker, setSelectedWorker] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [connected, setConnected] = useState(false);
  const [streamError, setStreamError] = useState(null);
  const [usingMjpeg, setUsingMjpeg] = useState(false);
  const imgRef = useRef(null);
  const wsRef = useRef(null);
  const canvasRef = useRef(null);
  const mjpegFallbackRef = useRef(false);
  const isStreamingRef = useRef(false);

  const startMjpegFallback = useCallback(() => {
    if (imgRef.current) {
      setUsingMjpeg(true);
      setStreamError(null);
      imgRef.current.src = detectionAPI.getLiveFeedUrl(confidence);
      // Check if image loads after a short delay
      const checkTimeout = setTimeout(() => {
        if (imgRef.current && imgRef.current.naturalWidth === 0) {
          setStreamError('Camera feed not available. Please check that your webcam is connected.');
          setIsStreaming(false);
          setUsingMjpeg(false);
        }
      }, 5000);
      imgRef.current._checkTimeout = checkTimeout;
    }
  }, [confidence]);

  const startStream = useCallback(() => {
    setIsStreaming(true);
    isStreamingRef.current = true;
    setStreamError(null);
    setUsingMjpeg(false);
    mjpegFallbackRef.current = false;
    
    // Connect via WebSocket for data
    try {
      const backendHost = window.location.hostname + ':8000';
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      
      // Try direct backend connection first, then proxy
      let wsUrl;
      if (window.location.port === '5173' || window.location.port === '3000') {
        // Dev mode: use Vite proxy
        wsUrl = `${protocol}//${window.location.host}/ws/live?confidence=${confidence}`;
      } else {
        wsUrl = `${protocol}//${window.location.host}/ws/live?confidence=${confidence}`;
      }
      
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        setConnected(true);
        setStreamError(null);
      };
      
      wsRef.current.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'frame') {
            // Display frame
            if (imgRef.current) {
              imgRef.current.src = `data:image/jpeg;base64,${msg.frame}`;
            }
            setWsData(msg.data);
            if (msg.tracking) setTracking(msg.tracking);

            // Track violations as alerts
            if (msg.data.violations_count > 0) {
              msg.data.detections
                .filter(d => d.is_violation)
                .forEach(d => {
                  setAlerts(prev => [{
                    id: Date.now() + Math.random(),
                    type: d.class_name,
                    confidence: d.confidence,
                    time: new Date().toLocaleTimeString(),
                  }, ...prev].slice(0, 20));
                });
            }
          } else if (msg.type === 'error') {
            console.error('Camera error from server:', msg.message);
            setStreamError(msg.message);
            // Try MJPEG fallback
            if (!mjpegFallbackRef.current) {
              mjpegFallbackRef.current = true;
              startMjpegFallback();
            }
          }
        } catch(e) {
          console.error('Failed to parse WS message:', e);
        }
      };

      wsRef.current.onclose = () => {
        setConnected(false);
        // If we haven't received any frames, try MJPEG fallback
        if (!mjpegFallbackRef.current && isStreamingRef.current) {
          mjpegFallbackRef.current = true;
          startMjpegFallback();
        } else if (!mjpegFallbackRef.current) {
          setIsStreaming(false);
          isStreamingRef.current = false;
        }
      };

      wsRef.current.onerror = (err) => {
        console.error('WebSocket error:', err);
        setConnected(false);
        // Fallback to MJPEG stream
        if (!mjpegFallbackRef.current) {
          mjpegFallbackRef.current = true;
          startMjpegFallback();
        }
      };
    } catch(e) {
      console.error('WebSocket creation error:', e);
      // Fallback to MJPEG stream
      if (!mjpegFallbackRef.current) {
        mjpegFallbackRef.current = true;
        startMjpegFallback();
      }
    }
  }, [confidence, startMjpegFallback]);

  const stopStream = useCallback(() => {
    setIsStreaming(false);
    isStreamingRef.current = false;
    setConnected(false);
    setUsingMjpeg(false);
    setStreamError(null);
    mjpegFallbackRef.current = false;
    wsRef.current?.close();
    if (imgRef.current) {
      if (imgRef.current._checkTimeout) clearTimeout(imgRef.current._checkTimeout);
      imgRef.current.src = '';
    }
  }, []);

  useEffect(() => {
    return () => {
      wsRef.current?.close();
    };
  }, []);

  const updateConfidence = (val) => {
    setConfidence(val);
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'set_confidence', value: val }));
    }
  };

  return (
    <div className="p-8 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Live Monitoring</h1>
          <p className="text-yaqiz-muted">Real-time PPE detection from camera feed</p>
        </div>
        <div className="flex items-center gap-3">
          {/* Connection Status */}
          <div className={`flex items-center gap-2 px-4 py-2 rounded-xl border ${
            connected 
              ? 'bg-yaqiz-success/10 border-yaqiz-success/30 text-yaqiz-success' 
              : usingMjpeg
                ? 'bg-yaqiz-warning/10 border-yaqiz-warning/30 text-yaqiz-warning'
                : 'bg-yaqiz-border border-yaqiz-border text-yaqiz-muted'
          }`}>
            {connected ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
            <span className="text-sm font-medium">
              {connected ? 'Connected' : usingMjpeg ? 'MJPEG Stream' : 'Disconnected'}
            </span>
          </div>

          {/* Start/Stop */}
          <button
            onClick={isStreaming ? stopStream : startStream}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-xl font-semibold transition-all duration-300 ${
              isStreaming
                ? 'bg-yaqiz-danger/20 border border-yaqiz-danger/30 text-yaqiz-danger hover:bg-yaqiz-danger/30'
                : 'btn-primary'
            }`}
          >
            {isStreaming ? (
              <><CameraOff className="w-5 h-5" /> Stop</>
            ) : (
              <><Camera className="w-5 h-5" /> Start Camera</>
            )}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Video Feed - 3 columns */}
        <div className="lg:col-span-3">
          <div className="glass-card overflow-hidden">
            {/* Live badge */}
            <div className="flex items-center justify-between p-4 border-b border-yaqiz-border">
              <div className="flex items-center gap-3">
                <Camera className="w-5 h-5 text-yaqiz-accent" />
                <span className="font-medium">Camera Feed</span>
              </div>
              {isStreaming && (
                <div className="flex items-center gap-2 px-3 py-1 bg-yaqiz-danger/20 border border-yaqiz-danger/30 rounded-full">
                  <div className="w-2 h-2 bg-yaqiz-danger rounded-full animate-pulse" />
                  <span className="text-xs text-yaqiz-danger font-bold">LIVE</span>
                </div>
              )}
            </div>

            {/* Video */}
            <div className="bg-black aspect-video flex items-center justify-center relative">
              {isStreaming ? (
                <>
                  <img ref={imgRef} alt="Live Detection Feed" className="w-full h-full object-contain" />
                  {streamError && !usingMjpeg && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black/80">
                      <div className="text-center p-6">
                        <AlertTriangle className="w-16 h-16 mx-auto mb-4 text-yaqiz-danger" />
                        <p className="text-lg text-yaqiz-danger font-semibold mb-2">Camera Error</p>
                        <p className="text-sm text-yaqiz-muted max-w-md">{streamError}</p>
                        <button onClick={stopStream} className="mt-4 px-4 py-2 bg-yaqiz-danger/20 border border-yaqiz-danger/30 text-yaqiz-danger rounded-lg hover:bg-yaqiz-danger/30 transition-all">
                          Close
                        </button>
                      </div>
                    </div>
                  )}
                  {usingMjpeg && (
                    <div className="absolute top-2 left-2 px-3 py-1 bg-yaqiz-warning/20 border border-yaqiz-warning/30 rounded-full">
                      <span className="text-xs text-yaqiz-warning font-medium">MJPEG Mode</span>
                    </div>
                  )}
                </>
              ) : (
                <div className="text-center text-yaqiz-muted">
                  <Camera className="w-20 h-20 mx-auto mb-4 opacity-20" />
                  <p className="text-lg">Camera feed paused</p>
                  <p className="text-sm mt-1 text-yaqiz-muted/50">Click "Start Camera" to begin detection</p>
                </div>
              )}
            </div>

            {/* Controls */}
            <div className="p-4 border-t border-yaqiz-border flex items-center gap-6">
              <div className="flex items-center gap-3 flex-1">
                <Gauge className="w-5 h-5 text-yaqiz-accent" />
                <span className="text-sm text-yaqiz-muted whitespace-nowrap">Confidence:</span>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={confidence}
                  onChange={(e) => updateConfidence(parseFloat(e.target.value))}
                  className="flex-1 accent-yaqiz-accent"
                />
                <span className="text-sm font-mono text-yaqiz-accent w-12 text-right">
                  {(confidence * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Side Panel - 1 column */}
        <div className="space-y-5">
          {/* Tracking Stats */}
          <div className="glass-card p-5">
            <h3 className="text-sm font-semibold text-yaqiz-muted uppercase tracking-wider mb-4">
              Worker Tracking
            </h3>
            <div className="grid grid-cols-2 gap-3 mb-4">
              <div className="bg-yaqiz-bg/50 rounded-xl p-3 text-center border border-yaqiz-accent/10">
                <Users className="w-5 h-5 text-yaqiz-accent mx-auto mb-1" />
                <p className="text-xl font-bold text-white">{tracking?.unique_workers ?? '-'}</p>
                <p className="text-[10px] text-yaqiz-muted">Unique Workers</p>
              </div>
              <div className="bg-yaqiz-bg/50 rounded-xl p-3 text-center border border-yaqiz-danger/10">
                <UserX className="w-5 h-5 text-yaqiz-danger mx-auto mb-1" />
                <p className="text-xl font-bold text-yaqiz-danger">{tracking?.unique_violators ?? '-'}</p>
                <p className="text-[10px] text-yaqiz-muted">Violators</p>
              </div>
              <div className="bg-yaqiz-bg/50 rounded-xl p-3 text-center border border-yaqiz-success/10">
                <UserCheck className="w-5 h-5 text-yaqiz-success mx-auto mb-1" />
                <p className="text-xl font-bold text-yaqiz-success">{tracking?.active_workers ?? '-'}</p>
                <p className="text-[10px] text-yaqiz-muted">Active Now</p>
              </div>
              <div className="bg-yaqiz-bg/50 rounded-xl p-3 text-center border border-yaqiz-warning/10">
                <AlertTriangle className="w-5 h-5 text-yaqiz-warning mx-auto mb-1" />
                <p className="text-xl font-bold text-yaqiz-warning">{tracking?.active_violators ?? '-'}</p>
                <p className="text-[10px] text-yaqiz-muted">Active Violators</p>
              </div>
            </div>
          </div>

          {/* Worker List */}
          <div className="glass-card p-5">
            <h3 className="text-sm font-semibold text-yaqiz-muted uppercase tracking-wider mb-4 flex items-center gap-2">
              <ClipboardList className="w-4 h-4" />
              Worker Log
            </h3>
            <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
              {tracking?.worker_records && Object.keys(tracking.worker_records).length > 0 ? (
                Object.values(tracking.worker_records)
                  .sort((a, b) => b.last_seen_frame - a.last_seen_frame)
                  .map((w) => (
                    <div
                      key={w.worker_id}
                      onClick={() => setSelectedWorker(selectedWorker === w.worker_id ? null : w.worker_id)}
                      className={`rounded-lg p-3 border cursor-pointer transition-all duration-200 ${
                        w.is_compliant
                          ? 'bg-yaqiz-success/5 border-yaqiz-success/20 hover:border-yaqiz-success/40'
                          : 'bg-yaqiz-danger/5 border-yaqiz-danger/20 hover:border-yaqiz-danger/40'
                      } ${selectedWorker === w.worker_id ? 'ring-1 ring-yaqiz-accent' : ''}`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm font-bold text-white">Worker #{w.worker_id}</span>
                        <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                          w.is_compliant
                            ? 'bg-yaqiz-success/20 text-yaqiz-success'
                            : 'bg-yaqiz-danger/20 text-yaqiz-danger'
                        }`}>
                          {w.is_compliant ? 'Compliant' : 'Violation'}
                        </span>
                      </div>
                      
                      {/* Current violations */}
                      {w.current_violations?.length > 0 && (
                        <div className="flex flex-wrap gap-1 mb-1.5">
                          {w.current_violations.map((v, i) => (
                            <span key={i} className="text-[9px] bg-yaqiz-danger/15 text-yaqiz-danger px-1.5 py-0.5 rounded">
                              {v}
                            </span>
                          ))}
                        </div>
                      )}

                      <div className="flex items-center gap-3 text-[10px] text-yaqiz-muted">
                        <span>{w.total_frames} frames</span>
                        <span>{w.duration_seconds}s</span>
                        <span className={w.violations?.total > 0 ? 'text-yaqiz-danger' : 'text-yaqiz-success'}>
                          {w.violations?.total || 0} violations
                        </span>
                      </div>

                      {/* Expanded detail */}
                      {selectedWorker === w.worker_id && (
                        <div className="mt-2 pt-2 border-t border-yaqiz-border/50 space-y-1.5 animate-fade-in">
                          <div className="grid grid-cols-3 gap-1 text-[10px]">
                            <div className="text-center p-1 rounded bg-yaqiz-bg/50">
                              <span className="text-yaqiz-muted">⛑️ Missing</span>
                              <p className="font-bold text-white">{w.violations?.no_hardhat || 0}</p>
                            </div>
                            <div className="text-center p-1 rounded bg-yaqiz-bg/50">
                              <span className="text-yaqiz-muted">🦺 Missing</span>
                              <p className="font-bold text-white">{w.violations?.no_vest || 0}</p>
                            </div>
                            <div className="text-center p-1 rounded bg-yaqiz-bg/50">
                              <span className="text-yaqiz-muted">😷 Missing</span>
                              <p className="font-bold text-white">{w.violations?.no_mask || 0}</p>
                            </div>
                          </div>
                          {w.log?.length > 0 && (
                            <div className="space-y-1 max-h-24 overflow-y-auto">
                              {w.log.slice(-5).reverse().map((entry, i) => (
                                <div key={i} className="text-[9px] text-yaqiz-muted flex items-center gap-1.5 py-0.5">
                                  <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
                                    entry.event === 'violation' ? 'bg-yaqiz-danger' : 'bg-yaqiz-success'
                                  }`} />
                                  <span className="truncate">
                                    F{entry.frame}: {entry.details || entry.event}
                                  </span>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ))
              ) : (
                <div className="text-center py-6 text-yaqiz-muted">
                  <Users className="w-8 h-8 mx-auto mb-2 opacity-30" />
                  <p className="text-xs">No workers tracked yet</p>
                  <p className="text-[10px] mt-1 opacity-50">Start the camera to begin tracking</p>
                </div>
              )}
            </div>
          </div>

          {/* Live Stats */}
          <div className="glass-card p-5">
            <h3 className="text-sm font-semibold text-yaqiz-muted uppercase tracking-wider mb-4">
              Frame Status
            </h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-yaqiz-accent" />
                  <span className="text-sm text-yaqiz-muted">Detections</span>
                </div>
                <span className="text-lg font-bold text-white">{wsData?.total_detections ?? '-'}</span>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-yaqiz-danger" />
                  <span className="text-sm text-yaqiz-muted">Violations</span>
                </div>
                <span className={`text-lg font-bold ${
                  (wsData?.violations_count || 0) > 0 ? 'text-yaqiz-danger' : 'text-yaqiz-success'
                }`}>{wsData?.violations_count ?? '-'}</span>
              </div>

              <hr className="border-yaqiz-border" />

              {/* Compliance meters */}
              {[
                { label: 'Helmet', value: wsData?.helmet_compliance, icon: '⛑️' },
                { label: 'Vest', value: wsData?.vest_compliance, icon: '🦺' },
                { label: 'Mask', value: wsData?.mask_compliance, icon: '😷' },
              ].map(({ label, value, icon }) => (
                <div key={label}>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs text-yaqiz-muted">{icon} {label}</span>
                    <span className={`text-xs font-bold ${
                      (value ?? 100) >= 80 ? 'text-yaqiz-success' :
                      (value ?? 100) >= 50 ? 'text-yaqiz-warning' : 'text-yaqiz-danger'
                    }`}>{value?.toFixed(0) ?? '--'}%</span>
                  </div>
                  <div className="w-full bg-yaqiz-bg rounded-full h-1.5">
                    <div className={`h-full rounded-full transition-all duration-500 ${
                      (value ?? 100) >= 80 ? 'bg-yaqiz-success' :
                      (value ?? 100) >= 50 ? 'bg-yaqiz-warning' : 'bg-yaqiz-danger'
                    }`} style={{ width: `${Math.min(value ?? 100, 100)}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Live Alerts */}
          <div className="glass-card p-5">
            <h3 className="text-sm font-semibold text-yaqiz-muted uppercase tracking-wider mb-4">
              Live Alerts
            </h3>
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {alerts.length > 0 ? alerts.map((alert) => (
                <div key={alert.id} className="flex items-center gap-2 py-2 px-3 rounded-lg bg-yaqiz-danger/10 border border-yaqiz-danger/20 animate-slide-up">
                  <AlertTriangle className="w-3.5 h-3.5 text-yaqiz-danger flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-xs text-white truncate">{alert.type}</p>
                    <p className="text-[10px] text-yaqiz-muted">{alert.time}</p>
                  </div>
                  <span className="text-[10px] text-yaqiz-danger font-mono">
                    {(alert.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              )) : (
                <div className="text-center py-6 text-yaqiz-muted">
                  <ShieldCheck className="w-8 h-8 mx-auto mb-2 opacity-30 text-yaqiz-success" />
                  <p className="text-xs">No alerts</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
