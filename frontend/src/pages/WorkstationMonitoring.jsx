import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Monitor, MonitorOff, Wifi, WifiOff, Activity, Eye, EyeOff,
  AlertTriangle, Brain, Coffee, Clock, User, UserX,
  Gauge, Hand, Keyboard, MousePointer,
  CheckCircle, XCircle, Minus
} from 'lucide-react';

// ── Gauge Component ──────────────────────────────────────
function CircularGauge({ value, max = 100, label, color, size = 120 }) {
  const pct = Math.min(value / max, 1);
  const radius = (size - 16) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - pct);
  const gradientId = `gauge-${label.replace(/\s/g, '')}`;

  const colorMap = {
    accent: { start: '#00d4ff', end: '#3b82f6' },
    danger: { start: '#ef4444', end: '#f97316' },
    success: { start: '#10b981', end: '#34d399' },
    warning: { start: '#f59e0b', end: '#fbbf24' },
  };
  const c = colorMap[color] || colorMap.accent;

  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={size} className="-rotate-90">
        <defs>
          <linearGradient id={gradientId}>
            <stop offset="0%" stopColor={c.start} />
            <stop offset="100%" stopColor={c.end} />
          </linearGradient>
        </defs>
        <circle cx={size / 2} cy={size / 2} r={radius}
          fill="none" stroke="#1e1e2e" strokeWidth="8" />
        <circle cx={size / 2} cy={size / 2} r={radius}
          fill="none" stroke={`url(#${gradientId})`} strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="transition-all duration-700 ease-out" />
      </svg>
      <div className="absolute flex flex-col items-center justify-center"
        style={{ width: size, height: size }}>
        <span className="text-2xl font-bold text-white">{Math.round(value)}</span>
        <span className="text-[10px] text-yaqiz-muted uppercase tracking-wider">{label}</span>
      </div>
    </div>
  );
}

// ── Mini Stat Card ───────────────────────────────────────
function MiniStat({ icon: Icon, label, value, color = 'text-yaqiz-muted' }) {
  return (
    <div className="bg-yaqiz-bg/50 rounded-xl p-3 border border-yaqiz-border/50">
      <div className="flex items-center gap-2 mb-1">
        <Icon className={`w-4 h-4 ${color}`} />
        <span className="text-[10px] text-yaqiz-muted uppercase tracking-wider">{label}</span>
      </div>
      <p className={`text-lg font-bold ${color === 'text-yaqiz-muted' ? 'text-white' : color}`}>
        {value}
      </p>
    </div>
  );
}

// ── Head Pose Indicator ──────────────────────────────────
function HeadPoseIndicator({ status }) {
  const poses = {
    forward: { label: 'Forward', color: 'text-yaqiz-success', bg: 'bg-yaqiz-success/15' },
    left: { label: 'Left', color: 'text-yaqiz-warning', bg: 'bg-yaqiz-warning/15' },
    right: { label: 'Right', color: 'text-yaqiz-warning', bg: 'bg-yaqiz-warning/15' },
    up: { label: 'Up', color: 'text-yaqiz-warning', bg: 'bg-yaqiz-warning/15' },
    down: { label: 'Down', color: 'text-yaqiz-danger', bg: 'bg-yaqiz-danger/15' },
    unknown: { label: 'N/A', color: 'text-yaqiz-muted', bg: 'bg-yaqiz-border' },
  };
  const p = poses[status] || poses.unknown;

  return (
    <div className={`${p.bg} rounded-lg px-3 py-1.5 flex items-center gap-2`}>
      <div className={`w-2 h-2 rounded-full ${
        status === 'forward' ? 'bg-yaqiz-success' :
        status === 'unknown' ? 'bg-yaqiz-muted' : 'bg-yaqiz-warning'
      }`} />
      <span className={`text-xs font-semibold ${p.color}`}>{p.label}</span>
    </div>
  );
}

// ── Live Alert Badge ─────────────────────────────────────
function AlertBadge({ alert }) {
  const severityColors = {
    critical: 'bg-yaqiz-danger/10 border-yaqiz-danger/20 text-yaqiz-danger',
    warning: 'bg-yaqiz-warning/10 border-yaqiz-warning/20 text-yaqiz-warning',
    info: 'bg-yaqiz-accent/10 border-yaqiz-accent/20 text-yaqiz-accent',
  };
  const cls = severityColors[alert.severity] || severityColors.info;

  return (
    <div className={`${cls} border rounded-lg px-3 py-2 animate-slide-up`}>
      <div className="flex items-center gap-2">
        <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0" />
        <span className="text-xs font-medium truncate">{alert.message}</span>
      </div>
      <p className="text-[10px] opacity-60 mt-0.5 ml-5.5">
        {new Date(alert.timestamp || Date.now()).toLocaleTimeString()}
      </p>
    </div>
  );
}

// ══════════════════════════════════════════════════════════
// MAIN PAGE COMPONENT
// ══════════════════════════════════════════════════════════

export default function WorkstationMonitoring() {
  const [isActive, setIsActive] = useState(false);
  const [connected, setConnected] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [frameNum, setFrameNum] = useState(0);
  const [cameraError, setCameraError] = useState('');

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);

  // ── Camera controls ────────────────────────────────────

  const startCamera = useCallback(async () => {
    setCameraError('');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch (err) {
      setCameraError('Camera access denied. Please allow camera permissions.');
      console.error('Camera error:', err);
      return false;
    }
    return true;
  }, []);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  // ── WebSocket ──────────────────────────────────────────

  const connectWs = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const token = localStorage.getItem('yaqiz_token');
    const tokenParam = token ? `?token=${encodeURIComponent(token)}` : '';
    const url = `${protocol}//${window.location.host}/ws/workstation${tokenParam}`;
    const ws = new WebSocket(url);

    ws.onopen = () => {
      setConnected(true);
      // Optionally start activity tracker
      ws.send(JSON.stringify({ type: 'start_activity_tracker' }));
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'metrics') {
          setMetrics(msg.data);
          setFrameNum(msg.frame_number || 0);

          // Accumulate alerts
          if (msg.data.alerts?.length > 0) {
            setAlerts((prev) => [
              ...msg.data.alerts.map((a, i) => ({
                ...a,
                id: `${Date.now()}-${i}`,
                timestamp: Date.now(),
              })),
              ...prev,
            ].slice(0, 30));
          }
        }
      } catch {}
    };

    ws.onclose = () => {
      setConnected(false);
    };

    ws.onerror = () => {
      setConnected(false);
    };

    wsRef.current = ws;
  }, []);

  const disconnectWs = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.send(JSON.stringify({ type: 'stop_activity_tracker' }));
      wsRef.current.close();
      wsRef.current = null;
    }
    setConnected(false);
  }, []);

  // ── Frame capture & send ───────────────────────────────

  const captureAndSend = useCallback(() => {
    if (!videoRef.current || !canvasRef.current || !wsRef.current) return;
    if (wsRef.current.readyState !== WebSocket.OPEN) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, 640, 480);

    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const reader = new FileReader();
        reader.onloadend = () => {
          const b64 = reader.result.split(',')[1];
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: 'frame', data: b64 }));
          }
        };
        reader.readAsDataURL(blob);
      },
      'image/jpeg',
      0.7
    );
  }, []);

  // ── Start / Stop orchestration ─────────────────────────

  const handleStart = useCallback(async () => {
    const ok = await startCamera();
    if (!ok) return;
    connectWs();
    setIsActive(true);

    // Start adaptive frame capture (~8 fps)
    intervalRef.current = setInterval(captureAndSend, 125);
  }, [startCamera, connectWs, captureAndSend]);

  const handleStop = useCallback(() => {
    clearInterval(intervalRef.current);
    disconnectWs();
    stopCamera();
    setIsActive(false);
    setMetrics(null);
    setFrameNum(0);
  }, [disconnectWs, stopCamera]);

  useEffect(() => {
    return () => {
      clearInterval(intervalRef.current);
      disconnectWs();
      stopCamera();
    };
  }, [disconnectWs, stopCamera]);

  // ── Computed display values ────────────────────────────

  const fatigue = metrics?.fatigue_score ?? 0;
  const attention = metrics?.attention_score ?? 100;
  const fatigueColor = fatigue >= 80 ? 'danger' : fatigue >= 50 ? 'warning' : 'success';
  const attentionColor = attention >= 70 ? 'success' : attention >= 40 ? 'warning' : 'danger';

  return (
    <div className="p-8 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Workstation Monitoring</h1>
          <p className="text-yaqiz-muted">Real-time fatigue & attention tracking via webcam</p>
        </div>
        <div className="flex items-center gap-3">
          {/* Connection badge */}
          <div className={`flex items-center gap-2 px-4 py-2 rounded-xl border ${
            connected
              ? 'bg-yaqiz-success/10 border-yaqiz-success/30 text-yaqiz-success'
              : 'bg-yaqiz-border border-yaqiz-border text-yaqiz-muted'
          }`}>
            {connected ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
            <span className="text-sm font-medium">{connected ? 'Connected' : 'Disconnected'}</span>
          </div>

          {/* Start / Stop */}
          <button
            onClick={isActive ? handleStop : handleStart}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-xl font-semibold transition-all duration-300 ${
              isActive
                ? 'bg-yaqiz-danger/20 border border-yaqiz-danger/30 text-yaqiz-danger hover:bg-yaqiz-danger/30'
                : 'btn-primary'
            }`}
          >
            {isActive ? (
              <><MonitorOff className="w-5 h-5" /> Stop</>
            ) : (
              <><Monitor className="w-5 h-5" /> Start Monitoring</>
            )}
          </button>
        </div>
      </div>

      {cameraError && (
        <div className="bg-yaqiz-danger/10 border border-yaqiz-danger/30 text-yaqiz-danger
          rounded-xl px-4 py-3 mb-6 text-sm animate-slide-up">
          {cameraError}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* ── Left: Webcam + Gauges (3 cols) ──────────── */}
        <div className="lg:col-span-3 space-y-6">
          {/* Webcam Preview */}
          <div className="glass-card overflow-hidden">
            <div className="flex items-center justify-between p-4 border-b border-yaqiz-border">
              <div className="flex items-center gap-3">
                <Monitor className="w-5 h-5 text-yaqiz-accent" />
                <span className="font-medium text-white">Webcam Feed</span>
              </div>
              {isActive && (
                <div className="flex items-center gap-3">
                  <span className="text-xs text-yaqiz-muted font-mono">Frame #{frameNum}</span>
                  <div className="flex items-center gap-2 px-3 py-1 bg-yaqiz-accent/20 border border-yaqiz-accent/30 rounded-full">
                    <div className="w-2 h-2 bg-yaqiz-accent rounded-full animate-pulse" />
                    <span className="text-xs text-yaqiz-accent font-bold">LIVE</span>
                  </div>
                </div>
              )}
            </div>

            <div className="bg-black aspect-video flex items-center justify-center relative">
              <video
                ref={videoRef}
                className={`w-full h-full object-contain ${isActive ? '' : 'hidden'}`}
                autoPlay
                playsInline
                muted
              />
              <canvas ref={canvasRef} className="hidden" />

              {!isActive && (
                <div className="text-center text-yaqiz-muted">
                  <Monitor className="w-20 h-20 mx-auto mb-4 opacity-20" />
                  <p className="text-lg">Camera preview</p>
                  <p className="text-sm mt-1 text-yaqiz-muted/50">
                    Click "Start Monitoring" to begin fatigue detection
                  </p>
                </div>
              )}

              {/* Presence overlay */}
              {isActive && metrics && !metrics.face_detected && (
                <div className="absolute inset-0 bg-black/60 flex items-center justify-center">
                  <div className="text-center">
                    <UserX className="w-16 h-16 text-yaqiz-danger mx-auto mb-3 animate-pulse" />
                    <p className="text-xl font-bold text-yaqiz-danger">No Face Detected</p>
                    <p className="text-sm text-yaqiz-muted mt-1">
                      Absent for {metrics.absence_duration?.toFixed(0) || 0}s
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Gauge Row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-5">
            <div className="glass-card p-5 flex flex-col items-center relative">
              <CircularGauge value={fatigue} label="Fatigue" color={fatigueColor} size={110} />
            </div>
            <div className="glass-card p-5 flex flex-col items-center relative">
              <CircularGauge value={attention} label="Attention" color={attentionColor} size={110} />
            </div>
            <div className="glass-card p-5">
              <h4 className="text-[10px] text-yaqiz-muted uppercase tracking-wider mb-3">EAR / MAR</h4>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-xs text-yaqiz-muted flex items-center gap-1">
                      <Eye className="w-3 h-3" /> EAR
                    </span>
                    <span className="text-xs font-mono text-white">{(metrics?.ear ?? 0).toFixed(3)}</span>
                  </div>
                  <div className="w-full bg-yaqiz-bg rounded-full h-1.5">
                    <div className={`h-full rounded-full transition-all duration-300 ${
                      (metrics?.ear ?? 0.3) < 0.2 ? 'bg-yaqiz-danger' : 'bg-yaqiz-success'
                    }`} style={{ width: `${Math.min((metrics?.ear ?? 0.3) / 0.4 * 100, 100)}%` }} />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-xs text-yaqiz-muted flex items-center gap-1">
                      <Minus className="w-3 h-3" /> MAR
                    </span>
                    <span className="text-xs font-mono text-white">{(metrics?.mar ?? 0).toFixed(3)}</span>
                  </div>
                  <div className="w-full bg-yaqiz-bg rounded-full h-1.5">
                    <div className={`h-full rounded-full transition-all duration-300 ${
                      (metrics?.mar ?? 0) > 0.6 ? 'bg-yaqiz-danger' : 'bg-yaqiz-accent'
                    }`} style={{ width: `${Math.min((metrics?.mar ?? 0) / 1 * 100, 100)}%` }} />
                  </div>
                </div>
              </div>
            </div>
            <div className="glass-card p-5">
              <h4 className="text-[10px] text-yaqiz-muted uppercase tracking-wider mb-3">Head Pose</h4>
              <HeadPoseIndicator status={metrics?.head_pose_status || 'unknown'} />
              <div className="mt-3 space-y-1 text-[10px] text-yaqiz-muted font-mono">
                <div className="flex justify-between">
                  <span>Yaw</span><span className="text-white">{(metrics?.head_yaw ?? 0).toFixed(1)}°</span>
                </div>
                <div className="flex justify-between">
                  <span>Pitch</span><span className="text-white">{(metrics?.head_pitch ?? 0).toFixed(1)}°</span>
                </div>
                <div className="flex justify-between">
                  <span>Roll</span><span className="text-white">{(metrics?.head_roll ?? 0).toFixed(1)}°</span>
                </div>
              </div>
            </div>
          </div>

          {/* Metric Counters */}
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
            <MiniStat icon={Eye} label="Blinks / 60s"
              value={metrics?.blink_rate ?? '-'} color="text-yaqiz-accent" />
            <MiniStat icon={Coffee} label="Yawn Count"
              value={metrics?.yawn_count ?? '-'}
              color={(metrics?.yawn_count ?? 0) >= 3 ? 'text-yaqiz-danger' : 'text-yaqiz-muted'} />
            <MiniStat icon={Hand} label="Eye Rubs"
              value={metrics?.eye_rub_count ?? '-'}
              color={(metrics?.eye_rub_count ?? 0) >= 3 ? 'text-yaqiz-warning' : 'text-yaqiz-muted'} />
            <MiniStat icon={User} label="Presence"
              value={metrics?.presence ? 'Present' : 'Absent'}
              color={metrics?.presence ? 'text-yaqiz-success' : 'text-yaqiz-danger'} />
            <MiniStat icon={Keyboard} label="Typing CPM"
              value={metrics?.typing_cpm ?? '-'} color="text-yaqiz-accent" />
            <MiniStat icon={MousePointer} label="Scroll SPM"
              value={metrics?.scroll_spm ?? '-'} color="text-yaqiz-accent" />
          </div>
        </div>

        {/* ── Right panel (1 col) ─────────────────────── */}
        <div className="space-y-5">
          {/* Status Overview */}
          <div className="glass-card p-5">
            <h3 className="text-sm font-semibold text-yaqiz-muted uppercase tracking-wider mb-4">
              Status
            </h3>
            <div className="space-y-4">
              {/* Fatigue Level */}
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-xs text-yaqiz-muted">Fatigue Level</span>
                  <span className={`text-xs font-bold ${
                    fatigue >= 80 ? 'text-yaqiz-danger' :
                    fatigue >= 50 ? 'text-yaqiz-warning' : 'text-yaqiz-success'
                  }`}>{fatigue.toFixed(0)}%</span>
                </div>
                <div className="w-full bg-yaqiz-bg rounded-full h-2">
                  <div className={`h-full rounded-full transition-all duration-500 ${
                    fatigue >= 80 ? 'bg-gradient-to-r from-yaqiz-danger to-red-400' :
                    fatigue >= 50 ? 'bg-gradient-to-r from-yaqiz-warning to-yellow-400' :
                    'bg-gradient-to-r from-yaqiz-success to-green-400'
                  }`} style={{ width: `${Math.min(fatigue, 100)}%` }} />
                </div>
              </div>

              {/* Attention Level */}
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-xs text-yaqiz-muted">Attention</span>
                  <span className={`text-xs font-bold ${
                    attention >= 70 ? 'text-yaqiz-success' :
                    attention >= 40 ? 'text-yaqiz-warning' : 'text-yaqiz-danger'
                  }`}>{attention.toFixed(0)}%</span>
                </div>
                <div className="w-full bg-yaqiz-bg rounded-full h-2">
                  <div className={`h-full rounded-full transition-all duration-500 ${
                    attention >= 70 ? 'bg-gradient-to-r from-yaqiz-success to-green-400' :
                    attention >= 40 ? 'bg-gradient-to-r from-yaqiz-warning to-yellow-400' :
                    'bg-gradient-to-r from-yaqiz-danger to-red-400'
                  }`} style={{ width: `${Math.min(attention, 100)}%` }} />
                </div>
              </div>

              <hr className="border-yaqiz-border" />

              {/* Idle Status */}
              <div className="flex items-center justify-between">
                <span className="text-xs text-yaqiz-muted flex items-center gap-2">
                  <Clock className="w-3.5 h-3.5" /> Activity
                </span>
                <span className={`text-xs font-bold ${
                  metrics?.user_idle ? 'text-yaqiz-danger' : 'text-yaqiz-success'
                }`}>
                  {metrics?.user_idle ? `Idle (${(metrics?.idle_time ?? 0).toFixed(0)}s)` : 'Active'}
                </span>
              </div>

              {/* Hand on head */}
              <div className="flex items-center justify-between">
                <span className="text-xs text-yaqiz-muted flex items-center gap-2">
                  <Hand className="w-3.5 h-3.5" /> Hand on Head
                </span>
                <span className={`text-xs font-bold ${
                  metrics?.hand_on_head ? 'text-yaqiz-warning' : 'text-yaqiz-success'
                }`}>
                  {metrics?.hand_on_head ? 'Yes' : 'No'}
                </span>
              </div>
            </div>
          </div>

          {/* Active Window */}
          <div className="glass-card p-5">
            <h3 className="text-sm font-semibold text-yaqiz-muted uppercase tracking-wider mb-3">
              Active Window
            </h3>
            <p className="text-sm text-white truncate" title={metrics?.active_window}>
              {metrics?.active_window || 'Unknown'}
            </p>
          </div>

          {/* Live Alerts */}
          <div className="glass-card p-5">
            <h3 className="text-sm font-semibold text-yaqiz-muted uppercase tracking-wider mb-4 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4" />
              Workstation Alerts
            </h3>
            <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
              {alerts.length > 0 ? (
                alerts.map((alert) => (
                  <AlertBadge key={alert.id} alert={alert} />
                ))
              ) : (
                <div className="text-center py-8 text-yaqiz-muted">
                  <CheckCircle className="w-8 h-8 mx-auto mb-2 opacity-30 text-yaqiz-success" />
                  <p className="text-xs">No alerts yet</p>
                  <p className="text-[10px] mt-1 opacity-50">Alerts will appear here in real time</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
