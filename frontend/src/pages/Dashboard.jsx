import React, { useState, useEffect } from 'react';
import {
  Activity, AlertTriangle, CheckCircle, Users, Camera, TrendingUp,
  Eye, Clock, BarChart3, Shield, UserCheck, UserX
} from 'lucide-react';
import { dashboardAPI, detectionAPI } from '../services/api';

function StatCard({ icon: Icon, label, value, subtext, color, trend }) {
  const colorClasses = {
    accent: 'from-yaqiz-accent/20 to-yaqiz-accent2/10 border-yaqiz-accent/20',
    success: 'from-yaqiz-success/20 to-yaqiz-success/5 border-yaqiz-success/20',
    danger: 'from-yaqiz-danger/20 to-yaqiz-danger/5 border-yaqiz-danger/20',
    warning: 'from-yaqiz-warning/20 to-yaqiz-warning/5 border-yaqiz-warning/20',
  };
  const iconColors = {
    accent: 'text-yaqiz-accent',
    success: 'text-yaqiz-success',
    danger: 'text-yaqiz-danger',
    warning: 'text-yaqiz-warning',
  };

  return (
    <div className={`glass-card p-6 bg-gradient-to-br ${colorClasses[color]} hover:scale-[1.02] transition-transform duration-200`}>
      <div className="flex items-start justify-between mb-4">
        <div className={`w-12 h-12 rounded-xl bg-yaqiz-bg/50 flex items-center justify-center ${iconColors[color]}`}>
          <Icon className="w-6 h-6" />
        </div>
        {trend && (
          <span className={`text-xs font-semibold px-2 py-1 rounded-lg ${
            trend > 0 ? 'bg-yaqiz-success/15 text-yaqiz-success' : 'bg-yaqiz-danger/15 text-yaqiz-danger'
          }`}>
            {trend > 0 ? '+' : ''}{trend}%
          </span>
        )}
      </div>
      <p className="text-3xl font-bold text-white mb-1">{value}</p>
      <p className="text-sm text-yaqiz-muted">{label}</p>
      {subtext && <p className="text-xs text-yaqiz-muted/70 mt-1">{subtext}</p>}
    </div>
  );
}

function RecentSession({ session }) {
  const statusColors = {
    completed: 'badge-success',
    processing: 'badge-warning',
    pending: 'badge-warning',
    failed: 'badge-danger',
  };

  return (
    <div className="flex items-center justify-between py-3 border-b border-yaqiz-border/50 last:border-0">
      <div className="flex items-center gap-3">
        <div className={`w-2 h-2 rounded-full ${
          session.status === 'completed' ? 'bg-yaqiz-success' :
          session.status === 'failed' ? 'bg-yaqiz-danger' : 'bg-yaqiz-warning animate-pulse'
        }`} />
        <div>
          <p className="text-sm font-medium text-white capitalize">{session.session_type} Analysis</p>
          <p className="text-xs text-yaqiz-muted">
            {new Date(session.created_at).toLocaleString()}
          </p>
        </div>
      </div>
      <div className="text-right">
        <span className={statusColors[session.status] || 'badge'}>{session.status}</span>
        {session.total_detections > 0 && (
          <p className="text-xs text-yaqiz-muted mt-1">{session.total_detections} detections</p>
        )}
      </div>
    </div>
  );
}

function RecentAlert({ alert }) {
  const severityIcons = {
    critical: AlertTriangle,
    warning: AlertTriangle,
    info: Eye,
  };
  const Icon = severityIcons[alert.severity] || Eye;

  return (
    <div className="flex items-start gap-3 py-3 border-b border-yaqiz-border/50 last:border-0">
      <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${
        alert.severity === 'critical' ? 'bg-yaqiz-danger/15 text-yaqiz-danger' :
        alert.severity === 'warning' ? 'bg-yaqiz-warning/15 text-yaqiz-warning' :
        'bg-yaqiz-accent/15 text-yaqiz-accent'
      }`}>
        <Icon className="w-4 h-4" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm text-white truncate">{alert.message}</p>
        <p className="text-xs text-yaqiz-muted">
          {new Date(alert.created_at).toLocaleString()}
        </p>
      </div>
      <span className={
        alert.severity === 'critical' ? 'badge-danger' : 
        alert.severity === 'warning' ? 'badge-warning' : 'badge-success'
      }>
        {alert.severity}
      </span>
    </div>
  );
}

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [trackerData, setTrackerData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadStats();
    loadTrackerData();
    const interval = setInterval(() => { loadStats(); loadTrackerData(); }, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadStats = async () => {
    try {
      const { data } = await dashboardAPI.getStats();
      setStats(data);
    } catch (err) {
      console.error('Failed to load stats:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadTrackerData = async () => {
    try {
      const { data } = await detectionAPI.getTrackedWorkers();
      setTrackerData(data);
    } catch (err) {
      // Tracker may not have data yet, that's fine
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-yaqiz-accent/20 border-t-yaqiz-accent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-yaqiz-muted">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-8 animate-fade-in">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Executive Dashboard</h1>
        <p className="text-yaqiz-muted">PPE safety compliance overview and analytics</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5 mb-8">
        <StatCard
          icon={Eye}
          label="Total Detections"
          value={stats?.total_detections?.toLocaleString() || '0'}
          subtext={`${stats?.sessions_today || 0} sessions today`}
          color="accent"
        />
        <StatCard
          icon={AlertTriangle}
          label="Violations"
          value={stats?.total_violations?.toLocaleString() || '0'}
          subtext={`${stats?.active_alerts || 0} active alerts`}
          color="danger"
        />
        <StatCard
          icon={CheckCircle}
          label="Compliance Rate"
          value={`${stats?.overall_compliance?.toFixed(1) || '100'}%`}
          subtext="Overall safety score"
          color="success"
        />
        <StatCard
          icon={Activity}
          label="Total Sessions"
          value={stats?.total_sessions?.toLocaleString() || '0'}
          subtext="Video, Image & Live"
          color="warning"
        />
      </div>

      {/* Compliance Bar */}
      <div className="glass-card p-6 mb-8">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <Shield className="w-5 h-5 text-yaqiz-accent" />
            Safety Compliance Overview
          </h3>
          <span className={`text-2xl font-bold ${
            (stats?.overall_compliance || 100) >= 80 ? 'text-yaqiz-success' :
            (stats?.overall_compliance || 100) >= 50 ? 'text-yaqiz-warning' : 'text-yaqiz-danger'
          }`}>
            {stats?.overall_compliance?.toFixed(1) || '100'}%
          </span>
        </div>
        <div className="w-full bg-yaqiz-bg rounded-full h-4 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-1000 ${
              (stats?.overall_compliance || 100) >= 80 ? 'bg-gradient-to-r from-yaqiz-success to-green-400' :
              (stats?.overall_compliance || 100) >= 50 ? 'bg-gradient-to-r from-yaqiz-warning to-yellow-400' :
              'bg-gradient-to-r from-yaqiz-danger to-red-400'
            }`}
            style={{ width: `${Math.min(stats?.overall_compliance || 100, 100)}%` }}
          />
        </div>
      </div>

      {/* Worker Tracking Summary */}
      {trackerData && (trackerData.unique_workers > 0) && (
        <div className="glass-card p-6 mb-8">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2 mb-4">
            <Users className="w-5 h-5 text-yaqiz-accent" />
            Worker Tracking (Live Session)
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <div className="bg-yaqiz-bg/50 rounded-xl p-4 text-center border border-yaqiz-accent/10">
              <Users className="w-6 h-6 text-yaqiz-accent mx-auto mb-2" />
              <p className="text-2xl font-bold text-white">{trackerData.unique_workers}</p>
              <p className="text-xs text-yaqiz-muted">Unique Workers</p>
            </div>
            <div className="bg-yaqiz-bg/50 rounded-xl p-4 text-center border border-yaqiz-danger/10">
              <UserX className="w-6 h-6 text-yaqiz-danger mx-auto mb-2" />
              <p className="text-2xl font-bold text-yaqiz-danger">{trackerData.unique_violators}</p>
              <p className="text-xs text-yaqiz-muted">Unique Violators</p>
            </div>
            <div className="bg-yaqiz-bg/50 rounded-xl p-4 text-center border border-yaqiz-success/10">
              <UserCheck className="w-6 h-6 text-yaqiz-success mx-auto mb-2" />
              <p className="text-2xl font-bold text-yaqiz-success">
                {trackerData.unique_workers - trackerData.unique_violators}
              </p>
              <p className="text-xs text-yaqiz-muted">Compliant Workers</p>
            </div>
            <div className="bg-yaqiz-bg/50 rounded-xl p-4 text-center border border-yaqiz-warning/10">
              <Shield className="w-6 h-6 text-yaqiz-warning mx-auto mb-2" />
              <p className="text-2xl font-bold text-yaqiz-warning">
                {trackerData.unique_workers > 0
                  ? ((1 - trackerData.unique_violators / trackerData.unique_workers) * 100).toFixed(0)
                  : '100'}%
              </p>
              <p className="text-xs text-yaqiz-muted">Worker Compliance</p>
            </div>
          </div>

          {/* Top Violators Table */}
          {trackerData.workers?.length > 0 && (
            <div>
              <h4 className="text-sm font-semibold text-yaqiz-muted mb-3">Tracked Workers</h4>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-yaqiz-muted text-xs border-b border-yaqiz-border">
                      <th className="text-left py-2 px-2">ID</th>
                      <th className="text-left py-2 px-2">Status</th>
                      <th className="text-center py-2 px-2">Frames</th>
                      <th className="text-center py-2 px-2">⛑️</th>
                      <th className="text-center py-2 px-2">🦺</th>
                      <th className="text-center py-2 px-2">😷</th>
                      <th className="text-center py-2 px-2">Total</th>
                    </tr>
                  </thead>
                  <tbody>
                    {trackerData.workers
                      .sort((a, b) => (b.violations?.total || 0) - (a.violations?.total || 0))
                      .slice(0, 10)
                      .map((w) => (
                        <tr key={w.worker_id} className="border-b border-yaqiz-border/30 hover:bg-yaqiz-bg/30">
                          <td className="py-2 px-2 font-mono text-white">#{w.worker_id}</td>
                          <td className="py-2 px-2">
                            <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${
                              w.is_compliant
                                ? 'bg-yaqiz-success/20 text-yaqiz-success'
                                : 'bg-yaqiz-danger/20 text-yaqiz-danger'
                            }`}>
                              {w.is_compliant ? 'OK' : 'Violation'}
                            </span>
                          </td>
                          <td className="py-2 px-2 text-center text-yaqiz-muted">{w.total_frames}</td>
                          <td className="py-2 px-2 text-center text-yaqiz-danger">{w.violations?.no_hardhat || 0}</td>
                          <td className="py-2 px-2 text-center text-yaqiz-danger">{w.violations?.no_vest || 0}</td>
                          <td className="py-2 px-2 text-center text-yaqiz-danger">{w.violations?.no_mask || 0}</td>
                          <td className="py-2 px-2 text-center font-bold text-white">{w.violations?.total || 0}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Bottom Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Sessions */}
        <div className="glass-card p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Clock className="w-5 h-5 text-yaqiz-accent" />
            Recent Sessions
          </h3>
          <div className="space-y-0">
            {stats?.recent_sessions?.length > 0 ? (
              stats.recent_sessions.map((s) => (
                <RecentSession key={s.id} session={s} />
              ))
            ) : (
              <div className="text-center py-8 text-yaqiz-muted">
                <Camera className="w-10 h-10 mx-auto mb-3 opacity-50" />
                <p>No sessions yet</p>
                <p className="text-xs mt-1">Start a live session or upload media</p>
              </div>
            )}
          </div>
        </div>

        {/* Recent Alerts */}
        <div className="glass-card p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-yaqiz-warning" />
            Recent Alerts
          </h3>
          <div className="space-y-0">
            {stats?.recent_alerts?.length > 0 ? (
              stats.recent_alerts.map((a) => (
                <RecentAlert key={a.id} alert={a} />
              ))
            ) : (
              <div className="text-center py-8 text-yaqiz-muted">
                <CheckCircle className="w-10 h-10 mx-auto mb-3 opacity-50 text-yaqiz-success" />
                <p>No alerts</p>
                <p className="text-xs mt-1">All clear - no violations detected</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
