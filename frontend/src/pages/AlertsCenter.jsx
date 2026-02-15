import React, { useState, useEffect, useCallback } from 'react';
import {
  Bell, BellOff, Filter, Search, AlertTriangle, CheckCircle,
  Info, Clock, Trash2, Eye, RefreshCw
} from 'lucide-react';
import { dashboardAPI } from '../services/api';

const SEVERITY_CONFIG = {
  critical: {
    icon: AlertTriangle,
    bg: 'bg-yaqiz-danger/10',
    border: 'border-yaqiz-danger/20',
    text: 'text-yaqiz-danger',
    badge: 'badge-danger',
    label: 'Critical',
  },
  warning: {
    icon: AlertTriangle,
    bg: 'bg-yaqiz-warning/10',
    border: 'border-yaqiz-warning/20',
    text: 'text-yaqiz-warning',
    badge: 'badge-warning',
    label: 'Warning',
  },
  info: {
    icon: Info,
    bg: 'bg-yaqiz-accent/10',
    border: 'border-yaqiz-accent/20',
    text: 'text-yaqiz-accent',
    badge: 'badge bg-yaqiz-accent/15 text-yaqiz-accent border border-yaqiz-accent/20',
    label: 'Info',
  },
};

function AlertCard({ alert, onMarkRead }) {
  const config = SEVERITY_CONFIG[alert.severity] || SEVERITY_CONFIG.info;
  const Icon = config.icon;

  return (
    <div className={`${config.bg} border ${config.border} rounded-xl p-4 transition-all duration-200
      hover:bg-opacity-20 ${alert.is_read ? 'opacity-60' : ''} animate-slide-up`}>
      <div className="flex items-start gap-3">
        <div className={`w-10 h-10 rounded-xl ${config.bg} flex items-center justify-center flex-shrink-0 ${config.text}`}>
          <Icon className="w-5 h-5" />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className={config.badge}>{config.label}</span>
            <span className="text-xs text-yaqiz-muted font-mono">
              {alert.alert_type.replace(/_/g, ' ')}
            </span>
            {!alert.is_read && (
              <div className="w-2 h-2 bg-yaqiz-accent rounded-full animate-pulse" />
            )}
          </div>

          <p className="text-sm text-white">{alert.message}</p>

          <div className="flex items-center gap-4 mt-2 text-xs text-yaqiz-muted">
            <span className="flex items-center gap-1">
              <Clock className="w-3 h-3" />
              {new Date(alert.created_at).toLocaleString()}
            </span>
            {alert.frame_number && (
              <span>Frame #{alert.frame_number}</span>
            )}
            <span className="font-mono">
              Conf: {(alert.confidence * 100).toFixed(0)}%
            </span>
          </div>
        </div>

        {!alert.is_read && (
          <button
            onClick={() => onMarkRead(alert.id)}
            className="text-yaqiz-muted hover:text-yaqiz-accent transition-colors p-1"
            title="Mark as read"
          >
            <Eye className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  );
}

export default function AlertsCenter() {
  const [alerts, setAlerts] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState({ severity: '', unread_only: false, search: '' });

  const loadAlerts = useCallback(async () => {
    setLoading(true);
    try {
      const params = {};
      if (filter.severity) params.severity = filter.severity;
      if (filter.unread_only) params.unread_only = true;

      const { data } = await dashboardAPI.getAlerts(params);
      setAlerts(data.alerts);
      setTotal(data.total);
    } catch (err) {
      console.error('Failed to load alerts:', err);
    } finally {
      setLoading(false);
    }
  }, [filter.severity, filter.unread_only]);

  useEffect(() => {
    loadAlerts();
    const interval = setInterval(loadAlerts, 15000);
    return () => clearInterval(interval);
  }, [loadAlerts]);

  const markRead = async (id) => {
    try {
      await dashboardAPI.markAlertRead(id);
      setAlerts((prev) => prev.map((a) => (a.id === id ? { ...a, is_read: true } : a)));
    } catch {}
  };

  const markAllRead = async () => {
    try {
      await dashboardAPI.markAllRead();
      setAlerts((prev) => prev.map((a) => ({ ...a, is_read: true })));
    } catch {}
  };

  const unreadCount = alerts.filter((a) => !a.is_read).length;

  const filteredAlerts = alerts.filter((a) => {
    if (filter.search) {
      const s = filter.search.toLowerCase();
      return a.message.toLowerCase().includes(s) || a.alert_type.toLowerCase().includes(s);
    }
    return true;
  });

  // Group by severity
  const criticalCount = alerts.filter((a) => a.severity === 'critical').length;
  const warningCount = alerts.filter((a) => a.severity === 'warning').length;
  const infoCount = alerts.filter((a) => a.severity === 'info').length;

  return (
    <div className="p-8 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Alerts Center</h1>
          <p className="text-yaqiz-muted">
            {total} total alerts · {unreadCount} unread
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={loadAlerts}
            className="btn-ghost flex items-center gap-2"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
          {unreadCount > 0 && (
            <button
              onClick={markAllRead}
              className="btn-ghost flex items-center gap-2 text-yaqiz-accent"
            >
              <CheckCircle className="w-4 h-4" />
              Mark All Read
            </button>
          )}
        </div>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {[
          { label: 'Critical', count: criticalCount, config: SEVERITY_CONFIG.critical },
          { label: 'Warning', count: warningCount, config: SEVERITY_CONFIG.warning },
          { label: 'Info', count: infoCount, config: SEVERITY_CONFIG.info },
        ].map(({ label, count, config }) => {
          const Icon = config.icon;
          return (
            <button
              key={label}
              onClick={() => setFilter({ ...filter, severity: filter.severity === label.toLowerCase() ? '' : label.toLowerCase() })}
              className={`glass-card p-4 flex items-center gap-4 transition-all duration-200
                ${filter.severity === label.toLowerCase() ? `${config.border} border-2` : 'hover:border-yaqiz-accent/20'}`}
            >
              <div className={`w-10 h-10 rounded-xl ${config.bg} flex items-center justify-center ${config.text}`}>
                <Icon className="w-5 h-5" />
              </div>
              <div className="text-left">
                <p className="text-2xl font-bold text-white">{count}</p>
                <p className="text-xs text-yaqiz-muted">{label}</p>
              </div>
            </button>
          );
        })}
      </div>

      {/* Filters */}
      <div className="glass-card p-4 mb-6">
        <div className="flex items-center gap-4 flex-wrap">
          <div className="flex items-center gap-2 flex-1 min-w-[200px]">
            <Search className="w-4 h-4 text-yaqiz-muted" />
            <input
              type="text"
              placeholder="Search alerts..."
              className="bg-transparent border-none outline-none text-sm text-white placeholder-yaqiz-muted/50 flex-1"
              value={filter.search}
              onChange={(e) => setFilter({ ...filter, search: e.target.value })}
            />
          </div>

          <button
            onClick={() => setFilter({ ...filter, unread_only: !filter.unread_only })}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition-colors ${
              filter.unread_only
                ? 'bg-yaqiz-accent/15 text-yaqiz-accent border border-yaqiz-accent/30'
                : 'text-yaqiz-muted hover:text-white'
            }`}
          >
            {filter.unread_only ? <Bell className="w-4 h-4" /> : <BellOff className="w-4 h-4" />}
            Unread Only
          </button>

          {(filter.severity || filter.unread_only || filter.search) && (
            <button
              onClick={() => setFilter({ severity: '', unread_only: false, search: '' })}
              className="text-xs text-yaqiz-muted hover:text-white flex items-center gap-1"
            >
              <Trash2 className="w-3 h-3" />
              Clear Filters
            </button>
          )}
        </div>
      </div>

      {/* Alerts List */}
      <div className="space-y-3">
        {loading && alerts.length === 0 ? (
          <div className="text-center py-16">
            <div className="w-10 h-10 border-4 border-yaqiz-accent/20 border-t-yaqiz-accent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-yaqiz-muted">Loading alerts...</p>
          </div>
        ) : filteredAlerts.length > 0 ? (
          filteredAlerts.map((alert) => (
            <AlertCard key={alert.id} alert={alert} onMarkRead={markRead} />
          ))
        ) : (
          <div className="text-center py-16 text-yaqiz-muted">
            <CheckCircle className="w-16 h-16 mx-auto mb-4 opacity-20 text-yaqiz-success" />
            <p className="text-lg font-medium">All Clear</p>
            <p className="text-sm mt-1">
              {filter.severity || filter.unread_only ? 'No alerts match your filters' : 'No safety violations detected'}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
