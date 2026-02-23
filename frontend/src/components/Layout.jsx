import React, { useState } from 'react';
import { Outlet, NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useWebSocket';
import {
  LayoutDashboard, Video, Camera, Bell, LogOut, Menu, X,
  ChevronRight, Monitor
} from 'lucide-react';

const navItems = [
  { path: '/', icon: LayoutDashboard, label: 'Dashboard', end: true },
  { path: '/live', icon: Camera, label: 'Live Monitoring' },
  { path: '/video', icon: Video, label: 'Video Analysis' },
  { path: '/alerts', icon: Bell, label: 'Alerts Center' },
  { path: '/workstation', icon: Monitor, label: 'Workstation' },
];

export default function Layout() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className={`${sidebarOpen ? 'w-64' : 'w-20'} bg-yaqiz-card border-r border-yaqiz-border
        flex flex-col transition-all duration-300 ease-in-out flex-shrink-0`}>
        
        {/* Logo */}
        <div className="p-5 flex items-center gap-3 border-b border-yaqiz-border">
          <div className="w-10 h-10 rounded-xl overflow-hidden flex-shrink-0 ring-1 ring-yaqiz-accent/20">
            <img src="/logo.jpeg" alt="YAQIZ" className="w-full h-full object-cover" />
          </div>
          {sidebarOpen && (
            <div className="animate-fade-in">
              <h1 className="text-lg font-bold bg-gradient-to-r from-yaqiz-accent to-yaqiz-accent2
                bg-clip-text text-transparent">YAQIZ</h1>
              <p className="text-[10px] text-yaqiz-muted uppercase tracking-widest">PPE Detection</p>
            </div>
          )}
        </div>

        {/* Nav Links */}
        <nav className="flex-1 py-4 px-3 space-y-1">
          {navItems.map(({ path, icon: Icon, label, end }) => (
            <NavLink
              key={path}
              to={path}
              end={end}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200
                ${isActive
                  ? 'bg-gradient-to-r from-yaqiz-accent/15 to-yaqiz-accent2/10 text-yaqiz-accent border border-yaqiz-accent/20'
                  : 'text-yaqiz-muted hover:text-white hover:bg-white/5'
                }`
              }
            >
              <Icon className="w-5 h-5 flex-shrink-0" />
              {sidebarOpen && <span className="text-sm font-medium">{label}</span>}
            </NavLink>
          ))}
        </nav>

        {/* User section */}
        <div className="p-4 border-t border-yaqiz-border">
          {sidebarOpen && user && (
            <div className="flex items-center gap-3 mb-3">
              <div className="w-8 h-8 bg-gradient-to-br from-yaqiz-accent to-yaqiz-accent2 rounded-lg
                flex items-center justify-center text-sm font-bold">
                {user.username?.charAt(0).toUpperCase()}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-white truncate">{user.username}</p>
                <p className="text-xs text-yaqiz-muted truncate">{user.role}</p>
              </div>
            </div>
          )}
          <button onClick={handleLogout}
            className="flex items-center gap-3 w-full px-3 py-2 rounded-lg text-yaqiz-muted
              hover:text-yaqiz-danger hover:bg-yaqiz-danger/10 transition-all duration-200">
            <LogOut className="w-5 h-5 flex-shrink-0" />
            {sidebarOpen && <span className="text-sm">Logout</span>}
          </button>
        </div>

        {/* Toggle */}
        <button onClick={() => setSidebarOpen(!sidebarOpen)}
          className="p-3 border-t border-yaqiz-border text-yaqiz-muted hover:text-white transition-colors">
          {sidebarOpen ? <X className="w-5 h-5 mx-auto" /> : <Menu className="w-5 h-5 mx-auto" />}
        </button>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto bg-yaqiz-bg">
        <div className="animate-page-in">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
