import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Eye, EyeOff, ArrowRight } from 'lucide-react';
import { authAPI } from '../services/api';
import { useAuth } from '../hooks/useWebSocket';

export default function Login() {
  const [form, setForm] = useState({ username: '', password: '' });
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const { data } = await authAPI.login(form);
      login(data.access_token, data.user);
      navigate('/');
    } catch (err) {
      setError(err.response?.data?.detail || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-yaqiz-bg flex">
      {/* Left - Branding */}
      <div className="hidden lg:flex lg:w-1/2 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-yaqiz-accent/20 via-yaqiz-bg to-yaqiz-accent2/20" />
        <div className="absolute inset-0" style={{
          backgroundImage: 'radial-gradient(circle at 25% 50%, rgba(0,212,255,0.08) 0%, transparent 50%), radial-gradient(circle at 75% 50%, rgba(59,130,246,0.08) 0%, transparent 50%)'
        }} />
        
        <div className="relative z-10 flex flex-col justify-center px-16">
          <div className="flex items-center gap-4 mb-8">
            <div className="w-16 h-16 rounded-2xl overflow-hidden shadow-lg shadow-yaqiz-accent/20
              ring-1 ring-yaqiz-accent/20">
              <img src="/logo.jpeg" alt="YAQIZ" className="w-full h-full object-cover" />
            </div>
            <div>
              <h1 className="text-4xl font-extrabold bg-gradient-to-r from-yaqiz-accent to-yaqiz-accent2
                bg-clip-text text-transparent">YAQIZ</h1>
              <p className="text-sm text-yaqiz-muted tracking-widest uppercase">PPE Detection Platform</p>
            </div>
          </div>

          <h2 className="text-3xl font-bold text-white mb-4 leading-tight">
            Intelligent Safety<br />Monitoring System
          </h2>
          <p className="text-yaqiz-muted text-lg leading-relaxed max-w-md">
            AI-powered real-time PPE detection and compliance monitoring
            for industrial safety management.
          </p>

          <div className="mt-12 grid grid-cols-2 gap-4">
            {[
              { label: 'Real-time Detection', value: 'YOLOv8' },
              { label: 'PPE Classes', value: '10' },
              { label: 'Streaming', value: 'WebSocket' },
              { label: 'Compliance', value: 'Analytics' },
            ].map((item, i) => (
              <div key={i} className="bg-white/5 rounded-xl p-4 border border-white/5 backdrop-blur-sm">
                <p className="text-yaqiz-accent font-bold text-lg">{item.value}</p>
                <p className="text-yaqiz-muted text-sm">{item.label}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Right - Login Form */}
      <div className="flex-1 flex items-center justify-center px-6 py-12">
        <div className="w-full max-w-md">
          {/* Mobile logo */}
          <div className="lg:hidden flex items-center gap-3 mb-10 justify-center">
            <div className="w-12 h-12 rounded-xl overflow-hidden ring-1 ring-yaqiz-accent/20">
              <img src="/logo.jpeg" alt="YAQIZ" className="w-full h-full object-cover" />
            </div>
            <h1 className="text-2xl font-extrabold bg-gradient-to-r from-yaqiz-accent to-yaqiz-accent2
              bg-clip-text text-transparent">YAQIZ</h1>
          </div>

          <div className="glass-card p-8">
            <h2 className="text-2xl font-bold text-white mb-2">Welcome Back</h2>
            <p className="text-yaqiz-muted mb-8">Sign in to your YAQIZ account</p>

            {error && (
              <div className="bg-yaqiz-danger/10 border border-yaqiz-danger/30 text-yaqiz-danger
                rounded-xl px-4 py-3 mb-6 text-sm animate-slide-up">
                {error}
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-5">
              <div>
                <label className="block text-sm font-medium text-yaqiz-muted mb-2">Username</label>
                <input
                  type="text"
                  className="input-field"
                  placeholder="Enter your username"
                  value={form.username}
                  onChange={(e) => setForm({ ...form, username: e.target.value })}
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-yaqiz-muted mb-2">Password</label>
                <div className="relative">
                  <input
                    type={showPassword ? 'text' : 'password'}
                    className="input-field pr-12"
                    placeholder="Enter your password"
                    value={form.password}
                    onChange={(e) => setForm({ ...form, password: e.target.value })}
                    required
                  />
                  <button type="button" onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-yaqiz-muted hover:text-white">
                    {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                  </button>
                </div>
              </div>

              <button type="submit" disabled={loading}
                className="btn-primary w-full flex items-center justify-center gap-2">
                {loading ? (
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                ) : (
                  <>Sign In <ArrowRight className="w-4 h-4" /></>
                )}
              </button>
            </form>

            <p className="text-center text-yaqiz-muted mt-6 text-sm">
              Don't have an account?{' '}
              <Link to="/register" className="text-yaqiz-accent hover:underline font-medium">
                Create Account
              </Link>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
