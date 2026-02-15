import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Eye, EyeOff, ArrowRight, UserPlus, CheckCircle } from 'lucide-react';
import { authAPI } from '../services/api';

export default function Register() {
  const [form, setForm] = useState({ username: '', email: '', password: '', full_name: '' });
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await authAPI.register(form);
      setSuccess(true);
      setTimeout(() => navigate('/login'), 2000);
    } catch (err) {
      setError(err.response?.data?.detail || 'Registration failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-yaqiz-bg flex items-center justify-center px-6 py-12">
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="flex items-center gap-3 mb-10 justify-center">
          <div className="w-12 h-12 rounded-xl overflow-hidden ring-1 ring-yaqiz-accent/20">
            <img src="/logo.jpeg" alt="YAQIZ" className="w-full h-full object-cover" />
          </div>
          <div>
            <h1 className="text-2xl font-extrabold bg-gradient-to-r from-yaqiz-accent to-yaqiz-accent2
              bg-clip-text text-transparent">YAQIZ</h1>
            <p className="text-[10px] text-yaqiz-muted tracking-widest uppercase">PPE Detection</p>
          </div>
        </div>

        <div className="glass-card p-8">
          <div className="flex items-center gap-3 mb-6">
            <UserPlus className="w-6 h-6 text-yaqiz-accent" />
            <h2 className="text-2xl font-bold text-white">Create Account</h2>
          </div>
          <p className="text-yaqiz-muted mb-8">Join YAQIZ safety monitoring platform</p>

          {error && (
            <div className="bg-yaqiz-danger/10 border border-yaqiz-danger/30 text-yaqiz-danger
              rounded-xl px-4 py-3 mb-6 text-sm animate-slide-up">
              {error}
            </div>
          )}

          {success && (
            <div className="bg-yaqiz-success/10 border border-yaqiz-success/30 text-yaqiz-success
              rounded-xl px-4 py-3 mb-6 text-sm animate-slide-up flex items-center gap-2">
              <CheckCircle className="w-4 h-4 flex-shrink-0" />
              Account created successfully! Redirecting to Sign In...
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-yaqiz-muted mb-2">Full Name</label>
              <input
                type="text"
                className="input-field"
                placeholder="Enter your full name"
                value={form.full_name}
                onChange={(e) => setForm({ ...form, full_name: e.target.value })}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-yaqiz-muted mb-2">Username</label>
              <input
                type="text"
                className="input-field"
                placeholder="Choose a username"
                value={form.username}
                onChange={(e) => setForm({ ...form, username: e.target.value })}
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-yaqiz-muted mb-2">Email</label>
              <input
                type="email"
                className="input-field"
                placeholder="your@email.com"
                value={form.email}
                onChange={(e) => setForm({ ...form, email: e.target.value })}
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-yaqiz-muted mb-2">Password</label>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  className="input-field pr-12"
                  placeholder="Choose a strong password"
                  value={form.password}
                  onChange={(e) => setForm({ ...form, password: e.target.value })}
                  required
                  minLength={6}
                />
                <button type="button" onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-yaqiz-muted hover:text-white">
                  {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
            </div>

            <button type="submit" disabled={loading}
              className="btn-primary w-full flex items-center justify-center gap-2 mt-6">
              {loading ? (
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                <>Create Account <ArrowRight className="w-4 h-4" /></>
              )}
            </button>
          </form>

          <p className="text-center text-yaqiz-muted mt-6 text-sm">
            Already have an account?{' '}
            <Link to="/login" className="text-yaqiz-accent hover:underline font-medium">
              Sign In
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
