import React, { useState, useEffect, useRef } from 'react';
import {
  Upload, Video, Play, CheckCircle, AlertTriangle, FileVideo,
  Loader2, BarChart3, Eye, Clock, X, Image
} from 'lucide-react';
import { detectionAPI } from '../services/api';

function UploadZone({ onUpload, type = 'video', loading }) {
  const [dragOver, setDragOver] = useState(false);
  const [fileName, setFileName] = useState('');
  const fileRef = useRef(null);

  const accept = type === 'video' ? 'video/*' : 'image/*';
  const icon = type === 'video' ? FileVideo : Image;
  const Icon = icon;

  const handleFile = (file) => {
    if (file) {
      setFileName(file.name);
      onUpload(file);
    }
  };

  return (
    <div
      className={`border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-300 cursor-pointer
        ${dragOver ? 'border-yaqiz-accent bg-yaqiz-accent/5' : 'border-yaqiz-border hover:border-yaqiz-accent/50'}
        ${loading ? 'pointer-events-none opacity-60' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]); }}
      onClick={() => fileRef.current?.click()}
    >
      <input
        ref={fileRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => handleFile(e.target.files[0])}
      />
      {loading ? (
        <Loader2 className="w-12 h-12 text-yaqiz-accent mx-auto mb-4 animate-spin" />
      ) : (
        <Icon className="w-12 h-12 text-yaqiz-accent/50 mx-auto mb-4" />
      )}
      <p className="text-white font-medium mb-1">
        {fileName || `Drop ${type} here or click to browse`}
      </p>
      <p className="text-xs text-yaqiz-muted">
        {type === 'video' ? 'MP4, AVI, MOV, MKV (max 500MB)' : 'JPG, PNG, BMP, WEBP'}
      </p>
    </div>
  );
}

export default function VideoAnalysis() {
  const [tab, setTab] = useState('video'); // video | image
  const [confidence, setConfidence] = useState(0.5);
  const [frameSkip, setFrameSkip] = useState(3);
  const [loading, setLoading] = useState(false);
  const [sessions, setSessions] = useState([]);
  const [result, setResult] = useState(null);
  const [imageResult, setImageResult] = useState(null);

  useEffect(() => {
    loadSessions();
  }, []);

  const loadSessions = async () => {
    try {
      const { data } = await detectionAPI.getSessions();
      setSessions(data);
    } catch {}
  };

  const handleVideoUpload = async (file) => {
    setLoading(true);
    setResult(null);
    try {
      const { data } = await detectionAPI.uploadVideo(file, confidence, frameSkip);
      setResult(data);
      loadSessions();
    } catch (err) {
      console.error('Upload failed:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleImageUpload = async (file) => {
    setLoading(true);
    setImageResult(null);
    try {
      const { data } = await detectionAPI.uploadImage(file, confidence);
      setImageResult(data);
      loadSessions();
    } catch (err) {
      console.error('Upload failed:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8 animate-fade-in">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Video & Image Analysis</h1>
        <p className="text-yaqiz-muted">Upload media for PPE detection and compliance analysis</p>
      </div>

      {/* Tab Selector */}
      <div className="flex gap-2 mb-6">
        {[
          { id: 'video', label: 'Video Upload', icon: Video },
          { id: 'image', label: 'Image Upload', icon: Image },
        ].map(({ id, label, icon: TabIcon }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={`flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${
              tab === id
                ? 'bg-gradient-to-r from-yaqiz-accent/20 to-yaqiz-accent2/10 text-yaqiz-accent border border-yaqiz-accent/30'
                : 'text-yaqiz-muted hover:text-white hover:bg-white/5 border border-transparent'
            }`}
          >
            <TabIcon className="w-4 h-4" />
            {label}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Upload Panel */}
        <div className="glass-card p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Upload className="w-5 h-5 text-yaqiz-accent" />
            Upload {tab === 'video' ? 'Video' : 'Image'}
          </h3>

          <UploadZone
            type={tab}
            onUpload={tab === 'video' ? handleVideoUpload : handleImageUpload}
            loading={loading}
          />

          {/* Confidence Slider */}
          <div className="mt-5">
            <label className="text-sm text-yaqiz-muted mb-2 block">
              Detection Confidence: <span className="text-yaqiz-accent font-mono">{(confidence * 100).toFixed(0)}%</span>
            </label>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={confidence}
              onChange={(e) => setConfidence(parseFloat(e.target.value))}
              className="w-full accent-yaqiz-accent"
            />
          </div>

          {/* Frame Skip Slider (only for video) */}
          {tab === 'video' && (
            <div className="mt-4">
              <label className="text-sm text-yaqiz-muted mb-2 block">
                Frame Skip: <span className="text-yaqiz-accent font-mono">{frameSkip}x</span>
                <span className="text-xs text-yaqiz-muted ml-1">
                  ({frameSkip === 1 ? 'no skip — slowest' : `process 1 of every ${frameSkip} frames — ${frameSkip}x faster`})
                </span>
              </label>
              <input
                type="range"
                min="1"
                max="15"
                step="1"
                value={frameSkip}
                onChange={(e) => setFrameSkip(parseInt(e.target.value))}
                className="w-full accent-yaqiz-accent"
              />
              <div className="flex justify-between text-[10px] text-yaqiz-muted mt-1">
                <span>Accurate</span>
                <span>Balanced</span>
                <span>Fast</span>
              </div>
            </div>
          )}

          {/* Processing indicator */}
          {loading && (
            <div className="mt-5 p-4 bg-yaqiz-accent/5 border border-yaqiz-accent/20 rounded-xl animate-pulse">
              <div className="flex items-center gap-3">
                <Loader2 className="w-5 h-5 text-yaqiz-accent animate-spin" />
                <div>
                  <p className="text-sm text-white font-medium">Processing...</p>
                  <p className="text-xs text-yaqiz-muted">Running YOLO detection</p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Results Panel */}
        <div className="lg:col-span-2 glass-card p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-yaqiz-accent" />
            Analysis Results
          </h3>

          {/* Image Result */}
          {imageResult && tab === 'image' && (
            <div className="space-y-5 animate-slide-up">
              <div className="bg-black rounded-xl overflow-hidden">
                <img
                  src={imageResult.result_image}
                  alt="Detection Result"
                  className="w-full max-h-[500px] object-contain"
                />
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="bg-yaqiz-bg rounded-xl p-4 text-center">
                  <p className="text-2xl font-bold text-white">{imageResult.total_detections}</p>
                  <p className="text-xs text-yaqiz-muted mt-1">Detections</p>
                </div>
                <div className="bg-yaqiz-bg rounded-xl p-4 text-center">
                  <p className={`text-2xl font-bold ${imageResult.violations_count > 0 ? 'text-yaqiz-danger' : 'text-yaqiz-success'}`}>
                    {imageResult.violations_count}
                  </p>
                  <p className="text-xs text-yaqiz-muted mt-1">Violations</p>
                </div>
                <div className="bg-yaqiz-bg rounded-xl p-4 text-center">
                  <p className="text-2xl font-bold text-yaqiz-accent">{imageResult.workers_count}</p>
                  <p className="text-xs text-yaqiz-muted mt-1">Workers</p>
                </div>
                <div className="bg-yaqiz-bg rounded-xl p-4 text-center">
                  <p className="text-2xl font-bold text-yaqiz-success">{imageResult.helmet_compliance?.toFixed(0)}%</p>
                  <p className="text-xs text-yaqiz-muted mt-1">Helmet Comp.</p>
                </div>
              </div>

              {/* Detection list */}
              {imageResult.detections?.length > 0 && (
                <div>
                  <h4 className="text-sm font-semibold text-yaqiz-muted mb-3">Detected Objects</h4>
                  <div className="space-y-2 max-h-48 overflow-y-auto">
                    {imageResult.detections.map((d, i) => (
                      <div key={i} className={`flex items-center justify-between px-3 py-2 rounded-lg ${
                        d.is_violation ? 'bg-yaqiz-danger/10 border border-yaqiz-danger/20' : 'bg-yaqiz-success/10 border border-yaqiz-success/20'
                      }`}>
                        <span className="text-sm text-white">{d.class_name}</span>
                        <span className={`text-xs font-mono ${d.is_violation ? 'text-yaqiz-danger' : 'text-yaqiz-success'}`}>
                          {(d.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Video Result */}
          {result && tab === 'video' && (
            <div className="space-y-4 animate-slide-up">
              <div className={`flex items-center gap-3 px-4 py-3 rounded-xl ${
                result.status === 'pending' || result.status === 'processing'
                  ? 'bg-yaqiz-warning/10 border border-yaqiz-warning/20'
                  : result.status === 'completed'
                  ? 'bg-yaqiz-success/10 border border-yaqiz-success/20'
                  : 'bg-yaqiz-danger/10 border border-yaqiz-danger/20'
              }`}>
                {result.status === 'completed' ? (
                  <CheckCircle className="w-5 h-5 text-yaqiz-success" />
                ) : result.status === 'failed' ? (
                  <X className="w-5 h-5 text-yaqiz-danger" />
                ) : (
                  <Loader2 className="w-5 h-5 text-yaqiz-warning animate-spin" />
                )}
                <span className="text-sm text-white capitalize">Video {result.status}</span>
              </div>

              <p className="text-sm text-yaqiz-muted">
                Session #{result.id} — Processing will continue in the background.
                Check the dashboard for results.
              </p>
            </div>
          )}

          {/* Empty State */}
          {!imageResult && !result && (
            <div className="text-center py-16 text-yaqiz-muted">
              <Eye className="w-16 h-16 mx-auto mb-4 opacity-20" />
              <p className="text-lg">No analysis results yet</p>
              <p className="text-sm mt-1">Upload a {tab} to start detection</p>
            </div>
          )}
        </div>
      </div>

      {/* Session History */}
      {sessions.length > 0 && (
        <div className="glass-card p-6 mt-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Clock className="w-5 h-5 text-yaqiz-accent" />
            Session History
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-yaqiz-muted border-b border-yaqiz-border">
                  <th className="text-left py-3 px-4">ID</th>
                  <th className="text-left py-3 px-4">Type</th>
                  <th className="text-left py-3 px-4">Status</th>
                  <th className="text-left py-3 px-4">Detections</th>
                  <th className="text-left py-3 px-4">Violations</th>
                  <th className="text-left py-3 px-4">Compliance</th>
                  <th className="text-left py-3 px-4">Date</th>
                </tr>
              </thead>
              <tbody>
                {sessions.map((s) => (
                  <tr key={s.id} className="border-b border-yaqiz-border/30 hover:bg-white/2">
                    <td className="py-3 px-4 font-mono text-yaqiz-accent">#{s.id}</td>
                    <td className="py-3 px-4 capitalize">{s.session_type}</td>
                    <td className="py-3 px-4">
                      <span className={
                        s.status === 'completed' ? 'badge-success' :
                        s.status === 'failed' ? 'badge-danger' : 'badge-warning'
                      }>{s.status}</span>
                    </td>
                    <td className="py-3 px-4">{s.total_detections}</td>
                    <td className="py-3 px-4">
                      <span className={s.violations_count > 0 ? 'text-yaqiz-danger' : 'text-yaqiz-success'}>
                        {s.violations_count}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <span className={
                        s.compliance_rate >= 80 ? 'text-yaqiz-success' :
                        s.compliance_rate >= 50 ? 'text-yaqiz-warning' : 'text-yaqiz-danger'
                      }>{s.compliance_rate?.toFixed(1)}%</span>
                    </td>
                    <td className="py-3 px-4 text-yaqiz-muted text-xs">
                      {new Date(s.created_at).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
