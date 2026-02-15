import axios from 'axios';

const API_BASE = '/api';

const api = axios.create({
  baseURL: API_BASE,
  headers: { 'Content-Type': 'application/json' },
});

// Attach JWT token to every request
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('yaqiz_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle 401 errors
api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('yaqiz_token');
      localStorage.removeItem('yaqiz_user');
      window.location.href = '/login';
    }
    return Promise.reject(err);
  }
);

// ── Auth ───────────────────────────────────────────────
export const authAPI = {
  register: (data) => api.post('/auth/register', data),
  login: (data) => api.post('/auth/login', data),
  getMe: () => api.get('/auth/me'),
  logout: () => api.post('/auth/logout'),
};

// ── Detection ──────────────────────────────────────────
export const detectionAPI = {
  uploadVideo: (file, confidence = 0.5, frameSkip = 3) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post(`/detection/upload-video?confidence=${confidence}&frame_skip=${frameSkip}`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  uploadImage: (file, confidence = 0.5) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post(`/detection/upload-image?confidence=${confidence}`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  getSessions: (skip = 0, limit = 20) => api.get(`/detection/sessions?skip=${skip}&limit=${limit}`),
  getSession: (id) => api.get(`/detection/sessions/${id}`),
  getResultUrl: (filename) => `/api/detection/result/${filename}`,
  getLiveFeedUrl: (confidence = 0.5, frameSkip = 2) => `/api/detection/live-feed?confidence=${confidence}&frame_skip=${frameSkip}`,
  getTrackedWorkers: () => api.get('/detection/workers'),
  getWorkerLog: (id) => api.get(`/detection/workers/${id}`),
};

// ── Workstation ────────────────────────────────────────
export const workstationAPI = {
  health: () => api.get('/workstation/health'),
  getSessions: (skip = 0, limit = 20) => api.get(`/workstation/sessions?skip=${skip}&limit=${limit}`),
};

// ── Dashboard ──────────────────────────────────────────
export const dashboardAPI = {
  getStats: () => api.get('/dashboard/stats'),
  getAlerts: (params = {}) => api.get('/dashboard/alerts', { params }),
  markAlertRead: (id) => api.put(`/dashboard/alerts/${id}/read`),
  markAllRead: () => api.put('/dashboard/alerts/mark-all-read'),
};

export default api;
