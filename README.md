<div align="center">

<img src="static/images/logo.png" alt="YAQIZ Logo" width="120" />

# YAQIZ — يقظ

### AI-Powered Workplace Safety & Fatigue Monitoring Platform

Real-time PPE compliance detection • Workstation fatigue tracking • Instant Telegram alerts

[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6F00?style=for-the-badge)](https://ultralytics.com)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Google-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev)
[![TailwindCSS](https://img.shields.io/badge/Tailwind-3.4-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)](https://tailwindcss.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 📋 Overview

**YAQIZ** (يقظ — Arabic for "Vigilant") is a full-stack AI platform that combines **PPE detection** with **workstation fatigue monitoring** to create a unified workplace safety system. It uses YOLOv8 for real-time PPE compliance and MediaPipe for fatigue/attention tracking — all wrapped in a modern React dashboard with WebSocket live updates and Telegram notifications.

### Two Detection Engines, One Unified Platform

| 🔧 PPE Compliance (YOLOv8) | 🧠 Fatigue Monitoring (MediaPipe) |
|---|---|
| Hardhat / NO-Hardhat | Eye closure detection (EAR) |
| Safety Vest / NO-Safety Vest | Yawn detection (MAR) |
| Mask / NO-Mask | Head pose tracking (yaw/pitch/roll) |
| Person, Machinery, Vehicle, Cone | Head drop detection |
| Worker tracking (ByteTrack) | Eye rub & hand-on-head detection |
| Live camera + video + image analysis | Keyboard & mouse activity tracking |

---

## ✨ Features

### 🎯 PPE Detection Engine
- **YOLOv8 model** with 10 detection classes and configurable confidence threshold
- **GPU acceleration** — auto-detects CUDA with FP16 half-precision inference
- **ByteTrack object tracking** — persistent worker IDs across video frames
- **Frame skipping** — configurable (1–30) for performance tuning
- **Three input modes**: live webcam, video upload, image upload
- **Annotated output** — bounding boxes, labels, compliance overlay, track IDs
- **Per-frame metrics**: helmet/vest/mask compliance %, worker count, violations

### 🧠 Workstation Fatigue Monitoring
- **MediaPipe FaceMesh** (468 landmarks) + **Hand Tracking** for fatigue analysis
- **Eye Aspect Ratio (EAR)** — blink detection & eye closure alerts
- **Mouth Aspect Ratio (MAR)** — yawn detection & counting
- **Head pose estimation** — solvePnP-based yaw/pitch/roll classification (forward, left, right, up, down)
- **Head drop detection** — detects nodding off with immediate alerts
- **Eye rub detection** — hand-to-eye proximity with debouncing
- **Absence detection** — alerts when user leaves for configurable threshold
- **Composite scoring** — fatigue score (0–100) and attention score (0–100)
- **System activity tracking** — keyboard CPM, mouse scroll SPM, idle detection via `pynput`
- **Active window tracking** — reads foreground window title

### 📊 Executive Dashboard
- Aggregated stats: total sessions, detections, violations, compliance rate
- Session history with status indicators (video, image, live, workstation)
- Recent alerts feed with severity indicators
- Auto-refresh every 30 seconds

### 📹 Live Monitoring
- Real-time WebSocket camera feed with YOLO overlay
- Adjustable confidence threshold slider
- Live worker tracking panel with individual compliance status
- Real-time violation alerts sidebar
- MJPEG fallback for older browsers

### 📁 Video & Image Analysis
- Drag-and-drop upload zone
- Background video processing with real-time progress tracking via WebSocket
- Annotated results with detection details
- Session history table with downloadable results

### 🖥️ Workstation Monitoring Page
- Webcam-based fatigue monitoring via WebSocket
- Circular gauges for fatigue and attention scores
- EAR/MAR progress bars
- Head pose indicator (yaw/pitch/roll visualization)
- Metric counters: blinks, yawns, eye rubs, presence, typing CPM, scroll SPM
- Live alert feed

### 🚨 Alerts Center
- Unified alert feed (PPE violations + workstation fatigue alerts)
- Severity-based cards: critical, high, warning, info
- Search and filter capabilities
- Mark read / mark all read
- Real-time WebSocket updates

### 📲 Telegram Notifications
- **PPE violation alerts** — sent during live monitoring (rate-limited 10s cooldown)
- **Video analysis summaries** — compliance report on processing completion
- **Workstation fatigue alerts** — critical fatigue/attention events
- Arabic-formatted messages with emoji
- Configurable via environment variables

### 🔐 Authentication & Security
- JWT-based authentication (HS256, 24-hour expiry)
- Secure password hashing via bcrypt
- Protected routes on both frontend and backend
- Role-based user model

---

## 🏗️ Architecture

```
YAQIZ/
├── backend/                      # FastAPI + YOLO + MediaPipe
│   ├── app/
│   │   ├── core/                 # Config, Database, Security (JWT)
│   │   ├── models/               # SQLAlchemy models + Pydantic schemas
│   │   │   ├── user.py           # User model (roles, auth)
│   │   │   ├── detection.py      # DetectionSession + Alert models
│   │   │   ├── workstation.py    # WorkstationSession model
│   │   │   └── schemas.py        # Request/response schemas
│   │   ├── routers/
│   │   │   ├── auth.py           # Register, Login, Profile
│   │   │   ├── detection.py      # Upload, Process, Stream, Workers
│   │   │   ├── dashboard.py      # Stats, Alerts management
│   │   │   ├── websocket.py      # Live feed, Alerts, Progress
│   │   │   └── workstation.py    # Fatigue monitoring WebSocket
│   │   ├── services/
│   │   │   ├── detection_service.py      # YOLO wrapper + compliance logic
│   │   │   ├── websocket_manager.py      # 4-channel WS connection manager
│   │   │   ├── telegram_service.py       # Telegram Bot API integration
│   │   │   ├── worker_tracker.py         # ByteTrack worker tracking
│   │   │   ├── workstation_service.py    # MediaPipe fatigue processing
│   │   │   ├── workstation_metrics.py    # Fatigue score computation
│   │   │   └── workstation_alert_bridge.py # Unified alert pipeline
│   │   └── utils/
│   │       └── logger.py         # Structured logging
│   ├── main.py                   # FastAPI app entry point
│   ├── Dockerfile
│   └── requirements.txt
│
├── detection_model/              # Standalone detection modules
│   ├── detection_core.py         # MediaPipe face/hand detection
│   ├── fatigue_detector.py       # Desktop fatigue GUI + alarms
│   └── activity_tracker.py       # Keyboard/mouse/idle tracking
│
├── frontend/                     # React 18 + Vite 5 + Tailwind
│   ├── src/
│   │   ├── components/Layout.jsx # Sidebar navigation, page transitions
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx     # Executive stats overview
│   │   │   ├── LiveMonitoring.jsx
│   │   │   ├── VideoAnalysis.jsx
│   │   │   ├── AlertsCenter.jsx
│   │   │   ├── WorkstationMonitoring.jsx
│   │   │   ├── Login.jsx
│   │   │   └── Register.jsx
│   │   ├── services/api.js       # Axios client + interceptors
│   │   └── hooks/useWebSocket.js # WebSocket + auth hooks
│   ├── Dockerfile
│   ├── nginx.conf
│   └── tailwind.config.js
│
├── docker-compose.yml            # Full-stack Docker deployment
├── YOLO-Weights/                 # Pre-trained model weights
└── Videos/                       # Sample test videos
```

---

## ⚡ Quick Start

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| Node.js | 18+ |
| Git | Latest |
| GPU (optional) | CUDA-compatible for acceleration |

### 1. Clone

```bash
git clone https://github.com/Demro7/YAQIZ.git
cd YAQIZ
```

### 2. Backend

```bash
cd backend

# Create & activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env         # Edit with your settings

# Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

> **API**: http://localhost:8000 — **Swagger Docs**: http://localhost:8000/docs

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

> **Dashboard**: http://localhost:5173

### 4. Get Started

1. Open http://localhost:5173
2. Click **Create Account** and register
3. Log in and explore the dashboard
4. Navigate to **Live Monitoring** for real-time PPE detection
5. Navigate to **Workstation** for fatigue monitoring

---

## 🐳 Docker Deployment

```bash
# Build and run the full stack
docker-compose up --build -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f backend
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Health Check | http://localhost:8000/health |

---

## 🔑 Environment Variables

Create `backend/.env`:

```env
# Security
SECRET_KEY=your-secret-key-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Database
DATABASE_URL=sqlite:///./yaqiz.db

# YOLO
YOLO_WEIGHTS_PATH=../YOLO-Weights/ppe.pt
YOLO_CONFIDENCE=0.5
YOLO_FRAME_SKIP=3

# CORS
CORS_ORIGINS=http://localhost:5173,http://localhost:3000

# Telegram (optional)
TELEGRAM_BOT_TOKEN=your-bot-token
TELEGRAM_CHAT_ID=your-chat-id
TELEGRAM_ENABLED=false

# Upload Limits
MAX_UPLOAD_SIZE_MB=500
```

---

## 📡 API Reference

### Authentication

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/auth/register` | Create new account |
| `POST` | `/api/auth/login` | Login → returns JWT |
| `GET`  | `/api/auth/me` | Get current user profile |

### PPE Detection

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/detection/upload-video` | Upload & process video (background) |
| `POST` | `/api/detection/upload-image` | Upload & analyze image (instant) |
| `GET`  | `/api/detection/sessions` | List detection sessions (paginated) |
| `GET`  | `/api/detection/sessions/{id}` | Get session details + summary |
| `GET`  | `/api/detection/result/{filename}` | Serve annotated result files |
| `GET`  | `/api/detection/live-feed` | MJPEG webcam stream with detections |
| `GET`  | `/api/detection/workers` | Get tracked workers from live session |
| `GET`  | `/api/detection/workers/{id}` | Get individual worker violation log |

### Dashboard

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/dashboard/stats` | Aggregated platform statistics |
| `GET` | `/api/dashboard/alerts` | Get alerts (filterable by severity) |
| `PUT` | `/api/dashboard/alerts/{id}/read` | Mark alert as read |
| `PUT` | `/api/dashboard/alerts/mark-all-read` | Mark all alerts as read |

### Workstation

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/workstation/health` | Check MediaPipe availability |
| `GET` | `/api/workstation/sessions` | List workstation sessions |

### WebSocket Channels

| Endpoint | Protocol | Purpose |
|---|---|---|
| `/ws/live` | WebSocket | Live camera PPE detection + worker tracking |
| `/ws/alerts` | WebSocket | Real-time alert notifications (PPE + workstation) |
| `/ws/processing` | WebSocket | Video processing progress updates |
| `/ws/workstation` | WebSocket | Fatigue monitoring (send frames, receive metrics) |

---

## 🗄️ Database Schema

| Table | Key Fields |
|---|---|
| `users` | id, username, email, hashed_password, role, is_active |
| `detection_sessions` | id, session_type (video/image/live/workstation), source_file, result_file, total_frames, violations_count, compliance_rate, status, summary (JSON) |
| `alerts` | id, alert_type, severity (critical/high/warning/info), message, confidence, frame_number, session_id (FK), is_read |
| `workstation_sessions` | id, user_id, status, total_frames, avg_fatigue_score, avg_attention_score, peak_fatigue_score, total_yawns, total_blinks, total_eye_rubs, total_alerts |

---

## 🚨 Alert Types

### PPE Alerts (via YOLO)

| Alert Type | Severity | Trigger |
|---|---|---|
| `no_hardhat` | critical | Worker detected without hardhat |
| `no_safety_vest` | high | Worker detected without safety vest |
| `no_mask` | warning | Worker detected without mask |

### Workstation Alerts (via MediaPipe)

| Alert Type | Severity | Trigger |
|---|---|---|
| `workstation_eye_closure` | critical | Eyes closed beyond threshold |
| `workstation_head_drop` | critical | Head dropped (nodding off) |
| `workstation_absence` | critical | User absent beyond threshold |
| `workstation_high_fatigue` | critical | Fatigue score ≥ 80 |
| `workstation_yawn` | warning | Yawn count exceeds limit |
| `workstation_head_pose` | warning | Not looking forward beyond threshold |
| `workstation_eye_rub` | warning | Eye rub count exceeds limit |
| `workstation_hand_head` | warning | Hand on head beyond threshold |
| `workstation_idle` | info | User idle beyond threshold |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.10+, FastAPI, Uvicorn, SQLAlchemy, Pydantic |
| **AI / CV** | YOLOv8 (Ultralytics), MediaPipe, OpenCV, PyTorch |
| **Frontend** | React 18, Vite 5, TailwindCSS 3.4, Recharts, Lucide Icons |
| **Real-time** | WebSocket (4 channels), MJPEG streaming |
| **Auth** | JWT (python-jose), bcrypt password hashing |
| **Notifications** | Telegram Bot API (httpx async) |
| **Activity Tracking** | pynput (keyboard/mouse), ctypes (active window) |
| **Database** | SQLite via SQLAlchemy ORM |
| **Deployment** | Docker, Docker Compose, Nginx reverse proxy |

---

## ☁️ Cloud Deployment

### Google Cloud Run (CPU)

```bash
# Backend
cd backend
gcloud builds submit --tag gcr.io/PROJECT_ID/yaqiz-backend
gcloud run deploy yaqiz-backend \
  --image gcr.io/PROJECT_ID/yaqiz-backend \
  --platform managed --memory 4Gi --cpu 2 \
  --allow-unauthenticated

# Frontend
cd ../frontend
gcloud builds submit --tag gcr.io/PROJECT_ID/yaqiz-frontend
gcloud run deploy yaqiz-frontend \
  --image gcr.io/PROJECT_ID/yaqiz-frontend \
  --platform managed --allow-unauthenticated
```

### Compute Engine (GPU — Production)

1. Create a VM with GPU (T4 or better)
2. Install NVIDIA drivers + Docker
3. Clone repo and run `docker-compose up -d`
4. Configure firewall rules for ports 3000, 8000

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built for workplace safety**

**YAQIZ — يقظ — Vigilant AI for Safety Compliance**

</div>




