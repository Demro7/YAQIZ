<div align="center">

# 🛡️ YAQIZ — AI-Powered PPE Compliance Platform

**Real-time Personal Protective Equipment detection & safety compliance monitoring**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev)
[![YOLO](https://img.shields.io/badge/YOLOv8-ultralytics-FF6F00)](https://ultralytics.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📋 Overview

YAQIZ transforms PPE detection from a simple script into a **production-grade AI safety platform**. It wraps a YOLOv8 model with a modern full-stack application featuring real-time monitoring, video analysis, alerting, and executive dashboards.

### What It Detects

| ✅ Equipment Present | ❌ Violation Detected |
|---|---|
| Hardhat | NO-Hardhat |
| Safety Vest | NO-Safety Vest |
| Mask | NO-Mask |
| Person, Safety Cone, Machinery, Vehicle | — |

---

## 🏗️ Architecture

```
YAQIZ/
├── backend/                  # FastAPI + YOLO inference
│   ├── app/
│   │   ├── core/             # Config, DB, Security (JWT)
│   │   ├── models/           # SQLAlchemy models + Pydantic schemas
│   │   ├── routers/          # Auth, Detection, Dashboard, WebSocket
│   │   ├── services/         # DetectionService, WebSocketManager
│   │   └── utils/            # Logging
│   ├── main.py               # FastAPI app entry point
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                 # React + Vite + Tailwind
│   ├── src/
│   │   ├── components/       # Layout, shared UI
│   │   ├── pages/            # Dashboard, LiveMonitoring, VideoAnalysis, Alerts
│   │   ├── services/         # Axios API client
│   │   └── hooks/            # useWebSocket, useAuth
│   ├── Dockerfile
│   └── nginx.conf
├── YOLO-Weights/             # Pre-trained model weights
├── docker-compose.yml
└── README.md
```

---

## ⚡ Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **Git**

### 1. Clone & Setup

```bash
git clone <repo-url>
cd PPE_detection_YOLO-main
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/Mac)
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API is now live at **http://localhost:8000**  
Swagger docs at **http://localhost:8000/docs**

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

The dashboard is now live at **http://localhost:5173**

### 4. Create Your Account

1. Open **http://localhost:5173**
2. Click **Create Account**
3. Register with username, email, and password
4. Log in and start monitoring!

---

## 🐳 Docker Deployment

```bash
# Build and run everything
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
CONFIDENCE_THRESHOLD=0.5

# CORS
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

---

## 📡 API Reference

### Authentication
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/auth/register` | Create new account |
| POST | `/api/auth/login` | Login (returns JWT) |
| GET | `/api/auth/me` | Get current user profile |

### Detection
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/detection/upload-video` | Upload & process video |
| POST | `/api/detection/upload-image` | Upload & analyze image |
| GET | `/api/detection/sessions` | List detection sessions |
| GET | `/api/detection/sessions/{id}` | Get session details |
| GET | `/api/detection/live-feed` | MJPEG live camera stream |

### Dashboard
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/dashboard/stats` | Aggregated platform stats |
| GET | `/api/dashboard/alerts` | Get alerts (filterable) |
| PUT | `/api/dashboard/alerts/{id}/read` | Mark alert as read |
| PUT | `/api/dashboard/alerts/mark-all-read` | Mark all alerts read |

### WebSocket
| Endpoint | Description |
|---|---|
| `ws://host/ws/live` | Real-time camera detections |
| `ws://host/ws/alerts` | Live alert notifications |
| `ws://host/ws/processing` | Video processing progress |

---

## 🖥️ Features

### Executive Dashboard
- Total detections, violations, compliance rate
- Session history with status indicators
- Recent alerts feed
- Auto-refresh every 30 seconds

### Live Monitoring
- Real-time camera feed via WebSocket
- Adjustable confidence threshold
- Worker count, helmet/vest/mask compliance meters
- Live violation alerts sidebar
- MJPEG fallback for older browsers

### Video & Image Analysis
- Drag-and-drop upload zone
- Background video processing with progress tracking
- Annotated image results with detection details
- Session history table

### Alerts Center
- Severity-based alert cards (critical, high, medium, low)
- Search and filter capabilities
- Mark read / mark all read
- Summary counts by severity

---

## ☁️ Google Cloud Deployment

### Cloud Run (CPU — recommended for testing)

```bash
# Build and push backend
cd backend
gcloud builds submit --tag gcr.io/PROJECT_ID/yaqiz-backend
gcloud run deploy yaqiz-backend \
  --image gcr.io/PROJECT_ID/yaqiz-backend \
  --platform managed \
  --memory 4Gi \
  --cpu 2 \
  --allow-unauthenticated

# Build and push frontend
cd ../frontend
gcloud builds submit --tag gcr.io/PROJECT_ID/yaqiz-frontend
gcloud run deploy yaqiz-frontend \
  --image gcr.io/PROJECT_ID/yaqiz-frontend \
  --platform managed \
  --allow-unauthenticated
```

### Compute Engine (GPU — recommended for production)

1. Create a VM with GPU (T4 or better)
2. Install NVIDIA drivers + Docker
3. Clone repo and run `docker-compose up -d`
4. Configure firewall rules for ports 3000, 8000

---

## 🧪 Project Structure Details

```
backend/
├── app/
│   ├── core/
│   │   ├── config.py          # Pydantic Settings — all env vars
│   │   ├── database.py        # SQLAlchemy engine, sessions
│   │   └── security.py        # JWT encode/decode, password hashing
│   ├── models/
│   │   ├── user.py            # User model (roles, auth)
│   │   ├── detection.py       # DetectionSession + Alert models
│   │   └── schemas.py         # All Pydantic request/response schemas
│   ├── routers/
│   │   ├── auth.py            # Register, Login, Profile
│   │   ├── detection.py       # Upload, Process, Stream
│   │   ├── dashboard.py       # Stats, Alerts
│   │   └── websocket.py       # Live feed, Alerts, Progress channels
│   ├── services/
│   │   ├── detection_service.py  # YOLO wrapper, compliance logic
│   │   └── websocket_manager.py  # Connection manager, broadcasting
│   └── utils/
│       └── logger.py          # Structured logging
├── main.py                    # App factory, lifespan, middleware
└── requirements.txt

frontend/
├── src/
│   ├── components/
│   │   └── Layout.jsx         # Sidebar nav, user panel
│   ├── pages/
│   │   ├── Dashboard.jsx      # Executive stats overview
│   │   ├── LiveMonitoring.jsx # Real-time camera + detections
│   │   ├── VideoAnalysis.jsx  # Upload & analyze media
│   │   ├── AlertsCenter.jsx   # Alert management
│   │   ├── Login.jsx          # Authentication
│   │   └── Register.jsx       # Account creation
│   ├── services/
│   │   └── api.js             # Axios client + interceptors
│   ├── hooks/
│   │   └── useWebSocket.js    # WS hook + useAuth hook
│   ├── App.jsx                # Router + protected routes
│   ├── main.jsx               # React entry point
│   └── index.css              # Tailwind + custom components
├── index.html
├── vite.config.js
├── tailwind.config.js
└── postcss.config.js
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ for workplace safety**

YAQIZ — يقظ — *Vigilant AI for PPE Compliance*

</div>
