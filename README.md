# SERENITY

Smart Emotion Recognition and Neural Intervention Technology (SERENITY) is a multimodal mental-health support platform that combines speech understanding, emotion sensing, clinical screening, and cloud-assisted therapeutic dialogue.

This README is intentionally written for first-time deployers and non-technical readers who want to set up the project on Raspberry Pi 5 from scratch.

---

## 1. Professional Introduction

SERENITY is an applied AI system designed to support structured, data-informed mental health conversations. It is not a diagnostic product. Instead, it provides:

- conversational support through an AI assistant,
- multimodal emotion sensing (voice and optional face frame),
- questionnaire-based monitoring (PHQ-9, GAD-7, PCL-5),
- professional-style risk and follow-up summaries in an admin view.

The architecture is designed for constrained hardware and practical field use, including edge devices such as Raspberry Pi 5.

---

## 2. Motivation

Mental health support systems often fail in one of two ways:

1. They are easy to use but not clinically structured.
2. They are clinically structured but difficult to deploy and operate.

SERENITY is designed to bridge that gap by combining:

- a user-friendly web interface,
- standard questionnaire interpretation,
- emotion-aware interaction,
- structured risk formulation and follow-up guidance,
- edge-compatible engineering choices.

The goal is not to replace clinicians. The goal is to provide continuity, observability, and early signal detection in supportive workflows.

---

## 3. Solution Overview

SERENITY has two major components:

1. Backend (FastAPI)
- Handles audio upload, optional image analysis, speech-to-text, emotion fusion, cloud LLM interaction, persistence, and admin analytics.

2. Frontend (React + Vite)
- Provides login, live interaction page, questionnaire workflows, and admin observatory dashboard.

Core pipeline:

1. User speaks into microphone.
2. Backend transcribes speech and predicts speech emotion.
3. Optional face frame is analyzed for facial emotion.
4. Results are fused into dominant emotion.
5. Text goes to cloud LLM for response generation.
6. Response streams back in near real time.
7. All relevant events are persisted and summarized for admin review.

---

## 4. Technical Implementation (High-Level)

### 4.1 Backend Layers

- API layer: FastAPI endpoints for auth, interaction, streaming, questionnaires, admin.
- Perception layer: STT + SER + FER.
- LLM layer: Async cloud client with fail-fast timeout, cooldown behavior, and stream safety filters.
- Data layer: SQLite with performance pragmas (WAL, memory-oriented settings).
- Clinical analytics layer: screening trends, distress signals, risk/protective factors, follow-up cadence.

### 4.2 Frontend Layers

- Interaction page with NDJSON stream consumer.
- Questionnaire page with scoring and history.
- Admin page with metrics dashboard, risk formulation, screening interpretation, and clinical summary.

### 4.3 Edge Optimization Strategy

- Reused model runtimes instead of loading model per request.
- Stream token deltas to reduce payload growth.
- LLM connection pooling and fallback URL support.
- Bounded cache for admin overview and summary.
- Reduced DB overhead with user-scoped counts and cached snapshots.

---

## 5. Technical Terminology (Plain-English Glossary)

- API: A software interface that lets one program communicate with another.
- FastAPI: A Python framework for building web APIs quickly.
- Endpoint: A specific URL path in the backend, such as /api/chat.
- NDJSON: Newline-delimited JSON; each line is one JSON event in a stream.
- Streaming response: Data sent in pieces over time instead of waiting for one final payload.
- STT (Speech-to-Text): Converting spoken audio to written text.
- Whisper / faster-whisper: Speech transcription backends.
- SER (Speech Emotion Recognition): Predicting emotional state from voice features.
- FER (Facial Emotion Recognition): Predicting emotional state from face image features.
- TFLite (TensorFlow Lite): Lightweight model runtime optimized for edge devices.
- Delegate (XNNPACK): CPU acceleration backend for faster inference.
- LLM (Large Language Model): Generative AI model used for assistant responses.
- Circuit breaker (cooldown): A resilience pattern that temporarily stops requests after repeated failures to avoid repeated long waits.
- Pooling: Reusing HTTP connections instead of recreating them each request.
- WAL (Write-Ahead Logging): SQLite mode that improves concurrency and performance.
- Risk formulation: Structured synthesis of risk indicators and protective factors.
- PHQ-9 / GAD-7 / PCL-5: Standard screening questionnaires for depression, anxiety, and trauma symptoms.
- Symptom burden: Combined estimate of current symptom intensity from screening scores.
- Protective factors: Indicators that may reduce immediate risk.
- Distress signals: Language patterns that may indicate increased acute burden.

---

## 6. Project Structure

- backend/main.py: API endpoints and orchestration.
- backend/cloud_llm_core.py: Cloud LLM client, streaming parser, resilience controls.
- backend/audio_core.py: Speech emotion inference runtime.
- backend/emotion_core.py: Face emotion inference runtime.
- backend/database.py: SQLite engine config, persistence helpers, query helpers.
- backend/models.py: ORM schema.
- backend/questionnaires_data.py: Questionnaire templates, scoring, flags.
- frontend/src/pages/UnifiedEmotionPage.jsx: Live interaction UX.
- frontend/src/pages/QuestionnairesPage.jsx: Screening workflow UX.
- frontend/src/pages/AdminPage.jsx: Clinical observability UX.

### 6.1 What Is Tracked In GitHub (And What Is Not)

Tracked in repository:

- source code,
- requirement files,
- startup/config scripts,
- frontend and backend application files.

Not tracked in repository:

- local virtual environments such as .venv,
- system packages installed via apt,
- machine-specific caches and compiled binary wheels.

This is expected behavior. Anyone cloning the project must create a fresh local environment and install dependencies on their own device.

---

## 7. Complete Raspberry Pi 5 Deployment Guide (From Scratch)

This section assumes you are starting from zero.

### 7.1 Hardware and OS Prerequisites

Required:

- Raspberry Pi 5 (recommended 8GB RAM).
- 64-bit Raspberry Pi OS Bookworm.
- MicroSD or SSD storage with enough free space (at least 16GB recommended).
- Stable internet.
- USB microphone.
- Optional USB camera for FER.

Strongly recommended:

- Wired network during setup.
- Active cooling for sustained inference load.

### 7.2 First Boot Setup

On the Pi terminal:

```bash
sudo apt update
sudo apt full-upgrade -y
sudo reboot
```

After reboot:

```bash
sudo apt install -y git curl build-essential pkg-config ffmpeg libsndfile1 libatlas-base-dev libopenblas-dev liblapack-dev libglib2.0-0 libgl1
```

### 7.3 Install Python Tooling

Install Python 3.11 and venv tooling (recommended for edge package compatibility):

```bash
sudo apt install -y python3.11 python3.11-venv python3-pip
python3.11 --version
```

Important:

- Supported runtime for this project: Python 3.10 to 3.12.
- Recommended runtime on Raspberry Pi 5: Python 3.11.
- Python 3.13 is currently unsupported for this stack due binary wheel compatibility (NumPy/OpenCV/TFLite/TensorFlow ecosystem constraints).
- tflite-runtime wheels are often unavailable for some Python and architecture combinations.

### 7.4 Install Node.js (Frontend)

Option A (simple apt path):

```bash
sudo apt install -y nodejs npm
node -v
npm -v
```

Option B (if apt Node is too old): install Node 20 using NodeSource.

### 7.5 Clone the Repository

```bash
git clone https://github.com/mtahaarif/Smart-Emotion-Recognition-and-Neural-Intervention-Technology-SERENITY-.git
cd Smart-Emotion-Recognition-and-Neural-Intervention-Technology-SERENITY-
```

### 7.5.1 Pull Git LFS Model Files (Required)

This repository stores `.tflite` model files with Git LFS. If Git LFS is not installed or pulled, model files may not load at runtime.

```bash
git lfs install
git lfs pull
```

### 7.6 Create and Activate Virtual Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 7.7 Install Backend Dependencies on Pi

Recommended edge install:

```bash
pip install --extra-index-url https://www.piwheels.org/simple -r requirements-edge.txt
```

If that succeeds, continue to section 7.8.

Optional automation:

```bash
chmod +x scripts/setup_rpi5.sh
./scripts/setup_rpi5.sh
```

### 7.8 If You Get "No matching distribution found for tflite-runtime"

This is common on incompatible Python or wheel combinations.

Run these checks:

```bash
python --version
uname -m
```

Expected for best compatibility:

- Python 3.10 or 3.11
- aarch64 (64-bit)

If you are on Python 3.13, recreate venv with Python 3.11 and retry:

```bash
deactivate 2>/dev/null || true
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Fallback path (use TensorFlow Lite via tensorflow package):

```bash
grep -v '^tflite-runtime' requirements-edge.txt > requirements-edge-no-tflite.txt
pip install --extra-index-url https://www.piwheels.org/simple -r requirements-edge-no-tflite.txt
pip install tensorflow==2.18.0
```

Why fallback works:

- Backend imports tflite_runtime first, and if unavailable it falls back to tensorflow.lite.

### 7.9 Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### 7.10 Run Backend (Development)

```bash
source .venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 5000
```

### 7.11 Run Frontend (Development)

Open another terminal:

```bash
cd frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

Then open in browser:

- http://PI_IP_ADDRESS:5173/login

### 7.12 Production Deployment (Recommended)

#### Step A: Build frontend static assets

```bash
cd frontend
npm run build
cd ..
```

#### Step B: Create backend systemd service

Create file /etc/systemd/system/serenity-backend.service:

```ini
[Unit]
Description=SERENITY FastAPI Backend
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/Smart-Emotion-Recognition-and-Neural-Intervention-Technology-SERENITY-/FYP
Environment="PATH=/home/pi/Smart-Emotion-Recognition-and-Neural-Intervention-Technology-SERENITY-/FYP/.venv/bin"
ExecStart=/home/pi/Smart-Emotion-Recognition-and-Neural-Intervention-Technology-SERENITY-/FYP/.venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 5000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable serenity-backend
sudo systemctl start serenity-backend
sudo systemctl status serenity-backend
```

#### Step C: Serve frontend with Nginx

Install nginx:

```bash
sudo apt install -y nginx
```

Create /etc/nginx/sites-available/serenity:

```nginx
server {
    listen 80;
    server_name _;

    root /home/pi/Smart-Emotion-Recognition-and-Neural-Intervention-Technology-SERENITY-/FYP/frontend/dist;
    index index.html;

    location / {
        try_files $uri /index.html;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable site:

```bash
sudo ln -s /etc/nginx/sites-available/serenity /etc/nginx/sites-enabled/serenity
sudo nginx -t
sudo systemctl restart nginx
```

Now open:

- http://PI_IP_ADDRESS/login

### 7.13 Configure Frontend API URL for Production

Before building frontend, set:

frontend/.env.production

```env
VITE_API_BASE_URL=/api
```

Then rebuild:

```bash
cd frontend
npm run build
cd ..
```

### 7.14 Windows Local Quick Start (Developer)

If you are running on Windows for local development:

1. Open PowerShell in the repository root (`FYP`).
2. Create and activate a local venv inside `FYP` (recommended):

```powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

3. Install frontend dependencies:

```powershell
cd frontend
npm install
cd ..
```

4. Start both services:

```powershell
.\Start_App.bat
```

Note: `Start_App.bat` first checks `FYP\.venv\Scripts\python.exe` and now also falls back to `..\.venv\Scripts\python.exe` when available.

---

## 8. Environment Variables

Common variables for tuning and deployment:

- SERENITY_CLOUD_LLM_URL
- SERENITY_CLOUD_LLM_FALLBACK_URLS
- SERENITY_CLOUD_LLM_TIMEOUT_SECONDS
- SERENITY_CLOUD_LLM_CONNECT_TIMEOUT_SECONDS
- SERENITY_CLOUD_LLM_FAILURE_THRESHOLD
- SERENITY_CLOUD_LLM_COOLDOWN_SECONDS
- SERENITY_TTS_ENABLED
- SERENITY_TTS_VOICE
- SERENITY_PREWARM_MODELS
- SERENITY_PREWARM_WHISPER
- SERENITY_WHISPER_MODEL_SIZE
- SERENITY_WHISPER_CPU_THREADS
- SERENITY_WHISPER_TIMEOUT_SECONDS
- SERENITY_EMOTION_TIMEOUT_SECONDS
- SERENITY_LLM_TIMEOUT_SECONDS
- SERENITY_ADMIN_OVERVIEW_CACHE_TTL_SECONDS
- SERENITY_ADMIN_SUMMARY_CACHE_TTL_SECONDS
- SERENITY_ADMIN_SUMMARY_TIMEOUT_SECONDS

Example shell exports:

```bash
export SERENITY_CLOUD_LLM_URL="http://YOUR_EC2_HOST:8000/chat"
export SERENITY_TTS_ENABLED="true"
export SERENITY_PREWARM_MODELS="true"
```

---

## 9. API Overview

Primary backend endpoints:

- POST /register
- POST /login
- POST /api/interact
- POST /api/interact/stream
- POST /api/chat
- POST /api/chat/stream
- GET /api/questionnaires/templates
- POST /api/questionnaires/submit
- GET /api/questionnaires/history
- GET /api/admin/overview
- GET /api/admin/summary/stream

---

## 10. Validation Checklist After Deployment

Run these checks in order:

1. Backend health smoke test:

```bash
curl -X POST http://127.0.0.1:5000/login -H "Content-Type: application/json" -d '{"username":"test","password":"test"}'
```

2. Open frontend in browser and verify login page loads.
3. Test text chat first (/api/chat).
4. Test voice interaction with microphone.
5. Test optional camera mode.
6. Submit one questionnaire and verify history.
7. Open admin page and verify:
- Metrics Dashboard,
- Risk Formulation,
- Screening interpretations,
- Summary generation.

---

## 11. Troubleshooting Guide

### 11.1 tflite-runtime Not Found

Symptom:

- ERROR: No matching distribution found for tflite-runtime

Fix:

1. Use Python 3.11 venv.
2. Use 64-bit Raspberry Pi OS.
3. Use piwheels index URL.
4. If still failing, install without tflite-runtime and add tensorflow==2.18.0 fallback.

### 11.2 NumPy Version Compatibility Errors

Symptom:

- pip says some numpy versions require different Python.
- Runtime errors such as:
    - A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
    - AttributeError: _ARRAY_API not found
    - ImportError: numpy.core.multiarray failed to import

Fix:

- Ensure you are not using Python 3.13 for this project.
- Recreate a clean Python 3.11 virtual environment.
- Reinstall from requirements-edge.txt using piwheels index.
- Do not mix incompatible wheel generations from previous failed installs.

### 11.3 Backend Starts But Frontend Cannot Reach API

Fix:

- Ensure backend host is 0.0.0.0.
- Ensure VITE_API_BASE_URL is correct.
- If using nginx, ensure /api is proxied to 127.0.0.1:5000.

### 11.4 Camera or Microphone Not Working

Fix:

- Check browser permissions.
- Test with Chromium on Pi.
- Confirm USB device visibility with arecord -l and v4l2-ctl --list-devices.

### 11.5 Slow Inference on Pi

Fix:

- Use edge dependency profile.
- Keep FER optional if camera not needed.
- Reduce concurrent load.
- Keep cooling active to avoid thermal throttling.

---

## 12. Security, Privacy, and Clinical Safety Notes

- SERENITY is a support and observability system, not a diagnostic authority.
- Distress keyword detection is heuristic and can produce false positives/negatives.
- Always align usage with local privacy regulations.
- Restrict CORS and enforce proper authentication before internet exposure.
- For high-risk contexts, integrate mandatory human escalation workflows.

---

## 13. Operational Recommendations for Raspberry Pi 5

- Keep swap enabled but monitor memory pressure.
- Use SSD storage if available for better IO consistency.
- Prefer wired ethernet for stable cloud LLM latency.
- Schedule regular updates:

```bash
sudo apt update && sudo apt upgrade -y
```

- Monitor service logs:

```bash
sudo journalctl -u serenity-backend -f
```

---

## 14. Final Notes for First-Time Deployers

If you are new to deployment, follow this order:

1. Get backend running first.
2. Test API with simple text request.
3. Start frontend and verify login and chat.
4. Add voice and camera testing.
5. Configure production services only after development flow works.

This approach isolates problems and avoids debugging everything at once.

---

## 15. License and Third-Party Components

Use of this repository must follow:

- repository license terms,
- third-party package licenses,
- model and service licenses,
- applicable health, privacy, and data governance requirements in your region.
