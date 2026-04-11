# SERENITY

Smart Emotion Recognition and Neural Intervention Technology (SERENITY) is a multimodal AI support assistant built for cloud-hybrid operation with edge-first optimizations (including Raspberry Pi 5).

## Current Implementation

- FastAPI backend with streaming NDJSON responses.
- React + Vite frontend with route-level lazy loading.
- Voice + optional camera emotional analysis.
- Whisper transcription (faster-whisper preferred).
- Cloud LLM integration with stream and fallback handling.
- Sentence-level Edge TTS support.
- Local SQLite persistence for users, chat turns, sessions/emotions, and questionnaire results.
- Admin analytics page and questionnaires workflow (PHQ-9, GAD-7, PCL-5).

## Frontend Routes

- /login
- /dashboard
- /emotion/live
- /questionnaires
- /admin

## Backend API Surface

### Auth

- POST /register
- POST /login

### Health

- GET /health

### Voice + Vision Interaction

- POST /api/interact
- POST /api/interact/stream

### Text Chat

- POST /api/chat
- POST /api/chat/stream

### Questionnaires

- GET /api/questionnaires/templates
- POST /api/questionnaires/submit
- GET /api/questionnaires/history

### Admin

- GET /api/admin/overview

## Streaming Event Contract

The frontend consumes one JSON object per line from stream endpoints:

- transcription
- user_text
- emotion_partial
- emotion
- assistant_delta
- assistant_sentence
- assistant_sentence_tts
- assistant_replace
- assistant_tts_reset
- assistant_tts_trim
- error
- final

## Database Tables

- users
- conversation_turns
- sessions
- emotions
- questionnaire_results

SQLite is tuned with WAL mode, NORMAL sync, memory temp store, and configurable cache size.

## Raspberry Pi 5 Optimization Profile

The project now includes additional optimizations focused on CPU/memory efficiency:

- Lazy initialization for FER, SER, STT, and cloud LLM client.
- Thread-safe shared TFLite interpreter invocation (prevents race conditions under concurrent requests).
- Reduced OpenCV and BLAS thread oversubscription for stable CPU usage.
- Faster FER preprocessing path (grayscale decode, configurable face min size, largest-face selection).
- Lighter cloud HTTP connection pool defaults.
- Admin payload controls (default bounded fetch, text clipping, optional questionnaire answer inclusion).
- Incremental rendering in Admin UI to avoid large DOM stalls.

Recommended edge environment:

```env
SERENITY_EDGE_OPTIMIZED_MODE=true
SERENITY_LAZY_RUNTIME_INIT=true
SERENITY_CLOUD_LLM_LAZY_INIT=true
SERENITY_WHISPER_PRELOAD_ENABLED=false
SERENITY_CLOUD_LLM_WARMUP_ENABLED=false
SERENITY_TTS_WARMUP_ENABLED=false

SERENITY_WHISPER_CPU_THREADS=2
SERENITY_SER_TFLITE_THREADS=2
SERENITY_FER_TFLITE_THREADS=2
SERENITY_FER_CV2_THREADS=1
SERENITY_FER_MAX_FRAME_SIDE=640
SERENITY_FER_FACE_MIN_SIZE=48

SERENITY_CLOUD_LLM_CONNECT_TIMEOUT_SECONDS=4
SERENITY_CLOUD_LLM_TIMEOUT_SECONDS=12
SERENITY_CLOUD_LLM_FAILURE_THRESHOLD=2
SERENITY_CLOUD_LLM_COOLDOWN_SECONDS=30
SERENITY_CLOUD_LLM_POOL_CONNECTIONS=2
SERENITY_CLOUD_LLM_POOL_MAXSIZE=4

SERENITY_ADMIN_DEFAULT_LIMIT=200
SERENITY_ADMIN_CHAT_TEXT_LIMIT=360
SERENITY_ADMIN_SESSION_TEXT_LIMIT=360

OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

These defaults are reflected in Start_App.bat.

## Setup

### 1) Python Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install Dependencies

General runtime:

```powershell
pip install -r requirements.txt
```

Edge runtime (recommended for Pi):

```powershell
pip install -r requirements-edge.txt
```

### 3) Frontend Dependencies

```powershell
cd frontend
npm install
cd ..
```

### 4) Run Backend

From repository root (folder containing backend):

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 5000
```

If launching from one level above the repository folder, use:

```powershell
python -m uvicorn backend.main:app --app-dir .\FYP --host 127.0.0.1 --port 5000
```

### 5) Run Frontend

```powershell
cd frontend
npm run dev
```

Open http://localhost:5173/login.

### Optional One-Click Startup

- Start_App.bat

## Notes

- Passwords are currently stored as plaintext and should be migrated to hashed storage.
- requirements-edge.txt should be preferred on low-resource devices.
- Optional offline FAISS assets remain available but the active runtime uses cloud LLM by default.
