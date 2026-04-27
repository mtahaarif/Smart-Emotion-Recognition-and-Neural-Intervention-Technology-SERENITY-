# SERENITY

Smart Emotion Recognition and Neural Intervention Technology (SERENITY)

## Table of Contents

1. Abstract
2. Introduction
3. Objectives and Success Criteria
4. Motivation and Problem Framing
5. Scope and Non-Scope
6. System Architecture (Deep Dive)
7. Backend Technical Implementation
8. Frontend Technical Implementation
9. Data Architecture and Persistence Model
10. API Contract Reference
11. Configuration and Environment Variables
12. Complete Directory Guide
13. Raspberry Pi 5 Implementation Guide (Complete)
14. Validation and Acceptance Checklist
15. Troubleshooting Playbook
16. Security, Privacy, and Clinical Safety
17. Known Limitations
18. Future Work
19. License and Third-Party Notes

---

## 1. Abstract

SERENITY is a multimodal mental-health support platform engineered to bridge conversational AI support with structured clinical observability. The project combines speech-to-text (STT), speech emotion recognition (SER), optional facial emotion recognition (FER), questionnaire-driven measurement-based care (PHQ-9, GAD-7, PCL-5), and cloud-assisted therapeutic dialogue routing.

Unlike purely conversational assistants, SERENITY applies deterministic clinical routing logic (DBT, CBT, ACT, Supportive modes), persists longitudinal signals, and provides clinician-facing summaries and handoff artifacts. Unlike rigid questionnaire-only systems, it supports live voice and text interaction with near real-time NDJSON streaming.

The implementation is optimized for constrained edge hardware (including Raspberry Pi 5) through lightweight inference paths, bounded caches, connection pooling, circuit-breaker controls for cloud unavailability, and SQLite performance pragmas.

Important: SERENITY is a support and observability prototype, not a diagnostic medical device.

---

## 2. Introduction

Digital mental-health tooling frequently fails on one of two axes:

- high empathy but low structure (chatbots without clinical workflow discipline), or
- high structure but low engagement (forms-heavy systems with weak conversational continuity).

SERENITY is designed as a hybrid model:

- emotionally aware interaction layer,
- structured longitudinal screening and trajectory analysis,
- deterministic risk and framework routing,
- professional administrative observability and handoff generation,
- edge-friendly runtime profile for field deployment.

The project is implemented as a FastAPI backend and React + Vite frontend with explicit separation of responsibilities between interaction, clinical state management, and administrative analytics.

---

## 3. Objectives and Success Criteria

### 3.1 Primary Objectives

- Build a deployable multimodal support system that integrates voice, optional face frame, and text interaction.
- Maintain clinically meaningful continuity through persistent user trajectories and event logging.
- Route therapeutic response mode based on deterministic risk and heuristic indicators.
- Deliver low-latency interaction with stream-based response UX.
- Support edge deployment constraints without discarding cloud model quality.

### 3.2 Measurable Success Signals

- End-to-end voice interactions complete with emotion extraction, transcription, and LLM response.
- Questionnaire submission updates trajectory and safety-review state.
- Admin overview produces coherent risk, trend, and follow-up outputs.
- System remains functional under cloud degradation via local fallback responses.
- Raspberry Pi 5 deployment completes using edge dependency profile.

---

## 4. Motivation and Problem Framing

### 4.1 Practical Gap

Existing systems typically separate conversational support from formal monitoring. This creates fragmented workflows where acute sentiment and longitudinal screening signals are not synthesized in one place.

### 4.2 Design Intention

SERENITY integrates three operational pillars:

- Passive Perception: infer affect from speech and optional facial frame.
- Active Screening: PHQ-9, GAD-7, PCL-5 tracking with trend and velocity logic.
- Deterministic Clinical Routing: mode lock and framework-specific prompting with escalation paths.

### 4.3 Edge-Centric Rationale

In low-resource or privacy-sensitive settings, edge execution reduces reliance on heavyweight local infrastructure and preserves responsiveness for media processing, while still enabling cloud LLM quality for final response generation.

---

## 5. Scope and Non-Scope

### 5.1 In Scope

- Voice and text therapeutic interaction
- Optional camera-assisted FER
- Questionnaire scoring and trajectory state
- Clinical route/event persistence
- Admin overview, narrative summary, and handoff export
- Diagnostics telemetry for latency and resource usage

### 5.2 Out of Scope

- Medical diagnosis
- Autonomous emergency services dispatch
- Regulatory-grade clinical decision support
- Full offline replacement for cloud LLM

---

## 6. System Architecture (Deep Dive)

### 6.1 High-Level Topology

- Frontend (React/Vite): user interaction, streaming render, clinical dashboards.
- Backend (FastAPI): orchestration, routing, model inference management, persistence, analytics.
- Local DB (SQLite): durable records for turns, questionnaires, routing events, safety events, diagnostics.
- Cloud LLM endpoint: response generation via streamed SSE-like payload.

### 6.2 Privacy Boundary

- Raw media is processed on backend runtime.
- Cloud LLM receives routed prompt text (not raw video frames).
- Local persistence stores structured events and bounded text excerpts.

### 6.3 Core Interaction Flow (Voice)

1. Frontend captures microphone audio and optional camera snapshot.
2. Backend runs `_perceive()` pipeline:
   - `_transcribe()` (Whisper backend)
   - `predict_audio_emotion()` (SER TFLite)
   - `analyze_face()` (FER TFLite, optional)
3. Backend fuses emotions into dominant affect.
4. Risk score computed from distress regex + negative affect + screening flags.
5. Router selects framework and lock state.
6. Prompt built with framework rule constraints and phase context.
7. Cloud stream consumed and emitted as NDJSON events.
8. Protocol control parsed (`advance_phase`, `detected_distortion`).
9. State and events persisted (turns, routing events, safety escalation if required).
10. Optional TTS generated in-memory and returned as audio base64 segments/final blob.

### 6.4 Core Interaction Flow (Text)

- Same route logic and clinical state transition path as voice.
- Speech/face emotions are neutral placeholders.
- Streaming and fallback behavior mirrors voice route.

### 6.5 Clinical Routing Model

Frameworks:

- DBT_Distress_Tolerance
- CBT_Restructuring
- ACT_Defusion
- Supportive_Stabilization

Mode triggers:

- high risk or acute distress phrases -> DBT
- absolutist/catastrophic language -> CBT
- rumination phrases -> ACT
- otherwise supportive mode

Additional behavior:

- Tarasoff-like violence heuristic sets duty-to-warn flag on user profile.
- 24-hour post-crisis cooldown can force DBT stabilization mode.
- Route lock is active for non-supportive modes.

### 6.6 Streaming Protocol

Primary media type: `application/x-ndjson`

Common event types seen by frontend:

- `user_text`
- `emotion`
- `clinical_protocol_status`
- `assistant_delta`
- `assistant_sentence`
- `assistant_sentence_tts`
- `assistant_replace`
- `protocol_control`
- `error`
- `final`

Streaming controls include hard-cutoff logic in the cloud client for disallowed token patterns and kill phrases.

---

## 7. Backend Technical Implementation

Backend root: `backend/`

### 7.1 Runtime and Lifecycle

- Framework: FastAPI
- Startup lifecycle (`_lifespan`): initializes app state, optional model prewarm, caches, and diagnostics ring buffer.
- Shutdown lifecycle: closes cloud HTTP client safely.
- CORS: permissive (`*`) for development flexibility.

### 7.2 Module Responsibilities

#### `backend/main.py`

- Central orchestrator for all endpoints.
- Implements auth, interaction, admin analytics, safety, diagnostics, and handoff export.
- Contains stream event pump `_stream_events` with protocol extraction and persistence hooks.

#### `backend/cloud_llm_core.py`

- Async HTTP client with connection pooling.
- Supports primary and fallback endpoint rotation.
- Circuit breaker:
  - failure threshold (default 3)
  - cooldown window (default 20s)
- Streams token deltas from SSE-like `data:` frames.
- Enforces hard token cutoffs and kill phrase truncation.

#### `backend/audio_core.py`

- STT preparation and SER runtime.
- Uses librosa + polyphase resampling (`scipy.signal.resample_poly`) for CPU-friendly conversion.
- Supports `tflite-runtime` first, TensorFlow Lite fallback.

#### `backend/emotion_core.py`

- FER runtime with OpenCV Haar cascade and TFLite model.
- Resizes frames to cap processing cost.
- Returns `No Face` gracefully when detection fails.

#### `backend/clinical_router.py`

- Deterministic route evaluation and framework prompt rules.
- Regex-based cognitive distortion and distress heuristics.

#### `backend/clinical_core.py`

- Phase registry and phase advancement.
- Structured payload parsing from LLM text.
- Weekly trajectory flag computation.
- Markdown and PDF handoff builders.

#### `backend/database.py`

- SQLite engine configuration and pragmas.
- Additive schema migrations for `users` safety/contact fields.
- Persistence and query helpers for all major entities.

#### `backend/models.py`

- SQLAlchemy ORM schema with user-centered relationship graph.

#### `backend/questionnaires_data.py`

- Questionnaire definitions, scoring logic, severity mappings, clinical threshold flags.

### 7.3 Runtime Optimization Decisions

- Shared runtime singletons for SER/FER/Whisper where possible.
- Threadpool offloading for blocking paths (`run_in_threadpool`).
- Bounded in-memory structures:
  - admin overview cache (max entries with TTL)
  - admin summary cache (TTL)
  - edge diagnostics deque
- SQLite WAL and memory-oriented pragmas.

### 7.4 Failure and Fallback Strategy

- LLM unavailable -> framework-aware local fallback response.
- STT/SER/FER failures -> partial degradation, not full request abort.
- TTS failures -> text response still delivered.
- DB persistence errors captured as response-side warning events where possible.

---

## 8. Frontend Technical Implementation

Frontend root: `frontend/`

### 8.1 Application Shell

- React 18 + Vite
- Routing via `react-router-dom`
- Auth guard based on `localStorage` username token
- Global clinical state via `ClinicalContext`

### 8.2 Clinical Context (`frontend/src/context/ClinicalContext.jsx`)

Tracks and exposes:

- active risk score
- crisis mode flag
- current therapy mode
- connection status

Supports ingestion of backend events (`ingestBackendEvent`) and optional WS/SSE clinical feeds.

### 8.3 Detailed Web Page Architecture

#### 8.3.1 Login Page (`frontend/src/components/Login.jsx`)

- Registration and login mode toggle.
- Axios calls to `/register` and `/login`.
- Persists authenticated username to `localStorage`.

#### 8.3.2 Dashboard (`frontend/src/components/Dashboard.jsx`)

- Module launcher for all major workflows:
  - Clinical Assessment
  - Live Support Session
  - MBC Hub
  - Safety Protocol
  - Admin Observatory
  - Edge Diagnostics

#### 8.3.3 Live Session Page (`frontend/src/pages/UnifiedEmotionPage.jsx`)

Core features:

- Session lifecycle controls (start/end, push-to-speak)
- Media capture:
  - microphone via `MediaRecorder`
  - optional camera snapshot via canvas
- NDJSON stream parser for `/api/interact/stream` and `/api/chat/stream`
- Incremental assistant rendering from `assistant_delta`
- Replace/final reconciliation logic (`assistant_replace`, `final`)
- Per-turn emotion logging and distribution counters
- Stream TTS queue playback from base64 segments
- Fallback to non-streaming endpoints if stream fails

#### 8.3.4 Questionnaires Page (`frontend/src/pages/QuestionnairesPage.jsx`)

- Dynamic template fetch (`/api/questionnaires/templates`)
- Multi-form response entry for PHQ-9/GAD-7/PCL-5
- Validation for complete item coverage before submission
- Submission to `/api/questionnaires/submit`
- Historical records panel via `/api/questionnaires/history`

#### 8.3.5 MBC Hub (`frontend/src/pages/MBCHubPage.jsx`)

- Pulls trajectory model from `/api/mbc/trajectory`
- Recharts trend visualization for PHQ-9, GAD-7, and scaled PCL-5
- Velocity badges and due-assessment indicators
- Daily adherence checklist UI for routine and intervention items
- Escalation signal propagation to clinical context when safety review is required

#### 8.3.6 Admin Observatory (`frontend/src/pages/AdminPage.jsx`)

- Overview load from `/api/admin/overview`
- Clinical narrative load from `/api/admin/clinical-report`
- Export handoff markdown from `/api/admin/handoff/{user_id}`
- Displays:
  - risk metrics
  - chat transcripts
  - framework fidelity counts
  - timeline events
  - emotion distributions

#### 8.3.7 Safety Toolkit (`frontend/src/pages/SafetyPlanPage.jsx`)

- 5-4-3-2-1 grounding stepper
- paced breathing state machine (inhale/hold/exhale loops)
- optional speech synthesis guide
- C-SSRS triage ladder with risk-level assignment
- emergency/SOS utilities:
  - crisis log endpoint usage
  - handoff payload retrieval
  - location link capture and SMS draft fallback

#### 8.3.8 Hardware Diagnostics (`frontend/src/pages/HardwareDiagnosticsPage.jsx`)

- Polls `/api/diagnostics/metrics` every 2.5s
- Maintains bounded chart history window
- Displays latency, CPU, RAM, and delegate status
- Dual chart view for inference and hardware trends

---

## 9. Data Architecture and Persistence Model

Database: SQLite (`serenity.db` in project root)

### 9.1 Engine Pragmas

- `journal_mode=WAL`
- `synchronous=NORMAL`
- `temp_store=MEMORY`
- configurable cache size (`SERENITY_SQLITE_CACHE_KB`)
- `mmap_size=268435456`

### 9.2 Core Tables

- `users`
- `sessions`
- `emotions`
- `conversation_turns`
- `questionnaire_results`
- `clinical_states`
- `clinical_routing_events`
- `clinical_distortion_events`
- `safety_escalation_events`
- `trajectory_snapshots`
- `edge_diagnostic_samples`

### 9.3 Derived Clinical State Logic

- `requires_safety_review` may be set from trajectory worsening or acute events.
- Weekly trajectory snapshots replace prior snapshot set per user.
- Clinical state stores active framework, phase, last risk score, and last distortion marker.

---

## 10. API Contract Reference

Base backend URL (dev default): `http://127.0.0.1:5000`

### 10.1 Auth

- `POST /register`
- `POST /login`

### 10.2 Interaction

- `POST /api/interact`
- `POST /api/interact/stream`
- `POST /api/chat`
- `POST /api/chat/stream`

### 10.3 Questionnaire and Trajectory

- `GET /api/questionnaires/templates`
- `POST /api/questionnaires/submit`
- `GET /api/questionnaires/history`
- `GET /api/mbc/trajectory`

### 10.4 Admin and Reporting

- `GET /api/admin/overview`
- `GET /api/admin/clinical-report`
- `GET /api/admin/summary/stream`
- `GET /api/admin/handoff/{user_id}`

### 10.5 Safety and Crisis

- `POST /api/safety/emergency-contact`
- `GET /api/safety/handoff`
- `POST /api/crisis/log`
- `POST /api/clinical/clear-safety`

### 10.6 Diagnostics

- `GET /api/diagnostics/edge`
- `GET /api/diagnostics/metrics`

### 10.7 Streaming Event Shape (NDJSON)

Typical sequence example:

```json
{"type":"user_text","text":"..."}
{"type":"emotion","dominant_emotion":"Sad","speech_emotion":"Sad","face_emotion":"Neutral"}
{"type":"clinical_protocol_status","framework":"DBT_Distress_Tolerance","risk_score":7}
{"type":"assistant_delta","delta":"I hear that this is heavy"}
{"type":"assistant_sentence_tts","sequence":1,"audio_base64":"..."}
{"type":"final","llm_response":"...","clinical":{"framework":"...","phase":"..."}}
```

---

## 11. Configuration and Environment Variables

### 11.1 Backend Core

- `SERENITY_WHISPER_MODEL_SIZE` (default `tiny`)
- `SERENITY_WHISPER_CPU_THREADS`
- `SERENITY_WHISPER_TIMEOUT_SECONDS`
- `SERENITY_EMOTION_TIMEOUT_SECONDS`
- `SERENITY_LLM_TIMEOUT_SECONDS`

### 11.2 TTS

- `SERENITY_TTS_ENABLED` (default `true`)
- `SERENITY_TTS_VOICE` (default `en-GB-RyanNeural`)
- `SERENITY_TTS_FALLBACK_VOICE`
- `SERENITY_TTS_TIMEOUT_SECONDS`
- `SERENITY_TTS_RETRIES`
- `SERENITY_TTS_STREAM_MODE` (`sentence` or `final`)

### 11.3 Clinical/Admin

- `SERENITY_CLINICAL_WEEKLY_WORSENING_DELTA`
- `SERENITY_ADMIN_DEFAULT_LIMIT`
- `SERENITY_ADMIN_MAX_LIMIT`
- `SERENITY_ADMIN_OVERVIEW_CACHE_TTL_SECONDS`
- `SERENITY_ADMIN_SUMMARY_CACHE_TTL_SECONDS`
- `SERENITY_ADMIN_SUMMARY_TIMEOUT_SECONDS`

### 11.4 Cloud LLM

- `SERENITY_CLOUD_LLM_URL`
- `SERENITY_CLOUD_LLM_FALLBACK_URLS`
- `SERENITY_CLOUD_LLM_TIMEOUT_SECONDS`
- `SERENITY_CLOUD_LLM_CONNECT_TIMEOUT_SECONDS`
- `SERENITY_CLOUD_LLM_POOL_MAXSIZE`
- `SERENITY_CLOUD_LLM_HTTP2`
- `SERENITY_CLOUD_LLM_KILL_PHRASES`
- `SERENITY_CLOUD_LLM_FAILURE_THRESHOLD`
- `SERENITY_CLOUD_LLM_COOLDOWN_SECONDS`

### 11.5 SER/FER Runtime

- `SERENITY_SER_AUDIO_SAMPLE_RATE`
- `SERENITY_SER_AUDIO_DURATION_SECONDS`
- `SERENITY_SER_AUDIO_OFFSET_SECONDS`
- `SERENITY_SER_TFLITE_THREADS`
- `SERENITY_FER_TFLITE_THREADS`
- `SERENITY_FER_MAX_FRAME_SIDE`
- `SERENITY_FER_FACE_SCALE_FACTOR`
- `SERENITY_FER_FACE_MIN_NEIGHBORS`
- `SERENITY_FER_FACE_MIN_SIZE`
- `SERENITY_FER_CV2_THREADS`

### 11.6 Delegate and DB

- `SERENITY_TFLITE_XNNPACK_DELEGATE`
- `SERENITY_TFLITE_USE_EXTERNAL_DELEGATE`
- `SERENITY_XNNPACK_DELEGATE_ACTIVE` (diagnostics display flag)
- `SERENITY_SQLITE_CACHE_KB`

### 11.7 Frontend

- `VITE_API_BASE_URL` (default `http://127.0.0.1:5000`)
- optional stream behavior flags in UI (for provisional rendering)

### 11.8 Example `.env` for production backend

```env
SERENITY_CLOUD_LLM_URL=http://YOUR_LLM_HOST:8000/chat
SERENITY_CLOUD_LLM_FALLBACK_URLS=
SERENITY_TTS_ENABLED=true
SERENITY_TTS_STREAM_MODE=final
SERENITY_PREWARM_MODELS=true
SERENITY_PREWARM_WHISPER=false
SERENITY_WHISPER_MODEL_SIZE=tiny
SERENITY_WHISPER_CPU_THREADS=2
SERENITY_SER_TFLITE_THREADS=2
SERENITY_FER_TFLITE_THREADS=2
SERENITY_FER_CV2_THREADS=1
```

---

## 12. Complete Directory Guide

Current workspace root: `FYP/`

```text
FYP/
|- .github/
|  |- appmod/
|- backend/
|  |- audio_core.py
|  |- clinical_core.py
|  |- clinical_router.py
|  |- cloud_llm_core.py
|  |- database.py
|  |- emotion_core.py
|  |- fer_model.tflite
|  |- main.py
|  |- models.py
|  |- questionnaires_data.py
|  |- ser_model.tflite
|- docs/                       (currently empty)
|- frontend/
|  |- index.html
|  |- package.json
|  |- package-lock.json
|  |- postcss.config.js
|  |- tailwind.config.js
|  |- vite.config.js
|  |- dist/                    (generated build output)
|  |- node_modules/            (generated dependency tree)
|  |- src/
|     |- App.jsx
|     |- main.jsx
|     |- index.css
|     |- components/
|     |  |- Dashboard.jsx
|     |  |- Layout.jsx
|     |  |- Login.jsx
|     |- context/
|     |  |- ClinicalContext.jsx
|     |- pages/
|        |- AdminPage.jsx
|        |- HardwareDiagnosticsPage.jsx
|        |- MBCHubPage.jsx
|        |- QuestionnairesPage.jsx
|        |- SafetyPlanPage.jsx
|        |- UnifiedEmotionPage.jsx
|- Presentations/
|  |- Abstract.docx/.pdf and additional project references
|- scripts/
|  |- setup_rpi5.sh
|- FYP.code-workspace
|- README.md
|- requirements.txt
|- requirements-edge.txt
|- Start_App.bat
|- serenity.db                (runtime-generated database)
|- test.py
```

### 12.1 What each top-level item is for

- `backend/`: all server logic, model runtimes, API endpoints, data operations.
- `frontend/`: user interface and SPA route modules.
- `scripts/`: deployment automation scripts.
- `Presentations/`: research and report assets used during FYP lifecycle.
- `docs/`: reserved for additional documentation artifacts.

---

## 13. Raspberry Pi 5 Implementation Guide (Complete)

This section is written as a full start-to-finish implementation path.

### 13.1 Hardware and OS prerequisites

Required:

- Raspberry Pi 5 (8 GB RAM recommended)
- 64-bit Raspberry Pi OS (Bookworm)
- stable network
- USB microphone
- optional USB camera

Strongly recommended:

- active cooling
- official PSU
- SSD storage for better sustained I/O

### 13.2 System update and packages

```bash
sudo apt update
sudo apt full-upgrade -y
sudo reboot
```

After reboot:

```bash
sudo apt install -y git curl build-essential pkg-config ffmpeg \
  libsndfile1 libatlas-base-dev libopenblas-dev liblapack-dev \
  libglib2.0-0 libgl1
```

### 13.3 Python runtime setup

Recommended: Python 3.11

```bash
sudo apt install -y python3.11 python3.11-venv python3-pip
python3.11 --version
```

Supported runtime window for this project: Python 3.10 to 3.12.

### 13.4 Node runtime setup

```bash
sudo apt install -y nodejs npm
node -v
npm -v
```

### 13.5 Clone and model retrieval

```bash
git lfs install
git clone https://github.com/mtahaarif/Smart-Emotion-Recognition-and-Neural-Intervention-Technology-SERENITY-.git
cd Smart-Emotion-Recognition-and-Neural-Intervention-Technology-SERENITY-/FYP
git lfs pull
```

### 13.6 Backend environment (manual)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install --extra-index-url https://www.piwheels.org/simple -r requirements-edge.txt
```

### 13.7 Backend environment (automated script)

```bash
chmod +x scripts/setup_rpi5.sh
./scripts/setup_rpi5.sh
```

What script does:

- validates Python version window
- checks architecture and warns if not aarch64
- recreates incompatible `.venv`
- installs edge requirements
- retries without `tflite-runtime` then installs TensorFlow fallback if needed

### 13.8 Frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 13.9 Development run

Terminal A:

```bash
source .venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 5000
```

Terminal B:

```bash
cd frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

Open:

- `http://<PI_IP>:5173/login`

### 13.10 Production run (recommended)

#### Step A: build frontend

```bash
cd frontend
npm run build
cd ..
```

#### Step B: backend systemd service

`/etc/systemd/system/serenity-backend.service`

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

Enable service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable serenity-backend
sudo systemctl start serenity-backend
sudo systemctl status serenity-backend
```

#### Step C: nginx reverse proxy

Install:

```bash
sudo apt install -y nginx
```

Create `/etc/nginx/sites-available/serenity`:

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

Enable and restart:

```bash
sudo ln -s /etc/nginx/sites-available/serenity /etc/nginx/sites-enabled/serenity
sudo nginx -t
sudo systemctl restart nginx
```

Set frontend production base URL before build:

`frontend/.env.production`

```env
VITE_API_BASE_URL=/api
```

Rebuild frontend after this change.

---

## 14. Validation and Acceptance Checklist

Run in order:

1. `POST /login` smoke test from localhost.
2. Open login page and authenticate.
3. Run text chat first (`/api/chat` or stream variant).
4. Run voice interaction with microphone.
5. Enable camera and verify FER optional path.
6. Submit one questionnaire of each type.
7. Confirm MBC trajectory updates.
8. Open admin page and verify summary + metrics.
9. Confirm diagnostics telemetry updates every poll interval.
10. Export handoff markdown and verify file content integrity.

---

## 15. Troubleshooting Playbook

### 15.1 `tflite-runtime` install failure

Symptoms:

- `No matching distribution found for tflite-runtime`

Actions:

1. verify Python version (`python --version`) is 3.10 to 3.12
2. verify architecture (`uname -m`) is `aarch64`
3. use piwheels index URL
4. if unresolved, install without `tflite-runtime` and use TensorFlow fallback

### 15.2 NumPy ABI mismatch errors

Symptoms:

- `_ARRAY_API not found`
- `numpy.core.multiarray failed to import`

Actions:

1. remove corrupted/partial environment
2. recreate `.venv`
3. reinstall from edge requirements in clean environment

### 15.3 Cloud LLM timeout/cooldown behavior

Symptoms:

- delayed responses or fallback text only

Actions:

1. validate `SERENITY_CLOUD_LLM_URL`
2. set fallback URLs
3. tune connect/read timeout and failure threshold
4. inspect backend logs for timeout class names

### 15.4 Edge TTS 403 errors

Actions:

1. sync system time (`timedatectl`)
2. update `edge-tts` and dependencies
3. clear proxy environment variables
4. switch to `SERENITY_TTS_STREAM_MODE=final`
5. disable TTS temporarily if service path remains blocked

### 15.5 Frontend cannot reach backend

Actions:

1. ensure backend host/port is reachable
2. verify `VITE_API_BASE_URL`
3. verify nginx `/api` reverse proxy path

### 15.6 Camera or mic unavailable

Actions:

1. grant browser permissions
2. verify devices with `arecord -l` and `v4l2-ctl --list-devices`
3. retry in Chromium on Pi

### 15.7 Delegate warnings

If external delegate library is missing, keep external delegate disabled and use CPU interpreter path.

---

## 16. Security, Privacy, and Clinical Safety

- This project is a support prototype, not a diagnostic authority.
- Distress detection is heuristic and may produce false positives/negatives.
- Enforce strict access control and CORS restrictions before internet exposure.
- Treat all logs and transcript data as sensitive.
- Keep human-in-the-loop escalation for high-risk contexts.
- Use local legal and institutional frameworks for mental-health data governance.

---

## 17. Known Limitations

- Cloud LLM dependency remains a critical path for best response quality.
- Heuristic routing and distress regex cannot replace clinician judgement.
- Current auth is minimal (username/password) and needs hardening for production.
- CORS is permissive by default and should be restricted in secure deployments.
- Some edge tuning variables in startup scripts may be legacy and not consumed by backend code directly.

---

## 18. Future Work

- stronger auth and role-based access controls
- encrypted secret management and transport hardening
- richer observability (structured logs, traces, audit events)
- optional offline local LLM fallback for disconnected operation
- improved multilingual support in STT and response routing
- wearable/physiological signal integration for multimodal confidence fusion

---

## 19. License and Third-Party Notes

Use of this repository must comply with:

- repository license terms,
- third-party package licenses,
- model and hosted service licenses,
- applicable data-protection and health governance regulations.

SERENITY is an academic engineering project and is not an approved medical device.
