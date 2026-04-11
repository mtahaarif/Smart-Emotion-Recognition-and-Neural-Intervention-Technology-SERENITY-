# SERENITY

SERENITY (Smart Emotion Recognition and Neural Intervention Technology) is a multimodal, edge-aware AI support system that combines speech analysis, optional face analysis, conversational assistance, and mental health questionnaire workflows into one integrated platform.

The system is designed to run smoothly on constrained hardware such as Raspberry Pi 5 while still supporting cloud-offloaded language generation when enabled.

## 1. Introduction

SERENITY provides a complete user journey:

1. Authenticate locally.
2. Start a live interaction session using microphone, with optional camera.
3. Receive emotional signal fusion and assistant responses in real time.
4. Complete standardized questionnaires (PHQ-9, GAD-7, PCL-5).
5. Review all local analytics from an admin observatory.

The architecture prioritizes low-latency feedback, deterministic cleanup/safety behavior, and memory-conscious execution for edge environments.

## 2. Motivation

The project was motivated by three practical goals:

1. Build an emotionally aware conversational experience that does not depend entirely on high-end hardware.
2. Support local-first persistence and analytics for environments with intermittent cloud access.
3. Offer an implementation that can scale from development laptops to Raspberry Pi-class deployments with explicit performance controls.

## 3. Core Objectives

1. Real-time multimodal interaction.
2. Robust streaming contract between backend and frontend.
3. Strong output cleaning to prevent internal prompt leakage.
4. Local questionnaire and admin analytics workflows.
5. Edge optimization with controlled CPU and memory pressure.

## 4. System Architecture

### 4.1 Frontend

- React + Vite application.
- Route-level lazy loading for lower initial bundle cost.
- Streaming NDJSON consumer for live response updates.
- Dedicated pages for live session, questionnaires, and admin analytics.

### 4.2 Backend

- FastAPI service with sync and streaming endpoints.
- Emotion stack:
	- SER using TFLite.
	- FER using TFLite + OpenCV Haar face detection.
- STT stack:
	- faster-whisper preferred.
	- openai-whisper fallback.
- Optional cloud LLM client with endpoint failover and cooldown behavior.
- SQLite persistence with performance pragmas.

### 4.3 Data Layer

- SQLAlchemy ORM.
- Local SQLite file (serenity.db).
- Tables for users, turns, sessions, emotions, and questionnaire results.

## 5. Feature Inventory

### 5.1 Authentication

- Register and login endpoints.
- Local credential storage (currently plaintext, slated for hashing migration).

### 5.2 Live Emotion Session

- Voice required.
- Camera optional.
- Parallel processing of transcription, speech emotion, and face emotion.
- Emotion fusion and dominant label selection.
- Real-time assistant streaming and sentence-level TTS events.

### 5.3 Text Chat Session

- Lightweight text-only route.
- Uses same LLM response sanitization and persistence flow.

### 5.4 Questionnaires

- Supported: PHQ-9, GAD-7, PCL-5.
- Selection supports one, multiple, or all questionnaires.
- Score and severity computed server-side.
- Dated history stored in local DB.

### 5.5 Admin Observatory

- Aggregates local chats, sessions, emotions, and questionnaire outcomes.
- Returns summary text plus metrics and top emotions.
- Supports bounded limits and optional answer payload inclusion.
- Frontend uses incremental rendering for large datasets.

## 6. Streaming Protocol

Backend emits NDJSON events. Frontend consumes one JSON object per line.

Primary event types:

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

This protocol enables progressive UX updates, early audio playback, and state correction when final polished text differs from provisional stream text.

## 7. Safety and Output Cleanup

To avoid exposing chain-of-thought or internal prompt artifacts:

1. Text is hard-cleaned during streaming and finalization.
2. Content after first asterisk marker is discarded.
3. Prompt-leak and policy-tail patterns are stripped.
4. Repetition collapse and sentence deduplication are applied.

## 8. Performance and Resource Optimization

The codebase includes explicit optimizations for Pi-class devices:

1. Lazy runtime initialization for FER, SER, STT, and cloud client.
2. Thread-safe shared TFLite interpreter usage (invoke locks).
3. Reduced thread oversubscription controls (OpenBLAS/MKL/OMP/OpenCV).
4. FER path optimization:
	- grayscale decode path,
	- largest-face selection,
	- configurable minimum face size.
5. Cloud HTTP pool sizing controls to reduce memory footprint.
6. Admin payload clipping and bounded default limits.
7. Frontend incremental rendering for large admin lists.

## 9. Cloud LLM Operating Modes

### 9.1 Cloud Enabled

- Normal cloud generation path is active.
- Streaming plus non-stream retry fallback is used.

### 9.2 Cloud Disabled

Set SERENITY_CLOUD_LLM_ENABLED=false.

- Backend skips cloud client initialization.
- Network attempts are avoided.
- Safe fallback assistant text path remains available.

This mode is useful for local-only testing and offline-like operation.

## 10. Backend API Reference

### Auth

- POST /register
- POST /login

### Health

- GET /health

### Live Interaction

- POST /api/interact
- POST /api/interact/stream

### Text Interaction

- POST /api/chat
- POST /api/chat/stream

### Questionnaires

- GET /api/questionnaires/templates
- POST /api/questionnaires/submit
- GET /api/questionnaires/history

### Admin

- GET /api/admin/overview

## 11. Database Schema (Runtime)

Active tables:

- users
- conversation_turns
- sessions
- emotions
- questionnaire_results

SQLite pragmas applied at connection time:

- journal_mode=WAL
- synchronous=NORMAL
- temp_store=MEMORY
- cache_size configurable via SERENITY_SQLITE_CACHE_KB

## 12. Frontend Routes

- /login
- /dashboard
- /emotion/live
- /questionnaires
- /admin

## 13. Environment Configuration

### 13.1 Recommended Pi Profile

```env
SERENITY_EDGE_OPTIMIZED_MODE=true
SERENITY_LAZY_RUNTIME_INIT=true
SERENITY_CLOUD_LLM_ENABLED=false
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

Start_App.bat already includes practical edge defaults for local launch.

## 14. Setup and Run

### 14.1 Python Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 14.2 Install Dependencies

General profile:

```powershell
pip install -r requirements.txt
```

Edge profile:

```powershell
pip install -r requirements-edge.txt
```

### 14.3 Frontend Setup

```powershell
cd frontend
npm install
cd ..
```

### 14.4 Run Backend

From repository root:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 5000
```

If launching from parent folder:

```powershell
python -m uvicorn backend.main:app --app-dir .\FYP --host 127.0.0.1 --port 5000
```

### 14.5 Run Frontend

```powershell
cd frontend
npm run dev
```

Open:

- http://localhost:5173/login

### 14.6 One-Click Startup

- Start_App.bat

## 15. Validation Commands

### Backend syntax check

```powershell
python -m py_compile backend/main.py backend/audio_core.py backend/emotion_core.py backend/cloud_llm_core.py backend/database.py
```

### Frontend production build

```powershell
cd frontend
npm run build
```

## 16. Repository Structure and Implementation Map

```text
FYP/
	README.md
	Start_App.bat
	requirements.txt
	requirements-edge.txt
	backend/
		main.py                  # FastAPI app, routes, streaming orchestration, safety cleanup
		cloud_llm_core.py        # Cloud client, endpoint failover, stream parsing, retries/cooldowns
		audio_core.py            # SER model runtime and inference pipeline
		emotion_core.py          # FER model runtime and face analysis pipeline
		database.py              # SQLAlchemy engine/session/pragmas and data helper methods
		models.py                # ORM models
		questionnaires_data.py   # PHQ-9/GAD-7/PCL-5 templates, scoring, severity thresholds
		fer_model.tflite         # Face emotion model artifact
		ser_model.tflite         # Speech emotion model artifact
		ec2_serenity_server.py   # Optional EC2-side LLM server implementation
	frontend/
		package.json
		vite.config.js
		src/
			App.jsx                        # Route wiring and auth gate
			index.css
			components/
				Login.jsx
				Dashboard.jsx
			pages/
				UnifiedEmotionPage.jsx       # Live stream UX and event consumption
				QuestionnairesPage.jsx       # Questionnaire submit/history UX
				AdminPage.jsx                # Analytics and metrics dashboard UX
```

## 17. Known Limitations and Next Improvements

1. Credentials are currently stored in plaintext.
2. Automated test coverage and CI pipelines should be expanded.
3. Admin analytics can be extended with trend windows and export support.
4. Optional local LLM mode can be further optimized for low-memory edge devices.

## 18. License and Attribution

Use and distribution should follow the repository license and the licenses of all third-party dependencies and model assets.
