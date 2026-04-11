# SERENITY

Smart Emotion Recognition and Neural Intervention Technology (SERENITY) is a multimodal AI therapist assistant focused on edge-friendly deployment (including Raspberry Pi 5) with cloud LLM offload.

## What This Repository Runs Today

- FastAPI backend with streaming NDJSON APIs
- React + Vite frontend with real-time conversation UX
- TFLite-based FER (face emotion recognition)
- TFLite-based SER (speech emotion recognition)
- Whisper transcription (faster-whisper preferred)
- Cloud LLM over HTTP/SSE (`/chat` endpoint)
- Sentence-level Edge TTS streaming while generation is still in progress
- SQLite persistence for users and conversation turns

## Key Runtime Behavior

### 1. Real-time Assistant Streaming

The backend streams `assistant_delta` events as tokens arrive from the cloud model.
Frontend renders provisional text immediately and updates incrementally.

### 2. Early Sentence TTS

As soon as a sentence boundary is detected, backend emits:

- `assistant_sentence`
- `assistant_sentence_tts` (base64 audio)

This allows speech playback before the full response is complete.

### 3. Output Safety and Cleanup

A strict cutoff is enforced at the first `*` in streamed and final text:

- Keep text before first `*`
- Ignore everything after first `*`

This prevents prompt/thought traces or policy tails from reaching UI/TTS.

### 4. Cloud Reliability

The cloud client supports:

- endpoint failover (`SERENITY_CLOUD_LLM_URL` + optional fallback list)
- timeout controls
- cooldown/circuit-breaker behavior after repeated failures
- stream-to-non-stream retry fallback

## Runtime Architecture

```text
Browser (React/Vite)
  -> /api/interact/stream (multipart, NDJSON)
  -> /api/chat/stream (JSON, NDJSON)

FastAPI (backend/main.py)
  -> lazy/eager runtime loading (FER, SER, STT)
  -> parallel transcription + speech emotion + optional face emotion
  -> emotion fusion
  -> cloud LLM streaming
  -> sentence-level TTS stream
  -> SQLite persistence

Model/Assets
  -> backend/fer_model.tflite
  -> backend/ser_model.tflite
  -> backend/serenity_faiss.index (optional/offline KB assets)
  -> backend/serenity_chunks.pkl (optional/offline KB assets)
```

## Streaming Event Contract (Backend -> Frontend)

The frontend consumes one JSON object per line.

- `transcription`: `{ type, text }`
- `user_text`: `{ type, text, source }`
- `emotion_partial`: `{ type, speech_emotion?, speech_confidence?, face_emotion?, face_confidence? }`
- `emotion`: `{ type, dominant_emotion, speech_emotion, face_emotion }`
- `assistant_delta`: `{ type, delta, text }`
- `assistant_sentence`: `{ type, text, sequence }`
- `assistant_sentence_tts`: `{ type, text, sequence, audio_base64 }`
- `assistant_replace`: `{ type, text }`
- `assistant_tts_reset`: `{ type }`
- `assistant_tts_trim`: `{ type, max_sequence }`
- `error`: `{ type, message }`
- `final`: `{ type, llm_response, transcription, dominant_emotion, speech_emotion, face_emotion }`

## API Surface

### Auth

- `POST /register`
- `POST /login`

### Health

- `GET /health`

### Multimodal Voice + Optional Vision

- `POST /api/interact`
- `POST /api/interact/stream`

### Text Chat

- `POST /api/chat`
- `POST /api/chat/stream`

## Raspberry Pi 5 Edge Optimization Defaults

SERENITY is tuned for low-resource operation via lazy loading and reduced thread pressure.

Recommended env profile:

```env
SERENITY_EDGE_OPTIMIZED_MODE=true
SERENITY_LAZY_RUNTIME_INIT=true
SERENITY_WHISPER_PRELOAD_ENABLED=false
SERENITY_CLOUD_LLM_WARMUP_ENABLED=false
SERENITY_TTS_WARMUP_ENABLED=false

SERENITY_WHISPER_CPU_THREADS=2
SERENITY_SER_TFLITE_THREADS=2
SERENITY_FER_TFLITE_THREADS=2
SERENITY_FER_MAX_FRAME_SIDE=640

SERENITY_STREAM_TOKEN_DELTA=true
SERENITY_STREAM_TTS_SENTENCE_AUDIO=true
SERENITY_STREAM_TTS_FINAL_TEXT_ONLY=false
SERENITY_STREAM_QUEUE_WAIT_SECONDS=0.015

SERENITY_CLOUD_LLM_CONNECT_TIMEOUT_SECONDS=4
SERENITY_CLOUD_LLM_TIMEOUT_SECONDS=12
SERENITY_CLOUD_LLM_FAILURE_THRESHOLD=2
SERENITY_CLOUD_LLM_COOLDOWN_SECONDS=30
```

These defaults are also reflected in `Start_App.bat`.

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

Edge profile (recommended on Pi):

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

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 5000
```

### 5) Run Frontend

```powershell
cd frontend
npm run dev
```

Open:

- `http://localhost:5173/login`

### Optional One-Click Startup

- `Start_App.bat`

## Dependency Notes

- `requirements.txt` is now focused on cloud-hybrid runtime.
- Heavy local LLM packages are commented as optional in `requirements.txt`.
- `requirements-edge.txt` is optimized for lower memory use.

## Persistence

SQLite file:

- `serenity.db`

Active tables used by runtime:

- `users`
- `conversation_turns`

Database connection sets SQLite pragmas for better edge-device performance (WAL, normal sync, memory temp store, cache tuning).

## Optional Offline Knowledge Assets

`build_kb.py` and optional FAISS assets remain for offline experimentation paths:

- `backend/serenity_faiss.index`
- `backend/serenity_chunks.pkl`

The active runtime chat path is cloud LLM first.

## Current Known Gaps

- Passwords are still stored in plaintext (needs hashing migration).
- Automated tests and CI are not yet complete.
- No production auth hardening (JWT/session security) yet.
- Observability dashboards and alerting are still pending.

## Repository Layout (Practical)

```text
FYP/
  backend/
    main.py
    cloud_llm_core.py
    emotion_core.py
    audio_core.py
    database.py
    models.py
    fer_model.tflite
    ser_model.tflite
  frontend/
    src/
      App.jsx
      components/
      pages/UnifiedEmotionPage.jsx
  requirements.txt
  requirements-edge.txt
  Start_App.bat
  README.md
```
