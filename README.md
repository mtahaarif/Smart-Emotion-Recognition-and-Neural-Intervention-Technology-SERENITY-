# SERENITY

Smart Emotion Recognition and Neural Intervention Technology (SERENITY) is a multimodal AI therapist assistant developed as a Final Year Project.

This repository currently runs a local-first stack with:
- FastAPI backend for orchestration and streaming APIs
- React + Vite frontend for real-time therapy loop UX
- TFLite FER/SER inference for facial and speech emotion signals
- Whisper-based transcription
- Emotion-aware RAG + Qwen2.5 generation
- Edge TTS with sentence-level streaming playback
- SQLite persistence for user accounts and conversation memory

## 1. Current Implementation Snapshot

### What Is Working Now
- User registration and login via FastAPI (`/register`, `/login`)
- Session-based live interaction UX (`/emotion/live`)
- Voice unit capture with optional camera snapshot
- Parallel inference pipeline (transcription + SER + optional FER)
- Emotion fusion (`speech + face`) with normalized probability vectors
- Streaming assistant responses over NDJSON
- Sentence-level TTS chunk generation and playback queue
- Text-only chat mode (`/api/chat`, `/api/chat/stream`)
- Local conversation memory using `conversation_turns` table
- Optional cloud LLM transport mode using AWS SQS

### Current Progress (Code-Verified)
- Core multimodal loop: done
- Streaming protocol (backend + frontend consumer): done
- Local RAG runtime integration: done
- Fallback handling (LLM/TTS/STT and endpoint fallback): done
- Cloud offload transport adapter (SQS path): done (optional mode)
- Security hardening for production auth/data: pending
- Automated tests and CI pipeline: pending
- Production deployment profiles and observability: pending

## 2. Runtime Architecture

```text
Browser (React/Vite)
  -> /api/interact/stream (multipart NDJSON)
  -> /api/chat/stream (JSON NDJSON)

FastAPI (backend/main.py)
  -> Lifespan preload:
     - FER runtime (TFLite)
     - SER runtime (TFLite)
     - STT backend (faster-whisper preferred, openai-whisper fallback)
     - LLM runtime (local RAG) OR cloud transport client (SQS)
  -> Parallel task orchestration + timeout guards
  -> Emotion fusion + response generation + TTS
  -> SQLite persistence (SQLAlchemy)

Model/Data Assets
  -> backend/fer_model.tflite
  -> backend/ser_model.tflite
  -> backend/serenity_faiss.index
  -> backend/serenity_chunks.pkl
  -> backend/tinyllama_local/* (artifact folder)
```

## 3. End-to-End Processing Flow

### Voice + Optional Vision Unit
1. Frontend records mic audio and optionally captures a camera frame.
2. Frontend posts multipart payload to `/api/interact/stream`.
3. Backend runs tasks concurrently:
   - STT transcription (`_transcribe_with_whisper`)
   - Speech emotion (`predict_audio_emotion`)
   - Optional face emotion (`analyze_face`)
4. Backend emits partial stream events as each task completes.
5. Backend computes fused dominant emotion.
6. Backend loads recent turns from SQLite (`fetch_recent_turns`).
7. Backend calls streaming generation (`generate_multimodal_streaming`).
8. Backend emits token/sentence deltas and optional sentence TTS audio events.
9. Backend persists full turn (`persist_turn`) and emits final event.

### Text-Only Prompt
1. Frontend posts JSON to `/api/chat/stream`.
2. Backend sets neutral modality context and streams response.
3. Turn is persisted in the same memory table for continuity.

## 4. Streaming Event Contract (NDJSON)

The frontend parser in `frontend/src/pages/UnifiedEmotionPage.jsx` consumes one JSON object per line.

### Event Types Actually Used
- `transcription`
  - `{ type, text }`
- `user_text`
  - `{ type, text, source }`
- `emotion_partial`
  - `{ type, speech_emotion?, speech_confidence?, face_emotion?, face_confidence? }`
- `emotion`
  - `{ type, dominant_emotion, speech_emotion, face_emotion }`
- `assistant_delta`
  - `{ type, delta, text }`
- `assistant_sentence`
  - `{ type, text, sequence }`
- `assistant_sentence_tts`
  - `{ type, text, sequence, audio_base64 }`
- `assistant_replace`
  - `{ type, text }`
- `error`
  - `{ type, message }`
- `final`
  - `{ type, llm_response, transcription, dominant_emotion, speech_emotion, face_emotion }`

Note: `generation_result` is an internal backend event used to finalize state, not forwarded directly to the frontend stream UI.

## 5. API Surface (Current)

### Auth
- `POST /register`
  - Body: `{ username, password }`
  - Creates account if username is unique
- `POST /login`
  - Body: `{ username, password }`
  - Returns username on success

### Health
- `GET /health`
  - Returns `{ status, rag_loaded }`

### Multimodal
- `POST /api/interact`
  - Multipart form
  - Required: `username`, `file` (mic audio)
  - Optional: `image`, `user_message`
  - Returns non-streaming response with optional base64 TTS

- `POST /api/interact/stream`
  - Multipart form
  - Required: `username`, `file`
  - Optional: `image`, `user_message`
  - Returns NDJSON event stream

### Text Chat
- `POST /api/chat`
  - JSON body: `{ username, message }`
  - Returns non-streaming response

- `POST /api/chat/stream`
  - JSON body: `{ username, message }`
  - Returns NDJSON event stream

## 6. AI and Inference Stack

### FER (Face Emotion Recognition)
- Module: `backend/emotion_core.py`
- Model: `backend/fer_model.tflite`
- Steps: decode base64 -> detect face (Haar cascade) -> resize to 48x48 -> infer

### SER (Speech Emotion Recognition)
- Module: `backend/audio_core.py`
- Model: `backend/ser_model.tflite`
- Steps: librosa load -> MFCC extraction -> dynamic tensor shaping -> infer

### STT
- Primary: `faster-whisper`
- Fallback: `openai-whisper`
- Model size configurable (`SERENITY_WHISPER_MODEL_SIZE`, default `tiny`)

### RAG + LLM
- Module: `backend/llm_core.py`
- Embedding model: `all-MiniLM-L6-v2`
- Vector store: FAISS index + serialized chunks
- Primary LLM: `Qwen/Qwen2.5-1.5B-Instruct`
- Fallback model list supported via env
- Generates short empathetic replies with guardrails and fallback templates

### TTS
- Module: backend helpers in `main.py`
- Engine: `edge-tts`
- Supports:
  - Full-turn base64 audio
  - Sentence-level streaming audio chunks
  - Cooldown after repeated failures
  - Browser speech synthesis fallback in frontend

## 7. Data Model and Persistence

SQLite DB: `serenity.db`

### Active Tables
- `users`
  - User credentials and identity
- `conversation_turns`
  - Stores user text, assistant text, dominant/speech/face emotion, timestamp

### Legacy/Partially Used Tables
- `sessions`
- `emotions`

The live loop primarily depends on `conversation_turns` for contextual memory.

## 8. Frontend Architecture

Main app flow:
- `frontend/src/App.jsx`
  - Route guards + localStorage auth key (`serenity_user`)
- `frontend/src/components/Login.jsx`
  - Register/login UI and backend auth calls
- `frontend/src/components/Dashboard.jsx`
  - Session launcher
- `frontend/src/pages/UnifiedEmotionPage.jsx`
  - Real-time multimodal loop:
    - mic recorder + optional camera
    - NDJSON parser
    - live emotion + transcript + assistant stream rendering
    - sentence TTS queue playback
    - fallback from streaming endpoints to non-streaming endpoints

## 9. Environment and Config

### High-Impact Backend Variables
- `SERENITY_LLM_MODE` (`local` or `cloud`)
- `SERENITY_SKIP_RAG_STARTUP`
- `SERENITY_WHISPER_MODEL_SIZE`
- `SERENITY_WHISPER_CPU_THREADS`
- `SERENITY_TTS_ENABLED`
- `SERENITY_TTS_STREAMING_ENABLED`
- `SERENITY_STREAM_TOKEN_DELTA`
- `SERENITY_STREAM_TTS_SENTENCE_AUDIO`
- `SERENITY_CLOUD_LLM_TIMEOUT_SECONDS`

### Cloud Mode Variables (AWS SQS)
- `AWS_REGION`
- `SERENITY_SQS_REQUEST_QUEUE_URL`
- `SERENITY_SQS_RESULT_QUEUE_URL`
- `SERENITY_SQS_GROUP_ID`

### Frontend Variable
- `VITE_API_BASE_URL` (defaults to `http://127.0.0.1:5000`)

## 10. Setup and Run

### 1) Install Python dependencies
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Install frontend dependencies
```powershell
cd frontend
npm install
cd ..
```

### 3) Run backend
```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 5000
```

### 4) Run frontend
```powershell
cd frontend
npm run dev
```

### 5) Open app
- `http://localhost:5173/login`

### Optional one-click startup
- `Start_App.bat`

## 11. Knowledge Base Build (Offline)

If FAISS assets are missing or need rebuilding:

```powershell
python build_kb.py
```

This regenerates:
- `backend/serenity_faiss.index`
- `backend/serenity_chunks.pkl`

## 12. Current Gaps and Deep Implementation Roadmap

### A. Security and Auth (High Priority)
1. Replace plaintext password storage with salted hashing (e.g., `passlib[bcrypt]`).
2. Add token-based auth (JWT/session cookies) and backend route protection.
3. Add rate limiting and request size limits for upload endpoints.
4. Restrict CORS in non-dev environments.

### B. Data Layer and Governance
1. Add Alembic migrations and schema versioning.
2. Add retention policy and archival strategy for conversation turns.
3. Add audit logging strategy for access and model decisions.
4. Normalize or remove legacy tables if no longer needed.

### C. Reliability and Scalability
1. Add request tracing and structured logs for each turn (`trace_id`).
2. Add retries/circuit breakers for cloud LLM result polling.
3. Add bounded worker queues for heavy operations.
4. Add graceful degradation profile for low-memory devices.

### D. Evaluation and QA
1. Add backend unit tests:
   - fusion logic
   - fallback behavior
   - event stream shape
2. Add frontend integration tests for NDJSON event handling.
3. Add golden test fixtures for stream event regression checks.
4. Add performance benchmarks (P50/P95 end-to-end latency).

### E. Model and Product Quality
1. Introduce stronger safety and crisis-routing policy layer.
2. Add confidence calibration for FER/SER outputs.
3. Add user controls for voice, persona, and response style constraints.
4. Add session summaries and clinician-readable exports.

### F. Deployment and Operations
1. Add Dockerized profiles (CPU local, edge-only, hybrid cloud).
2. Add CI pipeline (lint, tests, build, smoke checks).
3. Add production environment templates (`.env.example`, secrets policy).
4. Add metrics dashboards and alerting for service health.

## 13. Known Limitations (Current)

- Passwords are currently stored as plaintext in `users` table.
- `requirements.txt` still includes some legacy dependencies not used in active runtime path.
- No automated test suite or CI gates in this state.
- Conversation memory is simple recency context, not long-term summarization memory.
- TTS and transcription quality can vary by hardware/network/runtime availability.

## 14. Repository Layout (Current Practical View)

```text
FYP/
  backend/
    main.py
    llm_core.py
    emotion_core.py
    audio_core.py
    cloud_llm_core.py
    database.py
    models.py
    fer_model.tflite
    ser_model.tflite
    serenity_faiss.index
    serenity_chunks.pkl
  frontend/
    src/
      App.jsx
      components/
        Login.jsx
        Dashboard.jsx
      pages/
        UnifiedEmotionPage.jsx
  build_kb.py
  Start_App.bat
  requirements.txt
  README.md
```

## 15. Hybrid Target Direction (Edge + AWS)

The codebase already supports a practical hybrid migration path:
- Keep capture + FER/SER/STT + UI on edge/local node
- Offload LLM generation to cloud worker via SQS

Current enabler in code:
- `backend/cloud_llm_core.py` + `SERENITY_LLM_MODE=cloud`

Next required steps:
1. Implement/secure worker service contract.
2. Add signed payload validation and stricter IAM policies.
3. Add queue DLQ and observability dashboards.
4. Benchmark cost-latency tradeoffs for target hardware.

---

This README reflects the current code path in this repository (FastAPI streaming architecture) and identifies concrete pending work for production-grade readiness.
