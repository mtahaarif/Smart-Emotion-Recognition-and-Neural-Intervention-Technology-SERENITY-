# SERENITY

Smart Emotion Recognition and Neural Intervention Technology (SERENITY) is a multimodal mental-health support platform that combines:

- speech transcription,
- speech emotion recognition (SER),
- optional face emotion recognition (FER),
- cloud-assisted conversational response generation,
- standardized questionnaire workflows (PHQ-9, GAD-7, PCL-5),
- clinician-style analytics in a user-scoped admin observatory.

The project is designed for practical operation on constrained systems (including Raspberry Pi class devices) with explicit resource controls and safe fallback behavior.

---

## 1) Repository Status and Cleanup Note

This repository has been intentionally cleaned to keep source control focused on code and core runtime assets.

What was intentionally removed from git history in the current state:

- large legacy notebooks and one-off evaluation artifacts,
- historical presentation files,
- deprecated local LLM adapter snapshots,
- temporary experiments and generated caches,
- non-essential binaries not required for current runtime.

What remains as the canonical production-oriented stack:

- FastAPI backend in `backend/`,
- React + Vite frontend in `frontend/`,
- TFLite model artifacts currently referenced by backend runtime,
- local SQLite persistence layer,
- startup and dependency manifests.

---

## 2) End-to-End Product Scope

SERENITY currently supports the following user journey:

1. User authentication (register/login).
2. Live voice interaction (microphone required).
3. Optional camera frame analysis for FER.
4. Parallel perception pipeline (STT + SER + FER).
5. Emotion-fused assistant response generation.
6. Streaming NDJSON events for real-time UI updates.
7. Questionnaire completion and storage.
8. User-scoped admin report with structured risk framing.

---

## 3) System Architecture

### 3.1 Frontend (React + Vite)

Frontend entry points:

- `frontend/src/main.jsx`
- `frontend/src/App.jsx`

Primary UI areas:

- `components/Login.jsx`: registration/login forms.
- `components/Dashboard.jsx`: route hub for all workflows.
- `pages/UnifiedEmotionPage.jsx`: live session orchestration, streaming parser, incremental assistant rendering, audio playback queue.
- `pages/QuestionnairesPage.jsx`: template rendering, answer capture, submit, and history.
- `pages/AdminPage.jsx`: risk report, metrics, timelines, and profile analytics.

Key frontend architectural points:

- Route-level lazy loading via React lazy + Suspense.
- Local auth session persistence via `localStorage` key `serenity_user`.
- NDJSON streaming consumer for `/api/interact/stream` and `/api/chat/stream`.
- Browser speech fallback when backend TTS cannot be played.
- Incremental list rendering for large admin chat datasets.

### 3.2 Backend (FastAPI)

Backend core:

- `backend/main.py`: API surface, orchestration, stream pipeline, risk summary pipeline, DB persistence integration.
- `backend/cloud_llm_core.py`: optimized cloud client with pooled HTTP session, streaming token parser, hallucination/artifact cutoff logic.
- `backend/audio_core.py`: SER model loading and inference (TFLite backend with optional delegate).
- `backend/emotion_core.py`: FER model loading and inference (OpenCV + TFLite).
- `backend/database.py`: SQLAlchemy engine/session helpers, SQLite pragmas, query and persistence functions.
- `backend/models.py`: ORM schema.
- `backend/questionnaires_data.py`: questionnaire definitions, normalization, scoring, severity, and clinical flag logic.

### 3.3 Data Layer

Storage: local SQLite database (`serenity.db`).

Tables:

- `users`
- `conversation_turns`
- `sessions`
- `emotions`
- `questionnaire_results`

SQLite performance pragmas applied at connect-time:

- `journal_mode=WAL`
- `synchronous=NORMAL`
- `temp_store=MEMORY`
- negative `cache_size` in KB (configurable)
- `mmap_size=268435456`

---

## 4) Runtime Data Flows

### 4.1 Voice Interaction Flow

1. Frontend records audio (MediaRecorder) and optionally captures image frame.
2. Backend `/api/interact` or `/api/interact/stream` receives multipart payload.
3. Temporary audio file is created and auto-cleaned with context manager.
4. Perception tasks run concurrently:
	 - STT (`faster-whisper` preferred, `openai-whisper` fallback)
	 - SER (audio TFLite)
	 - FER (frame TFLite, if image is present)
5. Probability fusion determines dominant emotion.
6. User text is sent to cloud LLM client.
7. Assistant result is persisted with emotions.
8. Optional TTS audio is generated and returned or streamed as sentence segments.

### 4.2 Text Chat Flow

1. Frontend sends JSON to `/api/chat` or `/api/chat/stream`.
2. Backend bypasses perception stack (neutral emotions).
3. LLM result is generated and persisted.
4. Optional TTS is produced in non-stream mode.

### 4.3 Admin Observability Flow

1. Frontend requests `/api/admin/overview?username=...`.
2. Backend gathers:
	 - recent conversation turn summaries,
	 - recent sessions with emotion timeline,
	 - questionnaire results,
	 - aggregate counts.
3. Backend computes:
	 - top emotions,
	 - negative emotion ratio,
	 - distress keyword signals,
	 - screening trends,
	 - engagement score,
	 - risk score and risk band.
4. Summary text is generated using cloud LLM (with strict formatting prompt) or fallback heuristic summary.
5. User-scoped overview payload is cached with TTL and returned.

---

## 5) Backend API Contract

Base URL (default local): `http://127.0.0.1:5000`

### 5.1 Auth

#### POST /register
Request JSON:

```json
{
	"username": "alice",
	"password": "secret"
}
```

Response JSON:

```json
{
	"message": "Registration successful",
	"username": "alice"
}
```

Notes:

- Password hashing uses bcrypt in current implementation.

#### POST /login
Request JSON:

```json
{
	"username": "alice",
	"password": "secret"
}
```

Response JSON:

```json
{
	"message": "Login successful",
	"username": "alice"
}
```

### 5.2 Questionnaires

#### GET /api/questionnaires/templates
Optional query param:

- `types=PHQ-9,GAD-7` (comma-separated)

Returns template metadata, options, and question lists.

#### POST /api/questionnaires/submit
Request JSON:

```json
{
	"username": "alice",
	"questionnaire_type": "PHQ-9",
	"answers": [1,2,0,1,1,0,2,0,0],
	"submitted_at": "2026-04-12T12:00:00Z"
}
```

Returns stored record ID and computed score/severity.

#### GET /api/questionnaires/history
Query params:

- `username` (required)
- `limit` (optional)

### 5.3 Admin

#### GET /api/admin/overview
Query params:

- `username` (required)
- `limit` (optional, clamped)
- `include_answers` (optional)

Returns:

- professional 6-line summary,
- summary source,
- metrics,
- profile,
- computed clinical parameters,
- top emotions,
- recent chats/sessions/questionnaire rows,
- flagged user object (for active flags).

### 5.4 Interactions

#### POST /api/interact
Multipart form fields:

- `username` (required)
- `file` (required audio file)
- `image` (optional base64 data URL or base64 payload)
- `user_message` (optional fallback text)

Returns `InteractResponse` with emotion fields, transcription, assistant reply, optional TTS audio, and error list.

#### POST /api/interact/stream
Same multipart inputs as `/api/interact`.

Response media type: `application/x-ndjson`

#### POST /api/chat
Request JSON:

```json
{
	"username": "alice",
	"message": "I am feeling overwhelmed today."
}
```

Returns `InteractResponse` with neutral emotions and generated assistant text.

#### POST /api/chat/stream
Same payload as `/api/chat`.

Response media type: `application/x-ndjson`

---

## 6) Streaming NDJSON Event Protocol

Backend stream events currently emitted:

- `user_text`
- `emotion`
- `assistant_delta`
- `assistant_sentence`
- `assistant_sentence_tts`
- `error`
- `final`

Ordering rules:

- Voice stream (`/api/interact/stream`) emits `emotion` first, then `user_text`.
- Text stream (`/api/chat/stream`) emits `user_text` first, then `emotion`.

Frontend behavior:

- `assistant_delta` progressively appends assistant text.
- `assistant_sentence_tts` sentence audio segments are queued in sequence.
- `final` closes turn state and confirms final response payload.

---

## 7) Clinical Logic and Risk Formulation

Questionnaire engines:

- PHQ-9 scoring and severity thresholds.
- GAD-7 scoring and severity thresholds.
- PCL-5 scoring and severity thresholds.

Admin risk construction includes:

- active screening flags,
- severity point mapping,
- distress language pattern detection,
- negative emotion ratio,
- engagement score from activity volume.

Risk labels:

- `stable`
- `monitor`
- `elevated`

Summary generation strategy:

- Prefer cloud LLM professional summary prompt.
- If unavailable/timeouts/errors, deterministic fallback summary is returned.
- Recent summary is cached to reduce repeated generation overhead.

---

## 8) Edge Performance Strategy

### 8.1 Model Runtime

- TFLite interpreter reuse (SER and FER).
- Invoke locks for thread-safe interpreter calls.
- Optional XNNPACK delegate load with CPU fallback.
- OpenCV thread cap through environment configuration.

### 8.2 Streaming and LLM

- Shared requests session and configurable HTTP pools.
- Fast token-level artifact cutoff (`*` and `#`).
- Rolling kill-phrase detector with bounded memory tail.
- Conditional text normalization for lower per-token CPU cost.

### 8.3 Data and Memory

- SQLite WAL and memory-oriented pragmas.
- Bounded admin response limits and user-scoped TTL cache.
- Temporary audio file cleanup via context manager.

### 8.4 Frontend UX Efficiency

- Route lazy loading.
- RequestAnimationFrame-based throttled stream UI updates.
- Incremental chat pagination in admin view.

---

## 9) Configuration Reference

### 9.1 Backend Environment Variables

The following keys are currently referenced by backend code:

- `SERENITY_ADMIN_DEFAULT_LIMIT`
- `SERENITY_ADMIN_MAX_LIMIT`
- `SERENITY_ADMIN_OVERVIEW_CACHE_TTL_SECONDS`
- `SERENITY_ADMIN_SUMMARY_CACHE_TTL_SECONDS`
- `SERENITY_ADMIN_SUMMARY_TIMEOUT_SECONDS`
- `SERENITY_CLOUD_LLM_CONNECT_TIMEOUT_SECONDS`
- `SERENITY_CLOUD_LLM_KILL_PHRASES`
- `SERENITY_CLOUD_LLM_POOL_CONNECTIONS`
- `SERENITY_CLOUD_LLM_POOL_MAXSIZE`
- `SERENITY_CLOUD_LLM_TIMEOUT_SECONDS`
- `SERENITY_CLOUD_LLM_URL`
- `SERENITY_EDGE_OPTIMIZED_MODE`
- `SERENITY_FER_CV2_THREADS`
- `SERENITY_FER_FACE_MIN_NEIGHBORS`
- `SERENITY_FER_FACE_MIN_SIZE`
- `SERENITY_FER_FACE_SCALE_FACTOR`
- `SERENITY_FER_MAX_FRAME_SIDE`
- `SERENITY_FER_TFLITE_THREADS`
- `SERENITY_SER_AUDIO_DURATION_SECONDS`
- `SERENITY_SER_AUDIO_OFFSET_SECONDS`
- `SERENITY_SER_AUDIO_SAMPLE_RATE`
- `SERENITY_SER_TFLITE_THREADS`
- `SERENITY_SQLITE_CACHE_KB`
- `SERENITY_TFLITE_XNNPACK_DELEGATE`
- `SERENITY_TTS_ENABLED`
- `SERENITY_TTS_VOICE`
- `SERENITY_WHISPER_MODEL_SIZE`

### 9.2 Frontend Environment Variables

- `VITE_API_BASE_URL`
- `VITE_SHOW_PROVISIONAL_ASSISTANT_TEXT`

### 9.3 Practical Edge Defaults

`Start_App.bat` sets practical defaults for local edge-like operation, including:

- BLAS/OpenMP thread caps,
- cloud timeout/pool values,
- TFLite thread allocations,
- FER constraints,
- admin limit defaults.

---

## 10) Setup and Local Run

### 10.1 Prerequisites

- Python 3.10+ recommended.
- Node.js 18+ recommended.
- npm.

### 10.2 Python Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 10.3 Install Dependencies

General profile:

```powershell
pip install -r requirements.txt
```

Edge profile:

```powershell
pip install -r requirements-edge.txt
```

### 10.4 Frontend Dependencies

```powershell
cd frontend
npm install
cd ..
```

### 10.5 Start Backend

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 5000
```

### 10.6 Start Frontend

```powershell
cd frontend
npm run dev
```

Open `http://localhost:5173/login`.

### 10.7 One-Click Windows Launcher

```powershell
Start_App.bat
```

---

## 11) Development and Validation Commands

Backend syntax check:

```powershell
python -m py_compile backend/main.py backend/audio_core.py backend/emotion_core.py backend/cloud_llm_core.py backend/database.py backend/questionnaires_data.py backend/models.py
```

Frontend production build:

```powershell
cd frontend
npm run build
```

---

## 12) Project Structure (Current)

```text
FYP/
	README.md
	Start_App.bat
	requirements.txt
	requirements-edge.txt
	.gitignore
	backend/
		main.py
		cloud_llm_core.py
		audio_core.py
		emotion_core.py
		database.py
		models.py
		questionnaires_data.py
		fer_model.tflite
		ser_model.tflite
	frontend/
		package.json
		vite.config.js
		tailwind.config.js
		postcss.config.js
		index.html
		src/
			App.jsx
			main.jsx
			index.css
			components/
				Login.jsx
				Dashboard.jsx
			pages/
				UnifiedEmotionPage.jsx
				QuestionnairesPage.jsx
				AdminPage.jsx
```

---

## 13) Security, Privacy, and Safety Notes

- User credentials are hashed with bcrypt at registration.
- CORS is open by default for development convenience and should be restricted in production.
- Distress signal matching is keyword-based and should not be treated as diagnostic certainty.
- This platform is a support and analytics tool, not a substitute for licensed medical diagnosis.
- In emergency-risk contexts, local policy-compliant escalation workflows should be integrated.

---

## 14) Known Limitations

1. No JWT/session token auth layer; frontend uses localStorage user marker.
2. No comprehensive automated test suite yet.
3. Cloud LLM availability depends on external endpoint reliability.
4. Legacy session/emotion tables and modern conversation turns coexist; migration unification may be desirable.
5. Model artifact versioning and checksum validation are not yet formalized.

---

## 15) Recommended Next Improvements

1. Add token-based auth and role separation.
2. Add backend API and frontend integration tests.
3. Add schema migrations (Alembic) for controlled DB evolution.
4. Add structured observability (request IDs, latency histograms, error metrics).
5. Add model artifact manifest with hashes and startup integrity checks.
6. Add production deployment profiles (Docker/systemd) with secure env handling.

---

## 16) License and Third-Party Assets

Use and distribution must follow:

- this repository license,
- licenses of Python/Node dependencies,
- licenses and terms of model artifacts,
- deployment-region privacy and healthcare compliance requirements where applicable.
