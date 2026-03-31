# SERENITY

Smart Emotion Recognition and Neural Intervention Technology (SERENITY) is a local-first multimodal mental-care assistant developed as a Final Year Project.

The system combines:
- Facial emotion recognition (TFLite)
- Speech emotion recognition (TFLite)
- Speech-to-text transcription (Whisper Tiny)
- Retrieval-augmented response generation (Qwen 2.5 1.5B, 4-bit NF4)
- Text-to-speech playback (edge-tts)
- Conversation memory (SQLite)

It is optimized to run on a single-machine setup with an NVIDIA GTX 1650 (4 GB VRAM) by prioritizing GPU memory for the quantized LLM while keeping supporting models CPU-oriented.

## Table of Contents
- [Project Goals](#project-goals)
- [System Architecture](#system-architecture)
- [Core Pipeline](#core-pipeline)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [API Reference](#api-reference)
- [Memory and Hardware Routing Strategy](#memory-and-hardware-routing-strategy)
- [Setup](#setup)
- [Run](#run)
- [Validation and Testing](#validation-and-testing)
- [Troubleshooting](#troubleshooting)
- [Known Constraints](#known-constraints)

## Project Goals
1. Detect user emotional state from both voice and facial signals.
2. Fuse multimodal signals into a robust dominant affect estimate.
3. Generate empathetic, psychology-aware intervention text through local RAG + LLM.
4. Return audible feedback through TTS for a conversational therapist-like interaction loop.
5. Maintain short-term memory to personalize successive responses.

## System Architecture
High-level modules:

1. Frontend (React + Vite)
- Login and session routing.
- Unified live interaction page at `/emotion/live`.
- Media capture (camera + microphone) and optional text chat.
- Real-time rendering of detected emotions, transcript, and assistant reply.

2. Backend (FastAPI)
- Authentication endpoints.
- Multimodal orchestration endpoint (`/api/interact`).
- Text-only chat endpoint (`/api/chat`).
- Model wrappers for FER/SER/Whisper/Qwen/TTS.

3. Data Layer (SQLite + SQLAlchemy)
- Users table and turn-level conversation memory.
- Efficient retrieval of recent turns for prompt context.

4. Knowledge Layer (RAG)
- FAISS vector index + chunk store.
- Sentence-transformers embedding retrieval.

## Core Pipeline
Primary interaction endpoint: `POST /api/interact`

1. Frontend sends:
- `username`
- `image` (base64 frame)
- `file` (audio blob)
- optional fallback text

2. Backend executes parallel tasks:
- Whisper Tiny transcription
- Speech emotion inference
- Facial emotion inference

3. Late fusion:
- Converts speech and face outputs to probability vectors.
- Applies 50/50 averaging for dominant emotion selection.
- Handles partial modality failures with graceful fallback.

4. RAG + LLM generation:
- Retrieves recent turns from SQLite.
- Retrieves semantic context from FAISS.
- Builds prompt with user text + detected affect + history + KB context.
- Generates empathetic response from Qwen 2.5 1.5B (4-bit quantized).

5. TTS:
- edge-tts synthesizes response audio.
- Base64 payload returned to frontend.

6. Persistence:
- Logs turn, dominant emotion, modality-specific emotions, and assistant reply.

## Technology Stack
Backend:
- Python 3.10+
- FastAPI, Uvicorn
- SQLAlchemy, SQLite
- TensorFlow 2.18 (TFLite runtime path)
- OpenCV, librosa, NumPy
- openai-whisper
- transformers, bitsandbytes, torch
- sentence-transformers, faiss-cpu
- edge-tts

Frontend:
- React 18
- Vite 5
- react-router-dom
- axios
- Tailwind CSS
- lucide-react

## Repository Structure

```text
FYP/
  backend/
    main.py
    llm_core.py
    audio_core.py
    emotion_core.py
    database.py
    models.py
    fer_model.tflite
    ser_model.tflite
    serenity_faiss.index
    serenity_chunks.pkl
    tinyllama_local/
  frontend/
    src/
      App.jsx
      components/
      pages/UnifiedEmotionPage.jsx
    package.json
  presentations/
  requirements.txt
  README.md
```

## API Reference

Authentication:
- `POST /register`
- `POST /login`

Health:
- `GET /health`

Legacy/diagnostic emotion endpoints:
- `POST /detect_emotion`
- `POST /analyze_audio`

Unified multimodal:
- `POST /api/interact`
  - multipart form fields: `username`, optional `image`, optional `file`, optional `user_message`
  - returns:
    - dominant_emotion
    - speech_emotion
    - face_emotion
    - transcription
    - llm_response
    - tts_audio_base64 (optional)
    - errors[]

Text-only RAG chat:
- `POST /api/chat`
  - JSON: `{ "username": "...", "message": "..." }`
  - returns same response envelope as `/api/interact`

## Memory and Hardware Routing Strategy
Target hardware: GTX 1650 (4 GB VRAM)

Design intent:
- Reserve GPU primarily for quantized Qwen generation.
- Keep Whisper Tiny and TFLite inference CPU-biased for memory safety.
- Run RAG initialization in background to keep API responsive during boot.
- Serialize LLM generation with an async semaphore to avoid GPU contention.
- Use timeout guards for Whisper, emotion inference, LLM generation, and TTS.

## Setup

1. Clone:

```powershell
git clone https://github.com/mtahaarif/Smart-Emotion-Recognition-and-Neural-Intervention-Technology-SERENITY-.git
Set-Location Smart-Emotion-Recognition-and-Neural-Intervention-Technology-SERENITY-
```

2. Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Frontend dependencies:

```powershell
Set-Location frontend
npm install
Set-Location ..
```

## Run

Backend:

```powershell
Set-Location g:/FYP/FYP
& "g:/FYP/.venv/Scripts/python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 5000
```

Frontend:

```powershell
Set-Location g:/FYP/FYP/frontend
npm run dev
```

App route:
- Login page: `http://localhost:5173/login`
- Unified session page: `http://localhost:5173/emotion/live`

## Validation and Testing
Recommended quick checks:

1. Health:

```powershell
curl http://127.0.0.1:5000/health
```

2. Frontend build:

```powershell
Set-Location frontend
npm run build
```

3. Multimodal endpoint smoke test (audio + image payload) from script or UI.

## Troubleshooting

1. "Server not connecting"
- Confirm backend process is running.
- Check `/health` first.

2. Camera/mic device not found
- Verify browser permissions.
- Refresh devices from UI controls.
- Check OS-level privacy settings for camera/microphone.

3. Port in use

```powershell
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
```

4. Slow first response
- First request may trigger model warm-up and cache initialization.

5. TTS unavailable
- Network restrictions to TTS provider can cause fallback to text-only response.

## Known Constraints
- Passwords are currently stored in plaintext for prototype use; production requires hashing and stronger auth.
- Large model assets increase repository and setup footprint.
- End-to-end latency depends on local CPU/GPU saturation and first-load caching.

---

SERENITY is built as an end-to-end cyber-physical multimodal intervention stack for research and academic demonstration, emphasizing local inference, resilience, and practical deployment on constrained consumer hardware.
