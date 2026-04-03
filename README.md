# SERENITY

Smart Emotion Recognition and Neural Intervention Technology (SERENITY) is a multimodal, therapist-style assistant developed as a Final Year Project.

The project combines computer vision, speech processing, retrieval-augmented generation, and response synthesis into one interaction loop:
- Face emotion recognition (TFLite FER model)
- Speech emotion recognition (TFLite SER model)
- Speech-to-text transcription (Whisper Tiny)
- Empathetic response generation (Qwen2.5-1.5B-Instruct in 4-bit NF4)
- Text-to-speech synthesis (edge-tts)
- Local conversation memory (SQLite via SQLAlchemy)

The current implementation is local-first and was tuned to work on constrained hardware. The architecture is also ready to evolve into a hybrid Raspberry Pi + AWS model-offload design.

## Table of Contents
- [1. Project Vision](#1-project-vision)
- [2. Full Functional Overview](#2-full-functional-overview)
- [3. Runtime Architecture](#3-runtime-architecture)
- [4. End-to-End Processing Pipeline](#4-end-to-end-processing-pipeline)
- [5. API Contract (Detailed)](#5-api-contract-detailed)
- [6. Model and AI Stack](#6-model-and-ai-stack)
- [7. Data and Persistence Design](#7-data-and-persistence-design)
- [8. Full Repository Structure and Folder Purposes](#8-full-repository-structure-and-folder-purposes)
- [9. File-Level Responsibility Map](#9-file-level-responsibility-map)
- [10. Environment and Configuration](#10-environment-and-configuration)
- [11. Setup and Run](#11-setup-and-run)
- [12. Validation Checklist](#12-validation-checklist)
- [13. Troubleshooting Guide](#13-troubleshooting-guide)
- [14. Security and Limitations](#14-security-and-limitations)
- [15. Hybrid Deployment Target (Pi + Azure)](#15-hybrid-deployment-target-pi--azure)

## 1. Project Vision
SERENITY is designed to provide emotionally-aware conversational assistance by fusing visual and vocal affective signals with therapeutic language generation.

Core objectives:
1. Capture emotional cues from both face and voice.
2. Merge both modalities into one dominant affect signal.
3. Generate concise, empathetic intervention guidance using RAG + LLM.
4. Return output in both text and speech.
5. Persist conversation context per user for continuity.

## 2. Full Functional Overview

### User Journey
1. User authenticates from the frontend login interface.
2. User enters the unified live session page.
3. Browser captures camera frame + microphone audio.
4. Frontend submits multimodal payload to backend.
5. Backend performs parallel inference:
  - Whisper transcription
  - SER inference
  - FER inference
6. Backend fuses confidence outputs and computes dominant emotion.
7. Backend retrieves recent user turns from SQLite and RAG context from FAISS.
8. Qwen generates a short emotionally-aligned response.
9. edge-tts synthesizes audio reply.
10. Frontend renders emotions, transcript, generated response, and optional speech playback.
11. Backend stores complete turn log for future context.

### Supported Interaction Modes
1. Unified multimodal interaction through /api/interact.
2. Text-only fallback interaction through /api/chat.
3. Legacy single-modality diagnostics:
  - /detect_emotion for camera only
  - /analyze_audio for microphone only

## 3. Runtime Architecture

### Active Application Path
- Frontend: React + Vite SPA
- Backend: FastAPI app in backend/main.py
- Data store: SQLite (serenity.db)
- Models: local TFLite + Whisper + quantized Qwen + FAISS assets

### Control and Reliability Patterns Implemented
1. Background RAG initialization on startup to reduce API cold-start blocking.
2. Timeout guards for transcription, emotion inference, LLM generation, and TTS.
3. Async semaphore around LLM generation to prevent GPU contention.
4. Graceful fallback responses on model failure or timeout.
5. Per-request temporary files with cleanup to reduce cross-request collisions.

## 4. End-to-End Processing Pipeline

Primary endpoint: POST /api/interact

### Request Inputs
- username (required)
- image (optional base64 camera frame)
- file (optional audio blob)
- user_message (optional text fallback)

### Parallel Inference Stage
If audio is present:
1. Transcription task uses Whisper Tiny.
2. Speech emotion task uses SER TFLite model.

If image is present:
1. Face emotion task uses FER TFLite model with Haar cascade face detection.

### Emotion Fusion Stage
1. Speech and face outputs are normalized into probability vectors.
2. Alias normalization is applied (for example, surprised to surprise).
3. Weighted average (50/50) fusion selects dominant emotion.
4. If only one modality is available, that modality becomes dominant.

### Response Generation Stage
1. User input text is selected from transcript or fallback message.
2. Recent turn history is loaded from SQLite.
3. RAG context is retrieved from FAISS embeddings.
4. LLM prompt is built with emotion + history + retrieved context.
5. Qwen generates concise therapeutic guidance.

### Output Stage
1. Response is converted to speech with edge-tts (if available).
2. Turn data is persisted in conversation_turns table.
3. Unified payload returns inference, response, audio, and non-fatal errors.

## 5. API Contract (Detailed)

### Authentication
1. POST /register
  - Request JSON: username, password
  - Creates user if username is unique

2. POST /login
  - Request JSON: username, password
  - Validates credentials and returns username

### Health
1. GET /health
  - Returns service status and rag_loaded flag

### Legacy Diagnostic Endpoints
1. POST /detect_emotion
  - Request JSON: image, optional user_message
  - Returns emotion, confidence, optional ai_message, optional error

2. POST /analyze_audio
  - Multipart file upload
  - Returns emotion, confidence, optional ai_message, optional error

### Unified Multimodal Endpoint
1. POST /api/interact
  - Multipart form
  - Required: username
  - Optional: image, file, user_message
  - Response fields:
    - dominant_emotion
    - speech_emotion
    - face_emotion
    - transcription
    - llm_response
    - tts_audio_base64
    - errors (array)

### Text Chat Endpoint
1. POST /api/chat
  - Request JSON: username, message
  - Returns same response schema as /api/interact

## 6. Model and AI Stack

### FER (Face Emotion Recognition)
- File: backend/fer_model.tflite
- Inference module: backend/emotion_core.py
- Input processing:
  - Base64 decode
  - OpenCV frame decode
  - Grayscale conversion
  - Haar face detection
  - ROI resize to 48x48
  - Normalization and tensor shaping

### SER (Speech Emotion Recognition)
- File: backend/ser_model.tflite
- Inference module: backend/audio_core.py
- Input processing:
  - librosa audio load (3 sec window)
  - MFCC extraction
  - Dynamic tensor shaping based on model input schema
  - Supports 2D, 3D, and 4D model signatures

### STT (Transcription)
- Library: openai-whisper
- Model: tiny (CPU route by default in current backend)

### RAG and LLM
- Runtime module: backend/llm_core.py
- Embeddings: all-MiniLM-L6-v2
- Vector index: FAISS inner-product index
- LLM: Qwen/Qwen2.5-1.5B-Instruct with bitsandbytes 4-bit NF4
- Generation style:
  - concise therapeutic response
  - emotion-aware
  - one actionable next step orientation

### TTS
- Library: edge-tts
- Output returned as base64 audio payload

## 7. Data and Persistence Design

### Database
- Engine: SQLite
- ORM: SQLAlchemy
- Primary runtime file: serenity.db

### Tables and Purpose
1. users
  - credentials and user identity
2. sessions
  - legacy session conversation metadata
3. emotions
  - legacy emotion timeline records
4. conversation_turns
  - active multimodal memory table used by current pipeline
  - stores user text, assistant text, dominant/speech/face emotion, timestamp

### Memory Strategy
1. Recent turns are fetched with recency order and reversed for chronological prompt context.
2. Persisted turns are used for continuity in subsequent requests.

## 8. Full Repository Structure and Folder Purposes

```text
FYP/
  .git/                         -> Git metadata (repository internals)
  .github/                      -> Repository workflows and migration notes
  .vscode/                      -> Local editor settings
  backend/                      -> Main backend runtime, AI modules, model artifacts
  database_app/                 -> Isolated DB connectivity tests
  frontend/                     -> Active React frontend used by current app
  FYP_frontend/                 -> Legacy/alternate frontend deployment workspace
  mess/                         -> Experimental artifacts, exports, and archived assets
  presentations/                -> Academic defense and progress presentation files
  node_modules/                 -> Root-level JS dependencies (local environment artifact)
  __pycache__/                  -> Python bytecode cache
  README.md                     -> Project documentation
  requirements.txt              -> Python dependency specification
  build_kb.py                   -> Offline FAISS knowledge base build script
  Start_App.bat                 -> Legacy one-click launcher (Flask + frontend)
  test_vision.py                -> Legacy vision test helper
  tflite_loader.py              -> Generic helper class for loading TFLite models
  FYP.code-workspace            -> VS Code workspace descriptor
  database.db                   -> Local SQLite artifact (environment dependent)
  serenity.db                   -> Main local SQLite database used by runtime
  resnet_emotion_model.pth      -> Legacy/training artifact for vision pipeline
  SER_model.keras               -> Legacy/training artifact for audio pipeline
  facial-emotion-model-tflite-conversion.ipynb -> FER conversion notebook
  speech-emotion-model-tflite-conversion.ipynb -> SER conversion notebook
  tmp_*.wav                     -> Temporary runtime audio artifacts
```

### backend folder (detailed)
```text
backend/
  main.py                       -> Active FastAPI app and orchestration pipeline
  llm_core.py                   -> RAG engine, embedding pipeline, quantized LLM runtime
  emotion_core.py               -> FER inference with OpenCV + TFLite
  audio_core.py                 -> SER inference with librosa + TFLite
  database.py                   -> SQLAlchemy engine/session and helper persistence methods
  models.py                     -> SQLAlchemy table definitions
  schemas.py                    -> Pydantic schema definitions (legacy/minimal)
  app.py                        -> Legacy Flask backend path
  llm-finallll.ipynb            -> LLM experimentation notebook
  fer_model.tflite              -> Face emotion runtime model
  ser_model.tflite              -> Speech emotion runtime model
  serenity_faiss.index          -> Persisted FAISS vector index
  serenity_chunks.pkl           -> Persisted knowledge chunks for RAG
  tinyllama_local/              -> Local fine-tuned adapter/tokenizer assets
  offload_weights/              -> Placeholder folder for model offload weights
  venv/                         -> Local backend virtual environment
  __pycache__/                  -> Python cache files
```

### frontend folder (detailed)
```text
frontend/
  src/
   App.jsx                     -> Route guards and authenticated navigation
   main.jsx                    -> React bootstrap entrypoint
   index.css                   -> Tailwind/global styles
   pages/
    UnifiedEmotionPage.jsx    -> Main multimodal live session experience
   components/
    Login.jsx                 -> Registration/login UI and backend auth calls
    Dashboard.jsx             -> Session launcher UI
    CameraEmotionDetection.jsx -> Legacy camera-only emotion screen
    SpeechEmotionDetection.jsx -> Legacy voice-only emotion screen
  package.json                  -> Frontend dependencies and scripts
  package-lock.json             -> NPM lockfile
  postcss.config.js             -> PostCSS config
  tailwind.config.js            -> Tailwind config
  vite.config.js                -> Vite bundler config
  index.html                    -> Vite HTML template
  MIGRATION_PLAN.md             -> Internal migration planning notes
  node_modules/                 -> Frontend dependencies
```

### database_app folder
```text
database_app/
  test_app.py                   -> Minimal FastAPI route to validate SQLite write/read
  serenity.db                   -> Local DB test artifact
  venv/                         -> Isolated test environment
  __pycache__/                  -> Python cache
```

### FYP_frontend folder
```text
FYP_frontend/
  FYP_deployment/               -> Legacy deployment-oriented frontend/backend sample bundle
```

### mess folder
```text
mess/
  Database framework guidance.pdf -> Research/reference material
  finetuned LLM.zip             -> Archived model artifact
  results/                      -> Mirrored TinyLlama adapter artifacts
```

## 9. File-Level Responsibility Map

### Core backend entrypoint
1. backend/main.py
  - Defines FastAPI app, CORS, startup/shutdown events
  - Initializes DB tables
  - Implements auth, health, multimodal, and chat routes
  - Performs async orchestration of STT/SER/FER tasks
  - Handles emotion fusion and response generation
  - Persists conversation turns and handles TTS

### AI orchestration and RAG
1. backend/llm_core.py
  - Scrapes knowledge sources (offline builder)
  - Chunks and embeds text corpus
  - Builds and persists FAISS index
  - Loads quantized Qwen model
  - Generates single-modality and multimodal responses

### Modality inference modules
1. backend/emotion_core.py
  - Loads FER TFLite model once
  - Executes safe face prediction with fallback error payload

2. backend/audio_core.py
  - Loads SER TFLite model once
  - Dynamically adapts feature tensor to model shape
  - Executes safe speech prediction with fallback error payload

### Data layer
1. backend/database.py
  - SQLAlchemy engine/session factory
  - Fetches recent turns
  - Persists new turns

2. backend/models.py
  - Defines user/session/emotion/conversation_turn schema

## 10. Environment and Configuration

### Runtime Variables
1. SERENITY_SKIP_RAG_STARTUP
  - true skips RAG model startup at boot
  - useful for low-memory boot diagnostics

2. VITE_API_BASE_URL
  - frontend API target override
  - defaults to http://127.0.0.1:5000

### Dependency Overview
Python dependencies are managed via requirements.txt.
Frontend dependencies are managed via frontend/package.json.

## 11. Setup and Run

### Clone
```powershell
git clone https://github.com/mtahaarif/Smart-Emotion-Recognition-and-Neural-Intervention-Technology-SERENITY-.git
Set-Location Smart-Emotion-Recognition-and-Neural-Intervention-Technology-SERENITY-
```

### Python environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Frontend dependencies
```powershell
Set-Location frontend
npm install
Set-Location ..
```

### Run backend (active FastAPI path)
```powershell
Set-Location g:/FYP/FYP
& "g:/FYP/.venv/Scripts/python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 5000
```

### Run frontend
```powershell
Set-Location g:/FYP/FYP/frontend
npm run dev
```

### Application routes
1. Login page: http://localhost:5173/login
2. Dashboard: http://localhost:5173/dashboard
3. Unified live session: http://localhost:5173/emotion/live

## 12. Validation Checklist
1. Backend health check
```powershell
curl http://127.0.0.1:5000/health
```

2. Frontend production build
```powershell
Set-Location frontend
npm run build
```

3. Interaction smoke test
  - open live page
  - provide camera and microphone permission
  - submit one multimodal interaction
  - verify response text, emotion labels, and optional TTS

## 13. Troubleshooting Guide
1. Backend unreachable
  - verify uvicorn is running
  - call /health

2. Camera or microphone unavailable
  - check browser permissions
  - refresh device list in UI
  - verify OS privacy settings

3. Slow first response
  - first call includes model warm-up and cache loading

4. TTS missing
  - edge-tts unavailable or blocked network path
  - text response still returns

5. Port conflicts
```powershell
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
```

## 14. Security and Limitations
1. Passwords are stored plaintext in current prototype and must be hashed before production.
2. Model artifacts are large and increase clone/setup time.
3. Mixed Flask and FastAPI legacy files are still present for backward compatibility and reference.
4. Latency depends on local CPU/GPU saturation and model warm-up state.
5. TTS path can be network-dependent.

## 15. Hybrid Deployment Target (Pi + AWS)
Planned architecture for the hardware-compulsory deployment to bypass local compute constraints while adhering to a strict student cloud budget:

### Keep on Raspberry Pi 5 (Edge Node)
1. Frontend UI, camera/mic capture
2. Local auth/session management
3. Sensitive local data storage (SQLite) and logs
4. TFLite Edge Inference (Facial & Speech Emotion via XNNPACK delegate)
5. Transcription (faster-whisper / Whisper Tiny)
6. Final response rendering and Edge-TTS audio playback

### Move to AWS Cloud Compute (Cloud Node)
1. Heavy LLM inference worker (Qwen 2.5 1.5B)
2. RAG embedding and FAISS retrieval service

### Recommended AWS Serverless / CPU Architecture
*To circumvent student GPU quota limits and minimize costs:*
1. **Amazon SQS (Simple Queue Service):** FIFO queues (`requests.fifo` and `results.fifo`) for asynchronous, resilient Edge-to-Cloud messaging that prevents Pi timeouts.
2. **AWS EC2 (Graviton CPU):** A `c7g.xlarge` ARM-based Spot Instance to run the LLM inference.
3. **llama.cpp (GGUF):** The Qwen LLM is quantized to 4-bit GGUF format to run blazingly fast on the EC2 Graviton CPU without needing an expensive NVIDIA GPU.
4. **AWS IAM:** Strictly scoped access keys to secure the SQS payload transmission.

This hybrid event-driven architecture preserves privacy-sensitive data on the Edge while offloading heavy LLM generation to a highly cost-optimized AWS CPU instance.

