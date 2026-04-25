# SERENITY: Smart Emotion Recognition and Neural Intervention Technology

**An Edge-Optimized, Multimodal AI Platform for Measurement-Based Mental Health Support**

SERENITY is an applied Artificial Intelligence Electronic Health Record (EHR) and telehealth platform. It combines real-time speech and facial emotion recognition, clinical psychometric screening, and cloud-assisted Large Language Model (LLM) therapeutic dialogue to provide data-informed, continuous mental health support.

This repository serves as the complete technical documentary, deployment guide, and architectural overview for the SERENITY Final Year Project (FYP), specifically optimized for edge deployment on the **Raspberry Pi 5**.

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Background & Motivation](#2-background--motivation)
3. [Core Features & System Workflows](#3-core-features--system-workflows)
4. [System Architecture & Technical Innovations](#4-system-architecture--technical-innovations)
5. [Complete Directory Structure](#5-complete-directory-structure)
6. [Raspberry Pi 5 Deployment Guide](#6-raspberry-pi-5-deployment-guide)
7. [Environment Configuration](#7-environment-configuration)
8. [Troubleshooting & Edge Limitations](#8-troubleshooting--edge-limitations)
9. [Academic & Clinical Disclaimer](#9-academic--clinical-disclaimer)

---

## 1. Executive Summary

Mental health support systems typically fail by either being user-friendly but clinically shallow, or clinically rigid but highly inaccessible. SERENITY bridges this gap through a **Tri-Pillar Architecture**:
* **Passive Perception:** Continuous, edge-computed monitoring of vocal and facial affect.
* **Active Screening:** Measurement-Based Care (MBC) tracking longitudinal symptom trajectories via standardized psychometrics (PHQ-9, GAD-7, PCL-5).
* **Dynamic Intervention:** A clinical routing engine that dynamically shifts the LLM's therapeutic framework (CBT, DBT, ACT) based on real-time risk scores and detected cognitive distortions.

SERENITY provides users with empathetic, unrestricted support while automatically synthesizing structured **SBAR (Situation, Background, Assessment, Recommendation)** handoff reports for human psychologists.

---

## 2. Background & Motivation

The global shortage of mental health professionals has driven a surge in AI chatbots. However, standard LLMs lack clinical memory, multimodal awareness, and safety guardrails. They cannot detect when a patient's tone of voice contradicts their text, nor can they track symptom degradation over months.

**The SERENITY Objective:**
To engineer a low-latency, edge-deployable system that acts as a "clinical co-pilot." It does not diagnose; rather, it observes, stabilizes, and structures patient data. By running heavy perception models (STT, SER, FER) locally on a Raspberry Pi 5, the system ensures raw biometric data (audio/video) never leaves the user's local network, sending only transcribed text and emotion labels to the cloud LLM.

---

## 3. Core Features & System Workflows

### 1. Live Support Session (Telehealth UI)
* **Multimodal Fusion:** Captures microphone and webcam input to generate a fused "Dominant Emotion" (Speech + Face).
* **Zero-Latency Streaming:** Responses stream word-by-word onto the UI, synchronized with background-generated Edge Text-to-Speech (TTS).
* **Immersive EHR Design:** Deep-slate UI with dynamic hardware telemetry, affective logs, and active clinical protocol visualization.

### 2. Measurement-Based Care (MBC) Hub
* **Longitudinal Tracking:** Interactive charting of PHQ-9 (Depression), GAD-7 (Anxiety), and PCL-5 (Trauma) scores over time.
* **Dynamic Care Plans:** Automatically prescribes daily routines (e.g., Morning Light Exposure) and micro-interventions (e.g., Progressive Muscle Relaxation) based on current symptom velocity.

### 3. Admin Observatory & SBAR Exports
* **Clinical Analytics:** Aggregates user data to calculate Engagement Scores, Emotion Volatility, and Negative Affect Ratios.
* **Lossless PDF Handoffs:** Generates downloadable clinical summary reports, utilizing text-wrapping algorithms to ensure complete preservation of patient transcripts.

### 4. Safety Protocol & Escalation
* **Unrestricted Support:** The system tracks high-acuity distress (C-SSRS logic) without locking the user out of the chat, preserving patient autonomy.
* **Manual Resolution:** A dedicated UI workflow for patients to acknowledge stabilization and manually clear backend safety flags.

---

## 4. System Architecture & Technical Innovations

To achieve real-time performance on a Raspberry Pi 5, SERENITY employs several advanced engineering techniques:

### 4.1 Backend (FastAPI & Python 3.11)
* **Unbound Event Loops:** Heavy database queries (SQLite WAL mode) are offloaded to background threads using `run_in_threadpool`, preventing the async LLM stream from freezing.
* **In-Memory TTS (Zero Disk I/O):** Audio is generated and piped directly into RAM (`bytearray`) and encoded to Base64, completely bypassing slow SD-card read/write bottlenecks.
* **Sliding-Window Stream Parsing:** Replaced heavy Regex payload extraction with an O(1) sliding-window delimiter search, drastically reducing CPU overhead during token streaming.
* **Connection Pooling:** Utilizes aggressive `httpx.AsyncClient` keep-alive pooling to eliminate TLS handshake latency on sequential chat turns.

### 4.2 Edge AI Perception (TFLite + Whisper)
* **Polyphase Resampling:** Replaced standard Fourier Transform (FFT) audio resampling with `scipy.signal.resample_poly`, reducing audio preprocessing CPU time by up to 80%.
* **Vectorized Tensors:** MFCC (Mel-Frequency Cepstral Coefficients) extraction uses pure NumPy vectorization to dynamically reshape 1D/2D/3D/4D tensors for the TensorFlow Lite models.
* **XNNPACK Acceleration:** Forces ARM-optimized CPU delegates for real-time neural network inference.

### 4.3 Frontend (React + Vite)
* **Global Temporal Sync:** All timestamps are explicitly locked to Pakistan Standard Time (`Asia/Karachi`), ensuring chronological integrity regardless of the client device's local clock.
* **Glassmorphic Tailwind Design:** High-performance, low-DOM-node CSS rendering for smooth animations on constrained graphical hardware.

---

## 5. Complete Directory Structure

```text
SERENITY/
│
├── backend/
│   ├── main.py                   # FastAPI application, streaming endpoints, FSM orchestration
│   ├── audio_core.py             # Whisper STT & TFLite Speech Emotion (Polyphase optimized)
│   ├── emotion_core.py           # Facial Emotion Recognition (FER) inference
│   ├── cloud_llm_core.py         # Async SSE client, connection pooling, sliding-window parser
│   ├── clinical_router.py        # Framework selection (CBT/DBT/ACT) & risk scoring
│   ├── clinical_core.py          # SBAR generation, symptom trajectory math, PDF rendering
│   ├── database.py               # SQLite engine, schema migrations, async threadpool queries
│   ├── models.py                 # SQLAlchemy ORM definitions
│   └── questionnaires_data.py    # PHQ-9, GAD-7, PCL-5 templates and scoring logic
│
├── frontend/
│   ├── index.html                # Vite entry point
│   ├── vite.config.js            # Build configuration & port mapping
│   ├── src/
│   │   ├── App.jsx               # React Router & Clinical Context Provider
│   │   ├── main.jsx              # StrictMode root render
│   │   ├── index.css             # Tailwind imports & custom scrollbar definitions
│   │   ├── context/
│   │   │   └── ClinicalContext.jsx # Global state for therapy modes & crisis flags
│   │   ├── components/
│   │   │   ├── Dashboard.jsx     # Main module selector (Clinical AI Interface)
│   │   │   └── Login.jsx         # Authentication & Bcrypt fallback UI
│   │   └── pages/
│   │       ├── UnifiedEmotionPage.jsx      # Live Video/Voice/Chat Support Session
│   │       ├── MBCHubPage.jsx              # Measurement-Based Care charts & Care Plan
│   │       ├── AdminPage.jsx               # Clinical Observatory & SBAR exports
│   │       ├── QuestionnairesPage.jsx      # Standardized psychometric screening forms
│   │       ├── HardwareDiagnosticsPage.jsx # Edge telemetry (CPU, RAM, Inference Latency)
│   │       └── SafetyPlanPage.jsx          # C-SSRS Triage & Grounding Protocols
│
├── .env                          # Environment variables (Cloud URLs, TTS keys)
├── requirements-edge.txt         # Pi-optimized Python dependencies
└── README.md                     # Master project documentary
This is a comprehensive, academic, and professionally structured README.md designed specifically to serve as the master documentary for your Final Year Project (FYP).It highlights the advanced architectural decisions, performance optimizations (like Polyphase Resampling and Zero-Disk-I/O TTS), and the clinical frameworks you have successfully integrated.Markdown# SERENITY: Smart Emotion Recognition and Neural Intervention Technology

**An Edge-Optimized, Multimodal AI Platform for Measurement-Based Mental Health Support**

SERENITY is an applied Artificial Intelligence Electronic Health Record (EHR) and telehealth platform. It combines real-time speech and facial emotion recognition, clinical psychometric screening, and cloud-assisted Large Language Model (LLM) therapeutic dialogue to provide data-informed, continuous mental health support.

This repository serves as the complete technical documentary, deployment guide, and architectural overview for the SERENITY Final Year Project (FYP), specifically optimized for edge deployment on the **Raspberry Pi 5**.

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Background & Motivation](#2-background--motivation)
3. [Core Features & System Workflows](#3-core-features--system-workflows)
4. [System Architecture & Technical Innovations](#4-system-architecture--technical-innovations)
5. [Complete Directory Structure](#5-complete-directory-structure)
6. [Raspberry Pi 5 Deployment Guide](#6-raspberry-pi-5-deployment-guide)
7. [Environment Configuration](#7-environment-configuration)
8. [Troubleshooting & Edge Limitations](#8-troubleshooting--edge-limitations)
9. [Academic & Clinical Disclaimer](#9-academic--clinical-disclaimer)

---

## 1. Executive Summary

Mental health support systems typically fail by either being user-friendly but clinically shallow, or clinically rigid but highly inaccessible. SERENITY bridges this gap through a **Tri-Pillar Architecture**:
* **Passive Perception:** Continuous, edge-computed monitoring of vocal and facial affect.
* **Active Screening:** Measurement-Based Care (MBC) tracking longitudinal symptom trajectories via standardized psychometrics (PHQ-9, GAD-7, PCL-5).
* **Dynamic Intervention:** A clinical routing engine that dynamically shifts the LLM's therapeutic framework (CBT, DBT, ACT) based on real-time risk scores and detected cognitive distortions.

SERENITY provides users with empathetic, unrestricted support while automatically synthesizing structured **SBAR (Situation, Background, Assessment, Recommendation)** handoff reports for human psychologists.

---

## 2. Background & Motivation

The global shortage of mental health professionals has driven a surge in AI chatbots. However, standard LLMs lack clinical memory, multimodal awareness, and safety guardrails. They cannot detect when a patient's tone of voice contradicts their text, nor can they track symptom degradation over months.

**The SERENITY Objective:**
To engineer a low-latency, edge-deployable system that acts as a "clinical co-pilot." It does not diagnose; rather, it observes, stabilizes, and structures patient data. By running heavy perception models (STT, SER, FER) locally on a Raspberry Pi 5, the system ensures raw biometric data (audio/video) never leaves the user's local network, sending only transcribed text and emotion labels to the cloud LLM.

---

## 3. Core Features & System Workflows

### 1. Live Support Session (Telehealth UI)
* **Multimodal Fusion:** Captures microphone and webcam input to generate a fused "Dominant Emotion" (Speech + Face).
* **Zero-Latency Streaming:** Responses stream word-by-word onto the UI, synchronized with background-generated Edge Text-to-Speech (TTS).
* **Immersive EHR Design:** Deep-slate UI with dynamic hardware telemetry, affective logs, and active clinical protocol visualization.

### 2. Measurement-Based Care (MBC) Hub
* **Longitudinal Tracking:** Interactive charting of PHQ-9 (Depression), GAD-7 (Anxiety), and PCL-5 (Trauma) scores over time.
* **Dynamic Care Plans:** Automatically prescribes daily routines (e.g., Morning Light Exposure) and micro-interventions (e.g., Progressive Muscle Relaxation) based on current symptom velocity.

### 3. Admin Observatory & SBAR Exports
* **Clinical Analytics:** Aggregates user data to calculate Engagement Scores, Emotion Volatility, and Negative Affect Ratios.
* **Lossless PDF Handoffs:** Generates downloadable clinical summary reports, utilizing text-wrapping algorithms to ensure complete preservation of patient transcripts.

### 4. Safety Protocol & Escalation
* **Unrestricted Support:** The system tracks high-acuity distress (C-SSRS logic) without locking the user out of the chat, preserving patient autonomy.
* **Manual Resolution:** A dedicated UI workflow for patients to acknowledge stabilization and manually clear backend safety flags.

---

## 4. System Architecture & Technical Innovations

To achieve real-time performance on a Raspberry Pi 5, SERENITY employs several advanced engineering techniques:

### 4.1 Backend (FastAPI & Python 3.11)
* **Unbound Event Loops:** Heavy database queries (SQLite WAL mode) are offloaded to background threads using `run_in_threadpool`, preventing the async LLM stream from freezing.
* **In-Memory TTS (Zero Disk I/O):** Audio is generated and piped directly into RAM (`bytearray`) and encoded to Base64, completely bypassing slow SD-card read/write bottlenecks.
* **Sliding-Window Stream Parsing:** Replaced heavy Regex payload extraction with an O(1) sliding-window delimiter search, drastically reducing CPU overhead during token streaming.
* **Connection Pooling:** Utilizes aggressive `httpx.AsyncClient` keep-alive pooling to eliminate TLS handshake latency on sequential chat turns.

### 4.2 Edge AI Perception (TFLite + Whisper)
* **Polyphase Resampling:** Replaced standard Fourier Transform (FFT) audio resampling with `scipy.signal.resample_poly`, reducing audio preprocessing CPU time by up to 80%.
* **Vectorized Tensors:** MFCC (Mel-Frequency Cepstral Coefficients) extraction uses pure NumPy vectorization to dynamically reshape 1D/2D/3D/4D tensors for the TensorFlow Lite models.
* **XNNPACK Acceleration:** Forces ARM-optimized CPU delegates for real-time neural network inference.

### 4.3 Frontend (React + Vite)
* **Global Temporal Sync:** All timestamps are explicitly locked to Pakistan Standard Time (`Asia/Karachi`), ensuring chronological integrity regardless of the client device's local clock.
* **Glassmorphic Tailwind Design:** High-performance, low-DOM-node CSS rendering for smooth animations on constrained graphical hardware.

---

## 5. Complete Directory Structure

```text
SERENITY/
│
├── backend/
│   ├── main.py                   # FastAPI application, streaming endpoints, FSM orchestration
│   ├── audio_core.py             # Whisper STT & TFLite Speech Emotion (Polyphase optimized)
│   ├── emotion_core.py           # Facial Emotion Recognition (FER) inference
│   ├── cloud_llm_core.py         # Async SSE client, connection pooling, sliding-window parser
│   ├── clinical_router.py        # Framework selection (CBT/DBT/ACT) & risk scoring
│   ├── clinical_core.py          # SBAR generation, symptom trajectory math, PDF rendering
│   ├── database.py               # SQLite engine, schema migrations, async threadpool queries
│   ├── models.py                 # SQLAlchemy ORM definitions
│   └── questionnaires_data.py    # PHQ-9, GAD-7, PCL-5 templates and scoring logic
│
├── frontend/
│   ├── index.html                # Vite entry point
│   ├── vite.config.js            # Build configuration & port mapping
│   ├── src/
│   │   ├── App.jsx               # React Router & Clinical Context Provider
│   │   ├── main.jsx              # StrictMode root render
│   │   ├── index.css             # Tailwind imports & custom scrollbar definitions
│   │   ├── context/
│   │   │   └── ClinicalContext.jsx # Global state for therapy modes & crisis flags
│   │   ├── components/
│   │   │   ├── Dashboard.jsx     # Main module selector (Clinical AI Interface)
│   │   │   └── Login.jsx         # Authentication & Bcrypt fallback UI
│   │   └── pages/
│   │       ├── UnifiedEmotionPage.jsx      # Live Video/Voice/Chat Support Session
│   │       ├── MBCHubPage.jsx              # Measurement-Based Care charts & Care Plan
│   │       ├── AdminPage.jsx               # Clinical Observatory & SBAR exports
│   │       ├── QuestionnairesPage.jsx      # Standardized psychometric screening forms
│   │       ├── HardwareDiagnosticsPage.jsx # Edge telemetry (CPU, RAM, Inference Latency)
│   │       └── SafetyPlanPage.jsx          # C-SSRS Triage & Grounding Protocols
│
├── .env                          # Environment variables (Cloud URLs, TTS keys)
├── requirements-edge.txt         # Pi-optimized Python dependencies
└── README.md                     # Master project documentary
6. Raspberry Pi 5 Deployment GuideThis section details deployment from scratch on a clean Raspberry Pi OS.6.1 Hardware & OS PrerequisitesDevice: Raspberry Pi 5 (8GB RAM strongly recommended).OS: 64-bit Raspberry Pi OS Bookworm.Peripherals: USB Microphone, USB Webcam (Optional).6.2 System & Tooling SetupBashsudo apt update && sudo apt full-upgrade -y
sudo reboot

# Install required compilers and audio libraries
sudo apt install -y git curl build-essential pkg-config ffmpeg libsndfile1 libatlas-base-dev libopenblas-dev libglib2.0-0 libgl1

# Install Python 3.11 (Required for TFLite compatibility)
sudo apt install -y python3.11 python3.11-venv python3-pip
6.3 Clone & Model RetrievalSERENITY uses Git Large File Storage (LFS) for .tflite weights. You must install LFS before cloning.Bashgit lfs install
git clone [https://github.com/mtahaarif/Smart-Emotion-Recognition-and-Neural-Intervention-Technology-SERENITY-.git](https://github.com/mtahaarif/Smart-Emotion-Recognition-and-Neural-Intervention-Technology-SERENITY-.git)
cd Smart-Emotion-Recognition-and-Neural-Intervention-Technology-SERENITY-
git lfs pull
6.4 Python Virtual Environment (Backend)Bashpython3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Install Edge-Optimized Wheels
pip install --extra-index-url [https://www.piwheels.org/simple](https://www.piwheels.org/simple) -r requirements-edge.txt
(Note: If tflite-runtime fails to build, downgrade to tensorflow==2.18.0 as the backend has automatic fallbacks).6.5 Node.js Setup (Frontend)Bashsudo apt install -y nodejs npm
cd frontend
npm install
6.6 Running the SystemTerminal 1 (Backend):Bashsource .venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 5000
Terminal 2 (Frontend):Bashcd frontend
npm run dev -- --host 0.0.0.0 --port 5173
Access the application from any device on your local network at http://<PI_IP_ADDRESS>:5173.7. Environment ConfigurationCreate a .env file in the root directory.VariableDescriptionDefaultSERENITY_CLOUD_LLM_URLCloud LLM endpoint (EC2, Vertex, etc.)RequiredSERENITY_CLOUD_LLM_FALLBACK_URLSComma-separated backup endpoints""SERENITY_TTS_ENABLEDEnable Edge-TTS generationtrueSERENITY_TTS_VOICEPrimary Azure TTS voiceen-US-AriaNeuralSERENITY_WHISPER_MODEL_SIZELocal STT model sizetinySERENITY_SER_AUDIO_SAMPLE_RATEPolyphase target Hz16000SERENITY_TFLITE_XNNPACK_DELEGATEPath to ARM CPU acceleratorlibtensorflowlite_xnnpack_delegate.so8. Troubleshooting & Edge Limitations8.1 "No matching distribution found for tflite-runtime"Cause: Python 3.12/3.13 does not have pre-compiled wheels for TFLite on ARM64.Fix: Ensure you created the .venv using exactly python3.11.8.2 Blank Screen on Chat SendCause: Missing Lucide React icons (Loader2, AlertTriangle) crashing the render tree.Fix: Handled in v2.1. Ensure npm install is up to date and Vite cache is cleared (npm run dev -- --force).8.3 TTS 403 Forbidden ErrorCause: Microsoft Edge TTS endpoint throttling or clock de-sync.Fix: 1. Ensure the Raspberry Pi's clock is synced: sudo timedatectl set-ntp true2. If network-blocked, set SERENITY_TTS_ENABLED=false to rely on the browser's native fallback SpeechSynthesis.8.4 High Latency (Slow Responses)Fix:Ensure the Pi has adequate power (5V 5A official supply).Attach an active cooler. Thermal throttling will immediately drop STT inference speed.Ensure .env contains SERENITY_TTS_STREAM_MODE="sentence" to interleave audio generation with LLM token streaming.9. Academic & Clinical DisclaimerFinal Year Project Notice: This software was developed as an academic engineering prototype to demonstrate the fusion of edge-AI and cloud-LLMs.Not a Medical Device: SERENITY is not an FDA/HIPAA-approved medical device. It does not provide medical diagnoses, and its risk-scoring heuristics (PHQ-9/GAD-7/PCL-5 calculations) are for demonstrative observability only. Any use of this system in a real-world scenario must be accompanied by mandatory human-in-the-loop escalation workflows and compliance with local data privacy laws.
