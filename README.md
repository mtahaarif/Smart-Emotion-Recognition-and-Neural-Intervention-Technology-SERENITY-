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
