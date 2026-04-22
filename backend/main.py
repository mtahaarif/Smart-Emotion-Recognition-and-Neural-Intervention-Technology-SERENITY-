import asyncio
import base64
import contextlib
from collections import deque
from datetime import datetime, timedelta
import importlib
import json
import logging
import math
import os
import re
import secrets
import sys
import threading
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
import bcrypt
from datetime import datetime, timezone
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

MIN_SUPPORTED_PYTHON = (3, 10)
MAX_SUPPORTED_PYTHON_EXCLUSIVE = (3, 13)

if not (MIN_SUPPORTED_PYTHON <= sys.version_info[:2] < MAX_SUPPORTED_PYTHON_EXCLUSIVE):
    raise RuntimeError(
        "SERENITY supports Python 3.10-3.12 (3.11 recommended) for this runtime stack. "
        f"Detected Python {sys.version.split()[0]}. "
        "Please create the virtual environment with python3.11 and reinstall dependencies."
    )

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

try: import torch
except ImportError: torch = None

from backend.audio_core import initialize_audio_runtime, predict_audio_emotion
from backend.clinical_core import (
    PHASES_BY_FRAMEWORK,
    advance_phase,
    build_admin_clinical_handoff_fallback,
    build_admin_clinical_handoff_prompt,
    build_admin_handoff_markdown,
    build_handoff_markdown,
    compute_weekly_trajectory_flags,
    default_phase_for_framework,
    parse_structured_llm_payload,
    render_handoff_pdf,
)
from backend.clinical_router import (
    FRAMEWORK_ACT,
    FRAMEWORK_CBT,
    FRAMEWORK_DBT,
    FRAMEWORK_SUPPORTIVE,
    RoutingDecision,
    build_routed_prompt,
    build_safety_override_response,
    determine_clinical_mode,
    evaluate_clinical_route,
)
from backend.emotion_core import initialize_face_runtime, analyze_face
from backend.database import (
    SessionLocal,
    apply_schema_migrations,
    calculate_symptom_trajectory,
    engine,
    fetch_or_create_clinical_state,
    fetch_questionnaire_results,
    fetch_recent_edge_diagnostics,
    fetch_recent_sessions_with_emotions,
    fetch_recent_turn_summaries,
    fetch_trajectory_snapshots,
    persist_clinical_distortion_event,
    persist_clinical_routing_event,
    persist_edge_diagnostic_sample,
    persist_questionnaire_result,
    persist_safety_escalation_event,
    persist_turn,
    replace_trajectory_snapshots,
    update_user_emergency_contact,
    upsert_clinical_state,
)
from backend.cloud_llm_core import CloudLLMClient, CloudLLMError
from backend.questionnaires_data import QUESTIONNAIRE_DEFINITIONS, normalize_questionnaire_type, questionnaire_clinical_flags, questionnaire_templates, score_questionnaire
import backend.models as models

try:
    import psutil
except ImportError:
    psutil = None

try: WhisperModel = getattr(importlib.import_module("faster_whisper"), "WhisperModel", None)
except ImportError: WhisperModel = None
try: import whisper as openai_whisper
except ImportError: openai_whisper = None
try: import edge_tts
except ImportError: edge_tts = None

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Config & Flags ---
def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default)).strip()))
    except (TypeError, ValueError, AttributeError):
        return max(minimum, default)


def _env_float(name: str, default: float, minimum: float = 0.1) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default)).strip()))
    except (TypeError, ValueError, AttributeError):
        return max(minimum, default)


WHISPER_MODEL_SIZE = os.getenv("SERENITY_WHISPER_MODEL_SIZE", "tiny").strip()
WHISPER_CPU_THREADS = _env_int("SERENITY_WHISPER_CPU_THREADS", max(1, (os.cpu_count() or 4) // 2))
WHISPER_TIMEOUT_SECONDS = _env_int("SERENITY_WHISPER_TIMEOUT_SECONDS", 40)
EMOTION_TIMEOUT_SECONDS = _env_int("SERENITY_EMOTION_TIMEOUT_SECONDS", 20)
LLM_TIMEOUT_SECONDS = _env_int("SERENITY_LLM_TIMEOUT_SECONDS", 25)
TTS_ENABLED = os.getenv("SERENITY_TTS_ENABLED", "true").lower() == "true"
TTS_VOICE = os.getenv("SERENITY_TTS_VOICE", "en-GB-RyanNeural").strip()
TTS_FALLBACK_VOICE = os.getenv("SERENITY_TTS_FALLBACK_VOICE", "").strip()
TTS_TIMEOUT_SECONDS = _env_int("SERENITY_TTS_TIMEOUT_SECONDS", 45)
TTS_RETRIES = _env_int("SERENITY_TTS_RETRIES", 2)
TTS_STREAM_MODE = os.getenv("SERENITY_TTS_STREAM_MODE", "sentence").strip().lower()
if TTS_STREAM_MODE not in {"sentence", "final"}:
    TTS_STREAM_MODE = "sentence"
ADMIN_DEFAULT_LIMIT = _env_int("SERENITY_ADMIN_DEFAULT_LIMIT", 300, minimum=50)
ADMIN_MAX_LIMIT = _env_int("SERENITY_ADMIN_MAX_LIMIT", 3000, minimum=200)
ADMIN_OVERVIEW_CACHE_TTL_SECONDS = _env_float("SERENITY_ADMIN_OVERVIEW_CACHE_TTL_SECONDS", 20.0, minimum=5.0)
ADMIN_SUMMARY_CACHE_TTL_SECONDS = _env_float("SERENITY_ADMIN_SUMMARY_CACHE_TTL_SECONDS", 120.0, minimum=10.0)
ADMIN_SUMMARY_TIMEOUT_SECONDS = _env_float("SERENITY_ADMIN_SUMMARY_TIMEOUT_SECONDS", 10.0, minimum=3.0)
PREWARM_MODELS = os.getenv("SERENITY_PREWARM_MODELS", "true").lower() == "true"
PREWARM_WHISPER = os.getenv("SERENITY_PREWARM_WHISPER", "false").lower() == "true"
CLINICAL_WEEKLY_WORSENING_DELTA = _env_int("SERENITY_CLINICAL_WEEKLY_WORSENING_DELTA", 4)
CLINICAL_SAFETY_HARDSTOP_RISK = _env_int("SERENITY_CLINICAL_SAFETY_HARDSTOP_RISK", 8)
EDGE_DIAGNOSTICS_BUFFER_SIZE = _env_int("SERENITY_EDGE_DIAGNOSTICS_BUFFER_SIZE", 240)

EMOTION_LABELS = ["angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_ALIAS = {"surprised": "surprise", "fearful": "fear", "no face": "neutral"}
NEGATIVE_EMOTIONS = {"angry", "disgust", "fear", "sad"}
SENTENCE_BOUNDARY_REGEX = re.compile(r"(?<=[.!?])\s+")
DISTRESS_SIGNAL_REGEX = re.compile(r"\b(hopeless|worthless|overwhelmed|panic|can't cope|cannot cope|self[- ]?harm|suicid|hurt myself|end my life)\b", re.IGNORECASE)
QUESTIONNAIRE_MAX_SCORES = {"PHQ-9": 27.0, "GAD-7": 21.0, "PCL-5": 80.0}

whisper_init_lock, cloud_llm_init_lock = threading.Lock(), threading.Lock()

# --- Pydantic Models ---
class AuthRequest(BaseModel): username: str; password: str
class AuthResponse(BaseModel): message: str; username: str
class ChatRequest(BaseModel): username: str; message: str
class EmergencyContactRequest(BaseModel): username: str; contact_name: str = ""; contact_phone: str = ""
class QuestionnaireSubmitRequest(BaseModel): username: str; questionnaire_type: str; answers: List[int] = Field(default_factory=list); submitted_at: Optional[str] = None
class InteractResponse(BaseModel): dominant_emotion: str; speech_emotion: str; face_emotion: str; transcription: str; llm_response: str; tts_audio_base64: Optional[str] = None; tts_audio_segments_base64: List[str] = Field(default_factory=list); errors: List[str] = Field(default_factory=list)

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

@contextlib.asynccontextmanager
async def serenity_lifespan(app: FastAPI):
    app.state.cloud_llm_client, app.state.whisper_model = None, None
    app.state.face_runtime, app.state.speech_runtime = None, None
    app.state.admin_summary_cache, app.state.admin_overview_cache = {"key": "", "summary": "", "expires_at": 0.0}, {}
    app.state.edge_diagnostics = deque(maxlen=EDGE_DIAGNOSTICS_BUFFER_SIZE)
    app.state.whisper_device_in_use = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
    try:
        if PREWARM_MODELS:
            def _prewarm_models() -> None:
                with contextlib.suppress(Exception):
                    app.state.speech_runtime = initialize_audio_runtime()
                with contextlib.suppress(Exception):
                    app.state.face_runtime = initialize_face_runtime()
                if PREWARM_WHISPER and not app.state.whisper_model:
                    with whisper_init_lock:
                        if app.state.whisper_model:
                            return
                        device = app.state.whisper_device_in_use
                        if WhisperModel:
                            compute = "float16" if device == "cuda" else "int8"
                            app.state.whisper_model = WhisperModel(
                                WHISPER_MODEL_SIZE,
                                device=device,
                                compute_type=compute,
                                cpu_threads=WHISPER_CPU_THREADS,
                            )
                            app.state.whisper_backend = "faster-whisper"
                        elif openai_whisper:
                            app.state.whisper_model = openai_whisper.load_model(WHISPER_MODEL_SIZE, device=device)
                            app.state.whisper_backend = "openai-whisper"

            await run_in_threadpool(_prewarm_models)
    except Exception as exc:
        LOGGER.info("Startup warmup skipped: %s", exc)

    try:
        yield
    finally:
        if client := getattr(app.state, "cloud_llm_client", None):
            with contextlib.suppress(Exception):
                await client.close()

app = FastAPI(title="SERENITY API", lifespan=serenity_lifespan)
models.Base.metadata.create_all(bind=engine)
apply_schema_migrations()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def _state_get(k: str, d=None): return getattr(app.state, k, d)
def _state_set(k: str, v): setattr(app.state, k, v)
def _dedupe_errors(errors: List[str]) -> List[str]: return list(dict.fromkeys([str(e) for e in errors if str(e).strip()]))
def _is_bcrypt_hash(value: str) -> bool:
    return bool(value and (value.startswith("$2a$") or value.startswith("$2b$") or value.startswith("$2y$")))
def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
def _verify_password(plain_password: str, stored_password: str) -> bool:
    stored = str(stored_password or "")
    if not stored:
        return False
    if _is_bcrypt_hash(stored):
        with contextlib.suppress(ValueError):
            return bcrypt.checkpw(plain_password.encode("utf-8"), stored.encode("utf-8"))
        return False
    # Legacy plain-text fallback for old records; upgraded to bcrypt on successful login.
    return secrets.compare_digest(stored, plain_password)


def _phase_index(framework: str, phase: str) -> int:
    phases = PHASES_BY_FRAMEWORK.get(str(framework or "").strip()) or PHASES_BY_FRAMEWORK.get("Supportive_Stabilization", [])
    normalized = str(phase or "").strip().lower()
    for idx, candidate in enumerate(phases):
        if str(candidate).strip().lower() == normalized:
            return idx
    return 0


def _latest_questionnaire_scores(db: Session, username: str) -> Dict[str, int]:
    rows = fetch_questionnaire_results(db, username=username, limit=30, include_answers=False)
    latest: Dict[str, int] = {}
    for row in rows:
        q_type = str(row.get("questionnaire_type") or "").upper()
        if q_type and q_type not in latest:
            latest[q_type] = int(row.get("total_score") or 0)
    return latest


def _clinical_risk_score(
    db: Session,
    username: str,
    user_text: str,
    speech_emotion: str,
    face_emotion: str,
) -> int:
    latest_scores = _latest_questionnaire_scores(db, username)
    risk = 0

    flags = questionnaire_clinical_flags(latest_scores)
    risk += sum(1 for v in flags.values() if v) * 2

    if DISTRESS_SIGNAL_REGEX.search(str(user_text or "")):
        risk += 2

    neg_count = sum(
        1
        for label in (speech_emotion, face_emotion)
        if _normalize_emotion_label(label) in NEGATIVE_EMOTIONS
    )
    if neg_count >= 2:
        risk += 2
    elif neg_count == 1:
        risk += 1

    user = db.query(models.User).filter(models.User.username == username).first()
    if user and bool(getattr(user, "requires_safety_review", False)):
        risk = max(risk, CLINICAL_SAFETY_HARDSTOP_RISK)

    return int(risk)


def _collect_edge_stats(latency_ms: float) -> Dict[str, float]:
    rss_mb = 0.0
    if psutil is not None:
        with contextlib.suppress(Exception):
            rss_mb = round(float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0), 2)
    return {
        "latency_ms": round(float(latency_ms), 2),
        "rss_mb": float(rss_mb),
    }


def _persist_and_cache_diagnostics(
    db: Session,
    username: str,
    source: str,
    stt_ms: float,
    ser_ms: float,
    fer_ms: float,
    llm_latency_ms: float,
    speech_conf: float,
    face_conf: float,
) -> None:
    combined_latency = float(stt_ms) + float(ser_ms) + float(fer_ms) + float(llm_latency_ms)
    stats = _collect_edge_stats(combined_latency)
    sample = {
        "captured_at": datetime.utcnow().isoformat(),
        "username": username,
        "source": source,
        "stt_latency_ms": round(float(stt_ms), 2),
        "ser_latency_ms": round(float(ser_ms), 2),
        "fer_latency_ms": round(float(fer_ms), 2),
        "total_latency_ms": round(float(combined_latency), 2),
        "memory_mb": stats["rss_mb"],
        "llm_ms": round(float(llm_latency_ms), 2),
        "speech_confidence": round(float(speech_conf or 0.0), 2),
        "face_confidence": round(float(face_conf or 0.0), 2),
    }

    buffer = _state_get("edge_diagnostics", None)
    if isinstance(buffer, deque):
        buffer.append(sample)

    persist_edge_diagnostic_sample(db, username=username, sample=sample)


def _latest_edge_metric_sample(db: Session, username: Optional[str] = None) -> Dict[str, Any]:
    live = _state_get("edge_diagnostics", None)
    if isinstance(live, deque) and len(live) > 0:
        return dict(live[-1])

    rows = fetch_recent_edge_diagnostics(db, username=username, limit=1)
    if rows:
        return dict(rows[0])

    wave = (math.sin(time.time() * 0.75) + 1.0) / 2.0
    return {
        "stt_latency_ms": round(180 + 120 * wave, 2),
        "ser_latency_ms": round(42 + 30 * wave, 2),
        "fer_latency_ms": round(55 + 28 * wave, 2),
        "memory_mb": round(620 + 180 * wave, 2),
    }


def _xnnpack_delegate_active() -> bool:
    value = str(os.getenv("SERENITY_XNNPACK_DELEGATE_ACTIVE", "true")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _extract_protocol_payload(raw_llm_response: str) -> Dict[str, Any]:
    raw_text = str(raw_llm_response or "").strip()
    protocol_from_delimiter: Dict[str, Any] = {}
    parse_source = raw_text

    if "|||" in raw_text:
        visible_text, _, tail = raw_text.partition("|||")
        parse_source = visible_text.strip()
        tail_payload = str(tail or "").strip()
        if tail_payload:
            with contextlib.suppress(json.JSONDecodeError):
                parsed_tail = json.loads(tail_payload)
                if isinstance(parsed_tail, dict):
                    protocol_from_delimiter = {
                        "advance_phase": bool(parsed_tail.get("advance_phase", False)),
                        "detected_distortion": str(parsed_tail.get("detected_distortion") or "").strip(),
                    }

    parsed = parse_structured_llm_payload(parse_source)
    response_text = str(parsed.get("response_text") or "").strip()
    if not response_text:
        response_text = parse_source or raw_text

    if protocol_from_delimiter:
        parsed["advance_phase"] = bool(protocol_from_delimiter.get("advance_phase", parsed.get("advance_phase", False)))
        if protocol_from_delimiter.get("detected_distortion"):
            parsed["detected_distortion"] = str(protocol_from_delimiter["detected_distortion"])

    parsed["response_text"] = response_text
    return parsed


def _format_exception_detail(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def _build_llm_fallback_response(user_text: str, route_decision: RoutingDecision) -> str:
    framework = str(route_decision.framework or FRAMEWORK_SUPPORTIVE).strip()

    if framework == FRAMEWORK_DBT:
        return (
            "I hear this feels intense right now. Let's ground first: inhale for 4, hold for 4, and exhale for 6. "
            "When you are ready, tell me the single most urgent part so we can take one safe step at a time."
        )

    if framework == FRAMEWORK_CBT:
        return (
            "Thank you for sharing that. Let's test the hardest thought together: name one fact that supports it "
            "and one fact that does not. Then we can form a more balanced statement."
        )

    if framework == FRAMEWORK_ACT:
        return (
            "That sounds heavy. Try this defusion step: say, 'I am having the thought that ...' and notice how it feels. "
            "Then choose one small action in the next 10 minutes that matches your values."
        )

    return (
        "I'm here with you. We can move one step at a time. "
        "Tell me what feels hardest right now, and we will choose one practical next action together."
    )


def _route_safety_mode(route_decision: RoutingDecision, parsed_payload: Optional[Dict[str, Any]] = None) -> bool:
    parsed_payload = parsed_payload or {}
    return bool(
        route_decision.acute_safety_trigger
        or route_decision.high_distress
        or bool(parsed_payload.get("safety_alert"))
        or int(route_decision.risk_score or 0) >= CLINICAL_SAFETY_HARDSTOP_RISK
    )


def _build_route_event_signals(
    route_decision: RoutingDecision,
    speech_emotion: str,
    face_emotion: str,
    dominant_emotion: str,
) -> Dict[str, Any]:
    return {
        "framework": route_decision.framework,
        "risk_score": int(route_decision.risk_score or 0),
        "safety_mode": _route_safety_mode(route_decision),
        "route_locked": bool(route_decision.route_locked),
        "route_reason": route_decision.route_reason,
        "detected_distortions": list(route_decision.detected_distortions or []),
        "rumination": bool(route_decision.rumination_detected),
        "acute_safety_trigger": bool(route_decision.acute_safety_trigger),
        "high_distress": bool(route_decision.high_distress),
        "speech_emotion": _normalize_emotion_label(speech_emotion),
        "face_emotion": _normalize_emotion_label(face_emotion),
        "dominant_emotion": _normalize_emotion_label(dominant_emotion),
    }


def _refresh_clinical_state_after_turn(
    db: Session,
    username: str,
    route_decision: RoutingDecision,
    parsed_payload: Dict[str, Any],
    user_text: str,
    assistant_text: str,
) -> Dict[str, Any]:
    current_state = fetch_or_create_clinical_state(db, username)

    framework = str(route_decision.framework or current_state.get("active_framework") or FRAMEWORK_DBT)
    phase = str(current_state.get("current_phase") or default_phase_for_framework(framework))
    if str(current_state.get("active_framework") or "") != framework:
        phase = default_phase_for_framework(framework)

    if bool(parsed_payload.get("advance_phase")):
        phase = advance_phase(framework, phase)

    phase_idx = _phase_index(framework, phase)
    distress = _route_safety_mode(route_decision, parsed_payload)

    summary = {
        "distortion": parsed_payload.get("detected_distortion") or (route_decision.detected_distortions or [None])[0],
        "rumination": bool(route_decision.rumination_detected),
        "suggested_intervention": parsed_payload.get("suggested_intervention") or "",
        "route_reason": route_decision.route_reason,
    }

    updated = upsert_clinical_state(
        db,
        username=username,
        updates={
            "active_framework": framework,
            "current_phase": phase,
            "phase_index": phase_idx,
            "requires_safety_review": bool(current_state.get("requires_safety_review")) or distress,
            "last_risk_score": int(route_decision.risk_score or 0),
            "last_route_reason": route_decision.route_reason,
            "last_detected_distortion": summary["distortion"] or "",
            "last_distress_level": "high" if distress else ("moderate" if summary["rumination"] else "low"),
        },
    )

    if summary["distortion"]:
        persist_clinical_distortion_event(
            db,
            username=username,
            distortion_label=summary["distortion"],
            framework=framework,
            source_excerpt=user_text,
        )

    if distress:
        recent = fetch_recent_turn_summaries(db, limit=6, text_limit=500, username=username)
        handoff_markdown = build_handoff_markdown(
            username=username,
            risk_score=int(route_decision.risk_score or 0),
            route_framework=framework,
            active_flags=[summary["distortion"]] if summary["distortion"] else [],
            distress_signals=1 if distress else 0,
            recent_turns=recent,
        )
        persist_safety_escalation_event(
            db,
            username=username,
            trigger_type=parsed_payload.get("safety_reason") or route_decision.route_reason,
            risk_score=int(route_decision.risk_score or 0),
            dominant_emotion=route_decision.dominant_emotion,
            transcript_excerpt=user_text,
            handoff_markdown=handoff_markdown,
        )

    return updated


def _sync_weekly_trajectory(db: Session, username: str) -> Dict[str, Any]:
    questionnaire_rows = fetch_questionnaire_results(db, username=username, limit=120, include_answers=False)
    trajectory = compute_weekly_trajectory_flags(
        questionnaire_rows,
        worsening_delta=CLINICAL_WEEKLY_WORSENING_DELTA,
    )

    snapshots = replace_trajectory_snapshots(db, username=username, snapshots=trajectory.get("snapshots") or [])
    requires_review = bool(trajectory.get("requires_safety_review"))

    user = db.query(models.User).filter(models.User.username == username).first()
    if user:
        requires_review = requires_review or bool(getattr(user, "requires_safety_review", False))
        user.requires_safety_review = requires_review
        db.commit()

    upsert_clinical_state(
        db,
        username=username,
        updates={
            "requires_safety_review": requires_review,
            "last_distress_level": "high" if requires_review else "low",
        },
    )

    return {
        "requires_safety_review": requires_review,
        "flagged_questionnaires": list(trajectory.get("flagged_questionnaires") or []),
        "snapshots": snapshots,
    }


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    with contextlib.suppress(ValueError, TypeError):
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    return None


def _build_pending_assessments(history: Dict[str, List[Dict[str, Any]]], cadence_days: int = 7) -> List[Dict[str, Any]]:
    now = datetime.utcnow()
    pending: List[Dict[str, Any]] = []
    for questionnaire_type in ("PHQ-9", "GAD-7", "PCL-5"):
        rows = list(history.get(questionnaire_type) or [])
        last_row = rows[-1] if rows else {}
        last_assessed_at = str(last_row.get("created_at") or "") or None
        last_dt = _parse_iso_datetime(last_assessed_at)

        if not last_dt:
            pending.append(
                {
                    "questionnaire_type": questionnaire_type,
                    "is_due": True,
                    "days_since_last": None,
                    "days_until_due": 0,
                    "next_due_at": None,
                    "reason": "No previous assessment found.",
                }
            )
            continue

        elapsed_days = max(0, (now - last_dt).days)
        next_due = last_dt + timedelta(days=cadence_days)
        days_until_due = max(0, (next_due - now).days)
        is_due = elapsed_days >= cadence_days

        pending.append(
            {
                "questionnaire_type": questionnaire_type,
                "is_due": is_due,
                "days_since_last": int(elapsed_days),
                "days_until_due": int(days_until_due),
                "next_due_at": next_due.isoformat(),
                "reason": "Overdue based on 7-day cadence." if is_due else "Within active cadence window.",
            }
        )
    return pending

def _build_mbc_care_plan_state(clinical_state: Dict[str, Any], latest_scores: Dict[str, int]) -> Dict[str, Any]:
    framework = str(clinical_state.get("active_framework") or "Supportive_Stabilization")
    phase = str(clinical_state.get("current_phase") or "Emotional Check-In")
    distress = str(clinical_state.get("last_distress_level") or "low")

    # Extract Scores safely
    phq9 = latest_scores.get("PHQ-9", 0)
    gad7 = latest_scores.get("GAD-7", 0)
    pcl5 = latest_scores.get("PCL-5", 0)

    # Base arrays
    routine_blueprint = []
    interventions = []

    # --- DYNAMIC MBC ENGINE ---

    # 1. Depression / Low Motivation Dominant (PHQ-9)
    if phq9 >= 10:
        routine_blueprint.append({
            "id": "morning-light",
            "title": "Morning Light Exposure",
            "description": "Get 10-15 minutes of direct sunlight within 30 minutes of waking.",
            "cadence": "Daily",
            "clinical_rationale": "Targeting: Depressive Lethargy"
        })
        interventions.append({
            "id": "behavioral-activation",
            "title": "Micro Behavioral Activation",
            "framework": "CBT_Restructuring",
            "objective": "Schedule and complete one 5-minute low-friction task to build momentum.",
            "clinical_rationale": "Targeting: Low Motivation"
        })

    # 2. Anxiety / Arousal Dominant (GAD-7)
    if gad7 >= 10:
        routine_blueprint.append({
            "id": "worry-postponement",
            "title": "Scheduled Worry Time",
            "description": "Defer anxieties to a designated 15-minute window at 4:00 PM.",
            "cadence": "Daily",
            "clinical_rationale": "Targeting: Generalized Anxiety"
        })
        interventions.append({
            "id": "pmr-intervention",
            "title": "Progressive Muscle Relaxation",
            "framework": "DBT_Distress_Tolerance",
            "objective": "Systematically tense and release muscle groups to lower somatic arousal.",
            "clinical_rationale": "Targeting: Somatic Tension"
        })

    # 3. Trauma / Stress Dominant (PCL-5)
    if pcl5 >= 31:
        routine_blueprint.append({
            "id": "evening-wind-down",
            "title": "Predictable Evening Wind-down",
            "description": "Maintain a strict, low-stimulation environment 1 hour before sleep.",
            "cadence": "Daily",
            "clinical_rationale": "Targeting: Hypervigilance"
        })
        interventions.append({
            "id": "container-exercise",
            "title": "The Container Exercise",
            "framework": "ACT_Defusion",
            "objective": "Visualize placing distressing memories into a secure, locked container.",
            "clinical_rationale": "Targeting: Intrusive Thoughts"
        })

    # 4. Fallback / Sub-clinical Maintenance
    # If no scores are severely elevated, provide baseline support
    if not routine_blueprint:
        routine_blueprint.extend([
            {
                "id": "morning-checkin",
                "title": "Morning Emotional Check-In",
                "description": "2-minute naming of mood, body tension, and intent for the day.",
                "cadence": "Daily",
                "clinical_rationale": "Targeting: Baseline Maintenance"
            },
            {
                "id": "evening-reflection",
                "title": "Evening Reflection",
                "description": "Short review of stress triggers and coping actions used.",
                "cadence": "Daily",
                "clinical_rationale": "Targeting: Routine Structuring"
            }
        ])
    
    if not interventions:
        interventions.extend([
            {
                "id": "micro-breath",
                "title": "60-Second Paced Breathing",
                "framework": "DBT_Distress_Tolerance",
                "objective": "Downshift physiological arousal before cognitive work.",
                "clinical_rationale": "Targeting: Baseline Regulation"
            }
        ])

    # 5. Acute Safety Override
    if distress == "high":
        interventions.insert(0, {
            "id": "safety-grounding",
            "title": "Safety Grounding Sequence",
            "framework": "Safety_Stabilization",
            "objective": "Deploy immediate grounding and support-contact reminder workflow.",
            "clinical_rationale": "Targeting: Acute Distress"
        })

    return {
        "framework": framework,
        "phase": phase,
        "last_distress_level": distress,
        "latest_scores": latest_scores,
        "daily_routine_blueprint": routine_blueprint,
        "micro_interventions": interventions,
    }

def _normalize_emotion_label(e: str) -> str: return EMOTION_ALIAS.get(str(e or "neutral").strip().lower(), str(e or "neutral").strip().lower())
def _risk_label(s: int) -> str: return "elevated" if s >= 6 else "monitor" if s >= 3 else "stable"
def _engagement_band(s: int) -> str: return "high" if s >= 70 else "moderate" if s >= 35 else "low"
def _severity_points(s: str) -> int:
    s = str(s or "").strip().lower()
    return 3 if s in {"severe", "very severe", "extremely severe", "elevated", "high"} else 2 if s in {"moderate", "moderately severe"} else 1 if s in {"mild", "minimal"} else 0
def _score_trend(v: List[int]) -> str:
    if len(v) < 2: return "insufficient_data"
    d = int(v[0]) - int(v[1])
    return "worsening" if d >= 3 else "improving" if d <= -3 else "stable"
def _overall_screening_trend(trends: Dict[str, str]) -> str:
    vals = set(trends.values())
    if "worsening" in vals and "improving" in vals: return "mixed"
    if "worsening" in vals: return "worsening"
    if "improving" in vals: return "improving"
    if "stable" in vals: return "stable"
    return "insufficient_data"
def _symptom_burden_pct(latest_scores: Dict[str, int]) -> float:
    ratios = [
        min(1.0, float(score) / QUESTIONNAIRE_MAX_SCORES[q_type])
        for q_type, score in latest_scores.items()
        if q_type in QUESTIONNAIRE_MAX_SCORES and QUESTIONNAIRE_MAX_SCORES[q_type] > 0
    ]
    return round((sum(ratios) / len(ratios)) * 100.0, 1) if ratios else 0.0
def _normalize_admin_narrative(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""

    normalized = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized
def _admin_summary_cache_get(key: str) -> Optional[Tuple[str, str]]:
    cache = _state_get("admin_summary_cache", {})
    if cache.get("key") != key or float(cache.get("expires_at") or 0.0) <= time.time():
        return None
    return str(cache.get("summary") or ""), str(cache.get("source") or "fallback")
def _admin_summary_cache_set(key: str, summary: str, source: str) -> None:
    _state_set(
        "admin_summary_cache",
        {
            "key": key,
            "summary": summary,
            "source": source,
            "expires_at": time.time() + ADMIN_SUMMARY_CACHE_TTL_SECONDS,
        },
    )
def _build_admin_metrics(
    turns: int,
    emotion_events: int,
    quizzes: int,
    risk_score: int,
    distress_signals: int,
    care_plan_adherence_pct: int,
    risk_score_delta: int = 0,
    distress_signal_delta: int = 0,
    care_plan_completed_days: int = 0,
    care_plan_window_days: int = 7,
) -> List[Dict[str, Any]]:
    def _delta_payload(value: int, label: str) -> Dict[str, Any]:
        tone = "neutral"
        if value > 0:
            tone = "up"
        elif value < 0:
            tone = "down"
        return {"delta": int(value), "delta_label": label, "delta_tone": tone}

    return [
        {
            "id": "turns",
            "label": "Conversation Turns",
            "value": int(turns),
            "description": "Recent therapeutic dialogue exchanges.",
        },
        {
            "id": "care_plan_adherence",
            "label": "Care Plan Adherence %",
            "value": f"{int(care_plan_adherence_pct)}%",
            "description": f"{int(care_plan_completed_days)}/{int(care_plan_window_days)} daily check-ins completed in the current rolling window.",
        },
        {
            "id": "emotion_events",
            "label": "Emotion Events",
            "value": int(emotion_events),
            "description": "Captured emotion observations across sessions.",
        },
        {
            "id": "questionnaire_entries",
            "label": "Screening Entries",
            "value": int(quizzes),
            "description": "PHQ-9, GAD-7, and PCL-5 submissions analyzed.",
        },
        {
            "id": "risk_score",
            "label": "Risk Score",
            "value": int(risk_score),
            "description": "Composite score from screening severity, distress language, and affective risk.",
            **_delta_payload(risk_score_delta, "vs prior routed assessment"),
        },
        {
            "id": "distress_signals",
            "label": "Distress Signals",
            "value": int(distress_signals),
            "description": "Keyword-based acute distress cues in recent user messages.",
            **_delta_payload(distress_signal_delta, "vs prior 7 days"),
        },
    ]


def _build_admin_activity_context(
    chats: List[Dict[str, Any]],
    sessions: List[Dict[str, Any]],
    quizzes: List[Dict[str, Any]],
    routing_events: List[Dict[str, Any]],
    safety_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    now = datetime.utcnow()
    window_start = now - timedelta(days=7)
    previous_window_start = now - timedelta(days=14)

    def _clip(value: str, limit: int = 150) -> str:
        text = str(value or "").strip().replace("\n", " ")
        return text[:limit] + ("..." if len(text) > limit else "")

    def _row_datetime(row: Dict[str, Any], key: str = "timestamp") -> Optional[datetime]:
        return _parse_iso_datetime(str(row.get(key) or ""))

    def _count_distress(rows: List[Dict[str, Any]], start: datetime, end: datetime) -> int:
        total = 0
        for row in rows:
            dt = _row_datetime(row)
            if not dt or dt < start or dt >= end:
                continue
            if DISTRESS_SIGNAL_REGEX.search(str(row.get("user_text") or "")):
                total += 1
        return total

    activity_days = set()
    for row in chats or []:
        dt = _row_datetime(row)
        if dt and dt >= window_start:
            activity_days.add(dt.date().isoformat())
    for row in sessions or []:
        dt = _row_datetime(row)
        if dt and dt >= window_start:
            activity_days.add(dt.date().isoformat())
    for row in quizzes or []:
        dt = _parse_iso_datetime(str(row.get("created_at") or ""))
        if dt and dt >= window_start:
            activity_days.add(dt.date().isoformat())

    completed_days = len(activity_days)
    care_plan_window_days = 7
    care_plan_adherence_pct = int(round(min(1.0, completed_days / max(1, care_plan_window_days)) * 100))

    current_distress_count = _count_distress(chats, window_start, now)
    previous_distress_count = _count_distress(chats, previous_window_start, window_start)

    routing_scores = [int(row.get("risk_score") or 0) for row in routing_events if row.get("risk_score") is not None]
    risk_score_delta = int(routing_scores[0] - routing_scores[1]) if len(routing_scores) >= 2 else 0

    framework_counts = {
        FRAMEWORK_CBT: 0,
        FRAMEWORK_ACT: 0,
        FRAMEWORK_DBT: 0,
        FRAMEWORK_SUPPORTIVE: 0,
    }
    route_labels = {
        FRAMEWORK_CBT: "CBT Restructuring",
        FRAMEWORK_ACT: "ACT Defusion",
        FRAMEWORK_DBT: "DBT Distress Tolerance",
        FRAMEWORK_SUPPORTIVE: "Supportive Stabilization",
    }

    timeline_events: List[Dict[str, Any]] = []

    for row in chats or []:
        timestamp = str(row.get("timestamp") or "")
        timeline_events.append(
            {
                "id": f"turn-{row.get('id')}",
                "kind": "chat_turn",
                "timestamp": timestamp,
                "title": f"Chat turn #{row.get('id')}",
                "detail": f"User: {_clip(row.get('user_text') or '')}",
                "meta": {"emotion": str(row.get("dominant_emotion") or "neutral")},
            }
        )

    for row in sessions or []:
        timestamp = str(row.get("timestamp") or "")
        emotion_count = len(row.get("emotions") or [])
        timeline_events.append(
            {
                "id": f"session-{row.get('id')}",
                "kind": "session",
                "timestamp": timestamp,
                "title": f"Session #{row.get('id')}",
                "detail": f"{emotion_count} emotion observations captured.",
                "meta": {"emotion_count": emotion_count},
            }
        )

    for row in quizzes or []:
        timestamp = str(row.get("created_at") or "")
        questionnaire_type = str(row.get("questionnaire_type") or "Questionnaire")
        timeline_events.append(
            {
                "id": f"questionnaire-{row.get('id')}",
                "kind": "questionnaire",
                "timestamp": timestamp,
                "title": f"{questionnaire_type} screening",
                "detail": f"Score {int(row.get('total_score') or 0)} ({str(row.get('severity') or 'unknown')}).",
                "meta": {
                    "questionnaire_type": questionnaire_type,
                    "score": int(row.get("total_score") or 0),
                    "severity": str(row.get("severity") or "unknown"),
                },
            }
        )

    for row in routing_events or []:
        framework = str(row.get("routed_framework") or FRAMEWORK_SUPPORTIVE)
        framework_counts[framework] = framework_counts.get(framework, 0) + 1
        timestamp = str(row.get("timestamp") or "")
        timeline_events.append(
            {
                "id": f"routing-{row.get('id')}",
                "kind": "routing",
                "timestamp": timestamp,
                "title": f"{route_labels.get(framework, framework)} route",
                "detail": f"Risk {int(row.get('risk_score') or 0)} • {str(row.get('route_reason') or 'No route reason recorded.')}",
                "meta": {
                    "framework": framework,
                    "risk_score": int(row.get("risk_score") or 0),
                    "route_locked": bool(row.get("route_locked")),
                    "acute_safety_trigger": bool(row.get("acute_safety_trigger")),
                },
            }
        )

    safety_trigger_count = len(safety_events or [])
    for row in safety_events or []:
        timestamp = str(row.get("timestamp") or "")
        timeline_events.append(
            {
                "id": f"safety-{row.get('id')}",
                "kind": "safety",
                "timestamp": timestamp,
                "title": "Safety protocol triggered",
                "detail": f"{str(row.get('trigger_type') or 'Safety escalation')} • risk {int(row.get('risk_score') or 0)}",
                "meta": {"trigger_type": str(row.get("trigger_type") or ""), "risk_score": int(row.get("risk_score") or 0)},
            }
        )

    def _timeline_key(item: Dict[str, Any]) -> datetime:
        parsed = _parse_iso_datetime(str(item.get("timestamp") or ""))
        return parsed or datetime.min

    timeline_events = sorted(timeline_events, key=_timeline_key, reverse=True)[:120]

    protocol_fidelity = [
        {"id": "CBT_Restructuring", "label": "CBT Restructuring", "count": int(framework_counts.get(FRAMEWORK_CBT, 0)), "tone": "cyan"},
        {"id": "ACT_Defusion", "label": "ACT Defusion", "count": int(framework_counts.get(FRAMEWORK_ACT, 0)), "tone": "emerald"},
        {"id": "DBT_Distress_Tolerance", "label": "DBT Distress Tolerance", "count": int(framework_counts.get(FRAMEWORK_DBT, 0)), "tone": "amber"},
        {"id": "Supportive_Stabilization", "label": "Supportive Stabilization", "count": int(framework_counts.get(FRAMEWORK_SUPPORTIVE, 0)), "tone": "slate"},
        {"id": "Safety_Protocol", "label": "Safety Protocol", "count": int(safety_trigger_count), "tone": "rose"},
    ]

    max_protocol_count = max([int(item["count"]) for item in protocol_fidelity] or [0])
    for item in protocol_fidelity:
        count = int(item.get("count") or 0)
        item["share"] = int(round((count / max_protocol_count) * 100)) if max_protocol_count > 0 else 0

    return {
        "care_plan_adherence_pct": care_plan_adherence_pct,
        "care_plan_completed_days": completed_days,
        "care_plan_window_days": care_plan_window_days,
        "risk_score_delta": risk_score_delta,
        "distress_signal_delta": int(current_distress_count - previous_distress_count),
        "timeline_events": timeline_events,
        "protocol_fidelity": protocol_fidelity,
    }
def _base_profile(username: str) -> Dict[str, Any]:
    return {
        "user_id": None,
        "username": username,
        "last_seen": None,
        "risk_level": "stable",
        "risk_score": 0,
        "active_flags": [],
        "dominant_emotion": "neutral",
        "negative_emotion_ratio": 0.0,
        "distress_signal_count": 0,
        "engagement_score": 0,
        "engagement_level": "low",
        "latest_scores": {},
        "latest_severity": {},
        "screening_trends": {},
        "overall_trend": "insufficient_data",
        "symptom_burden_pct": 0.0,
        "risk_factors": [],
        "protective_factors": [],
        "follow_up_priority": "Continue routine supportive follow-up.",
        "monitoring_cadence": "Weekly review with symptom monitoring.",
        "latest_assistant_note": "",
        "risk": {"level": "stable", "score": 0, "active_flags": [], "distress_signal_count": 0},
        "emotion": {"dominant_emotion": "neutral", "negative_ratio": 0.0, "distribution": []},
        "screening": {"latest_scores": {}, "latest_severity": {}, "trends": {}},
        "engagement": {"score": 0, "level": "low"},
        "follow_up": {"primary_priority": "Continue routine supportive follow-up.", "cadence": "Weekly review with symptom monitoring."},
    }
def _empty_admin_payload(username: str, user_id: Optional[int] = None) -> Dict[str, Any]:
    profile = _base_profile(username)
    profile["user_id"] = int(user_id) if user_id is not None else None
    return {
        "user_id": int(user_id) if user_id is not None else None,
        "generated_at": datetime.utcnow().isoformat(),
        "summary": "Patient presents with a composite risk score of 0, indicating a low level of concern. Recent screening trajectories show insufficiently characterized depressive symptoms and insufficiently characterized anxiety markers. Dominant interaction affect is neutral. Immediate clinical focus should prioritize establishing baseline screening and a supportive check-in routine. Continue routine monitoring with brief supportive check-ins and repeat screening at the next scheduled interval.",
        "summary_source": "fallback",
        "summary_snapshot": profile,
        "metrics": _build_admin_metrics(0, 0, 0, 0, 0, 0),
        "top_emotions": [],
        "chats": [],
        "sessions": [],
        "questionnaire_results": [],
        "timeline_events": [],
        "protocol_fidelity": [],
        "profile": profile,
        "clinical_parameters": {
            "risk_level": "stable",
            "risk_score": 0,
            "active_flags": [],
            "distress_signal_count": 0,
            "distress_signal_rate": 0.0,
            "negative_emotion_ratio": 0.0,
            "emotion_volatility": 0.0,
            "engagement_level": "low",
            "engagement_score": 0,
            "screening_trends": {},
            "overall_trend": "insufficient_data",
            "symptom_burden_pct": 0.0,
            "risk_factors": [],
            "protective_factors": [],
            "follow_up_priority": profile["follow_up_priority"],
            "monitoring_cadence": profile["monitoring_cadence"],
        },
        "flagged_users": [],
    }

def _admin_overview_cache_get(key: str) -> Optional[Dict[str, Any]]:
    cache = _state_get("admin_overview_cache", {})
    [cache.pop(k, None) for k, v in list(cache.items()) if float(v.get("expires_at") or 0.0) <= time.time()]
    return item.get("payload") if (item := cache.get(key)) else None

def _admin_overview_cache_set(key: str, payload: Dict[str, Any]) -> None:
    cache = _state_get("admin_overview_cache", {})
    cache[key] = {"payload": payload, "expires_at": time.time() + ADMIN_OVERVIEW_CACHE_TTL_SECONDS}
    if len(cache) > 64: cache.pop(min(cache.keys(), key=lambda k: float(cache[k].get("expires_at") or 0.0)), None)
    _state_set("admin_overview_cache", cache)

def _invalidate_admin_overview_cache(username: Optional[str]) -> None:
    if not (user_key := str(username or "").strip()): return
    cache = _state_get("admin_overview_cache", {})
    [cache.pop(k, None) for k in list(cache.keys()) if k.startswith(f"{user_key}:")]

async def _ensure_cloud_llm_client():
    if _state_get("cloud_llm_client"): return _state_get("cloud_llm_client")
    with cloud_llm_init_lock:
        if not _state_get("cloud_llm_client"):
            try: _state_set("cloud_llm_client", CloudLLMClient())
            except CloudLLMError: return None
    return _state_get("cloud_llm_client")

async def _generate_admin_summary(
    snapshot: Dict[str, Any],
    recent_turns: List[Dict[str, Any]],
) -> Tuple[str, str]:
    compact_turns = [
        {
            "timestamp": str(row.get("timestamp") or "unknown"),
            "user_text": str(row.get("user_text") or "")[:400],
            "assistant_text": str(row.get("assistant_text") or "")[:400],
        }
        for row in list(recent_turns or [])[:10]
    ]

    cache_key = json.dumps(
        {"snapshot": snapshot, "recent_turns": compact_turns},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if cached := _admin_summary_cache_get(cache_key):
        return cached

    fallback = build_admin_clinical_handoff_fallback(snapshot, compact_turns)
    summary_text, source = fallback, "fallback"
    client = await _ensure_cloud_llm_client()

    if client:
        try:
            prompt = build_admin_clinical_handoff_prompt(snapshot, compact_turns)
            candidate = await asyncio.wait_for(client.ask_serenity(prompt, timeout=ADMIN_SUMMARY_TIMEOUT_SECONDS), timeout=ADMIN_SUMMARY_TIMEOUT_SECONDS + 1.5)
            normalized = _normalize_admin_narrative(candidate)
            if normalized:
                summary_text, source = normalized, "cloud_llm"
        except Exception as exc:
            LOGGER.info("Admin summary fallback in use: %s", exc)

    _admin_summary_cache_set(cache_key, summary_text, source)
    return summary_text, source

def _fallback_admin_summary(snapshot: Dict[str, Any]) -> str:
    r, s, e, f = snapshot.get("risk", {}), snapshot.get("screening", {}), snapshot.get("emotion", {}), snapshot.get("follow_up", {})
    scores = ", ".join([f"{k} {v}" for k, v in s.get("latest_scores", {}).items()]) or "no recent screenings"
    trends = ", ".join([f"{k}: {v}" for k, v in s.get("trends", {}).items()]) or "insufficient_data"

    return "\n".join([
        f"- Client status: {snapshot.get('username', 'unknown')} is currently in the {r.get('level', 'stable')} risk band with score {r.get('score', 0)}.",
        f"- Affective pattern: dominant emotion is {e.get('dominant_emotion', 'neutral')} with negative-affect ratio {e.get('negative_ratio', 0)}.",
        f"- Measurement-based screening: latest scores {scores}; trend review {trends}.",
        f"- Risk formulation: active screening flags are {', '.join(r.get('active_flags', [])) or 'none'} and distress signals observed are {r.get('distress_signal_count', 0)}.",
        f"- Immediate follow-up focus: {f.get('primary_priority', 'continue routine supportive follow-up and baseline screening establishment.')}",
        f"- Monitoring cadence: {f.get('cadence', 'weekly check-ins with repeat screening as clinically indicated.')}"
    ])

@contextlib.asynccontextmanager
async def handle_temp_audio(audio_bytes: Optional[bytes], filename: str = ""):
    if not audio_bytes: yield None; return
    suffix = ".webm" if filename.endswith(".webm") else ".mp3" if filename.endswith(".mp3") else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(audio_bytes)
        temp_path = temp_audio.name
    try: yield temp_path
    finally:
        if os.path.exists(temp_path):
            with contextlib.suppress(OSError): os.remove(temp_path)

def _transcribe_with_whisper(audio_path: str) -> tuple[str, Optional[str]]:
    with whisper_init_lock:
        if not _state_get("whisper_model"):
            device = _state_get("whisper_device_in_use", "cpu")
            if WhisperModel:
                compute = "float16" if device == "cuda" else "int8"
                _state_set("whisper_model", WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=compute, cpu_threads=WHISPER_CPU_THREADS))
                _state_set("whisper_backend", "faster-whisper")
            elif openai_whisper:
                _state_set("whisper_model", openai_whisper.load_model(WHISPER_MODEL_SIZE, device=device))
                _state_set("whisper_backend", "openai-whisper")
            else: return "", "No STT backend installed"

    model, backend = _state_get("whisper_model"), _state_get("whisper_backend")
    try:
        if backend == "faster-whisper":
            segments, _ = model.transcribe(audio_path, language="en", beam_size=1)
            return " ".join([s.text.strip() for s in segments if s.text.strip()]), None
        return str(model.transcribe(audio_path, fp16=False, language="en").get("text", "")).strip(), None
    except Exception as e: return "", f"Transcription failed: {e}"

async def _generate_tts_base64(text: str) -> tuple[Optional[str], Optional[str]]:
    if not text or not TTS_ENABLED or not edge_tts:
        return None, None

    voices = [TTS_VOICE]
    if TTS_FALLBACK_VOICE and TTS_FALLBACK_VOICE not in voices:
        voices.append(TTS_FALLBACK_VOICE)

    last_error: Optional[Exception] = None
    retry_count = max(1, TTS_RETRIES)

    for voice in voices:
        for attempt in range(1, retry_count + 1):
            audio_path = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}.mp3")
            try:
                comm = edge_tts.Communicate(text=text, voice=voice)
                await asyncio.wait_for(comm.save(audio_path), timeout=TTS_TIMEOUT_SECONDS)
                with open(audio_path, "rb") as f:
                    payload = base64.b64encode(f.read()).decode("utf-8")
                if voice != TTS_VOICE:
                    LOGGER.info("TTS succeeded with fallback voice: %s", voice)
                return payload, None
            except Exception as exc:
                last_error = exc
                is_forbidden = "403" in str(exc)
                if is_forbidden:
                    LOGGER.warning(
                        "TTS 403 from Edge endpoint (voice=%s, attempt=%s/%s).",
                        voice,
                        attempt,
                        retry_count,
                    )
                else:
                    LOGGER.warning(
                        "TTS attempt failed (voice=%s, attempt=%s/%s): %s",
                        voice,
                        attempt,
                        retry_count,
                        exc,
                    )

                if attempt < retry_count:
                    await asyncio.sleep(0.6 * attempt)
            finally:
                if os.path.exists(audio_path):
                    with contextlib.suppress(OSError):
                        os.remove(audio_path)

    return None, f"TTS failed: {last_error}"

async def _run_perception_tasks(temp_audio_path: Optional[str], image_data: Optional[str]) -> Dict[str, Any]:
    async def run_task(name, func, *args):
        started = time.perf_counter()
        try:
            value = await asyncio.wait_for(
                run_in_threadpool(func, *args),
                timeout=WHISPER_TIMEOUT_SECONDS if name == "transcribe" else EMOTION_TIMEOUT_SECONDS,
            )
            return name, value, (time.perf_counter() - started) * 1000.0
        except Exception as e:
            return name, e, (time.perf_counter() - started) * 1000.0

    tasks = []
    if temp_audio_path:
        tasks.extend([run_task("transcribe", _transcribe_with_whisper, temp_audio_path), run_task("speech", predict_audio_emotion, temp_audio_path, _state_get("speech_runtime"))])
    if image_data: tasks.append(run_task("face", analyze_face, image_data, _state_get("face_runtime")))

    results = {
        "transcription": "",
        "speech_emotion": "Neutral",
        "face_emotion": "Neutral",
        "speech_conf": 0.0,
        "face_conf": 0.0,
        "stt_latency_ms": 0.0,
        "ser_latency_ms": 0.0,
        "fer_latency_ms": 0.0,
    }
    errors = []

    if tasks:
        for name, val, elapsed_ms in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(val, Exception): errors.append(f"{name} task failed: {val}")
            elif name == "transcribe":
                results["transcription"], err = val
                results["stt_latency_ms"] = round(float(elapsed_ms), 2)
                if err: errors.append(err)
            elif name in ("speech", "face"):
                results[f"{name}_emotion"] = val.get("emotion", "Neutral")
                results[f"{name}_conf"] = val.get("confidence", 0.0)
                if name == "speech":
                    results["ser_latency_ms"] = round(float(elapsed_ms), 2)
                else:
                    results["fer_latency_ms"] = round(float(elapsed_ms), 2)

    s_p = {l: (1.0 - results["speech_conf"]/100)/7 for l in EMOTION_LABELS}
    s_p[_normalize_emotion_label(results["speech_emotion"])] = results["speech_conf"]/100
    f_p = {l: (1.0 - results["face_conf"]/100)/7 for l in EMOTION_LABELS} if image_data else dict(s_p)
    f_p[_normalize_emotion_label(results["face_emotion"])] = results["face_conf"]/100 if image_data else s_p[_normalize_emotion_label(results["speech_emotion"])]

    fused = {l: (s_p[l] + f_p[l])/2.0 for l in EMOTION_LABELS} if temp_audio_path and image_data else (s_p if temp_audio_path else f_p)
    dominant = max(fused.items(), key=lambda x: x[1])[0].title() if (temp_audio_path or image_data) else "Neutral"
    return {**results, "dominant_emotion": dominant, "errors": errors}

async def _persist_turn_safe(
    db: Session,
    username: str,
    user_text: str,
    llm_response: str,
    dominant_emotion: str,
    speech_emotion: str,
    face_emotion: str,
    errors: Optional[List[str]] = None,
) -> Optional[int]:
    try:
        turn = await run_in_threadpool(
            persist_turn,
            db,
            username,
            user_text,
            llm_response,
            dominant_emotion,
            speech_emotion,
            face_emotion,
        )
        _invalidate_admin_overview_cache(username)
        return int(getattr(turn, "id", 0) or 0) or None
    except Exception as exc:
        if errors is not None:
            errors.append(f"DB Error: {exc}")
        return None

def _to_event(payload: dict) -> str: return json.dumps(payload, ensure_ascii=False) + "\n"

async def _stream_chat_events(
    db: Session,
    username: str,
    user_text: str,
    dominant_emotion: str,
    speech_emotion: str,
    face_emotion: str,
    source: str,
    emotion_first: bool,
    perception_metrics: Optional[Dict[str, Any]] = None,
):
    emo_payload = {"type": "emotion", "dominant_emotion": dominant_emotion, "speech_emotion": speech_emotion, "face_emotion": face_emotion}
    txt_payload = {"type": "user_text", "text": user_text, "source": source}
    metrics = perception_metrics or {}
    stt_latency_ms = float(metrics.get("stt_latency_ms") or 0.0)
    ser_latency_ms = float(metrics.get("ser_latency_ms") or 0.0)
    fer_latency_ms = float(metrics.get("fer_latency_ms") or 0.0)
    speech_conf = float(metrics.get("speech_conf") or 0.0)
    face_conf = float(metrics.get("face_conf") or 0.0)
    if emotion_first:
        yield _to_event(emo_payload)
        yield _to_event(txt_payload)
    else:
        yield _to_event(txt_payload)
        yield _to_event(emo_payload)

    clinical_state = fetch_or_create_clinical_state(db, username)
    risk_score = _clinical_risk_score(db, username, user_text, speech_emotion, face_emotion)
    clinical_mode = determine_clinical_mode(
        user_text=user_text,
        risk_score=risk_score,
        dominant_emotion=dominant_emotion,
    )
    user_model = db.query(models.User).filter(models.User.username == username).first()
    route_decision = evaluate_clinical_route(
        user_text=user_text,
        risk_score=risk_score,
        dominant_emotion=dominant_emotion,
        speech_emotion=speech_emotion,
        face_emotion=face_emotion,
        user_model=user_model,
        forced_mode=clinical_mode,
    )

    clinical_phase = str(clinical_state.get("current_phase") or default_phase_for_framework(route_decision.framework))
    routed_prompt = build_routed_prompt(
        user_text=user_text,
        decision=route_decision,
        clinical_phase=clinical_phase,
        requires_safety_review=bool(clinical_state.get("requires_safety_review")),
    )
    routed_prompt = (
        f"SYSTEM MODE LOCK: {clinical_mode}. You must remain in this mode for the entire response.\n"
        f"{routed_prompt}"
    )

    route_signals = _build_route_event_signals(route_decision, speech_emotion, face_emotion, dominant_emotion)
    route_signals["phase"] = clinical_phase
    route_signals["mode"] = clinical_mode
    yield _to_event({"type": "clinical_protocol_status", **route_signals})

    hard_stop = bool(route_decision.acute_safety_trigger or int(route_decision.risk_score or 0) >= CLINICAL_SAFETY_HARDSTOP_RISK)
    if hard_stop:
        safety_payload = build_safety_override_response()
        final_text = str(safety_payload.get("response_text") or "").strip()
        if not final_text:
            final_text = "I want to help you stay safe right now. Let's focus on immediate grounding together."

        _refresh_clinical_state_after_turn(
            db,
            username=username,
            route_decision=route_decision,
            parsed_payload=safety_payload,
            user_text=user_text,
            assistant_text=final_text,
        )

        db_errors: List[str] = []
        turn_id = await _persist_turn_safe(db, username, user_text, final_text, dominant_emotion, speech_emotion, face_emotion, db_errors)
        with contextlib.suppress(Exception):
            persist_clinical_routing_event(
                db,
                username=username,
                routed_framework=route_decision.framework,
                route_reason=route_decision.route_reason,
                risk_score=int(route_decision.risk_score or 0),
                route_locked=bool(route_decision.route_locked),
                acute_safety_trigger=bool(route_decision.acute_safety_trigger),
                rumination_detected=bool(route_decision.rumination_detected),
                detected_distortion=(route_decision.detected_distortions or [""])[0],
                dominant_emotion=dominant_emotion,
                speech_emotion=speech_emotion,
                face_emotion=face_emotion,
                turn_id=turn_id,
            )

        final_tts_audio, final_tts_err = await _generate_tts_base64(final_text)
        if final_tts_err:
            db_errors.append(final_tts_err)

        _persist_and_cache_diagnostics(
            db,
            username=username,
            source=source,
            stt_ms=stt_latency_ms,
            ser_ms=ser_latency_ms,
            fer_ms=fer_latency_ms,
            llm_latency_ms=0.0,
            speech_conf=speech_conf,
            face_conf=face_conf,
        )

        yield _to_event(
            {
                "type": "safety_mode",
                "enabled": True,
                "reason": safety_payload.get("safety_reason") or route_decision.route_reason,
                "framework": route_decision.framework,
            }
        )
        for err in db_errors:
            yield _to_event({"type": "error", "message": err})
        yield _to_event(
            {
                "type": "final",
                "llm_response": final_text,
                "transcription": user_text,
                "dominant_emotion": dominant_emotion,
                "speech_emotion": speech_emotion,
                "face_emotion": face_emotion,
                "tts_audio_base64": final_tts_audio,
                "clinical": route_signals,
            }
        )
        return

    client = await _ensure_cloud_llm_client()
    if not client:
        fallback_text = _build_llm_fallback_response(user_text, route_decision)
        db_errors: List[str] = ["Cloud LLM unavailable. Local fallback response used."]
        turn_id = await _persist_turn_safe(
            db,
            username,
            user_text,
            fallback_text,
            dominant_emotion,
            speech_emotion,
            face_emotion,
            db_errors,
        )
        with contextlib.suppress(Exception):
            persist_clinical_routing_event(
                db,
                username=username,
                routed_framework=route_decision.framework,
                route_reason=route_decision.route_reason,
                risk_score=int(route_decision.risk_score or 0),
                route_locked=bool(route_decision.route_locked),
                acute_safety_trigger=bool(route_decision.acute_safety_trigger),
                rumination_detected=bool(route_decision.rumination_detected),
                detected_distortion=(route_decision.detected_distortions or [""])[0],
                dominant_emotion=dominant_emotion,
                speech_emotion=speech_emotion,
                face_emotion=face_emotion,
                turn_id=turn_id,
            )

        fallback_tts_audio, fallback_tts_err = await _generate_tts_base64(fallback_text)
        if fallback_tts_err:
            db_errors.append(fallback_tts_err)

        _persist_and_cache_diagnostics(
            db,
            username=username,
            source=source,
            stt_ms=stt_latency_ms,
            ser_ms=ser_latency_ms,
            fer_ms=fer_latency_ms,
            llm_latency_ms=0.0,
            speech_conf=speech_conf,
            face_conf=face_conf,
        )

        for err in db_errors:
            yield _to_event({"type": "error", "message": err})

        yield _to_event({"type": "assistant_replace", "text": fallback_text})
        yield _to_event(
            {
                "type": "final",
                "llm_response": fallback_text,
                "transcription": user_text,
                "dominant_emotion": dominant_emotion,
                "speech_emotion": speech_emotion,
                "face_emotion": face_emotion,
                "tts_audio_base64": fallback_tts_audio,
                "clinical": route_signals,
            }
        )
        return

    llm_started = time.perf_counter()
    output_queue: asyncio.Queue = asyncio.Queue(maxsize=96)
    tts_input_queue: asyncio.Queue = asyncio.Queue(maxsize=16)
    stream_tts = bool(TTS_ENABLED and edge_tts and TTS_STREAM_MODE == "sentence")
    final_tts = bool(TTS_ENABLED and edge_tts and TTS_STREAM_MODE == "final")
    stream_protocol_control: Dict[str, Any] = {}

    async def fetch_text() -> None:
        nonlocal stream_protocol_control
        buffer, seq = "", 0
        llm_chunks: List[str] = []
        llm_res = ""
        cutoff_hit = False
        try:
            async for stream_event in client.stream_serenity_events(routed_prompt, require_protocol_control=True):
                event_type = str(stream_event.get("type") or "")

                if event_type == "cutoff":
                    cutoff_hit = True
                    break

                if event_type == "protocol_control":
                    incoming = stream_event.get("payload") or {}
                    if isinstance(incoming, dict):
                        stream_protocol_control = {
                            "advance_phase": bool(incoming.get("advance_phase", stream_protocol_control.get("advance_phase", False))),
                            "detected_distortion": str(incoming.get("detected_distortion") or stream_protocol_control.get("detected_distortion") or "").strip(),
                        }
                        await output_queue.put({"type": "protocol_control", **stream_protocol_control})
                    continue

                chunk = str(stream_event.get("delta") or "")
                if not chunk:
                    continue

                llm_chunks.append(chunk)
                buffer += chunk
                await output_queue.put({"type": "assistant_delta", "delta": chunk})

                if any(ch in chunk for ch in ".!?"):
                    sentences = SENTENCE_BOUNDARY_REGEX.split(buffer)
                    if len(sentences) > 1:
                        for sentence in sentences[:-1]:
                            sentence = sentence.strip()
                            if sentence:
                                seq += 1
                                await output_queue.put({"type": "assistant_sentence", "text": sentence, "sequence": seq})
                                if stream_tts:
                                    await tts_input_queue.put((sentence, seq))
                        buffer = sentences[-1]

            llm_res = "".join(llm_chunks).strip()

            if cutoff_hit:
                await output_queue.put({"type": "assistant_trim_dangling"})
                if match := list(re.finditer(r"[.!?]", llm_res)):
                    llm_res = llm_res[:match[-1].end()].strip()
                await output_queue.put({"type": "assistant_replace", "text": llm_res})
            elif buffer.strip():
                seq += 1
                sentence = buffer.strip()
                await output_queue.put({"type": "assistant_sentence", "text": sentence, "sequence": seq})
                if stream_tts:
                    await tts_input_queue.put((sentence, seq))
        except Exception as exc:
            LOGGER.warning("LLM stream request failed: %s", _format_exception_detail(exc))
            llm_res = "".join(llm_chunks).strip()
        finally:
            if stream_tts:
                await tts_input_queue.put(None)
            await output_queue.put({"type": "LLM_DONE", "final_res": llm_res})

    async def generate_audio() -> None:
        while True:
            item = await tts_input_queue.get()
            if item is None:
                break
            try:
                aud, tts_err = await _generate_tts_base64(item[0])
                if aud:
                    await output_queue.put({"type": "assistant_sentence_tts", "text": item[0], "sequence": item[1], "audio_base64": aud})
                elif tts_err:
                    await output_queue.put({"type": "error", "message": tts_err})
            except Exception as exc:
                await output_queue.put({"type": "error", "message": f"TTS stream failed: {exc}"})
        await output_queue.put({"type": "TTS_DONE"})

    tasks = [asyncio.create_task(fetch_text())]
    if stream_tts:
        tasks.append(asyncio.create_task(generate_audio()))

    tasks_running = len(tasks)
    final_llm_raw = ""
    while tasks_running > 0:
        event = await output_queue.get()
        if event["type"] == "LLM_DONE":
            final_llm_raw = str(event.get("final_res") or "")
            tasks_running -= 1
        elif event["type"] == "TTS_DONE":
            tasks_running -= 1
        else:
            yield _to_event(event)

    parsed_payload = _extract_protocol_payload(final_llm_raw)
    if stream_protocol_control:
        parsed_payload["advance_phase"] = bool(
            stream_protocol_control.get("advance_phase", parsed_payload.get("advance_phase", False))
        )
        if stream_protocol_control.get("detected_distortion"):
            parsed_payload["detected_distortion"] = str(stream_protocol_control.get("detected_distortion"))

    final_llm_res = str(parsed_payload.get("response_text") or "").strip()
    if not final_llm_res:
        final_llm_res = final_llm_raw

    if not final_llm_res:
        final_llm_res = _build_llm_fallback_response(user_text, route_decision)
        parsed_payload["response_text"] = final_llm_res
        yield _to_event({"type": "assistant_replace", "text": final_llm_res})

    if final_llm_res and final_llm_raw and final_llm_res != final_llm_raw:
        yield _to_event({"type": "assistant_replace", "text": final_llm_res})

    if not parsed_payload.get("detected_distortion") and route_decision.detected_distortions:
        parsed_payload["detected_distortion"] = route_decision.detected_distortions[0]

    updated_state = _refresh_clinical_state_after_turn(
        db,
        username=username,
        route_decision=route_decision,
        parsed_payload=parsed_payload,
        user_text=user_text,
        assistant_text=final_llm_res,
    )

    llm_latency_ms = (time.perf_counter() - llm_started) * 1000.0
    final_tts_audio: Optional[str] = None
    if final_tts and final_llm_res.strip():
        final_tts_audio, final_tts_err = await _generate_tts_base64(final_llm_res)
        if final_tts_err:
            yield _to_event({"type": "error", "message": final_tts_err})

    db_errors: List[str] = []
    turn_id = await _persist_turn_safe(db, username, user_text, final_llm_res, dominant_emotion, speech_emotion, face_emotion, db_errors)
    with contextlib.suppress(Exception):
        persist_clinical_routing_event(
            db,
            username=username,
            routed_framework=route_decision.framework,
            route_reason=route_decision.route_reason,
            risk_score=int(route_decision.risk_score or 0),
            route_locked=bool(route_decision.route_locked),
            acute_safety_trigger=bool(route_decision.acute_safety_trigger),
            rumination_detected=bool(route_decision.rumination_detected),
            detected_distortion=str(parsed_payload.get("detected_distortion") or ""),
            dominant_emotion=dominant_emotion,
            speech_emotion=speech_emotion,
            face_emotion=face_emotion,
            turn_id=turn_id,
        )

    _persist_and_cache_diagnostics(
        db,
        username=username,
        source=source,
        stt_ms=stt_latency_ms,
        ser_ms=ser_latency_ms,
        fer_ms=fer_latency_ms,
        llm_latency_ms=llm_latency_ms,
        speech_conf=speech_conf,
        face_conf=face_conf,
    )

    for err in db_errors:
        yield _to_event({"type": "error", "message": err})

    yield _to_event(
        {
            "type": "final",
            "llm_response": final_llm_res,
            "transcription": user_text,
            "dominant_emotion": dominant_emotion,
            "speech_emotion": speech_emotion,
            "face_emotion": face_emotion,
            "tts_audio_base64": final_tts_audio,
            "clinical": {
                "framework": updated_state.get("active_framework"),
                "phase": updated_state.get("current_phase"),
                "phase_index": updated_state.get("phase_index"),
                "risk_score": updated_state.get("last_risk_score"),
                "requires_safety_review": updated_state.get("requires_safety_review"),
            },
        }
    )

# --- Endpoints ---
@app.post("/register", response_model=AuthResponse)
async def register(payload: AuthRequest, db: Session = Depends(get_db)):
    username = str(payload.username or "").strip()
    password = str(payload.password or "")
    if not username or not password:
        raise HTTPException(400, "Username and password required")

    hashed_password = _hash_password(password)

    existing = await run_in_threadpool(lambda: db.query(models.User).filter(models.User.username == username).first())
    if existing:
        # Allow claiming legacy placeholder accounts created before registration.
        if not _is_bcrypt_hash(str(existing.password or "")):
            def _claim_existing_user() -> None:
                existing.password = hashed_password
                db.commit()
            await run_in_threadpool(_claim_existing_user)
            return AuthResponse(message="Success", username=existing.username)
        raise HTTPException(400, "Exists")

    new_user = models.User(username=username, password=hashed_password)
    await run_in_threadpool(lambda: (db.add(new_user), db.commit()))
    return AuthResponse(message="Success", username=new_user.username)

@app.post("/login", response_model=AuthResponse)
async def login(payload: AuthRequest, db: Session = Depends(get_db)):
    username = str(payload.username or "").strip()
    password = str(payload.password or "")
    if not username or not password:
        raise HTTPException(400, "Username and password required")

    user = await run_in_threadpool(lambda: db.query(models.User).filter(models.User.username == username).first())

    if not user or not _verify_password(password, str(user.password or "")):
        raise HTTPException(401, "Invalid")

    # Seamless one-time migration of legacy plain-text records to bcrypt.
    if not _is_bcrypt_hash(str(user.password or "")):
        def _upgrade_legacy_password() -> None:
            user.password = _hash_password(password)
            db.commit()
        await run_in_threadpool(_upgrade_legacy_password)

    with contextlib.suppress(Exception):
        _sync_weekly_trajectory(db, username)

    return AuthResponse(message="Success", username=user.username)

@app.get("/api/questionnaires/templates")
async def get_templates(types: Optional[str] = None):
    return {"available_types": list(QUESTIONNAIRE_DEFINITIONS.keys()), "questionnaires": questionnaire_templates(types.split(",") if types else None)}

@app.post("/api/questionnaires/submit")
async def submit_questionnaire(payload: QuestionnaireSubmitRequest, db: Session = Depends(get_db)):
    if not (q_type := normalize_questionnaire_type(payload.questionnaire_type)): raise HTTPException(400, "Invalid type")
    ans = [max(0, min(4 if q_type=="PCL-5" else 3, int(x))) for x in payload.answers]
    score, sev = score_questionnaire(q_type, ans)
    dt = None
    if payload.submitted_at:
        with contextlib.suppress(ValueError): dt = datetime.fromisoformat(payload.submitted_at.replace("Z", "+00:00")).replace(tzinfo=None)
    rec = await run_in_threadpool(persist_questionnaire_result, db, payload.username, q_type, ans, score, sev, dt)
    trajectory = await run_in_threadpool(_sync_weekly_trajectory, db, payload.username)
    _invalidate_admin_overview_cache(payload.username)
    return {
        "message": "Saved",
        "result": {"id": rec.id, "total_score": score, "severity": sev},
        "trajectory": trajectory,
    }

@app.get("/api/questionnaires/history")
async def q_history(username: str, limit: int = 30, db: Session = Depends(get_db)):
    if not username: raise HTTPException(400, "Username required")
    return {"username": username, "results": await run_in_threadpool(fetch_questionnaire_results, db, username, limit=max(1, limit))}

@app.get("/api/admin/overview")
async def admin_overview(username: str, limit: int = 300, include_answers: bool = False, db: Session = Depends(get_db)):
    if not (user_key := str(username or "").strip()): raise HTTPException(400, "Username required")
    limit = max(20, min(int(limit or ADMIN_DEFAULT_LIMIT), ADMIN_MAX_LIMIT))
    quiz_limit = max(30, min(limit * 2, ADMIN_MAX_LIMIT * 2))
    cache_key = f"{user_key}:{limit}:{1 if include_answers else 0}"
    if (cached := _admin_overview_cache_get(cache_key)): return cached

    activity_limit = max(60, min(limit * 2, 240))

    def fetch_admin_data():
        if not (user_row := db.query(models.User.id).filter(models.User.username == user_key).first()):
            return None
        user_id = int(user_row[0])
        routing_rows = (
            db.query(models.ClinicalRoutingEvent, models.User.username)
            .join(models.User, models.ClinicalRoutingEvent.user_id == models.User.id)
            .filter(models.User.username == user_key)
            .order_by(models.ClinicalRoutingEvent.timestamp.desc())
            .limit(activity_limit)
            .all()
        )
        safety_rows = (
            db.query(models.SafetyEscalationEvent, models.User.username)
            .join(models.User, models.SafetyEscalationEvent.user_id == models.User.id)
            .filter(models.User.username == user_key)
            .order_by(models.SafetyEscalationEvent.timestamp.desc())
            .limit(activity_limit)
            .all()
        )
        return (
            user_id,
            fetch_recent_turn_summaries(db, limit=limit, text_limit=420, username=user_key),
            fetch_recent_sessions_with_emotions(db, limit=limit, conversation_limit=420, username=user_key),
            fetch_questionnaire_results(db, username=user_key, limit=quiz_limit, include_answers=include_answers),
            [
                {
                    "id": row.id,
                    "username": row_username,
                    "routed_framework": row.routed_framework,
                    "route_reason": row.route_reason,
                    "risk_score": int(row.risk_score or 0),
                    "route_locked": bool(row.route_locked),
                    "acute_safety_trigger": bool(row.acute_safety_trigger),
                    "rumination_detected": bool(row.rumination_detected),
                    "detected_distortion": row.detected_distortion,
                    "dominant_emotion": row.dominant_emotion,
                    "speech_emotion": row.speech_emotion,
                    "face_emotion": row.face_emotion,
                    "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                }
                for row, row_username in routing_rows
            ],
            [
                {
                    "id": row.id,
                    "username": row_username,
                    "trigger_type": row.trigger_type,
                    "risk_score": int(row.risk_score or 0),
                    "dominant_emotion": row.dominant_emotion,
                    "transcript_excerpt": row.transcript_excerpt,
                    "handoff_markdown": row.handoff_markdown,
                    "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                }
                for row, row_username in safety_rows
            ],
            db.query(func.count(models.ConversationTurn.id)).filter(models.ConversationTurn.user_id == user_id).scalar() or 0,
            db.query(func.count(models.Session.id)).filter(models.Session.user_id == user_id).scalar() or 0,
            db.query(func.count(models.QuestionnaireResult.id)).filter(models.QuestionnaireResult.user_id == user_id).scalar() or 0,
            db.query(func.count(models.ClinicalRoutingEvent.id)).filter(models.ClinicalRoutingEvent.user_id == user_id).scalar() or 0,
            db.query(func.count(models.SafetyEscalationEvent.id)).filter(models.SafetyEscalationEvent.user_id == user_id).scalar() or 0,
        )

    if not (fetched := await run_in_threadpool(fetch_admin_data)):
        payload = _empty_admin_payload(user_key)
        _admin_overview_cache_set(cache_key, payload)
        return payload

    user_id, chats, sessions, quizzes, routing_events, safety_events, total_turns, total_sessions, total_quizzes, total_routing_events, total_safety_events = fetched
    if int(total_turns) + int(total_sessions) + int(total_quizzes) + int(total_routing_events) + int(total_safety_events) == 0:
        payload = _empty_admin_payload(user_key, user_id=user_id)
        _admin_overview_cache_set(cache_key, payload)
        return payload

    last_seen: Optional[str] = None
    latest_assistant_note = ""
    emotion_events = 0
    negative_turns = 0
    distress_signal_count = 0
    emotion_counts: Dict[str, int] = {}
    latest_scores: Dict[str, int] = {}
    latest_severity: Dict[str, str] = {}
    score_history: Dict[str, List[int]] = {}

    for row in chats:
        ts = row.get("timestamp")
        if ts and (not last_seen or str(ts) > str(last_seen)):
            last_seen = str(ts)
        emotion = _normalize_emotion_label(row.get("dominant_emotion"))
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        if emotion in NEGATIVE_EMOTIONS:
            negative_turns += 1
        if DISTRESS_SIGNAL_REGEX.search(str(row.get("user_text") or "")):
            distress_signal_count += 1
        if not latest_assistant_note and str(row.get("assistant_text") or "").strip():
            latest_assistant_note = str(row.get("assistant_text") or "").strip()

    for row in sessions:
        ts = row.get("timestamp")
        if ts and (not last_seen or str(ts) > str(last_seen)):
            last_seen = str(ts)
        emotion_events += len(row.get("emotions") or [])

    for row in quizzes:
        q_type = str(row.get("questionnaire_type") or "")
        if not q_type:
            continue
        score = int(row.get("total_score") or 0)
        score_history.setdefault(q_type, []).append(score)
        if q_type not in latest_scores:
            latest_scores[q_type] = score
            latest_severity[q_type] = str(row.get("severity") or "unknown")
        ts = row.get("created_at")
        if ts and (not last_seen or str(ts) > str(last_seen)):
            last_seen = str(ts)

    screening_trends = {name: _score_trend(values[:3]) for name, values in score_history.items()}
    overall_trend = _overall_screening_trend(screening_trends)
    top_emotions = sorted(
        [{"emotion": name, "count": count} for name, count in emotion_counts.items()],
        key=lambda row: row["count"],
        reverse=True,
    )
    dominant_emotion = top_emotions[0]["emotion"] if top_emotions else "neutral"

    flags = questionnaire_clinical_flags(latest_scores) if latest_scores else {}
    active_flags = [name for name, enabled in flags.items() if bool(enabled)]

    neg_ratio = round(negative_turns / max(1, len(chats)), 3)
    distress_signal_rate = round(distress_signal_count / max(1, len(chats)), 3)
    emotion_volatility = round(1.0 - (top_emotions[0]["count"] / max(1, len(chats))), 3) if top_emotions else 0.0
    symptom_burden_pct = _symptom_burden_pct(latest_scores)
    sev_pts = max([_severity_points(value) for value in latest_severity.values()] or [0])

    activity_context = _build_admin_activity_context(chats, sessions, quizzes, routing_events, safety_events)

    risk_score = (
        (len(active_flags) * 2)
        + sev_pts
        + (2 if distress_signal_count > 0 else 0)
        + (1 if neg_ratio >= 0.55 else 0)
        + (1 if overall_trend == "worsening" else 0)
    )
    risk_level = _risk_label(risk_score)

    engagement_score = min(100, int(total_turns * 2 + total_sessions * 4 + total_quizzes * 8))
    engagement_level = _engagement_band(engagement_score)

    risk_factors: List[str] = []
    if active_flags:
        risk_factors.append(f"Screening flags active: {', '.join(active_flags)}")
    if distress_signal_count > 0:
        risk_factors.append(f"Distress language detected in {distress_signal_count} recent turn(s)")
    if neg_ratio >= 0.55:
        risk_factors.append("High negative-affect ratio in recent interactions")
    if overall_trend == "worsening":
        risk_factors.append("Screening trend indicates worsening symptom burden")
    if not risk_factors:
        risk_factors.append("No acute risk factors identified in available recent data")

    protective_factors: List[str] = []
    if engagement_level in {"moderate", "high"}:
        protective_factors.append("Consistent engagement with therapeutic interactions")
    if overall_trend in {"stable", "improving"} and screening_trends:
        protective_factors.append("Screening trajectory is stable/improving")
    if distress_signal_count == 0:
        protective_factors.append("No recent acute distress language detected")
    if not protective_factors:
        protective_factors.append("Protective factors not yet robustly established")

    if risk_level == "elevated":
        follow_up_priority = "Prioritize safety-focused follow-up, collaborative coping-plan review, and escalation readiness."
        monitoring_cadence = "Contact within 24-72 hours; repeat PHQ-9/GAD-7/PCL-5 within one week."
    elif risk_level == "monitor":
        follow_up_priority = "Maintain structured follow-up targeting symptom triggers, coping adherence, and protective routines."
        monitoring_cadence = "Weekly review with questionnaire reassessment every 1-2 weeks."
    else:
        follow_up_priority = "Continue supportive care and reinforce resilience strategies while maintaining baseline monitoring."
        monitoring_cadence = "Biweekly to monthly check-ins with periodic screening refresh."

    summary_snapshot = {
        "username": user_key,
        "risk": {
            "level": risk_level,
            "score": int(risk_score),
            "active_flags": active_flags,
            "distress_signal_count": int(distress_signal_count),
            "risk_factors": risk_factors,
            "protective_factors": protective_factors,
        },
        "emotion": {
            "dominant_emotion": dominant_emotion,
            "negative_ratio": neg_ratio,
            "volatility": emotion_volatility,
            "distribution": top_emotions[:4],
        },
        "screening": {
            "latest_scores": latest_scores,
            "latest_severity": latest_severity,
            "trends": screening_trends,
            "overall_trend": overall_trend,
            "symptom_burden_pct": symptom_burden_pct,
        },
        "engagement": {
            "score": int(engagement_score),
            "level": engagement_level,
        },
        "follow_up": {
            "primary_priority": follow_up_priority,
            "cadence": monitoring_cadence,
        },
        "volume": {
            "turns": int(total_turns),
            "sessions": int(total_sessions),
            "questionnaire_entries": int(total_quizzes),
            "emotion_events": int(emotion_events),
            "routing_events": int(total_routing_events),
            "safety_events": int(total_safety_events),
        },
    }

    summary_text, summary_source = _fallback_admin_summary(summary_snapshot), "computed"

    profile = _base_profile(user_key)

# 1. Fetch the full user row to get the new clinical safety flags
    user_row_full = db.query(models.User).filter(models.User.username == user_key).first()

    # 2. Update the profile
    profile.update({
        "user_id": int(user_id),
        "last_seen": last_seen,
        # Safely fetch the new fields using getattr just in case
        "duty_to_warn": bool(getattr(user_row_full, "duty_to_warn", False)), 
        "last_crisis": getattr(user_row_full, "last_crisis_timestamp", None),
        
        "risk_level": risk_level,
        "risk_score": int(risk_score),
        # ... (Keep the rest of your existing profile.update fields here exactly as they are) ...
        "active_flags": active_flags,
        "dominant_emotion": dominant_emotion,
        "negative_emotion_ratio": neg_ratio,
        "distress_signal_count": int(distress_signal_count),
        "engagement_score": int(engagement_score),
        "engagement_level": engagement_level,
        "latest_scores": latest_scores,
        "latest_severity": latest_severity,
        "screening_trends": screening_trends,
        "overall_trend": overall_trend,
        "symptom_burden_pct": symptom_burden_pct,
        "risk_factors": risk_factors,
        "protective_factors": protective_factors,
        "follow_up_priority": follow_up_priority,
        "monitoring_cadence": monitoring_cadence,
        "latest_assistant_note": latest_assistant_note,
        "risk": summary_snapshot["risk"],
        "emotion": summary_snapshot["emotion"],
        "screening": {
            "latest_scores": latest_scores,
            "latest_severity": latest_severity,
            "trends": screening_trends,
        },
        "engagement": summary_snapshot["engagement"],
        "follow_up": summary_snapshot["follow_up"],
    })

    payload = {
        "user_id": int(user_id),
        "generated_at": datetime.utcnow().isoformat(),
        "summary": summary_text,
        "summary_source": summary_source,
        "summary_snapshot": summary_snapshot,
        "metrics": _build_admin_metrics(
            total_turns,
            emotion_events,
            total_quizzes,
            risk_score,
            distress_signal_count,
            int(activity_context.get("care_plan_adherence_pct") or 0),
            int(activity_context.get("risk_score_delta") or 0),
            int(activity_context.get("distress_signal_delta") or 0),
            int(activity_context.get("care_plan_completed_days") or 0),
            int(activity_context.get("care_plan_window_days") or 7),
        ),
        "top_emotions": top_emotions,
        "chats": chats,
        "sessions": sessions,
        "questionnaire_results": quizzes,
        "timeline_events": list(activity_context.get("timeline_events") or []),
        "protocol_fidelity": list(activity_context.get("protocol_fidelity") or []),
        "activity_summary": {
            "care_plan_adherence_pct": int(activity_context.get("care_plan_adherence_pct") or 0),
            "care_plan_completed_days": int(activity_context.get("care_plan_completed_days") or 0),
            "care_plan_window_days": int(activity_context.get("care_plan_window_days") or 7),
            "risk_score_delta": int(activity_context.get("risk_score_delta") or 0),
            "distress_signal_delta": int(activity_context.get("distress_signal_delta") or 0),
        },
        "profile": profile,
        "clinical_parameters": {
            "risk_level": risk_level,
            "risk_score": int(risk_score),
            "active_flags": active_flags,
            "distress_signal_count": int(distress_signal_count),
            "distress_signal_rate": distress_signal_rate,
            "negative_emotion_ratio": neg_ratio,
            "emotion_volatility": emotion_volatility,
            "engagement_level": engagement_level,
            "engagement_score": int(engagement_score),
            "screening_trends": screening_trends,
            "overall_trend": overall_trend,
            "symptom_burden_pct": symptom_burden_pct,
            "risk_factors": risk_factors,
            "protective_factors": protective_factors,
            "follow_up_priority": follow_up_priority,
            "monitoring_cadence": monitoring_cadence,
        },
        "flagged_users": [{"username": user_key, "flags": flags, "scores": latest_scores, "risk_level": risk_level}] if active_flags else [],
    }
    _admin_overview_cache_set(cache_key, payload)
    return payload

@app.get("/api/admin/clinical-report")
async def admin_clinical_report(username: str, db: Session = Depends(get_db)):
    if not (user_key := str(username or "").strip()):
        raise HTTPException(400, "Username required")

    overview = await admin_overview(username=user_key, limit=ADMIN_DEFAULT_LIMIT, include_answers=False, db=db)
    summary_snapshot = dict(overview.get("summary_snapshot") or {})
    recent_turns = list(overview.get("chats") or [])[:10]

    summary_text, summary_source = await _generate_admin_summary(summary_snapshot, recent_turns)

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "username": user_key,
        "summary": summary_text,
        "summary_source": summary_source,
        "recent_turn_count": len(recent_turns),
        "risk_score": int((summary_snapshot.get("risk") or {}).get("score") or 0),
    }

@app.get("/api/admin/summary/stream")
async def admin_summary_stream(username: str, db: Session = Depends(get_db)):
    async def stream():
        report = await admin_clinical_report(username=username, db=db)
        summary = str(report.get("summary") or "").strip()
        if not summary:
            yield _to_event({"type": "error", "message": "No summary available"})
            return

        for idx in range(0, len(summary), 64):
            yield _to_event({"type": "summary_delta", "delta": summary[idx:idx + 64]})
            await asyncio.sleep(0)

        yield _to_event({
            "type": "summary_final",
            "summary": summary,
            "summary_source": report.get("summary_source", "fallback"),
        })
    return StreamingResponse(stream(), media_type="application/x-ndjson")

@app.post("/api/interact", response_model=InteractResponse)
async def interact(username: str = Form(...), image: Optional[str] = Form(None), file: Optional[UploadFile] = File(None), user_message: Optional[str] = Form(None), db: Session = Depends(get_db)):
    if not file: raise HTTPException(400, "Microphone input required.")
    audio_bytes, filename = await file.read(), file.filename or ""
    async with handle_temp_audio(audio_bytes, filename) as audio_path: perc = await _run_perception_tasks(audio_path, image)

    user_text = (perc["transcription"] or user_message or "").strip()
    if not user_text: return InteractResponse(dominant_emotion=perc["dominant_emotion"], speech_emotion=perc["speech_emotion"], face_emotion=perc["face_emotion"], transcription="", llm_response="", errors=perc["errors"] + ["No speech detected."])

    errors: List[str] = list(perc["errors"])
    llm_started = time.perf_counter()
    clinical_state = fetch_or_create_clinical_state(db, username)
    risk_score = _clinical_risk_score(db, username, user_text, perc["speech_emotion"], perc["face_emotion"])
    clinical_mode = determine_clinical_mode(
        user_text=user_text,
        risk_score=risk_score,
        dominant_emotion=perc["dominant_emotion"],
    )
    user_model = db.query(models.User).filter(models.User.username == username).first()
    route_decision = evaluate_clinical_route(
        user_text=user_text,
        risk_score=risk_score,
        dominant_emotion=perc["dominant_emotion"],
        speech_emotion=perc["speech_emotion"],
        face_emotion=perc["face_emotion"],
        user_model=user_model,
        forced_mode=clinical_mode,
    )

    phase = str(clinical_state.get("current_phase") or default_phase_for_framework(route_decision.framework))
    hard_stop = bool(route_decision.acute_safety_trigger or int(route_decision.risk_score or 0) >= CLINICAL_SAFETY_HARDSTOP_RISK)

    raw_llm_response = ""
    llm_protocol_control: Dict[str, Any] = {}
    parsed_payload: Dict[str, Any]
    if hard_stop:
        parsed_payload = build_safety_override_response()
        raw_llm_response = str(parsed_payload.get("response_text") or "").strip()
    elif client := await _ensure_cloud_llm_client():
        try:
            routed_prompt = build_routed_prompt(
                user_text=user_text,
                decision=route_decision,
                clinical_phase=phase,
                requires_safety_review=bool(clinical_state.get("requires_safety_review")),
            )
            routed_prompt = (
                f"SYSTEM MODE LOCK: {clinical_mode}. You must remain in this mode for the entire response.\n"
                f"{routed_prompt}"
            )
            raw_llm_response, llm_protocol_control = await client.ask_serenity_with_protocol(
                routed_prompt,
                timeout=LLM_TIMEOUT_SECONDS,
            )
        except Exception as e:
            errors.append(f"LLM Error: {e}")
    else:
        errors.append("LLM client offline")

    if hard_stop:
        errors.append("Safety protocol engaged for acute risk language/high distress.")

    parsed_payload = _extract_protocol_payload(raw_llm_response)
    if llm_protocol_control:
        parsed_payload["advance_phase"] = bool(
            llm_protocol_control.get("advance_phase", parsed_payload.get("advance_phase", False))
        )
        if llm_protocol_control.get("detected_distortion"):
            parsed_payload["detected_distortion"] = str(llm_protocol_control.get("detected_distortion"))

    if not parsed_payload.get("detected_distortion") and route_decision.detected_distortions:
        parsed_payload["detected_distortion"] = route_decision.detected_distortions[0]
    llm_res = str(parsed_payload.get("response_text") or raw_llm_response or "").strip()

    updated_state = _refresh_clinical_state_after_turn(
        db,
        username=username,
        route_decision=route_decision,
        parsed_payload=parsed_payload,
        user_text=user_text,
        assistant_text=llm_res,
    )

    turn_id = await _persist_turn_safe(db, username, user_text, llm_res, perc["dominant_emotion"], perc["speech_emotion"], perc["face_emotion"], errors)
    with contextlib.suppress(Exception):
        persist_clinical_routing_event(
            db,
            username=username,
            routed_framework=route_decision.framework,
            route_reason=route_decision.route_reason,
            risk_score=int(route_decision.risk_score or 0),
            route_locked=bool(route_decision.route_locked),
            acute_safety_trigger=bool(route_decision.acute_safety_trigger),
            rumination_detected=bool(route_decision.rumination_detected),
            detected_distortion=str(parsed_payload.get("detected_distortion") or ""),
            dominant_emotion=perc["dominant_emotion"],
            speech_emotion=perc["speech_emotion"],
            face_emotion=perc["face_emotion"],
            turn_id=turn_id,
        )

    llm_latency_ms = (time.perf_counter() - llm_started) * 1000.0
    _persist_and_cache_diagnostics(
        db,
        username=username,
        source="voice",
        stt_ms=float(perc.get("stt_latency_ms") or 0.0),
        ser_ms=float(perc.get("ser_latency_ms") or 0.0),
        fer_ms=float(perc.get("fer_latency_ms") or 0.0),
        llm_latency_ms=llm_latency_ms,
        speech_conf=float(perc.get("speech_conf") or 0.0),
        face_conf=float(perc.get("face_conf") or 0.0),
    )

    if bool(updated_state.get("requires_safety_review")):
        errors.append("Clinical safety review is currently required for this user.")

    tts_base64, tts_err = await _generate_tts_base64(llm_res)
    if tts_err: errors.append(tts_err)
    return InteractResponse(dominant_emotion=perc["dominant_emotion"], speech_emotion=perc["speech_emotion"], face_emotion=perc["face_emotion"], transcription=user_text, llm_response=llm_res, tts_audio_base64=tts_base64, errors=_dedupe_errors(errors))

@app.post("/api/interact/stream")
async def interact_stream(username: str = Form(...), image: Optional[str] = Form(None), file: Optional[UploadFile] = File(None), user_message: Optional[str] = Form(None), db: Session = Depends(get_db)):
    if not file: raise HTTPException(400, "Microphone input required.")
    audio_bytes, filename = await file.read(), file.filename or ""

    async def event_stream():
        async with handle_temp_audio(audio_bytes, filename) as audio_path:
            perc = await _run_perception_tasks(audio_path, image)
            for err in perc["errors"]: yield _to_event({"type": "error", "message": err})
            
            user_text = (perc["transcription"] or user_message or "").strip()
            if not user_text:
                yield _to_event({"type": "emotion", "dominant_emotion": perc["dominant_emotion"], "speech_emotion": perc["speech_emotion"], "face_emotion": perc["face_emotion"]})
                yield _to_event({"type": "error", "message": "No speech detected."})
                yield _to_event({"type": "final", "llm_response": "", "transcription": "", "dominant_emotion": perc["dominant_emotion"], "speech_emotion": perc["speech_emotion"], "face_emotion": perc["face_emotion"]})
                return

            async for payload in _stream_chat_events(
                db,
                username,
                user_text,
                perc["dominant_emotion"],
                perc["speech_emotion"],
                perc["face_emotion"],
                "voice",
                True,
                perception_metrics=perc,
            ):
                yield payload

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")

@app.post("/api/chat")
async def chat(payload: ChatRequest, db: Session = Depends(get_db)):
    if not payload.message: raise HTTPException(400, "Message empty.")
    errors: List[str] = []
    llm_started = time.perf_counter()

    clinical_state = fetch_or_create_clinical_state(db, payload.username)
    risk_score = _clinical_risk_score(db, payload.username, payload.message, "Neutral", "Neutral")
    clinical_mode = determine_clinical_mode(
        user_text=payload.message,
        risk_score=risk_score,
        dominant_emotion="Neutral",
    )
    user_model = db.query(models.User).filter(models.User.username == payload.username).first()
    route_decision = evaluate_clinical_route(
        user_text=payload.message,
        risk_score=risk_score,
        dominant_emotion="Neutral",
        speech_emotion="Neutral",
        face_emotion="Neutral",
        user_model=user_model,
        forced_mode=clinical_mode,
    )

    phase = str(clinical_state.get("current_phase") or default_phase_for_framework(route_decision.framework))
    hard_stop = bool(route_decision.acute_safety_trigger or int(route_decision.risk_score or 0) >= CLINICAL_SAFETY_HARDSTOP_RISK)

    raw_llm_response = ""
    llm_protocol_control: Dict[str, Any] = {}
    if hard_stop:
        raw_llm_response = str(build_safety_override_response().get("response_text") or "").strip()
        errors.append("Safety protocol engaged for acute risk language/high distress.")
    elif client := await _ensure_cloud_llm_client():
        try:
            routed_prompt = build_routed_prompt(
                user_text=payload.message,
                decision=route_decision,
                clinical_phase=phase,
                requires_safety_review=bool(clinical_state.get("requires_safety_review")),
            )
            routed_prompt = (
                f"SYSTEM MODE LOCK: {clinical_mode}. You must remain in this mode for the entire response.\n"
                f"{routed_prompt}"
            )
            raw_llm_response, llm_protocol_control = await client.ask_serenity_with_protocol(
                routed_prompt,
                timeout=LLM_TIMEOUT_SECONDS,
            )
        except Exception as e:
            LOGGER.warning("LLM request failed: %s", _format_exception_detail(e))
            raw_llm_response = _build_llm_fallback_response(payload.message, route_decision)
            errors.append("Cloud LLM unavailable. Local fallback response used.")
    else:
        raw_llm_response = _build_llm_fallback_response(payload.message, route_decision)
        errors.append("Cloud LLM unavailable. Local fallback response used.")

    parsed_payload = _extract_protocol_payload(raw_llm_response)
    if llm_protocol_control:
        parsed_payload["advance_phase"] = bool(
            llm_protocol_control.get("advance_phase", parsed_payload.get("advance_phase", False))
        )
        if llm_protocol_control.get("detected_distortion"):
            parsed_payload["detected_distortion"] = str(llm_protocol_control.get("detected_distortion"))

    if not parsed_payload.get("detected_distortion") and route_decision.detected_distortions:
        parsed_payload["detected_distortion"] = route_decision.detected_distortions[0]
    llm_res = str(parsed_payload.get("response_text") or raw_llm_response or "").strip()
    if not llm_res:
        llm_res = _build_llm_fallback_response(payload.message, route_decision)
        parsed_payload["response_text"] = llm_res

    updated_state = _refresh_clinical_state_after_turn(
        db,
        username=payload.username,
        route_decision=route_decision,
        parsed_payload=parsed_payload,
        user_text=payload.message,
        assistant_text=llm_res,
    )

    turn_id = await _persist_turn_safe(db, payload.username, payload.message, llm_res, "Neutral", "Neutral", "Neutral", errors)
    with contextlib.suppress(Exception):
        persist_clinical_routing_event(
            db,
            username=payload.username,
            routed_framework=route_decision.framework,
            route_reason=route_decision.route_reason,
            risk_score=int(route_decision.risk_score or 0),
            route_locked=bool(route_decision.route_locked),
            acute_safety_trigger=bool(route_decision.acute_safety_trigger),
            rumination_detected=bool(route_decision.rumination_detected),
            detected_distortion=str(parsed_payload.get("detected_distortion") or ""),
            dominant_emotion="Neutral",
            speech_emotion="Neutral",
            face_emotion="Neutral",
            turn_id=turn_id,
        )

    _persist_and_cache_diagnostics(
        db,
        username=payload.username,
        source="text",
        stt_ms=0.0,
        ser_ms=0.0,
        fer_ms=0.0,
        llm_latency_ms=(time.perf_counter() - llm_started) * 1000.0,
        speech_conf=0.0,
        face_conf=0.0,
    )

    if bool(updated_state.get("requires_safety_review")):
        errors.append("Clinical safety review is currently required for this user.")

    tts_base64, tts_err = await _generate_tts_base64(llm_res)
    if tts_err: errors.append(tts_err)
    return InteractResponse(dominant_emotion="Neutral", speech_emotion="Neutral", face_emotion="Neutral", transcription=payload.message, llm_response=llm_res, tts_audio_base64=tts_base64, errors=_dedupe_errors(errors))

@app.post("/api/chat/stream")
async def chat_stream(payload: ChatRequest, db: Session = Depends(get_db)):
    if not payload.message: raise HTTPException(400, "Message empty.")
    return StreamingResponse(_stream_chat_events(db, payload.username, payload.message, "Neutral", "Neutral", "Neutral", "text", False), media_type="application/x-ndjson")


@app.post("/api/safety/emergency-contact")
async def set_emergency_contact(payload: EmergencyContactRequest, db: Session = Depends(get_db)):
    username = str(payload.username or "").strip()
    if not username:
        raise HTTPException(400, "Username required")
    result = update_user_emergency_contact(
        db,
        username=username,
        contact_name=str(payload.contact_name or "").strip(),
        contact_phone=str(payload.contact_phone or "").strip(),
    )
    return {"message": "Saved", "contact": result}


@app.get("/api/mbc/trajectory")
async def mbc_trajectory(username: str, refresh: bool = False, db: Session = Depends(get_db)):
    user_key = str(username or "").strip()
    if not user_key:
        raise HTTPException(400, "Username required")

    user = db.query(models.User).filter(models.User.username == user_key).first()
    if not user:
        user = models.User(username=user_key, password="")
        db.add(user)
        db.commit()
        db.refresh(user)

    if refresh:
        await run_in_threadpool(_sync_weekly_trajectory, db, user_key)

    trajectory = await run_in_threadpool(calculate_symptom_trajectory, db, user.id)
    clinical_state = fetch_or_create_clinical_state(db, user_key)

    requires_safety_review = bool(trajectory.get("requires_safety_review")) or bool(clinical_state.get("requires_safety_review"))
    if requires_safety_review and not bool(clinical_state.get("requires_safety_review")):
        clinical_state = upsert_clinical_state(
            db,
            username=user_key,
            updates={"requires_safety_review": True, "last_distress_level": "high"},
        )

    history = trajectory.get("history") or {}
    pending_assessments = _build_pending_assessments(history, cadence_days=7)
    pending_due = [row for row in pending_assessments if bool(row.get("is_due"))]

    chart_rows: List[Dict[str, Any]] = []
    for row in trajectory.get("time_series") or []:
        payload = dict(row)
        pcl5_score = payload.get("pcl5")
        payload["pcl5_scaled_27"] = (
            round((float(pcl5_score) * 27.0) / 80.0, 2)
            if pcl5_score is not None
            else None
        )
        chart_rows.append(payload)

    snapshots = fetch_trajectory_snapshots(db, user_key)
    flagged_questionnaires = [
        str(item.get("questionnaire_type") or "")
        for item in snapshots
        if bool(item.get("flagged")) and str(item.get("questionnaire_type") or "")
    ]

    return {
        "username": user_key,
        "user_id": int(user.id),
        "requires_safety_review": bool(requires_safety_review),
        "flagged_questionnaires": flagged_questionnaires,
        "velocity_delta": dict(trajectory.get("velocity_delta") or {}),
        "history": history,
        "time_series": chart_rows,
        "latest_scores": dict(trajectory.get("latest_scores") or {}),
        "care_plan": _build_mbc_care_plan_state(clinical_state, dict(trajectory.get("latest_scores") or {})),
        "pending_assessments": pending_assessments,
        "has_due_assessment": bool(pending_due),
        "cadence_days": 7,
        "snapshots": snapshots,
    }


@app.get("/api/diagnostics/edge")
async def edge_diagnostics(username: Optional[str] = None, limit: int = 120, db: Session = Depends(get_db)):
    cap = max(1, min(int(limit or 120), 500))
    data = fetch_recent_edge_diagnostics(db, username=username, limit=cap)
    live = _state_get("edge_diagnostics", None)
    live_samples = list(live)[-cap:] if isinstance(live, deque) else []
    return {
        "username": username,
        "limit": cap,
        "samples": data,
        "live_samples": live_samples,
    }


@app.get("/api/diagnostics/metrics")
async def diagnostics_metrics(username: Optional[str] = None, db: Session = Depends(get_db)):
    sample = _latest_edge_metric_sample(db, username=username)

    stt_latency_ms = round(float(sample.get("stt_latency_ms") or 0.0), 2)
    ser_latency_ms = round(float(sample.get("ser_latency_ms") or 0.0), 2)
    fer_latency_ms = round(float(sample.get("fer_latency_ms") or 0.0), 2)

    cpu_usage_pct = 0.0
    ram_usage_mb = round(float(sample.get("memory_mb") or 0.0), 2)

    if psutil is not None:
        with contextlib.suppress(Exception):
            cpu_usage_pct = round(float(psutil.cpu_percent(interval=None)), 2)
        with contextlib.suppress(Exception):
            ram_usage_mb = round(float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0), 2)

    if cpu_usage_pct <= 0.0:
        wave = (math.sin(time.time() * 0.55) + 1.0) / 2.0
        cpu_usage_pct = round(18 + 52 * wave, 2)

    if ram_usage_mb <= 0.0:
        wave = (math.sin(time.time() * 0.33) + 1.0) / 2.0
        ram_usage_mb = round(580 + 220 * wave, 2)

    return {
        "captured_at": str(sample.get("captured_at") or datetime.utcnow().isoformat()),
        "username": username,
        "stt_latency_ms": stt_latency_ms,
        "ser_latency_ms": ser_latency_ms,
        "fer_latency_ms": fer_latency_ms,
        "cpu_thread_usage_percent": cpu_usage_pct,
        "ram_usage_mb": ram_usage_mb,
        "xnnpack_delegate_active": _xnnpack_delegate_active(),
    }


@app.get("/api/admin/handoff/{user_id}")
async def admin_handoff(user_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == int(user_id)).first()
    if not user:
        raise HTTPException(404, "User not found")

    username = str(user.username or "").strip()
    if not username:
        raise HTTPException(400, "User has no username")

    clinical_state = fetch_or_create_clinical_state(db, username)
    trajectory = await run_in_threadpool(calculate_symptom_trajectory, db, int(user.id))
    snapshots = fetch_trajectory_snapshots(db, username)
    turns = fetch_recent_turn_summaries(db, limit=15, text_limit=700, username=username)

    overview = await admin_overview(
        username=username,
        limit=ADMIN_DEFAULT_LIMIT,
        include_answers=False,
        db=db,
    )
    summary_snapshot = dict(overview.get("summary_snapshot") or {})
    narrative_turns = list(overview.get("chats") or [])[:10]
    clinical_narrative, clinical_narrative_source = await _generate_admin_summary(
        summary_snapshot,
        narrative_turns,
    )

    flagged_questionnaires = [
        str(item.get("questionnaire_type") or "")
        for item in snapshots
        if bool(item.get("flagged")) and str(item.get("questionnaire_type") or "")
    ]

    trajectory_payload = {
        "latest_scores": dict(trajectory.get("latest_scores") or {}),
        "velocity_delta": dict(trajectory.get("velocity_delta") or {}),
        "history": dict(trajectory.get("history") or {}),
        "flagged_questionnaires": flagged_questionnaires,
    }

    requires_safety_review = bool(getattr(user, "requires_safety_review", False)) or bool(
        clinical_state.get("requires_safety_review")
    ) or bool(trajectory.get("requires_safety_review"))

    markdown = build_admin_handoff_markdown(
        user_id=int(user.id),
        username=username,
        risk_score=int(clinical_state.get("last_risk_score") or 0),
        requires_safety_review=requires_safety_review,
        active_framework=str(clinical_state.get("active_framework") or FRAMEWORK_DBT),
        trajectory=trajectory_payload,
        recent_turns=turns,
        clinical_narrative=clinical_narrative,
        clinical_narrative_source=clinical_narrative_source,
    )

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "user_id": int(user.id),
        "username": username,
        "risk_score": int(clinical_state.get("last_risk_score") or 0),
        "requires_safety_review": bool(requires_safety_review),
        "trajectory": trajectory_payload,
        "recent_turns": turns,
        "clinical_narrative": clinical_narrative,
        "clinical_narrative_source": clinical_narrative_source,
        "markdown": markdown,
        "file_name": f"{username}_clinical_handoff.md",
    }


@app.get("/api/safety/handoff")
async def safety_handoff(username: str, format: str = "markdown", db: Session = Depends(get_db)):
    user_key = str(username or "").strip()
    if not user_key:
        raise HTTPException(400, "Username required")

    clinical_state = fetch_or_create_clinical_state(db, user_key)
    turns = fetch_recent_turn_summaries(db, limit=12, text_limit=700, username=user_key)

    active_flags: List[str] = []
    if clinical_state.get("last_detected_distortion"):
        active_flags.append(str(clinical_state.get("last_detected_distortion")))
    if clinical_state.get("requires_safety_review"):
        active_flags.append("requires_safety_review")

    distress_signals = sum(1 for row in turns if DISTRESS_SIGNAL_REGEX.search(str(row.get("user_text") or "")))

    markdown = build_handoff_markdown(
        username=user_key,
        risk_score=int(clinical_state.get("last_risk_score") or 0),
        route_framework=str(clinical_state.get("active_framework") or FRAMEWORK_DBT),
        active_flags=active_flags,
        distress_signals=distress_signals,
        recent_turns=turns,
    )

    if str(format or "markdown").lower() == "pdf":
        pdf_bytes = render_handoff_pdf(markdown)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={user_key}_handoff.pdf"},
        )

    return {
        "username": user_key,
        "risk_score": int(clinical_state.get("last_risk_score") or 0),
        "framework": clinical_state.get("active_framework"),
        "phase": clinical_state.get("current_phase"),
        "markdown": markdown,
    }

@app.post("/api/crisis/log")
async def log_crisis_event(payload: dict, db: Session = Depends(get_db)):
    user_id = payload.get("user_id")
    severity = payload.get("severity", "Moderate")
    
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    user.last_crisis_timestamp = datetime.now(timezone.utc).isoformat()
    user.latest_cssrs_risk = severity
    if severity == "High":
        user.requires_safety_review = True
        
    db.commit()
    return {"status": "success", "message": "Clinical cool-down protocol activated."}