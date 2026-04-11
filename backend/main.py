import asyncio
import base64
import contextlib
from datetime import datetime
import importlib
import json
import logging
import os
import re
import threading
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func

try:
    import torch
except ImportError:
    torch = None

try:
    from .audio_core import initialize_audio_runtime, predict_audio_emotion
    from .emotion_core import initialize_face_runtime, analyze_face
    from .database import (
        SessionLocal,
        engine,
        fetch_questionnaire_results,
        fetch_recent_sessions_with_emotions,
        fetch_recent_turn_summaries,
        fetch_recent_turns,
        persist_questionnaire_result,
        persist_turn,
    )
    from .cloud_llm_core import CloudLLMClient, CloudLLMError
    from .questionnaires_data import (
        QUESTIONNAIRE_DEFINITIONS,
        normalize_questionnaire_type,
        questionnaire_clinical_flags,
        questionnaire_templates,
        score_questionnaire,
    )
    from . import models
except ImportError:
    # Allow both package and direct module execution patterns.
    from backend.audio_core import initialize_audio_runtime, predict_audio_emotion
    from backend.emotion_core import initialize_face_runtime, analyze_face
    from backend.database import (
        SessionLocal,
        engine,
        fetch_questionnaire_results,
        fetch_recent_sessions_with_emotions,
        fetch_recent_turn_summaries,
        fetch_recent_turns,
        persist_questionnaire_result,
        persist_turn,
    )
    from backend.cloud_llm_core import CloudLLMClient, CloudLLMError
    from backend.questionnaires_data import (
        QUESTIONNAIRE_DEFINITIONS,
        normalize_questionnaire_type,
        questionnaire_clinical_flags,
        questionnaire_templates,
        score_questionnaire,
    )
    import backend.models as models

try:
    faster_whisper_module = importlib.import_module("faster_whisper")
    WhisperModel = getattr(faster_whisper_module, "WhisperModel", None)
except ImportError:
    WhisperModel = None

try:
    import whisper as openai_whisper
except ImportError:
    openai_whisper = None

try:
    import edge_tts
except ImportError:
    edge_tts = None

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# --- Pydantic Models ---
class AuthRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    message: str
    username: str


class HealthResponse(BaseModel):
    status: str
    rag_loaded: bool


class InteractResponse(BaseModel):
    dominant_emotion: str
    speech_emotion: str
    face_emotion: str
    transcription: str
    llm_response: str
    tts_audio_base64: Optional[str] = None
    tts_audio_segments_base64: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    username: str
    message: str


class QuestionnaireSubmitRequest(BaseModel):
    username: str
    questionnaire_type: str
    answers: List[int] = Field(default_factory=list)
    submitted_at: Optional[str] = None

EMOTION_LABELS = ["angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_ALIAS = {
    "surprised": "surprise",
    "fearful": "fear",
    "no face": "neutral",
}
NEGATIVE_EMOTIONS = {"angry", "disgust", "fear", "sad"}
POSITIVE_EMOTIONS = {"happy", "calm", "surprise"}
SENTENCE_BOUNDARY_REGEX = re.compile(r"(?<=[.!?])\s+")
SENTENCE_TERMINATOR_REGEX = re.compile(r"[.!?][\"')\]]*$")
CAMEL_BOUNDARY_REGEX = re.compile(r"([a-z])([A-Z])")
WHITESPACE_REGEX = re.compile(r"\s+")
HORIZONTAL_SPACE_REGEX = re.compile(r"[ \t]+")
SPACE_BEFORE_PUNCT_REGEX = re.compile(r"\s+([,.;:!?])")
PROMPT_LEAK_PATTERNS = [
    re.compile(r"Reflecting feelings, then asking ONE follow-up question\.?", flags=re.IGNORECASE),
    re.compile(r"\bReflecting\s*feelings?\b.*$", flags=re.IGNORECASE),
    re.compile(r"\bthen asking(?:\s+one)?\s+follow-?up\s+question\b.*$", flags=re.IGNORECASE),
    re.compile(r"\bdialogue while reflecting on personal responses\.?", flags=re.IGNORECASE),
    re.compile(
        r"\bserenity\s*is\s*an\s*empathetic\s*therapist\b.*$",
        flags=re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\bthis\s*dialogue\s*should\s*only\s*be\s*used\s*as\s*a\s*general\s*conversation\s*starter\b.*$",
        flags=re.IGNORECASE | re.DOTALL,
    ),
]


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() == "true"


EDGE_OPTIMIZED_MODE = _env_bool("SERENITY_EDGE_OPTIMIZED_MODE", True)
LAZY_RUNTIME_INIT = _env_bool("SERENITY_LAZY_RUNTIME_INIT", EDGE_OPTIMIZED_MODE)
WHISPER_PRELOAD_ENABLED = _env_bool("SERENITY_WHISPER_PRELOAD_ENABLED", not EDGE_OPTIMIZED_MODE)

WHISPER_MODEL_SIZE = os.getenv("SERENITY_WHISPER_MODEL_SIZE", "tiny").strip()
DEFAULT_WHISPER_CPU_THREADS = (
    max(1, (os.cpu_count() or 4) // 2)
    if EDGE_OPTIMIZED_MODE
    else max(1, (os.cpu_count() or 4) - 1)
)
WHISPER_CPU_THREADS = int(os.getenv("SERENITY_WHISPER_CPU_THREADS", str(DEFAULT_WHISPER_CPU_THREADS)))
WHISPER_COMPUTE_TYPE_CPU = os.getenv("SERENITY_WHISPER_COMPUTE_TYPE_CPU", "int8").strip()
REQUIRE_STT_BACKEND = _env_bool("SERENITY_REQUIRE_STT_BACKEND", False)
WHISPER_TIMEOUT_SECONDS = int(os.getenv("SERENITY_WHISPER_TIMEOUT_SECONDS", "40"))
EMOTION_TIMEOUT_SECONDS = int(os.getenv("SERENITY_EMOTION_TIMEOUT_SECONDS", "20"))
LLM_TIMEOUT_SECONDS = int(os.getenv("SERENITY_LLM_TIMEOUT_SECONDS", "90" if EDGE_OPTIMIZED_MODE else "180"))
TTS_TIMEOUT_SECONDS = 30
TTS_ENABLED = _env_bool("SERENITY_TTS_ENABLED", True)
TTS_FAILURE_THRESHOLD = int(os.getenv("SERENITY_TTS_FAILURE_THRESHOLD", "3"))
TTS_COOLDOWN_SECONDS = int(os.getenv("SERENITY_TTS_COOLDOWN_SECONDS", "180"))
TTS_VOICE = os.getenv("SERENITY_TTS_VOICE", "en-GB-RyanNeural").strip()
TTS_RATE = os.getenv("SERENITY_TTS_RATE", "+0%").strip()
TTS_PITCH = os.getenv("SERENITY_TTS_PITCH", "+0Hz").strip()
TTS_STREAMING_ENABLED = _env_bool("SERENITY_TTS_STREAMING_ENABLED", True)
TTS_WARMUP_ENABLED = _env_bool("SERENITY_TTS_WARMUP_ENABLED", not EDGE_OPTIMIZED_MODE)
STREAM_TOKEN_DELTA = _env_bool("SERENITY_STREAM_TOKEN_DELTA", True)
STREAM_TTS_SENTENCE_AUDIO = _env_bool("SERENITY_STREAM_TTS_SENTENCE_AUDIO", True)
STREAM_TTS_FINAL_TEXT_ONLY = _env_bool("SERENITY_STREAM_TTS_FINAL_TEXT_ONLY", False)
TRUST_CLOUD_POLISHED_RESPONSE = _env_bool("SERENITY_TRUST_CLOUD_POLISHED_RESPONSE", True)
CLOUD_LLM_LAZY_INIT = _env_bool("SERENITY_CLOUD_LLM_LAZY_INIT", EDGE_OPTIMIZED_MODE)
CLOUD_LLM_WARMUP_ENABLED = _env_bool("SERENITY_CLOUD_LLM_WARMUP_ENABLED", not EDGE_OPTIMIZED_MODE)
CLOUD_LLM_WARMUP_TEXT = os.getenv("SERENITY_CLOUD_LLM_WARMUP_TEXT", "Hello").strip() or "Hello"
CLOUD_LLM_WARMUP_TIMEOUT_SECONDS = int(os.getenv("SERENITY_CLOUD_LLM_WARMUP_TIMEOUT_SECONDS", "45"))
ADMIN_DEFAULT_LIMIT = max(20, int(os.getenv("SERENITY_ADMIN_DEFAULT_LIMIT", "300")))
ADMIN_MAX_LIMIT = max(100, int(os.getenv("SERENITY_ADMIN_MAX_LIMIT", "5000")))
ADMIN_CHAT_TEXT_LIMIT = max(64, int(os.getenv("SERENITY_ADMIN_CHAT_TEXT_LIMIT", "420")))
ADMIN_SESSION_TEXT_LIMIT = max(64, int(os.getenv("SERENITY_ADMIN_SESSION_TEXT_LIMIT", "420")))

_stream_queue_wait_default = "0.015" if EDGE_OPTIMIZED_MODE else "0.01"
try:
    STREAM_QUEUE_WAIT_SECONDS = max(
        0.001,
        float(os.getenv("SERENITY_STREAM_QUEUE_WAIT_SECONDS", _stream_queue_wait_default).strip()),
    )
except ValueError:
    STREAM_QUEUE_WAIT_SECONDS = 0.015 if EDGE_OPTIMIZED_MODE else 0.01

llm_generation_semaphore = asyncio.Semaphore(1)
tts_consecutive_failures = 0
tts_disabled_until_epoch = 0.0
whisper_init_lock = threading.Lock()
cloud_llm_init_lock = threading.Lock()


def _state_get(key: str, default=None):
    return getattr(app.state, key, default)


def _state_set(key: str, value) -> None:
    setattr(app.state, key, value)


def _ensure_cloud_llm_client_sync() -> Tuple[Optional[CloudLLMClient], Optional[str], bool]:
    cloud_llm_client = _state_get("cloud_llm_client")
    if cloud_llm_client is not None:
        return cloud_llm_client, None, False

    with cloud_llm_init_lock:
        cloud_llm_client = _state_get("cloud_llm_client")
        if cloud_llm_client is not None:
            return cloud_llm_client, None, False

        try:
            cloud_llm_client = CloudLLMClient()
            _state_set("cloud_llm_client", cloud_llm_client)
            _state_set("rag_init_error", None)
            LOGGER.info("Cloud LLM API client initialized on-demand")
            return cloud_llm_client, None, True
        except CloudLLMError as exc:
            error_text = str(exc)
            _state_set("rag_init_error", error_text)
            LOGGER.warning("Cloud LLM lazy initialization failed: %s", exc)
            return None, error_text, False


async def _ensure_cloud_llm_client() -> Tuple[Optional[CloudLLMClient], Optional[str]]:
    cloud_llm_client = _state_get("cloud_llm_client")
    if cloud_llm_client is not None:
        return cloud_llm_client, None

    client, error_text, created = await run_in_threadpool(_ensure_cloud_llm_client_sync)
    if client is not None and created and CLOUD_LLM_WARMUP_ENABLED:
        warmup_task = _state_get("cloud_llm_warmup_task")
        if warmup_task is None or warmup_task.done():
            _state_set("cloud_llm_warmup_task", asyncio.create_task(_warmup_cloud_llm_once()))

    return client, error_text


def _parse_optional_datetime(value: Optional[str]) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None

    normalized = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)

    return parsed


def _normalize_emotion_label(emotion: str) -> str:
    value = str(emotion or "neutral").strip().lower()
    if not value:
        return "neutral"
    return EMOTION_ALIAS.get(value, value)


def _build_admin_summary(
    total_users: int,
    emotion_counts: Dict[str, int],
    latest_scores_by_type: Dict[str, List[int]],
    flagged_user_count: int,
) -> str:
    if total_users <= 0:
        return "No user records found yet. Ask users to complete chats or questionnaires to generate insights."

    total_emotion_samples = sum(int(count) for count in emotion_counts.values())
    negative_samples = sum(
        int(count)
        for emotion, count in emotion_counts.items()
        if _normalize_emotion_label(emotion) in NEGATIVE_EMOTIONS
    )
    positive_samples = sum(
        int(count)
        for emotion, count in emotion_counts.items()
        if _normalize_emotion_label(emotion) in POSITIVE_EMOTIONS
    )

    summary_parts: List[str] = []
    if total_emotion_samples > 0:
        negative_ratio = round((negative_samples / total_emotion_samples) * 100.0, 1)
        positive_ratio = round((positive_samples / total_emotion_samples) * 100.0, 1)
        summary_parts.append(
            f"Emotion logs show {negative_ratio}% negative-leaning and {positive_ratio}% positive-leaning observations."
        )

    if latest_scores_by_type:
        score_bits: List[str] = []
        for questionnaire_type in ("PHQ-9", "GAD-7", "PCL-5"):
            values = latest_scores_by_type.get(questionnaire_type) or []
            if not values:
                continue
            average_score = round(sum(values) / len(values), 1)
            score_bits.append(f"{questionnaire_type} avg {average_score}")
        if score_bits:
            summary_parts.append("Latest screening averages: " + ", ".join(score_bits) + ".")

    if flagged_user_count > 0:
        summary_parts.append(
            f"{flagged_user_count} user(s) currently meet elevated screening thresholds and may benefit from additional follow-up support."
        )
    else:
        summary_parts.append("No users currently exceed elevated screening thresholds based on available questionnaire entries.")

    return " ".join(summary_parts).strip()


def _normalize_questionnaire_answers(questionnaire_type: str, answers: List[int]) -> List[int]:
    definition = QUESTIONNAIRE_DEFINITIONS.get(questionnaire_type, {})
    question_count = len(definition.get("questions", []))
    max_per_item = 4 if questionnaire_type == "PCL-5" else 3

    normalized: List[int] = []
    for index in range(question_count):
        raw = answers[index] if index < len(answers) else 0
        try:
            value = int(raw)
        except Exception:
            value = 0
        value = max(0, min(max_per_item, value))
        normalized.append(value)

    return normalized


def _ensure_whisper_runtime_sync() -> Optional[str]:
    whisper_model = _state_get("whisper_model")
    if whisper_model is not None:
        return None

    if _state_get("whisper_backend") == "none":
        message = str(_state_get("whisper_init_error") or "No STT backend available.")
        if REQUIRE_STT_BACKEND:
            raise RuntimeError(message)
        return message

    with whisper_init_lock:
        whisper_model = _state_get("whisper_model")
        if whisper_model is not None:
            return None

        if _state_get("whisper_backend") == "none":
            message = str(_state_get("whisper_init_error") or "No STT backend available.")
            if REQUIRE_STT_BACKEND:
                raise RuntimeError(message)
            return message

        whisper_device = _state_get("whisper_device_in_use", "cpu")
        try:
            if WhisperModel is not None:
                whisper_compute_type = "float16" if whisper_device == "cuda" else WHISPER_COMPUTE_TYPE_CPU
                loaded_model = WhisperModel(
                    WHISPER_MODEL_SIZE,
                    device=whisper_device,
                    compute_type=whisper_compute_type,
                    cpu_threads=WHISPER_CPU_THREADS,
                    num_workers=1,
                )
                _state_set("whisper_backend", "faster-whisper")
                _state_set("whisper_model", loaded_model)
                _state_set("whisper_init_error", None)
                LOGGER.info(
                    "Loaded faster-whisper '%s' on %s (%s)",
                    WHISPER_MODEL_SIZE,
                    whisper_device.upper(),
                    whisper_compute_type,
                )
                return None

            if openai_whisper is not None:
                loaded_model = openai_whisper.load_model(WHISPER_MODEL_SIZE, device=whisper_device)
                _state_set("whisper_backend", "openai-whisper")
                _state_set("whisper_model", loaded_model)
                _state_set("whisper_init_error", None)
                LOGGER.warning(
                    "faster-whisper not installed; loaded openai-whisper '%s' on %s",
                    WHISPER_MODEL_SIZE,
                    whisper_device.upper(),
                )
                return None

            message = "No STT backend available. Install faster-whisper or openai-whisper."
            _state_set("whisper_backend", "none")
            _state_set("whisper_model", None)
            _state_set("whisper_init_error", message)
            if REQUIRE_STT_BACKEND:
                raise RuntimeError(message)
            LOGGER.warning("%s Voice transcription endpoints will return a runtime warning.", message)
            return message
        except Exception as exc:
            message = f"Failed to initialize STT backend: {exc}"
            _state_set("whisper_backend", "none")
            _state_set("whisper_model", None)
            _state_set("whisper_init_error", message)
            if REQUIRE_STT_BACKEND:
                raise RuntimeError(message) from exc
            LOGGER.warning("%s", message)
            return message


@contextlib.asynccontextmanager
async def serenity_lifespan(fastapi_app: FastAPI):
    LOGGER.info("Preloading SERENITY edge runtime into FastAPI app.state")

    fastapi_app.state.rag_init_error = None
    fastapi_app.state.cloud_llm_client = None
    fastapi_app.state.cloud_llm_warmup_task = None
    fastapi_app.state.tts_warmed_up = False
    fastapi_app.state.tts_warmup_task = None
    fastapi_app.state.whisper_model = None
    fastapi_app.state.whisper_backend = "uninitialized"
    fastapi_app.state.whisper_init_error = None
    fastapi_app.state.face_runtime = None
    fastapi_app.state.speech_runtime = None

    # Edge profile can defer heavy runtimes until first real request.
    if not LAZY_RUNTIME_INIT:
        fastapi_app.state.face_runtime = await run_in_threadpool(initialize_face_runtime)
        fastapi_app.state.speech_runtime = await run_in_threadpool(initialize_audio_runtime)
    else:
        LOGGER.info("Lazy runtime init enabled: FER/SER models will load on first use.")

    # STT backend can also be deferred for edge memory savings.
    whisper_device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    fastapi_app.state.whisper_device_in_use = whisper_device

    if WHISPER_PRELOAD_ENABLED:
        await run_in_threadpool(_ensure_whisper_runtime_sync)
    else:
        LOGGER.info("Whisper preload disabled: STT model will load on first transcription request.")

    if CLOUD_LLM_LAZY_INIT:
        LOGGER.info("Cloud LLM lazy init enabled: client will initialize on first chat request.")
    else:
        try:
            fastapi_app.state.cloud_llm_client = CloudLLMClient()
            LOGGER.info("Cloud LLM API client initialized successfully")
        except CloudLLMError as exc:
            fastapi_app.state.rag_init_error = str(exc)
            LOGGER.warning("Cloud LLM client initialization failed: %s", exc)

    if CLOUD_LLM_WARMUP_ENABLED and fastapi_app.state.cloud_llm_client is not None:
        fastapi_app.state.cloud_llm_warmup_task = asyncio.create_task(_warmup_cloud_llm_once())

    if TTS_WARMUP_ENABLED and TTS_ENABLED and edge_tts is not None:
        fastapi_app.state.tts_warmup_task = asyncio.create_task(_warmup_edge_tts_once())

    try:
        yield
    except asyncio.CancelledError:
        # Uvicorn shutdown on Windows can cancel the lifespan receive queue during Ctrl+C.
        # Treat this as a normal shutdown path to avoid noisy traceback logs.
        LOGGER.info("Lifespan cancellation received during shutdown.")
    finally:
        warmup_task = getattr(fastapi_app.state, "tts_warmup_task", None)
        if warmup_task is not None and not warmup_task.done():
            warmup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await warmup_task

        cloud_warmup_task = getattr(fastapi_app.state, "cloud_llm_warmup_task", None)
        if cloud_warmup_task is not None and not cloud_warmup_task.done():
            cloud_warmup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cloud_warmup_task

        cloud_client = getattr(fastapi_app.state, "cloud_llm_client", None)
        if cloud_client is not None:
            with contextlib.suppress(Exception):
                cloud_client.close()


app = FastAPI(title="SERENITY API", version="1.0.0", lifespan=serenity_lifespan)

# Create database tables on startup
models.Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- AUTH ENDPOINTS ---
@app.post("/register", response_model=AuthResponse)
async def register(payload: AuthRequest) -> AuthResponse:
    """Register a new user account."""
    if not payload.username or not payload.password:
        raise HTTPException(status_code=400, detail="Username and password required")
    
    db = SessionLocal()
    try:
        # Check if user exists
        existing_user = db.query(models.User).filter(
            models.User.username == payload.username
        ).first()
        
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")
        
        # Create new user
        new_user = models.User(username=payload.username, password=payload.password)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        LOGGER.info("New user registered: %s", payload.username)
        return AuthResponse(message="Registration successful", username=new_user.username)
    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        LOGGER.exception("Registration failed: %s", exc)
        raise HTTPException(status_code=500, detail="Registration failed") from exc
    finally:
        db.close()


@app.post("/login", response_model=AuthResponse)
async def login(payload: AuthRequest) -> AuthResponse:
    """Login with username and password."""
    if not payload.username or not payload.password:
        raise HTTPException(status_code=400, detail="Username and password required")
    
    db = SessionLocal()
    try:
        user = db.query(models.User).filter(
            models.User.username == payload.username,
            models.User.password == payload.password
        ).first()
        
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        LOGGER.info("User logged in: %s", payload.username)
        return AuthResponse(message="Login successful", username=user.username)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Login failed: %s", exc)
        raise HTTPException(status_code=500, detail="Login failed") from exc
    finally:
        db.close()


@app.get("/api/questionnaires/templates")
async def get_questionnaire_templates(types: Optional[str] = None) -> Dict[str, object]:
    requested_types: Optional[List[str]] = None
    if types:
        requested_types = [item.strip() for item in types.split(",") if item.strip()]

    templates = questionnaire_templates(requested_types)
    return {
        "available_types": list(QUESTIONNAIRE_DEFINITIONS.keys()),
        "questionnaires": templates,
    }


@app.post("/api/questionnaires/submit")
async def submit_questionnaire(payload: QuestionnaireSubmitRequest) -> Dict[str, object]:
    username = str(payload.username or "").strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username is required")

    questionnaire_type = normalize_questionnaire_type(payload.questionnaire_type)
    if questionnaire_type is None:
        raise HTTPException(status_code=400, detail="Invalid questionnaire type")

    normalized_answers = _normalize_questionnaire_answers(questionnaire_type, payload.answers)
    total_score, severity = score_questionnaire(questionnaire_type, normalized_answers)
    submitted_dt = _parse_optional_datetime(payload.submitted_at)

    db = SessionLocal()
    try:
        record = persist_questionnaire_result(
            db,
            username=username,
            questionnaire_type=questionnaire_type,
            answers=normalized_answers,
            total_score=total_score,
            severity=severity,
            created_at=submitted_dt,
        )
    except Exception as exc:
        LOGGER.exception("Failed to persist questionnaire result: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to save questionnaire result") from exc
    finally:
        db.close()

    return {
        "message": "Questionnaire saved",
        "result": {
            "id": record.id,
            "username": username,
            "questionnaire_type": questionnaire_type,
            "answers": normalized_answers,
            "total_score": total_score,
            "severity": severity,
            "created_at": record.created_at.isoformat() if record.created_at else None,
        },
    }


@app.get("/api/questionnaires/history")
async def questionnaire_history(username: str, limit: int = 30) -> Dict[str, object]:
    cleaned_username = str(username or "").strip()
    if not cleaned_username:
        raise HTTPException(status_code=400, detail="Username is required")

    safe_limit = max(1, min(200, int(limit or 30)))

    db = SessionLocal()
    try:
        rows = fetch_questionnaire_results(db, username=cleaned_username, limit=safe_limit)
    except Exception as exc:
        LOGGER.exception("Failed to fetch questionnaire history: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to fetch questionnaire history") from exc
    finally:
        db.close()

    return {
        "username": cleaned_username,
        "results": rows,
    }


@app.get("/api/admin/overview")
async def admin_overview(limit: Optional[int] = None, include_answers: bool = False) -> Dict[str, object]:

    db = SessionLocal()
    try:
        total_users = int(db.query(func.count(models.User.id)).scalar() or 0)
        total_conversation_turns = int(db.query(func.count(models.ConversationTurn.id)).scalar() or 0)
        total_sessions = int(db.query(func.count(models.Session.id)).scalar() or 0)
        total_emotion_events = int(db.query(func.count(models.Emotion.id)).scalar() or 0)
        total_questionnaire_entries = int(db.query(func.count(models.QuestionnaireResult.id)).scalar() or 0)

        if limit is None:
            safe_limit = min(ADMIN_DEFAULT_LIMIT, ADMIN_MAX_LIMIT)
            session_limit = min(ADMIN_DEFAULT_LIMIT, ADMIN_MAX_LIMIT)
            questionnaire_limit = min(max(100, ADMIN_DEFAULT_LIMIT * 3), ADMIN_MAX_LIMIT * 3)
        elif int(limit) <= 0:
            safe_limit = min(max(1, total_conversation_turns), ADMIN_MAX_LIMIT)
            session_limit = min(max(1, total_sessions), ADMIN_MAX_LIMIT)
            questionnaire_limit = min(max(1, total_questionnaire_entries), ADMIN_MAX_LIMIT * 3)
        else:
            safe_limit = max(20, min(ADMIN_MAX_LIMIT, int(limit)))
            session_limit = max(20, min(ADMIN_MAX_LIMIT, safe_limit))
            questionnaire_limit = max(100, min(ADMIN_MAX_LIMIT * 3, safe_limit * 3))

        conversation_turns = fetch_recent_turn_summaries(
            db,
            limit=safe_limit,
            text_limit=ADMIN_CHAT_TEXT_LIMIT,
        )
        sessions = fetch_recent_sessions_with_emotions(
            db,
            limit=session_limit,
            conversation_limit=ADMIN_SESSION_TEXT_LIMIT,
        )
        questionnaire_results = fetch_questionnaire_results(
            db,
            limit=questionnaire_limit,
            include_answers=include_answers,
        )
    except Exception as exc:
        LOGGER.exception("Failed to build admin overview: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to build admin overview") from exc
    finally:
        db.close()

    emotion_counts: Dict[str, int] = {}
    for turn in conversation_turns:
        label = _normalize_emotion_label(str(turn.get("dominant_emotion", "neutral")))
        emotion_counts[label] = emotion_counts.get(label, 0) + 1

    for session in sessions:
        for emotion_item in session.get("emotions", []):
            label = _normalize_emotion_label(str(emotion_item.get("emotion", "neutral")))
            emotion_counts[label] = emotion_counts.get(label, 0) + 1

    latest_by_user: Dict[str, Dict[str, Dict[str, Any]]] = {}
    latest_scores_by_type: Dict[str, List[int]] = {}
    flagged_users: List[Dict[str, Any]] = []

    for item in questionnaire_results:
        username = str(item.get("username") or "unknown")
        questionnaire_type = str(item.get("questionnaire_type") or "")
        if not questionnaire_type:
            continue

        user_bucket = latest_by_user.setdefault(username, {})
        if questionnaire_type in user_bucket:
            continue
        user_bucket[questionnaire_type] = item

    for username, user_latest in latest_by_user.items():
        score_map: Dict[str, int] = {}
        for questionnaire_type, item in user_latest.items():
            score = int(item.get("total_score") or 0)
            score_map[questionnaire_type] = score
            latest_scores_by_type.setdefault(questionnaire_type, []).append(score)

        flags = questionnaire_clinical_flags(score_map)
        if any(flags.values()):
            flagged_users.append(
                {
                    "username": username,
                    "scores": score_map,
                    "flags": flags,
                }
            )

    top_emotions = [
        {"emotion": emotion, "count": count}
        for emotion, count in sorted(emotion_counts.items(), key=lambda pair: pair[1], reverse=True)
    ]

    summary = _build_admin_summary(
        total_users=total_users,
        emotion_counts=emotion_counts,
        latest_scores_by_type=latest_scores_by_type,
        flagged_user_count=len(flagged_users),
    )

    metrics = [
        {
            "id": "users",
            "label": "Registered Users",
            "value": total_users,
            "description": "Accounts in local database",
        },
        {
            "id": "turns",
            "label": "Conversation Turns",
            "value": total_conversation_turns,
            "description": "Stored user-assistant exchanges",
        },
        {
            "id": "sessions",
            "label": "Sessions",
            "value": total_sessions,
            "description": "Legacy session records",
        },
        {
            "id": "emotion_events",
            "label": "Emotion Events",
            "value": total_emotion_events,
            "description": "Session-level emotion rows",
        },
        {
            "id": "questionnaires",
            "label": "Questionnaire Entries",
            "value": total_questionnaire_entries,
            "description": "Saved PHQ-9, GAD-7, and PCL-5 submissions",
        },
        {
            "id": "flagged_users",
            "label": "Elevated Screens",
            "value": len(flagged_users),
            "description": "Users with latest scores above screening thresholds",
        },
    ]

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "summary": summary,
        "metrics": metrics,
        "top_emotions": top_emotions,
        "chats": conversation_turns,
        "sessions": sessions,
        "questionnaire_results": questionnaire_results,
        "latest_questionnaires_by_user": latest_by_user,
        "flagged_users": flagged_users,
    }


# --- HEALTH & EMOTION ENDPOINTS ---


def _normalize_prediction(prediction: object) -> tuple[str, float, Optional[str]]:
    if isinstance(prediction, dict):
        emotion = str(prediction.get("emotion", "Neutral"))
        confidence = float(prediction.get("confidence", 0.0))
        error = prediction.get("error")
        return emotion, confidence, str(error) if error else None

    if isinstance(prediction, tuple) and len(prediction) >= 2:
        return str(prediction[0]), float(prediction[1]), None

    return "Neutral", 0.0, "Invalid prediction payload"


def _to_probability_vector(emotion: str, confidence: float) -> Dict[str, float]:
    normalized_emotion = EMOTION_ALIAS.get(str(emotion).strip().lower(), str(emotion).strip().lower())
    if normalized_emotion not in EMOTION_LABELS:
        normalized_emotion = "neutral"

    confidence = float(confidence)
    if confidence > 1.0:
        confidence = confidence / 100.0
    confidence = max(0.0, min(confidence, 1.0))

    background = (1.0 - confidence) / (len(EMOTION_LABELS) - 1)
    probs = {label: background for label in EMOTION_LABELS}
    probs[normalized_emotion] = confidence
    return probs


def _fuse_probability_vectors(
    speech_probs: Dict[str, float],
    face_probs: Dict[str, float],
) -> Dict[str, float]:
    return {
        label: (speech_probs[label] + face_probs[label]) / 2.0
        for label in EMOTION_LABELS
    }


def _dominant_from_probabilities(probabilities: Dict[str, float]) -> str:
    if not probabilities:
        return "Neutral"
    dominant = max(probabilities.items(), key=lambda item: item[1])[0]
    return dominant.title()


def _serialize_turns(turns: list) -> List[Dict[str, str]]:
    history = []
    for turn in turns:
        history.append(
            {
                "user_text": str(getattr(turn, "user_text", "")),
                "assistant_text": str(getattr(turn, "assistant_text", "")),
                "emotion": str(getattr(turn, "dominant_emotion", "Neutral")),
            }
        )
    return history

def _to_ndjson_event(payload: Dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _extract_text_from_cloud_payload(payload: object) -> str:
    if isinstance(payload, str):
        return payload.strip()

    if isinstance(payload, dict):
        for key in (
            "response",
            "reply",
            "answer",
            "text",
            "llm_response",
            "message",
            "delta",
            "token",
            "content",
        ):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value

        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                delta = first.get("delta")
                if isinstance(delta, dict):
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        return content

                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str) and content:
                        return content

    return ""


def _extract_text_from_cloud_blob(blob: str) -> str:
    text = str(blob or "").strip()
    if not text:
        return ""

    try:
        parsed = json.loads(text)
    except ValueError:
        return ""

    return _extract_text_from_cloud_payload(parsed)


def _strip_starred_segments(text: str, preserve_edges: bool = False) -> str:
    source = str(text or "")
    if not source:
        return ""

    star_index = source.find("*")
    cleaned = source if star_index == -1 else source[:star_index]
    cleaned = HORIZONTAL_SPACE_REGEX.sub(" ", cleaned)
    cleaned = SPACE_BEFORE_PUNCT_REGEX.sub(r"\1", cleaned)
    return cleaned if preserve_edges else cleaned.strip()


def _hard_clean_text(text: str, preserve_edges: bool = False) -> str:
    cleaned = _strip_starred_segments(str(text or ""), preserve_edges=True)
    if not cleaned:
        return ""

    # Match the same hard-clean style used by the standalone stream client.
    cleaned = CAMEL_BOUNDARY_REGEX.sub(r"\1 \2", cleaned)
    cleaned = WHITESPACE_REGEX.sub(" ", cleaned)
    return cleaned if preserve_edges else cleaned.strip()


def _clean_stream_token(token: str, reached_first_asterisk: bool) -> tuple[str, bool]:
    source = str(token or "")
    if not source:
        return "", reached_first_asterisk

    if reached_first_asterisk:
        return "", True

    star_index = source.find("*")
    if star_index != -1:
        source = source[:star_index]
        reached_first_asterisk = True

    cleaned = source
    cleaned = HORIZONTAL_SPACE_REGEX.sub(" ", cleaned)
    cleaned = SPACE_BEFORE_PUNCT_REGEX.sub(r"\1", cleaned)
    return cleaned, reached_first_asterisk


def _normalize_polished_cloud_text(text: str, preserve_edges: bool = False) -> str:
    normalized = str(text or "")
    if not normalized:
        return ""

    normalized = normalized.replace("\r", "")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    return normalized if preserve_edges else normalized.strip()


def _normalize_cloud_stream_piece(piece: str) -> str:
    text = str(piece or "")
    if not text.strip():
        return ""

    if TRUST_CLOUD_POLISHED_RESPONSE:
        extracted = _extract_text_from_cloud_blob(text)
        if extracted:
            text = extracted
        return _normalize_polished_cloud_text(text, preserve_edges=True)

    # If a full JSON object arrives as a single chunk, unwrap it to assistant text.
    extracted = _extract_text_from_cloud_blob(text)
    if extracted:
        return _hard_clean_text(extracted, preserve_edges=True)

    response_match = re.search(
        r'"response"\s*:\s*"((?:\\.|[^"\\])*)"',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if response_match:
        escaped_value = response_match.group(1)
        try:
            text = json.loads(f'"{escaped_value}"')
        except ValueError:
            text = escaped_value.replace("\\n", "\n").replace('\\"', '"')

    text = re.sub(
        r"Reflecting feelings, then asking ONE follow-up question\.?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bReflecting\s*feelings?\b.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthen asking(?:\s+one)?\s+follow-?up\s+question\b.*$", "", text, flags=re.IGNORECASE)
    text = re.split(r"\bUser:|\bAssistant:", text, maxsplit=1, flags=re.IGNORECASE)[0]
    return _hard_clean_text(text, preserve_edges=True)


def _strip_prompt_leakage(text: str) -> str:
    cleaned = str(text or "")
    if not cleaned:
        return ""

    for pattern in PROMPT_LEAK_PATTERNS:
        cleaned = pattern.sub("", cleaned)

    cleaned = re.split(r"\bUser:|\bAssistant:", cleaned, maxsplit=1, flags=re.IGNORECASE)[0]
    return cleaned


def _remove_disallowed_tail(text: str) -> str:
    """Remove fixed policy/disclaimer tails that should not reach UI or TTS."""
    return _strip_prompt_leakage(text)


def _collapse_repeated_ngrams(text: str) -> str:
    cleaned = str(text or "")
    if not cleaned:
        return ""

    for n in (10, 8, 6, 5, 4):
        previous = None
        while cleaned != previous:
            previous = cleaned
            cleaned = re.sub(
                rf"(?i)\b((?:[\w']+\W+){{{n - 1}}}[\w']+)(?:\W+\1\b)+",
                r"\1",
                cleaned,
            )

    return cleaned


def _dedupe_sentences(text: str, max_sentences: Optional[int] = None) -> str:
    normalized = str(text or "")
    if not normalized:
        return ""

    normalized = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", normalized)
    normalized = re.sub(r"([,;:])([A-Za-z])", r"\1 \2", normalized)
    normalized = WHITESPACE_REGEX.sub(" ", normalized).strip()
    if not normalized:
        return ""

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
    if not sentences:
        return normalized

    unique_sentences: List[str] = []
    seen_keys = set()
    for sentence in sentences:
        key = re.sub(r"\W+", "", sentence.lower())
        if key and key in seen_keys:
            continue

        if unique_sentences:
            previous = unique_sentences[-1]
            previous_key = re.sub(r"\W+", "", previous.lower())
            if previous_key and key and (key in previous_key or previous_key in key):
                continue

            sentence_tokens = set(re.findall(r"[\w']+", sentence.lower()))
            previous_tokens = set(re.findall(r"[\w']+", previous.lower()))
            union = sentence_tokens.union(previous_tokens)
            if union:
                overlap_ratio = len(sentence_tokens.intersection(previous_tokens)) / len(union)
                if overlap_ratio > 0.82:
                    continue

        if key:
            seen_keys.add(key)
        unique_sentences.append(sentence)

        if max_sentences is not None and len(unique_sentences) >= max_sentences:
            break

    return " ".join(unique_sentences).strip() if unique_sentences else normalized


def _sanitize_cloud_llm_response(text: str, max_words: int = 60) -> str:
    if TRUST_CLOUD_POLISHED_RESPONSE:
        cleaned = _extract_text_from_cloud_blob(str(text or "")) or str(text or "")
        cleaned = _normalize_polished_cloud_text(cleaned)
        cleaned = _strip_starred_segments(cleaned, preserve_edges=True)
        cleaned = _remove_disallowed_tail(cleaned)
        cleaned = _strip_prompt_leakage(cleaned)
        cleaned = _collapse_repeated_ngrams(cleaned)
        cleaned = _dedupe_sentences(cleaned, max_sentences=3)
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        cleaned = re.sub(r"([.!?])(?:\s*[.!?])+", r"\1", cleaned)

        words = cleaned.split()
        if len(words) > max_words:
            cleaned = " ".join(words[:max_words]).strip().rstrip(",;:") + "."

        cleaned = cleaned.strip()
        if not cleaned:
            return "I am here with you. Let's take one small step together."

        if cleaned[-1] not in ".!?":
            cleaned = f"{cleaned}."

        return cleaned

    cleaned = _hard_clean_text(str(text or ""))
    if not cleaned:
        return "I am here with you. Let's take one small step together."

    extracted_blob_text = _extract_text_from_cloud_blob(cleaned)
    if extracted_blob_text:
        cleaned = _hard_clean_text(extracted_blob_text)
    else:
        response_match = re.search(
            r'"response"\s*:\s*"((?:\\.|[^"\\])*)"',
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if response_match:
            escaped_value = response_match.group(1)
            try:
                cleaned = json.loads(f'"{escaped_value}"')
            except ValueError:
                cleaned = escaped_value.replace("\\n", "\n").replace('\\"', '"')
            cleaned = _hard_clean_text(cleaned)

    cleaned = _remove_disallowed_tail(cleaned)
    cleaned = _strip_prompt_leakage(cleaned)

    # Normalize whitespace and strip markdown artifacts.
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(
        r"(?i)(is there anything specific you'd like to talk about\??)(?:\s+\1)+",
        r"\1",
        cleaned,
    )

    cleaned = _collapse_repeated_ngrams(cleaned)

    cleaned = WHITESPACE_REGEX.sub(" ", cleaned).strip()
    cleaned = _dedupe_sentences(cleaned, max_sentences=3)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([.!?])(?:\s*[.!?])+", r"\1", cleaned)

    # Keep response concise for frontend UX.
    words = cleaned.split()
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words]).strip().rstrip(",;:") + "."

    if not cleaned:
        return "I am here with you. Let's take one small step together."

    if cleaned[-1] not in ".!?":
        cleaned = f"{cleaned}."

    return cleaned


def _compact_cloud_error(error: Exception) -> str:
    raw = str(error or "").strip()
    if not raw:
        return "Cloud LLM unavailable"

    simplified = raw.replace("\n", " ").replace("\r", " ").strip()
    if "(Caused by" in simplified:
        simplified = simplified.split("(Caused by", maxsplit=1)[0].strip()

    simplified = WHITESPACE_REGEX.sub(" ", simplified).strip()
    if len(simplified) > 220:
        simplified = simplified[:217].rstrip() + "..."
    return simplified or "Cloud LLM unavailable"


def _drain_complete_sentences(buffer: str) -> Tuple[List[str], str]:
    if not buffer:
        return [], ""

    segments = SENTENCE_BOUNDARY_REGEX.split(buffer)
    if len(segments) <= 1:
        stripped_buffer = buffer.strip()
        # Trigger sentence-TTS as soon as a sentence-ending punctuation arrives,
        # even if the model has not emitted trailing whitespace yet.
        if (
            stripped_buffer
            and SENTENCE_TERMINATOR_REGEX.search(stripped_buffer)
            and len(stripped_buffer.split()) >= 3
        ):
            return [stripped_buffer], ""
        return [], buffer

    completed = [segment.strip() for segment in segments[:-1] if segment.strip()]
    pending = segments[-1]
    return completed, pending


async def _generate_tts_base64(text: str) -> tuple[Optional[str], Optional[str]]:
    global tts_consecutive_failures, tts_disabled_until_epoch

    if not text:
        return None, "TTS skipped: empty text"

    if not TTS_ENABLED:
        return None, "TTS disabled by configuration"

    if edge_tts is None:
        return None, "TTS unavailable: edge-tts is not installed"

    now_epoch = time.time()
    if now_epoch < tts_disabled_until_epoch:
        remaining = int(tts_disabled_until_epoch - now_epoch)
        return None, f"TTS temporarily disabled for {remaining}s after repeated failures"

    audio_path = None
    try:
        audio_path = os.path.join(tempfile.gettempdir(), f"serenity_tts_{uuid.uuid4().hex}.mp3")
        communicator = edge_tts.Communicate(
            text=text,
            voice=TTS_VOICE,
            rate=TTS_RATE,
            pitch=TTS_PITCH,
        )
        await asyncio.wait_for(communicator.save(audio_path), timeout=TTS_TIMEOUT_SECONDS)

        with open(audio_path, "rb") as audio_file:
            encoded = base64.b64encode(audio_file.read()).decode("utf-8")

        tts_consecutive_failures = 0
        tts_disabled_until_epoch = 0.0
        return encoded, None
    except asyncio.TimeoutError:
        tts_consecutive_failures += 1
        if tts_consecutive_failures >= TTS_FAILURE_THRESHOLD:
            tts_disabled_until_epoch = time.time() + TTS_COOLDOWN_SECONDS
            LOGGER.warning(
                "TTS disabled for %ss after %s consecutive failures.",
                TTS_COOLDOWN_SECONDS,
                tts_consecutive_failures,
            )
        LOGGER.warning("TTS generation timed out after %s seconds", TTS_TIMEOUT_SECONDS)
        return None, "TTS timeout. Returning text response only."
    except Exception as exc:
        tts_consecutive_failures += 1
        if tts_consecutive_failures >= TTS_FAILURE_THRESHOLD:
            tts_disabled_until_epoch = time.time() + TTS_COOLDOWN_SECONDS
            LOGGER.warning(
                "TTS disabled for %ss after %s consecutive failures.",
                TTS_COOLDOWN_SECONDS,
                tts_consecutive_failures,
            )

        failure_text = str(exc)
        if "403" in failure_text:
            LOGGER.warning("TTS service returned 403. Falling back to text response only.")
            return None, "TTS service unavailable (HTTP 403). Returning text response only."

        LOGGER.warning("TTS generation failed: %s", exc)
        return None, f"TTS failed: {exc}. Returning text response only."
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass


async def _warmup_edge_tts_once() -> None:
    if not TTS_WARMUP_ENABLED or not TTS_ENABLED or edge_tts is None:
        return

    if _state_get("tts_warmed_up", False):
        return

    warmup_text = os.getenv("SERENITY_TTS_WARMUP_TEXT", "Hello.").strip() or "Hello."
    started = time.time()
    _encoded_audio, tts_error = await _generate_tts_base64(warmup_text)
    elapsed = time.time() - started

    if tts_error:
        LOGGER.warning("TTS warmup completed with warning after %.2fs: %s", elapsed, tts_error)
    else:
        LOGGER.info("TTS warmup completed in %.2fs", elapsed)

    _state_set("tts_warmed_up", True)


async def _warmup_cloud_llm_once() -> None:
    cloud_llm_client = _state_get("cloud_llm_client")
    if cloud_llm_client is None:
        return

    try:
        elapsed = await asyncio.wait_for(
            run_in_threadpool(cloud_llm_client.warmup, CLOUD_LLM_WARMUP_TEXT),
            timeout=CLOUD_LLM_WARMUP_TIMEOUT_SECONDS,
        )
        LOGGER.info("Cloud LLM warmup completed in %.2fs", elapsed)
    except asyncio.TimeoutError:
        LOGGER.warning(
            "Cloud LLM warmup timed out after %ss",
            CLOUD_LLM_WARMUP_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        LOGGER.warning("Cloud LLM warmup failed: %s", exc)


def _transcribe_with_whisper(audio_path: str) -> tuple[str, Optional[str]]:
    init_error = _ensure_whisper_runtime_sync()
    if init_error and _state_get("whisper_model") is None:
        return "", str(init_error)

    whisper_model = _state_get("whisper_model")
    whisper_device_in_use = _state_get("whisper_device_in_use", "cpu")
    whisper_backend = _state_get("whisper_backend", "faster-whisper")

    if whisper_model is None:
        return "", "Whisper runtime unavailable"

    try:
        if whisper_backend == "faster-whisper":
            segments, _info = whisper_model.transcribe(
                audio_path,
                language="en",
                beam_size=1,
                vad_filter=True,
            )
            text = " ".join([str(segment.text).strip() for segment in segments if str(segment.text).strip()]).strip()
        else:
            # Force fp16 off for openai-whisper to avoid NaN logits seen on some CUDA stacks.
            use_fp16 = False
            transcript = whisper_model.transcribe(audio_path, fp16=use_fp16, language="en")
            text = str(transcript.get("text", "")).strip()
        return text, None
    except Exception as exc:
        error_text = str(exc).lower()
        unstable_logits = (
            "nan" in error_text
            or "categorical" in error_text
            or "expected parameter logits" in error_text
        )

        if whisper_backend == "openai-whisper" and whisper_device_in_use == "cuda" and unstable_logits:
            LOGGER.warning("OpenAI Whisper CUDA became unstable; retrying transcription with CPU whisper runtime.")
            try:
                cpu_whisper_model = _state_get("whisper_model_cpu")
                if cpu_whisper_model is None:
                    if openai_whisper is None:
                        raise RuntimeError("openai-whisper package is unavailable for CPU retry")
                    cpu_whisper_model = openai_whisper.load_model(WHISPER_MODEL_SIZE, device="cpu")
                    _state_set("whisper_model_cpu", cpu_whisper_model)

                cpu_transcript = cpu_whisper_model.transcribe(audio_path, fp16=False, language="en")
                cpu_text = str(cpu_transcript.get("text", "")).strip()
                return cpu_text, None
            except Exception as cpu_exc:
                LOGGER.exception("Whisper CPU retry failed after CUDA instability: %s", cpu_exc)
                return "", f"Whisper transcription failed after CPU retry: {cpu_exc}"

        LOGGER.exception(
            "Whisper transcription failed on %s using %s: %s",
            whisper_device_in_use,
            whisper_backend,
            exc,
        )
        return "", f"Whisper transcription failed: {exc}"


async def _generate_llm_response(
    user_text: str,
    dominant_emotion: str,
    serialized_history: List[Dict[str, str]],
    username: str = "anonymous",
    emotion_probabilities: Optional[Dict[str, Dict[str, float]]] = None,
) -> tuple[str, Optional[str], List[str], List[str]]:
    cloud_llm_client = _state_get("cloud_llm_client")
    if cloud_llm_client is None:
        cloud_llm_client, init_error = await _ensure_cloud_llm_client()
    else:
        init_error = None

    if cloud_llm_client is None:
        init_error = _state_get("rag_init_error")
        error_msg = "Cloud LLM client unavailable"
        if init_error:
            error_msg = f"Cloud LLM client unavailable: {init_error}"
        return (
            "I am here with you. Let's take one small step together.",
            error_msg,
            [],
            [],
        )

    # user_text is the only field expected by the deployed EC2 /chat API.
    _ = (dominant_emotion, serialized_history, username, emotion_probabilities)

    try:
        async with llm_generation_semaphore:
            response = await asyncio.wait_for(
                run_in_threadpool(cloud_llm_client.ask_serenity, user_text),
                timeout=LLM_TIMEOUT_SECONDS,
            )
        cleaned_response = _sanitize_cloud_llm_response(str(response))
        return cleaned_response, None, [], []
    except asyncio.TimeoutError:
        LOGGER.warning("LLM generation timed out after %s seconds", LLM_TIMEOUT_SECONDS)
        return (
            "I am with you. Let's breathe slowly and focus on one manageable next step.",
            "LLM timeout. Using safe fallback response.",
            [],
            [],
        )
    except Exception as exc:
        if isinstance(exc, CloudLLMError):
            LOGGER.warning("LLM generation failed: %s", _compact_cloud_error(exc))
        else:
            LOGGER.exception("LLM generation failed: %s", exc)
        return (
            "I am here with you. Let's take one small step together.",
            "LLM generation failed. Using safe fallback response.",
            [],
            [],
        )


async def _stream_multimodal_generation(
    username: str,
    user_text: str,
    dominant_emotion: str,
    serialized_history: List[Dict[str, str]],
    emotion_probabilities: Optional[Dict[str, Dict[str, float]]] = None,
):
    cloud_llm_client = _state_get("cloud_llm_client")
    init_error = None

    if cloud_llm_client is None:
        cloud_llm_client, init_error = await _ensure_cloud_llm_client()
    if cloud_llm_client is None and init_error is None:
        init_error = _state_get("rag_init_error")

    if cloud_llm_client is None:
        errors = ["Cloud LLM client unavailable"]
        if init_error:
            errors = [f"Cloud LLM client unavailable: {init_error}"]
        yield {
            "type": "generation_result",
            "llm_response": "I am here with you. Let's take one small step together.",
            "errors": errors,
        }
        return

    # Current deployed cloud endpoint consumes only text.
    _ = (username, dominant_emotion, serialized_history, emotion_probabilities)

    loop = asyncio.get_running_loop()
    chunk_queue: "asyncio.Queue[Optional[str]]" = asyncio.Queue()

    stream_sentence_tts_enabled = (
        STREAM_TTS_SENTENCE_AUDIO
        and TTS_STREAMING_ENABLED
        and TTS_ENABLED
        and edge_tts is not None
    )
    stream_sentence_tts_live = stream_sentence_tts_enabled and not STREAM_TTS_FINAL_TEXT_ONLY
    emit_provisional_text = True

    def _producer() -> str:
        full_text_parts: List[str] = []
        try:
            try:
                for chunk in cloud_llm_client.stream_serenity(user_text):
                    piece = str(chunk or "")
                    if not piece:
                        continue
                    full_text_parts.append(piece)
                    loop.call_soon_threadsafe(chunk_queue.put_nowait, piece)
            except CloudLLMError as stream_exc:
                # Retry once using non-stream generation when streaming endpoint
                # is temporarily unavailable.
                LOGGER.warning(
                    "Cloud stream failed, retrying non-stream response: %s",
                    _compact_cloud_error(stream_exc),
                )
                fallback_text = str(cloud_llm_client.ask_serenity(user_text) or "")
                if fallback_text:
                    full_text_parts.append(fallback_text)
                    loop.call_soon_threadsafe(chunk_queue.put_nowait, fallback_text)
        finally:
            loop.call_soon_threadsafe(chunk_queue.put_nowait, None)
        return "".join(full_text_parts)

    generation_task = asyncio.create_task(
        asyncio.wait_for(
            run_in_threadpool(_producer),
            timeout=LLM_TIMEOUT_SECONDS,
        )
    )

    producer_done = False
    stream_errors: List[str] = []
    raw_stream_text = ""
    display_text = ""
    sentence_stream_buffer = ""
    reached_first_asterisk = False
    final_response = ""
    sentence_sequence = 0
    sentence_key_to_sequence: Dict[str, int] = {}

    sentence_queue: Optional["asyncio.Queue[Optional[Tuple[int, str]]]"] = None
    tts_event_queue: Optional["asyncio.Queue[Optional[Tuple[int, str, Optional[str], Optional[str]]]]"] = None
    tts_worker_task: Optional[asyncio.Task] = None

    def _sentence_key(text: str) -> str:
        return re.sub(r"\W+", "", str(text or "").lower())

    def _register_sentence(
        sentence_text: str,
        allow_existing: bool = False,
    ) -> Optional[Tuple[int, str, bool]]:
        nonlocal sentence_sequence
        cleaned_sentence = _hard_clean_text(sentence_text)
        if not cleaned_sentence:
            return None

        key = _sentence_key(cleaned_sentence)
        if key:
            existing_sequence = sentence_key_to_sequence.get(key)
            if existing_sequence is not None:
                if allow_existing:
                    return existing_sequence, cleaned_sentence, False
                return None

        sentence_sequence += 1
        if key:
            sentence_key_to_sequence[key] = sentence_sequence
        return sentence_sequence, cleaned_sentence, True

    async def _tts_sentence_worker() -> None:
        if sentence_queue is None or tts_event_queue is None:
            return

        while True:
            item = await sentence_queue.get()
            if item is None:
                break

            sequence, sentence_text = item
            encoded_audio, tts_error = await _generate_tts_base64(sentence_text)
            await tts_event_queue.put((sequence, sentence_text, encoded_audio, tts_error))

        await tts_event_queue.put(None)

    async def _drain_ready_tts_events() -> tuple[List[Dict[str, object]], List[str], bool]:
        events: List[Dict[str, object]] = []
        errors: List[str] = []
        worker_done = False

        if tts_event_queue is None:
            return events, errors, worker_done

        while not tts_event_queue.empty():
            item = tts_event_queue.get_nowait()
            if item is None:
                worker_done = True
                break

            sequence, sentence_text, encoded_audio, tts_error = item
            if encoded_audio:
                events.append(
                    {
                        "type": "assistant_sentence_tts",
                        "text": sentence_text,
                        "sequence": sequence,
                        "audio_base64": encoded_audio,
                    }
                )
            if tts_error:
                errors.append(tts_error)

        return events, errors, worker_done

    if stream_sentence_tts_enabled:
        sentence_queue = asyncio.Queue()
        tts_event_queue = asyncio.Queue()
        tts_worker_task = asyncio.create_task(_tts_sentence_worker())

    try:
        while True:
            emitted = False
            pending_pieces: List[Optional[str]] = []

            if chunk_queue.empty() and not producer_done and not generation_task.done():
                try:
                    pending_pieces.append(
                        await asyncio.wait_for(
                        chunk_queue.get(),
                        timeout=STREAM_QUEUE_WAIT_SECONDS,
                        )
                    )
                except asyncio.TimeoutError:
                    pass

            while pending_pieces or not chunk_queue.empty():
                piece = pending_pieces.pop(0) if pending_pieces else chunk_queue.get_nowait()
                if piece is None:
                    producer_done = True
                    continue

                piece = _normalize_cloud_stream_piece(str(piece))
                if not piece:
                    continue

                raw_stream_text += piece

                clean_piece, reached_first_asterisk = _clean_stream_token(piece, reached_first_asterisk)
                if not clean_piece:
                    continue

                candidate_text = _remove_disallowed_tail(display_text + clean_piece)
                if candidate_text == display_text:
                    continue

                # Tail stripping can retract already-shown provisional text; emit replace
                # so UI/TTS state remains aligned with the sanitized transcript.
                if not candidate_text.startswith(display_text):
                    display_text = candidate_text
                    if emit_provisional_text:
                        yield {
                            "type": "assistant_replace",
                            "text": display_text,
                        }
                        emitted = True
                    if stream_sentence_tts_live:
                        sentence_stream_buffer = ""
                    continue

                delta_piece = candidate_text[len(display_text):]
                if not delta_piece:
                    continue

                display_text = candidate_text
                if emit_provisional_text:
                    yield {
                        "type": "assistant_delta",
                        "delta": delta_piece,
                        "text": display_text,
                    }
                    emitted = True

                if stream_sentence_tts_live:
                    sentence_stream_buffer += delta_piece
                    completed_sentences, sentence_stream_buffer = _drain_complete_sentences(sentence_stream_buffer)
                    for sentence_text in completed_sentences:
                        registered = _register_sentence(sentence_text)
                        if registered is None:
                            continue

                        sequence, cleaned_sentence, _is_new = registered
                        yield {
                            "type": "assistant_sentence",
                            "text": cleaned_sentence,
                            "sequence": sequence,
                        }
                        emitted = True

                        if sentence_queue is not None:
                            await sentence_queue.put((sequence, cleaned_sentence))

            tts_events, tts_errors, _worker_done = await _drain_ready_tts_events()
            for event in tts_events:
                yield event
                emitted = True
            if tts_errors:
                stream_errors.extend(tts_errors)

            if producer_done and chunk_queue.empty() and generation_task.done():
                break

            if not emitted:
                await asyncio.sleep(0.005)

        producer_final_text = str(await generation_task or "").strip()
        if not raw_stream_text:
            raw_stream_text = producer_final_text

        if not display_text:
            display_text = _hard_clean_text(producer_final_text or raw_stream_text, preserve_edges=True)

        final_response = _sanitize_cloud_llm_response(display_text or producer_final_text or raw_stream_text)
    except asyncio.TimeoutError:
        generation_task.cancel()
        with contextlib.suppress(Exception):
            await generation_task
        LOGGER.warning("Streaming LLM generation timed out after %s seconds", LLM_TIMEOUT_SECONDS)
        stream_errors.append("LLM timeout. Returning partial response.")
        final_response = _sanitize_cloud_llm_response(display_text or raw_stream_text)
        if not final_response:
            final_response = "I am with you. Let's breathe slowly and focus on one manageable next step."
    except Exception as exc:
        generation_task.cancel()
        with contextlib.suppress(Exception):
            await generation_task
        if isinstance(exc, CloudLLMError):
            LOGGER.warning("Streaming LLM generation failed: %s", _compact_cloud_error(exc))
        else:
            LOGGER.exception("Streaming LLM generation failed: %s", exc)
        stream_errors.append(f"LLM generation failed: {_compact_cloud_error(exc)}")
        final_response = _sanitize_cloud_llm_response(display_text or raw_stream_text)
        if not final_response:
            final_response = "I am here with you. Let's take one small step together."

    display_text = display_text.strip()
    final_response = _sanitize_cloud_llm_response(final_response or display_text or raw_stream_text)
    if final_response and (final_response != display_text or not emit_provisional_text):
        yield {
            "type": "assistant_replace",
            "text": final_response,
        }

    # Ensure TTS playback aligns with the finalized assistant response text.
    if stream_sentence_tts_enabled and final_response:
        if STREAM_TTS_FINAL_TEXT_ONLY:
            # Instruct clients to stop any previously queued/playing provisional audio.
            yield {"type": "assistant_tts_reset"}

        final_sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", final_response) if part.strip()]
        final_sentence_sequences: List[int] = []
        for sentence_text in final_sentences:
            registered = _register_sentence(sentence_text, allow_existing=True)
            if registered is None:
                continue

            sequence, cleaned_sentence, is_new = registered
            final_sentence_sequences.append(sequence)
            if is_new:
                yield {
                    "type": "assistant_sentence",
                    "text": cleaned_sentence,
                    "sequence": sequence,
                }
                if sentence_queue is not None:
                    await sentence_queue.put((sequence, cleaned_sentence))

        # Live TTS can begin early; once final polishing is known, trim queued/playing
        # audio to the finalized sentence boundary.
        if stream_sentence_tts_live and final_sentence_sequences:
            yield {
                "type": "assistant_tts_trim",
                "max_sequence": max(final_sentence_sequences),
            }

    if sentence_queue is not None:
        await sentence_queue.put(None)

    if tts_worker_task is not None:
        worker_completed = False
        while not worker_completed:
            tts_events, tts_errors, worker_done = await _drain_ready_tts_events()
            for event in tts_events:
                yield event
            if tts_errors:
                stream_errors.extend(tts_errors)

            if worker_done:
                worker_completed = True
                break

            if tts_worker_task.done() and (tts_event_queue is None or tts_event_queue.empty()):
                worker_completed = True
                break

            await asyncio.sleep(0.01)

        with contextlib.suppress(Exception):
            await tts_worker_task

    deduped_errors = list(dict.fromkeys([err for err in stream_errors if err]))
    yield {
        "type": "generation_result",
        "llm_response": final_response,
        "errors": deduped_errors,
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="running", rag_loaded=_state_get("cloud_llm_client") is not None)


@app.post("/api/interact", response_model=InteractResponse)
async def interact(
    username: str = Form(...),
    image: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
    user_message: Optional[str] = Form(default=None),
) -> InteractResponse:
    if not username:
        raise HTTPException(status_code=400, detail="Missing username.")
    if file is None:
        raise HTTPException(status_code=400, detail="Microphone input is required for interaction.")

    errors: List[str] = []
    temp_audio_path: Optional[str] = None
    audio_available = False
    image_available = bool(image)

    speech_emotion = "Neutral"
    speech_confidence = 0.0
    face_emotion = "Neutral"
    face_confidence = 0.0
    transcription = ""
    speech_probs: Dict[str, float] = _to_probability_vector("Neutral", 0.0)
    face_probs: Optional[Dict[str, float]] = None
    fused_probs: Dict[str, float] = speech_probs

    try:
        tasks: List[tuple[str, asyncio.Future]] = []

        if file is not None:
            audio_bytes = await file.read()
            if not audio_bytes:
                raise HTTPException(status_code=400, detail="Audio upload was empty.")

            suffix = ".wav"
            filename = (file.filename or "").lower()
            if filename.endswith(".webm"):
                suffix = ".webm"
            elif filename.endswith(".mp3"):
                suffix = ".mp3"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio_path = temp_audio.name
                audio_available = True

            tasks.append(
                (
                    "transcribe",
                    asyncio.create_task(
                        asyncio.wait_for(
                            run_in_threadpool(_transcribe_with_whisper, temp_audio_path),
                            timeout=WHISPER_TIMEOUT_SECONDS,
                        )
                    ),
                )
            )
            tasks.append(
                (
                    "speech",
                    asyncio.create_task(
                        asyncio.wait_for(
                                run_in_threadpool(predict_audio_emotion, temp_audio_path, _state_get("speech_runtime")),
                            timeout=EMOTION_TIMEOUT_SECONDS,
                        )
                    ),
                )
            )

        if image:
            tasks.append(
                (
                    "face",
                    asyncio.create_task(
                        asyncio.wait_for(
                                run_in_threadpool(analyze_face, image, _state_get("face_runtime")),
                            timeout=EMOTION_TIMEOUT_SECONDS,
                        )
                    ),
                )
            )

        if tasks:
            task_names = [name for name, _ in tasks]
            task_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

            for name, result in zip(task_names, task_results):
                if name == "transcribe":
                    if isinstance(result, asyncio.TimeoutError):
                        errors.append("Transcription timeout")
                    elif isinstance(result, Exception):
                        errors.append(f"Transcription failed: {result}")
                    else:
                        transcription, transcribe_error = result
                        if transcribe_error:
                            errors.append(transcribe_error)

                if name == "speech":
                    if isinstance(result, asyncio.TimeoutError):
                        errors.append("Speech emotion timeout")
                    elif isinstance(result, Exception):
                        errors.append(f"Speech emotion failed: {result}")
                    else:
                        speech_emotion, speech_confidence, speech_error = _normalize_prediction(result)
                        speech_probs = _to_probability_vector(speech_emotion, speech_confidence)
                        if speech_error:
                            errors.append(f"Speech model: {speech_error}")

                if name == "face":
                    if isinstance(result, asyncio.TimeoutError):
                        errors.append("Face emotion timeout")
                    elif isinstance(result, Exception):
                        errors.append(f"Face emotion failed: {result}")
                    else:
                        face_emotion, face_confidence, face_error = _normalize_prediction(result)
                        face_probs = _to_probability_vector(face_emotion, face_confidence)
                        if face_error:
                            errors.append(f"Face model: {face_error}")

        if audio_available and image_available:
            face_probs = face_probs or _to_probability_vector(face_emotion, face_confidence)
            fused_probs = _fuse_probability_vectors(
                speech_probs=speech_probs,
                face_probs=face_probs,
            )
            dominant_emotion = _dominant_from_probabilities(fused_probs)
        elif audio_available:
            fused_probs = speech_probs
            dominant_emotion = _dominant_from_probabilities(fused_probs)
        elif image_available:
            face_probs = face_probs or _to_probability_vector(face_emotion, face_confidence)
            fused_probs = face_probs
            dominant_emotion = _dominant_from_probabilities(fused_probs)
        else:
            dominant_emotion = "Neutral"

        emotion_probabilities: Dict[str, Dict[str, float]] = {
            "speech": speech_probs,
            "fused": fused_probs,
        }
        if face_probs is not None:
            emotion_probabilities["face"] = face_probs

        user_text = (transcription or user_message or "").strip()
        if not user_text:
            errors.append("No speech transcription detected. Please try speaking again.")
            errors = list(dict.fromkeys(errors))
            return InteractResponse(
                dominant_emotion=dominant_emotion,
                speech_emotion=str(speech_emotion),
                face_emotion=str(face_emotion),
                transcription="",
                llm_response="",
                tts_audio_base64=None,
                tts_audio_segments_base64=[],
                errors=errors,
            )
        llm_response = ""

        db = SessionLocal()
        try:
            history_turns = fetch_recent_turns(db, username=username, limit=6)
            serialized_history = _serialize_turns(history_turns)

            llm_response, llm_error, tts_audio_segments, stream_tts_errors = await _generate_llm_response(
                user_text=user_text,
                dominant_emotion=dominant_emotion,
                serialized_history=serialized_history,
                username=username,
                emotion_probabilities=emotion_probabilities,
            )
            if llm_error:
                errors.append(llm_error)
            if stream_tts_errors:
                errors.extend(stream_tts_errors)

            try:
                persist_turn(
                    db,
                    username=username,
                    user_text=user_text,
                    assistant_text=llm_response,
                    dominant_emotion=dominant_emotion,
                    speech_emotion=speech_emotion,
                    face_emotion=face_emotion,
                )
            except Exception as exc:
                LOGGER.exception("Failed to persist conversation turn: %s", exc)
                errors.append(f"DB log failed: {exc}")
        finally:
            db.close()

        tts_audio_base64: Optional[str] = None
        if not tts_audio_segments and not stream_tts_errors:
            tts_audio_base64, tts_error = await _generate_tts_base64(llm_response)
            if tts_error:
                errors.append(tts_error)

        errors = list(dict.fromkeys(errors))

        return InteractResponse(
            dominant_emotion=dominant_emotion,
            speech_emotion=str(speech_emotion),
            face_emotion=str(face_emotion),
            transcription=user_text,
            llm_response=llm_response,
            tts_audio_base64=tts_audio_base64,
            tts_audio_segments_base64=tts_audio_segments,
            errors=errors,
        )
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("/api/interact failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Interaction failed: {exc}") from exc
    finally:
        if file is not None:
            await file.close()
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except OSError as exc:
                LOGGER.warning("Failed to remove temp interaction audio %s: %s", temp_audio_path, exc)


@app.post("/api/interact/stream")
async def interact_stream(
    username: str = Form(...),
    image: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
    user_message: Optional[str] = Form(default=None),
):
    if not username:
        raise HTTPException(status_code=400, detail="Missing username.")
    if file is None:
        raise HTTPException(status_code=400, detail="Microphone input is required for interaction.")

    audio_bytes = await file.read()
    await file.close()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio upload was empty.")

    suffix = ".wav"
    filename = (file.filename or "").lower()
    if filename.endswith(".webm"):
        suffix = ".webm"
    elif filename.endswith(".mp3"):
        suffix = ".mp3"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name

    async def event_stream():
        errors: List[str] = []
        speech_emotion = "Neutral"
        speech_confidence = 0.0
        face_emotion = "Neutral"
        face_confidence = 0.0
        transcription = ""
        speech_probs: Dict[str, float] = _to_probability_vector("Neutral", 0.0)
        face_probs: Optional[Dict[str, float]] = None
        fused_probs: Dict[str, float] = speech_probs

        async def _named_task(name: str, coro):
            try:
                result = await coro
                return name, result, None
            except Exception as exc:
                return name, None, exc

        tasks = [
            asyncio.create_task(
                _named_task(
                    "transcribe",
                    asyncio.wait_for(
                        run_in_threadpool(_transcribe_with_whisper, temp_audio_path),
                        timeout=WHISPER_TIMEOUT_SECONDS,
                    ),
                )
            ),
            asyncio.create_task(
                _named_task(
                    "speech",
                    asyncio.wait_for(
                        run_in_threadpool(predict_audio_emotion, temp_audio_path, _state_get("speech_runtime")),
                        timeout=EMOTION_TIMEOUT_SECONDS,
                    ),
                )
            ),
        ]

        if image:
            tasks.append(
                asyncio.create_task(
                    _named_task(
                        "face",
                        asyncio.wait_for(
                            run_in_threadpool(analyze_face, image, _state_get("face_runtime")),
                            timeout=EMOTION_TIMEOUT_SECONDS,
                        ),
                    )
                )
            )

        try:
            for completed in asyncio.as_completed(tasks):
                name, result, task_error = await completed

                if task_error is not None:
                    message = str(task_error)
                    if name == "transcribe":
                        errors.append(f"Transcription failed: {message}")
                    elif name == "speech":
                        errors.append(f"Speech emotion failed: {message}")
                    elif name == "face":
                        errors.append(f"Face emotion failed: {message}")
                    yield _to_ndjson_event({"type": "error", "message": errors[-1]})
                    continue

                if name == "transcribe":
                    transcription, transcribe_error = result
                    if transcribe_error:
                        errors.append(transcribe_error)
                        yield _to_ndjson_event({"type": "error", "message": transcribe_error})
                    yield _to_ndjson_event(
                        {
                            "type": "transcription",
                            "text": transcription,
                        }
                    )

                if name == "speech":
                    speech_emotion, speech_confidence, speech_error = _normalize_prediction(result)
                    speech_probs = _to_probability_vector(speech_emotion, speech_confidence)
                    if speech_error:
                        errors.append(f"Speech model: {speech_error}")
                        yield _to_ndjson_event({"type": "error", "message": errors[-1]})
                    yield _to_ndjson_event(
                        {
                            "type": "emotion_partial",
                            "speech_emotion": str(speech_emotion),
                            "speech_confidence": float(speech_confidence),
                        }
                    )

                if name == "face":
                    face_emotion, face_confidence, face_error = _normalize_prediction(result)
                    face_probs = _to_probability_vector(face_emotion, face_confidence)
                    if face_error:
                        errors.append(f"Face model: {face_error}")
                        yield _to_ndjson_event({"type": "error", "message": errors[-1]})
                    yield _to_ndjson_event(
                        {
                            "type": "emotion_partial",
                            "face_emotion": str(face_emotion),
                            "face_confidence": float(face_confidence),
                        }
                    )

            if image:
                face_probs = face_probs or _to_probability_vector(face_emotion, face_confidence)
                fused_probs = _fuse_probability_vectors(speech_probs=speech_probs, face_probs=face_probs)
                dominant_emotion = _dominant_from_probabilities(fused_probs)
            else:
                fused_probs = speech_probs
                dominant_emotion = _dominant_from_probabilities(fused_probs)

            emotion_probabilities: Dict[str, Dict[str, float]] = {
                "speech": speech_probs,
                "fused": fused_probs,
            }
            if face_probs is not None:
                emotion_probabilities["face"] = face_probs

            yield _to_ndjson_event(
                {
                    "type": "emotion",
                    "dominant_emotion": dominant_emotion,
                    "speech_emotion": str(speech_emotion),
                    "face_emotion": str(face_emotion),
                }
            )

            user_text = (transcription or user_message or "").strip()
            if not user_text:
                message = "No speech transcription detected. Please try speaking again."
                errors.append(message)
                yield _to_ndjson_event({"type": "error", "message": message})
                yield _to_ndjson_event(
                    {
                        "type": "final",
                        "llm_response": "",
                        "transcription": "",
                        "dominant_emotion": dominant_emotion,
                        "speech_emotion": str(speech_emotion),
                        "face_emotion": str(face_emotion),
                    }
                )
                return
            yield _to_ndjson_event({"type": "user_text", "text": user_text, "source": "voice"})

            db = SessionLocal()
            llm_response = ""
            try:
                history_turns = fetch_recent_turns(db, username=username, limit=6)
                serialized_history = _serialize_turns(history_turns)

                stream = _stream_multimodal_generation(
                    username=username,
                    user_text=user_text,
                    dominant_emotion=dominant_emotion,
                    serialized_history=serialized_history,
                    emotion_probabilities=emotion_probabilities,
                )

                async for event in stream:
                    if event.get("type") == "generation_result":
                        llm_response = str(event.get("llm_response") or llm_response)
                        llm_response = _sanitize_cloud_llm_response(llm_response)
                        llm_errors = event.get("errors") or []
                        errors.extend([str(item) for item in llm_errors if item])
                        continue
                    yield _to_ndjson_event(event)

                try:
                    persist_turn(
                        db,
                        username=username,
                        user_text=user_text,
                        assistant_text=llm_response,
                        dominant_emotion=dominant_emotion,
                        speech_emotion=str(speech_emotion),
                        face_emotion=str(face_emotion),
                    )
                except Exception as exc:
                    LOGGER.exception("Failed to persist streaming interaction turn: %s", exc)
                    errors.append(f"DB log failed: {exc}")
            finally:
                db.close()

            for message in list(dict.fromkeys(errors)):
                yield _to_ndjson_event({"type": "error", "message": message})

            yield _to_ndjson_event(
                {
                    "type": "final",
                    "llm_response": llm_response,
                    "transcription": user_text,
                    "dominant_emotion": dominant_emotion,
                    "speech_emotion": str(speech_emotion),
                    "face_emotion": str(face_emotion),
                }
            )
        finally:
            if os.path.exists(temp_audio_path):
                with contextlib.suppress(OSError):
                    os.remove(temp_audio_path)

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.post("/api/chat", response_model=InteractResponse)
async def chat(payload: ChatRequest) -> InteractResponse:
    username = (payload.username or "").strip()
    user_text = (payload.message or "").strip()
    if not username:
        raise HTTPException(status_code=400, detail="Missing username.")
    if not user_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    errors: List[str] = []
    dominant_emotion = "Neutral"
    llm_response = ""

    db = SessionLocal()
    try:
        history_turns = fetch_recent_turns(db, username=username, limit=8)
        serialized_history = _serialize_turns(history_turns)

        llm_response, llm_error, tts_audio_segments, stream_tts_errors = await _generate_llm_response(
            user_text=user_text,
            dominant_emotion=dominant_emotion,
            serialized_history=serialized_history,
            username=username,
        )
        if llm_error:
            errors.append(llm_error)
        if stream_tts_errors:
            errors.extend(stream_tts_errors)

        try:
            persist_turn(
                db,
                username=username,
                user_text=user_text,
                assistant_text=llm_response,
                dominant_emotion=dominant_emotion,
                speech_emotion="Neutral",
                face_emotion="Neutral",
            )
        except Exception as exc:
            LOGGER.exception("Failed to persist chat turn: %s", exc)
            errors.append(f"DB log failed: {exc}")
    finally:
        db.close()

    tts_audio_base64: Optional[str] = None
    if not tts_audio_segments and not stream_tts_errors:
        tts_audio_base64, tts_error = await _generate_tts_base64(llm_response)
        if tts_error:
            errors.append(tts_error)

    errors = list(dict.fromkeys(errors))

    return InteractResponse(
        dominant_emotion=dominant_emotion,
        speech_emotion="Neutral",
        face_emotion="Neutral",
        transcription=user_text,
        llm_response=llm_response,
        tts_audio_base64=tts_audio_base64,
        tts_audio_segments_base64=tts_audio_segments,
        errors=errors,
    )


@app.post("/api/chat/stream")
async def chat_stream(payload: ChatRequest):
    username = (payload.username or "").strip()
    user_text = (payload.message or "").strip()
    if not username:
        raise HTTPException(status_code=400, detail="Missing username.")
    if not user_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    async def event_stream():
        errors: List[str] = []
        dominant_emotion = "Neutral"

        yield _to_ndjson_event({"type": "user_text", "text": user_text, "source": "text"})
        yield _to_ndjson_event(
            {
                "type": "emotion",
                "dominant_emotion": dominant_emotion,
                "speech_emotion": "Neutral",
                "face_emotion": "Neutral",
            }
        )

        db = SessionLocal()
        llm_response = ""
        try:
            history_turns = fetch_recent_turns(db, username=username, limit=8)
            serialized_history = _serialize_turns(history_turns)

            stream = _stream_multimodal_generation(
                username=username,
                user_text=user_text,
                dominant_emotion=dominant_emotion,
                serialized_history=serialized_history,
                emotion_probabilities=None,
            )

            async for event in stream:
                if event.get("type") == "generation_result":
                    llm_response = str(event.get("llm_response") or llm_response)
                    llm_response = _sanitize_cloud_llm_response(llm_response)
                    llm_errors = event.get("errors") or []
                    errors.extend([str(item) for item in llm_errors if item])
                    continue
                yield _to_ndjson_event(event)

            try:
                persist_turn(
                    db,
                    username=username,
                    user_text=user_text,
                    assistant_text=llm_response,
                    dominant_emotion=dominant_emotion,
                    speech_emotion="Neutral",
                    face_emotion="Neutral",
                )
            except Exception as exc:
                LOGGER.exception("Failed to persist streaming chat turn: %s", exc)
                errors.append(f"DB log failed: {exc}")
        finally:
            db.close()

        for message in list(dict.fromkeys(errors)):
            yield _to_ndjson_event({"type": "error", "message": message})

        yield _to_ndjson_event(
            {
                "type": "final",
                "llm_response": llm_response,
                "transcription": user_text,
                "dominant_emotion": dominant_emotion,
                "speech_emotion": "Neutral",
                "face_emotion": "Neutral",
            }
        )

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
