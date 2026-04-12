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

import bcrypt
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

try:
    import torch
except ImportError:
    torch = None

from backend.audio_core import initialize_audio_runtime, predict_audio_emotion
from backend.emotion_core import initialize_face_runtime, analyze_face
from backend.database import (
    SessionLocal, engine, fetch_questionnaire_results, fetch_recent_sessions_with_emotions,
    fetch_recent_turn_summaries, persist_questionnaire_result, persist_turn
)
from backend.cloud_llm_core import CloudLLMClient, CloudLLMError
from backend.questionnaires_data import (
    QUESTIONNAIRE_DEFINITIONS, normalize_questionnaire_type, questionnaire_clinical_flags,
    questionnaire_templates, score_questionnaire
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

# --- Config & Flags ---
EDGE_OPTIMIZED_MODE = os.getenv("SERENITY_EDGE_OPTIMIZED_MODE", "true").lower() == "true"
WHISPER_MODEL_SIZE = os.getenv("SERENITY_WHISPER_MODEL_SIZE", "tiny").strip()
WHISPER_CPU_THREADS = max(1, (os.cpu_count() or 4) // 2)
WHISPER_TIMEOUT_SECONDS = 40
EMOTION_TIMEOUT_SECONDS = 20
LLM_TIMEOUT_SECONDS = 60
TTS_ENABLED = os.getenv("SERENITY_TTS_ENABLED", "true").lower() == "true"
TTS_VOICE = os.getenv("SERENITY_TTS_VOICE", "en-GB-RyanNeural").strip()
ADMIN_DEFAULT_LIMIT = max(50, int(os.getenv("SERENITY_ADMIN_DEFAULT_LIMIT", "300")))
ADMIN_MAX_LIMIT = max(200, int(os.getenv("SERENITY_ADMIN_MAX_LIMIT", "3000")))
ADMIN_SUMMARY_TIMEOUT_SECONDS = max(4, int(os.getenv("SERENITY_ADMIN_SUMMARY_TIMEOUT_SECONDS", "14")))
ADMIN_SUMMARY_CACHE_TTL_SECONDS = max(10, int(os.getenv("SERENITY_ADMIN_SUMMARY_CACHE_TTL_SECONDS", "120")))
ADMIN_OVERVIEW_CACHE_TTL_SECONDS = max(5, int(os.getenv("SERENITY_ADMIN_OVERVIEW_CACHE_TTL_SECONDS", "20")))

EMOTION_LABELS = ["angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_ALIAS = {"surprised": "surprise", "fearful": "fear", "no face": "neutral"}
NEGATIVE_EMOTIONS = {"angry", "disgust", "fear", "sad"}
SENTENCE_BOUNDARY_REGEX = re.compile(r"(?<=[.!?])\s+")
DISTRESS_SIGNAL_REGEX = re.compile(
    r"\b(hopeless|worthless|overwhelmed|panic|can't cope|cannot cope|self[- ]?harm|suicid|hurt myself|end my life)\b",
    re.IGNORECASE,
)

whisper_init_lock = threading.Lock()
cloud_llm_init_lock = threading.Lock()

# --- Pydantic Models ---
class AuthRequest(BaseModel):
    username: str
    password: str

class AuthResponse(BaseModel):
    message: str
    username: str

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

# --- DB Dependency ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- App Lifespan ---
@contextlib.asynccontextmanager
async def serenity_lifespan(fastapi_app: FastAPI):
    fastapi_app.state.cloud_llm_client = None
    fastapi_app.state.whisper_model = None
    fastapi_app.state.face_runtime = None
    fastapi_app.state.speech_runtime = None
    fastapi_app.state.admin_summary_cache = {"key": "", "summary": "", "expires_at": 0.0}
    fastapi_app.state.admin_overview_cache = {}
    fastapi_app.state.whisper_device_in_use = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
    yield
    client = getattr(fastapi_app.state, "cloud_llm_client", None)
    if client:
        with contextlib.suppress(Exception): client.close()

app = FastAPI(title="SERENITY API", lifespan=serenity_lifespan)
models.Base.metadata.create_all(bind=engine)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Helpers ---
def _state_get(key: str, default=None): return getattr(app.state, key, default)
def _state_set(key: str, value): setattr(app.state, key, value)

def _dedupe_errors(errors: List[str]) -> List[str]:
    return list(dict.fromkeys([str(err) for err in errors if str(err).strip()]))

def _admin_overview_cache_get(key: str) -> Optional[Dict[str, Any]]:
    now = time.time()
    cache = _state_get("admin_overview_cache", {}) or {}
    stale_keys = [k for k, v in cache.items() if float(v.get("expires_at") or 0.0) <= now]
    for stale_key in stale_keys:
        cache.pop(stale_key, None)

    item = cache.get(key)
    if not item:
        return None
    if float(item.get("expires_at") or 0.0) <= now:
        cache.pop(key, None)
        return None
    return item.get("payload")

def _admin_overview_cache_set(key: str, payload: Dict[str, Any]) -> None:
    cache = _state_get("admin_overview_cache", {}) or {}
    cache[key] = {
        "payload": payload,
        "expires_at": time.time() + ADMIN_OVERVIEW_CACHE_TTL_SECONDS,
    }
    if len(cache) > 64:
        oldest_key = min(cache.keys(), key=lambda k: float(cache[k].get("expires_at") or 0.0))
        cache.pop(oldest_key, None)
    _state_set("admin_overview_cache", cache)

def _invalidate_admin_overview_cache(username: Optional[str]) -> None:
    user_key = str(username or "").strip()
    if not user_key:
        return
    cache = _state_get("admin_overview_cache", {}) or {}
    dead_keys = [key for key in cache.keys() if key.startswith(f"{user_key}:")]
    for key in dead_keys:
        cache.pop(key, None)
    _state_set("admin_overview_cache", cache)

async def _ensure_cloud_llm_client():
    if _state_get("cloud_llm_client"): return _state_get("cloud_llm_client")
    with cloud_llm_init_lock:
        if not _state_get("cloud_llm_client"):
            try: _state_set("cloud_llm_client", CloudLLMClient())
            except CloudLLMError: return None
    return _state_get("cloud_llm_client")

def _normalize_emotion_label(emotion: str) -> str:
    val = str(emotion or "neutral").strip().lower()
    return EMOTION_ALIAS.get(val, val)

def _risk_label(score: int) -> str:
    if score >= 6: return "elevated"
    if score >= 3: return "monitor"
    return "stable"

def _severity_points(severity: str) -> int:
    sev = str(severity or "").strip().lower()
    if sev in {"severe", "very severe", "extremely severe", "elevated", "high"}: return 3
    if sev in {"moderate", "moderately severe"}: return 2
    if sev in {"mild", "minimal"}: return 1
    return 0

def _score_trend(values: List[int]) -> str:
    if len(values) < 2: return "insufficient_data"
    delta = int(values[0]) - int(values[1])
    if delta >= 3: return "worsening"
    if delta <= -3: return "improving"
    return "stable"

def _engagement_band(score: int) -> str:
    if score >= 70: return "high"
    if score >= 35: return "moderate"
    return "low"

def _fallback_admin_summary(snapshot: Dict[str, Any]) -> str:
    username = str(snapshot.get("username") or "unknown")
    risk = snapshot.get("risk") or {}
    screening = snapshot.get("screening") or {}
    emotion = snapshot.get("emotion") or {}
    engagement = snapshot.get("engagement") or {}
    follow_up = snapshot.get("follow_up") or {}

    score_pairs = ", ".join([f"{name} {value}" for name, value in (screening.get("latest_scores") or {}).items()]) or "no recent screenings"
    trend_pairs = ", ".join([f"{name}: {value}" for name, value in (screening.get("trends") or {}).items()]) or "insufficient_data"
    flags = ", ".join(risk.get("active_flags") or []) or "none"

    return "\n".join([
        f"- Client status: {username} is currently in {risk.get('level', 'stable')} risk band (score {risk.get('score', 0)}).",
        f"- Affective pattern: dominant emotion is {emotion.get('dominant_emotion', 'neutral')} with negative-affect ratio {emotion.get('negative_ratio', 0)} and {risk.get('distress_signal_count', 0)} distress-language signal(s).",
        f"- Measurement-based screening: latest scores {score_pairs}; trend review {trend_pairs}.",
        f"- Risk formulation: active screening flags are {flags}; engagement level is {engagement.get('level', 'low')} (score {engagement.get('score', 0)}).",
        f"- Immediate follow-up focus: {follow_up.get('primary_priority', 'continue routine supportive follow-up and reassessment.')}",
        f"- Monitoring cadence: {follow_up.get('cadence', 'weekly check-ins with repeated screening as clinically appropriate.')}",
    ])

def _build_admin_summary_prompt(snapshot: Dict[str, Any]) -> str:
    compact = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
    return (
        "You are SERENITY, a professional AI psychologist writing a single-client clinical progress report. "
        "Use evidence-based framing from measurement-based care, structured risk formulation, and trauma-informed communication. "
        "Output plain text only in exactly 6 concise bullet lines starting with '- '. "
        "No emojis, no hashtags, no roleplay, no markdown headers, and no diagnosis claims. "
        "Line 1: current risk status. "
        "Line 2: emotional or behavioral pattern. "
        "Line 3: PHQ-9, GAD-7, PCL-5 interpretation and trend. "
        "Line 4: risk factors and protective factors. "
        "Line 5: immediate follow-up priorities. "
        "Line 6: monitoring cadence and measurable targets. "
        f"Data:{compact}"
    )

async def _generate_admin_summary(snapshot: Dict[str, Any]) -> Tuple[str, str]:
    key = json.dumps(snapshot, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    now = time.time()
    cache = _state_get("admin_summary_cache", {}) or {}
    if cache.get("key") == key and float(cache.get("expires_at") or 0.0) > now:
        return str(cache.get("summary") or ""), str(cache.get("source") or "cloud_llm_cache")

    client = await _ensure_cloud_llm_client()
    fallback_summary = _fallback_admin_summary(snapshot)
    fallback_ttl = ADMIN_SUMMARY_CACHE_TTL_SECONDS
    if not client:
        _state_set("admin_summary_cache", {
            "key": key,
            "summary": fallback_summary,
            "source": "fallback",
            "expires_at": now + fallback_ttl,
        })
        return fallback_summary, "fallback"

    if client:
        try:
            prompt = _build_admin_summary_prompt(snapshot)
            summary = await asyncio.wait_for(
                run_in_threadpool(client.ask_serenity, prompt),
                timeout=ADMIN_SUMMARY_TIMEOUT_SECONDS,
            )
            raw = str(summary or "").strip()
            lines = [line.strip() for line in raw.splitlines() if line.strip()]
            if len(lines) <= 1:
                lines = [line.strip() for line in re.split(r"(?<=[.!?])\s+", raw) if line.strip()]

            normalized = []
            for line in lines:
                cleaned_line = re.sub(r"\s+", " ", line).lstrip("-*• ").strip()
                if cleaned_line:
                    normalized.append(f"- {cleaned_line}")
                if len(normalized) >= 6:
                    break

            cleaned = "\n".join(normalized).strip()
            if cleaned:
                _state_set(
                    "admin_summary_cache",
                    {
                        "key": key,
                        "summary": cleaned,
                        "source": "cloud_llm",
                        "expires_at": now + ADMIN_SUMMARY_CACHE_TTL_SECONDS,
                    },
                )
                return cleaned, "cloud_llm"
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}" if str(exc).strip() else type(exc).__name__
            LOGGER.info("Admin summary cloud generation unavailable; using fallback (%s)", detail)

    _state_set("admin_summary_cache", {
        "key": key,
        "summary": fallback_summary,
        "source": "fallback",
        "expires_at": now + fallback_ttl,
    })
    return fallback_summary, "fallback"

@contextlib.asynccontextmanager
async def handle_temp_audio(audio_bytes: Optional[bytes], filename: str = ""):
    if not audio_bytes:
        yield None
        return
    
    suffix = ".wav"
    if filename.endswith(".webm"): suffix = ".webm"
    elif filename.endswith(".mp3"): suffix = ".mp3"
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(audio_bytes)
        temp_path = temp_audio.name
        
    try:
        yield temp_path
    finally:
        if os.path.exists(temp_path):
            with contextlib.suppress(OSError):
                os.remove(temp_path)

def _transcribe_with_whisper(audio_path: str) -> tuple[str, Optional[str]]:
    with whisper_init_lock:
        if _state_get("whisper_model") is None:
            device = _state_get("whisper_device_in_use", "cpu")
            if WhisperModel:
                _state_set("whisper_model", WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type="int8", cpu_threads=WHISPER_CPU_THREADS))
                _state_set("whisper_backend", "faster-whisper")
            elif openai_whisper:
                _state_set("whisper_model", openai_whisper.load_model(WHISPER_MODEL_SIZE, device=device))
                _state_set("whisper_backend", "openai-whisper")
            else:
                return "", "No STT backend installed"

    model = _state_get("whisper_model")
    backend = _state_get("whisper_backend")
    
    try:
        if backend == "faster-whisper":
            segments, _ = model.transcribe(audio_path, language="en", beam_size=1)
            return " ".join([s.text.strip() for s in segments if s.text.strip()]), None
        else:
            return str(model.transcribe(audio_path, fp16=False, language="en").get("text", "")).strip(), None
    except Exception as e:
        return "", f"Transcription failed: {e}"

async def _generate_tts_base64(text: str) -> tuple[Optional[str], Optional[str]]:
    if not text or not TTS_ENABLED or not edge_tts: return None, None
    audio_path = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}.mp3")
    try:
        comm = edge_tts.Communicate(text=text, voice=TTS_VOICE)
        await asyncio.wait_for(comm.save(audio_path), timeout=30)
        with open(audio_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8"), None
    except Exception as e:
        return None, f"TTS failed: {e}"
    finally:
        if os.path.exists(audio_path):
            with contextlib.suppress(OSError): os.remove(audio_path)

async def _ask_serenity_text(user_text: str, errors: List[str]) -> str:
    client = await _ensure_cloud_llm_client()
    if not client:
        errors.append("LLM client offline")
        return ""
    try:
        return await asyncio.wait_for(run_in_threadpool(client.ask_serenity, user_text), timeout=LLM_TIMEOUT_SECONDS)
    except Exception as exc:
        errors.append(f"LLM Error: {exc}")
        return ""

async def _persist_turn_safe(
    db: Session,
    username: str,
    user_text: str,
    llm_response: str,
    dominant_emotion: str,
    speech_emotion: str,
    face_emotion: str,
    errors: Optional[List[str]] = None,
) -> None:
    try:
        await run_in_threadpool(
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
    except Exception as exc:
        if errors is not None:
            errors.append(f"DB Error: {exc}")

# --- Shared Perception Pipeline ---
async def _run_perception_tasks(temp_audio_path: Optional[str], image_data: Optional[str]) -> Dict[str, Any]:
    errors = []
    tasks = []
    
    if temp_audio_path:
        tasks.append(("transcribe", asyncio.create_task(run_in_threadpool(_transcribe_with_whisper, temp_audio_path))))
        tasks.append(("speech", asyncio.create_task(run_in_threadpool(predict_audio_emotion, temp_audio_path, _state_get("speech_runtime")))))
    if image_data:
        tasks.append(("face", asyncio.create_task(run_in_threadpool(analyze_face, image_data, _state_get("face_runtime")))))

    results = {"transcription": "", "speech_emotion": "Neutral", "face_emotion": "Neutral", "speech_conf": 0.0, "face_conf": 0.0}
    
    for name, task in tasks:
        try:
            res = await asyncio.wait_for(task, timeout=WHISPER_TIMEOUT_SECONDS if name=="transcribe" else EMOTION_TIMEOUT_SECONDS)
            if name == "transcribe":
                results["transcription"], err = res
                if err: errors.append(err)
            elif name == "speech":
                results["speech_emotion"] = res.get("emotion", "Neutral")
                results["speech_conf"] = res.get("confidence", 0.0)
            elif name == "face":
                results["face_emotion"] = res.get("emotion", "Neutral")
                results["face_conf"] = res.get("confidence", 0.0)
        except Exception as e:
            errors.append(f"{name} task failed: {e}")

    def _probs(emo, conf):
        p = {l: (1.0 - conf/100)/7 for l in EMOTION_LABELS}
        p[_normalize_emotion_label(emo)] = conf/100
        return p

    s_probs = _probs(results["speech_emotion"], results["speech_conf"])
    f_probs = _probs(results["face_emotion"], results["face_conf"]) if image_data else s_probs
    
    fused = {l: (s_probs[l] + f_probs[l])/2.0 for l in EMOTION_LABELS} if temp_audio_path and image_data else (s_probs if temp_audio_path else f_probs)
    dominant = max(fused.items(), key=lambda x: x[1])[0].title() if (temp_audio_path or image_data) else "Neutral"
    
    return {**results, "dominant_emotion": dominant, "errors": errors}

# --- Shared LLM & TTS Streaming Generator ---
async def _stream_llm_and_tts(client, user_text: str):
    """Centralized pipeline for chunking text and low-overhead TTS event generation."""
    buffer, seq = "", 0
    tts_out_queue: asyncio.Queue = asyncio.Queue()
    tts_in_queue: asyncio.Queue = asyncio.Queue(maxsize=6)

    async def tts_worker_loop():
        while True:
            item = await tts_in_queue.get()
            if item is None:
                return
            text, sequence = item
            aud, _ = await _generate_tts_base64(text)
            if aud:
                await tts_out_queue.put({"type": "assistant_sentence_tts", "text": text, "sequence": sequence, "audio_base64": aud})

    tts_worker_task = asyncio.create_task(tts_worker_loop()) if (TTS_ENABLED and edge_tts) else None

    async def emit_sentence(sentence: str):
        nonlocal seq
        seq += 1
        yield {"type": "assistant_sentence", "text": sentence, "sequence": seq}
        if tts_worker_task:
            with contextlib.suppress(asyncio.QueueFull):
                tts_in_queue.put_nowait((sentence, seq))

    generator = client.stream_serenity(user_text)

    # Safely fetch next chunk without StopIteration crashing the threadpool
    def _fetch_next():
        try:
            return next(generator), False
        except StopIteration:
            return None, True
        except Exception as e:
            return None, e

    try:
        while True:
            chunk, status = await run_in_threadpool(_fetch_next)
            if status is True:  # Generator finished successfully
                break
            if isinstance(status, Exception):
                yield {"type": "error", "message": f"LLM Stream Error: {status}"}
                break
            
            yield {"type": "assistant_delta", "delta": chunk}
            
            buffer += chunk
            sentences = SENTENCE_BOUNDARY_REGEX.split(buffer)
            if len(sentences) > 1:
                for s in sentences[:-1]:
                    s = s.strip()
                    if s:
                        async for event in emit_sentence(s):
                            yield event
                buffer = sentences[-1]
                
            # Yield any ready TTS events immediately
            while not tts_out_queue.empty():
                yield tts_out_queue.get_nowait()

        # Flush remaining buffer into a final sentence
        if buffer.strip():
            async for event in emit_sentence(buffer.strip()):
                yield event

        # Let the single TTS worker drain before closing.
        if tts_worker_task:
            await tts_in_queue.put(None)
            await tts_worker_task
            while not tts_out_queue.empty():
                yield tts_out_queue.get_nowait()

    except Exception as e:
        yield {"type": "error", "message": f"Pipeline Error: {e}"}
    finally:
        if tts_worker_task and not tts_worker_task.done():
            with contextlib.suppress(Exception):
                await tts_in_queue.put(None)
                await tts_worker_task

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
):
    emotion_payload = {
        "type": "emotion",
        "dominant_emotion": dominant_emotion,
        "speech_emotion": speech_emotion,
        "face_emotion": face_emotion,
    }
    user_text_payload = {"type": "user_text", "text": user_text, "source": source}
    if emotion_first:
        yield _to_event(emotion_payload)
        yield _to_event(user_text_payload)
    else:
        yield _to_event(user_text_payload)
        yield _to_event(emotion_payload)

    llm_res = ""
    client = await _ensure_cloud_llm_client()
    if not client:
        yield _to_event({"type": "error", "message": "LLM client offline"})
    else:
        async for event in _stream_llm_and_tts(client, user_text):
            if event["type"] == "assistant_delta":
                llm_res += event["delta"]
            event["text"] = llm_res
            yield _to_event(event)

    stream_errors: List[str] = []
    await _persist_turn_safe(db, username, user_text, llm_res, dominant_emotion, speech_emotion, face_emotion, stream_errors)
    for err in stream_errors:
        yield _to_event({"type": "error", "message": err})

    yield _to_event({"type": "final", "llm_response": llm_res, "transcription": user_text, "dominant_emotion": dominant_emotion})

# --- Endpoints ---
@app.post("/register", response_model=AuthResponse)
async def register(payload: AuthRequest, db: Session = Depends(get_db)):
    existing = await run_in_threadpool(lambda: db.query(models.User).filter(models.User.username == payload.username).first())
    if existing: raise HTTPException(400, "Username exists")
    hashed = bcrypt.hashpw(payload.password.encode('utf-8'), bcrypt.gensalt())
    new_user = models.User(username=payload.username, password=hashed.decode('utf-8'))
    def _save():
        db.add(new_user)
        db.commit()
    await run_in_threadpool(_save)
    return AuthResponse(message="Registration successful", username=new_user.username)

@app.post("/login", response_model=AuthResponse)
async def login(payload: AuthRequest, db: Session = Depends(get_db)):
    user = await run_in_threadpool(lambda: db.query(models.User).filter(models.User.username == payload.username).first())
    if not user or not bcrypt.checkpw(payload.password.encode('utf-8'), user.password.encode('utf-8')):
        raise HTTPException(401, "Invalid credentials")
    return AuthResponse(message="Login successful", username=user.username)

@app.get("/api/questionnaires/templates")
async def get_templates(types: Optional[str] = None):
    return {"available_types": list(QUESTIONNAIRE_DEFINITIONS.keys()), "questionnaires": questionnaire_templates(types.split(",") if types else None)}

@app.post("/api/questionnaires/submit")
async def submit_questionnaire(payload: QuestionnaireSubmitRequest, db: Session = Depends(get_db)):
    q_type = normalize_questionnaire_type(payload.questionnaire_type)
    if not q_type: raise HTTPException(400, "Invalid type")
    ans = [max(0, min(4 if q_type=="PCL-5" else 3, int(x))) for x in payload.answers]
    score, sev = score_questionnaire(q_type, ans)
    dt = None
    if payload.submitted_at:
        try: dt = datetime.fromisoformat(payload.submitted_at.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError: pass
    rec = await run_in_threadpool(persist_questionnaire_result, db, payload.username, q_type, ans, score, sev, dt)
    _invalidate_admin_overview_cache(payload.username)
    return {"message": "Saved", "result": {"id": rec.id, "total_score": score, "severity": sev}}

@app.get("/api/questionnaires/history")
async def q_history(username: str, limit: int = 30, db: Session = Depends(get_db)):
    if not username: raise HTTPException(400, "Username required")
    res = await run_in_threadpool(fetch_questionnaire_results, db, username, limit=max(1, limit))
    return {"username": username, "results": res}

@app.get("/api/admin/overview")
async def admin_overview(username: str, limit: int = 300, include_answers: bool = False, db: Session = Depends(get_db)):
    user_key = str(username or "").strip()
    if not user_key:
        raise HTTPException(400, "Username required")

    limit = max(20, min(int(limit or ADMIN_DEFAULT_LIMIT), ADMIN_MAX_LIMIT))
    quiz_limit = max(30, min(limit * 2, ADMIN_MAX_LIMIT * 2))
    cache_key = f"{user_key}:{limit}:{1 if include_answers else 0}"
    cached_payload = _admin_overview_cache_get(cache_key)
    if cached_payload is not None:
        return cached_payload

    chats, sessions, quizzes, total_turns, total_sessions, total_quizzes, user_exists = await run_in_threadpool(lambda: (
        fetch_recent_turn_summaries(db, limit=limit, text_limit=420, username=user_key),
        fetch_recent_sessions_with_emotions(db, limit=limit, conversation_limit=420, username=user_key),
        fetch_questionnaire_results(db, username=user_key, limit=quiz_limit, include_answers=include_answers),
        db.query(func.count(models.ConversationTurn.id))
        .join(models.User, models.ConversationTurn.user_id == models.User.id)
        .filter(models.User.username == user_key)
        .scalar() or 0,
        db.query(func.count(models.Session.id))
        .join(models.User, models.Session.user_id == models.User.id)
        .filter(models.User.username == user_key)
        .scalar() or 0,
        db.query(func.count(models.QuestionnaireResult.id))
        .join(models.User, models.QuestionnaireResult.user_id == models.User.id)
        .filter(models.User.username == user_key)
        .scalar() or 0,
        db.query(models.User.id).filter(models.User.username == user_key).first() is not None,
    ))

    if not user_exists:
        raise HTTPException(404, "User not found")

    last_seen = None
    latest_assistant_note = ""
    emotion_events = 0
    emotion_counts: Dict[str, int] = {}
    negative_turns = 0
    distress_signal_count = 0

    for row in chats:
        ts = row.get("timestamp")
        if ts and (not last_seen or str(ts) > str(last_seen)):
            last_seen = ts

        emotion = _normalize_emotion_label(row.get("dominant_emotion"))
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        if emotion in NEGATIVE_EMOTIONS:
            negative_turns += 1

        if DISTRESS_SIGNAL_REGEX.search(str(row.get("user_text") or "")):
            distress_signal_count += 1

        if not latest_assistant_note:
            latest_assistant_note = str(row.get("assistant_text") or "").strip()

    for row in sessions:
        ts = row.get("timestamp")
        if ts and (not last_seen or str(ts) > str(last_seen)):
            last_seen = ts
        emotion_events += len(row.get("emotions") or [])

    latest_scores: Dict[str, int] = {}
    latest_severity: Dict[str, str] = {}
    score_history: Dict[str, List[int]] = {}
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
            last_seen = ts

    top_emotions = sorted(
        [{"emotion": name, "count": count} for name, count in emotion_counts.items()],
        key=lambda row: row["count"],
        reverse=True,
    )
    dominant_emotion = top_emotions[0]["emotion"] if top_emotions else "neutral"

    screening_trends = {name: _score_trend(scores[:3]) for name, scores in score_history.items()}
    flags = questionnaire_clinical_flags(latest_scores) if latest_scores else {}
    active_flags = [name for name, enabled in flags.items() if bool(enabled)]

    negative_ratio = round(negative_turns / max(1, len(chats)), 3)
    severity_points = max([_severity_points(value) for value in latest_severity.values()] or [0])
    risk_score = (len(active_flags) * 2) + severity_points + (2 if distress_signal_count > 0 else 0) + (1 if negative_ratio >= 0.55 else 0)
    risk_level = _risk_label(risk_score)

    engagement_score = min(100, int(total_turns * 2 + total_sessions * 4 + total_quizzes * 8))
    engagement_level = _engagement_band(engagement_score)
    primary_priority = (
        "Conduct short-interval follow-up with focused safety check and coping-plan review."
        if risk_level == "elevated"
        else "Maintain weekly structured follow-up with measurement-based symptom tracking."
        if risk_level == "monitor"
        else "Continue routine supportive follow-up with periodic symptom screening."
    )
    cadence = (
        "24-72 hour follow-up until risk indicators decrease."
        if risk_level == "elevated"
        else "Weekly review of symptoms and coping adherence."
        if risk_level == "monitor"
        else "Biweekly to monthly review with early escalation criteria."
    )

    profile = {
        "username": user_key,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "active_flags": active_flags,
        "dominant_emotion": dominant_emotion,
        "turn_count": int(total_turns),
        "session_count": int(total_sessions),
        "questionnaire_count": int(total_quizzes),
        "last_seen": last_seen,
        "latest_scores": latest_scores,
        "latest_severity": latest_severity,
        "screening_trends": screening_trends,
        "emotion_mix": top_emotions[:3],
        "negative_emotion_ratio": negative_ratio,
        "distress_signal_count": distress_signal_count,
        "engagement_score": engagement_score,
        "engagement_level": engagement_level,
        "latest_assistant_note": latest_assistant_note,
    }

    summary_snapshot = {
        "username": user_key,
        "risk": {
            "level": risk_level,
            "score": risk_score,
            "active_flags": active_flags,
            "distress_signal_count": distress_signal_count,
        },
        "emotion": {
            "dominant_emotion": dominant_emotion,
            "negative_ratio": negative_ratio,
            "distribution": top_emotions[:4],
        },
        "screening": {
            "latest_scores": latest_scores,
            "latest_severity": latest_severity,
            "trends": screening_trends,
        },
        "engagement": {
            "score": engagement_score,
            "level": engagement_level,
        },
        "follow_up": {
            "primary_priority": primary_priority,
            "cadence": cadence,
        },
        "volume": {
            "conversation_turns": int(total_turns),
            "sessions": int(total_sessions),
            "questionnaire_entries": int(total_quizzes),
            "emotion_events": int(emotion_events),
        },
    }
    summary_text, summary_source = await _generate_admin_summary(summary_snapshot)

    metrics = [
        {"id": "turns", "label": "Conversation Turns", "value": int(total_turns), "description": "Recorded user-assistant exchanges"},
        {"id": "sessions", "label": "Sessions", "value": int(total_sessions), "description": "Legacy session records for this user"},
        {"id": "emotion_events", "label": "Emotion Events", "value": int(emotion_events), "description": "Session-level emotion observations"},
        {"id": "questionnaire_entries", "label": "Screening Entries", "value": int(total_quizzes), "description": "PHQ-9, GAD-7, and PCL-5 submissions"},
        {"id": "risk_score", "label": "Risk Score", "value": int(risk_score), "description": "Structured score from screening, affect, and distress signals"},
        {"id": "distress_signals", "label": "Distress Signals", "value": int(distress_signal_count), "description": "Keyword-based distress indicators in recent user messages"},
    ]

    flagged_users = [{
        "username": user_key,
        "flags": flags,
        "scores": latest_scores,
        "risk_level": risk_level,
    }] if active_flags else []

    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "summary": summary_text,
        "summary_source": summary_source,
        "summary_snapshot": summary_snapshot,
        "metrics": metrics,
        "profile": profile,
        "clinical_parameters": {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "active_flags": active_flags,
            "distress_signal_count": distress_signal_count,
            "negative_emotion_ratio": negative_ratio,
            "engagement_level": engagement_level,
            "screening_trends": screening_trends,
        },
        "top_emotions": top_emotions,
        "user_profiles": [profile],
        "chats": chats,
        "sessions": sessions,
        "questionnaire_results": quizzes,
        "flagged_users": flagged_users,
    }
    _admin_overview_cache_set(cache_key, payload)
    return payload

@app.post("/api/interact", response_model=InteractResponse)
async def interact(username: str = Form(...), image: Optional[str] = Form(None), file: Optional[UploadFile] = File(None), user_message: Optional[str] = Form(None), db: Session = Depends(get_db)):
    if not file: raise HTTPException(400, "Microphone input required.")
    audio_bytes, filename = await file.read(), file.filename or ""
    async with handle_temp_audio(audio_bytes, filename) as audio_path:
        perc = await _run_perception_tasks(audio_path, image)
    user_text = (perc["transcription"] or user_message or "").strip()
    if not user_text:
        return InteractResponse(dominant_emotion=perc["dominant_emotion"], speech_emotion=perc["speech_emotion"], face_emotion=perc["face_emotion"], transcription="", llm_response="", errors=perc["errors"] + ["No speech detected."])
    errors = list(perc["errors"])
    llm_res = await _ask_serenity_text(user_text, errors)
    await _persist_turn_safe(db, username, user_text, llm_res, perc["dominant_emotion"], perc["speech_emotion"], perc["face_emotion"], errors)
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
                yield _to_event({"type": "final", "llm_response": "", "transcription": "", "dominant_emotion": perc["dominant_emotion"]})
                return

            async for payload in _stream_chat_events(
                db=db,
                username=username,
                user_text=user_text,
                dominant_emotion=perc["dominant_emotion"],
                speech_emotion=perc["speech_emotion"],
                face_emotion=perc["face_emotion"],
                source="voice",
                emotion_first=True,
            ):
                yield payload

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")

@app.post("/api/chat")
async def chat(payload: ChatRequest, db: Session = Depends(get_db)):
    if not payload.message: raise HTTPException(400, "Message empty.")
    errors: List[str] = []
    llm_res = await _ask_serenity_text(payload.message, errors)
    await _persist_turn_safe(db, payload.username, payload.message, llm_res, "Neutral", "Neutral", "Neutral", errors)
    tts_base64, tts_err = await _generate_tts_base64(llm_res)
    if tts_err: errors.append(tts_err)
    return InteractResponse(dominant_emotion="Neutral", speech_emotion="Neutral", face_emotion="Neutral", transcription=payload.message, llm_response=llm_res, tts_audio_base64=tts_base64, errors=_dedupe_errors(errors))

@app.post("/api/chat/stream")
async def chat_stream(payload: ChatRequest, db: Session = Depends(get_db)):
    if not payload.message: raise HTTPException(400, "Message empty.")

    return StreamingResponse(
        _stream_chat_events(
            db=db,
            username=payload.username,
            user_text=payload.message,
            dominant_emotion="Neutral",
            speech_emotion="Neutral",
            face_emotion="Neutral",
            source="text",
            emotion_first=False,
        ),
        media_type="application/x-ndjson",
    )