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

try: import torch
except ImportError: torch = None

from backend.audio_core import initialize_audio_runtime, predict_audio_emotion
from backend.emotion_core import initialize_face_runtime, analyze_face
from backend.database import SessionLocal, engine, fetch_questionnaire_results, fetch_recent_sessions_with_emotions, fetch_recent_turn_summaries, persist_questionnaire_result, persist_turn
from backend.cloud_llm_core import CloudLLMClient, CloudLLMError
from backend.questionnaires_data import QUESTIONNAIRE_DEFINITIONS, normalize_questionnaire_type, questionnaire_clinical_flags, questionnaire_templates, score_questionnaire
import backend.models as models

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
ADMIN_DEFAULT_LIMIT = _env_int("SERENITY_ADMIN_DEFAULT_LIMIT", 300, minimum=50)
ADMIN_MAX_LIMIT = _env_int("SERENITY_ADMIN_MAX_LIMIT", 3000, minimum=200)
ADMIN_OVERVIEW_CACHE_TTL_SECONDS = _env_float("SERENITY_ADMIN_OVERVIEW_CACHE_TTL_SECONDS", 20.0, minimum=5.0)
ADMIN_SUMMARY_CACHE_TTL_SECONDS = _env_float("SERENITY_ADMIN_SUMMARY_CACHE_TTL_SECONDS", 120.0, minimum=10.0)
ADMIN_SUMMARY_TIMEOUT_SECONDS = _env_float("SERENITY_ADMIN_SUMMARY_TIMEOUT_SECONDS", 10.0, minimum=3.0)
PREWARM_MODELS = os.getenv("SERENITY_PREWARM_MODELS", "true").lower() == "true"
PREWARM_WHISPER = os.getenv("SERENITY_PREWARM_WHISPER", "false").lower() == "true"

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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def _state_get(k: str, d=None): return getattr(app.state, k, d)
def _state_set(k: str, v): setattr(app.state, k, v)
def _dedupe_errors(errors: List[str]) -> List[str]: return list(dict.fromkeys([str(e) for e in errors if str(e).strip()]))
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
def _normalize_summary_lines(text: str, max_lines: int = 6) -> str:
    if not text: return ""
    raw = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    if len(raw) <= 1:
        raw = [seg.strip() for seg in re.split(r"(?<=[.!?])\s+", str(text)) if seg.strip()]
    out = []
    for line in raw:
        cleaned = re.sub(r"\s+", " ", line).lstrip("-*• ").strip()
        if cleaned:
            out.append(f"- {cleaned}")
        if len(out) >= max_lines:
            break
    return "\n".join(out)
def _build_admin_summary_prompt(snapshot: Dict[str, Any]) -> str:
    compact = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
    return (
        "You are SERENITY, a professional AI psychologist writing a concise client progress note. "
        "Output plain text only in exactly 6 bullet lines beginning with '- '. "
        "No diagnosis claims, no emojis, no markdown headers, and no role labels. "
        "Line 1: current risk status with risk score interpretation. "
        "Line 2: dominant affective pattern and emotional context. "
        "Line 3: questionnaire interpretation (PHQ-9/GAD-7/PCL-5) and trend direction. "
        "Line 4: salient risk and protective factors. "
        "Line 5: immediate follow-up priorities with trauma-informed tone. "
        "Line 6: monitoring cadence and measurable follow-up targets. "
        f"Data:{compact}"
    )
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
def _build_admin_metrics(turns: int, sessions: int, emotion_events: int, quizzes: int, risk_score: int, distress_signals: int) -> List[Dict[str, Any]]:
    return [
        {"id": "turns", "label": "Conversation Turns", "value": int(turns), "description": "Recent therapeutic dialogue exchanges."},
        {"id": "sessions", "label": "Sessions", "value": int(sessions), "description": "Stored session records for the current client."},
        {"id": "emotion_events", "label": "Emotion Events", "value": int(emotion_events), "description": "Captured emotion observations across sessions."},
        {"id": "questionnaire_entries", "label": "Screening Entries", "value": int(quizzes), "description": "PHQ-9, GAD-7, and PCL-5 submissions analyzed."},
        {"id": "risk_score", "label": "Risk Score", "value": int(risk_score), "description": "Composite score from screening severity, distress language, and affective risk."},
        {"id": "distress_signals", "label": "Distress Signals", "value": int(distress_signals), "description": "Keyword-based acute distress cues in recent user messages."},
    ]
def _base_profile(username: str) -> Dict[str, Any]:
    return {
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
def _empty_admin_payload(username: str) -> Dict[str, Any]:
    profile = _base_profile(username)
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "summary": "- Client status: no sufficient clinical data yet for formal risk stratification.\n- Affective pattern: insufficient interaction data for dominant emotion analysis.\n- Measurement-based screening: no recent PHQ-9, GAD-7, or PCL-5 results available.\n- Risk formulation: no active screening flags identified from available records.\n- Immediate follow-up focus: establish baseline screening and supportive check-in.\n- Monitoring cadence: weekly follow-up until baseline metrics are available.",
        "summary_source": "fallback",
        "summary_snapshot": profile,
        "metrics": _build_admin_metrics(0, 0, 0, 0, 0, 0),
        "top_emotions": [],
        "chats": [],
        "sessions": [],
        "questionnaire_results": [],
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

async def _generate_admin_summary(snapshot: Dict[str, Any]) -> Tuple[str, str]:
    cache_key = json.dumps(snapshot, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    if cached := _admin_summary_cache_get(cache_key):
        return cached

    fallback = _fallback_admin_summary(snapshot)
    summary_text, source = fallback, "fallback"
    client = await _ensure_cloud_llm_client()

    if client:
        try:
            prompt = _build_admin_summary_prompt(snapshot)
            candidate = await asyncio.wait_for(client.ask_serenity(prompt, timeout=ADMIN_SUMMARY_TIMEOUT_SECONDS), timeout=ADMIN_SUMMARY_TIMEOUT_SECONDS + 1.5)
            normalized = _normalize_summary_lines(candidate)
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
    if not text or not TTS_ENABLED or not edge_tts: return None, None
    audio_path = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}.mp3")
    try:
        comm = edge_tts.Communicate(text=text, voice=TTS_VOICE)
        await asyncio.wait_for(comm.save(audio_path), timeout=30)
        with open(audio_path, "rb") as f: return base64.b64encode(f.read()).decode("utf-8"), None
    except Exception as e: return None, f"TTS failed: {e}"
    finally:
        if os.path.exists(audio_path):
            with contextlib.suppress(OSError): os.remove(audio_path)

async def _run_perception_tasks(temp_audio_path: Optional[str], image_data: Optional[str]) -> Dict[str, Any]:
    async def run_task(name, func, *args):
        try: return name, await asyncio.wait_for(run_in_threadpool(func, *args), timeout=WHISPER_TIMEOUT_SECONDS if name=="transcribe" else EMOTION_TIMEOUT_SECONDS)
        except Exception as e: return name, e

    tasks = []
    if temp_audio_path:
        tasks.extend([run_task("transcribe", _transcribe_with_whisper, temp_audio_path), run_task("speech", predict_audio_emotion, temp_audio_path, _state_get("speech_runtime"))])
    if image_data: tasks.append(run_task("face", analyze_face, image_data, _state_get("face_runtime")))

    results = {"transcription": "", "speech_emotion": "Neutral", "face_emotion": "Neutral", "speech_conf": 0.0, "face_conf": 0.0}
    errors = []

    if tasks:
        for name, val in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(val, Exception): errors.append(f"{name} task failed: {val}")
            elif name == "transcribe":
                results["transcription"], err = val
                if err: errors.append(err)
            elif name in ("speech", "face"):
                results[f"{name}_emotion"] = val.get("emotion", "Neutral")
                results[f"{name}_conf"] = val.get("confidence", 0.0)

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

def _to_event(payload: dict) -> str: return json.dumps(payload, ensure_ascii=False) + "\n"

async def _stream_chat_events(db: Session, username: str, user_text: str, dominant_emotion: str, speech_emotion: str, face_emotion: str, source: str, emotion_first: bool):
    emo_payload = {"type": "emotion", "dominant_emotion": dominant_emotion, "speech_emotion": speech_emotion, "face_emotion": face_emotion}
    txt_payload = {"type": "user_text", "text": user_text, "source": source}

    if emotion_first: yield _to_event(emo_payload); yield _to_event(txt_payload)
    else: yield _to_event(txt_payload); yield _to_event(emo_payload)

    client = await _ensure_cloud_llm_client()
    if not client:
        yield _to_event({"type": "error", "message": "LLM client offline"})
        await _persist_turn_safe(db, username, user_text, "", dominant_emotion, speech_emotion, face_emotion)
        yield _to_event({
            "type": "final",
            "llm_response": "",
            "transcription": user_text,
            "dominant_emotion": dominant_emotion,
            "speech_emotion": speech_emotion,
            "face_emotion": face_emotion,
        })
        return

    output_queue: asyncio.Queue = asyncio.Queue(maxsize=96)
    tts_input_queue: asyncio.Queue = asyncio.Queue(maxsize=16)
    stream_tts = bool(TTS_ENABLED and edge_tts)

    async def fetch_text():
        buffer, seq = "", 0
        llm_chunks: List[str] = []
        llm_res = ""
        cutoff_hit = False
        try:
            async for chunk in client.stream_serenity(user_text):
                if chunk == "<CUTOFF>":
                    cutoff_hit = True
                    break

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
                if match := list(re.finditer(r'[.!?]', llm_res)):
                    llm_res = llm_res[:match[-1].end()].strip()
                await output_queue.put({"type": "assistant_replace", "text": llm_res})
            elif buffer.strip():
                seq += 1
                await output_queue.put({"type": "assistant_sentence", "text": buffer.strip(), "sequence": seq})
                if stream_tts: await tts_input_queue.put((buffer.strip(), seq))

        except Exception as exc:
            await output_queue.put({"type": "error", "message": f"LLM Stream Error: {exc}"})
            llm_res = "".join(llm_chunks).strip()
        finally:
            if stream_tts: await tts_input_queue.put(None)
            await output_queue.put({"type": "LLM_DONE", "final_res": llm_res})

    async def generate_audio():
        while True:
            if (item := await tts_input_queue.get()) is None: break
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
    if stream_tts: tasks.append(asyncio.create_task(generate_audio()))
    
    tasks_running = len(tasks)
    final_llm_res = ""

    while tasks_running > 0:
        event = await output_queue.get()
        if event["type"] == "LLM_DONE":
            final_llm_res = event["final_res"]
            tasks_running -= 1
        elif event["type"] == "TTS_DONE":
            tasks_running -= 1
        else:
            yield _to_event(event)

    db_errors: List[str] = []
    await _persist_turn_safe(db, username, user_text, final_llm_res, dominant_emotion, speech_emotion, face_emotion, db_errors)
    for err in db_errors:
        yield _to_event({"type": "error", "message": err})
    yield _to_event({
        "type": "final",
        "llm_response": final_llm_res,
        "transcription": user_text,
        "dominant_emotion": dominant_emotion,
        "speech_emotion": speech_emotion,
        "face_emotion": face_emotion,
    })

# --- Endpoints ---
@app.post("/register", response_model=AuthResponse)
async def register(payload: AuthRequest, db: Session = Depends(get_db)):
    if await run_in_threadpool(lambda: db.query(models.User).filter(models.User.username == payload.username).first()): raise HTTPException(400, "Exists")
    new_user = models.User(username=payload.username, password=bcrypt.hashpw(payload.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8'))
    await run_in_threadpool(lambda: (db.add(new_user), db.commit()))
    return AuthResponse(message="Success", username=new_user.username)

@app.post("/login", response_model=AuthResponse)
async def login(payload: AuthRequest, db: Session = Depends(get_db)):
    user = await run_in_threadpool(lambda: db.query(models.User).filter(models.User.username == payload.username).first())
    if not user or not bcrypt.checkpw(payload.password.encode('utf-8'), user.password.encode('utf-8')): raise HTTPException(401, "Invalid")
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
    _invalidate_admin_overview_cache(payload.username)
    return {"message": "Saved", "result": {"id": rec.id, "total_score": score, "severity": sev}}

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

    def fetch_admin_data():
        if not (user_row := db.query(models.User.id).filter(models.User.username == user_key).first()):
            return None
        user_id = int(user_row[0])
        return (
            fetch_recent_turn_summaries(db, limit=limit, text_limit=420, username=user_key),
            fetch_recent_sessions_with_emotions(db, limit=limit, conversation_limit=420, username=user_key),
            fetch_questionnaire_results(db, username=user_key, limit=quiz_limit, include_answers=include_answers),
            db.query(func.count(models.ConversationTurn.id)).filter(models.ConversationTurn.user_id == user_id).scalar() or 0,
            db.query(func.count(models.Session.id)).filter(models.Session.user_id == user_id).scalar() or 0,
            db.query(func.count(models.QuestionnaireResult.id)).filter(models.QuestionnaireResult.user_id == user_id).scalar() or 0,
        )

    if not (fetched := await run_in_threadpool(fetch_admin_data)):
        payload = _empty_admin_payload(user_key)
        _admin_overview_cache_set(cache_key, payload)
        return payload

    chats, sessions, quizzes, total_turns, total_sessions, total_quizzes = fetched
    if int(total_turns) + int(total_sessions) + int(total_quizzes) == 0:
        payload = _empty_admin_payload(user_key)
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
        },
    }

    summary_text, summary_source = await _generate_admin_summary(summary_snapshot)

    profile = _base_profile(user_key)
    profile.update({
        "last_seen": last_seen,
        "risk_level": risk_level,
        "risk_score": int(risk_score),
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
        "generated_at": datetime.utcnow().isoformat(),
        "summary": summary_text,
        "summary_source": summary_source,
        "summary_snapshot": summary_snapshot,
        "metrics": _build_admin_metrics(total_turns, total_sessions, emotion_events, total_quizzes, risk_score, distress_signal_count),
        "top_emotions": top_emotions,
        "chats": chats,
        "sessions": sessions,
        "questionnaire_results": quizzes,
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

@app.get("/api/admin/summary/stream")
async def admin_summary_stream(username: str, db: Session = Depends(get_db)):
    async def stream():
        overview = await admin_overview(username=username, db=db)
        summary = str(overview.get("summary") or "").strip()
        if not summary:
            yield _to_event({"type": "error", "message": "No summary available"})
            return

        for idx in range(0, len(summary), 64):
            yield _to_event({"type": "summary_delta", "delta": summary[idx:idx + 64]})
            await asyncio.sleep(0)

        yield _to_event({
            "type": "summary_final",
            "summary": summary,
            "summary_source": overview.get("summary_source", "fallback"),
        })
    return StreamingResponse(stream(), media_type="application/x-ndjson")

@app.post("/api/interact", response_model=InteractResponse)
async def interact(username: str = Form(...), image: Optional[str] = Form(None), file: Optional[UploadFile] = File(None), user_message: Optional[str] = Form(None), db: Session = Depends(get_db)):
    if not file: raise HTTPException(400, "Microphone input required.")
    audio_bytes, filename = await file.read(), file.filename or ""
    async with handle_temp_audio(audio_bytes, filename) as audio_path: perc = await _run_perception_tasks(audio_path, image)

    user_text = (perc["transcription"] or user_message or "").strip()
    if not user_text: return InteractResponse(dominant_emotion=perc["dominant_emotion"], speech_emotion=perc["speech_emotion"], face_emotion=perc["face_emotion"], transcription="", llm_response="", errors=perc["errors"] + ["No speech detected."])

    errors, llm_res = list(perc["errors"]), ""
    if client := await _ensure_cloud_llm_client():
        try: llm_res = await client.ask_serenity(user_text, timeout=LLM_TIMEOUT_SECONDS)
        except Exception as e: errors.append(f"LLM Error: {e}")
    else:
        errors.append("LLM client offline")

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
                yield _to_event({"type": "final", "llm_response": "", "transcription": "", "dominant_emotion": perc["dominant_emotion"], "speech_emotion": perc["speech_emotion"], "face_emotion": perc["face_emotion"]})
                return

            async for payload in _stream_chat_events(db, username, user_text, perc["dominant_emotion"], perc["speech_emotion"], perc["face_emotion"], "voice", True): yield payload

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")

@app.post("/api/chat")
async def chat(payload: ChatRequest, db: Session = Depends(get_db)):
    if not payload.message: raise HTTPException(400, "Message empty.")
    errors, llm_res = [], ""
    if client := await _ensure_cloud_llm_client():
        try: llm_res = await client.ask_serenity(payload.message, timeout=LLM_TIMEOUT_SECONDS)
        except Exception as e: errors.append(f"LLM Error: {e}")
    else:
        errors.append("LLM client offline")

    await _persist_turn_safe(db, payload.username, payload.message, llm_res, "Neutral", "Neutral", "Neutral", errors)

    tts_base64, tts_err = await _generate_tts_base64(llm_res)
    if tts_err: errors.append(tts_err)
    return InteractResponse(dominant_emotion="Neutral", speech_emotion="Neutral", face_emotion="Neutral", transcription=payload.message, llm_response=llm_res, tts_audio_base64=tts_base64, errors=_dedupe_errors(errors))

@app.post("/api/chat/stream")
async def chat_stream(payload: ChatRequest, db: Session = Depends(get_db)):
    if not payload.message: raise HTTPException(400, "Message empty.")
    return StreamingResponse(_stream_chat_events(db, payload.username, payload.message, "Neutral", "Neutral", "Neutral", "text", False), media_type="application/x-ndjson")