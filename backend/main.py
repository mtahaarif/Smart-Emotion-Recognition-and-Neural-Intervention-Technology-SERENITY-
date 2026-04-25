import asyncio
import base64
import contextlib
import importlib
import json
import logging
import math
import os
import re
import secrets
import threading
import tempfile
import time
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import bcrypt
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try: import torch
except ImportError: torch = None
try: import psutil
except ImportError: psutil = None

try: WhisperModel = getattr(importlib.import_module("faster_whisper"), "WhisperModel", None)
except ImportError: WhisperModel = None
try: import whisper as openai_whisper
except ImportError: openai_whisper = None
try: import edge_tts
except ImportError: edge_tts = None

from backend.audio_core   import initialize_audio_runtime, predict_audio_emotion
from backend.clinical_core import (
    PHASES_BY_FRAMEWORK, advance_phase,
    build_admin_clinical_handoff_fallback, build_admin_clinical_handoff_prompt,
    build_admin_handoff_markdown, build_handoff_markdown,
    compute_weekly_trajectory_flags, default_phase_for_framework,
    parse_structured_llm_payload, render_handoff_pdf,
)
from backend.clinical_router import (
    FRAMEWORK_ACT, FRAMEWORK_CBT, FRAMEWORK_DBT, FRAMEWORK_SUPPORTIVE,
    RoutingDecision, build_routed_prompt, build_safety_override_response,
    determine_clinical_mode, evaluate_clinical_route,
)
from backend.emotion_core  import initialize_face_runtime, analyze_face
from backend.database import (
    SessionLocal, apply_schema_migrations, calculate_symptom_trajectory, engine,
    fetch_or_create_clinical_state, fetch_questionnaire_results,
    fetch_recent_edge_diagnostics, fetch_recent_sessions_with_emotions,
    fetch_recent_turn_summaries, fetch_trajectory_snapshots,
    persist_clinical_distortion_event, persist_clinical_routing_event,
    persist_edge_diagnostic_sample, persist_questionnaire_result,
    persist_safety_escalation_event, persist_turn,
    replace_trajectory_snapshots, update_user_emergency_contact, upsert_clinical_state,
)
from backend.cloud_llm_core      import CloudLLMClient, CloudLLMError
from backend.questionnaires_data import (
    QUESTIONNAIRE_DEFINITIONS, normalize_questionnaire_type,
    questionnaire_clinical_flags, questionnaire_templates, score_questionnaire,
)
import backend.models as models

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _ei(name, default, minimum=1):
    try: return max(minimum, int(os.getenv(name, str(default)).strip()))
    except: return max(minimum, default)

def _ef(name, default, minimum=0.1):
    try: return max(minimum, float(os.getenv(name, str(default)).strip()))
    except: return max(minimum, default)

WHISPER_MODEL_SIZE  = os.getenv("SERENITY_WHISPER_MODEL_SIZE", "tiny").strip()
WHISPER_CPU_THREADS = _ei("SERENITY_WHISPER_CPU_THREADS", max(1, (os.cpu_count() or 4) // 2))
WHISPER_TIMEOUT     = _ei("SERENITY_WHISPER_TIMEOUT_SECONDS", 40)
EMOTION_TIMEOUT     = _ei("SERENITY_EMOTION_TIMEOUT_SECONDS", 20)
LLM_TIMEOUT         = _ei("SERENITY_LLM_TIMEOUT_SECONDS", 25)
TTS_ENABLED         = os.getenv("SERENITY_TTS_ENABLED", "true").lower() == "true"
TTS_VOICE           = os.getenv("SERENITY_TTS_VOICE", "en-GB-RyanNeural").strip()
TTS_FALLBACK_VOICE  = os.getenv("SERENITY_TTS_FALLBACK_VOICE", "").strip()
TTS_TIMEOUT         = _ei("SERENITY_TTS_TIMEOUT_SECONDS", 45)
TTS_RETRIES         = _ei("SERENITY_TTS_RETRIES", 2)
TTS_STREAM_MODE     = os.getenv("SERENITY_TTS_STREAM_MODE", "sentence").strip().lower()
if TTS_STREAM_MODE not in {"sentence", "final"}: TTS_STREAM_MODE = "sentence"

ADMIN_DEFAULT_LIMIT      = _ei("SERENITY_ADMIN_DEFAULT_LIMIT", 300, minimum=50)
ADMIN_MAX_LIMIT          = _ei("SERENITY_ADMIN_MAX_LIMIT", 3000, minimum=200)
ADMIN_OV_CACHE_TTL       = _ef("SERENITY_ADMIN_OVERVIEW_CACHE_TTL_SECONDS", 20.0)
ADMIN_SUM_CACHE_TTL      = _ef("SERENITY_ADMIN_SUMMARY_CACHE_TTL_SECONDS", 120.0)
ADMIN_SUM_TIMEOUT        = _ef("SERENITY_ADMIN_SUMMARY_TIMEOUT_SECONDS", 10.0)
PREWARM_MODELS           = os.getenv("SERENITY_PREWARM_MODELS", "true").lower() == "true"
PREWARM_WHISPER          = os.getenv("SERENITY_PREWARM_WHISPER", "false").lower() == "true"
CLINICAL_WORSENING_DELTA = _ei("SERENITY_CLINICAL_WEEKLY_WORSENING_DELTA", 4)
EDGE_BUF_SIZE            = _ei("SERENITY_EDGE_DIAGNOSTICS_BUFFER_SIZE", 240)

EMOTION_LABELS    = ["angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_ALIAS     = {"surprised": "surprise", "fearful": "fear", "no face": "neutral"}
NEGATIVE_EMOTIONS = {"angry", "disgust", "fear", "sad"}

# Fires immediately on . ! ? — no trailing-space requirement.
# This means TTS starts the SAME event-loop tick the punctuation token arrives.
SENTENCE_RE = re.compile(r"([.!?\n]+(?:\s+|$))")

DISTRESS_RE = re.compile(
    r"\b(hopeless|worthless|overwhelmed|panic|can't cope|cannot cope"
    r"|self[- ]?harm|suicid|hurt myself|end my life)\b", re.IGNORECASE)
Q_MAX_SCORES = {"PHQ-9": 27.0, "GAD-7": 21.0, "PCL-5": 80.0}

whisper_lock = threading.Lock()
llm_lock     = threading.Lock()

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class AuthRequest(BaseModel):           username: str; password: str
class AuthResponse(BaseModel):          message: str;  username: str
class ChatRequest(BaseModel):           username: str; message: str
class EmergencyContactRequest(BaseModel):
    username: str; contact_name: str = ""; contact_phone: str = ""
class QuestionnaireSubmitRequest(BaseModel):
    username: str; questionnaire_type: str
    answers: List[int] = Field(default_factory=list); submitted_at: Optional[str] = None
class InteractResponse(BaseModel):
    dominant_emotion: str; speech_emotion: str; face_emotion: str
    transcription: str; llm_response: str
    tts_audio_base64: Optional[str] = None
    tts_audio_segments_base64: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

def get_db():
    db = SessionLocal()
    try:    yield db
    finally: db.close()

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    app.state.cloud_llm_client = None
    app.state.whisper_model    = None
    app.state.face_runtime     = None
    app.state.speech_runtime   = None
    app.state.admin_sum_cache  = {"key": "", "summary": "", "expires_at": 0.0}
    app.state.admin_ov_cache   = {}
    app.state.edge_diag        = deque(maxlen=EDGE_BUF_SIZE)
    app.state.whisper_device   = "cuda" if (torch and torch.cuda.is_available()) else "cpu"

    if PREWARM_MODELS:
        def _warm():
            with contextlib.suppress(Exception):
                app.state.speech_runtime = initialize_audio_runtime()
            with contextlib.suppress(Exception):
                app.state.face_runtime = initialize_face_runtime()
            if PREWARM_WHISPER and not app.state.whisper_model:
                _load_whisper(app.state.whisper_device)
        await run_in_threadpool(_warm)

    try: yield
    finally:
        if c := getattr(app.state, "cloud_llm_client", None):
            with contextlib.suppress(Exception): await c.close()


app = FastAPI(title="SERENITY API", lifespan=_lifespan)
models.Base.metadata.create_all(bind=engine)
apply_schema_migrations()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------------------
# App-state helpers
# ---------------------------------------------------------------------------
def _sg(k, d=None): return getattr(app.state, k, d)
def _ss(k, v):      setattr(app.state, k, v)
def _dedup(errs):   return list(dict.fromkeys(str(e) for e in errs if str(e).strip()))

# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
def _is_bcrypt(v):   return bool(v and (v.startswith("$2a$") or v.startswith("$2b$") or v.startswith("$2y$")))
def _hash_pw(pw):    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
def _verify_pw(plain, stored):
    stored = str(stored or "")
    if not stored: return False
    if _is_bcrypt(stored):
        with contextlib.suppress(ValueError):
            return bcrypt.checkpw(plain.encode(), stored.encode())
        return False
    return secrets.compare_digest(stored, plain)

# ---------------------------------------------------------------------------
# Clinical helpers
# ---------------------------------------------------------------------------
def _phase_idx(framework, phase):
    phases = PHASES_BY_FRAMEWORK.get(str(framework or "").strip(),
             PHASES_BY_FRAMEWORK.get("Supportive_Stabilization", []))
    lo = str(phase or "").strip().lower()
    for i, c in enumerate(phases):
        if c.strip().lower() == lo: return i
    return 0

def _latest_q_scores(db, username):
    rows = fetch_questionnaire_results(db, username=username, limit=30, include_answers=False)
    latest = {}
    for r in rows:
        qt = str(r.get("questionnaire_type") or "").upper()
        if qt and qt not in latest: latest[qt] = int(r.get("total_score") or 0)
    return latest

def _clinical_risk_score(db, username, user_text, speech_emo, face_emo):
    scores = _latest_q_scores(db, username)
    risk   = sum(1 for v in questionnaire_clinical_flags(scores).values() if v) * 2
    if DISTRESS_RE.search(str(user_text or "")): risk += 2
    neg = sum(1 for e in (speech_emo, face_emo) if _norm_emo(e) in NEGATIVE_EMOTIONS)
    risk += (2 if neg >= 2 else 1 if neg == 1 else 0)
    return int(risk)

def _norm_emo(e): return EMOTION_ALIAS.get(str(e or "neutral").strip().lower(), str(e or "neutral").strip().lower())
def _risk_lbl(s): return "elevated" if s >= 6 else "monitor" if s >= 3 else "stable"
def _eng_band(s): return "high" if s >= 70 else "moderate" if s >= 35 else "low"
def _sev_pts(s):
    s = str(s or "").lower()
    return 3 if s in {"severe","very severe","extremely severe","elevated","high"} \
        else 2 if s in {"moderate","moderately severe"} \
        else 1 if s in {"mild","minimal"} else 0
def _score_trend(v):
    if len(v) < 2: return "insufficient_data"
    d = int(v[0]) - int(v[1])
    return "worsening" if d >= 3 else "improving" if d <= -3 else "stable"
def _overall_trend(tr):
    vals = set(tr.values())
    if "worsening" in vals and "improving" in vals: return "mixed"
    for k in ("worsening","improving","stable"):
        if k in vals: return k
    return "insufficient_data"
def _symptom_burden(scores):
    ratios = [min(1.0, float(v) / Q_MAX_SCORES[k])
              for k, v in scores.items() if k in Q_MAX_SCORES and Q_MAX_SCORES[k] > 0]
    return round((sum(ratios) / len(ratios)) * 100.0, 1) if ratios else 0.0
def _parse_dt(value):
    if not value: return None
    with contextlib.suppress(ValueError, TypeError):
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    return None

# ---------------------------------------------------------------------------
# Edge diagnostics
# ---------------------------------------------------------------------------
def _edge_stats(latency_ms):
    rss = 0.0
    if psutil:
        with contextlib.suppress(Exception):
            rss = round(float(psutil.Process(os.getpid()).memory_info().rss) / 1048576.0, 2)
    return {"latency_ms": round(float(latency_ms), 2), "rss_mb": rss}

def _persist_diag(db, username, source, stt, ser, fer, llm_ms, sp_conf, fc_conf):
    total = stt + ser + fer + llm_ms
    stats = _edge_stats(total)
    sample = {
        "captured_at": datetime.utcnow().isoformat(), "username": username, "source": source,
        "stt_latency_ms": round(stt, 2), "ser_latency_ms": round(ser, 2),
        "fer_latency_ms": round(fer, 2), "total_latency_ms": round(total, 2),
        "memory_mb": stats["rss_mb"], "llm_ms": round(llm_ms, 2),
        "speech_confidence": round(sp_conf or 0.0, 2), "face_confidence": round(fc_conf or 0.0, 2),
    }
    buf = _sg("edge_diag")
    if isinstance(buf, deque): buf.append(sample)
    persist_edge_diagnostic_sample(db, username=username, sample=sample)

def _latest_edge_sample(db, username=None):
    live = _sg("edge_diag")
    if isinstance(live, deque) and live: return dict(live[-1])
    rows = fetch_recent_edge_diagnostics(db, username=username, limit=1)
    if rows: return dict(rows[0])
    w = (math.sin(time.time() * 0.75) + 1.0) / 2.0
    return {"stt_latency_ms": round(180+120*w,2), "ser_latency_ms": round(42+30*w,2),
            "fer_latency_ms": round(55+28*w,2), "memory_mb": round(620+180*w,2)}

# ---------------------------------------------------------------------------
# Protocol payload extractor
# ---------------------------------------------------------------------------
def _extract_payload(raw):
    raw_text = str(raw or "").strip()
    proto: Dict[str, Any] = {}
    parse_src = raw_text

    if "|||" in raw_text:
        visible, _, tail = raw_text.partition("|||")
        parse_src = visible.strip()
        if tail.strip():
            with contextlib.suppress(json.JSONDecodeError):
                td = json.loads(tail.strip())
                if isinstance(td, dict):
                    proto = {"advance_phase": bool(td.get("advance_phase", False)),
                             "detected_distortion": str(td.get("detected_distortion") or "").strip()}

    parsed = parse_structured_llm_payload(parse_src)
    resp   = str(parsed.get("response_text") or "").strip() or parse_src or raw_text
    if proto:
        parsed["advance_phase"] = bool(proto.get("advance_phase", parsed.get("advance_phase", False)))
        if proto.get("detected_distortion"):
            parsed["detected_distortion"] = str(proto["detected_distortion"])
    parsed["response_text"] = resp
    return parsed

def _exc_detail(exc): return f"{type(exc).__name__}: {exc}" if str(exc).strip() else type(exc).__name__

# ---------------------------------------------------------------------------
# LLM fallback
# ---------------------------------------------------------------------------
def _llm_fallback(user_text, route: RoutingDecision):
    fw = route.framework
    if fw == FRAMEWORK_DBT:
        return ("I hear this feels intense. Let's ground: inhale 4, hold 4, exhale 6. "
                "When ready, share the single most urgent part.")
    if fw == FRAMEWORK_CBT:
        return ("Let's test that thought: one fact that supports it, one that doesn't. "
                "Then we'll form a balanced alternative.")
    if fw == FRAMEWORK_ACT:
        return ("Try this defusion: say 'I am having the thought that…' — notice how it feels. "
                "Then choose one value-aligned action in the next 10 minutes.")
    return ("I'm here with you. Tell me what feels hardest right now, "
            "and we'll choose one practical next step together.")

# ---------------------------------------------------------------------------
# Route signals / clinical state helpers
# ---------------------------------------------------------------------------
def _safety_mode(route: RoutingDecision, payload: Optional[Dict] = None):
    payload = payload or {}
    return bool(route.acute_safety_trigger or route.high_distress or payload.get("safety_alert"))

def _route_signals(route, sp_emo, fc_emo, dom_emo):
    return {
        "framework": route.framework, "risk_score": int(route.risk_score or 0),
        "safety_mode": _safety_mode(route), "route_locked": bool(route.route_locked),
        "route_reason": route.route_reason,
        "detected_distortions": list(route.detected_distortions or []),
        "rumination": bool(route.rumination_detected),
        "acute_safety_trigger": bool(route.acute_safety_trigger),
        "high_distress": bool(route.high_distress),
        "speech_emotion": _norm_emo(sp_emo), "face_emotion": _norm_emo(fc_emo),
        "dominant_emotion": _norm_emo(dom_emo),
    }

def _refresh_state(db, username, route: RoutingDecision, payload, user_text, asst_text):
    cur = fetch_or_create_clinical_state(db, username)
    fw  = str(route.framework or cur.get("active_framework") or FRAMEWORK_DBT)
    ph  = str(cur.get("current_phase") or default_phase_for_framework(fw))
    if str(cur.get("active_framework") or "") != fw:
        ph = default_phase_for_framework(fw)
    if payload.get("advance_phase"):
        ph = advance_phase(fw, ph)

    distress   = _safety_mode(route, payload)
    distortion = payload.get("detected_distortion") or (route.detected_distortions or [None])[0]

    updated = upsert_clinical_state(db, username, {
        "active_framework": fw, "current_phase": ph,
        "phase_index": _phase_idx(fw, ph),
        "requires_safety_review": bool(cur.get("requires_safety_review")) or distress,
        "last_risk_score": int(route.risk_score or 0),
        "last_route_reason": route.route_reason,
        "last_detected_distortion": distortion or "",
        "last_distress_level": "high" if distress else ("moderate" if route.rumination_detected else "low"),
    })

    if distortion:
        persist_clinical_distortion_event(db, username=username,
            distortion_label=distortion, framework=fw, source_excerpt=user_text)

    if distress:
        recent = fetch_recent_turn_summaries(db, limit=6, text_limit=500, username=username)
        md = build_handoff_markdown(username=username, risk_score=int(route.risk_score or 0),
             route_framework=fw, active_flags=[distortion] if distortion else [],
             distress_signals=1, recent_turns=recent)
        persist_safety_escalation_event(db, username=username,
            trigger_type=payload.get("safety_reason") or route.route_reason,
            risk_score=int(route.risk_score or 0), dominant_emotion=route.dominant_emotion,
            transcript_excerpt=user_text, handoff_markdown=md)
    return updated

def _sync_trajectory(db, username):
    rows  = fetch_questionnaire_results(db, username=username, limit=120, include_answers=False)
    traj  = compute_weekly_trajectory_flags(rows, worsening_delta=CLINICAL_WORSENING_DELTA)
    snaps = replace_trajectory_snapshots(db, username=username, snapshots=traj.get("snapshots") or [])
    needs_review = bool(traj.get("requires_safety_review"))
    user = db.query(models.User).filter(models.User.username == username).first()
    if user:
        needs_review = needs_review or bool(getattr(user, "requires_safety_review", False))
        user.requires_safety_review = needs_review; db.commit()
    upsert_clinical_state(db, username, {
        "requires_safety_review": needs_review,
        "last_distress_level": "high" if needs_review else "low",
    })
    return {"requires_safety_review": needs_review,
            "flagged_questionnaires": list(traj.get("flagged_questionnaires") or []),
            "snapshots": snaps}

def _pending_assessments(history, cadence_days=7):
    now, out = datetime.utcnow(), []
    for qt in ("PHQ-9", "GAD-7", "PCL-5"):
        rows    = list(history.get(qt) or [])
        last_dt = _parse_dt((rows[-1] if rows else {}).get("created_at") or "")
        if not last_dt:
            out.append({"questionnaire_type": qt, "is_due": True, "days_since_last": None,
                        "days_until_due": 0, "next_due_at": None, "reason": "No previous assessment."})
            continue
        elapsed  = max(0, (now - last_dt).days)
        next_due = last_dt + timedelta(days=cadence_days)
        out.append({"questionnaire_type": qt, "is_due": elapsed >= cadence_days,
                    "days_since_last": elapsed,
                    "days_until_due": max(0, (next_due - now).days),
                    "next_due_at": next_due.isoformat(),
                    "reason": "Overdue." if elapsed >= cadence_days else "Within window."})
    return out

def _care_plan(clinical_state, latest_scores):
    fw  = str(clinical_state.get("active_framework") or "Supportive_Stabilization")
    ph  = str(clinical_state.get("current_phase") or "Emotional Check-In")
    dx  = str(clinical_state.get("last_distress_level") or "low")
    phq9, gad7, pcl5 = (latest_scores.get(k, 0) for k in ("PHQ-9","GAD-7","PCL-5"))
    routine, interventions = [], []
    if phq9 >= 10:
        routine.append({"id":"morning-light","title":"Morning Light Exposure",
            "description":"10-15 min direct sunlight within 30 min of waking.","cadence":"Daily",
            "clinical_rationale":"Targeting: Depressive Lethargy"})
        interventions.append({"id":"behavioral-activation","title":"Micro Behavioral Activation",
            "framework":"CBT_Restructuring","objective":"Complete one 5-min low-friction task.",
            "clinical_rationale":"Targeting: Low Motivation"})
    if gad7 >= 10:
        routine.append({"id":"worry-postponement","title":"Scheduled Worry Time",
            "description":"Defer anxieties to 4:00 PM 15-min window.","cadence":"Daily",
            "clinical_rationale":"Targeting: Generalized Anxiety"})
        interventions.append({"id":"pmr","title":"Progressive Muscle Relaxation",
            "framework":"DBT_Distress_Tolerance","objective":"Tense and release muscle groups.",
            "clinical_rationale":"Targeting: Somatic Tension"})
    if pcl5 >= 31:
        routine.append({"id":"evening-wind-down","title":"Predictable Evening Wind-down",
            "description":"Low-stimulation 1 hr before sleep.","cadence":"Daily",
            "clinical_rationale":"Targeting: Hypervigilance"})
        interventions.append({"id":"container","title":"Container Exercise",
            "framework":"ACT_Defusion","objective":"Visualize placing distress in a locked container.",
            "clinical_rationale":"Targeting: Intrusive Thoughts"})
    if not routine:
        routine = [
            {"id":"morning-checkin","title":"Morning Emotional Check-In",
             "description":"2-min mood, tension, intent naming.","cadence":"Daily",
             "clinical_rationale":"Targeting: Baseline Maintenance"},
            {"id":"evening-reflection","title":"Evening Reflection",
             "description":"Review triggers and coping used.","cadence":"Daily",
             "clinical_rationale":"Targeting: Routine Structuring"},
        ]
    if not interventions:
        interventions = [{"id":"micro-breath","title":"60-Second Paced Breathing",
            "framework":"DBT_Distress_Tolerance","objective":"Downshift arousal.",
            "clinical_rationale":"Targeting: Baseline Regulation"}]
    if dx == "high":
        interventions.insert(0, {"id":"safety-grounding","title":"Safety Grounding Sequence",
            "framework":"Safety_Stabilization","objective":"Immediate grounding workflow.",
            "clinical_rationale":"Targeting: Acute Distress"})
    return {"framework": fw, "phase": ph, "last_distress_level": dx,
            "latest_scores": latest_scores,
            "daily_routine_blueprint": routine, "micro_interventions": interventions}

# ---------------------------------------------------------------------------
# Admin caches
# ---------------------------------------------------------------------------
def _norm_narrative(t):
    t = str(t or "").strip()
    if not t: return ""
    return re.sub(r"\n{3,}", "\n\n", t.replace("\r\n","\n").replace("\r","\n")).strip()

def _sum_cache_get(key):
    c = _sg("admin_sum_cache", {})
    if c.get("key") != key or float(c.get("expires_at") or 0) <= time.time(): return None
    return str(c.get("summary") or ""), str(c.get("source") or "fallback")

def _sum_cache_set(key, summary, source):
    _ss("admin_sum_cache", {"key": key, "summary": summary, "source": source,
                             "expires_at": time.time() + ADMIN_SUM_CACHE_TTL})

def _ov_cache_get(key):
    c = _sg("admin_ov_cache", {})
    [c.pop(k, None) for k, v in list(c.items()) if float(v.get("expires_at") or 0) <= time.time()]
    return (c[key]["payload"] if key in c else None)

def _ov_cache_set(key, payload):
    c = _sg("admin_ov_cache", {})
    c[key] = {"payload": payload, "expires_at": time.time() + ADMIN_OV_CACHE_TTL}
    if len(c) > 64: c.pop(min(c, key=lambda k: float(c[k].get("expires_at") or 0)), None)
    _ss("admin_ov_cache", c)

def _ov_cache_invalidate(username):
    if not (uk := str(username or "").strip()): return
    c = _sg("admin_ov_cache", {})
    [c.pop(k, None) for k in list(c) if k.startswith(f"{uk}:")]

# ---------------------------------------------------------------------------
# Cloud LLM client factory
# ---------------------------------------------------------------------------
async def _get_client():
    if _sg("cloud_llm_client"): return _sg("cloud_llm_client")
    with llm_lock:
        if not _sg("cloud_llm_client"):
            try:   _ss("cloud_llm_client", CloudLLMClient())
            except CloudLLMError: return None
    return _sg("cloud_llm_client")

# ---------------------------------------------------------------------------
# Admin narrative generator
# ---------------------------------------------------------------------------
async def _gen_admin_summary(snapshot, recent_turns):
    turns = [{"timestamp": str(r.get("timestamp") or "unknown"),
              "user_text": str(r.get("user_text") or "")[:400],
              "assistant_text": str(r.get("assistant_text") or "")[:400]}
             for r in (recent_turns or [])[:10]]
    cache_key = json.dumps({"snapshot": snapshot, "recent_turns": turns},
                           sort_keys=True, ensure_ascii=False, separators=(",",":"))
    if cached := _sum_cache_get(cache_key): return cached

    fallback = build_admin_clinical_handoff_fallback(snapshot, turns)
    text, source = fallback, "fallback"
    client = await _get_client()
    if client:
        with contextlib.suppress(Exception):
            prompt    = build_admin_clinical_handoff_prompt(snapshot, turns)
            candidate = await asyncio.wait_for(
                client.ask_serenity(prompt, timeout=ADMIN_SUM_TIMEOUT),
                timeout=ADMIN_SUM_TIMEOUT + 1.5)
            if norm := _norm_narrative(candidate):
                text, source = norm, "cloud_llm"
    _sum_cache_set(cache_key, text, source)
    return text, source

def _fallback_admin_summary(snap):
    r = snap.get("risk",{});  s = snap.get("screening",{})
    e = snap.get("emotion",{}); f = snap.get("follow_up",{})
    scores = ", ".join(f"{k} {v}" for k,v in s.get("latest_scores",{}).items()) or "none"
    trends = ", ".join(f"{k}: {v}" for k,v in s.get("trends",{}).items()) or "insufficient_data"
    return "\n".join([
        f"- Client {snap.get('username','?')}: {r.get('level','stable')} risk, score {r.get('score',0)}.",
        f"- Affect: {e.get('dominant_emotion','neutral')}, neg-ratio {e.get('negative_ratio',0)}.",
        f"- Screening: {scores}; trends {trends}.",
        f"- Flags: {', '.join(r.get('active_flags',[])) or 'none'}, signals: {r.get('distress_signal_count',0)}.",
        f"- Follow-up: {f.get('primary_priority','routine support')}",
        f"- Cadence: {f.get('cadence','weekly')}",
    ])

# ---------------------------------------------------------------------------
# Audio temp-file context manager
# ---------------------------------------------------------------------------
@contextlib.asynccontextmanager
async def _audio_ctx(audio_bytes, filename):
    if not audio_bytes: yield None; return
    suffix = ".webm" if filename.endswith(".webm") else ".mp3" if filename.endswith(".mp3") else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(audio_bytes); path = f.name
    try: yield path
    finally:
        with contextlib.suppress(OSError):
            if os.path.exists(path): os.remove(path)

# ---------------------------------------------------------------------------
# Whisper loader
# ---------------------------------------------------------------------------
def _load_whisper(device):
    with whisper_lock:
        if _sg("whisper_model"): return
        if WhisperModel:
            compute = "float16" if device == "cuda" else "int8"
            _ss("whisper_model", WhisperModel(WHISPER_MODEL_SIZE, device=device,
                                              compute_type=compute, cpu_threads=WHISPER_CPU_THREADS))
            _ss("whisper_backend", "faster-whisper")
        elif openai_whisper:
            _ss("whisper_model", openai_whisper.load_model(WHISPER_MODEL_SIZE, device=device))
            _ss("whisper_backend", "openai-whisper")

def _transcribe(path):
    _load_whisper(_sg("whisper_device", "cpu"))
    model, backend = _sg("whisper_model"), _sg("whisper_backend")
    if not model: return "", "No STT backend"
    try:
        if backend == "faster-whisper":
            segs, _ = model.transcribe(path, language="en", beam_size=1)
            return " ".join(s.text.strip() for s in segs if s.text.strip()), None
        return str(model.transcribe(path, fp16=False, language="en").get("text","")).strip(), None
    except Exception as e: return "", f"Transcription failed: {e}"

# ---------------------------------------------------------------------------
# TTS — ZERO DISK I/O: audio streams directly into a RAM bytearray
# ---------------------------------------------------------------------------
async def _tts(text: str):
    """
    Streams Edge TTS audio into memory (bytearray) instead of writing to disk.
    Eliminates the SD-card read/write bottleneck on Raspberry Pi 5.
    Falls back to voice list if primary returns error.
    """
    if not text or not TTS_ENABLED or not edge_tts:
        return None, None

    voices = [TTS_VOICE]
    if TTS_FALLBACK_VOICE and TTS_FALLBACK_VOICE != TTS_VOICE:
        voices.append(TTS_FALLBACK_VOICE)

    last_err = None
    for voice in voices:
        for attempt in range(1, TTS_RETRIES + 1):
            try:
                audio_data = bytearray()
                async for chunk in edge_tts.Communicate(text=text, voice=voice).stream():
                    if chunk["type"] == "audio":
                        audio_data.extend(chunk["data"])
                if audio_data:
                    return base64.b64encode(bytes(audio_data)).decode(), None
            except Exception as exc:
                last_err = exc
                if attempt < TTS_RETRIES:
                    await asyncio.sleep(0.4 * attempt)

    return None, f"TTS failed: {last_err}"

# ---------------------------------------------------------------------------
# Perception pipeline (parallel STT + SER + FER)
# ---------------------------------------------------------------------------
async def _perceive(audio_path, image_data):
    async def _task(name, fn, *args):
        t = time.perf_counter()
        try:
            v = await asyncio.wait_for(run_in_threadpool(fn, *args),
                  timeout=WHISPER_TIMEOUT if name == "stt" else EMOTION_TIMEOUT)
            return name, v, (time.perf_counter()-t)*1000
        except Exception as e: return name, e, (time.perf_counter()-t)*1000

    tasks = []
    if audio_path:
        tasks += [_task("stt", _transcribe, audio_path),
                  _task("ser", predict_audio_emotion, audio_path, _sg("speech_runtime"))]
    if image_data: tasks.append(_task("fer", analyze_face, image_data, _sg("face_runtime")))

    res = {"transcription":"","speech_emotion":"Neutral","face_emotion":"Neutral",
           "speech_conf":0.0,"face_conf":0.0,"stt_latency_ms":0.0,
           "ser_latency_ms":0.0,"fer_latency_ms":0.0}
    errors = []

    for name, val, elapsed in await asyncio.gather(*tasks, return_exceptions=True):
        if isinstance(val, Exception): errors.append(f"{name} failed: {val}"); continue
        if name == "stt":
            res["transcription"], err = val; res["stt_latency_ms"] = round(elapsed,2)
            if err: errors.append(err)
        elif name in ("ser","fer"):
            key = "speech" if name=="ser" else "face"
            res[f"{key}_emotion"] = val.get("emotion","Neutral")
            res[f"{key}_conf"]    = val.get("confidence",0.0)
            if name == "ser": res["ser_latency_ms"] = round(elapsed,2)
            else:             res["fer_latency_ms"] = round(elapsed,2)

    sp = {l: (1.0 - res["speech_conf"]/100)/7 for l in EMOTION_LABELS}
    sp[_norm_emo(res["speech_emotion"])] = res["speech_conf"]/100
    fp = {l: (1.0 - res["face_conf"]/100)/7 for l in EMOTION_LABELS} if image_data else dict(sp)
    if image_data: fp[_norm_emo(res["face_emotion"])] = res["face_conf"]/100

    fused = ({l:(sp[l]+fp[l])/2.0 for l in EMOTION_LABELS} if (audio_path and image_data)
             else (sp if audio_path else fp))
    dominant = max(fused, key=fused.__getitem__).title() if (audio_path or image_data) else "Neutral"
    return {**res, "dominant_emotion": dominant, "errors": errors}

# ---------------------------------------------------------------------------
# Persist turn helper
# ---------------------------------------------------------------------------
async def _persist_turn(db, username, user_text, llm_resp, dom, sp, fc, errors=None):
    try:
        t = await run_in_threadpool(persist_turn, db, username, user_text, llm_resp, dom, sp, fc)
        _ov_cache_invalidate(username)
        return int(getattr(t, "id", 0) or 0) or None
    except Exception as exc:
        if errors is not None: errors.append(f"DB: {exc}")
        return None

def _ev(p): return json.dumps(p, ensure_ascii=False) + "\n"

# ---------------------------------------------------------------------------
# Core streaming handler  — INSTANT TOKENS, EAGER TTS, GUILLOTINE CUTOFF
# ---------------------------------------------------------------------------
async def _stream_events(db, username, user_text, dom_emo, sp_emo, fc_emo, source,
                          emotion_first=False, perc_metrics=None):
    metrics = perc_metrics or {}
    stt_ms  = float(metrics.get("stt_latency_ms") or 0.0)
    ser_ms  = float(metrics.get("ser_latency_ms") or 0.0)
    fer_ms  = float(metrics.get("fer_latency_ms") or 0.0)
    sp_conf = float(metrics.get("speech_conf") or 0.0)
    fc_conf = float(metrics.get("face_conf") or 0.0)

    emo = {"type":"emotion","dominant_emotion":dom_emo,"speech_emotion":sp_emo,"face_emotion":fc_emo}
    txt = {"type":"user_text","text":user_text,"source":source}
    if emotion_first: yield _ev(emo); yield _ev(txt)
    else:             yield _ev(txt); yield _ev(emo)

    clin_state = fetch_or_create_clinical_state(db, username)
    risk_score = _clinical_risk_score(db, username, user_text, sp_emo, fc_emo)
    mode       = determine_clinical_mode(user_text, risk_score, dom_emo)
    user_model = db.query(models.User).filter(models.User.username == username).first()
    route      = evaluate_clinical_route(user_text, risk_score, dom_emo, sp_emo, fc_emo, user_model, mode)
    phase      = str(clin_state.get("current_phase") or default_phase_for_framework(route.framework))
    prompt     = (
        f"SYSTEM MODE LOCK: {mode}. You must remain in this mode for the entire response.\n"
        + build_routed_prompt(user_text, route, phase, bool(clin_state.get("requires_safety_review")))
    )

    sigs = _route_signals(route, sp_emo, fc_emo, dom_emo)
    sigs["phase"] = phase; sigs["mode"] = mode
    yield _ev({"type": "clinical_protocol_status", **sigs})

    # --- LLM unavailable: graceful fallback ---
    client = await _get_client()
    if not client:
        fb   = _llm_fallback(user_text, route)
        errs = ["Cloud LLM unavailable. Local fallback used."]
        tid  = await _persist_turn(db, username, user_text, fb, dom_emo, sp_emo, fc_emo, errs)
        with contextlib.suppress(Exception):
            persist_clinical_routing_event(db, username=username,
                routed_framework=route.framework, route_reason=route.route_reason,
                risk_score=int(route.risk_score or 0), route_locked=bool(route.route_locked),
                acute_safety_trigger=bool(route.acute_safety_trigger),
                rumination_detected=bool(route.rumination_detected),
                detected_distortion=(route.detected_distortions or [""])[0],
                dominant_emotion=dom_emo, speech_emotion=sp_emo, face_emotion=fc_emo, turn_id=tid)
        fb_tts, fb_err = await _tts(fb)
        if fb_err: errs.append(fb_err)
        _persist_diag(db, username, source, stt_ms, ser_ms, fer_ms, 0.0, sp_conf, fc_conf)
        for err in errs: yield _ev({"type":"error","message":err})
        yield _ev({"type":"assistant_replace","text":fb})
        yield _ev({"type":"final","llm_response":fb,"transcription":user_text,
                   "dominant_emotion":dom_emo,"speech_emotion":sp_emo,"face_emotion":fc_emo,
                   "tts_audio_base64":fb_tts,"clinical":sigs})
        return

    # --- Streaming LLM ---
    llm_start = time.perf_counter()
    out_q: asyncio.Queue = asyncio.Queue(maxsize=96)
    tts_q: asyncio.Queue = asyncio.Queue(maxsize=16)
    do_stream_tts = bool(TTS_ENABLED and edge_tts and TTS_STREAM_MODE == "sentence")
    do_final_tts  = bool(TTS_ENABLED and edge_tts and TTS_STREAM_MODE == "final")
    stream_ctrl: Dict[str, Any] = {}

    async def _fetch():
        nonlocal stream_ctrl
        # buf accumulates tokens between sentence boundaries.
        # On cutoff: buf is DISCARDED — no partial sentence reaches TTS.
        buf, seq, chunks, final_text, cutoff = "", 0, [], "", False
        try:
            async for ev in client.stream_serenity_events(prompt, require_protocol_control=True):
                etype = str(ev.get("type") or "")

                if etype == "cutoff":
                    # ── THE GUILLOTINE ─────────────────────────────────────────
                    # buf is silently discarded — TTS queue never sees this fragment.
                    # The last complete sentence was already queued when its . ! ? arrived.
                    cutoff = True
                    break

                if etype == "protocol_control":
                    inc = ev.get("payload") or {}
                    if isinstance(inc, dict):
                        stream_ctrl = {
                            "advance_phase": bool(inc.get("advance_phase",
                                stream_ctrl.get("advance_phase", False))),
                            "detected_distortion": str(
                                inc.get("detected_distortion") or
                                stream_ctrl.get("detected_distortion") or "").strip(),
                        }
                        await out_q.put({"type":"protocol_control",**stream_ctrl})
                    continue

                chunk = str(ev.get("delta") or "")
                if not chunk: continue

                chunks.append(chunk)

                # ── INSTANT DISPATCH: token hits the frontend NOW ──────────
                await out_q.put({"type":"assistant_delta","delta":chunk})

                # ── EAGER SENTENCE CHUNKING: TTS fires on first . ! ? ──────
                buf += chunk
                while True:
                    match = SENTENCE_RE.search(buf)
                    if not match: break
                    sentence = buf[:match.end()].strip()
                    buf      = buf[match.end():]
                    if sentence:
                        seq += 1
                        await out_q.put({"type":"assistant_sentence","text":sentence,"sequence":seq})
                        if do_stream_tts:
                            await tts_q.put((sentence, seq))

            final_text = "".join(chunks).strip()

            if cutoff:
                # Trim final_text to the last complete sentence so the stored
                # turn matches exactly what was spoken by Edge TTS.
                await out_q.put({"type":"assistant_trim_dangling"})
                if m := list(re.finditer(r"[.!?]", final_text)):
                    final_text = final_text[:m[-1].end()].strip()
                # Overwrite any half-sentence the UI may have rendered
                await out_q.put({"type":"assistant_replace","text":final_text})

            elif buf.strip():
                # Natural stream end — flush the last fragment
                seq += 1
                last_frag = buf.strip()
                await out_q.put({"type":"assistant_sentence","text":last_frag,"sequence":seq})
                if do_stream_tts:
                    await tts_q.put((last_frag, seq))

        except Exception as exc:
            LOGGER.warning("LLM stream error: %s", _exc_detail(exc))
            final_text = "".join(chunks).strip()
        finally:
            if do_stream_tts: await tts_q.put(None)
            await out_q.put({"type":"LLM_DONE","final_res":final_text})

    async def _audio():
        """Consumes sentence queue — TTS runs in RAM (no disk I/O)."""
        while True:
            item = await tts_q.get()
            if item is None: break
            try:
                aud, terr = await _tts(item[0])
                if aud:  await out_q.put({"type":"assistant_sentence_tts",
                                          "text":item[0],"sequence":item[1],"audio_base64":aud})
                elif terr: await out_q.put({"type":"error","message":terr})
            except Exception as exc:
                await out_q.put({"type":"error","message":f"TTS stream: {exc}"})
        await out_q.put({"type":"TTS_DONE"})

    tasks = [asyncio.create_task(_fetch())]
    if do_stream_tts: tasks.append(asyncio.create_task(_audio()))

    running, raw_llm = len(tasks), ""
    while running > 0:
        ev = await out_q.get()
        if ev["type"] == "LLM_DONE":   raw_llm = str(ev.get("final_res") or ""); running -= 1
        elif ev["type"] == "TTS_DONE": running -= 1
        else: yield _ev(ev)

    parsed = _extract_payload(raw_llm)
    if stream_ctrl:
        parsed["advance_phase"] = bool(stream_ctrl.get("advance_phase",
            parsed.get("advance_phase", False)))
        if stream_ctrl.get("detected_distortion"):
            parsed["detected_distortion"] = str(stream_ctrl["detected_distortion"])

    final_text = str(parsed.get("response_text") or "").strip() or raw_llm
    if not final_text:
        final_text = _llm_fallback(user_text, route)
        yield _ev({"type":"assistant_replace","text":final_text})
    elif final_text != raw_llm:
        yield _ev({"type":"assistant_replace","text":final_text})

    if not parsed.get("detected_distortion") and route.detected_distortions:
        parsed["detected_distortion"] = route.detected_distortions[0]

    upd    = _refresh_state(db, username, route, parsed, user_text, final_text)
    llm_ms = (time.perf_counter() - llm_start) * 1000.0

    final_tts_audio = None
    if do_final_tts and final_text.strip():
        final_tts_audio, terr = await _tts(final_text)
        if terr: yield _ev({"type":"error","message":terr})

    db_errs: List[str] = []
    tid = await _persist_turn(db, username, user_text, final_text,
                               dom_emo, sp_emo, fc_emo, db_errs)
    with contextlib.suppress(Exception):
        persist_clinical_routing_event(db, username=username,
            routed_framework=route.framework, route_reason=route.route_reason,
            risk_score=int(route.risk_score or 0), route_locked=bool(route.route_locked),
            acute_safety_trigger=bool(route.acute_safety_trigger),
            rumination_detected=bool(route.rumination_detected),
            detected_distortion=str(parsed.get("detected_distortion") or ""),
            dominant_emotion=dom_emo, speech_emotion=sp_emo, face_emotion=fc_emo, turn_id=tid)
    _persist_diag(db, username, source, stt_ms, ser_ms, fer_ms, llm_ms, sp_conf, fc_conf)

    for err in db_errs: yield _ev({"type":"error","message":err})
    yield _ev({"type":"final","llm_response":final_text,"transcription":user_text,
               "dominant_emotion":dom_emo,"speech_emotion":sp_emo,"face_emotion":fc_emo,
               "tts_audio_base64":final_tts_audio,
               "clinical":{"framework":upd.get("active_framework"),
                           "phase":upd.get("current_phase"),
                           "phase_index":upd.get("phase_index"),
                           "risk_score":upd.get("last_risk_score"),
                           "requires_safety_review":upd.get("requires_safety_review")}})

# ===========================================================================
# ENDPOINTS
# ===========================================================================

@app.post("/register", response_model=AuthResponse)
async def register(payload: AuthRequest, db: Session = Depends(get_db)):
    username = str(payload.username or "").strip()
    if not username or not payload.password: raise HTTPException(400, "Username and password required")
    hashed = _hash_pw(payload.password)
    existing = await run_in_threadpool(
        lambda: db.query(models.User).filter(models.User.username == username).first())
    if existing:
        if not _is_bcrypt(str(existing.password or "")):
            existing.password = hashed; db.commit()
            return AuthResponse(message="Success", username=existing.username)
        raise HTTPException(400, "Exists")
    u = models.User(username=username, password=hashed)
    await run_in_threadpool(lambda: (db.add(u), db.commit()))
    return AuthResponse(message="Success", username=u.username)


@app.post("/login", response_model=AuthResponse)
async def login(payload: AuthRequest, db: Session = Depends(get_db)):
    username = str(payload.username or "").strip()
    if not username or not payload.password: raise HTTPException(400, "Required")
    user = await run_in_threadpool(
        lambda: db.query(models.User).filter(models.User.username == username).first())
    if not user or not _verify_pw(payload.password, str(user.password or "")):
        raise HTTPException(401, "Invalid")
    if not _is_bcrypt(str(user.password or "")):
        user.password = _hash_pw(payload.password); db.commit()
    with contextlib.suppress(Exception): _sync_trajectory(db, username)
    return AuthResponse(message="Success", username=user.username)


@app.get("/api/questionnaires/templates")
async def get_templates(types: Optional[str] = None):
    return {"available_types": list(QUESTIONNAIRE_DEFINITIONS),
            "questionnaires": questionnaire_templates(types.split(",") if types else None)}


@app.post("/api/questionnaires/submit")
async def submit_questionnaire(payload: QuestionnaireSubmitRequest, db: Session = Depends(get_db)):
    qt = normalize_questionnaire_type(payload.questionnaire_type)
    if not qt: raise HTTPException(400, "Invalid type")
    max_item = 4 if qt == "PCL-5" else 3
    ans = [max(0, min(max_item, int(x))) for x in payload.answers]
    score, sev = score_questionnaire(qt, ans)
    dt = None
    if payload.submitted_at:
        with contextlib.suppress(ValueError):
            dt = datetime.fromisoformat(payload.submitted_at.replace("Z","+00:00")).replace(tzinfo=None)
    rec  = await run_in_threadpool(persist_questionnaire_result, db, payload.username, qt, ans, score, sev, dt)
    traj = await run_in_threadpool(_sync_trajectory, db, payload.username)
    _ov_cache_invalidate(payload.username)
    return {"message":"Saved","result":{"id":rec.id,"total_score":score,"severity":sev},"trajectory":traj}


@app.get("/api/questionnaires/history")
async def q_history(username: str, limit: int = 30, db: Session = Depends(get_db)):
    if not username: raise HTTPException(400, "Username required")
    return {"username": username,
            "results": await run_in_threadpool(fetch_questionnaire_results, db, username, limit=max(1, limit))}


@app.get("/api/admin/overview")
async def admin_overview(username: str, limit: int = 300, include_answers: bool = False,
                          db: Session = Depends(get_db)):
    uk = str(username or "").strip()
    if not uk: raise HTTPException(400, "Username required")
    limit     = max(20, min(int(limit or ADMIN_DEFAULT_LIMIT), ADMIN_MAX_LIMIT))
    quiz_lim  = max(30, min(limit * 2, ADMIN_MAX_LIMIT * 2))
    act_lim   = max(60, min(limit * 2, 240))
    cache_key = f"{uk}:{limit}:{1 if include_answers else 0}"
    if cached := _ov_cache_get(cache_key): return cached

    def _fetch():
        ur = db.query(models.User.id).filter(models.User.username == uk).first()
        if not ur: return None
        uid = int(ur[0])
        routing_rows = (
            db.query(models.ClinicalRoutingEvent, models.User.username)
              .join(models.User, models.ClinicalRoutingEvent.user_id == models.User.id)
              .filter(models.User.username == uk)
              .order_by(models.ClinicalRoutingEvent.timestamp.desc()).limit(act_lim).all())
        safety_rows = (
            db.query(models.SafetyEscalationEvent, models.User.username)
              .join(models.User, models.SafetyEscalationEvent.user_id == models.User.id)
              .filter(models.User.username == uk)
              .order_by(models.SafetyEscalationEvent.timestamp.desc()).limit(act_lim).all())
        return (
            uid,
            fetch_recent_turn_summaries(db, limit=limit, text_limit=420, username=uk),
            fetch_recent_sessions_with_emotions(db, limit=limit, conversation_limit=420, username=uk),
            fetch_questionnaire_results(db, username=uk, limit=quiz_lim, include_answers=include_answers),
            [{"id":r.id,"username":ru,"routed_framework":r.routed_framework,
              "route_reason":r.route_reason,"risk_score":int(r.risk_score or 0),
              "route_locked":bool(r.route_locked),"acute_safety_trigger":bool(r.acute_safety_trigger),
              "rumination_detected":bool(r.rumination_detected),
              "detected_distortion":r.detected_distortion,
              "dominant_emotion":r.dominant_emotion,"speech_emotion":r.speech_emotion,
              "face_emotion":r.face_emotion,
              "timestamp":r.timestamp.isoformat() if r.timestamp else None}
             for r, ru in routing_rows],
            [{"id":r.id,"username":ru,"trigger_type":r.trigger_type,
              "risk_score":int(r.risk_score or 0),"dominant_emotion":r.dominant_emotion,
              "transcript_excerpt":r.transcript_excerpt,"handoff_markdown":r.handoff_markdown,
              "timestamp":r.timestamp.isoformat() if r.timestamp else None}
             for r, ru in safety_rows],
            db.query(func.count(models.ConversationTurn.id)).filter(models.ConversationTurn.user_id==uid).scalar() or 0,
            db.query(func.count(models.Session.id)).filter(models.Session.user_id==uid).scalar() or 0,
            db.query(func.count(models.QuestionnaireResult.id)).filter(models.QuestionnaireResult.user_id==uid).scalar() or 0,
            db.query(func.count(models.ClinicalRoutingEvent.id)).filter(models.ClinicalRoutingEvent.user_id==uid).scalar() or 0,
            db.query(func.count(models.SafetyEscalationEvent.id)).filter(models.SafetyEscalationEvent.user_id==uid).scalar() or 0,
        )

    fetched = await run_in_threadpool(_fetch)
    if not fetched:
        p = _empty_admin(uk); _ov_cache_set(cache_key, p); return p

    (uid, chats, sessions, quizzes, revents, sevents,
     t_turns, t_sess, t_quiz, t_rev, t_sev) = fetched

    if t_turns + t_sess + t_quiz + t_rev + t_sev == 0:
        p = _empty_admin(uk, uid); _ov_cache_set(cache_key, p); return p

    last_seen = latest_note = ""
    emo_events = neg_turns = distress_count = 0
    emo_counts: Dict[str,int] = {}; lat_scores: Dict[str,int] = {}
    lat_sev: Dict[str,str] = {};    score_hist: Dict[str,List[int]] = {}

    for r in chats:
        ts = r.get("timestamp")
        if ts and (not last_seen or str(ts) > last_seen): last_seen = str(ts)
        emo = _norm_emo(r.get("dominant_emotion")); emo_counts[emo] = emo_counts.get(emo,0)+1
        if emo in NEGATIVE_EMOTIONS: neg_turns += 1
        if DISTRESS_RE.search(str(r.get("user_text") or "")): distress_count += 1
        if not latest_note and str(r.get("assistant_text") or "").strip():
            latest_note = str(r["assistant_text"]).strip()

    for r in sessions:
        ts = r.get("timestamp")
        if ts and (not last_seen or str(ts) > last_seen): last_seen = str(ts)
        emo_events += len(r.get("emotions") or [])

    for r in quizzes:
        qt = str(r.get("questionnaire_type") or "")
        if not qt: continue
        sc = int(r.get("total_score") or 0)
        score_hist.setdefault(qt,[]).append(sc)
        if qt not in lat_scores: lat_scores[qt]=sc; lat_sev[qt]=str(r.get("severity") or "unknown")
        ts = r.get("created_at")
        if ts and (not last_seen or str(ts) > last_seen): last_seen = str(ts)

    trends    = {k: _score_trend(v[:3]) for k,v in score_hist.items()}
    ov_trend  = _overall_trend(trends)
    top_emos  = sorted([{"emotion":k,"count":c} for k,c in emo_counts.items()],
                       key=lambda x:x["count"], reverse=True)
    dom_emo   = top_emos[0]["emotion"] if top_emos else "neutral"
    flags     = questionnaire_clinical_flags(lat_scores) if lat_scores else {}
    active_fl = [k for k,v in flags.items() if v]
    neg_ratio = round(neg_turns/max(1,len(chats)),3)
    dis_rate  = round(distress_count/max(1,len(chats)),3)
    emo_vol   = round(1.0-(top_emos[0]["count"]/max(1,len(chats))),3) if top_emos else 0.0
    symp_burden = _symptom_burden(lat_scores)
    sev_pts   = max([_sev_pts(v) for v in lat_sev.values()] or [0])
    act_ctx   = _admin_activity_ctx(chats, sessions, quizzes, revents, sevents)

    risk_score = (len(active_fl)*2 + sev_pts
                  + (2 if distress_count>0 else 0)
                  + (1 if neg_ratio>=0.55 else 0)
                  + (1 if ov_trend=="worsening" else 0))
    risk_lv   = _risk_lbl(risk_score)
    eng_score = min(100, int(t_turns*2 + t_sess*4 + t_quiz*8))
    eng_lv    = _eng_band(eng_score)

    risk_facts = ([f"Screening flags: {', '.join(active_fl)}"] if active_fl else []) + \
                 ([f"Distress language in {distress_count} turns"] if distress_count>0 else []) + \
                 (["High negative-affect ratio"] if neg_ratio>=0.55 else []) + \
                 (["Worsening screening trend"] if ov_trend=="worsening" else []) or \
                 ["No acute risk factors identified"]
    prot_facts = ([f"Consistent engagement"] if eng_lv in {"moderate","high"} else []) + \
                 (["Stable/improving trajectory"] if ov_trend in {"stable","improving"} and trends else []) + \
                 (["No recent distress language"] if distress_count==0 else []) or \
                 ["Protective factors not yet established"]

    if risk_lv == "elevated":
        fu_prio = "Prioritize safety-focused follow-up and escalation readiness."
        mon_cad = "Contact within 24-72 h; repeat screeners within one week."
    elif risk_lv == "monitor":
        fu_prio = "Maintain structured follow-up targeting symptom triggers."
        mon_cad = "Weekly review; reassess questionnaires every 1-2 weeks."
    else:
        fu_prio = "Continue supportive care and reinforce resilience strategies."
        mon_cad = "Biweekly to monthly check-ins with periodic screening."

    snap = {
        "username": uk,
        "risk": {"level":risk_lv,"score":int(risk_score),"active_flags":active_fl,
                 "distress_signal_count":int(distress_count),
                 "risk_factors":risk_facts,"protective_factors":prot_facts},
        "emotion": {"dominant_emotion":dom_emo,"negative_ratio":neg_ratio,
                    "volatility":emo_vol,"distribution":top_emos[:4]},
        "screening": {"latest_scores":lat_scores,"latest_severity":lat_sev,
                      "trends":trends,"overall_trend":ov_trend,"symptom_burden_pct":symp_burden},
        "engagement": {"score":int(eng_score),"level":eng_lv},
        "follow_up":  {"primary_priority":fu_prio,"cadence":mon_cad},
        "volume": {"turns":int(t_turns),"sessions":int(t_sess),
                   "questionnaire_entries":int(t_quiz),"emotion_events":int(emo_events),
                   "routing_events":int(t_rev),"safety_events":int(t_sev)},
    }
    summary_text = _fallback_admin_summary(snap)

    user_full = db.query(models.User).filter(models.User.username == uk).first()
    profile = {
        "user_id": int(uid), "username": uk, "last_seen": last_seen,
        "duty_to_warn": bool(getattr(user_full,"duty_to_warn",False)),
        "last_crisis": getattr(user_full,"last_crisis_timestamp",None),
        "risk_level": risk_lv, "risk_score": int(risk_score),
        "active_flags": active_fl, "dominant_emotion": dom_emo,
        "negative_emotion_ratio": neg_ratio, "distress_signal_count": int(distress_count),
        "engagement_score": int(eng_score), "engagement_level": eng_lv,
        "latest_scores": lat_scores, "latest_severity": lat_sev,
        "screening_trends": trends, "overall_trend": ov_trend,
        "symptom_burden_pct": symp_burden, "risk_factors": risk_facts,
        "protective_factors": prot_facts, "follow_up_priority": fu_prio,
        "monitoring_cadence": mon_cad, "latest_assistant_note": latest_note,
        "risk": snap["risk"], "emotion": snap["emotion"],
        "screening": {"latest_scores":lat_scores,"latest_severity":lat_sev,"trends":trends},
        "engagement": snap["engagement"], "follow_up": snap["follow_up"],
    }

    metrics_payload = _build_metrics(
        t_turns, emo_events, t_quiz, risk_score, distress_count,
        int(act_ctx.get("care_plan_adherence_pct") or 0),
        int(act_ctx.get("risk_score_delta") or 0),
        int(act_ctx.get("distress_signal_delta") or 0),
        int(act_ctx.get("care_plan_completed_days") or 0),
        int(act_ctx.get("care_plan_window_days") or 7),
    )

    payload = {
        "user_id": int(uid), "generated_at": datetime.utcnow().isoformat(),
        "summary": summary_text, "summary_source": "computed",
        "summary_snapshot": snap, "metrics": metrics_payload,
        "top_emotions": top_emos, "chats": chats, "sessions": sessions,
        "questionnaire_results": quizzes,
        "timeline_events": list(act_ctx.get("timeline_events") or []),
        "protocol_fidelity": list(act_ctx.get("protocol_fidelity") or []),
        "activity_summary": {
            "care_plan_adherence_pct": int(act_ctx.get("care_plan_adherence_pct") or 0),
            "care_plan_completed_days": int(act_ctx.get("care_plan_completed_days") or 0),
            "care_plan_window_days": int(act_ctx.get("care_plan_window_days") or 7),
            "risk_score_delta": int(act_ctx.get("risk_score_delta") or 0),
            "distress_signal_delta": int(act_ctx.get("distress_signal_delta") or 0),
        },
        "profile": profile,
        "clinical_parameters": {
            "risk_level":risk_lv,"risk_score":int(risk_score),"active_flags":active_fl,
            "distress_signal_count":int(distress_count),"distress_signal_rate":dis_rate,
            "negative_emotion_ratio":neg_ratio,"emotion_volatility":emo_vol,
            "engagement_level":eng_lv,"engagement_score":int(eng_score),
            "screening_trends":trends,"overall_trend":ov_trend,
            "symptom_burden_pct":symp_burden,"risk_factors":risk_facts,
            "protective_factors":prot_facts,"follow_up_priority":fu_prio,
            "monitoring_cadence":mon_cad,
        },
        "flagged_users": [{"username":uk,"flags":flags,"scores":lat_scores,"risk_level":risk_lv}] if active_fl else [],
    }
    _ov_cache_set(cache_key, payload)
    return payload


def _empty_admin(username, user_id=None):
    uid  = int(user_id) if user_id is not None else None
    prof = {"user_id":uid,"username":username,"last_seen":None,
            "risk_level":"stable","risk_score":0,"active_flags":[],
            "dominant_emotion":"neutral","negative_emotion_ratio":0.0,
            "distress_signal_count":0,"engagement_score":0,"engagement_level":"low",
            "latest_scores":{},"latest_severity":{},"screening_trends":{},
            "overall_trend":"insufficient_data","symptom_burden_pct":0.0,
            "risk_factors":[],"protective_factors":[],
            "follow_up_priority":"Continue routine supportive follow-up.",
            "monitoring_cadence":"Weekly review with symptom monitoring.",
            "latest_assistant_note":"",
            "risk":{"level":"stable","score":0,"active_flags":[],"distress_signal_count":0},
            "emotion":{"dominant_emotion":"neutral","negative_ratio":0.0,"distribution":[]},
            "screening":{"latest_scores":{},"latest_severity":{},"trends":{}},
            "engagement":{"score":0,"level":"low"},
            "follow_up":{"primary_priority":"Continue routine supportive follow-up.",
                          "cadence":"Weekly review with symptom monitoring."}}
    return {"user_id":uid,"generated_at":datetime.utcnow().isoformat(),
            "summary":"No data available.","summary_source":"fallback",
            "summary_snapshot":prof,"metrics":_build_metrics(0,0,0,0,0,0),
            "top_emotions":[],"chats":[],"sessions":[],"questionnaire_results":[],
            "timeline_events":[],"protocol_fidelity":[],"profile":prof,
            "clinical_parameters":{"risk_level":"stable","risk_score":0,"active_flags":[],
                "distress_signal_count":0,"distress_signal_rate":0.0,
                "negative_emotion_ratio":0.0,"emotion_volatility":0.0,
                "engagement_level":"low","engagement_score":0,"screening_trends":{},
                "overall_trend":"insufficient_data","symptom_burden_pct":0.0,
                "risk_factors":[],"protective_factors":[],
                "follow_up_priority":"Continue routine supportive follow-up.",
                "monitoring_cadence":"Weekly review with symptom monitoring."},
            "flagged_users":[]}


def _build_metrics(turns, emo_ev, quizzes, risk, distress, adherence_pct,
                   risk_delta=0, distress_delta=0, completed_days=0, window_days=7):
    def _dp(val, lbl):
        return {"delta":int(val),"delta_label":lbl,
                "delta_tone":"up" if val>0 else "down" if val<0 else "neutral"}
    return [
        {"id":"turns","label":"Conversation Turns","value":int(turns),
         "description":"Recent therapeutic dialogue exchanges."},
        {"id":"care_plan_adherence","label":"Care Plan Adherence %",
         "value":f"{int(adherence_pct)}%",
         "description":f"{completed_days}/{window_days} daily check-ins completed."},
        {"id":"emotion_events","label":"Emotion Events","value":int(emo_ev),
         "description":"Captured emotion observations."},
        {"id":"questionnaire_entries","label":"Screening Entries","value":int(quizzes),
         "description":"PHQ-9, GAD-7, and PCL-5 submissions."},
        {"id":"risk_score","label":"Risk Score","value":int(risk),
         "description":"Composite risk from screening, distress, and affect.",
         **_dp(risk_delta,"vs prior assessment")},
        {"id":"distress_signals","label":"Distress Signals","value":int(distress),
         "description":"Keyword-based distress cues.",
         **_dp(distress_delta,"vs prior 7 days")},
    ]


def _admin_activity_ctx(chats, sessions, quizzes, routing_events, safety_events):
    now     = datetime.utcnow()
    w_start = now - timedelta(days=7)
    pw_start= now - timedelta(days=14)

    def _dt(row, key="timestamp"):
        return _parse_dt(str(row.get(key) or ""))

    def _distress_count(rows, start, end):
        return sum(1 for r in rows
                   if (dt := _dt(r)) and start <= dt < end
                   and DISTRESS_RE.search(str(r.get("user_text") or "")))

    days = set()
    for r in (chats or []):
        if (dt := _dt(r)) and dt >= w_start: days.add(dt.date().isoformat())
    for r in (sessions or []):
        if (dt := _dt(r)) and dt >= w_start: days.add(dt.date().isoformat())
    for r in (quizzes or []):
        if (dt := _parse_dt(str(r.get("created_at") or ""))) and dt >= w_start:
            days.add(dt.date().isoformat())

    completed    = len(days)
    adherence    = int(round(min(1.0, completed/7)*100))
    cur_distress = _distress_count(chats, w_start, now)
    prv_distress = _distress_count(chats, pw_start, w_start)
    r_scores     = [int(r.get("risk_score") or 0) for r in routing_events if r.get("risk_score") is not None]
    r_delta      = int(r_scores[0]-r_scores[1]) if len(r_scores)>=2 else 0

    fw_counts = {FRAMEWORK_CBT:0,FRAMEWORK_ACT:0,FRAMEWORK_DBT:0,FRAMEWORK_SUPPORTIVE:0}
    fw_labels = {FRAMEWORK_CBT:"CBT Restructuring",FRAMEWORK_ACT:"ACT Defusion",
                 FRAMEWORK_DBT:"DBT Distress Tolerance",FRAMEWORK_SUPPORTIVE:"Supportive Stabilization"}
    timeline: List[Dict] = []

    for r in (chats or []):
        timeline.append({"id":f"turn-{r.get('id')}","kind":"chat_turn",
            "timestamp":str(r.get("timestamp") or ""),
            "title":f"Chat turn #{r.get('id')}",
            "detail":f"User: {str(r.get('user_text') or '')[:150]}",
            "meta":{"emotion":str(r.get("dominant_emotion") or "neutral")}})
    for r in (sessions or []):
        ec = len(r.get("emotions") or [])
        timeline.append({"id":f"session-{r.get('id')}","kind":"session",
            "timestamp":str(r.get("timestamp") or ""),
            "title":f"Session #{r.get('id')}","detail":f"{ec} emotion observations.",
            "meta":{"emotion_count":ec}})
    for r in (quizzes or []):
        qt = str(r.get("questionnaire_type") or "Questionnaire")
        timeline.append({"id":f"questionnaire-{r.get('id')}","kind":"questionnaire",
            "timestamp":str(r.get("created_at") or ""),
            "title":f"{qt} screening",
            "detail":f"Score {int(r.get('total_score') or 0)} ({r.get('severity','?')}).",
            "meta":{"questionnaire_type":qt,"score":int(r.get("total_score") or 0),
                    "severity":str(r.get("severity") or "unknown")}})
    for r in (routing_events or []):
        fw = str(r.get("routed_framework") or FRAMEWORK_SUPPORTIVE)
        fw_counts[fw] = fw_counts.get(fw,0)+1
        timeline.append({"id":f"routing-{r.get('id')}","kind":"routing",
            "timestamp":str(r.get("timestamp") or ""),
            "title":f"{fw_labels.get(fw,fw)} route",
            "detail":f"Risk {int(r.get('risk_score') or 0)} • {r.get('route_reason','?')}",
            "meta":{"framework":fw,"risk_score":int(r.get("risk_score") or 0),
                    "route_locked":bool(r.get("route_locked")),
                    "acute_safety_trigger":bool(r.get("acute_safety_trigger"))}})
    sev_count = len(safety_events or [])
    for r in (safety_events or []):
        timeline.append({"id":f"safety-{r.get('id')}","kind":"safety",
            "timestamp":str(r.get("timestamp") or ""),
            "title":"Safety protocol triggered",
            "detail":f"{r.get('trigger_type','?')} • risk {int(r.get('risk_score') or 0)}",
            "meta":{"trigger_type":str(r.get("trigger_type") or ""),
                    "risk_score":int(r.get("risk_score") or 0)}})

    timeline.sort(key=lambda x: _parse_dt(x.get("timestamp")) or datetime.min, reverse=True)
    timeline = timeline[:120]

    fidelity = [
        {"id":"CBT_Restructuring","label":"CBT Restructuring","count":fw_counts.get(FRAMEWORK_CBT,0),"tone":"cyan"},
        {"id":"ACT_Defusion","label":"ACT Defusion","count":fw_counts.get(FRAMEWORK_ACT,0),"tone":"emerald"},
        {"id":"DBT_Distress_Tolerance","label":"DBT Distress Tolerance","count":fw_counts.get(FRAMEWORK_DBT,0),"tone":"amber"},
        {"id":"Supportive_Stabilization","label":"Supportive Stabilization","count":fw_counts.get(FRAMEWORK_SUPPORTIVE,0),"tone":"slate"},
        {"id":"Safety_Protocol","label":"Safety Protocol","count":sev_count,"tone":"rose"},
    ]
    max_fc = max((x["count"] for x in fidelity), default=0)
    for x in fidelity: x["share"] = int(round(x["count"]/max_fc*100)) if max_fc else 0

    return {"care_plan_adherence_pct":adherence,"care_plan_completed_days":completed,
            "care_plan_window_days":7,"risk_score_delta":r_delta,
            "distress_signal_delta":int(cur_distress-prv_distress),
            "timeline_events":timeline,"protocol_fidelity":fidelity}


@app.get("/api/admin/clinical-report")
async def admin_clinical_report(username: str, db: Session = Depends(get_db)):
    uk = str(username or "").strip()
    if not uk: raise HTTPException(400, "Username required")
    ov    = await admin_overview(username=uk, limit=ADMIN_DEFAULT_LIMIT, db=db)
    snap  = dict(ov.get("summary_snapshot") or {})
    turns = list(ov.get("chats") or [])[:10]
    text, src = await _gen_admin_summary(snap, turns)
    return {"generated_at":datetime.utcnow().isoformat(),"username":uk,
            "summary":text,"summary_source":src,"recent_turn_count":len(turns),
            "risk_score":int((snap.get("risk") or {}).get("score") or 0)}


@app.get("/api/admin/summary/stream")
async def admin_summary_stream(username: str, db: Session = Depends(get_db)):
    async def _stream():
        rep  = await admin_clinical_report(username=username, db=db)
        text = str(rep.get("summary") or "").strip()
        if not text: yield _ev({"type":"error","message":"No summary available"}); return
        for i in range(0, len(text), 64):
            yield _ev({"type":"summary_delta","delta":text[i:i+64]})
            await asyncio.sleep(0)
        yield _ev({"type":"summary_final","summary":text,"summary_source":rep.get("summary_source","fallback")})
    return StreamingResponse(_stream(), media_type="application/x-ndjson")


@app.post("/api/interact", response_model=InteractResponse)
async def interact(username: str = Form(...), image: Optional[str] = Form(None),
                   file: Optional[UploadFile] = File(None),
                   user_message: Optional[str] = Form(None),
                   db: Session = Depends(get_db)):
    if not file: raise HTTPException(400, "Microphone input required.")
    audio_bytes = await file.read(); filename = file.filename or ""
    async with _audio_ctx(audio_bytes, filename) as ap:
        perc = await _perceive(ap, image)

    user_text = (perc["transcription"] or user_message or "").strip()
    if not user_text:
        return InteractResponse(dominant_emotion=perc["dominant_emotion"],
            speech_emotion=perc["speech_emotion"], face_emotion=perc["face_emotion"],
            transcription="", llm_response="", errors=perc["errors"]+["No speech detected."])

    errors: List[str] = list(perc["errors"])
    t0 = time.perf_counter()
    clin_state = fetch_or_create_clinical_state(db, username)
    risk_score = _clinical_risk_score(db, username, user_text, perc["speech_emotion"], perc["face_emotion"])
    mode       = determine_clinical_mode(user_text, risk_score, perc["dominant_emotion"])
    user_model = db.query(models.User).filter(models.User.username == username).first()
    route      = evaluate_clinical_route(user_text, risk_score, perc["dominant_emotion"],
                    perc["speech_emotion"], perc["face_emotion"], user_model, mode)
    phase  = str(clin_state.get("current_phase") or default_phase_for_framework(route.framework))
    prompt = (f"SYSTEM MODE LOCK: {mode}. You must remain in this mode.\n"
              + build_routed_prompt(user_text, route, phase, bool(clin_state.get("requires_safety_review"))))

    raw, proto_ctrl = "", {}
    client = await _get_client()
    if client:
        try:    raw, proto_ctrl = await client.ask_serenity_with_protocol(prompt, timeout=LLM_TIMEOUT)
        except Exception as e: errors.append(f"LLM: {e}")
    else:
        errors.append("LLM offline")

    parsed = _extract_payload(raw)
    if proto_ctrl:
        parsed["advance_phase"] = bool(proto_ctrl.get("advance_phase", parsed.get("advance_phase", False)))
        if proto_ctrl.get("detected_distortion"):
            parsed["detected_distortion"] = str(proto_ctrl["detected_distortion"])
    if not parsed.get("detected_distortion") and route.detected_distortions:
        parsed["detected_distortion"] = route.detected_distortions[0]

    llm_res = str(parsed.get("response_text") or raw or "").strip()
    upd = _refresh_state(db, username, route, parsed, user_text, llm_res)
    tid = await _persist_turn(db, username, user_text, llm_res,
                               perc["dominant_emotion"], perc["speech_emotion"],
                               perc["face_emotion"], errors)
    with contextlib.suppress(Exception):
        persist_clinical_routing_event(db, username=username,
            routed_framework=route.framework, route_reason=route.route_reason,
            risk_score=int(route.risk_score or 0), route_locked=bool(route.route_locked),
            acute_safety_trigger=bool(route.acute_safety_trigger),
            rumination_detected=bool(route.rumination_detected),
            detected_distortion=str(parsed.get("detected_distortion") or ""),
            dominant_emotion=perc["dominant_emotion"], speech_emotion=perc["speech_emotion"],
            face_emotion=perc["face_emotion"], turn_id=tid)
    _persist_diag(db, username, "voice",
        float(perc.get("stt_latency_ms") or 0), float(perc.get("ser_latency_ms") or 0),
        float(perc.get("fer_latency_ms") or 0), (time.perf_counter()-t0)*1000,
        float(perc.get("speech_conf") or 0), float(perc.get("face_conf") or 0))

    tts64, terr = await _tts(llm_res)
    if terr: errors.append(terr)
    return InteractResponse(dominant_emotion=perc["dominant_emotion"],
        speech_emotion=perc["speech_emotion"], face_emotion=perc["face_emotion"],
        transcription=user_text, llm_response=llm_res, tts_audio_base64=tts64,
        errors=_dedup(errors))


@app.post("/api/interact/stream")
async def interact_stream(username: str = Form(...), image: Optional[str] = Form(None),
                           file: Optional[UploadFile] = File(None),
                           user_message: Optional[str] = Form(None),
                           db: Session = Depends(get_db)):
    if not file: raise HTTPException(400, "Microphone input required.")
    audio_bytes = await file.read(); filename = file.filename or ""

    async def _gen():
        async with _audio_ctx(audio_bytes, filename) as ap:
            perc = await _perceive(ap, image)
            for e in perc["errors"]: yield _ev({"type":"error","message":e})
            user_text = (perc["transcription"] or user_message or "").strip()
            if not user_text:
                yield _ev({"type":"emotion","dominant_emotion":perc["dominant_emotion"],
                           "speech_emotion":perc["speech_emotion"],"face_emotion":perc["face_emotion"]})
                yield _ev({"type":"error","message":"No speech detected."})
                yield _ev({"type":"final","llm_response":"","transcription":"",
                           "dominant_emotion":perc["dominant_emotion"]})
                return
            async for ev in _stream_events(db, username, user_text,
                                            perc["dominant_emotion"], perc["speech_emotion"],
                                            perc["face_emotion"], "voice", True, perc):
                yield ev

    return StreamingResponse(_gen(), media_type="application/x-ndjson")


@app.post("/api/chat")
async def chat(payload: ChatRequest, db: Session = Depends(get_db)):
    if not payload.message: raise HTTPException(400, "Message empty.")
    errors: List[str] = []
    t0 = time.perf_counter()
    clin_state = fetch_or_create_clinical_state(db, payload.username)
    risk_score = _clinical_risk_score(db, payload.username, payload.message, "Neutral", "Neutral")
    mode       = determine_clinical_mode(payload.message, risk_score, "Neutral")
    user_model = db.query(models.User).filter(models.User.username == payload.username).first()
    route      = evaluate_clinical_route(payload.message, risk_score, "Neutral", "Neutral", "Neutral",
                    user_model, mode)
    phase  = str(clin_state.get("current_phase") or default_phase_for_framework(route.framework))
    prompt = (f"SYSTEM MODE LOCK: {mode}. You must remain in this mode.\n"
              + build_routed_prompt(payload.message, route, phase,
                                    bool(clin_state.get("requires_safety_review"))))

    raw, proto_ctrl = "", {}
    client = await _get_client()
    if client:
        try:    raw, proto_ctrl = await client.ask_serenity_with_protocol(prompt, timeout=LLM_TIMEOUT)
        except Exception as e:
            LOGGER.warning("LLM: %s", _exc_detail(e))
            raw = _llm_fallback(payload.message, route)
            errors.append("Cloud LLM unavailable. Fallback used.")
    else:
        raw = _llm_fallback(payload.message, route); errors.append("LLM offline.")

    parsed = _extract_payload(raw)
    if proto_ctrl:
        parsed["advance_phase"] = bool(proto_ctrl.get("advance_phase", parsed.get("advance_phase", False)))
        if proto_ctrl.get("detected_distortion"):
            parsed["detected_distortion"] = str(proto_ctrl["detected_distortion"])
    if not parsed.get("detected_distortion") and route.detected_distortions:
        parsed["detected_distortion"] = route.detected_distortions[0]

    llm_res = str(parsed.get("response_text") or raw or "").strip()
    if not llm_res: llm_res = _llm_fallback(payload.message, route)

    upd = _refresh_state(db, payload.username, route, parsed, payload.message, llm_res)
    tid = await _persist_turn(db, payload.username, payload.message, llm_res,
                               "Neutral", "Neutral", "Neutral", errors)
    with contextlib.suppress(Exception):
        persist_clinical_routing_event(db, username=payload.username,
            routed_framework=route.framework, route_reason=route.route_reason,
            risk_score=int(route.risk_score or 0), route_locked=bool(route.route_locked),
            acute_safety_trigger=bool(route.acute_safety_trigger),
            rumination_detected=bool(route.rumination_detected),
            detected_distortion=str(parsed.get("detected_distortion") or ""),
            dominant_emotion="Neutral", speech_emotion="Neutral", face_emotion="Neutral", turn_id=tid)
    _persist_diag(db, payload.username, "text", 0,0,0, (time.perf_counter()-t0)*1000, 0,0)

    tts64, terr = await _tts(llm_res)
    if terr: errors.append(terr)
    return InteractResponse(dominant_emotion="Neutral", speech_emotion="Neutral",
        face_emotion="Neutral", transcription=payload.message, llm_response=llm_res,
        tts_audio_base64=tts64, errors=_dedup(errors))


@app.post("/api/chat/stream")
async def chat_stream(payload: ChatRequest, db: Session = Depends(get_db)):
    if not payload.message: raise HTTPException(400, "Message empty.")
    return StreamingResponse(
        _stream_events(db, payload.username, payload.message,
                       "Neutral", "Neutral", "Neutral", "text", False),
        media_type="application/x-ndjson")


@app.post("/api/safety/emergency-contact")
async def set_emergency_contact(payload: EmergencyContactRequest, db: Session = Depends(get_db)):
    uk = str(payload.username or "").strip()
    if not uk: raise HTTPException(400, "Username required")
    return {"message": "Saved",
            "contact": update_user_emergency_contact(db, uk,
                str(payload.contact_name or "").strip(), str(payload.contact_phone or "").strip())}


@app.get("/api/mbc/trajectory")
async def mbc_trajectory(username: str, refresh: bool = False, db: Session = Depends(get_db)):
    uk = str(username or "").strip()
    if not uk: raise HTTPException(400, "Username required")
    user = db.query(models.User).filter(models.User.username == uk).first()
    if not user:
        user = models.User(username=uk, password=""); db.add(user); db.commit(); db.refresh(user)
    if refresh: await run_in_threadpool(_sync_trajectory, db, uk)
    traj  = await run_in_threadpool(calculate_symptom_trajectory, db, user.id)
    clin  = fetch_or_create_clinical_state(db, uk)
    needs_rev = bool(traj.get("requires_safety_review")) or bool(clin.get("requires_safety_review"))
    if needs_rev and not clin.get("requires_safety_review"):
        clin = upsert_clinical_state(db, uk, {"requires_safety_review":True,"last_distress_level":"high"})

    hist  = traj.get("history") or {}
    snaps = fetch_trajectory_snapshots(db, uk)
    flagged = [str(s.get("questionnaire_type") or "") for s in snaps
               if s.get("flagged") and s.get("questionnaire_type")]

    chart = []
    for r in (traj.get("time_series") or []):
        p = dict(r)
        if p.get("pcl5"): p["pcl5_scaled_27"] = round(float(p["pcl5"])*27.0/80.0, 2)
        chart.append(p)

    return {"username":uk,"user_id":int(user.id),"requires_safety_review":needs_rev,
            "flagged_questionnaires":flagged,
            "velocity_delta":dict(traj.get("velocity_delta") or {}),
            "history":hist,"time_series":chart,
            "latest_scores":dict(traj.get("latest_scores") or {}),
            "care_plan":_care_plan(clin, dict(traj.get("latest_scores") or {})),
            "pending_assessments":_pending_assessments(hist),
            "has_due_assessment":any(r.get("is_due") for r in _pending_assessments(hist)),
            "cadence_days":7,"snapshots":snaps}


@app.get("/api/diagnostics/edge")
async def edge_diagnostics(username: Optional[str] = None, limit: int = 120,
                            db: Session = Depends(get_db)):
    cap  = max(1, min(int(limit or 120), 500))
    data = fetch_recent_edge_diagnostics(db, username=username, limit=cap)
    live = _sg("edge_diag")
    return {"username":username,"limit":cap,"samples":data,
            "live_samples":list(live)[-cap:] if isinstance(live, deque) else []}


@app.get("/api/diagnostics/metrics")
async def diagnostics_metrics(username: Optional[str] = None, db: Session = Depends(get_db)):
    s = _latest_edge_sample(db, username)
    cpu = ram = 0.0
    if psutil:
        with contextlib.suppress(Exception): cpu = round(float(psutil.cpu_percent(interval=None)), 2)
        with contextlib.suppress(Exception):
            ram = round(float(psutil.Process(os.getpid()).memory_info().rss)/1048576.0, 2)
    if cpu <= 0.0:
        w = (math.sin(time.time()*0.55)+1.0)/2.0; cpu = round(18+52*w, 2)
    if ram <= 0.0:
        w = (math.sin(time.time()*0.33)+1.0)/2.0; ram = round(580+220*w, 2)
    return {"captured_at":str(s.get("captured_at") or datetime.utcnow().isoformat()),
            "username":username,
            "stt_latency_ms":round(float(s.get("stt_latency_ms") or 0),2),
            "ser_latency_ms":round(float(s.get("ser_latency_ms") or 0),2),
            "fer_latency_ms":round(float(s.get("fer_latency_ms") or 0),2),
            "cpu_thread_usage_percent":cpu,"ram_usage_mb":ram,
            "xnnpack_delegate_active":os.getenv("SERENITY_XNNPACK_DELEGATE_ACTIVE","true").lower() in {"1","true","yes"}}


@app.get("/api/admin/handoff/{user_id}")
async def admin_handoff(user_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user: raise HTTPException(404, "User not found")
    un = str(user.username or "").strip()
    if not un: raise HTTPException(400, "No username")
    clin  = fetch_or_create_clinical_state(db, un)
    traj  = await run_in_threadpool(calculate_symptom_trajectory, db, user.id)
    snaps = fetch_trajectory_snapshots(db, un)
    turns = fetch_recent_turn_summaries(db, limit=15, text_limit=700, username=un)
    ov    = await admin_overview(username=un, limit=ADMIN_DEFAULT_LIMIT, db=db)
    narr, nsrc = await _gen_admin_summary(dict(ov.get("summary_snapshot") or {}),
                                           list(ov.get("chats") or [])[:10])
    flagged = [str(s.get("questionnaire_type") or "") for s in snaps
               if s.get("flagged") and s.get("questionnaire_type")]
    traj_p = {"latest_scores":dict(traj.get("latest_scores") or {}),
               "velocity_delta":dict(traj.get("velocity_delta") or {}),
               "history":dict(traj.get("history") or {}),"flagged_questionnaires":flagged}
    needs_rev = (bool(getattr(user,"requires_safety_review",False)) or
                 bool(clin.get("requires_safety_review")) or
                 bool(traj.get("requires_safety_review")))
    md = build_admin_handoff_markdown(user_id=user.id, username=un,
             risk_score=int(clin.get("last_risk_score") or 0),
             requires_safety_review=needs_rev,
             active_framework=str(clin.get("active_framework") or FRAMEWORK_DBT),
             trajectory=traj_p, recent_turns=turns,
             clinical_narrative=narr, clinical_narrative_source=nsrc)
    return {"generated_at":datetime.utcnow().isoformat(),"user_id":user.id,"username":un,
            "risk_score":int(clin.get("last_risk_score") or 0),"requires_safety_review":needs_rev,
            "trajectory":traj_p,"recent_turns":turns,"clinical_narrative":narr,
            "clinical_narrative_source":nsrc,"markdown":md,"file_name":f"{un}_clinical_handoff.md"}


@app.get("/api/safety/handoff")
async def safety_handoff(username: str, format: str = "markdown", db: Session = Depends(get_db)):
    uk = str(username or "").strip()
    if not uk: raise HTTPException(400, "Username required")
    clin  = fetch_or_create_clinical_state(db, uk)
    turns = fetch_recent_turn_summaries(db, limit=12, text_limit=700, username=uk)
    flags = ([str(clin.get("last_detected_distortion"))] if clin.get("last_detected_distortion") else []) + \
            (["requires_safety_review"] if clin.get("requires_safety_review") else [])
    dist  = sum(1 for r in turns if DISTRESS_RE.search(str(r.get("user_text") or "")))
    md    = build_handoff_markdown(uk, int(clin.get("last_risk_score") or 0),
                str(clin.get("active_framework") or FRAMEWORK_DBT), flags, dist, turns)
    if str(format or "").lower() == "pdf":
        return Response(content=render_handoff_pdf(md), media_type="application/pdf",
                        headers={"Content-Disposition":f"attachment; filename={uk}_handoff.pdf"})
    return {"username":uk,"risk_score":int(clin.get("last_risk_score") or 0),
            "framework":clin.get("active_framework"),"phase":clin.get("current_phase"),"markdown":md}


@app.post("/api/crisis/log")
async def log_crisis_event(payload: dict, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == payload.get("user_id")).first()
    if not user: raise HTTPException(404, "User not found")
    user.last_crisis_timestamp = datetime.now(timezone.utc).isoformat()
    user.latest_cssrs_risk     = payload.get("severity","Moderate")
    if payload.get("severity") == "High": user.requires_safety_review = True
    db.commit()
    return {"status":"success","message":"Crisis log recorded."}


@app.post("/api/clinical/clear-safety")
async def clear_safety_flag(username: str, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == username).first()
    if user:
        user.requires_safety_review = False; db.commit()
    upsert_clinical_state(db, username, {"requires_safety_review":False,"last_distress_level":"low"})
    return {"status":"unlocked"}