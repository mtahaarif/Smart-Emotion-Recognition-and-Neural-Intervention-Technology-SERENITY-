import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, joinedload, sessionmaker

DATABASE_URL    = "sqlite:///./serenity.db"
SQLITE_CACHE_KB = max(1024, int(os.getenv("SERENITY_SQLITE_CACHE_KB", "20000")))

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,
)

@event.listens_for(engine, "connect")
def _set_pragmas(conn, _):
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute(f"PRAGMA cache_size=-{SQLITE_CACHE_KB};")
    cur.execute("PRAGMA mmap_size=268435456;")
    cur.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def apply_schema_migrations() -> None:
    """Additive migrations — safe to run on every startup."""
    with engine.begin() as conn:
        if not conn.exec_driver_sql(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
        ).fetchone():
            return
        cols = {r[1] for r in conn.exec_driver_sql("PRAGMA table_info(users)").fetchall()}
        for col, defn in [
            ("requires_safety_review",  "INTEGER DEFAULT 0"),
            ("emergency_contact_name",  "TEXT DEFAULT ''"),
            ("emergency_contact_phone", "TEXT DEFAULT ''"),
            ("duty_to_warn",            "INTEGER DEFAULT 0"),
            ("last_crisis_timestamp",   "TEXT"),
            ("latest_cssrs_risk",       "TEXT DEFAULT 'Unassessed'"),
        ]:
            if col not in cols:
                conn.exec_driver_sql(f"ALTER TABLE users ADD COLUMN {col} {defn}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _models():
    from backend import models
    return models

def _clamp(text: str, n: int = 4000) -> str:
    if not text: return ""
    return text[:n]

def _get_or_create_user(db, username: str, models):
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user:
        user = models.User(username=username, password="")
        db.add(user)
        db.flush()
    return user

def _parse_dt(value) -> Optional[datetime]:
    if not value: return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Clinical state
# ---------------------------------------------------------------------------
def fetch_or_create_clinical_state(db, username: str) -> Dict[str, Any]:
    m    = _models()
    user = _get_or_create_user(db, username, m)
    state = db.query(m.ClinicalState).filter(m.ClinicalState.user_id == user.id).first()
    if not state:
        state = m.ClinicalState(user_id=user.id)
        db.add(state)
        db.commit()
        db.refresh(state)
    return {
        "id": state.id, "user_id": user.id, "username": user.username,
        "active_framework": state.active_framework,
        "current_phase":    state.current_phase,
        "phase_index":      state.phase_index,
        "requires_safety_review": bool(state.requires_safety_review),
        "last_risk_score":        int(state.last_risk_score or 0),
        "last_route_reason":      state.last_route_reason or "",
        "last_detected_distortion": state.last_detected_distortion or "",
        "last_distress_level":    state.last_distress_level or "",
        "updated_at": state.updated_at.isoformat() if state.updated_at else None,
        "emergency_contact_name":  user.emergency_contact_name or "",
        "emergency_contact_phone": user.emergency_contact_phone or "",
    }


def upsert_clinical_state(db, username: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    m    = _models()
    user = _get_or_create_user(db, username, m)
    state = db.query(m.ClinicalState).filter(m.ClinicalState.user_id == user.id).first()
    if not state:
        state = m.ClinicalState(user_id=user.id)
        db.add(state)
    for k, v in (updates or {}).items():
        if hasattr(state, k):
            setattr(state, k, v)
    state.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(state)
    return fetch_or_create_clinical_state(db, username)


def update_user_emergency_contact(db, username: str, contact_name: str, contact_phone: str) -> Dict[str, str]:
    m    = _models()
    user = _get_or_create_user(db, username, m)
    user.emergency_contact_name  = _clamp(contact_name, 128)
    user.emergency_contact_phone = _clamp(contact_phone, 64)
    db.commit()
    return {"username": user.username,
            "emergency_contact_name":  user.emergency_contact_name,
            "emergency_contact_phone": user.emergency_contact_phone}


# ---------------------------------------------------------------------------
# Conversation turns
# ---------------------------------------------------------------------------
def persist_turn(db, username: str, user_text: str, assistant_text: str,
                 dominant_emotion: str, speech_emotion: str, face_emotion: str):
    m    = _models()
    user = _get_or_create_user(db, username, m)
    turn = m.ConversationTurn(
        user_id          = user.id,
        user_text        = _clamp(user_text),
        assistant_text   = _clamp(assistant_text),
        dominant_emotion = _clamp(dominant_emotion, 32),
        speech_emotion   = _clamp(speech_emotion, 32),
        face_emotion     = _clamp(face_emotion, 32),
    )
    db.add(turn)
    db.commit()
    return turn


def fetch_recent_turn_summaries(db, limit: int = 200, text_limit: Optional[int] = None,
                                 username: Optional[str] = None) -> List[Dict[str, Any]]:
    m = _models()
    q = (db.query(m.ConversationTurn, m.User.username)
           .join(m.User, m.ConversationTurn.user_id == m.User.id))
    if username:
        q = q.filter(m.User.username == username)
    rows = q.order_by(m.ConversationTurn.timestamp.desc()).limit(max(1, limit)).all()
    cl   = max(64, int(text_limit)) if text_limit else None
    return [{
        "id": t.id, "username": u,
        "user_text":      _clamp(t.user_text, cl) if cl else t.user_text,
        "assistant_text": _clamp(t.assistant_text, cl) if cl else t.assistant_text,
        "dominant_emotion": t.dominant_emotion, "speech_emotion": t.speech_emotion,
        "face_emotion": t.face_emotion,
        "timestamp": t.timestamp.isoformat() if t.timestamp else None,
    } for t, u in rows]


# ---------------------------------------------------------------------------
# Questionnaires
# ---------------------------------------------------------------------------
def persist_questionnaire_result(db, username: str, questionnaire_type: str,
                                  answers: List[int], total_score: int,
                                  severity: str, created_at: Optional[datetime] = None):
    m    = _models()
    user = _get_or_create_user(db, username, m)
    rec  = m.QuestionnaireResult(
        user_id            = user.id,
        questionnaire_type = _clamp(questionnaire_type, 16),
        answers_json       = json.dumps(answers, separators=(",", ":")),
        total_score        = int(total_score),
        severity           = _clamp(severity, 32),
        created_at         = created_at or datetime.utcnow(),
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return rec


def fetch_questionnaire_results(db, username: Optional[str] = None,
                                 questionnaire_type: Optional[str] = None,
                                 limit: int = 100,
                                 include_answers: bool = True) -> List[Dict[str, Any]]:
    m = _models()
    q = (db.query(m.QuestionnaireResult, m.User.username)
           .join(m.User, m.QuestionnaireResult.user_id == m.User.id))
    if username:
        q = q.filter(m.User.username == username)
    if questionnaire_type:
        q = q.filter(m.QuestionnaireResult.questionnaire_type == questionnaire_type)
    rows = q.order_by(m.QuestionnaireResult.created_at.desc()).limit(max(1, limit)).all()
    out  = []
    for r, u in rows:
        d = {
            "id": r.id, "username": u,
            "questionnaire_type": r.questionnaire_type,
            "total_score": r.total_score or 0,
            "severity":    r.severity or "unknown",
            "created_at":  r.created_at.isoformat() if r.created_at else None,
        }
        if include_answers:
            try:    d["answers"] = json.loads(r.answers_json or "[]")
            except: d["answers"] = []
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Clinical events
# ---------------------------------------------------------------------------
def persist_clinical_routing_event(db, username: str, routed_framework: str,
                                    route_reason: str, risk_score: int,
                                    route_locked: bool, acute_safety_trigger: bool,
                                    rumination_detected: bool, detected_distortion: str,
                                    dominant_emotion: str, speech_emotion: str,
                                    face_emotion: str, turn_id: Optional[int] = None):
    m    = _models()
    user = _get_or_create_user(db, username, m)
    row  = m.ClinicalRoutingEvent(
        user_id=user.id, turn_id=turn_id,
        routed_framework    =_clamp(routed_framework, 64),
        route_reason        =_clamp(route_reason, 300),
        risk_score          =int(risk_score or 0),
        route_locked        =bool(route_locked),
        acute_safety_trigger=bool(acute_safety_trigger),
        rumination_detected =bool(rumination_detected),
        detected_distortion =_clamp(detected_distortion or "", 64),
        dominant_emotion    =_clamp(dominant_emotion or "neutral", 32),
        speech_emotion      =_clamp(speech_emotion or "neutral", 32),
        face_emotion        =_clamp(face_emotion or "neutral", 32),
    )
    db.add(row); db.commit()
    return row


def persist_clinical_distortion_event(db, username: str, distortion_label: str,
                                       framework: str, source_excerpt: str,
                                       turn_id: Optional[int] = None):
    m    = _models()
    user = _get_or_create_user(db, username, m)
    row  = m.ClinicalDistortionEvent(
        user_id=user.id, turn_id=turn_id,
        distortion_label=_clamp(distortion_label, 64),
        framework       =_clamp(framework, 64),
        source_excerpt  =_clamp(source_excerpt, 2000),
    )
    db.add(row); db.commit()
    return row


def persist_safety_escalation_event(db, username: str, trigger_type: str,
                                     risk_score: int, dominant_emotion: str,
                                     transcript_excerpt: str, handoff_markdown: str,
                                     turn_id: Optional[int] = None):
    m    = _models()
    user = _get_or_create_user(db, username, m)
    row  = m.SafetyEscalationEvent(
        user_id=user.id, turn_id=turn_id,
        trigger_type      =_clamp(trigger_type, 128),
        risk_score        =int(risk_score or 0),
        dominant_emotion  =_clamp(dominant_emotion or "neutral", 32),
        transcript_excerpt=_clamp(transcript_excerpt, 4000),
        handoff_markdown  =_clamp(handoff_markdown, 20000),
    )
    db.add(row); db.commit()
    return row


# ---------------------------------------------------------------------------
# Trajectory snapshots
# ---------------------------------------------------------------------------
def replace_trajectory_snapshots(db, username: str,
                                  snapshots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    m    = _models()
    user = _get_or_create_user(db, username, m)
    db.query(m.TrajectorySnapshot).filter(m.TrajectorySnapshot.user_id == user.id).delete()
    db.flush()
    for s in (snapshots or []):
        computed = _parse_dt(s.get("computed_at")) or datetime.utcnow()
        db.add(m.TrajectorySnapshot(
            user_id            = user.id,
            questionnaire_type = _clamp(s.get("questionnaire_type") or "", 16),
            baseline_score     = int(s.get("baseline_score") or 0),
            latest_score       = int(s.get("latest_score") or 0),
            delta_score        = int(s.get("delta_score") or 0),
            window_days        = int(s.get("window_days") or 7),
            flagged            = bool(s.get("flagged", False)),
            computed_at        = computed,
        ))
    db.commit()
    return fetch_trajectory_snapshots(db, username)


def fetch_trajectory_snapshots(db, username: str) -> List[Dict[str, Any]]:
    m = _models()
    rows = (
        db.query(m.TrajectorySnapshot, m.User.username)
          .join(m.User, m.TrajectorySnapshot.user_id == m.User.id)
          .filter(m.User.username == username)
          .order_by(m.TrajectorySnapshot.questionnaire_type.asc())
          .all()
    )
    return [{
        "id": r.id, "username": u,
        "questionnaire_type": r.questionnaire_type,
        "baseline_score": int(r.baseline_score or 0),
        "latest_score":   int(r.latest_score or 0),
        "delta_score":    int(r.delta_score or 0),
        "window_days":    int(r.window_days or 7),
        "flagged":        bool(r.flagged),
        "computed_at":    r.computed_at.isoformat() if r.computed_at else None,
    } for r, u in rows]


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
def fetch_recent_sessions_with_emotions(db, limit: int = 100,
                                         conversation_limit: Optional[int] = None,
                                         username: Optional[str] = None) -> List[Dict[str, Any]]:
    m = _models()
    q = (db.query(m.Session)
           .options(joinedload(m.Session.emotions), joinedload(m.Session.user)))
    if username:
        q = q.join(m.User, m.Session.user_id == m.User.id).filter(m.User.username == username)
    sessions = q.order_by(m.Session.timestamp.desc()).limit(max(1, limit)).all()
    cl = max(64, int(conversation_limit)) if conversation_limit else None
    return [{
        "id": s.id,
        "username": s.user.username if s.user else None,
        "timestamp": s.timestamp.isoformat() if s.timestamp else None,
        "conversation": _clamp(s.conversation, cl) if cl else s.conversation,
        "emotions": [{"id": e.id, "emotion": e.emotion,
                      "confidence": float(e.confidence or 0.0),
                      "timestamp": e.timestamp.isoformat() if e.timestamp else None}
                     for e in sorted(s.emotions, key=lambda x: x.timestamp or datetime.min)],
    } for s in sessions]


# ---------------------------------------------------------------------------
# Edge diagnostics
# ---------------------------------------------------------------------------
def persist_edge_diagnostic_sample(db, username: Optional[str],
                                    sample: Dict[str, Any]) -> Dict[str, Any]:
    m    = _models()
    user = _get_or_create_user(db, username, m) if username else None
    row  = m.EdgeDiagnosticSample(
        user_id          = user.id if user else None,
        source           = _clamp(sample.get("source") or "voice", 32),
        stt_latency_ms   = float(sample.get("stt_latency_ms") or 0.0),
        ser_latency_ms   = float(sample.get("ser_latency_ms") or 0.0),
        fer_latency_ms   = float(sample.get("fer_latency_ms") or 0.0),
        total_latency_ms = float(sample.get("total_latency_ms") or 0.0),
        memory_mb        = float(sample.get("memory_mb") or 0.0),
        speech_confidence= float(sample.get("speech_confidence") or 0.0),
        face_confidence  = float(sample.get("face_confidence") or 0.0),
    )
    db.add(row); db.commit(); db.refresh(row)
    return {
        "id": row.id, "username": username, "source": row.source,
        "stt_latency_ms": row.stt_latency_ms, "ser_latency_ms": row.ser_latency_ms,
        "fer_latency_ms": row.fer_latency_ms, "total_latency_ms": row.total_latency_ms,
        "memory_mb": row.memory_mb, "speech_confidence": row.speech_confidence,
        "face_confidence": row.face_confidence,
        "timestamp": row.timestamp.isoformat() if row.timestamp else None,
    }


def fetch_recent_edge_diagnostics(db, username: Optional[str] = None,
                                   limit: int = 120) -> List[Dict[str, Any]]:
    m = _models()
    q = db.query(m.EdgeDiagnosticSample)
    if username:
        q = (q.join(m.User, m.EdgeDiagnosticSample.user_id == m.User.id)
              .filter(m.User.username == username))
    rows = q.order_by(m.EdgeDiagnosticSample.timestamp.desc()).limit(max(1, limit)).all()
    return [{
        "id": r.id, "source": r.source,
        "stt_latency_ms": float(r.stt_latency_ms or 0.0),
        "ser_latency_ms": float(r.ser_latency_ms or 0.0),
        "fer_latency_ms": float(r.fer_latency_ms or 0.0),
        "total_latency_ms": float(r.total_latency_ms or 0.0),
        "memory_mb": float(r.memory_mb or 0.0),
        "speech_confidence": float(r.speech_confidence or 0.0),
        "face_confidence":   float(r.face_confidence or 0.0),
        "timestamp": r.timestamp.isoformat() if r.timestamp else None,
    } for r in rows]


# ---------------------------------------------------------------------------
# Symptom trajectory
# ---------------------------------------------------------------------------
def calculate_symptom_trajectory(db, user_id: int) -> Dict[str, Any]:
    m = _models()
    _empty = {"user_id": None, "requires_safety_review": False,
              "history": {"PHQ-9": [], "GAD-7": [], "PCL-5": []},
              "velocity_delta": {"PHQ-9": None, "GAD-7": None},
              "time_series": [], "latest_scores": {}}
    try:
        uid = int(user_id)
    except (TypeError, ValueError):
        return _empty

    user = db.query(m.User).filter(m.User.id == uid).first()
    if not user:
        return {**_empty, "user_id": uid}

    tracked = ("PHQ-9", "GAD-7", "PCL-5")
    history: Dict[str, List[Dict[str, Any]]] = {}
    for qt in tracked:
        rows = (
            db.query(m.QuestionnaireResult)
              .filter(m.QuestionnaireResult.user_id == uid,
                      m.QuestionnaireResult.questionnaire_type == qt)
              .order_by(m.QuestionnaireResult.created_at.desc())
              .limit(3).all()
        )
        history[qt] = [{"id": r.id, "score": int(r.total_score or 0),
                         "severity": r.severity or "unknown",
                         "created_at": r.created_at.isoformat() if r.created_at else None}
                        for r in reversed(rows)]

    velocity: Dict[str, Optional[int]] = {"PHQ-9": None, "GAD-7": None}
    flag_vel = False
    for qt in ("PHQ-9", "GAD-7"):
        pts = history.get(qt) or []
        if len(pts) >= 2:
            delta = pts[-1]["score"] - pts[-2]["score"]
            velocity[qt] = delta
            if delta >= 5:
                flag_vel = True

    if flag_vel and not bool(getattr(user, "requires_safety_review", False)):
        user.requires_safety_review = True
        state = db.query(m.ClinicalState).filter(m.ClinicalState.user_id == uid).first()
        if state:
            state.requires_safety_review = True
            state.updated_at = datetime.utcnow()
        db.commit()

    # Build time-series
    timeline: Dict[str, Dict] = {}
    keys = {"PHQ-9": "phq9", "GAD-7": "gad7", "PCL-5": "pcl5"}
    for qt, pts in history.items():
        dk = keys[qt]
        for i, pt in enumerate(pts):
            ts = pt.get("created_at") or f"{qt}:{i}"
            bucket = timeline.setdefault(ts, {"timestamp": ts, "phq9": None, "gad7": None, "pcl5": None})
            bucket[dk] = pt["score"]

    def _sort_key(row):
        try:
            return datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00"))
        except ValueError:
            return datetime.min

    return {
        "user_id":               uid,
        "requires_safety_review": bool(getattr(user, "requires_safety_review", False)),
        "history":               history,
        "velocity_delta":        velocity,
        "time_series":           sorted(timeline.values(), key=_sort_key),
        "latest_scores":         {qt: pts[-1]["score"] for qt, pts in history.items() if pts},
    }