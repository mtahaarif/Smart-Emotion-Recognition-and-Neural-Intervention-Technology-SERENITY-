import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, joinedload, sessionmaker

DATABASE_URL = "sqlite:///./serenity.db"
SQLITE_CACHE_KB = max(1024, int(os.getenv("SERENITY_SQLITE_CACHE_KB", "20000")))

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,
)

@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    # High-performance SQLite pragmas for Edge/ARM architecture
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.execute("PRAGMA temp_store=MEMORY;")
    cursor.execute(f"PRAGMA cache_size=-{SQLITE_CACHE_KB};")
    cursor.execute("PRAGMA mmap_size=268435456;") # 256MB mmap for blazing fast reads
    cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def apply_schema_migrations() -> None:
    """Apply additive SQLite migrations for existing local databases."""
    with engine.begin() as conn:
        users_exists = conn.exec_driver_sql(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
        ).fetchone()
        if not users_exists:
            return

        cols = {
            row[1]
            for row in conn.exec_driver_sql("PRAGMA table_info(users)").fetchall()
        }

        if "requires_safety_review" not in cols:
            conn.exec_driver_sql("ALTER TABLE users ADD COLUMN requires_safety_review INTEGER DEFAULT 0")
        if "emergency_contact_name" not in cols:
            conn.exec_driver_sql("ALTER TABLE users ADD COLUMN emergency_contact_name TEXT DEFAULT ''")
        if "emergency_contact_phone" not in cols:
            conn.exec_driver_sql("ALTER TABLE users ADD COLUMN emergency_contact_phone TEXT DEFAULT ''")

def _get_models():
    """Lazy load models centrally to completely avoid try/except import overhead."""
    from backend import models
    return models

def _clamp_text(text: str, max_chars: int = 4000) -> str:
    if not text: 
        return ""
    return text[:max_chars] if len(text) > max_chars else text.strip()

def _get_or_create_user(db, username: str, models):
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user:
        user = models.User(username=username, password="")
        db.add(user)
        db.flush() # Flush to assign an ID without committing the full transaction yet
    return user


def fetch_or_create_clinical_state(db, username: str) -> Dict[str, Any]:
    models = _get_models()
    user = _get_or_create_user(db, username, models)
    state = db.query(models.ClinicalState).filter(models.ClinicalState.user_id == user.id).first()
    if not state:
        state = models.ClinicalState(user_id=user.id)
        db.add(state)
        db.commit()
        db.refresh(state)

    return {
        "id": state.id,
        "user_id": user.id,
        "username": user.username,
        "active_framework": state.active_framework,
        "current_phase": state.current_phase,
        "phase_index": state.phase_index,
        "requires_safety_review": bool(state.requires_safety_review),
        "last_risk_score": int(state.last_risk_score or 0),
        "last_route_reason": state.last_route_reason or "",
        "last_detected_distortion": state.last_detected_distortion or "",
        "last_distress_level": state.last_distress_level or "",
        "updated_at": state.updated_at.isoformat() if state.updated_at else None,
        "emergency_contact_name": user.emergency_contact_name or "",
        "emergency_contact_phone": user.emergency_contact_phone or "",
    }


def upsert_clinical_state(db, username: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    models = _get_models()
    user = _get_or_create_user(db, username, models)
    state = db.query(models.ClinicalState).filter(models.ClinicalState.user_id == user.id).first()
    if not state:
        state = models.ClinicalState(user_id=user.id)
        db.add(state)

    for key, value in (updates or {}).items():
        if hasattr(state, key):
            setattr(state, key, value)

    state.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(state)
    return fetch_or_create_clinical_state(db, username)


def update_user_emergency_contact(db, username: str, contact_name: str, contact_phone: str) -> Dict[str, str]:
    models = _get_models()
    user = _get_or_create_user(db, username, models)
    user.emergency_contact_name = _clamp_text(contact_name or "", 128)
    user.emergency_contact_phone = _clamp_text(contact_phone or "", 64)
    db.commit()
    return {
        "username": user.username,
        "emergency_contact_name": user.emergency_contact_name or "",
        "emergency_contact_phone": user.emergency_contact_phone or "",
    }

def fetch_recent_turns(db, username: str, limit: int = 6):
    models = _get_models()
    turns = (
        db.query(models.ConversationTurn)
        .join(models.User, models.ConversationTurn.user_id == models.User.id)
        .filter(models.User.username == username)
        .order_by(models.ConversationTurn.timestamp.desc())
        .limit(limit)
        .all()
    )
    # Python C-level slicing is significantly faster than reversed()
    return turns[::-1] 

def persist_turn(db, username: str, user_text: str, assistant_text: str, dominant_emotion: str, speech_emotion: str, face_emotion: str):
    models = _get_models()
    user = _get_or_create_user(db, username, models)

    turn = models.ConversationTurn(
        user_id=user.id,
        user_text=_clamp_text(user_text),
        assistant_text=_clamp_text(assistant_text),
        dominant_emotion=_clamp_text(dominant_emotion, 32),
        speech_emotion=_clamp_text(speech_emotion, 32),
        face_emotion=_clamp_text(face_emotion, 32),
    )
    db.add(turn)
    db.commit()
    # Removed db.refresh() to save a redundant SELECT query
    return turn


def persist_clinical_routing_event(
    db,
    username: str,
    routed_framework: str,
    route_reason: str,
    risk_score: int,
    route_locked: bool,
    acute_safety_trigger: bool,
    rumination_detected: bool,
    detected_distortion: str,
    dominant_emotion: str,
    speech_emotion: str,
    face_emotion: str,
    turn_id: Optional[int] = None,
):
    models = _get_models()
    user = _get_or_create_user(db, username, models)
    event_row = models.ClinicalRoutingEvent(
        user_id=user.id,
        turn_id=turn_id,
        routed_framework=_clamp_text(routed_framework, 64),
        route_reason=_clamp_text(route_reason, 300),
        risk_score=int(risk_score or 0),
        route_locked=bool(route_locked),
        acute_safety_trigger=bool(acute_safety_trigger),
        rumination_detected=bool(rumination_detected),
        detected_distortion=_clamp_text(detected_distortion or "", 64),
        dominant_emotion=_clamp_text(dominant_emotion or "neutral", 32),
        speech_emotion=_clamp_text(speech_emotion or "neutral", 32),
        face_emotion=_clamp_text(face_emotion or "neutral", 32),
    )
    db.add(event_row)
    db.commit()
    return event_row


def persist_clinical_distortion_event(
    db,
    username: str,
    distortion_label: str,
    framework: str,
    source_excerpt: str,
    turn_id: Optional[int] = None,
):
    models = _get_models()
    user = _get_or_create_user(db, username, models)
    row = models.ClinicalDistortionEvent(
        user_id=user.id,
        turn_id=turn_id,
        distortion_label=_clamp_text(distortion_label, 64),
        framework=_clamp_text(framework, 64),
        source_excerpt=_clamp_text(source_excerpt, 2000),
    )
    db.add(row)
    db.commit()
    return row


def persist_safety_escalation_event(
    db,
    username: str,
    trigger_type: str,
    risk_score: int,
    dominant_emotion: str,
    transcript_excerpt: str,
    handoff_markdown: str,
    turn_id: Optional[int] = None,
):
    models = _get_models()
    user = _get_or_create_user(db, username, models)
    row = models.SafetyEscalationEvent(
        user_id=user.id,
        turn_id=turn_id,
        trigger_type=_clamp_text(trigger_type, 128),
        risk_score=int(risk_score or 0),
        dominant_emotion=_clamp_text(dominant_emotion or "neutral", 32),
        transcript_excerpt=_clamp_text(transcript_excerpt, 4000),
        handoff_markdown=_clamp_text(handoff_markdown, 20000),
    )
    db.add(row)
    db.commit()
    return row


def replace_trajectory_snapshots(db, username: str, snapshots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    models = _get_models()
    user = _get_or_create_user(db, username, models)

    db.query(models.TrajectorySnapshot).filter(models.TrajectorySnapshot.user_id == user.id).delete()
    db.flush()

    rows: List[Any] = []
    for snapshot in snapshots or []:
        computed = snapshot.get("computed_at")
        try:
            computed_at = datetime.fromisoformat(str(computed).replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            computed_at = datetime.utcnow()

        row = models.TrajectorySnapshot(
            user_id=user.id,
            questionnaire_type=_clamp_text(snapshot.get("questionnaire_type") or "", 16),
            baseline_score=int(snapshot.get("baseline_score") or 0),
            latest_score=int(snapshot.get("latest_score") or 0),
            delta_score=int(snapshot.get("delta_score") or 0),
            window_days=int(snapshot.get("window_days") or 7),
            flagged=bool(snapshot.get("flagged", False)),
            computed_at=computed_at,
        )
        rows.append(row)
        db.add(row)

    db.commit()
    return fetch_trajectory_snapshots(db, username)


def fetch_trajectory_snapshots(db, username: str) -> List[Dict[str, Any]]:
    models = _get_models()
    query = (
        db.query(models.TrajectorySnapshot, models.User.username)
        .join(models.User, models.TrajectorySnapshot.user_id == models.User.id)
        .filter(models.User.username == username)
        .order_by(models.TrajectorySnapshot.questionnaire_type.asc())
    )

    rows = query.all()
    return [
        {
            "id": row.id,
            "username": row_username,
            "questionnaire_type": row.questionnaire_type,
            "baseline_score": int(row.baseline_score or 0),
            "latest_score": int(row.latest_score or 0),
            "delta_score": int(row.delta_score or 0),
            "window_days": int(row.window_days or 7),
            "flagged": bool(row.flagged),
            "computed_at": row.computed_at.isoformat() if row.computed_at else None,
        }
        for row, row_username in rows
    ]


def persist_edge_diagnostic_sample(db, username: Optional[str], sample: Dict[str, Any]) -> Dict[str, Any]:
    models = _get_models()
    user = None
    if username:
        user = _get_or_create_user(db, username, models)

    row = models.EdgeDiagnosticSample(
        user_id=user.id if user else None,
        source=_clamp_text(sample.get("source") or "voice", 32),
        stt_latency_ms=float(sample.get("stt_latency_ms") or 0.0),
        ser_latency_ms=float(sample.get("ser_latency_ms") or 0.0),
        fer_latency_ms=float(sample.get("fer_latency_ms") or 0.0),
        total_latency_ms=float(sample.get("total_latency_ms") or 0.0),
        memory_mb=float(sample.get("memory_mb") or 0.0),
        speech_confidence=float(sample.get("speech_confidence") or 0.0),
        face_confidence=float(sample.get("face_confidence") or 0.0),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return {
        "id": row.id,
        "username": username,
        "source": row.source,
        "stt_latency_ms": row.stt_latency_ms,
        "ser_latency_ms": row.ser_latency_ms,
        "fer_latency_ms": row.fer_latency_ms,
        "total_latency_ms": row.total_latency_ms,
        "memory_mb": row.memory_mb,
        "speech_confidence": row.speech_confidence,
        "face_confidence": row.face_confidence,
        "timestamp": row.timestamp.isoformat() if row.timestamp else None,
    }


def fetch_recent_edge_diagnostics(db, username: Optional[str], limit: int = 120) -> List[Dict[str, Any]]:
    models = _get_models()
    query = db.query(models.EdgeDiagnosticSample)
    if username:
        query = query.join(models.User, models.EdgeDiagnosticSample.user_id == models.User.id).filter(models.User.username == username)

    rows = query.order_by(models.EdgeDiagnosticSample.timestamp.desc()).limit(max(1, int(limit))).all()
    return [
        {
            "id": row.id,
            "source": row.source,
            "stt_latency_ms": float(row.stt_latency_ms or 0.0),
            "ser_latency_ms": float(row.ser_latency_ms or 0.0),
            "fer_latency_ms": float(row.fer_latency_ms or 0.0),
            "total_latency_ms": float(row.total_latency_ms or 0.0),
            "memory_mb": float(row.memory_mb or 0.0),
            "speech_confidence": float(row.speech_confidence or 0.0),
            "face_confidence": float(row.face_confidence or 0.0),
            "timestamp": row.timestamp.isoformat() if row.timestamp else None,
        }
        for row in rows
    ]

def persist_questionnaire_result(db, username: str, questionnaire_type: str, answers: List[int], total_score: int, severity: str, created_at: Optional[datetime] = None):
    models = _get_models()
    user = _get_or_create_user(db, username, models)

    result = models.QuestionnaireResult(
        user_id=user.id,
        questionnaire_type=_clamp_text(questionnaire_type, 16),
        answers_json=json.dumps(answers, separators=(",", ":")),
        total_score=int(total_score),
        severity=_clamp_text(severity, 32),
        created_at=created_at or datetime.utcnow(),
    )
    db.add(result)
    db.commit()
    db.refresh(result) # Kept because the API needs to return the specific result ID
    return result

def fetch_questionnaire_results(db, username: Optional[str] = None, questionnaire_type: Optional[str] = None, limit: int = 100, include_answers: bool = True) -> List[Dict[str, Any]]:
    models = _get_models()
    query = db.query(models.QuestionnaireResult, models.User.username).join(models.User, models.QuestionnaireResult.user_id == models.User.id)

    if username:
        query = query.filter(models.User.username == username)
    if questionnaire_type:
        query = query.filter(models.QuestionnaireResult.questionnaire_type == questionnaire_type)

    rows = query.order_by(models.QuestionnaireResult.created_at.desc()).limit(max(1, limit)).all()

    results = []
    for result, row_username in rows:
        payload = {
            "id": result.id,
            "username": row_username,
            "questionnaire_type": result.questionnaire_type,
            "total_score": result.total_score or 0,
            "severity": result.severity or "unknown",
            "created_at": result.created_at.isoformat() if result.created_at else None,
        }
        if include_answers:
            try:
                payload["answers"] = json.loads(result.answers_json or "[]")
            except json.JSONDecodeError:
                payload["answers"] = []
        results.append(payload)
    return results


def calculate_symptom_trajectory(db, user_id: int) -> Dict[str, Any]:
    """Build 3-point questionnaire trajectories and enforce velocity-based safety flagging."""
    models = _get_models()

    try:
        normalized_user_id = int(user_id)
    except (TypeError, ValueError):
        return {
            "user_id": None,
            "requires_safety_review": False,
            "history": {"PHQ-9": [], "GAD-7": [], "PCL-5": []},
            "velocity_delta": {"PHQ-9": None, "GAD-7": None},
            "time_series": [],
            "latest_scores": {},
        }

    user = db.query(models.User).filter(models.User.id == normalized_user_id).first()
    if not user:
        return {
            "user_id": normalized_user_id,
            "requires_safety_review": False,
            "history": {"PHQ-9": [], "GAD-7": [], "PCL-5": []},
            "velocity_delta": {"PHQ-9": None, "GAD-7": None},
            "time_series": [],
            "latest_scores": {},
        }

    tracked_types = ("PHQ-9", "GAD-7", "PCL-5")
    history: Dict[str, List[Dict[str, Any]]] = {key: [] for key in tracked_types}

    for questionnaire_type in tracked_types:
        rows = (
            db.query(models.QuestionnaireResult)
            .filter(
                models.QuestionnaireResult.user_id == user.id,
                models.QuestionnaireResult.questionnaire_type == questionnaire_type,
            )
            .order_by(models.QuestionnaireResult.created_at.desc())
            .limit(3)
            .all()
        )

        # Convert to ascending time order for charting and delta computations.
        normalized_rows = list(reversed(rows))
        history[questionnaire_type] = [
            {
                "id": row.id,
                "score": int(row.total_score or 0),
                "severity": str(row.severity or "unknown"),
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in normalized_rows
        ]

    velocity_delta: Dict[str, Optional[int]] = {"PHQ-9": None, "GAD-7": None}
    velocity_trigger = False
    for questionnaire_type in ("PHQ-9", "GAD-7"):
        series = history.get(questionnaire_type) or []
        if len(series) >= 2:
            delta = int(series[-1].get("score") or 0) - int(series[-2].get("score") or 0)
            velocity_delta[questionnaire_type] = delta
            if delta >= 5:
                velocity_trigger = True

    if velocity_trigger and not bool(getattr(user, "requires_safety_review", False)):
        user.requires_safety_review = True
        state = db.query(models.ClinicalState).filter(models.ClinicalState.user_id == user.id).first()
        if state and not bool(getattr(state, "requires_safety_review", False)):
            state.requires_safety_review = True
            state.updated_at = datetime.utcnow()
        db.commit()

    def _series_key(questionnaire_type: str) -> str:
        if questionnaire_type == "PHQ-9":
            return "phq9"
        if questionnaire_type == "GAD-7":
            return "gad7"
        return "pcl5"

    def _display_date(timestamp_value: Optional[str]) -> str:
        if not timestamp_value:
            return "Unknown"
        try:
            parsed = datetime.fromisoformat(str(timestamp_value).replace("Z", "+00:00"))
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            return str(timestamp_value)

    timeline: Dict[str, Dict[str, Any]] = {}
    for questionnaire_type, points in history.items():
        data_key = _series_key(questionnaire_type)
        for idx, point in enumerate(points):
            timestamp = point.get("created_at") or f"{questionnaire_type}:{idx}"
            bucket = timeline.get(timestamp)
            if not bucket:
                bucket = {
                    "timestamp": timestamp,
                    "date": _display_date(str(timestamp) if isinstance(timestamp, str) else None),
                    "phq9": None,
                    "gad7": None,
                    "pcl5": None,
                }
                timeline[timestamp] = bucket
            bucket[data_key] = int(point.get("score") or 0)

    def _timeline_sort_key(item: Dict[str, Any]) -> datetime:
        raw = str(item.get("timestamp") or "")
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return datetime.min

    sorted_series = sorted(timeline.values(), key=_timeline_sort_key)

    latest_scores = {
        questionnaire_type: int(points[-1].get("score") or 0)
        for questionnaire_type, points in history.items()
        if points
    }

    return {
        "user_id": user.id,
        "requires_safety_review": bool(getattr(user, "requires_safety_review", False)),
        "history": history,
        "velocity_delta": velocity_delta,
        "time_series": sorted_series,
        "latest_scores": latest_scores,
    }

def fetch_recent_turn_summaries(db, limit: int = 200, text_limit: Optional[int] = None, username: Optional[str] = None) -> List[Dict[str, Any]]:
    models = _get_models()
    query = (
        db.query(models.ConversationTurn, models.User.username)
        .join(models.User, models.ConversationTurn.user_id == models.User.id)
    )
    if username:
        query = query.filter(models.User.username == username)

    rows = query.order_by(models.ConversationTurn.timestamp.desc()).limit(max(1, limit)).all()

    char_limit = max(64, int(text_limit)) if text_limit is not None else None
    
    return [{
        "id": turn.id,
        "username": username,
        "user_text": _clamp_text(turn.user_text, char_limit) if char_limit else turn.user_text,
        "assistant_text": _clamp_text(turn.assistant_text, char_limit) if char_limit else turn.assistant_text,
        "dominant_emotion": turn.dominant_emotion,
        "speech_emotion": turn.speech_emotion,
        "face_emotion": turn.face_emotion,
        "timestamp": turn.timestamp.isoformat() if turn.timestamp else None,
    } for turn, username in rows]

def fetch_recent_sessions_with_emotions(db, limit: int = 100, conversation_limit: Optional[int] = None, username: Optional[str] = None) -> List[Dict[str, Any]]:
    models = _get_models()
    query = db.query(models.Session).options(joinedload(models.Session.emotions), joinedload(models.Session.user))
    if username:
        query = query.join(models.User, models.Session.user_id == models.User.id).filter(models.User.username == username)

    sessions = query.order_by(models.Session.timestamp.desc()).limit(max(1, limit)).all()

    char_limit = max(64, int(conversation_limit)) if conversation_limit is not None else None
    
    return [{
        "id": session.id,
        "username": session.user.username if session.user else None,
        "timestamp": session.timestamp.isoformat() if session.timestamp else None,
        "conversation": _clamp_text(session.conversation, char_limit) if char_limit else session.conversation,
        "emotions": [{
            "id": e.id,
            "emotion": e.emotion,
            "confidence": float(e.confidence or 0.0),
            "timestamp": e.timestamp.isoformat() if e.timestamp else None
        } for e in sorted(session.emotions, key=lambda val: val.timestamp or datetime.min)]
    } for session in sessions]