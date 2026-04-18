import json
import os
from datetime import datetime, timedelta
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


def persist_care_plan_checkin(
    db,
    username: str,
    mood_rating: int,
    stress_rating: int,
    energy_rating: int,
    sleep_hours: float,
    completed_targets: Optional[List[str]] = None,
    note: str = "",
):
    models = _get_models()
    user = _get_or_create_user(db, username, models)

    normalized_targets = [
        _clamp_text(str(item), 128)
        for item in (completed_targets or [])
        if str(item).strip()
    ]

    checkin = models.CarePlanCheckin(
        user_id=user.id,
        mood_rating=max(1, min(10, int(mood_rating))),
        stress_rating=max(1, min(10, int(stress_rating))),
        energy_rating=max(1, min(10, int(energy_rating))),
        sleep_hours=max(0.0, min(24.0, float(sleep_hours))),
        completed_targets_json=json.dumps(normalized_targets, separators=(",", ":")),
        note=_clamp_text(note, 800),
    )
    db.add(checkin)
    db.commit()
    db.refresh(checkin)
    return checkin


def fetch_care_plan_checkins(db, username: str, limit: int = 30) -> List[Dict[str, Any]]:
    models = _get_models()
    rows = (
        db.query(models.CarePlanCheckin)
        .join(models.User, models.CarePlanCheckin.user_id == models.User.id)
        .filter(models.User.username == username)
        .order_by(models.CarePlanCheckin.created_at.desc())
        .limit(max(1, limit))
        .all()
    )

    results: List[Dict[str, Any]] = []
    for row in rows:
        try:
            completed_targets = json.loads(row.completed_targets_json or "[]")
            if not isinstance(completed_targets, list):
                completed_targets = []
        except json.JSONDecodeError:
            completed_targets = []

        results.append(
            {
                "id": row.id,
                "mood_rating": int(row.mood_rating or 5),
                "stress_rating": int(row.stress_rating or 5),
                "energy_rating": int(row.energy_rating or 5),
                "sleep_hours": float(row.sleep_hours or 0.0),
                "completed_targets": [str(item) for item in completed_targets],
                "note": row.note or "",
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
        )
    return results


def persist_cbt_thought_record(
    db,
    username: str,
    situation: str,
    automatic_thought: str,
    emotion_label: str,
    intensity_before: int,
    cognitive_distortions: Optional[List[str]] = None,
    evidence_for: str = "",
    evidence_against: str = "",
    balanced_thought: str = "",
    intensity_after: int = 5,
    action_plan: str = "",
    created_at: Optional[datetime] = None,
):
    models = _get_models()
    user = _get_or_create_user(db, username, models)

    normalized_distortions = [
        _clamp_text(str(item), 64)
        for item in (cognitive_distortions or [])
        if str(item).strip()
    ]

    record = models.CBTThoughtRecord(
        user_id=user.id,
        situation=_clamp_text(situation, 1000),
        automatic_thought=_clamp_text(automatic_thought, 1000),
        emotion_label=_clamp_text(emotion_label, 64),
        intensity_before=max(0, min(10, int(intensity_before))),
        cognitive_distortions_json=json.dumps(normalized_distortions, separators=(",", ":")),
        evidence_for=_clamp_text(evidence_for, 1000),
        evidence_against=_clamp_text(evidence_against, 1000),
        balanced_thought=_clamp_text(balanced_thought, 1000),
        intensity_after=max(0, min(10, int(intensity_after))),
        action_plan=_clamp_text(action_plan, 1000),
        created_at=created_at or datetime.utcnow(),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def fetch_cbt_thought_records(db, username: str, limit: int = 50) -> List[Dict[str, Any]]:
    models = _get_models()
    rows = (
        db.query(models.CBTThoughtRecord)
        .join(models.User, models.CBTThoughtRecord.user_id == models.User.id)
        .filter(models.User.username == username)
        .order_by(models.CBTThoughtRecord.created_at.desc())
        .limit(max(1, limit))
        .all()
    )

    payload: List[Dict[str, Any]] = []
    for row in rows:
        try:
            distortions = json.loads(row.cognitive_distortions_json or "[]")
            if not isinstance(distortions, list):
                distortions = []
        except json.JSONDecodeError:
            distortions = []

        payload.append(
            {
                "id": row.id,
                "situation": row.situation or "",
                "automatic_thought": row.automatic_thought or "",
                "emotion_label": row.emotion_label or "",
                "intensity_before": int(row.intensity_before or 0),
                "cognitive_distortions": [str(item) for item in distortions],
                "evidence_for": row.evidence_for or "",
                "evidence_against": row.evidence_against or "",
                "balanced_thought": row.balanced_thought or "",
                "intensity_after": int(row.intensity_after or 0),
                "action_plan": row.action_plan or "",
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
        )
    return payload


def fetch_cbt_weekly_progress(db, username: str, days: int = 7) -> Dict[str, Any]:
    models = _get_models()
    window_days = max(1, min(int(days or 7), 30))
    cutoff = datetime.utcnow() - timedelta(days=window_days)

    rows = (
        db.query(models.CBTThoughtRecord)
        .join(models.User, models.CBTThoughtRecord.user_id == models.User.id)
        .filter(models.User.username == username)
        .filter(models.CBTThoughtRecord.created_at >= cutoff)
        .order_by(models.CBTThoughtRecord.created_at.desc())
        .all()
    )

    if not rows:
        return {
            "window_days": window_days,
            "total_records": 0,
            "avg_intensity_before": 0.0,
            "avg_intensity_after": 0.0,
            "avg_intensity_reduction": 0.0,
            "improvement_pct": 0.0,
            "completion_rate": 0.0,
            "streak_days": 0,
            "top_distortions": [],
            "trend": "insufficient_data",
        }

    total = len(rows)
    before_values = [int(row.intensity_before or 0) for row in rows]
    after_values = [int(row.intensity_after or 0) for row in rows]
    avg_before = round(sum(before_values) / max(1, total), 2)
    avg_after = round(sum(after_values) / max(1, total), 2)
    reduction = round(avg_before - avg_after, 2)
    improvement_pct = round((reduction / avg_before) * 100.0, 2) if avg_before > 0 else 0.0

    complete_count = 0
    distortion_counts: Dict[str, int] = {}
    days_with_records = set()

    for row in rows:
        if str(row.balanced_thought or "").strip() and str(row.action_plan or "").strip():
            complete_count += 1
        if row.created_at:
            days_with_records.add(row.created_at.date())

        try:
            distortions = json.loads(row.cognitive_distortions_json or "[]")
            if not isinstance(distortions, list):
                distortions = []
        except json.JSONDecodeError:
            distortions = []

        for distortion in distortions:
            key = str(distortion).strip().lower()
            if key:
                distortion_counts[key] = distortion_counts.get(key, 0) + 1

    completion_rate = round((complete_count / max(1, total)) * 100.0, 2)

    today = datetime.utcnow().date()
    streak = 0
    for i in range(window_days):
        day = today - timedelta(days=i)
        if day in days_with_records:
            streak += 1
        else:
            break

    ranked_distortions = sorted(
        [{"distortion": key, "count": count} for key, count in distortion_counts.items()],
        key=lambda item: item["count"],
        reverse=True,
    )

    if reduction >= 1.0 and improvement_pct >= 10.0:
        trend = "improving"
    elif reduction <= -0.5:
        trend = "worsening"
    else:
        trend = "stable"

    return {
        "window_days": window_days,
        "total_records": total,
        "avg_intensity_before": avg_before,
        "avg_intensity_after": avg_after,
        "avg_intensity_reduction": reduction,
        "improvement_pct": improvement_pct,
        "completion_rate": completion_rate,
        "streak_days": streak,
        "top_distortions": ranked_distortions[:6],
        "trend": trend,
    }