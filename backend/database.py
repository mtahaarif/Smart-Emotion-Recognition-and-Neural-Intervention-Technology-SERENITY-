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