import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, joinedload, sessionmaker

# Local SQLite database
DATABASE_URL = "sqlite:///./serenity.db"
SQLITE_CACHE_KB = int(os.getenv("SERENITY_SQLITE_CACHE_KB", "20000"))

engine = create_engine(
	DATABASE_URL,
	connect_args={"check_same_thread": False},
	pool_pre_ping=True,
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
	cursor = dbapi_connection.cursor()
	cursor.execute("PRAGMA journal_mode=WAL")
	cursor.execute("PRAGMA synchronous=NORMAL")
	cursor.execute("PRAGMA temp_store=MEMORY")
	cursor.execute(f"PRAGMA cache_size=-{max(1024, SQLITE_CACHE_KB)}")
	cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def _clamp_text(text: str, max_chars: int = 4000) -> str:
	value = str(text or "").strip()
	if len(value) <= max_chars:
		return value
	return value[:max_chars]


def _get_or_create_user(db, username: str):
	try:
		from . import models
	except ImportError:
		import models

	user = db.query(models.User).filter(models.User.username == username).first()
	if not user:
		user = models.User(username=username, password="")
		db.add(user)
		db.flush()
	return user


def _decode_answers_json(payload: str) -> List[int]:
	try:
		raw = json.loads(payload or "[]")
	except Exception:
		return []

	if not isinstance(raw, list):
		return []

	result: List[int] = []
	for value in raw:
		try:
			result.append(int(value))
		except Exception:
			result.append(0)
	return result


def fetch_recent_turns(db, username: str, limit: int = 6):
	try:
		from . import models
	except ImportError:
		import models

	turns = (
		db.query(models.ConversationTurn)
		.join(models.User, models.ConversationTurn.user_id == models.User.id)
		.filter(models.User.username == username)
		.order_by(models.ConversationTurn.timestamp.desc())
		.limit(max(1, int(limit)))
		.all()
	)

	return list(reversed(turns))


def persist_turn(
	db,
	username: str,
	user_text: str,
	assistant_text: str,
	dominant_emotion: str,
	speech_emotion: str,
	face_emotion: str,
):
	try:
		from . import models
	except ImportError:
		import models

	user = _get_or_create_user(db, username)

	turn = models.ConversationTurn(
		user_id=user.id,
		user_text=_clamp_text(user_text),
		assistant_text=_clamp_text(assistant_text),
		dominant_emotion=_clamp_text(dominant_emotion, max_chars=32),
		speech_emotion=_clamp_text(speech_emotion, max_chars=32),
		face_emotion=_clamp_text(face_emotion, max_chars=32),
	)
	db.add(turn)
	db.commit()
	db.refresh(turn)
	return turn


def persist_questionnaire_result(
	db,
	username: str,
	questionnaire_type: str,
	answers: List[int],
	total_score: int,
	severity: str,
	created_at: Optional[datetime] = None,
):
	try:
		from . import models
	except ImportError:
		import models

	user = _get_or_create_user(db, username)

	result = models.QuestionnaireResult(
		user_id=user.id,
		questionnaire_type=_clamp_text(questionnaire_type, max_chars=16),
		answers_json=json.dumps(list(answers), ensure_ascii=True, separators=(",", ":")),
		total_score=int(total_score),
		severity=_clamp_text(severity, max_chars=32),
		created_at=created_at or datetime.utcnow(),
	)
	db.add(result)
	db.commit()
	db.refresh(result)
	return result


def questionnaire_result_to_dict(result, username: Optional[str] = None) -> Dict[str, Any]:
	return {
		"id": result.id,
		"username": username,
		"questionnaire_type": result.questionnaire_type,
		"answers": _decode_answers_json(result.answers_json),
		"total_score": int(result.total_score or 0),
		"severity": result.severity or "unknown",
		"created_at": result.created_at.isoformat() if result.created_at else None,
	}


def fetch_questionnaire_results(
	db,
	username: Optional[str] = None,
	questionnaire_type: Optional[str] = None,
	limit: int = 100,
) -> List[Dict[str, Any]]:
	try:
		from . import models
	except ImportError:
		import models

	query = (
		db.query(models.QuestionnaireResult, models.User.username)
		.join(models.User, models.QuestionnaireResult.user_id == models.User.id)
	)

	if username:
		query = query.filter(models.User.username == username)
	if questionnaire_type:
		query = query.filter(models.QuestionnaireResult.questionnaire_type == questionnaire_type)

	rows = (
		query.order_by(models.QuestionnaireResult.created_at.desc())
		.limit(max(1, int(limit)))
		.all()
	)

	return [questionnaire_result_to_dict(result, row_username) for result, row_username in rows]


def fetch_recent_turn_summaries(db, limit: int = 200) -> List[Dict[str, Any]]:
	try:
		from . import models
	except ImportError:
		import models

	rows = (
		db.query(models.ConversationTurn, models.User.username)
		.join(models.User, models.ConversationTurn.user_id == models.User.id)
		.order_by(models.ConversationTurn.timestamp.desc())
		.limit(max(1, int(limit)))
		.all()
	)

	output: List[Dict[str, Any]] = []
	for turn, username in rows:
		output.append(
			{
				"id": turn.id,
				"username": username,
				"user_text": turn.user_text,
				"assistant_text": turn.assistant_text,
				"dominant_emotion": turn.dominant_emotion,
				"speech_emotion": turn.speech_emotion,
				"face_emotion": turn.face_emotion,
				"timestamp": turn.timestamp.isoformat() if turn.timestamp else None,
			}
		)

	return output


def fetch_recent_sessions_with_emotions(db, limit: int = 100) -> List[Dict[str, Any]]:
	try:
		from . import models
	except ImportError:
		import models

	sessions = (
		db.query(models.Session)
		.options(joinedload(models.Session.emotions), joinedload(models.Session.user))
		.order_by(models.Session.timestamp.desc())
		.limit(max(1, int(limit)))
		.all()
	)

	items: List[Dict[str, Any]] = []
	for session in sessions:
		emotions = []
		for emotion in sorted(session.emotions, key=lambda value: value.timestamp or datetime.min):
			emotions.append(
				{
					"id": emotion.id,
					"emotion": emotion.emotion,
					"confidence": float(emotion.confidence or 0.0),
					"timestamp": emotion.timestamp.isoformat() if emotion.timestamp else None,
				}
			)

		items.append(
			{
				"id": session.id,
				"username": session.user.username if session.user else None,
				"timestamp": session.timestamp.isoformat() if session.timestamp else None,
				"conversation": session.conversation,
				"emotions": emotions,
			}
		)

	return items
