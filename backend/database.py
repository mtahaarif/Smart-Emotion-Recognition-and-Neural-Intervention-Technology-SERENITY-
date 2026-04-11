import os

from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker

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

	user = db.query(models.User).filter(models.User.username == username).first()
	if not user:
		user = models.User(username=username, password="")
		db.add(user)
		db.flush()

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
