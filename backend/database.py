from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Local SQLite database
DATABASE_URL = "sqlite:///./serenity.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def fetch_recent_turns(db, username: str, limit: int = 6):
	try:
		from . import models
	except ImportError:
		import models

	user = db.query(models.User).filter(models.User.username == username).first()
	if not user:
		return []

	turns = (
		db.query(models.ConversationTurn)
		.filter(models.ConversationTurn.user_id == user.id)
		.order_by(models.ConversationTurn.timestamp.desc())
		.limit(limit)
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
		db.commit()
		db.refresh(user)

	turn = models.ConversationTurn(
		user_id=user.id,
		user_text=user_text,
		assistant_text=assistant_text,
		dominant_emotion=dominant_emotion,
		speech_emotion=speech_emotion,
		face_emotion=face_emotion,
	)
	db.add(turn)
	db.commit()
	db.refresh(turn)
	return turn
