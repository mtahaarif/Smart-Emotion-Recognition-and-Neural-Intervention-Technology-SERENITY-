from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime

try:
    from .database import Base
except ImportError:
    from database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)
    sessions = relationship("Session", back_populates="user")
    turns = relationship("ConversationTurn", back_populates="user", cascade="all, delete-orphan")

class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    conversation = Column(String)
    emotions = relationship("Emotion", back_populates="session")
    user = relationship("User", back_populates="sessions")

class Emotion(Base):
    __tablename__ = "emotions"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))
    emotion = Column(String)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    session = relationship("Session", back_populates="emotions")


class ConversationTurn(Base):
    __tablename__ = "conversation_turns"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    user_text = Column(String, nullable=False)
    assistant_text = Column(String, nullable=False)
    dominant_emotion = Column(String, default="Neutral")
    speech_emotion = Column(String, default="Neutral")
    face_emotion = Column(String, default="Neutral")
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="turns")
