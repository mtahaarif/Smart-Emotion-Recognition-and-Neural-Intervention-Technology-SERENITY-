from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Text
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
    questionnaire_results = relationship(
        "QuestionnaireResult",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    care_plan_checkins = relationship(
        "CarePlanCheckin",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    cbt_thought_records = relationship(
        "CBTThoughtRecord",
        back_populates="user",
        cascade="all, delete-orphan",
    )

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


class QuestionnaireResult(Base):
    __tablename__ = "questionnaire_results"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    questionnaire_type = Column(String, nullable=False, index=True)
    answers_json = Column(Text, nullable=False)
    total_score = Column(Integer, nullable=False)
    severity = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="questionnaire_results")


class CarePlanCheckin(Base):
    __tablename__ = "care_plan_checkins"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    mood_rating = Column(Integer, nullable=False, default=5)
    stress_rating = Column(Integer, nullable=False, default=5)
    energy_rating = Column(Integer, nullable=False, default=5)
    sleep_hours = Column(Float, nullable=False, default=7.0)
    completed_targets_json = Column(Text, nullable=False, default="[]")
    note = Column(Text, nullable=False, default="")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="care_plan_checkins")


class CBTThoughtRecord(Base):
    __tablename__ = "cbt_thought_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    situation = Column(Text, nullable=False, default="")
    automatic_thought = Column(Text, nullable=False, default="")
    emotion_label = Column(String, nullable=False, default="")
    intensity_before = Column(Integer, nullable=False, default=5)
    cognitive_distortions_json = Column(Text, nullable=False, default="[]")
    evidence_for = Column(Text, nullable=False, default="")
    evidence_against = Column(Text, nullable=False, default="")
    balanced_thought = Column(Text, nullable=False, default="")
    intensity_after = Column(Integer, nullable=False, default=5)
    action_plan = Column(Text, nullable=False, default="")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="cbt_thought_records")
