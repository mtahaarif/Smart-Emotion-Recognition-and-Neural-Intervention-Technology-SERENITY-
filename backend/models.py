from sqlalchemy import Boolean, Column, Integer, String, Float, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship
from datetime import datetime

try:
    from .database import Base
except ImportError:
    from database import Base

# Replace your current User class in models.py with this unified version:
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)
    requires_safety_review = Column(Boolean, default=False)
    
    # --- NEW CLINICAL PERSISTENCE FIELDS ---
    last_crisis_timestamp = Column(String, nullable=True) # ISO format timestamp
    duty_to_warn = Column(Boolean, default=False) # Tarasoff Rule flag
    latest_cssrs_risk = Column(String, default="Unassessed") # Low, Moderate, High
    
    emergency_contact_name = Column(String, default="")
    emergency_contact_phone = Column(String, default="")
    
    # Relationships
    sessions = relationship("Session", back_populates="user")
    turns = relationship("ConversationTurn", back_populates="user", cascade="all, delete-orphan")
    questionnaire_results = relationship("QuestionnaireResult", back_populates="user", cascade="all, delete-orphan")
    clinical_state = relationship("ClinicalState", back_populates="user", uselist=False, cascade="all, delete-orphan")
    routing_events = relationship("ClinicalRoutingEvent", back_populates="user", cascade="all, delete-orphan")
    distortion_events = relationship("ClinicalDistortionEvent", back_populates="user", cascade="all, delete-orphan")
    safety_escalations = relationship("SafetyEscalationEvent", back_populates="user", cascade="all, delete-orphan")
    trajectory_snapshots = relationship("TrajectorySnapshot", back_populates="user", cascade="all, delete-orphan")
    diagnostics_samples = relationship("EdgeDiagnosticSample", back_populates="user", cascade="all, delete-orphan")

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


class ClinicalState(Base):
    __tablename__ = "clinical_states"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True, index=True)
    active_framework = Column(String, default="Supportive_Stabilization", index=True)
    current_phase = Column(String, default="Emotional Check-In")
    phase_index = Column(Integer, default=0)
    requires_safety_review = Column(Boolean, default=False)
    last_risk_score = Column(Integer, default=0)
    last_route_reason = Column(String, default="")
    last_detected_distortion = Column(String, default="")
    last_distress_level = Column(String, default="")
    updated_at = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="clinical_state")


class ClinicalRoutingEvent(Base):
    __tablename__ = "clinical_routing_events"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    turn_id = Column(Integer, ForeignKey("conversation_turns.id"), nullable=True, index=True)
    routed_framework = Column(String, nullable=False, index=True)
    route_reason = Column(String, default="")
    risk_score = Column(Integer, default=0)
    route_locked = Column(Boolean, default=False)
    acute_safety_trigger = Column(Boolean, default=False)
    rumination_detected = Column(Boolean, default=False)
    detected_distortion = Column(String, default="")
    dominant_emotion = Column(String, default="neutral")
    speech_emotion = Column(String, default="neutral")
    face_emotion = Column(String, default="neutral")
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="routing_events")


class ClinicalDistortionEvent(Base):
    __tablename__ = "clinical_distortion_events"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    turn_id = Column(Integer, ForeignKey("conversation_turns.id"), nullable=True, index=True)
    distortion_label = Column(String, nullable=False, index=True)
    framework = Column(String, default="Supportive_Stabilization")
    source_excerpt = Column(Text, default="")
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="distortion_events")


class SafetyEscalationEvent(Base):
    __tablename__ = "safety_escalation_events"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    turn_id = Column(Integer, ForeignKey("conversation_turns.id"), nullable=True, index=True)
    trigger_type = Column(String, default="")
    risk_score = Column(Integer, default=0)
    dominant_emotion = Column(String, default="neutral")
    transcript_excerpt = Column(Text, default="")
    handoff_markdown = Column(Text, default="")
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="safety_escalations")


class TrajectorySnapshot(Base):
    __tablename__ = "trajectory_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    questionnaire_type = Column(String, nullable=False, index=True)
    baseline_score = Column(Integer, default=0)
    latest_score = Column(Integer, default=0)
    delta_score = Column(Integer, default=0)
    window_days = Column(Integer, default=7)
    flagged = Column(Boolean, default=False)
    computed_at = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="trajectory_snapshots")


class EdgeDiagnosticSample(Base):
    __tablename__ = "edge_diagnostic_samples"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    source = Column(String, default="voice")
    stt_latency_ms = Column(Float, default=0.0)
    ser_latency_ms = Column(Float, default=0.0)
    fer_latency_ms = Column(Float, default=0.0)
    total_latency_ms = Column(Float, default=0.0)
    memory_mb = Column(Float, default=0.0)
    speech_confidence = Column(Float, default=0.0)
    face_confidence = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="diagnostics_samples")
