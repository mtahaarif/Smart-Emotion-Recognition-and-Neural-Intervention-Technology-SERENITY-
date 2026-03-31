import asyncio
import base64
import contextlib
import logging
import os
import tempfile
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from .audio_core import predict_audio_emotion
    from .emotion_core import analyze_face
    from .llm_core import init_rag_system
    from .database import SessionLocal, engine, fetch_recent_turns, persist_turn
    from . import models
except ImportError:
    # Allow direct module execution patterns where relative imports are unavailable.
    from audio_core import predict_audio_emotion
    from emotion_core import analyze_face
    from llm_core import init_rag_system
    from database import SessionLocal, engine, fetch_recent_turns, persist_turn
    import models

try:
    import whisper
except ImportError:
    whisper = None

try:
    import edge_tts
except ImportError:
    edge_tts = None

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# --- Pydantic Models ---
class AuthRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    message: str
    username: str


class DetectEmotionRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image payload.")
    user_message: Optional[str] = Field(default=None, description="Optional user text context.")


class EmotionAnalysisResponse(BaseModel):
    emotion: str
    confidence: float
    ai_message: str
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    rag_loaded: bool


class InteractResponse(BaseModel):
    dominant_emotion: str
    speech_emotion: str
    face_emotion: str
    transcription: str
    llm_response: str
    tts_audio_base64: Optional[str] = None
    errors: List[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    username: str
    message: str


app = FastAPI(title="SERENITY API", version="1.0.0")

# Create database tables on startup
models.Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

serenity_bot = None
whisper_model = None
rag_init_task = None
llm_generation_semaphore = asyncio.Semaphore(1)

EMOTION_LABELS = ["angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_ALIAS = {
    "surprised": "surprise",
    "fearful": "fear",
    "no face": "neutral",
}

WHISPER_TIMEOUT_SECONDS = 40
EMOTION_TIMEOUT_SECONDS = 20
LLM_TIMEOUT_SECONDS = 90
TTS_TIMEOUT_SECONDS = 30


# --- AUTH ENDPOINTS ---
@app.post("/register", response_model=AuthResponse)
async def register(payload: AuthRequest) -> AuthResponse:
    """Register a new user account."""
    if not payload.username or not payload.password:
        raise HTTPException(status_code=400, detail="Username and password required")
    
    db = SessionLocal()
    try:
        # Check if user exists
        existing_user = db.query(models.User).filter(
            models.User.username == payload.username
        ).first()
        
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")
        
        # Create new user
        new_user = models.User(username=payload.username, password=payload.password)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        LOGGER.info("New user registered: %s", payload.username)
        return AuthResponse(message="Registration successful", username=new_user.username)
    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        LOGGER.exception("Registration failed: %s", exc)
        raise HTTPException(status_code=500, detail="Registration failed") from exc
    finally:
        db.close()


@app.post("/login", response_model=AuthResponse)
async def login(payload: AuthRequest) -> AuthResponse:
    """Login with username and password."""
    if not payload.username or not payload.password:
        raise HTTPException(status_code=400, detail="Username and password required")
    
    db = SessionLocal()
    try:
        user = db.query(models.User).filter(
            models.User.username == payload.username,
            models.User.password == payload.password
        ).first()
        
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        LOGGER.info("User logged in: %s", payload.username)
        return AuthResponse(message="Login successful", username=user.username)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Login failed: %s", exc)
        raise HTTPException(status_code=500, detail="Login failed") from exc
    finally:
        db.close()


# --- HEALTH & EMOTION ENDPOINTS ---


def _normalize_prediction(prediction: object) -> tuple[str, float, Optional[str]]:
    if isinstance(prediction, dict):
        emotion = str(prediction.get("emotion", "Neutral"))
        confidence = float(prediction.get("confidence", 0.0))
        error = prediction.get("error")
        return emotion, confidence, str(error) if error else None

    if isinstance(prediction, tuple) and len(prediction) >= 2:
        return str(prediction[0]), float(prediction[1]), None

    return "Neutral", 0.0, "Invalid prediction payload"


def _to_probability_vector(emotion: str, confidence: float) -> Dict[str, float]:
    normalized_emotion = EMOTION_ALIAS.get(str(emotion).strip().lower(), str(emotion).strip().lower())
    if normalized_emotion not in EMOTION_LABELS:
        normalized_emotion = "neutral"

    confidence = float(confidence)
    if confidence > 1.0:
        confidence = confidence / 100.0
    confidence = max(0.0, min(confidence, 1.0))

    background = (1.0 - confidence) / (len(EMOTION_LABELS) - 1)
    probs = {label: background for label in EMOTION_LABELS}
    probs[normalized_emotion] = confidence
    return probs


def _fuse_emotions(
    speech_emotion: str,
    speech_confidence: float,
    face_emotion: str,
    face_confidence: float,
) -> str:
    speech_probs = _to_probability_vector(speech_emotion, speech_confidence)
    face_probs = _to_probability_vector(face_emotion, face_confidence)

    fused = {
        label: (speech_probs[label] + face_probs[label]) / 2.0
        for label in EMOTION_LABELS
    }
    dominant = max(fused.items(), key=lambda item: item[1])[0]
    return dominant.title()


def _serialize_turns(turns: list) -> List[Dict[str, str]]:
    history = []
    for turn in turns:
        history.append(
            {
                "user_text": str(getattr(turn, "user_text", "")),
                "assistant_text": str(getattr(turn, "assistant_text", "")),
                "emotion": str(getattr(turn, "dominant_emotion", "Neutral")),
            }
        )
    return history


async def _generate_tts_base64(text: str) -> tuple[Optional[str], Optional[str]]:
    if not text:
        return None, "TTS skipped: empty text"

    if edge_tts is None:
        return None, "TTS unavailable: edge-tts is not installed"

    audio_path = None
    try:
        audio_path = os.path.join(tempfile.gettempdir(), f"serenity_tts_{uuid.uuid4().hex}.mp3")
        communicator = edge_tts.Communicate(text=text, voice="en-US-JennyNeural", rate="+0%")
        await asyncio.wait_for(communicator.save(audio_path), timeout=TTS_TIMEOUT_SECONDS)

        with open(audio_path, "rb") as audio_file:
            encoded = base64.b64encode(audio_file.read()).decode("utf-8")

        return encoded, None
    except asyncio.TimeoutError:
        LOGGER.warning("TTS generation timed out after %s seconds", TTS_TIMEOUT_SECONDS)
        return None, "TTS timeout"
    except Exception as exc:
        LOGGER.exception("TTS generation failed: %s", exc)
        return None, f"TTS failed: {exc}"
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass


def _transcribe_with_whisper(audio_path: str) -> tuple[str, Optional[str]]:
    global whisper_model

    if whisper is None:
        return "", "Whisper unavailable: package not installed"

    if whisper_model is None:
        try:
            whisper_model = whisper.load_model("tiny", device="cpu")
            LOGGER.info("Whisper tiny model loaded on CPU.")
        except Exception as exc:
            LOGGER.exception("Whisper load failed: %s", exc)
            return "", f"Whisper load failed: {exc}"

    try:
        transcript = whisper_model.transcribe(audio_path, fp16=False, language="en")
        text = str(transcript.get("text", "")).strip()
        return text, None
    except Exception as exc:
        LOGGER.exception("Whisper transcription failed: %s", exc)
        return "", f"Whisper transcription failed: {exc}"


async def _generate_llm_response(
    user_text: str,
    dominant_emotion: str,
    serialized_history: List[Dict[str, str]],
) -> tuple[str, Optional[str]]:
    if serenity_bot is None:
        return "I am here with you. Let's take one small step together.", "RAG model unavailable"

    try:
        async with llm_generation_semaphore:
            response = await asyncio.wait_for(
                run_in_threadpool(
                    serenity_bot.generate_multimodal,
                    user_text,
                    dominant_emotion,
                    serialized_history,
                ),
                timeout=LLM_TIMEOUT_SECONDS,
            )
        return str(response), None
    except asyncio.TimeoutError:
        LOGGER.warning("LLM generation timed out after %s seconds", LLM_TIMEOUT_SECONDS)
        return "I am with you. Let's breathe slowly and focus on one manageable next step.", "LLM timeout"
    except Exception as exc:
        LOGGER.exception("LLM generation failed: %s", exc)
        return "I am here with you. Let's take one small step together.", f"LLM failed: {exc}"


@app.on_event("startup")
async def startup_event() -> None:
    global serenity_bot, rag_init_task

    async def _initialize_rag_background() -> None:
        global serenity_bot
        try:
            serenity_bot = await run_in_threadpool(init_rag_system)
            LOGGER.info("RAG system initialized successfully.")
        except Exception as exc:
            serenity_bot = None
            LOGGER.exception("Failed to initialize RAG system in background: %s", exc)

    if os.getenv("SERENITY_SKIP_RAG_STARTUP", "false").lower() == "true":
        serenity_bot = None
        LOGGER.warning("Skipping RAG startup initialization due to SERENITY_SKIP_RAG_STARTUP=true")
        return

    # Keep API responsive immediately; load RAG asynchronously in background.
    if rag_init_task is None or rag_init_task.done():
        rag_init_task = asyncio.create_task(_initialize_rag_background())
        LOGGER.info("RAG initialization started in background.")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global rag_init_task
    if rag_init_task is not None and not rag_init_task.done():
        rag_init_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await rag_init_task


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="running", rag_loaded=serenity_bot is not None)


@app.post("/detect_emotion", response_model=EmotionAnalysisResponse)
async def detect_emotion(payload: DetectEmotionRequest) -> EmotionAnalysisResponse:
    if not payload.image:
        raise HTTPException(status_code=400, detail="Missing image payload.")

    try:
        prediction = await run_in_threadpool(analyze_face, payload.image)
        emotion, confidence, inference_error = _normalize_prediction(prediction)
    except Exception as exc:
        LOGGER.exception("Face emotion inference failed: %s", exc)
        raise HTTPException(status_code=500, detail="Face emotion inference failed.") from exc

    ai_message = ""
    if serenity_bot is not None:
        try:
            prompt_message = payload.user_message or f"I am feeling {emotion}"
            ai_message = await run_in_threadpool(
                serenity_bot.generate,
                prompt_message,
                emotion,
            )
        except Exception as exc:
            LOGGER.exception("LLM generation failed for face path: %s", exc)
            ai_message = "I am here with you."

    return EmotionAnalysisResponse(
        emotion=str(emotion),
        confidence=float(confidence),
        ai_message=ai_message,
        error=inference_error,
    )


@app.post("/analyze_audio", response_model=EmotionAnalysisResponse)
async def analyze_audio(
    file: UploadFile = File(...),
    user_message: Optional[str] = Form(default=None),
) -> EmotionAnalysisResponse:
    if file is None:
        raise HTTPException(status_code=400, detail="No audio file provided.")

    temp_audio_path: Optional[str] = None
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio upload.")

        # Unique temp path per request prevents cross-request overwrite races.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name

        prediction = await run_in_threadpool(predict_audio_emotion, temp_audio_path)
        emotion, confidence, inference_error = _normalize_prediction(prediction)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Audio emotion inference failed: %s", exc)
        raise HTTPException(status_code=500, detail="Audio emotion inference failed.") from exc
    finally:
        await file.close()
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except OSError as exc:
                LOGGER.warning("Failed to remove temp audio file %s: %s", temp_audio_path, exc)

    ai_message = "I hear you."
    if serenity_bot is not None:
        try:
            prompt_message = user_message or f"I am speaking in a {emotion} tone"
            ai_message = await run_in_threadpool(
                serenity_bot.generate,
                prompt_message,
                emotion,
            )
        except Exception as exc:
            LOGGER.exception("LLM generation failed for audio path: %s", exc)
            ai_message = "Thank you for sharing that with me."

    return EmotionAnalysisResponse(
        emotion=str(emotion),
        confidence=float(confidence),
        ai_message=ai_message,
        error=inference_error,
    )


@app.post("/api/interact", response_model=InteractResponse)
async def interact(
    username: str = Form(...),
    image: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
    user_message: Optional[str] = Form(default=None),
) -> InteractResponse:
    if not username:
        raise HTTPException(status_code=400, detail="Missing username.")
    if not image and file is None and not user_message:
        raise HTTPException(status_code=400, detail="Provide at least one of image, audio, or text.")

    errors: List[str] = []
    temp_audio_path: Optional[str] = None
    audio_available = False
    image_available = bool(image)

    speech_emotion = "Neutral"
    speech_confidence = 0.0
    face_emotion = "Neutral"
    face_confidence = 0.0
    transcription = ""

    try:
        tasks: List[tuple[str, asyncio.Future]] = []

        if file is not None:
            audio_bytes = await file.read()
            if audio_bytes:
                suffix = ".wav"
                filename = (file.filename or "").lower()
                if filename.endswith(".webm"):
                    suffix = ".webm"
                elif filename.endswith(".mp3"):
                    suffix = ".mp3"

                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
                    temp_audio.write(audio_bytes)
                    temp_audio_path = temp_audio.name
                    audio_available = True

                tasks.append(
                    (
                        "transcribe",
                        asyncio.create_task(
                            asyncio.wait_for(
                                run_in_threadpool(_transcribe_with_whisper, temp_audio_path),
                                timeout=WHISPER_TIMEOUT_SECONDS,
                            )
                        ),
                    )
                )
                tasks.append(
                    (
                        "speech",
                        asyncio.create_task(
                            asyncio.wait_for(
                                run_in_threadpool(predict_audio_emotion, temp_audio_path),
                                timeout=EMOTION_TIMEOUT_SECONDS,
                            )
                        ),
                    )
                )
            else:
                errors.append("Audio upload was empty.")

        if image:
            tasks.append(
                (
                    "face",
                    asyncio.create_task(
                        asyncio.wait_for(
                            run_in_threadpool(analyze_face, image),
                            timeout=EMOTION_TIMEOUT_SECONDS,
                        )
                    ),
                )
            )

        if tasks:
            task_names = [name for name, _ in tasks]
            task_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

            for name, result in zip(task_names, task_results):
                if name == "transcribe":
                    if isinstance(result, asyncio.TimeoutError):
                        errors.append("Transcription timeout")
                    elif isinstance(result, Exception):
                        errors.append(f"Transcription failed: {result}")
                    else:
                        transcription, transcribe_error = result
                        if transcribe_error:
                            errors.append(transcribe_error)

                if name == "speech":
                    if isinstance(result, asyncio.TimeoutError):
                        errors.append("Speech emotion timeout")
                    elif isinstance(result, Exception):
                        errors.append(f"Speech emotion failed: {result}")
                    else:
                        speech_emotion, speech_confidence, speech_error = _normalize_prediction(result)
                        if speech_error:
                            errors.append(f"Speech model: {speech_error}")

                if name == "face":
                    if isinstance(result, asyncio.TimeoutError):
                        errors.append("Face emotion timeout")
                    elif isinstance(result, Exception):
                        errors.append(f"Face emotion failed: {result}")
                    else:
                        face_emotion, face_confidence, face_error = _normalize_prediction(result)
                        if face_error:
                            errors.append(f"Face model: {face_error}")

        if audio_available and image_available:
            dominant_emotion = _fuse_emotions(
                speech_emotion=speech_emotion,
                speech_confidence=speech_confidence,
                face_emotion=face_emotion,
                face_confidence=face_confidence,
            )
        elif audio_available:
            dominant_emotion = str(speech_emotion).title()
        elif image_available:
            dominant_emotion = str(face_emotion).title()
        else:
            dominant_emotion = "Neutral"

        user_text = (transcription or user_message or "I need help.").strip()
        llm_response = "I am here with you. Let's take one small step together."

        db = SessionLocal()
        try:
            history_turns = fetch_recent_turns(db, username=username, limit=6)
            serialized_history = _serialize_turns(history_turns)

            llm_response, llm_error = await _generate_llm_response(
                user_text=user_text,
                dominant_emotion=dominant_emotion,
                serialized_history=serialized_history,
            )
            if llm_error:
                errors.append(llm_error)

            try:
                persist_turn(
                    db,
                    username=username,
                    user_text=user_text,
                    assistant_text=llm_response,
                    dominant_emotion=dominant_emotion,
                    speech_emotion=speech_emotion,
                    face_emotion=face_emotion,
                )
            except Exception as exc:
                LOGGER.exception("Failed to persist conversation turn: %s", exc)
                errors.append(f"DB log failed: {exc}")
        finally:
            db.close()

        tts_audio_base64, tts_error = await _generate_tts_base64(llm_response)
        if tts_error:
            errors.append(tts_error)

        return InteractResponse(
            dominant_emotion=dominant_emotion,
            speech_emotion=str(speech_emotion),
            face_emotion=str(face_emotion),
            transcription=user_text,
            llm_response=llm_response,
            tts_audio_base64=tts_audio_base64,
            errors=errors,
        )
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("/api/interact failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Interaction failed: {exc}") from exc
    finally:
        if file is not None:
            await file.close()
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except OSError as exc:
                LOGGER.warning("Failed to remove temp interaction audio %s: %s", temp_audio_path, exc)


@app.post("/api/chat", response_model=InteractResponse)
async def chat(payload: ChatRequest) -> InteractResponse:
    username = (payload.username or "").strip()
    user_text = (payload.message or "").strip()
    if not username:
        raise HTTPException(status_code=400, detail="Missing username.")
    if not user_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    errors: List[str] = []
    dominant_emotion = "Neutral"
    llm_response = "I am here with you."

    db = SessionLocal()
    try:
        history_turns = fetch_recent_turns(db, username=username, limit=8)
        serialized_history = _serialize_turns(history_turns)

        llm_response, llm_error = await _generate_llm_response(
            user_text=user_text,
            dominant_emotion=dominant_emotion,
            serialized_history=serialized_history,
        )
        if llm_error:
            errors.append(llm_error)

        try:
            persist_turn(
                db,
                username=username,
                user_text=user_text,
                assistant_text=llm_response,
                dominant_emotion=dominant_emotion,
                speech_emotion="Neutral",
                face_emotion="Neutral",
            )
        except Exception as exc:
            LOGGER.exception("Failed to persist chat turn: %s", exc)
            errors.append(f"DB log failed: {exc}")
    finally:
        db.close()

    tts_audio_base64, tts_error = await _generate_tts_base64(llm_response)
    if tts_error:
        errors.append(tts_error)

    return InteractResponse(
        dominant_emotion=dominant_emotion,
        speech_emotion="Neutral",
        face_emotion="Neutral",
        transcription=user_text,
        llm_response=llm_response,
        tts_audio_base64=tts_audio_base64,
        errors=errors,
    )
