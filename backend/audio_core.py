import logging
import os
import threading
import warnings
import ctypes
from typing import Any, Dict, Optional
import librosa
import numpy as np
import scipy.signal
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_BACKEND = "tflite-runtime"
except ImportError:
    import tensorflow.lite as tflite
    TFLITE_BACKEND = "tensorflow-lite"

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "ser_model.tflite")
DEFAULT_EMOTIONS = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]

SER_AUDIO_SAMPLE_RATE = int(os.getenv("SERENITY_SER_AUDIO_SAMPLE_RATE", "16000"))
SER_AUDIO_DURATION_SECONDS = float(os.getenv("SERENITY_SER_AUDIO_DURATION_SECONDS", "3"))
SER_AUDIO_OFFSET_SECONDS = float(os.getenv("SERENITY_SER_AUDIO_OFFSET_SECONDS", "0.5"))

LOGGER = logging.getLogger(__name__)

_DEFAULT_RUNTIME: Optional[Dict[str, Any]] = None
_RUNTIME_LOCK = threading.Lock()

# Suppress librosa/PySoundFile warnings globally (Runs once at startup)
warnings.filterwarnings("ignore", message=r"PySoundFile failed.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r"librosa\.core\.audio\.__audioread_load.*", category=FutureWarning)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _is_delegate_loadable(delegate_name: str) -> bool:
    if not delegate_name:
        return False
    try:
        ctypes.CDLL(delegate_name)
        return True
    except OSError:
        return False


def _build_interpreter(model_path: str, num_threads: int) -> tuple[tflite.Interpreter, Optional[object]]:
    delegate_name = os.getenv("SERENITY_TFLITE_XNNPACK_DELEGATE", "libtensorflowlite_xnnpack_delegate.so").strip()
    # On tflite-runtime (Pi path), rely on built-in CPU/XNNPACK unless explicitly requested.
    use_external_delegate = _env_flag(
        "SERENITY_TFLITE_USE_EXTERNAL_DELEGATE",
        default=(TFLITE_BACKEND == "tensorflow-lite"),
    )

    if not use_external_delegate:
        interpreter = tflite.Interpreter(model_path=model_path, num_threads=num_threads)
        interpreter.allocate_tensors()
        LOGGER.info("SER initialized without external delegate (backend=%s).", TFLITE_BACKEND)
        return interpreter, None

    if not _is_delegate_loadable(delegate_name):
        LOGGER.warning(
            "XNNPACK delegate '%s' not loadable, using default CPU backend.",
            delegate_name,
        )
        interpreter = tflite.Interpreter(model_path=model_path, num_threads=num_threads)
        interpreter.allocate_tensors()
        return interpreter, None

    try:
        delegate = tflite.load_delegate(delegate_name)
        interpreter = tflite.Interpreter(
            model_path=model_path, 
            experimental_delegates=[delegate], 
            num_threads=num_threads
        )
        LOGGER.info("SER initialized with external XNNPACK delegate.")
    except Exception as exc:
        LOGGER.warning("XNNPACK delegate unavailable, using default CPU: %s", exc)
        delegate = None
        interpreter = tflite.Interpreter(model_path=model_path, num_threads=num_threads)

    interpreter.allocate_tensors()
    return interpreter, delegate


def initialize_audio_runtime(model_path: str = MODEL_PATH) -> Dict[str, Any]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"SER model not found at {model_path}")

    num_threads = int(os.getenv("SERENITY_SER_TFLITE_THREADS", max(1, (os.cpu_count() or 4) // 2)))
    interpreter, delegate = _build_interpreter(model_path, num_threads)
    
    # Cache the details so we don't have to call .get_input_details() on every request
    return {
        "interpreter": interpreter,
        "delegate": delegate,
        "input_details": interpreter.get_input_details()[0],
        "output_details": interpreter.get_output_details()[0],
        "labels": DEFAULT_EMOTIONS,
        "invoke_lock": threading.Lock(),
    }


def get_audio_runtime() -> Dict[str, Any]:
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        with _RUNTIME_LOCK:
            if _DEFAULT_RUNTIME is None:
                _DEFAULT_RUNTIME = initialize_audio_runtime()
    return _DEFAULT_RUNTIME


def _prepare_features(y: np.ndarray, sr: int, model_shape: list) -> np.ndarray:
    """Extract MFCCs and reshape instantly based on target model shape."""
    feature_dim = int(model_shape[2] if len(model_shape) in (3, 4) else model_shape[1])
    
    # Extract MFCCs once
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=feature_dim).T
    
    if len(model_shape) == 2:
        mfcc_mean = np.mean(mfcc, axis=0)
        return np.expand_dims(mfcc_mean, axis=0).astype(np.float32)

    # For 3D or 4D shapes, efficiently pad or truncate timesteps
    timesteps = int(model_shape[1])
    if mfcc.shape[0] < timesteps:
        pad_width = timesteps - mfcc.shape[0]
        mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:timesteps, :]

    # Return correct dimension expansions
    if len(model_shape) == 3:
        return np.expand_dims(mfcc, axis=0).astype(np.float32)
    if len(model_shape) == 4:
        return np.expand_dims(np.expand_dims(mfcc, axis=0), axis=-1).astype(np.float32)

    raise ValueError(f"Unsupported SER input shape: {model_shape}")

def predict_audio_emotion(file_path: str, runtime: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rt = runtime or get_audio_runtime()
    interpreter = rt["interpreter"]
    input_details = rt["input_details"]
    output_details = rt["output_details"]
    
    y = features = output_data = None
    
    try:
        # 1. Load native audio (sr=None completely bypasses the 'resampy' crash)
        y, sr = librosa.load(
            file_path,
            sr=None,
            mono=True,
            duration=SER_AUDIO_DURATION_SECONDS,
            offset=SER_AUDIO_OFFSET_SECONDS
        )

        if y.size == 0:
            return {"emotion": "Neutral", "confidence": 0.0, "error": "Audio input was empty"}

        # 2. Resample manually using scipy (which is already installed alongside librosa)
        if sr != SER_AUDIO_SAMPLE_RATE:
            num_samples = int(len(y) * SER_AUDIO_SAMPLE_RATE / sr)
            y = scipy.signal.resample(y, num_samples)
            sr = SER_AUDIO_SAMPLE_RATE

        features = _prepare_features(y, sr, input_details["shape"])

        with rt["invoke_lock"]:
            interpreter.set_tensor(input_details["index"], features)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details["index"])

        confidence = float(np.max(output_data))
        emotion = rt["labels"][int(np.argmax(output_data))]
        
        return {
            "emotion": emotion, 
            "confidence": round(confidence * 100, 2), 
            "error": ""
        }
        
    except Exception as exc:
        LOGGER.exception("SER inference failed.")
        return {"emotion": "Neutral", "confidence": 0.0, "error": f"Inference failed: {exc}"}
    finally:
        y = features = output_data = None