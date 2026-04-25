import ctypes
import logging
import os
import threading
import warnings
from typing import Any, Dict, Optional

import librosa
import numpy as np
import scipy.signal

try:
    import tflite_runtime.interpreter as tflite
    _BACKEND = "tflite-runtime"
except ImportError:
    import tensorflow.lite as tflite
    _BACKEND = "tensorflow-lite"

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ser_model.tflite")
EMOTIONS   = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]
SR         = int(os.getenv("SERENITY_SER_AUDIO_SAMPLE_RATE", "16000"))
DUR        = float(os.getenv("SERENITY_SER_AUDIO_DURATION_SECONDS", "3"))
OFF        = float(os.getenv("SERENITY_SER_AUDIO_OFFSET_SECONDS", "0.5"))
LOGGER     = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message=r".*PySoundFile.*|.*__audioread_load.*")

_RUNTIME: Optional[Dict[str, Any]] = None
_LOCK = threading.Lock()


def _delegate_loadable(path: str) -> bool:
    try:
        ctypes.CDLL(path)
        return True
    except OSError:
        return False


def initialize_audio_runtime(model_path: str = MODEL_PATH) -> Dict[str, Any]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"SER model missing: {model_path}")

    threads   = int(os.getenv("SERENITY_SER_TFLITE_THREADS", max(1, (os.cpu_count() or 4) // 2)))
    del_path  = os.getenv("SERENITY_TFLITE_XNNPACK_DELEGATE", "libtensorflowlite_xnnpack_delegate.so").strip()
    use_ext   = os.getenv("SERENITY_TFLITE_USE_EXTERNAL_DELEGATE", "").strip().lower()
    want_ext  = use_ext in {"1", "true", "yes"} if use_ext else (_BACKEND == "tensorflow-lite")

    delegates = []
    if want_ext and _delegate_loadable(del_path):
        try:
            delegates = [tflite.load_delegate(del_path)]
            LOGGER.info("SER: XNNPACK delegate loaded.")
        except Exception as exc:
            LOGGER.warning("SER: delegate load failed, CPU fallback: %s", exc)

    interp = tflite.Interpreter(model_path=model_path, num_threads=threads,
                                 experimental_delegates=delegates)
    interp.allocate_tensors()
    return {
        "interp":   interp,
        "in_det":   interp.get_input_details()[0],
        "out_det":  interp.get_output_details()[0],
        "lock":     threading.Lock(),
    }


def get_audio_runtime() -> Dict[str, Any]:
    global _RUNTIME
    if _RUNTIME is None:
        with _LOCK:
            if _RUNTIME is None:
                _RUNTIME = initialize_audio_runtime()
    return _RUNTIME


def _prepare_features(y: np.ndarray, sr: int, shape: list) -> np.ndarray:
    ndim     = len(shape)
    feat_dim = shape[2] if ndim in (3, 4) else shape[1]
    mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=feat_dim).T   # (T, F)

    if ndim == 2:
        return np.mean(mfcc, axis=0, keepdims=True).astype(np.float32)   # (1, F)

    ts   = shape[1]
    diff = ts - mfcc.shape[0]
    mfcc = np.pad(mfcc, ((0, max(0, diff)), (0, 0)))[:ts]              # (ts, F)

    out = mfcc[None]                                                    # (1, ts, F)
    return (out[..., None] if ndim == 4 else out).astype(np.float32)


def predict_audio_emotion(file_path: str,
                          runtime: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rt = runtime or get_audio_runtime()
    try:
        y, orig_sr = librosa.load(file_path, sr=None, mono=True, duration=DUR, offset=OFF)
        if y.size == 0:
            return {"emotion": "Neutral", "confidence": 0.0, "error": "Empty audio"}

        # Polyphase resampling — ~80 % faster than Fourier on ARM
        if orig_sr != SR:
            y = scipy.signal.resample_poly(y, SR, orig_sr)

        feat = _prepare_features(y, SR, rt["in_det"]["shape"])

        with rt["lock"]:
            rt["interp"].set_tensor(rt["in_det"]["index"], feat)
            rt["interp"].invoke()
            out = rt["interp"].get_tensor(rt["out_det"]["index"])[0]

        return {
            "emotion":    EMOTIONS[int(np.argmax(out))],
            "confidence": round(float(np.max(out)) * 100, 2),
            "error":      "",
        }
    except Exception as exc:
        LOGGER.exception("SER inference failed.")
        return {"emotion": "Neutral", "confidence": 0.0, "error": str(exc)}