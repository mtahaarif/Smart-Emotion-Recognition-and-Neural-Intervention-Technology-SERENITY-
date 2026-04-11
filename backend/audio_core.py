import logging
import os
from typing import Any, Dict, Optional

import librosa
import numpy as np

try:
    import tflite_runtime.interpreter as tflite  # type: ignore[import-not-found]
    TFLITE_BACKEND = "tflite-runtime"
except ImportError:
    import tensorflow.lite as tflite
    TFLITE_BACKEND = "tensorflow-lite"

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "ser_model.tflite")
DEFAULT_EMOTIONS = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]

DEFAULT_TFLITE_THREADS = max(1, (os.cpu_count() or 4) // 2)
SER_AUDIO_SAMPLE_RATE = int(os.getenv("SERENITY_SER_AUDIO_SAMPLE_RATE", "16000"))
SER_AUDIO_DURATION_SECONDS = float(os.getenv("SERENITY_SER_AUDIO_DURATION_SECONDS", "3"))
SER_AUDIO_OFFSET_SECONDS = float(os.getenv("SERENITY_SER_AUDIO_OFFSET_SECONDS", "0.5"))
SER_AUDIO_RESAMPLE_TYPE = os.getenv("SERENITY_SER_AUDIO_RESAMPLE_TYPE", "polyphase").strip() or "polyphase"

LOGGER = logging.getLogger(__name__)

_DEFAULT_RUNTIME: Optional[Dict[str, Any]] = None


def _safe_result(emotion: str = "Neutral", confidence: float = 0.0, error: str = "") -> Dict[str, Any]:
    return {
        "emotion": str(emotion),
        "confidence": float(confidence),
        "error": error,
    }


def _build_interpreter(model_path: str, num_threads: int) -> tuple[tflite.Interpreter, Optional[object]]:
    delegate_name = os.getenv("SERENITY_TFLITE_XNNPACK_DELEGATE", "libtensorflowlite_xnnpack_delegate.so")
    require_xnnpack = os.getenv("SERENITY_REQUIRE_XNNPACK", "false").strip().lower() == "true"

    delegate_loader = getattr(tflite, "load_delegate", None)
    if delegate_loader is None:
        experimental = getattr(tflite, "experimental", None)
        delegate_loader = getattr(experimental, "load_delegate", None) if experimental is not None else None

    delegate = None
    try:
        # On Windows dev boxes, default .so delegate probing is invalid; only try explicit delegate paths there.
        if os.name == "nt" and not (os.path.isabs(delegate_name) or os.path.exists(delegate_name)):
            raise RuntimeError("Skipping explicit XNNPACK delegate load on Windows without explicit delegate path")

        if delegate_loader is None:
            raise RuntimeError("No TFLite delegate loader is available in this TensorFlow build")

        delegate = delegate_loader(delegate_name)
        interpreter = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=[delegate],
            num_threads=num_threads,
        )
        LOGGER.info("SER interpreter initialized with XNNPACK delegate (%s)", delegate_name)
    except Exception as exc:
        if require_xnnpack:
            raise RuntimeError(f"SER XNNPACK delegate load failed: {exc}") from exc

        LOGGER.warning("SER XNNPACK delegate unavailable, continuing without explicit delegate: %s", exc)
        interpreter = tflite.Interpreter(model_path=model_path, num_threads=num_threads)

    interpreter.allocate_tensors()
    return interpreter, delegate


def initialize_audio_runtime(
    model_path: str = MODEL_PATH,
    num_threads: Optional[int] = None,
) -> Dict[str, Any]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"SER model not found at {model_path}")

    if num_threads is None:
        num_threads = int(
            os.getenv(
                "SERENITY_SER_TFLITE_THREADS",
                os.getenv("SERENITY_TFLITE_THREADS", str(DEFAULT_TFLITE_THREADS)),
            )
        )

    interpreter, delegate = _build_interpreter(model_path=model_path, num_threads=num_threads)
    runtime = {
        "interpreter": interpreter,
        "delegate": delegate,
        "input_details": interpreter.get_input_details(),
        "output_details": interpreter.get_output_details(),
        "labels": DEFAULT_EMOTIONS,
    }
    LOGGER.info("SER model preloaded from %s using %s", model_path, TFLITE_BACKEND)
    return runtime


def get_audio_runtime() -> Dict[str, Any]:
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        _DEFAULT_RUNTIME = initialize_audio_runtime()
    return _DEFAULT_RUNTIME


def _prepare_features(y: np.ndarray, sr: int, model_shape: np.ndarray) -> np.ndarray:
    """Build input tensor dynamically from model input shape.

    Supports SER layouts such as [1,T,F], [1,F], [1,T,F,1].
    """
    if len(model_shape) == 3:
        timesteps = int(model_shape[1])
        feature_dim = int(model_shape[2])
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=feature_dim).T
        if mfcc.shape[0] < timesteps:
            pad = np.zeros((timesteps - mfcc.shape[0], feature_dim), dtype=np.float32)
            mfcc = np.vstack([mfcc, pad])
        else:
            mfcc = mfcc[:timesteps, :]
        return np.expand_dims(mfcc, axis=0).astype(np.float32)

    if len(model_shape) == 2:
        feature_dim = int(model_shape[1])
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=feature_dim).T, axis=0)
        return np.expand_dims(mfcc, axis=0).astype(np.float32)

    if len(model_shape) == 4:
        timesteps = int(model_shape[1])
        feature_dim = int(model_shape[2])
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=feature_dim).T
        if mfcc.shape[0] < timesteps:
            pad = np.zeros((timesteps - mfcc.shape[0], feature_dim), dtype=np.float32)
            mfcc = np.vstack([mfcc, pad])
        else:
            mfcc = mfcc[:timesteps, :]
        return np.expand_dims(np.expand_dims(mfcc, axis=0), axis=-1).astype(np.float32)

    raise ValueError(f"Unsupported SER input shape: {model_shape}")


def predict_audio_emotion(file_path: str, runtime: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run SER inference using a preloaded runtime."""
    active_runtime = runtime or get_audio_runtime()
    interpreter = active_runtime["interpreter"]
    input_details = active_runtime["input_details"]
    output_details = active_runtime["output_details"]
    labels = active_runtime["labels"]

    try:
        try:
            y, sr = librosa.load(
                file_path,
                sr=SER_AUDIO_SAMPLE_RATE,
                mono=True,
                duration=SER_AUDIO_DURATION_SECONDS,
                offset=SER_AUDIO_OFFSET_SECONDS,
                res_type=SER_AUDIO_RESAMPLE_TYPE,
            )
        except ModuleNotFoundError as exc:
            if exc.name != "resampy":
                raise

            LOGGER.warning(
                "SER resampler '%s' requires resampy, retrying with scipy polyphase.",
                SER_AUDIO_RESAMPLE_TYPE,
            )
            y, sr = librosa.load(
                file_path,
                sr=SER_AUDIO_SAMPLE_RATE,
                mono=True,
                duration=SER_AUDIO_DURATION_SECONDS,
                offset=SER_AUDIO_OFFSET_SECONDS,
                res_type="polyphase",
            )

        if y.size == 0:
            return _safe_result(error="Audio input was empty after decoding")
        features = _prepare_features(y=y, sr=sr, model_shape=input_details[0]["shape"])

        interpreter.set_tensor(input_details[0]["index"], features)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])

        prediction_index = int(np.argmax(output_data))
        confidence = float(np.max(output_data))
        return _safe_result(
            emotion=labels[prediction_index],
            confidence=round(confidence * 100, 2),
        )
    except Exception as exc:
        LOGGER.exception("SER inference failed.")
        return _safe_result(error=f"TFLite inference failed: {exc}")
    finally:
        # Explicitly drop large arrays to return memory to the interpreter process quickly.
        if "output_data" in locals():
            del output_data
        if "features" in locals():
            del features
        if "y" in locals():
            del y