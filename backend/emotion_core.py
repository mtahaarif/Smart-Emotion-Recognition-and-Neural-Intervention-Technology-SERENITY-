import base64
import logging
import os
from typing import Any, Dict, Optional

import cv2
import numpy as np

try:
    import tflite_runtime.interpreter as tflite  # type: ignore[import-not-found]
    TFLITE_BACKEND = "tflite-runtime"
except ImportError:
    import tensorflow.lite as tflite
    TFLITE_BACKEND = "tensorflow-lite"

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "fer_model.tflite")
DEFAULT_EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

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
        LOGGER.info("FER interpreter initialized with XNNPACK delegate (%s)", delegate_name)
    except Exception as exc:
        if require_xnnpack:
            raise RuntimeError(f"FER XNNPACK delegate load failed: {exc}") from exc

        LOGGER.warning("FER XNNPACK delegate unavailable, continuing without explicit delegate: %s", exc)
        interpreter = tflite.Interpreter(model_path=model_path, num_threads=num_threads)

    interpreter.allocate_tensors()
    return interpreter, delegate


def initialize_face_runtime(
    model_path: str = MODEL_PATH,
    num_threads: Optional[int] = None,
) -> Dict[str, Any]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"FER model not found at {model_path}")

    if num_threads is None:
        num_threads = int(os.getenv("SERENITY_TFLITE_THREADS", str(max(1, (os.cpu_count() or 4) - 1))))

    interpreter, delegate = _build_interpreter(model_path=model_path, num_threads=num_threads)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        raise RuntimeError("Failed to load OpenCV Haar cascade for FER face detection.")

    runtime = {
        "interpreter": interpreter,
        "delegate": delegate,
        "input_details": interpreter.get_input_details(),
        "output_details": interpreter.get_output_details(),
        "face_cascade": face_cascade,
        "labels": DEFAULT_EMOTIONS,
    }
    LOGGER.info("FER model preloaded from %s using %s", model_path, TFLITE_BACKEND)
    return runtime


def get_face_runtime() -> Dict[str, Any]:
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        _DEFAULT_RUNTIME = initialize_face_runtime()
    return _DEFAULT_RUNTIME


def analyze_face(image_data: str, runtime: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Decode base64 frame and run FER inference using a preloaded runtime."""
    active_runtime = runtime or get_face_runtime()
    interpreter = active_runtime["interpreter"]
    input_details = active_runtime["input_details"]
    output_details = active_runtime["output_details"]
    face_cascade = active_runtime["face_cascade"]
    labels = active_runtime["labels"]

    try:
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        np_img = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if frame is None:
            return _safe_result(error="Invalid frame data")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return _safe_result(emotion="No Face", confidence=0.0)

        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi_gray, (48, 48))
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(np.expand_dims(roi, axis=0), axis=-1)

        interpreter.set_tensor(input_details[0]["index"], roi)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])

        prediction_index = int(np.argmax(output_data))
        confidence = float(np.max(output_data))
        return _safe_result(
            emotion=labels[prediction_index],
            confidence=round(confidence * 100, 2),
        )
    except Exception as exc:
        LOGGER.exception("FER inference failed.")
        return _safe_result(error=f"TFLite inference failed: {exc}")
    finally:
        # Explicitly drop frame-sized arrays to reduce peak RSS between requests.
        if "output_data" in locals():
            del output_data
        if "roi" in locals():
            del roi
        if "roi_gray" in locals():
            del roi_gray
        if "faces" in locals():
            del faces
        if "gray" in locals():
            del gray
        if "frame" in locals():
            del frame
        if "np_img" in locals():
            del np_img