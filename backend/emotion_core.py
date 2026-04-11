import base64
import logging
import os
import threading
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

DEFAULT_TFLITE_THREADS = max(1, (os.cpu_count() or 4) // 2)
FER_MAX_FRAME_SIDE = int(os.getenv("SERENITY_FER_MAX_FRAME_SIDE", "640"))
FER_FACE_SCALE_FACTOR = float(os.getenv("SERENITY_FER_FACE_SCALE_FACTOR", "1.2"))
FER_FACE_MIN_NEIGHBORS = int(os.getenv("SERENITY_FER_FACE_MIN_NEIGHBORS", "5"))
FER_FACE_MIN_SIZE = int(os.getenv("SERENITY_FER_FACE_MIN_SIZE", "48"))
FER_CV2_THREADS = int(os.getenv("SERENITY_FER_CV2_THREADS", "1"))

LOGGER = logging.getLogger(__name__)

_DEFAULT_RUNTIME: Optional[Dict[str, Any]] = None
_RUNTIME_LOCK = threading.Lock()

if FER_CV2_THREADS > 0:
    try:
        cv2.setNumThreads(FER_CV2_THREADS)
    except Exception:
        # Best effort only; some builds may not expose thread controls.
        pass


def _safe_result(emotion: str = "Neutral", confidence: float = 0.0, error: str = "") -> Dict[str, Any]:
    return {
        "emotion": str(emotion),
        "confidence": float(confidence),
        "error": error,
    }


def _build_interpreter(model_path: str, num_threads: int) -> tuple[tflite.Interpreter, Optional[object]]:
    delegate_name = os.getenv("SERENITY_TFLITE_XNNPACK_DELEGATE", "libtensorflowlite_xnnpack_delegate.so")
    require_xnnpack = os.getenv("SERENITY_REQUIRE_XNNPACK", "false").strip().lower() == "true"

    # On Windows, rely on built-in CPU delegate path unless an explicit delegate library is provided.
    if os.name == "nt" and not (os.path.isabs(delegate_name) or os.path.exists(delegate_name)):
        interpreter = tflite.Interpreter(model_path=model_path, num_threads=num_threads)
        interpreter.allocate_tensors()
        LOGGER.info("FER using default TFLite CPU delegates on Windows.")
        return interpreter, None

    delegate_loader = getattr(tflite, "load_delegate", None)
    if delegate_loader is None:
        experimental = getattr(tflite, "experimental", None)
        delegate_loader = getattr(experimental, "load_delegate", None) if experimental is not None else None

    delegate = None
    try:
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
        num_threads = int(
            os.getenv(
                "SERENITY_FER_TFLITE_THREADS",
                os.getenv("SERENITY_TFLITE_THREADS", str(DEFAULT_TFLITE_THREADS)),
            )
        )

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
        "invoke_lock": threading.Lock(),
    }
    LOGGER.info("FER model preloaded from %s using %s", model_path, TFLITE_BACKEND)
    return runtime


def get_face_runtime() -> Dict[str, Any]:
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is not None:
        return _DEFAULT_RUNTIME

    with _RUNTIME_LOCK:
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
    invoke_lock = active_runtime["invoke_lock"]

    try:
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        try:
            image_bytes = base64.b64decode(image_data, validate=True)
        except Exception:
            return _safe_result(error="Invalid base64 image payload")

        np_img = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            return _safe_result(error="Invalid frame data")

        detection_frame = frame
        if FER_MAX_FRAME_SIDE > 0:
            frame_height, frame_width = frame.shape[:2]
            longest_side = max(frame_height, frame_width)
            if longest_side > FER_MAX_FRAME_SIDE:
                scale = FER_MAX_FRAME_SIDE / float(longest_side)
                detection_frame = cv2.resize(
                    frame,
                    (max(1, int(frame_width * scale)), max(1, int(frame_height * scale))),
                    interpolation=cv2.INTER_AREA,
                )

        faces = face_cascade.detectMultiScale(
            detection_frame,
            FER_FACE_SCALE_FACTOR,
            FER_FACE_MIN_NEIGHBORS,
            minSize=(FER_FACE_MIN_SIZE, FER_FACE_MIN_SIZE),
        )
        if len(faces) == 0:
            return _safe_result(emotion="No Face", confidence=0.0)

        # Track the most prominent face to avoid unstable emotion jumps with multi-face frames.
        (x, y, w, h) = max(faces, key=lambda item: int(item[2]) * int(item[3]))
        roi_gray = detection_frame[y:y + h, x:x + w]
        roi = cv2.resize(roi_gray, (48, 48))
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(np.expand_dims(roi, axis=0), axis=-1)

        with invoke_lock:
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
        if "frame" in locals():
            del frame
        if "detection_frame" in locals():
            del detection_frame
        if "np_img" in locals():
            del np_img
        if "image_bytes" in locals():
            del image_bytes