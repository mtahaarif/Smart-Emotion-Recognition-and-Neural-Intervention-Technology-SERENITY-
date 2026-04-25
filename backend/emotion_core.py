import base64
import logging
import os
import threading
from typing import Any, Dict, Optional

import cv2
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
    _BACKEND = "tflite-runtime"
except ImportError:
    import tensorflow.lite as tflite
    _BACKEND = "tensorflow-lite"

MODEL_PATH  = os.path.join(os.path.dirname(__file__), "fer_model.tflite")
EMOTIONS    = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
MAX_SIDE    = int(os.getenv("SERENITY_FER_MAX_FRAME_SIDE", "640"))
SCALE_FACTOR= float(os.getenv("SERENITY_FER_FACE_SCALE_FACTOR", "1.2"))
MIN_NEIGHBORS=int(os.getenv("SERENITY_FER_FACE_MIN_NEIGHBORS", "5"))
MIN_FACE    = int(os.getenv("SERENITY_FER_FACE_MIN_SIZE", "48"))
LOGGER      = logging.getLogger(__name__)

_RUNTIME: Optional[Dict[str, Any]] = None
_LOCK = threading.Lock()

# Limit OpenCV to 1 thread on Pi — avoids GIL contention with TFLite
try:
    cv2.setNumThreads(int(os.getenv("SERENITY_FER_CV2_THREADS", "1")))
except Exception:
    pass


def initialize_face_runtime(model_path: str = MODEL_PATH) -> Dict[str, Any]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"FER model missing: {model_path}")

    threads  = int(os.getenv("SERENITY_FER_TFLITE_THREADS", max(1, (os.cpu_count() or 4) // 2)))
    del_path = os.getenv("SERENITY_TFLITE_XNNPACK_DELEGATE",
                         "libtensorflowlite_xnnpack_delegate.so").strip()
    use_ext  = os.getenv("SERENITY_TFLITE_USE_EXTERNAL_DELEGATE", "").strip().lower()
    want_ext = use_ext in {"1", "true", "yes"} if use_ext else (_BACKEND == "tensorflow-lite")

    # Skip delegate entirely on Windows unless it exists
    if os.name == "nt" and not os.path.exists(del_path):
        want_ext = False

    delegates = []
    if want_ext:
        try:
            delegates = [tflite.load_delegate(del_path)]
            LOGGER.info("FER: XNNPACK delegate loaded.")
        except Exception as exc:
            LOGGER.warning("FER: delegate unavailable, CPU fallback: %s", exc)

    interp = tflite.Interpreter(model_path=model_path, num_threads=threads,
                                 experimental_delegates=delegates)
    interp.allocate_tensors()

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        raise RuntimeError("OpenCV Haar cascade failed to load.")

    return {
        "interp":   interp,
        "in_det":   interp.get_input_details()[0],
        "out_det":  interp.get_output_details()[0],
        "cascade":  cascade,
        "lock":     threading.Lock(),
    }


def get_face_runtime() -> Dict[str, Any]:
    global _RUNTIME
    if _RUNTIME is None:
        with _LOCK:
            if _RUNTIME is None:
                _RUNTIME = initialize_face_runtime()
    return _RUNTIME


def analyze_face(image_data: str,
                 runtime: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rt = runtime or get_face_runtime()
    frame = roi = output = None
    try:
        # Decode base64 (strip data-URL prefix if present)
        raw = image_data.split(",", 1)[1] if "," in image_data else image_data
        try:
            buf = base64.b64decode(raw)
        except Exception:
            return {"emotion": "Neutral", "confidence": 0.0, "error": "Invalid base64"}

        arr   = np.frombuffer(buf, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            return {"emotion": "Neutral", "confidence": 0.0, "error": "Invalid frame"}

        # Downscale for faster cascade detection
        h, w = frame.shape
        if MAX_SIDE > 0 and max(h, w) > MAX_SIDE:
            s     = MAX_SIDE / float(max(h, w))
            frame = cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))),
                               interpolation=cv2.INTER_AREA)

        faces = rt["cascade"].detectMultiScale(
            frame, scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS, minSize=(MIN_FACE, MIN_FACE),
        )
        if not len(faces):
            return {"emotion": "No Face", "confidence": 0.0, "error": ""}

        x, y, wf, hf = max(faces, key=lambda f: f[2] * f[3])
        roi = cv2.resize(frame[y:y + hf, x:x + wf], (48, 48))
        roi = (roi.astype(np.float32) / 255.0)[None, :, :, None]   # (1,48,48,1)

        with rt["lock"]:
            rt["interp"].set_tensor(rt["in_det"]["index"], roi)
            rt["interp"].invoke()
            output = rt["interp"].get_tensor(rt["out_det"]["index"])[0]

        return {
            "emotion":    EMOTIONS[int(np.argmax(output))],
            "confidence": round(float(np.max(output)) * 100, 2),
            "error":      "",
        }
    except Exception as exc:
        LOGGER.exception("FER inference failed.")
        return {"emotion": "Neutral", "confidence": 0.0, "error": str(exc)}
    finally:
        frame = roi = output = None          # free RAM immediately on Pi