import base64
import logging
import os
import threading
from typing import Any, Dict, Optional

import cv2
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_BACKEND = "tflite-runtime"
except ImportError:
    import tensorflow.lite as tflite
    TFLITE_BACKEND = "tensorflow-lite"

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "fer_model.tflite")
DEFAULT_EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

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
        pass


def _build_interpreter(model_path: str, num_threads: int) -> tuple[tflite.Interpreter, Optional[object]]:
    delegate_name = os.getenv("SERENITY_TFLITE_XNNPACK_DELEGATE", "libtensorflowlite_xnnpack_delegate.so")
    
    if os.name == "nt" and not os.path.exists(delegate_name):
        interpreter = tflite.Interpreter(model_path=model_path, num_threads=num_threads)
        interpreter.allocate_tensors()
        return interpreter, None

    try:
        delegate = tflite.load_delegate(delegate_name)
        interpreter = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=[delegate],
            num_threads=num_threads,
        )
        LOGGER.info("FER initialized with XNNPACK delegate.")
    except Exception as exc:
        LOGGER.warning("FER XNNPACK delegate unavailable, using CPU: %s", exc)
        delegate = None
        interpreter = tflite.Interpreter(model_path=model_path, num_threads=num_threads)

    interpreter.allocate_tensors()
    return interpreter, delegate


def initialize_face_runtime(model_path: str = MODEL_PATH) -> Dict[str, Any]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"FER model not found at {model_path}")

    num_threads = int(os.getenv("SERENITY_FER_TFLITE_THREADS", max(1, (os.cpu_count() or 4) // 2)))
    interpreter, delegate = _build_interpreter(model_path, num_threads)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        raise RuntimeError("Failed to load OpenCV Haar cascade.")

    return {
        "interpreter": interpreter,
        "delegate": delegate,
        "input_details": interpreter.get_input_details()[0],
        "output_details": interpreter.get_output_details()[0],
        "face_cascade": face_cascade,
        "labels": DEFAULT_EMOTIONS,
        "invoke_lock": threading.Lock(),
    }


def get_face_runtime() -> Dict[str, Any]:
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        with _RUNTIME_LOCK:
            if _DEFAULT_RUNTIME is None:
                _DEFAULT_RUNTIME = initialize_face_runtime()
    return _DEFAULT_RUNTIME


def analyze_face(image_data: str, runtime: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Decode base64 frame and run FER inference using a preloaded runtime."""
    rt = runtime or get_face_runtime()
    interpreter = rt["interpreter"]
    input_details = rt["input_details"]
    output_details = rt["output_details"]
    face_cascade = rt["face_cascade"]
    
    # Pre-declare massive variables for guaranteed instantaneous GC cleanup
    image_bytes = np_img = frame = faces = roi = output_data = None
    
    try:
        # 1. Base64 Decode
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        try:
            image_bytes = base64.b64decode(image_data)
        except Exception:
            return {"emotion": "Neutral", "confidence": 0.0, "error": "Invalid base64 payload"}

        # 2. OpenCV Decode to Grayscale
        np_img = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            return {"emotion": "Neutral", "confidence": 0.0, "error": "Invalid frame data"}

        # 3. Downscale for faster detection (overwrites existing frame to save RAM)
        if FER_MAX_FRAME_SIDE > 0:
            h, w = frame.shape
            longest_side = max(h, w)
            if longest_side > FER_MAX_FRAME_SIDE:
                scale = FER_MAX_FRAME_SIDE / float(longest_side)
                frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

        # 4. Detect Faces
        faces = face_cascade.detectMultiScale(
            frame,
            scaleFactor=FER_FACE_SCALE_FACTOR,
            minNeighbors=FER_FACE_MIN_NEIGHBORS,
            minSize=(FER_FACE_MIN_SIZE, FER_FACE_MIN_SIZE),
        )
        
        if len(faces) == 0:
            return {"emotion": "No Face", "confidence": 0.0, "error": ""}

        # 5. Extract Largest Face (Area = width * height)
        x, y, w_f, h_f = max(faces, key=lambda f: f[2] * f[3])
        roi = cv2.resize(frame[y:y + h_f, x:x + w_f], (48, 48))
        
        # 6. Normalize and Reshape Tensor
        roi = roi.astype(np.float32) / 255.0
        roi = np.expand_dims(np.expand_dims(roi, axis=0), axis=-1)

        # 7. TFLite Inference
        with rt["invoke_lock"]:
            interpreter.set_tensor(input_details["index"], roi)
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
        LOGGER.exception("FER inference failed.")
        return {"emotion": "Neutral", "confidence": 0.0, "error": f"Inference failed: {exc}"}
    finally:
        # Instantly decrease reference counts so Raspberry Pi RAM is freed immediately
        image_bytes = np_img = frame = faces = roi = output_data = None