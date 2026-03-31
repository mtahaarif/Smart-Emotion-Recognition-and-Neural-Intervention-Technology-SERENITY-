import cv2
import numpy as np
import base64
import tensorflow.lite as tflite
import os
import logging
from typing import Any, Dict

# ==========================================
# PART 1: SETUP (Load Model Once)
# ==========================================

# Path to your Face Emotion Model
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "fer_model.tflite")
interpreter = None
input_details = None
output_details = None
init_error = None

LOGGER = logging.getLogger(__name__)


def _safe_result(emotion: str = "Neutral", confidence: float = 0.0, error: str = "") -> Dict[str, Any]:
    return {
        "emotion": str(emotion),
        "confidence": float(confidence),
        "error": error,
    }

# Try to load the model
if os.path.exists(MODEL_PATH):
    try:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        LOGGER.info("FER model loaded successfully from %s", MODEL_PATH)
    except Exception as e:
        init_error = f"TFLite init failed: {e}"
        LOGGER.exception("FER model initialization failed.")
else:
    init_error = f"TFLite init failed: model not found at {MODEL_PATH}"
    LOGGER.error(init_error)

# ==========================================
# PART 2: ACTION (The Function app.py needs)
# ==========================================

def analyze_face(image_data):
    """
    Takes a base64 image string, detects a face, and returns the emotion.
    """
    # 1. Safety Check
    if interpreter is None:
        return _safe_result(error=init_error or "TFLite init failed")

    try:
        # 2. Decode the Base64 Image
        # Remove the header if present (e.g., "data:image/jpeg;base64,")
        if "," in image_data:
            image_data = image_data.split(",")[1]
        
        # Convert string to numpy array
        np_img = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        
        # 3. Pre-process for Model (Grayscale + Face Detection)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Load Face Detector (Haar Cascade)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # If no face is found, return early
        if len(faces) == 0:
            return _safe_result(emotion="No Face", confidence=0.0)

        # 4. Prepare Face for TFLite Model
        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        
        # Resize to 48x48 (Standard for FER models)
        roi = cv2.resize(roi_gray, (48, 48))
        
        # Normalize pixel values (0-255 -> 0.0-1.0)
        roi = roi.astype("float32") / 255.0
        
        # Expand dimensions to match model input shape: (1, 48, 48, 1)
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        # 5. Run Prediction
        interpreter.set_tensor(input_details[0]['index'], roi)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # 6. Decode the Result
        # Ensure this list matches the order of your training classes!
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        
        prediction_index = np.argmax(output_data)
        confidence = float(np.max(output_data))
        
        predicted_emotion = emotions[prediction_index]

        return _safe_result(emotion=predicted_emotion, confidence=round(confidence * 100, 2))
    
    except Exception as e:
        LOGGER.exception("FER inference failed.")
        return _safe_result(error=f"TFLite inference failed: {e}")