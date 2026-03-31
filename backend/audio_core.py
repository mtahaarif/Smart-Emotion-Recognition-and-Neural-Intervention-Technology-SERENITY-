import os
import numpy as np
import librosa
import tensorflow.lite as tflite
import logging
from typing import Any, Dict

# 1. Config
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "ser_model.tflite")
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


def _prepare_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Build input tensor dynamically from model input shape.

    Supports common SER layouts such as:
    - [1, timesteps, features]
    - [1, features]
    - [1, timesteps, features, 1]
    """
    model_shape = input_details[0]['shape']

    if len(model_shape) == 3:
        # Example: [1, 128, 42]
        timesteps = int(model_shape[1])
        feature_dim = int(model_shape[2])
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=feature_dim).T  # [frames, feature_dim]

        if mfcc.shape[0] < timesteps:
            pad = np.zeros((timesteps - mfcc.shape[0], feature_dim), dtype=np.float32)
            mfcc = np.vstack([mfcc, pad])
        else:
            mfcc = mfcc[:timesteps, :]

        return np.expand_dims(mfcc, axis=0).astype(np.float32)

    if len(model_shape) == 2:
        # Example: [1, 40]
        feature_dim = int(model_shape[1])
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=feature_dim).T, axis=0)
        return np.expand_dims(mfcc, axis=0).astype(np.float32)

    if len(model_shape) == 4:
        # Example: [1, 128, 42, 1]
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

# 2. Safe Loader (Prevents Crash)
if os.path.exists(MODEL_PATH):
    try:
        # Attempt to load the model
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        LOGGER.info("SER model loaded from %s", MODEL_PATH)
    except Exception as e:
        init_error = f"TFLite init failed: {e}"
        LOGGER.exception("SER model initialization failed.")
        interpreter = None
else:
    init_error = f"TFLite init failed: model not found at {MODEL_PATH}"
    LOGGER.error(init_error)

def predict_audio_emotion(file_path):
    """
    Runs audio inference. If model is broken, returns 'Neutral' 
    so the app doesn't crash.
    """
    if interpreter is None:
        return _safe_result(error=init_error or "TFLite init failed")

    try:
        # Load audio (3 seconds)
        y, sr = librosa.load(file_path, duration=3, offset=0.5)

        features = _prepare_features(y, sr)
        
        # Run Inference
        interpreter.set_tensor(input_details[0]['index'], features.astype(np.float32))
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Decode
        emotions = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
        prediction_index = np.argmax(output_data)
        confidence = float(np.max(output_data))
        
        return _safe_result(emotion=emotions[prediction_index], confidence=round(confidence * 100, 2))

    except Exception as e:
        LOGGER.exception("SER inference failed.")
        return _safe_result(error=f"TFLite inference failed: {e}")