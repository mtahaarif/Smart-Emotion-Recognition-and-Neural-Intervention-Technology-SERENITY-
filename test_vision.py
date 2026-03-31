# Save this as backend/test_vision.py
from emotion_core import load_emotion_model, predict_emotion_from_bytes
import requests

# 1. Load the model
load_emotion_model()

# 2. Download a dummy face image for testing
print("Downloading test image...")
img_url = "https://raw.githubusercontent.com/microsoft/ferplus/master/ferplus/data/fer2013_new.csv/fer2013new.csv/fer2013new/Training/Happy/1.jpg"
try:
    # Use a dummy byte string if internet fails, or try to open a local image
    # For this test, we create a blank black image just to see if the pipeline crashes
    from PIL import Image
    import io
    
    # Create a dummy black image (48x48)
    img = Image.new('RGB', (48, 48), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()

    print("Running Prediction...")
    emotion, conf = predict_emotion_from_bytes(img_bytes)
    print(f"✅ SUCCESS! Detected: {emotion} ({conf:.2f})")
    
except Exception as e:
    print(f"❌ Failed: {e}")