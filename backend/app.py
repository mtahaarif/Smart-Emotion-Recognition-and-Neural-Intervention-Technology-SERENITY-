import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from database import SessionLocal, engine
import models 

# --- AI CORES (Safe Import) ---
try:
    from emotion_core import analyze_face
    print("✅ Vision Core Loaded")
except ImportError:
    analyze_face = None
    print("⚠️ Vision Core Failed")

try:
    from audio_core import predict_audio_emotion
    print("✅ Audio Core Loaded")
except ImportError:
    predict_audio_emotion = None
    print("⚠️ Audio Core Failed")

try:
    from llm_core import init_rag_system
    serenity_bot = init_rag_system() # Load RAG immediately
    print("✅ RAG AI Loaded")
except Exception as e:
    serenity_bot = None
    print(f"⚠️ RAG AI Failed: {e}")

app = Flask(__name__)
CORS(app)
models.Base.metadata.create_all(bind=engine)

# --- ROUTES ---
@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400
    
    db = SessionLocal()
    
    # Check if user already exists
    existing_user = db.query(models.User).filter(models.User.username == username).first()
    if existing_user:
        db.close()
        return jsonify({"error": "Username already exists"}), 400
    
    # Create new user
    new_user = models.User(username=username, password=password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    db.close()
    
    return jsonify({"message": "Registration successful", "username": new_user.username}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    db = SessionLocal()
    user = db.query(models.User).filter(models.User.username == data.get('username'), models.User.password == data.get('password')).first()
    db.close()
    if user: return jsonify({"message": "Success", "username": user.username}), 200
    return jsonify({"error": "Invalid credentials"}), 401

@app.route('/detect_emotion', methods=['POST'])
def detect_face():
    if not analyze_face: return jsonify({"error": "Vision Module Offline"}), 503
    data = request.json
    emotion, confidence = analyze_face(data['image'])
    
    # AI Reacts to Face
    ai_msg = ""
    if serenity_bot and confidence > 80:
        ai_msg = serenity_bot.generate(f"I am feeling {emotion}", emotion_context=emotion)
        
    return jsonify({"emotion": emotion, "confidence": confidence, "ai_message": ai_msg})

@app.route('/analyze_audio', methods=['POST'])
def detect_voice():
    if not predict_audio_emotion: return jsonify({"error": "Audio Module Offline"}), 503
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    
    file = request.files['file']
    file.save("temp.wav")
    
    emotion, confidence = predict_audio_emotion("temp.wav")
    
    # AI Reacts to Voice
    ai_msg = "I hear you."
    if serenity_bot:
        ai_msg = serenity_bot.generate(f"I am speaking in a {emotion} tone", emotion_context=emotion)
        
    return jsonify({"emotion": emotion, "confidence": confidence, "ai_message": ai_msg})

if __name__ == '__main__':
    app.run(debug=True, port=5000)