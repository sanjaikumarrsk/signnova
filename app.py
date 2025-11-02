import os
import io
import mediapipe as mp
import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import joblib

# --- Configuration ---
MODEL_DIR = 'server/model'
MODEL_PATH = os.path.join(MODEL_DIR, 'gesture_model.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')

# -

app = Flask(__name__)
CORS(app) 

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=0, # Optimized for speed
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load the trained model and label encoder
model = None
label_encoder = None
try:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")


# --- Feature Extraction Function ---
def extract_features(hand_landmarks):
    features = []
    if hand_landmarks:
        wrist = hand_landmarks.landmark[0]
        for landmark in hand_landmarks.landmark:
            features.extend([
                landmark.x - wrist.x, 
                landmark.y - wrist.y, 
                landmark.z - wrist.z
            ])
    if len(features) < 63:
        features.extend([0.0] * (63 - len(features)))
        
    return np.array(features[:63]).flatten()

# --- Helper Function for Landmark Data ---
def get_landmark_list(hand_landmarks):
    """Converts MediaPipe landmarks into a list of normalized (x, y, z) objects for JSON."""
    landmark_list = []
    for lm in hand_landmarks.landmark:
        landmark_list.append({
            'x': lm.x, 
            'y': lm.y,
            'z': lm.z
        })
    return landmark_list


# --- Flask API Route ---
@app.route('/classify_gesture', methods=['POST'])
def classify_gesture():
    if model is None or label_encoder is None:
        return jsonify({"gesture": "ERROR: Model not loaded", "landmarks": []}), 500

    if 'image' not in request.files:
        return jsonify({"gesture": "No image part in the request", "landmarks": []}), 400

    image_file = request.files['image']
    
    try:
        image_data = image_file.read()
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image.convert('RGB'))
        
        # Process the image
        image_np.flags.writeable = False 
        results = hands.process(image_np)
        image_np.flags.writeable = True 

        predicted_label = "No Hand Detected"
        landmark_data = []
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract landmarks for visualization
            landmark_data = get_landmark_list(hand_landmarks)
            
            # Prediction
            features = extract_features(hand_landmarks)
            prediction = model.predict([features])
            predicted_label = label_encoder.inverse_transform(prediction)[0]
            
        # Return both the gesture and the landmarks
        return jsonify({
            "gesture": predicted_label,
            "landmarks": landmark_data
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"gesture": "Prediction Error", "landmarks": []}), 500

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
