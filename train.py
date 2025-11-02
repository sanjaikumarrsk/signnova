import pandas as pd
import numpy as np
import os
import joblib 
import time

# --- Scikit-learn and Model Imports ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Configuration ---
# NOTE: Ensure this list matches your raw_images folders exactly
CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
    'nothing', 'space', 'del'
]
OUTPUT_CSV_PATH = os.path.join('data', 'processed_keypoints.csv')
MODEL_SAVE_PATH = os.path.join('server', 'model', 'gesture_model.pkl')
ENCODER_SAVE_PATH = os.path.join('server', 'model', 'label_encoder.pkl')


# ***************************************************************
# *** FEATURE EXTRACTION CODE MUST BE COMMENTED OUT OR REMOVED HERE ***
# ***************************************************************


# --- 1. Load the Processed Data (The FAST Part) ---
print("\n--- Starting Model Training ---")
start_time = time.time()

try:
    df = pd.read_csv(OUTPUT_CSV_PATH)
except FileNotFoundError:
    print(f"Error: CSV file not found at {OUTPUT_CSV_PATH}.")
    print("ACTION REQUIRED: Ensure the feature extraction step was run successfully first.")
    exit()

# Clean up any potential unnamed index columns
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# Separate features (X) from the target label (y)
X = df.drop('Label', axis=1) # All keypoint columns
y = df['Label'] # The sign name (A, B, C, etc.)

# --- 2. Encode the Labels ---
# Convert text labels ('A', 'B', etc.) into numbers (0, 1, 2, etc.)
le = LabelEncoder()
y_encoded = le.fit_transform(y) 


# --- 3. Split Data for Training and Testing ---
# Standard split: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)


# --- 4. Choose and Train the Model (Random Forest Classifier) ---
print(f"Training Random Forest on {len(X_train)} samples...")

# Initialize the model (using all available cores for speed)
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) 

# Train the model
model.fit(X_train, y_train)

# --- 5. Evaluate the Model ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Training Complete.")
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# --- 6. Save the Trained Model and Encoder ---

# Ensure the model directory exists
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# Save the trained model (.pkl is standard for scikit-learn)
joblib.dump(model, MODEL_SAVE_PATH) 

# Save the LabelEncoder (CRITICAL for server-side letter conversion)
joblib.dump(le, ENCODER_SAVE_PATH)

end_time = time.time()
print("\n--- Model Saving Complete ---")
print(f"Trained Model saved to: {MODEL_SAVE_PATH}")
print(f"Label Encoder saved to: {ENCODER_SAVE_PATH}")
print(f"Total Training Time: {round(end_time - start_time, 2)} seconds")