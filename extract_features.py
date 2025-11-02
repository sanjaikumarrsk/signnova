import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time

# --- Configuration ---
# 1. UPDATED CLASSES LIST: Matches the folders in your dataset.
CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
    'nothing', 'space', 'del'
]
DATA_PATH = os.path.join('data', 'raw_images')
OUTPUT_CSV_PATH = os.path.join('data', 'processed_keypoints.csv')

# 2. Initialize MediaPipe Hands
# static_image_mode=True is necessary for processing a large batch of images efficiently
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

all_data = [] # List to store all rows: [features] + [label]

print("Starting Feature Extraction...")
start_time = time.time()

# --- Core Processing Loop ---
for sign_class in CLASSES:
    class_path = os.path.join(DATA_PATH, sign_class)
    if not os.path.exists(class_path):
        print(f"Warning: Folder not found: {class_path}. Skipping.")
        continue
        
    print(f"Processing class: {sign_class}")
    
    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)
        
        try:
            # 1. Load and Prepare Image
            image = cv2.imread(image_path)
            if image is None:
                continue # Skip if file couldn't be loaded (e.g., corrupt file)
            
            # Convert BGR to RGB (required by MediaPipe)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False 
            
            # 2. Process Image and Detect Hands
            results = hands.process(image)
            
            # 3. Extract and Normalize Features
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    keypoints = []
                    
                    # Find the wrist (Landmark 0) for normalization
                    wrist_x = hand_landmarks.landmark[0].x
                    wrist_y = hand_landmarks.landmark[0].y
                    wrist_z = hand_landmarks.landmark[0].z
                    
                    # Loop through all 21 landmarks
                    for landmark in hand_landmarks.landmark:
                        # Normalize coordinates relative to the wrist (Landmark 0)
                        # This removes the hand's absolute position from the data.
                        normalized_x = landmark.x - wrist_x
                        normalized_y = landmark.y - wrist_y
                        normalized_z = landmark.z - wrist_z
                        
                        keypoints.extend([normalized_x, normalized_y, normalized_z])
                    
                    # 4. Add Label and Store Row (63 features + 1 label)
                    all_data.append(keypoints + [sign_class]) 

        except Exception as e:
            # Catch errors during image processing to prevent the script from crashing
            # print(f"Error processing {image_path}: {e}") 
            continue


# --- Final CSV Creation ---

# Create the column names (LM_0_x up to LM_20_z, plus Label)
# Note: We still list LM_0 (wrist) in the header, even if its normalized coordinates are near zero.
header = []
for i in range(21):
    header.extend([f'LM_{i}x', f'LM{i}y', f'LM{i}_z']) 
header.append('Label')

# Convert the list of lists into a Pandas DataFrame
df = pd.DataFrame(all_data, columns=header)

# Save the DataFrame to the CSV file
df.to_csv(OUTPUT_CSV_PATH, index=False)

end_time = time.time()
print("\n--- Extraction Complete ---")
print(f"Successfully processed {len(df)} total samples.")
print(f"Output saved to: {OUTPUT_CSV_PATH}")
print(f"Time taken: {round(end_time - start_time, 2)} seconds")

