import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# --- CONFIGURATION ---
MODEL_PATH = 'model_weights.h5'
EMOTION_DICT = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# --- LOAD RESOURCES ---
# Load the Haar Cascade for Face Detection (Standard OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Try to load the trained model, else use Mock Mode (For GitHub Demo)
try:
    if os.path.exists(MODEL_PATH):
        emotion_model = load_model(MODEL_PATH)
        print("✅ Trained Model Loaded Successfully")
        MODE = "ACTIVE"
    else:
        print("⚠️ Model file not found. Running in DEMO MODE (Face Detection Only)")
        MODE = "DEMO"
except Exception as e:
    print(f"Error loading model: {e}")
    MODE = "DEMO"

def start_camera():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (1280, 720))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect Faces
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Draw Bounding Box
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            
            if MODE == "ACTIVE":
                # Preprocess ROI for Model
                roi_gray = gray_frame[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

                # Predict
                prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                predicted_emotion = EMOTION_DICT[maxindex]
                
                # Display Text
                cv2.putText(frame, predicted_emotion, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            else:
                # Demo Mode Text
                cv2.putText(frame, "Analyzing...", (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Retail Vision - Customer Sentiment Analysis', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_camera()
