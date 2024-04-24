import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from scipy.spatial.distance import cosine

# Initialize MediaPipe solutions
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

# Load the MobileNetV2 model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze layers
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
])

# Simulated known faces database
known_faces_db = {
    "Mejbah Ahammad": np.random.rand(1, 64),
    "Rashedul Alam": np.random.rand(1, 64),
}

def preprocess_face_image(image):
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)
    return preprocess_input(image)

def extract_face(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
    face_images = [frame[y:y+h, x:x+w] for (x, y, w, h) in faces]
    return face_images, faces

def recognize_face(extracted_features, known_faces_db):
    best_match = "Unknown"
    lowest_dist = 1.0
    for name, features in known_faces_db.items():
        dist = cosine(features.flatten(), extracted_features.flatten())
        if dist < lowest_dist:
            lowest_dist = dist
            best_match = name
    return best_match, lowest_dist

# Real-time processing
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Pose and hand keypoints detection
    results_pose = pose.process(frame_rgb)
    results_hands = hands.process(frame_rgb)

    # Draw detections
    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Extract and process faces for recognition
    faces, face_coords = extract_face(frame)
    for face, (x, y, w, h) in zip(faces, face_coords):
        preprocessed_face = preprocess_face_image(face)
        extracted_features = model.predict(preprocessed_face)

        identity, similarity = recognize_face(extracted_features, known_faces_db)
        cv2.putText(frame, f"{identity}: {similarity:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Real-time Pose, Hand, and Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

pose.close()
hands.close()
video_capture.release()
cv2.destroyAllWindows()
