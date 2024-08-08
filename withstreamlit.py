import cv2
from deepface import DeepFace
import streamlit as st
from PIL import Image
import numpy as np

# Initialize Streamlit
st.title("Real-time Emotion Detection")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video from the default camera (usually the webcam)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while run:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        st.write("Failed to capture frame")
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format (DeepFace expects RGB format)
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]

        # Perform emotion analysis on the face ROI
        try:
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            # Determine the dominant emotion
            emotion = result[0]['dominant_emotion']

            # Draw rectangle around face and label with predicted emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        except Exception as e:
            st.write(f"Error analyzing face ROI: {e}")

    # Display the resulting frame
    FRAME_WINDOW.image(frame, channels="BGR")

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
