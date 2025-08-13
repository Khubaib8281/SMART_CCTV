import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import time

# Title
st.title("📹 Smart CCTV - Crowd Detection")
st.write("Real-time crowd detection using YOLOv8 and Streamlit")
st.set_page_config(layout="wide", page_title="📹 Smart CCTV")

# Load YOLO model     
model = YOLO("D:\PROGRAMMING\AI\SMART_CCTV\yolov8s.pt")

# Video upload
video_file = st.file_uploader("Upload a video or use webcam", type=["mp4", "avi", "mov"])

use_webcam = st.checkbox("Use webcam")

if use_webcam:
    video_source = 0
elif video_file:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(video_file.read())
    video_source = temp_file.name
else:
    video_source = None

if video_source is not None:
    cap = cv2.VideoCapture(video_source)
    stframe = st.empty()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        annotated_frame = results[0].plot()
        
        # Crowd detection logic (count people)
        person_count = sum(1 for c in results[0].boxes.cls if int(c) == 0)  # class 0 = person
        if person_count > 5:
            cv2.putText(annotated_frame, f"⚠ Crowd Alert: {person_count} people!", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        stframe.image(annotated_frame, channels="BGR")
        
        time.sleep(0.03)  # 30 FPS limit

    cap.release()

# streamlit run app.py  