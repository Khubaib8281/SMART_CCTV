import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import time
import numpy as np
from tensorflow.keras.models import load_model
from playsound import playsound
import threading
import os

# -------------------
# Streamlit page setup
# -------------------
st.set_page_config(layout="wide", page_title="📹 VigilantEye", page_icon="👁️‍🗨️")

# -------------------
# Inject custom CSS for professional, techy theme
# -------------------
st.markdown(
    """
    <style>
    /* Page background */
    .stApp {
        background-color: #330000;
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Title styling */
    .title {
        color: #00ffc3;
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0px;
    }

    /* Subtitle / description */
    .subtitle {
        color: #a0a0a0;
        font-size: 20px;
        text-align: center;
        margin-bottom: 20px;
    }

    /* Video frame border */
    .stImage img {
        border-radius: 12px;
        border: 2px solid #00ffc3;
    }

    /* Checkbox & uploader font color */
    .css-1d391kg, .css-1n76uvr {
        color: #ffffff;
    }

    /* Sidebar style */
    [data-testid="stSidebar"] {
        background-color: #1a1c29;
        color: #ffffff;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# -------------------
# App title & subtitle
# -------------------
st.markdown('<div class="title">VigilantEye 👁️‍🗨️</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect crowds and violence in real-time or from uploaded videos</div>', unsafe_allow_html=True)

# -------------------
# Load models
# -------------------
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")  # crowd detector

@st.cache_resource
def load_violence_model():
    return load_model("modelnew.h5")  # violence detector

yolo_model = load_yolo_model()
violence_model = load_violence_model()

# -------------------
# Upload video / webcam
# -------------------
video_file = st.file_uploader("Upload a video (for web) or use webcam locally", type=["mp4", "avi", "mov"])
use_webcam = st.checkbox("Use webcam (local only)")

if use_webcam:
    video_source = 0
elif video_file:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(video_file.read())
    video_source = temp_file.name
else:
    video_source = None

# -------------------
# Helper to play sound in background
# -------------------
def buzz_sound():
    if os.path.exists("buzz.mp3"):
        playsound("buzz.mp3")
    else:
        print("Buzz! Violence detected with crowd >5")

# -------------------
# Process video
# -------------------
if video_source is not None:
    cap = cv2.VideoCapture(video_source)
    stframe = st.empty()
    
    sequence = []  # for violence detector
    FRAME_SIZE = (128, 128)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crowd detection (YOLO)
        results = yolo_model(frame)
        annotated_frame = results[0].plot()
        person_count = sum(1 for c in results[0].boxes.cls if int(c) == 0)
        if person_count > 5:
            cv2.putText(annotated_frame, f"⚠ Crowd Alert: {person_count} people!", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        
        # Violence detection
        resized = cv2.resize(frame, FRAME_SIZE) / 255.0
        sequence.append(resized)
        if len(sequence) > 1:
            clip = np.expand_dims(sequence[-1], axis=0)
            pred = violence_model.predict(clip, verbose=0)[0][0]
            label = "Violence" if pred > 0.5 else "Non-Violence"
            color = (0,0,255) if label=="Violence" else (0,255,0)
            cv2.putText(annotated_frame, f"{label} ({pred:.2f})", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            if person_count > 5 and label=="Violence":
                threading.Thread(target=buzz_sound).start()
        
        stframe.image(annotated_frame, channels="BGR")
        time.sleep(0.03)

    cap.release()


