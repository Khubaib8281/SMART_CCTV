import streamlit as st
from ultralytics import YOLO
import tempfile
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2

st.set_page_config(layout="wide", page_title="📹 Smart CCTV")
st.title("📹 Smart CCTV - Crowd Detection")
st.write("Real-time crowd detection using YOLOv8 and Streamlit")

# Cache model so it doesn't reload every time
@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")

model = load_model()

# -------------------
# Webcam option (browser-based)
# -------------------
class YOLOVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img)
        annotated_frame = results[0].plot()

        # Crowd detection logic
        person_count = sum(1 for c in results[0].boxes.cls if int(c) == 0)
        if person_count > 5:
            cv2.putText(
                annotated_frame,
                f"⚠ Crowd Alert: {person_count} people!",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

use_webcam = st.checkbox("Use webcam")

if use_webcam:
    webrtc_streamer(
        key="yolo-webcam",
        video_transformer_factory=YOLOVideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},  # Free STUN server
                # TURN server example (replace with your own)
                {
                    "urls": ["turn:your-turn-server-ip:3478"],
                    "username": "user",
                    "credential": "pass"
                }
            ]
        }
    )

else:
    # -------------------
    # Video file upload
    # -------------------
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if video_file:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(video_file.read())

        cap = cv2.VideoCapture(temp_file.name)
        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated_frame = results[0].plot()

            person_count = sum(1 for c in results[0].boxes.cls if int(c) == 0)
            if person_count > 5:
                cv2.putText(
                    annotated_frame,
                    f"⚠ Crowd Alert: {person_count} people!",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

            stframe.image(annotated_frame, channels="BGR")
            time.sleep(0.03)

        cap.release()

