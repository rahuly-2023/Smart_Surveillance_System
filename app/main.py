# app/main.py (reconstructed directly from working version)

import streamlit as st
import cv2
import numpy as np
import time
from collections import deque
import mediapipe as mp
from tensorflow.keras.models import load_model
import threading
import queue
import os

# Initialize state
if 'run' not in st.session_state:
    st.session_state.run = False
if 'emergency' not in st.session_state:
    st.session_state.emergency = False
if 'latest_frame' not in st.session_state:
    st.session_state.latest_frame = None
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []
if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = 0

# Page config
st.set_page_config(page_title="Smart Surveillance", page_icon="🛡️", layout="wide")
st.title("🛡️ Smart Surveillance System")

# UI Layout
col1, col2 = st.columns([3, 1])
frame_display = col1.empty()

with col2:
    st.subheader("System Controls")
    run_checkbox = st.checkbox("Start Surveillance")
    st.session_state.run = run_checkbox

    st.markdown("---")
    st.subheader("Alert Status")
    alert_status = st.empty()
    alert_status.info("System ready. No threats detected.")

    st.markdown("---")
    st.subheader("Alert History")
    alert_box = st.empty()

# Constants
GENDER_LABELS = ['Female', 'Male']
GENDER_COLORS = {'Female': (255, 105, 180), 'Male': (0, 255, 255)}
FRAME_SEQUENCE_LENGTH = 10
FRAME_HEIGHT, FRAME_WIDTH = 64, 64
ALERT_COOLDOWN = 10
WAVE_COOLDOWN = 1.0
MIN_WAVE_DISTANCE = 0.2
IMG_SIZE = 96

# Buffers and flags
frame_buffer = deque(maxlen=FRAME_SEQUENCE_LENGTH)
x_history = deque(maxlen=15)
frame_lock = threading.Lock()

# Models
@st.cache_resource
def load_models():
    models = {}
    models['gender'] = load_model('app/models/Gender_detection.keras')
    models['violence'] = load_model('app/models/Violence_detection.h5', compile=False)
    mp_hands = mp.solutions.hands
    models['hands'] = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )
    return models

models = load_models()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Functions
def trigger_alert(reason):
    now = time.time()
    if now - st.session_state.last_alert_time > ALERT_COOLDOWN:
        st.session_state.emergency = True
        st.session_state.last_alert_time = now
        st.session_state.alert_history.insert(0, f"{time.strftime('%H:%M:%S')} - ALERT: {reason}")
        if len(st.session_state.alert_history) > 10:
            st.session_state.alert_history.pop()
        alert_status.error(f"🚨 EMERGENCY: {reason}", icon="⚠️")
        st.markdown('<audio autoplay><source src="https://www.soundjay.com/buttons/sounds/beep-07.mp3"></audio>', unsafe_allow_html=True)

def is_valid_wave(history, width):
    return len(history) >= 10 and (max(history) - min(history)) > width * MIN_WAVE_DISTANCE

def process_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w, _ = frame.shape
    gender_counts = {'Female': 0, 'Male': 0}
    violence_detected = False
    wave_detected = False

    # Gender
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w_f, h_f) in faces:
        face = frame[y:y+h_f, x:x+w_f]
        try:
            inp = cv2.resize(face, (IMG_SIZE, IMG_SIZE)) / 255.0
            inp = np.expand_dims(inp, axis=0)
            pred = models['gender'].predict(inp, verbose=0)
            gender = GENDER_LABELS[np.argmax(pred)]
            conf = np.max(pred)
            if conf > 0.8:
                color = GENDER_COLORS[gender]
                label = f"{gender} ({conf*100:.1f}%)"
                cv2.rectangle(rgb, (x, y), (x+w_f, y+h_f), color, 2)
                cv2.putText(rgb, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                gender_counts[gender] += 1
        except:
            continue

    # Violence
    resized = cv2.resize(rgb, (FRAME_WIDTH, FRAME_HEIGHT)) / 255.0
    frame_buffer.append(resized)
    if len(frame_buffer) == FRAME_SEQUENCE_LENGTH:
        input_tensor = np.expand_dims(frame_buffer, axis=0)
        pred = models['violence'].predict(input_tensor, verbose=0)[0][0]
        if pred > 0.6:
            violence_detected = True
            cv2.putText(rgb, "VIOLENCE DETECTED", (w//2 - 120, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            trigger_alert("Violence Detected")

    # Hand wave
    results = models['hands'].process(rgb)
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        mp.solutions.drawing_utils.draw_landmarks(rgb, lm, mp.solutions.hands.HAND_CONNECTIONS)
        wrist_x = int(lm.landmark[0].x * w)
        x_history.append(wrist_x)
        palm_open = all(lm.landmark[tip].y < lm.landmark[tip-2].y for tip in [8, 12, 16, 20])
        if palm_open and is_valid_wave(x_history, w):
            wave_detected = True
            trigger_alert("Distress Wave Detected")

    return rgb

# Webcam thread
cap = None

if st.session_state.run:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while st.session_state.run and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        st.session_state.latest_frame = frame
        processed = process_frame(frame)
        frame_display.image(processed, channels="RGB", use_container_width=True)
        alert_box.markdown("\n".join(st.session_state.alert_history))
        if (st.session_state.emergency and (time.time() - st.session_state.last_alert_time > ALERT_COOLDOWN / 2)):
            st.session_state.emergency = False
            alert_status.warning("Monitoring... No active threats", icon="👁️")
        time.sleep(0.01)

    cap.release()
else:
    col1.info("Enable surveillance to start monitoring")
    alert_status.info("System ready. No threats detected.")
    alert_box.markdown("\n".join(st.session_state.alert_history))
