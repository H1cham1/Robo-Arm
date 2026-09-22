import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
)

MODEL = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Model eenmalig downloaden als hij nog niet naast hand.py staat
if not os.path.exists(MODEL):
    print("Model downloaden...")
    urllib.request.urlretrieve(MODEL_URL, MODEL)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
)
landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kan webcam niet openen (index 0).")
timestamp_ms = 0

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)  # spiegelen
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms += 33  # ~30 fps, moet strikt oplopen
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        # connecties tekenen
        for c in HandLandmarksConnections.HAND_CONNECTIONS:
            cv2.line(frame, points[c.start], points[c.end], (255, 255, 255), 2)
        # landmarks tekenen
        for p in points:
            cv2.circle(frame, p, 4, (0, 0, 255), -1)

        # wijsvingertop = landmark 8
        tip = landmarks[8]
        print(f"wijsvingertop x={tip.x:.3f} y={tip.y:.3f} z={tip.z:.3f}")

    cv2.imshow("hand", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Esc
        break

cap.release()
cv2.destroyAllWindows()
