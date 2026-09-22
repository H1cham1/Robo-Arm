import math
import os
import time
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


class OneEuroFilter:
    """Smoothing die weinig vertraagt bij snelle beweging en sterk filtert bij stilstand.

    min_cutoff: lager = gladder bij stilstand (maar trager)
    beta:       hoger = minder vertraging bij snelle beweging
    """

    def __init__(self, min_cutoff=0.5, beta=5.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


# Eén filter per landmark per as (21 landmarks x 3 assen)
filters = [[OneEuroFilter() for _ in range(3)] for _ in range(21)]

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_tracking_confidence=0.7,  # hoger = blijft de hand volgen i.p.v. elke frame opnieuw zoeken
)
landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kan webcam niet openen (index 0).")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # hogere resolutie vragen (webcam kiest dichtstbijzijnde)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# venster op volledig scherm, beeldverhouding blijft behouden
cv2.namedWindow("hand", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.setWindowProperty("hand", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
timestamp_ms = 0
smooth = True  # toets 'f' schakelt tussen gefilterd en rauw

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
        now = time.time()
        coords = [(lm.x, lm.y, lm.z) for lm in result.hand_landmarks[0]]
        if smooth:
            coords = [
                tuple(filters[i][axis](coords[i][axis], now) for axis in range(3))
                for i in range(21)
            ]
        points = [(int(x * w), int(y * h)) for x, y, _ in coords]

        # connecties tekenen
        for c in HandLandmarksConnections.HAND_CONNECTIONS:
            cv2.line(frame, points[c.start], points[c.end], (255, 255, 255), 2)
        # landmarks tekenen
        for p in points:
            cv2.circle(frame, p, 4, (0, 0, 255), -1)

        # wijsvingertop = landmark 8
        x, y, z = coords[8]
        print(f"wijsvingertop x={x:.3f} y={y:.3f} z={z:.3f}")
    else:
        # hand kwijt: filters resetten zodat ze niet vanaf een oude positie 'inzwaaien'
        for f_xyz in filters:
            for f in f_xyz:
                f.x_prev = None

    label = "smooth (f)" if smooth else "raw (f)"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("hand", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Esc
        break
    if key == ord("f"):
        smooth = not smooth

cap.release()
cv2.destroyAllWindows()
