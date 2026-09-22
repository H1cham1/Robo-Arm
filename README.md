# robo-arm

Stap 1: hand tracking via de webcam met MediaPipe. De 21 hand-landmarks worden live op het beeld getekend en de positie van de wijsvingertop wordt naar de terminal geprint.

## Installatie (Windows)

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python hand.py
```

Bij de eerste start wordt het model `hand_landmarker.task` (ca. 8 MB) automatisch gedownload.

Druk op **Esc** om af te sluiten.

## Opmerking

Python 3.13 heeft geen mediapipe-versie meer met `mp.solutions`. Daarom gebruikt `hand.py` de nieuwere Tasks-API (`HandLandmarker`) van mediapipe 1.0.1.
