from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

app = Flask(__name__)

# ---------------- VOLUME SETUP ---------------- #

devices = AudioUtilities.GetSpeakers()

interface = devices.Activate(
    IAudioEndpointVolume._iid_,
    CLSCTX_ALL,
    None
)

volume = cast(interface, POINTER(IAudioEndpointVolume))

minVol, maxVol, _ = volume.GetVolumeRange()
# ---------------- MEDIAPIPE ---------------- #

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# ---------------- CAMERA ---------------- #

cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                thumb = hand_landmarks.landmark[4]
                index = hand_landmarks.landmark[8]

                x1, y1 = int(thumb.x * w), int(thumb.y * h)
                x2, y2 = int(index.x * w), int(index.y * h)

                cv2.circle(frame, (x1, y1), 10, (0,255,0), -1)
                cv2.circle(frame, (x2, y2), 10, (0,255,0), -1)
                cv2.line(frame, (x1,y1), (x2,y2), (255,0,0), 3)

                distance = np.hypot(x2-x1, y2-y1)

                vol = np.interp(distance, [20,200], [minVol,maxVol])
                volume.SetMasterVolumeLevel(vol, None)

                vol_percent = np.interp(distance,[20,200],[0,100])

                cv2.putText(frame,
                            f'Volume: {int(vol_percent)} %',
                            (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,(0,0,255),3)

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)