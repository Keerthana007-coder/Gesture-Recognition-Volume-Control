from flask import Flask, render_template, Response, jsonify
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import cv2
import mediapipe as mp
import numpy as np
import time

prev_time = 0
distance =0
total_predictions = 0
correct_predictions = 0
system_running = True
app = Flask(__name__)

# ---------------- DATA ---------------- #

detection_data = {
    "status": "No Hand",
    "distance": 0,
    "gesture": "None",
    "volume_percent": 0,
    "landmarks": 0,
    "resolution": "",
    "hands_detected": 0,
    "fps": 0,
    "accuracy": 0,
    "response_time": 0
}

# ---------------- VOLUME SETUP ---------------- #

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_,
    CLSCTX_ALL,
    None
)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

# ---------------- MEDIAPIPE ---------------- #

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# ---------------- CAMERA ---------------- #

cap = cv2.VideoCapture(0)

# ---------------- FRAME GENERATOR ---------------- #

def generate_frames():
    global prev_time, total_predictions, correct_predictions
    while True:
        start_time = time.time()
        try:
            success, frame = cap.read()
            if not success:
                break
            
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
            prev_time = current_time
            detection_data["fps"] = int(fps)

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            h, w, c = frame.shape
            detection_data["resolution"] = f"{w}x{h}"

            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:

                num_hands = len(results.multi_hand_landmarks)
                detection_data["status"] = "Hand Detected"
                detection_data["hands_detected"] = num_hands
        

                for hand_landmarks in results.multi_hand_landmarks:

                    lmList = []
                
                    for lm in hand_landmarks.landmark:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append((cx, cy))

                    detection_data["landmarks"] = len(lmList)

                    if len(lmList) > 8:
                        
                        x1, y1 = lmList[4]   # thumb tip
                        x2, y2 = lmList[8]   # index tip

                        distance_pixels = np.hypot(x2 - x1, y2 - y1)
                        distance_pixels = np.clip(distance_pixels, 20, 200)

                        distance_mm = int(np.interp(distance_pixels, [20, 200], [10, 120]))
                        detection_data["distance"] = distance_mm

                        vol = np.interp(distance_pixels, [20, 200], [minVol, maxVol])

                        if system_running:
                            try:
                                volume.SetMasterVolumeLevel(vol, None)
                            except:
                                pass

                        vol_percent = int(np.interp(distance_pixels, [20, 200], [0, 100]))
                        detection_data["volume_percent"] = vol_percent

        # ⭐ Gesture Detection
                        if distance_mm < 30:
                            gesture = "Closed ✊"
                        elif distance_mm < 80:
                            gesture = "Pinch 🤏"
                        else:
                            gesture = "Open ✋"
                        detection_data["gesture"] = gesture
                        total_predictions += 1

                    if (gesture == "Closed ✊" and distance_mm < 30) or \
                        (gesture == "Pinch 🤏" and 30 <= distance_mm < 80) or \
                        (gesture == "Open ✋" and distance_mm >= 80):
                        correct_predictions += 1

                    accuracy = (correct_predictions / total_predictions) * 100
                    detection_data["accuracy"] = round(accuracy, 2)

                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            else:
                detection_data["status"] = "No Hand"
                detection_data["gesture"] = "None"
                detection_data["distance"] = 0
                detection_data["volume_percent"] = 0
                detection_data["landmarks"] = 0
                detection_data["hands_detected"] = 0

            response_time = (time.time() - start_time) * 1000
            detection_data["response_time"] = int(response_time)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   frame + b'\r\n')

        except Exception as e:
            print("ERROR:", e)
            continue


# ---------------- ROUTES ---------------- #

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection_data')
def get_detection_data():
   return jsonify(detection_data)

@app.route('/start')
def start():
    global system_running
    system_running = True
    print("START BUTTON PRESSED")
    return "Started"

@app.route('/pause')
def pause():
    global system_running
    system_running = False
    print("PAUSE BUTTON PRESSED")
    return "Paused"

# ---------------- MAIN ---------------- #

if __name__ == "__main__":
    app.run(debug=False)