import cv2
import sys
import math
import time
import serial
import pickle
import random
from pathlib import Path
import numpy as np
import mediapipe as mp
import speech_recognition as sr

try:
    import sklearn
except ImportError:
    sys.exit(1)

BASE_DIR = Path(__file__).resolve().parent
ARDUINO_PORT = "COM4"
ARDUINO_BAUD_RATE = 9600
COUNTDOWN_DURATION = 3.0
FEATURE_LOCK_BEFORE_END = 0.3
FEATURE_LOCK_TIME = COUNTDOWN_DURATION - FEATURE_LOCK_BEFORE_END
ROBOT_DECISION_TIME = COUNTDOWN_DURATION
MIN_FINGERS = 0
MAX_FINGERS = 5

ROBOT_COMMANDS = {"ROCK": 0, "PAPER": 5, "SCISSORS": 2}
WINNING_MOVE = {"ROCK": "PAPER", "PAPER": "SCISSORS", "SCISSORS": "ROCK"}

game_state = "READY"
countdown_start, last_state_change_time = 0, 0
locked_prediction_features, locked_features_time, has_predicted_this_round = None, None, False
predicted_user_move, actual_user_move, robot_move = "UNKNOWN", "UNKNOWN", "UNKNOWN"
confidence_percent = 0.0
result_text, result_color = "", (255, 255, 255)
prev_distances, ser = None, None

def init_serial_connection():
    try:
        arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD_RATE, timeout=1)
        time.sleep(2)
        return arduino
    except Exception: return None

def send_to_arduino_fingers(number):
    global ser
    safe = max(MIN_FINGERS, min(MAX_FINGERS, int(number)))
    if ser is not None and ser.is_open:
        try: ser.write(str(safe).encode())
        except Exception: ser = None

def get_dist(p1, p2): return math.hypot(p1.x - p2.x, p1.y - p2.y)

def count_fingers_from_landmarks(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])
    if hand_scale <= 0: return -1
    fingers = []
    if get_dist(hand_landmarks.landmark[4], hand_landmarks.landmark[5]) > hand_scale * 0.6: fingers.append(1)
    else: fingers.append(0)
    tips, mips = [8, 12, 16, 20], [6, 10, 14, 18]
    for t, m in zip(tips, mips):
        if get_dist(wrist, hand_landmarks.landmark[m]) > 0 and get_dist(wrist, hand_landmarks.landmark[t]) / get_dist(wrist, hand_landmarks.landmark[m]) > 1.15: fingers.append(1)
        else: fingers.append(0)
    return max(MIN_FINGERS, min(MAX_FINGERS, sum(fingers)))

def is_ok_gesture(hand_landmarks):
    """ משמש כעת למחוות חזרה (👌) """
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])
    if hand_scale <= 0: return False
    thumb_index_dist = get_dist(hand_landmarks.landmark[4], hand_landmarks.landmark[8])
    def is_finger_open(tip_idx, mip_idx):
        mip_dist = get_dist(wrist, hand_landmarks.landmark[mip_idx])
        return mip_dist > 0 and (get_dist(wrist, hand_landmarks.landmark[tip_idx]) / mip_dist) > 1.10
    return thumb_index_dist < hand_scale * 0.35 and is_finger_open(12, 10) and is_finger_open(16, 14) and is_finger_open(20, 18)

def is_thumbs_up_gesture(hand_landmarks):
    """ משמש כעת להתחלה ומשחק חוזר (👍) """
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])
    if hand_scale <= 0: return False
    thumb_open = get_dist(hand_landmarks.landmark[4], hand_landmarks.landmark[5]) > hand_scale * 0.65
    def is_finger_closed(tip_idx, mip_idx):
        mip_dist = get_dist(wrist, hand_landmarks.landmark[mip_idx])
        return mip_dist > 0 and (get_dist(wrist, hand_landmarks.landmark[tip_idx]) / mip_dist) < 1.10
    return thumb_open and is_finger_closed(8, 6) and is_finger_closed(12, 10) and is_finger_closed(16, 14) and is_finger_closed(20, 18)

def extract_prediction_features(hand_landmarks, previous_distances):
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])
    if hand_scale <= 0: return None, previous_distances
    tips = [4, 8, 12, 16, 20]
    current_distances = [get_dist(wrist, hand_landmarks.landmark[t]) / hand_scale for t in tips]
    speeds = [0.0] * 5 if previous_distances is None else [cur - prv for cur, prv in zip(current_distances, previous_distances)]
    inter = [get_dist(hand_landmarks.landmark[tips[i]], hand_landmarks.landmark[tips[i+1]]) / hand_scale for i in range(4)]
    return current_distances + speeds + inter, current_distances

def map_model_output_to_rps(raw):
    p = int(raw)
    if p == 0: return "ROCK"
    if p == 2: return "SCISSORS"
    if p == 5: return "PAPER"
    return "ROCK" if p <= 1 else "SCISSORS" if p in [2, 3] else "PAPER"

def voice_callback(recognizer, audio):
    global game_state, last_state_change_time
    try:
        command = recognizer.recognize_google(audio, language="en-US").upper()
        if game_state == "READY" and any(w in command for w in ["START", "BEGIN", "GO", "PLAY"]):
            countdown_start, game_state, last_state_change_time = time.time(), "COUNTDOWN", time.time()
        elif game_state == "RESULT" and any(w in command for w in ["AGAIN", "GAIN", "GAME", "REMATCH"]):
            send_to_arduino_fingers(0)
            game_state, last_state_change_time = "READY", time.time()
    except Exception: pass

r, m = sr.Recognizer(), sr.Microphone()
r.energy_threshold, r.dynamic_energy_threshold, r.non_speaking_duration, r.pause_threshold = 1000, False, 0.3, 0.3
with m as source: r.adjust_for_ambient_noise(source, duration=1)
stop_listening = r.listen_in_background(m, voice_callback, phrase_time_limit=1.2)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

with open(BASE_DIR / "hand_predictor.pkl", "rb") as f: model = pickle.load(f)
ser = init_serial_connection()
cap = cv2.VideoCapture(0)

try:
    with mp_hands.Hands(model_complexity=1, max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: continue
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            current_finger_count, current_ok_gesture, current_thumbs_up, current_features = -1, False, False, None
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    current_finger_count = count_fingers_from_landmarks(hand_landmarks)
                    current_ok_gesture = is_ok_gesture(hand_landmarks)
                    current_thumbs_up = is_thumbs_up_gesture(hand_landmarks)
                    current_features, prev_distances = extract_prediction_features(hand_landmarks, prev_distances)
            else: prev_distances = None

            current_time = time.time()

            if game_state == "READY":
                cv2.putText(frame, "Say 'START' or show 👍 (Thumbs Up)", (30, h // 2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 2)
                cv2.putText(frame, "👌 (OK Sign): return to Game Menu", (30, h // 2 + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)
                if current_time - last_state_change_time > 1.0:
                    if current_ok_gesture:
                        send_to_arduino_fingers(0)
                        sys.exit(10)
                    elif current_thumbs_up:
                        countdown_start, game_state, last_state_change_time = current_time, "COUNTDOWN", current_time

            elif game_state == "COUNTDOWN":
                elapsed = current_time - countdown_start
                count = int(math.ceil(max(0.0, COUNTDOWN_DURATION - elapsed)))
                if count > 0: cv2.putText(frame, str(count), (w // 2 - 50, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 10)

                if locked_prediction_features is None and elapsed >= FEATURE_LOCK_TIME:
                    if current_features is not None:
                        locked_prediction_features = list(current_features)
                        locked_features_time = elapsed

                if elapsed >= ROBOT_DECISION_TIME and not has_predicted_this_round:
                    if locked_prediction_features is not None:
                        probabilities = model.predict_proba([locked_prediction_features])[0]
                        raw_pred = int(np.argmax(probabilities))
                        confidence_percent = float(probabilities[raw_pred] * 100)
                        predicted_user_move = map_model_output_to_rps(raw_pred)
                        robot_move = WINNING_MOVE.get(predicted_user_move, "ROCK")
                    else:
                        robot_move = random.choice(["ROCK", "PAPER", "SCISSORS"])
                    
                    send_to_arduino_fingers(ROBOT_COMMANDS[robot_move])
                    has_predicted_this_round = True
                    actual_user_move = "ROCK" if current_finger_count == 0 else "SCISSORS" if current_finger_count == 2 else "PAPER" if current_finger_count == 5 else "UNKNOWN"
                    
                    if actual_user_move == "UNKNOWN": result_text, result_color = "No clear hand detected", (0, 165, 255)
                    elif actual_user_move == robot_move: result_text, result_color = "TIE", (0, 255, 255)
                    elif WINNING_MOVE[actual_user_move] == robot_move: result_text, result_color = "ROBOT WINS!", (0, 255, 0)
                    else: result_text, result_color = "USER WINS!", (0, 0, 255)
                    
                    game_state, last_state_change_time = "RESULT", current_time

            elif game_state == "RESULT":
                cv2.putText(frame, result_text, (50, 310), cv2.FONT_HERSHEY_SIMPLEX, 1.5, result_color, 4)
                cv2.putText(frame, "👌 / AGAIN = rematch | 👌 (OK Sign) = Game Menu", (30, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                if current_time - last_state_change_time > 1.0:
                    if current_ok_gesture:
                        send_to_arduino_fingers(0)
                        sys.exit(10)
                    elif current_thumbs_up:
                        send_to_arduino_fingers(0)
                        game_state, last_state_change_time = "READY", current_time

            cv2.imshow("Predictive Rock Paper Scissors - Voice + Gesture", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
finally:
    stop_listening(wait_for_stop=False)
    cap.release()
    cv2.destroyAllWindows()
    if ser is not None and ser.is_open:
        send_to_arduino_fingers(0)
        ser.close()