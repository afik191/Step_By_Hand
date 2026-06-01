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
BACK_EXIT_CODE = 10

game_state = "CHOOSING"
user_side = ""
countdown_start = 0.0
has_predicted_this_round = False
robot_move = 0
predicted_user_move = -1
actual_user_move = -1
confidence_percent = 0.0
result_text = ""
result_color = (255, 255, 255)
prev_distances = None
last_state_change_time = 0.0
ser = None
locked_prediction_features = None
locked_features_time = None

def init_serial_connection():
    try:
        arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD_RATE, timeout=1)
        time.sleep(2)
        return arduino
    except Exception:
        return None

def send_to_arduino(command):
    global ser
    safe_command = max(MIN_FINGERS, min(MAX_FINGERS, int(command)))
    if ser is not None and ser.is_open:
        try: ser.write(str(safe_command).encode())
        except Exception: ser = None

def get_dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def count_fingers_from_landmarks(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])
    if hand_scale <= 0: return -1
    fingers = []
    if get_dist(hand_landmarks.landmark[4], hand_landmarks.landmark[5]) > hand_scale * 0.6: fingers.append(1)
    else: fingers.append(0)
    tips_idx, mips_idx = [8, 12, 16, 20], [6, 10, 14, 18]
    for tip, mip in zip(tips_idx, mips_idx):
        if get_dist(wrist, hand_landmarks.landmark[mip]) > 0 and get_dist(wrist, hand_landmarks.landmark[tip]) / get_dist(wrist, hand_landmarks.landmark[mip]) > 1.15: fingers.append(1)
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
    """ משמש כעת למשחק חוזר (👍) """
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])
    if hand_scale <= 0: return False
    thumb_open = get_dist(hand_landmarks.landmark[4], hand_landmarks.landmark[5]) > hand_scale * 0.65
    def is_finger_closed(tip_idx, mip_idx):
        mip_dist = get_dist(wrist, hand_landmarks.landmark[mip_idx])
        return mip_dist > 0 and (get_dist(wrist, hand_landmarks.landmark[tip_idx]) / mip_dist) < 1.10
    return thumb_open and is_finger_closed(8, 6) and is_finger_closed(12, 10) and is_finger_closed(16, 14) and is_finger_closed(20, 18)

def return_to_previous_menu():
    send_to_arduino(0)
    sys.exit(BACK_EXIT_CODE)

def extract_prediction_features(hand_landmarks, previous_distances):
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])
    if hand_scale <= 0: return None, previous_distances
    tips = [4, 8, 12, 16, 20]
    current_distances = [get_dist(wrist, hand_landmarks.landmark[t]) / hand_scale for t in tips]
    speeds = [0.0] * 5 if previous_distances is None else [cur - prv for cur, prv in zip(current_distances, previous_distances)]
    inter_finger_dist = [get_dist(hand_landmarks.landmark[tips[i]], hand_landmarks.landmark[tips[i+1]]) / hand_scale for i in range(4)]
    return current_distances + speeds + inter_finger_dist, current_distances

def choose_robot_move_to_win(predicted_move, selected_side):
    pm = max(MIN_FINGERS, min(MAX_FINGERS, int(predicted_move)))
    if selected_side == "EVEN": return 1 if pm % 2 == 0 else 2
    if selected_side == "ODD": return 2 if pm % 2 == 0 else 1
    return random.randint(1, 2)

def calculate_result(actual_move, robot_choice, selected_side):
    if actual_move < 0: return "No clear hand detected", (0, 165, 255)
    total = actual_move + robot_choice
    user_wins = (selected_side == "EVEN" and total % 2 == 0) or (selected_side == "ODD" and total % 2 != 0)
    return ("USER WINS!", (0, 255, 255)) if user_wins else ("ROBOT WINS!", (0, 255, 0))

def reset_round(selected_side, start_time):
    global user_side, game_state, countdown_start, has_predicted_this_round, robot_move, predicted_user_move, actual_user_move, confidence_percent, result_text, result_color, locked_prediction_features, locked_features_time
    user_side, game_state, countdown_start, has_predicted_this_round = selected_side, "COUNTDOWN", start_time, False
    robot_move, predicted_user_move, actual_user_move, confidence_percent = 0, -1, -1, 0.0
    result_text, result_color, locked_prediction_features, locked_features_time = "", (255, 255, 255), None, None

def voice_callback(recognizer, audio):
    global game_state
    try:
        command = recognizer.recognize_google(audio, language="en-US").upper()
        if game_state == "CHOOSING":
            if any(w in command for w in ["EVEN", "EVENT", "EVAN"]): reset_round("EVEN", time.time())
            elif any(w in command for w in ["ODD", "ADD", "OLD"]): reset_round("ODD", time.time())
        elif game_state == "RESULT":
            if any(w in command for w in ["AGAIN", "GAIN", "GAME", "REMATCH", "BEGIN"]):
                send_to_arduino(0)
                game_state = "CHOOSING"
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

            current_features, current_finger_count, current_ok_gesture, current_thumbs_up = None, -1, False, False
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    current_finger_count = count_fingers_from_landmarks(hand_landmarks)
                    current_ok_gesture = is_ok_gesture(hand_landmarks)
                    current_thumbs_up = is_thumbs_up_gesture(hand_landmarks)
                    current_features, prev_distances = extract_prediction_features(hand_landmarks, prev_distances)
            else: prev_distances = None

            current_time = time.time()

            if game_state == "CHOOSING":
                cv2.putText(frame, "Say 'EVEN' or Show 2 Fingers", (30, h // 2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(frame, "Say 'ODD' or Show 1 Finger", (30, h // 2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(frame, "👌 (OK Sign): return to Game Menu", (30, h // 2 + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)
                
                if current_time - last_state_change_time > 1.0:
                    if current_ok_gesture: return_to_previous_menu()
                    elif current_finger_count == 2:
                        reset_round("EVEN", current_time)
                        last_state_change_time = current_time
                    elif current_finger_count == 1:
                        reset_round("ODD", current_time)
                        last_state_change_time = current_time

            elif game_state == "COUNTDOWN":
                elapsed = current_time - countdown_start
                remaining = max(0.0, COUNTDOWN_DURATION - elapsed)
                count = int(math.ceil(remaining))
                if count > 0: cv2.putText(frame, str(count), (w // 2 - 50, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 10)

                if elapsed >= FEATURE_LOCK_TIME and locked_prediction_features is None:
                    if current_features is not None:
                        locked_prediction_features = list(current_features)
                        locked_features_time = elapsed

                if elapsed >= ROBOT_DECISION_TIME and not has_predicted_this_round:
                    if locked_prediction_features is not None:
                        probabilities = model.predict_proba([locked_prediction_features])[0]
                        predicted_user_move = int(np.argmax(probabilities))
                        confidence_percent = probabilities[predicted_user_move] * 100
                        robot_move = choose_robot_move_to_win(predicted_user_move, user_side)
                        send_to_arduino(robot_move)
                        has_predicted_this_round = True
                    else:
                        robot_move = random.randint(1, 2)
                        send_to_arduino(robot_move)
                        has_predicted_this_round = True

                if elapsed >= COUNTDOWN_DURATION:
                    actual_user_move = current_finger_count
                    result_text, result_color = calculate_result(actual_user_move, robot_move, user_side)
                    game_state = "RESULT"
                    last_state_change_time = current_time

            elif game_state == "RESULT":
                cv2.putText(frame, result_text, (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, result_color, 4)
                cv2.putText(frame, "Say 'AGAIN' or 👍 (Thumbs Up) for Rematch", (30, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 2)
                cv2.putText(frame, "👌 (OK Sign): return to Game Menu", (30, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 2)

                if current_time - last_state_change_time > 1.0:
                    if current_ok_gesture: return_to_previous_menu()
                    elif current_thumbs_up:
                        send_to_arduino(0)
                        game_state = "CHOOSING"
                        last_state_change_time = current_time

            cv2.imshow("Predictive Even-Odd Game - Voice + Gesture", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
finally:
    stop_listening(wait_for_stop=False)
    cap.release()
    cv2.destroyAllWindows()
    if ser is not None and ser.is_open:
        send_to_arduino(0)
        ser.close()