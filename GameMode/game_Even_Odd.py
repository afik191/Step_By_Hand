
import cv2
import sys
import math
import time
import random
from pathlib import Path
from collections import deque
import mediapipe as mp

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    import serial
except Exception:
    serial = None

try:
    import numpy as np
except Exception:
    np = None

TITLE_COLOR = (0, 0, 0)
OPTION_COLOR = (0, 0, 220)
NOTE_COLOR = (60, 60, 60)
HOLD_COLOR = (100, 40, 0)
GOOD_COLOR = (0, 140, 0)
BAD_COLOR = (0, 0, 180)
INFO_COLOR = (50, 50, 50)

GESTURE_HOLD_SECONDS = 1.5
START_HOLD_SECONDS = 1.5
ANSWER_STABLE_SECONDS = 0.8

voice_queue = deque()
stop_listening = None
voice_enabled = False
hold_action = None
hold_start_time = 0.0


def get_dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def is_ok_gesture(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])
    if hand_scale <= 0:
        return False
    thumb_index_dist = get_dist(hand_landmarks.landmark[4], hand_landmarks.landmark[8])

    def is_open(tip_idx, mip_idx):
        mip_dist = get_dist(wrist, hand_landmarks.landmark[mip_idx])
        if mip_dist <= 0:
            return False
        tip_dist = get_dist(wrist, hand_landmarks.landmark[tip_idx])
        return tip_dist / mip_dist > 1.10

    return thumb_index_dist < hand_scale * 0.35 and is_open(12, 10) and is_open(16, 14) and is_open(20, 18)


def is_thumbs_up(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])
    if hand_scale <= 0:
        return False
    thumb_open = get_dist(hand_landmarks.landmark[4], hand_landmarks.landmark[5]) > hand_scale * 0.65

    def is_finger_open(tip_idx, mip_idx):
        mip_dist = get_dist(wrist, hand_landmarks.landmark[mip_idx])
        if mip_dist <= 0:
            return False
        tip_dist = get_dist(wrist, hand_landmarks.landmark[tip_idx])
        return tip_dist / mip_dist > 1.15

    index_open = is_finger_open(8, 6)
    middle_open = is_finger_open(12, 10)
    ring_open = is_finger_open(16, 14)
    pinky_open = is_finger_open(20, 18)
    return thumb_open and not index_open and not middle_open and not ring_open and not pinky_open


def limit_robot_fingers(number):
    return max(0, min(5, int(number)))


def count_fingers_single_hand(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])
    if hand_scale <= 0:
        return -1
    fingers = []
    fingers.append(1 if get_dist(hand_landmarks.landmark[4], hand_landmarks.landmark[5]) > hand_scale * 0.6 else 0)
    tips_idx = [8, 12, 16, 20]
    mips_idx = [6, 10, 14, 18]
    for tip, mip in zip(tips_idx, mips_idx):
        tip_dist = get_dist(wrist, hand_landmarks.landmark[tip])
        mip_dist = get_dist(wrist, hand_landmarks.landmark[mip])
        fingers.append(1 if mip_dist > 0 and tip_dist / mip_dist > 1.15 else 0)
    return limit_robot_fingers(sum(fingers))


def count_fingers_two_hands(all_hand_landmarks):
    if not all_hand_landmarks:
        return -1
    total = 0
    detected = 0
    for hand_landmarks in all_hand_landmarks:
        c = count_fingers_single_hand(hand_landmarks)
        if c >= 0:
            total += c
            detected += 1
    if detected == 0:
        return -1
    return max(0, min(10, total))


def update_hold(desired_action, current_time):
    global hold_action, hold_start_time
    if desired_action is None:
        hold_action = None
        hold_start_time = 0.0
        return None, 0.0
    if desired_action != hold_action:
        hold_action = desired_action
        hold_start_time = current_time
        return None, 0.0
    elapsed = current_time - hold_start_time
    if elapsed >= GESTURE_HOLD_SECONDS:
        action = hold_action
        hold_action = None
        hold_start_time = 0.0
        return action, elapsed
    return None, elapsed


def draw_lines(frame, lines, start_y, color, scale=0.9, thickness=2, step=46):
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (30, start_y + i * step), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_hold_status(frame, current_time):
    h, _, _ = frame.shape
    if hold_action:
        progress = min(GESTURE_HOLD_SECONDS, current_time - hold_start_time)
        cv2.putText(frame, f"Hold action: {hold_action}", (30, h - 92), cv2.FONT_HERSHEY_SIMPLEX, 0.72, HOLD_COLOR, 2)
        cv2.putText(frame, f"Hold selection: {progress:.1f}s / {GESTURE_HOLD_SECONDS:.1f}s", (30, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.72, HOLD_COLOR, 2)


def voice_callback(recognizer, audio):
    try:
        command = recognizer.recognize_google(audio, language="en-US").upper()
        voice_queue.append(command)
        print(f"Heard voice command: {command}")
    except Exception:
        pass


def init_voice():
    global stop_listening, voice_enabled
    voice_enabled = False
    stop_listening = None
    if sr is None:
        return
    try:
        r = sr.Recognizer()
        m = sr.Microphone()
        r.energy_threshold = 1000
        r.dynamic_energy_threshold = False
        r.non_speaking_duration = 0.3
        r.pause_threshold = 0.3
        with m as source:
            r.adjust_for_ambient_noise(source, duration=1)
        stop_listening = r.listen_in_background(m, voice_callback, phrase_time_limit=1.2)
        voice_enabled = True
        print("Voice control active")
    except Exception as e:
        print(f"Voice not available: {e}")


def stop_voice():
    global stop_listening
    if stop_listening is not None:
        try:
            stop_listening(wait_for_stop=False)
        except Exception:
            pass
        stop_listening = None


def init_serial_connection(port="COM4", baud_rate=9600):
    if serial is None:
        print("PySerial not available - running in simulation mode")
        return None
    try:
        ser = serial.Serial(port, baud_rate, timeout=1)
        time.sleep(2)
        print(f"Connected to Arduino on {port}")
        return ser
    except Exception as e:
        print(f"Arduino not connected. Running in simulation mode. {e}")
        return None


def send_robot_fingers(ser, number):
    safe_number = limit_robot_fingers(number)
    if ser is not None and ser.is_open:
        try:
            ser.write(str(safe_number).encode())
            print(f"Robot shows {safe_number} fingers")
        except Exception as e:
            print(f"Serial send failed, simulation fallback: {e}")
    else:
        print(f"Simulation mode - robot would show {safe_number} fingers")

import pickle

BASE_DIR = Path(__file__).resolve().parent
COUNTDOWN_DURATION = 3.0
FEATURE_LOCK_BEFORE_END = 0.3
FEATURE_LOCK_TIME = COUNTDOWN_DURATION - FEATURE_LOCK_BEFORE_END
VERIFY_ACTUAL_SECONDS = 1.2
BACK_EXIT_CODE = 10

state = "CHOOSE_SIDE"
selected_side = None
countdown_start = 0.0
verify_start = 0.0
candidate_actual = None
candidate_actual_start = 0.0
predicted_user_move = -1
actual_user_move = -1
robot_move = 0
confidence_percent = 0.0
locked_features = None
has_predicted = False
prev_distances = None
result_text = ""
result_color = INFO_COLOR


def extract_prediction_features(hand_landmarks, previous_distances):
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])
    if hand_scale <= 0:
        return None, previous_distances
    tips = [4, 8, 12, 16, 20]
    current_distances = [get_dist(wrist, hand_landmarks.landmark[t]) / hand_scale for t in tips]
    speeds = [0.0] * 5 if previous_distances is None else [cur - prv for cur, prv in zip(current_distances, previous_distances)]
    inter = [get_dist(hand_landmarks.landmark[tips[i]], hand_landmarks.landmark[tips[i+1]]) / hand_scale for i in range(4)]
    return current_distances + speeds + inter, current_distances


def choose_robot_move_to_win(predicted_move, selected_side):
    predicted_move = max(0, min(5, int(predicted_move)))
    if selected_side == "EVEN":
        return 1 if predicted_move % 2 == 0 else 2
    return 2 if predicted_move % 2 == 0 else 1


def calculate_result(actual_move, robot_choice, selected_side):
    if actual_move < 0:
        return "No clear final hand", NOTE_COLOR
    total = actual_move + robot_choice
    total_is_even = total % 2 == 0
    user_wins = (selected_side == "EVEN" and total_is_even) or (selected_side == "ODD" and not total_is_even)
    if user_wins:
        return "USER WINS!", BAD_COLOR
    return "ROBOT WINS!", GOOD_COLOR


def start_round():
    global state, countdown_start, verify_start, candidate_actual, candidate_actual_start
    global predicted_user_move, actual_user_move, robot_move, confidence_percent
    global locked_features, has_predicted, prev_distances, result_text, result_color
    state = "COUNTDOWN"
    countdown_start = time.time()
    verify_start = 0.0
    candidate_actual = None
    candidate_actual_start = 0.0
    predicted_user_move = -1
    actual_user_move = -1
    robot_move = 0
    confidence_percent = 0.0
    locked_features = None
    has_predicted = False
    prev_distances = None
    result_text = ""
    result_color = INFO_COLOR


def finish_to_result():
    global state, result_text, result_color
    result_text, result_color = calculate_result(actual_user_move, robot_move, selected_side)
    state = "RESULT"


def process_voice_command(command):
    global state, selected_side
    if "BACK" in command:
        send_robot_fingers(ser, 0)
        sys.exit(BACK_EXIT_CODE)
    if state == "CHOOSE_SIDE":
        if "EVEN" in command:
            selected_side = "EVEN"
            state = "READY"
            return True
        if "ODD" in command:
            selected_side = "ODD"
            state = "READY"
            return True
    if ("START" in command or "BEGIN" in command or "PLAY" in command or "AGAIN" in command) and state in ["READY", "RESULT"]:
        start_round()
        return True
    return False


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
try:
    with open(BASE_DIR / "hand_predictor.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Could not load model: {e}")
    sys.exit(1)
ser = init_serial_connection()
init_voice()
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    stop_voice()
    raise SystemExit("Camera Error")

try:
    with mp_hands.Hands(model_complexity=1, max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            current_fingers = -1
            current_ok = False
            current_thumbs = False
            current_features = None
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    current_fingers = count_fingers_single_hand(hand_landmarks)
                    current_ok = is_ok_gesture(hand_landmarks)
                    current_thumbs = is_thumbs_up(hand_landmarks)
                    current_features, prev_distances = extract_prediction_features(hand_landmarks, prev_distances)
            else:
                prev_distances = None
            current_time = time.time()
            while voice_queue:
                process_voice_command(voice_queue.popleft())
            if state == "CHOOSE_SIDE":
                draw_lines(frame, ["CHOOSE YOUR SIDE"], 65, TITLE_COLOR, scale=1.1, thickness=3)
                draw_lines(frame, ["Show 2 fingers = EVEN", "Show 1 finger = ODD"], 145, OPTION_COLOR, scale=0.95, thickness=2)
                draw_lines(frame, ["Voice: say EVEN or ODD", "OK sign = Back to Game Menu", "Voice: say BACK"], 255, NOTE_COLOR, scale=0.76, thickness=2, step=40)
                desired_action = None
                if current_ok:
                    desired_action = "BACK"
                elif current_fingers == 2:
                    desired_action = "EVEN"
                elif current_fingers == 1:
                    desired_action = "ODD"
                action, _ = update_hold(desired_action, current_time)
                if action == "BACK":
                    send_robot_fingers(ser, 0)
                    sys.exit(BACK_EXIT_CODE)
                elif action == "EVEN":
                    selected_side = "EVEN"
                    state = "READY"
                elif action == "ODD":
                    selected_side = "ODD"
                    state = "READY"
            elif state == "READY":
                draw_lines(frame, ["EVEN ODD GAME"], 65, TITLE_COLOR, scale=1.1, thickness=3)
                draw_lines(frame, [f"Your side: {selected_side}", "Thumbs up = Start game", "Voice: say START"], 135, OPTION_COLOR, scale=0.88, thickness=2)
                draw_lines(frame, ["OK sign = Back to Game Menu", "Voice: say BACK"], 275, NOTE_COLOR, scale=0.76, thickness=2, step=40)
                desired_action = "START" if current_thumbs else ("BACK" if current_ok else None)
                action, _ = update_hold(desired_action, current_time)
                if action == "START":
                    start_round()
                elif action == "BACK":
                    send_robot_fingers(ser, 0)
                    sys.exit(BACK_EXIT_CODE)
            elif state == "COUNTDOWN":
                elapsed = current_time - countdown_start
                remaining = max(0.0, COUNTDOWN_DURATION - elapsed)
                count = int(math.ceil(remaining))
                draw_lines(frame, ["EVEN ODD GAME"], 60, TITLE_COLOR, scale=1.05, thickness=3)
                draw_lines(frame, [f"Side: {selected_side}", "Show your number now"], 115, OPTION_COLOR, scale=0.88, thickness=2)
                draw_lines(frame, ["Robot predicts before the end", "Result will use your actual final number"], 195, NOTE_COLOR, scale=0.72, thickness=2, step=36)
                if count > 0:
                    cv2.putText(frame, str(count), (w // 2 - 45, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 4.6, OPTION_COLOR, 8)
                if current_fingers != -1:
                    cv2.putText(frame, f"Detected now: {current_fingers}", (30, h - 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, INFO_COLOR, 2)
                draw_lines(frame, ["OK sign = Back", "Voice: BACK"], h - 95, NOTE_COLOR, scale=0.72, thickness=2, step=35)
                desired_action = "BACK" if current_ok else None
                action, _ = update_hold(desired_action, current_time)
                if action == "BACK":
                    send_robot_fingers(ser, 0)
                    sys.exit(BACK_EXIT_CODE)
                if elapsed >= FEATURE_LOCK_TIME and locked_features is None and current_features is not None:
                    locked_features = list(current_features)
                if elapsed >= COUNTDOWN_DURATION and not has_predicted:
                    if locked_features is not None and np is not None:
                        probs = model.predict_proba([locked_features])[0]
                        predicted_user_move = int(np.argmax(probs))
                        confidence_percent = float(probs[predicted_user_move] * 100)
                    else:
                        predicted_user_move = random.randint(0, 5)
                    robot_move = choose_robot_move_to_win(predicted_user_move, selected_side)
                    send_robot_fingers(ser, robot_move)
                    has_predicted = True
                    state = "VERIFY_ACTUAL"
                    verify_start = current_time
                    candidate_actual = None
                    candidate_actual_start = 0.0
            elif state == "VERIFY_ACTUAL":
                elapsed = current_time - verify_start
                draw_lines(frame, ["FINAL NUMBER CHECK"], 60, TITLE_COLOR, scale=1.1, thickness=3)
                draw_lines(frame, [f"Robot played: {robot_move}", "Hold your real final number now"], 130, OPTION_COLOR, scale=0.88, thickness=2)
                draw_lines(frame, [f"Predicted move: {predicted_user_move} ({confidence_percent:.1f}%)", "We now confirm your actual final number"], 215, NOTE_COLOR, scale=0.72, thickness=2, step=36)
                if current_fingers != -1:
                    cv2.putText(frame, f"Detected final number: {current_fingers}", (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.82, INFO_COLOR, 2)
                    if candidate_actual != current_fingers:
                        candidate_actual = current_fingers
                        candidate_actual_start = current_time
                    else:
                        hold = current_time - candidate_actual_start
                        cv2.putText(frame, f"Stable hold: {hold:.1f}s / {ANSWER_STABLE_SECONDS:.1f}s", (30, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.75, HOLD_COLOR, 2)
                        if hold >= ANSWER_STABLE_SECONDS:
                            actual_user_move = current_fingers
                            finish_to_result()
                else:
                    candidate_actual = None
                    candidate_actual_start = 0.0
                draw_lines(frame, ["OK sign = Back", "Voice: BACK"], h - 95, NOTE_COLOR, scale=0.72, thickness=2, step=35)
                desired_action = "BACK" if current_ok else None
                action, _ = update_hold(desired_action, current_time)
                if action == "BACK":
                    send_robot_fingers(ser, 0)
                    sys.exit(BACK_EXIT_CODE)
                if elapsed >= VERIFY_ACTUAL_SECONDS and state == "VERIFY_ACTUAL":
                    actual_user_move = current_fingers
                    finish_to_result()
            elif state == "RESULT":
                total = actual_user_move + robot_move if actual_user_move >= 0 else "?"
                draw_lines(frame, ["RESULT"], 60, TITLE_COLOR, scale=1.1, thickness=3)
                draw_lines(frame, [f"Your side: {selected_side}", f"Predicted move: {predicted_user_move} ({confidence_percent:.1f}%)", f"Actual move: {actual_user_move}", f"Robot played: {robot_move}", f"Total: {total}"], 120, NOTE_COLOR, scale=0.79, thickness=2, step=40)
                draw_lines(frame, [result_text], 340, result_color, scale=1.08, thickness=3)
                draw_lines(frame, ["Thumbs up = Start again", "Voice: START", "OK sign = Back to Game Menu", "Voice: BACK"], 390, OPTION_COLOR, scale=0.75, thickness=2, step=35)
                desired_action = "START" if current_thumbs else ("BACK" if current_ok else None)
                action, _ = update_hold(desired_action, current_time)
                if action == "START":
                    send_robot_fingers(ser, 0)
                    start_round()
                elif action == "BACK":
                    send_robot_fingers(ser, 0)
                    sys.exit(BACK_EXIT_CODE)
            draw_hold_status(frame, current_time)
            cv2.imshow("Predictive Even-Odd Game - Voice + Gesture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    stop_voice()
    send_robot_fingers(ser, 0)
    cap.release()
    cv2.destroyAllWindows()
    if ser is not None and ser.is_open:
        ser.close()
