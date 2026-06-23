import cv2
import sys
import math
import time
import random
import pickle
from pathlib import Path
from collections import deque
import mediapipe as mp
# Add the project root so shared modules can be imported from subfolders.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from voice_instructions import VoiceInstructions

try:
    import sklearn
except ImportError:
    print("WARNING: sklearn not found. ML models may fail to load.")

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

PANEL_BG_COLOR = (255, 255, 255)
DIVIDER_COLOR = (170, 170, 170)

def create_split_screen(frame):
    camera_view = frame.copy()
    panel = frame.copy()
    panel[:] = PANEL_BG_COLOR
    panel_h = panel.shape[0]
    cv2.line(panel, (0, 0), (0, panel_h), DIVIDER_COLOR, 3)
    return camera_view, panel

GESTURE_HOLD_SECONDS = 1.5
START_HOLD_SECONDS = 1.5
ANSWER_STABLE_SECONDS = 0.8

voice_queue = deque()
stop_listening = None
voice_enabled = False
voice_guide = VoiceInstructions()
hold_action = None
hold_start_time = 0.0

def get_dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def is_ok_gesture(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])
    if hand_scale <= 0: return False
    thumb_index_dist = get_dist(hand_landmarks.landmark[4], hand_landmarks.landmark[8])
    def is_open(tip_idx, mip_idx):
        mip_dist = get_dist(wrist, hand_landmarks.landmark[mip_idx])
        if mip_dist <= 0: return False
        tip_dist = get_dist(wrist, hand_landmarks.landmark[tip_idx])
        return tip_dist / mip_dist > 1.10
    return thumb_index_dist < hand_scale * 0.35 and is_open(12, 10) and is_open(16, 14) and is_open(20, 18)

def is_thumbs_up(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])
    if hand_scale <= 0: return False
    thumb_open = get_dist(hand_landmarks.landmark[4], hand_landmarks.landmark[5]) > hand_scale * 0.65
    def is_finger_open(tip_idx, mip_idx):
        mip_dist = get_dist(wrist, hand_landmarks.landmark[mip_idx])
        if mip_dist <= 0: return False
        tip_dist = get_dist(wrist, hand_landmarks.landmark[tip_idx])
        return tip_dist / mip_dist > 1.15
    return thumb_open and not is_finger_open(8, 6) and not is_finger_open(12, 10) and not is_finger_open(16, 14) and not is_finger_open(20, 18)

def limit_robot_fingers(number):
    return max(0, min(5, int(number)))

def count_fingers_single_hand(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])
    if hand_scale <= 0: return -1
    fingers = [1 if get_dist(hand_landmarks.landmark[4], hand_landmarks.landmark[5]) > hand_scale * 0.6 else 0]
    for tip, mip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        fingers.append(1 if get_dist(wrist, hand_landmarks.landmark[mip]) > 0 and get_dist(wrist, hand_landmarks.landmark[tip]) / get_dist(wrist, hand_landmarks.landmark[mip]) > 1.15 else 0)
    return limit_robot_fingers(sum(fingers))

def update_hold(desired_action, current_time):
    global hold_action, hold_start_time
    if desired_action is None:
        hold_action, hold_start_time = None, 0.0
        return None, 0.0
    if desired_action != hold_action:
        hold_action, hold_start_time = desired_action, current_time
        return None, 0.0
    elapsed = current_time - hold_start_time
    if elapsed >= GESTURE_HOLD_SECONDS:
        action = hold_action
        hold_action, hold_start_time = None, 0.0
        return action, elapsed
    return None, elapsed

def draw_lines(frame, lines, start_y, color, scale=0.9, thickness=2, step=30):
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (30, start_y + i * step), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def draw_hold_status(frame, current_time):
    h, _, _ = frame.shape
    if hold_action:
        progress = min(GESTURE_HOLD_SECONDS, current_time - hold_start_time)
        cv2.putText(frame, f"Hold action: {hold_action}", (30, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HOLD_COLOR, 2)
        cv2.putText(frame, f"Hold selection: {progress:.1f}s / {GESTURE_HOLD_SECONDS:.1f}s", (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HOLD_COLOR, 2)

def voice_callback(recognizer, audio):
    if voice_guide.is_speaking:
        return
    try:
        command = recognizer.recognize_google(audio, language="en-US").upper()
        voice_queue.append(command)
    except Exception: pass

def init_voice():
    global stop_listening, voice_enabled
    voice_enabled, stop_listening = False, None
    if sr is None: return
    try:
        r, m = sr.Recognizer(), sr.Microphone()
        r.energy_threshold, r.dynamic_energy_threshold, r.non_speaking_duration, r.pause_threshold = 1000, False, 0.3, 0.3
        with m as source: r.adjust_for_ambient_noise(source, duration=1)
        stop_listening = r.listen_in_background(m, voice_callback, phrase_time_limit=1.2)
        voice_enabled = True
    except Exception: pass

def stop_voice():
    global stop_listening
    if stop_listening is not None:
        try: stop_listening(wait_for_stop=False)
        except Exception: pass
        stop_listening = None

def init_serial_connection(port="COM4", baud_rate=9600):
    if serial is None: return None
    try:
        ser = serial.Serial(port, baud_rate, timeout=1)
        time.sleep(2)
        return ser
    except Exception: return None

def send_robot_fingers(ser, number):
    safe_number = limit_robot_fingers(number)
    if ser is not None and ser.is_open:
        try: ser.write(str(safe_number).encode())
        except Exception: pass

BASE_DIR = Path(__file__).resolve().parent
COUNTDOWN_DURATION = 3.0
FEATURE_LOCK_BEFORE_END = 0.3
FEATURE_LOCK_TIME = COUNTDOWN_DURATION - FEATURE_LOCK_BEFORE_END
VERIFY_ACTUAL_SECONDS = 1.2
BACK_EXIT_CODE = 10

state = "CHOOSE_SIDE"
selected_side = candidate_actual = locked_features = prev_distances = None
countdown_start = verify_start = candidate_actual_start = 0.0
predicted_user_move = actual_user_move = -1
robot_move = 0
confidence_percent = 0.0
has_predicted = False
result_text = ""
result_color = INFO_COLOR

def extract_prediction_features(hand_landmarks, previous_distances):
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])
    if hand_scale <= 0: return None, previous_distances
    tips = [4, 8, 12, 16, 20]
    current_distances = [get_dist(wrist, hand_landmarks.landmark[t]) / hand_scale for t in tips]
    speeds = [0.0] * 5 if previous_distances is None else [cur - prv for cur, prv in zip(current_distances, previous_distances)]
    inter = [get_dist(hand_landmarks.landmark[tips[i]], hand_landmarks.landmark[tips[i+1]]) / hand_scale for i in range(4)]
    return current_distances + speeds + inter, current_distances

def choose_robot_move_to_win(predicted_move, selected_side):
    predicted_move = max(0, min(5, int(predicted_move)))
    if selected_side == "EVEN": return 1 if predicted_move % 2 == 0 else 2
    return 2 if predicted_move % 2 == 0 else 1

def calculate_result(actual_move, robot_choice, selected_side):
    if actual_move < 0: return "No clear final hand", NOTE_COLOR
    total = actual_move + robot_choice
    user_wins = (selected_side == "EVEN" and total % 2 == 0) or (selected_side == "ODD" and total % 2 != 0)
    if user_wins: return "USER WINS", BAD_COLOR
    return "ROBOT WINS", GOOD_COLOR

def start_round():
    global state, countdown_start, verify_start, candidate_actual, candidate_actual_start
    global predicted_user_move, actual_user_move, robot_move, result_text, result_color
    global confidence_percent, locked_features, has_predicted, prev_distances
    state = "COUNTDOWN"
    countdown_start = time.time()
    verify_start = candidate_actual_start = 0.0
    candidate_actual = None
    predicted_user_move = actual_user_move = -1
    robot_move = 0
    result_text, result_color = "", INFO_COLOR
    confidence_percent = 0.0
    locked_features = prev_distances = None
    has_predicted = False

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
        if "EVEN" in command: selected_side = "EVEN"; state = "READY"; return True
        if "ODD" in command: selected_side = "ODD"; state = "READY"; return True
    if ("START" in command or "BEGIN" in command or "PLAY" in command or "AGAIN" in command) and state in ["READY", "RESULT"]:
        start_round()
        return True
    return False


def get_spoken_instruction():
    if state == "CHOOSE_SIDE":
        return "Choose your side. Show two fingers for even, or one finger for odd. You can also say even or odd. Say back to return."
    if state == "READY":
        return f"You selected {selected_side.lower()}. Show thumbs up, or say start, to begin."
    if state == "COUNTDOWN":
        return "Get ready. Show a number from zero to five before the countdown ends."
    if state == "VERIFY_ACTUAL":
        return f"The robot showed {robot_move}. Hold your final number steady."
    if state == "RESULT":
        return f"{result_text}. Your number was {actual_user_move}, the robot showed {robot_move}. Show thumbs up, or say start, to play again."
    return ""

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
model = None

try:
    with open(BASE_DIR / "hand_predictor.pkl", "rb") as f: 
        model = pickle.load(f)
except Exception as e: 
    print(f"CRITICAL ERROR loading ML model: {e}")
    sys.exit(1)

ser = init_serial_connection()
init_voice()

cap = cv2.VideoCapture(0)
attempts = 0
while not cap.isOpened() and attempts < 5:
    print("Camera busy... waiting 0.5s before retrying...")
    time.sleep(0.5)
    cap = cv2.VideoCapture(0)
    attempts += 1

if not cap.isOpened():
    print("CRITICAL ERROR: Camera could not be opened by the game!")
    stop_voice()
    sys.exit(1)

try:
    with mp_hands.Hands(model_complexity=1, max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: continue
            frame = cv2.flip(frame, 1)
            camera_view, panel = create_split_screen(frame)
            h, w, _ = frame.shape
            
            results = hands.process(cv2.cvtColor(camera_view, cv2.COLOR_BGR2RGB))
            current_fingers = -1
            current_ok = current_thumbs = False
            current_features = None
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(camera_view, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    current_fingers = count_fingers_single_hand(hand_landmarks)
                    current_ok = is_ok_gesture(hand_landmarks)
                    current_thumbs = is_thumbs_up(hand_landmarks)
                    current_features, prev_distances = extract_prediction_features(hand_landmarks, prev_distances)
            else:
                prev_distances = None
                
            current_time = time.time()
            while voice_queue: process_voice_command(voice_queue.popleft())
            if state == "CHOOSE_SIDE":
                draw_lines(panel, ["CHOOSE YOUR SIDE"], 40, TITLE_COLOR, scale=1.0, thickness=2)
                draw_lines(panel, ["Show 2 fingers = EVEN", "Show 1 finger = ODD"], 90, OPTION_COLOR, scale=0.75, thickness=2, step=30)
                draw_lines(panel, ["Voice: say EVEN or ODD", "OK sign = Back to Menu"], 170, NOTE_COLOR, scale=0.65, thickness=2, step=25)
                
                action, _ = update_hold("BACK" if current_ok else ("EVEN" if current_fingers==2 else ("ODD" if current_fingers==1 else None)), current_time)
                if action == "BACK": send_robot_fingers(ser, 0); sys.exit(BACK_EXIT_CODE)
                elif action in ["EVEN", "ODD"]: selected_side = action; state = "READY"
                    
            elif state == "READY":
                draw_lines(panel, ["EVEN ODD GAME"], 40, TITLE_COLOR, scale=1.0, thickness=2)
                draw_lines(panel, [f"Your side: {selected_side}", "Thumbs up = Start game"], 90, OPTION_COLOR, scale=0.75, thickness=2, step=30)
                draw_lines(panel, ["OK sign = Back to Menu"], 170, NOTE_COLOR, scale=0.65, thickness=2, step=25)
                
                action, _ = update_hold("START" if current_thumbs else ("BACK" if current_ok else None), current_time)
                if action == "START": start_round()
                elif action == "BACK": send_robot_fingers(ser, 0); sys.exit(BACK_EXIT_CODE)
                    
            elif state == "COUNTDOWN":
                elapsed = current_time - countdown_start
                count = int(math.ceil(max(0.0, COUNTDOWN_DURATION - elapsed)))
                draw_lines(panel, ["EVEN ODD GAME"], 40, TITLE_COLOR, scale=1.0, thickness=2)
                draw_lines(panel, [f"Side: {selected_side}", "Show number now"], 80, OPTION_COLOR, scale=0.75, thickness=2, step=30)
                draw_lines(panel, ["Robot predicts before end"], 150, NOTE_COLOR, scale=0.65, thickness=2)
                
                if count > 0:
                    cv2.putText(camera_view, str(count), (w // 2 - 45, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 4.6, OPTION_COLOR, 8)
                if current_fingers != -1:
                    cv2.putText(panel, f"Detected: {current_fingers}", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, INFO_COLOR, 2)
                    
                draw_lines(panel, ["OK sign = Back"], 240, NOTE_COLOR, scale=0.65, thickness=2)
                
                action, _ = update_hold("BACK" if current_ok else None, current_time)
                if action == "BACK": send_robot_fingers(ser, 0); sys.exit(BACK_EXIT_CODE)
                
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
                    state, verify_start, candidate_actual = "VERIFY_ACTUAL", current_time, None
                    
            elif state == "VERIFY_ACTUAL":
                elapsed = current_time - verify_start
                draw_lines(panel, ["FINAL CHECK"], 40, TITLE_COLOR, scale=1.0, thickness=2)
                draw_lines(panel, [f"Robot played: {robot_move}", "Hold final number"], 80, OPTION_COLOR, scale=0.75, thickness=2, step=30)
                draw_lines(panel, [f"Predicted: {predicted_user_move} ({confidence_percent:.1f}%)"], 150, NOTE_COLOR, scale=0.65, thickness=2)
                
                if current_fingers != -1:
                    cv2.putText(panel, f"Detected: {current_fingers}", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, INFO_COLOR, 2)
                    if candidate_actual != current_fingers:
                        candidate_actual, candidate_actual_start = current_fingers, current_time
                    else:
                        hold = current_time - candidate_actual_start
                        cv2.putText(panel, f"Stable hold: {hold:.1f}s / {ANSWER_STABLE_SECONDS:.1f}s", (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.65, HOLD_COLOR, 2)
                        if hold >= ANSWER_STABLE_SECONDS:
                            actual_user_move = current_fingers
                            finish_to_result()
                else: candidate_actual = None
                
                action, _ = update_hold("BACK" if current_ok else None, current_time)
                if action == "BACK": send_robot_fingers(ser, 0); sys.exit(BACK_EXIT_CODE)
                if elapsed >= VERIFY_ACTUAL_SECONDS and state == "VERIFY_ACTUAL":
                    actual_user_move = current_fingers
                    finish_to_result()
                    
            elif state == "RESULT":
                total = actual_user_move + robot_move if actual_user_move >= 0 else "?"
                draw_lines(panel, ["RESULT"], 40, TITLE_COLOR, scale=1.0, thickness=2)
                draw_lines(panel, [f"Side: {selected_side}", f"Predict: {predicted_user_move} ({confidence_percent:.1f}%)", f"Actual: {actual_user_move}", f"Robot: {robot_move}", f"Total: {total}"], 80, NOTE_COLOR, scale=0.65, thickness=2, step=25)
                draw_lines(panel, [result_text], 220, result_color, scale=1.0, thickness=2)
                draw_lines(panel, ["Thumbs up = Start again", "Voice: START", "OK sign = Back to Menu"], 260, OPTION_COLOR, scale=0.65, thickness=2, step=25)
                
                action, _ = update_hold("START" if current_thumbs else ("BACK" if current_ok else None), current_time)
                if action == "START": send_robot_fingers(ser, 0); start_round()
                elif action == "BACK": send_robot_fingers(ser, 0); sys.exit(BACK_EXIT_CODE)
                    
            voice_guide.announce(state, get_spoken_instruction())

            draw_hold_status(panel, current_time)
            combined_screen = cv2.hconcat([camera_view, panel])
            combined_screen = cv2.resize(combined_screen, (1280, 520))
            window_name = "Predictive Even-Odd Game - Voice + Gesture"
            cv2.imshow(window_name, combined_screen)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
except KeyboardInterrupt:
    print("\nClosing game...")

finally:
    stop_voice()
    voice_guide.stop()
    send_robot_fingers(ser, 0)
    cap.release()
    cv2.destroyAllWindows()
    if ser is not None and ser.is_open: ser.close()