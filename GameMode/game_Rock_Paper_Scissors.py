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
    import sklearn  # Required because the pickle model was trained with scikit-learn
except ImportError:
    print(
        " scikit-learn is required to load the ML model. Install it with:\n"
        "  python -m pip install scikit-learn"
    )
    sys.exit(1)

BASE_DIR = Path(__file__).resolve().parent

# -------------------------------------------------
# Rock Paper Scissors Predictive Game Configuration
# -------------------------------------------------

ARDUINO_PORT = "COM4"
ARDUINO_BAUD_RATE = 9600

COUNTDOWN_DURATION = 3.0
FEATURE_LOCK_BEFORE_END = 0.3
FEATURE_LOCK_TIME = COUNTDOWN_DURATION - FEATURE_LOCK_BEFORE_END  # 2.7 seconds
ROBOT_DECISION_TIME = COUNTDOWN_DURATION                          # 3.0 seconds

MIN_FINGERS = 0
MAX_FINGERS = 5

# Robot commands:
# ROCK     -> 0 fingers
# PAPER    -> 5 fingers
# SCISSORS -> 2 fingers
ROBOT_COMMANDS = {
    "ROCK": 0,
    "PAPER": 5,
    "SCISSORS": 2
}

# What beats what
WINNING_MOVE = {
    "ROCK": "PAPER",
    "PAPER": "SCISSORS",
    "SCISSORS": "ROCK"
}

# -------------------------------------------------
# Global Game Variables
# -------------------------------------------------

game_state = "READY"  # READY -> COUNTDOWN -> RESULT
countdown_start = 0
last_state_change_time = 0

locked_prediction_features = None
locked_features_time = None
has_predicted_this_round = False

predicted_user_move = "UNKNOWN"
actual_user_move = "UNKNOWN"
robot_move = "UNKNOWN"
confidence_percent = 0.0
result_text = ""
result_color = (255, 255, 255)

prev_distances = None
ser = None

# -------------------------------------------------
# Arduino / Simulation Mode
# -------------------------------------------------


def limit_fingers(number):
    return max(MIN_FINGERS, min(MAX_FINGERS, int(number)))


def init_serial_connection():
    """
    Tries to connect to Arduino.
    If Arduino is not connected, the game keeps running in simulation mode.
    """
    try:
        arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"Connected to Arduino on {ARDUINO_PORT}")
        return arduino
    except Exception as e:
        print("Arduino not connected. Running in simulation mode.")
        print(f"  Serial details: {e}")
        return None


def send_to_arduino_fingers(number):
    """
    Sends a finger-count command to Arduino if connected.
    Otherwise prints what would have been sent.
    """
    global ser

    safe_command = limit_fingers(number)

    if ser is not None and ser.is_open:
        try:
            ser.write(str(safe_command).encode())
            print(f"Sent to Arduino: {safe_command}")
        except Exception as e:
            print(f"Arduino send failed. Switching to simulation mode. Details: {e}")
            ser = None
            print(f"Simulation mode - would send to Arduino: {safe_command}")
    else:
        print(f"Simulation mode - would send to Arduino: {safe_command}")


def send_robot_move(move_name):
    """
    Converts ROCK/PAPER/SCISSORS to the finger command that the robot should show.
    """
    command = ROBOT_COMMANDS.get(move_name, 0)
    send_to_arduino_fingers(command)


# -------------------------------------------------
# Helper Functions
# -------------------------------------------------


def get_dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def count_fingers_from_landmarks(hand_landmarks):
    """
    Stable rule-based finger counter.
    Used for final result verification and gesture helpers.
    """
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])

    if hand_scale <= 0:
        return -1

    fingers = []

    # Thumb
    if get_dist(hand_landmarks.landmark[4], hand_landmarks.landmark[5]) > hand_scale * 0.6:
        fingers.append(1)
    else:
        fingers.append(0)

    # Index, middle, ring, pinky
    tips_idx = [8, 12, 16, 20]
    mips_idx = [6, 10, 14, 18]

    for tip, mip in zip(tips_idx, mips_idx):
        tip_dist = get_dist(wrist, hand_landmarks.landmark[tip])
        mip_dist = get_dist(wrist, hand_landmarks.landmark[mip])

        if mip_dist > 0 and tip_dist / mip_dist > 1.15:
            fingers.append(1)
        else:
            fingers.append(0)

    return limit_fingers(sum(fingers))


def classify_rps_from_fingers(finger_count):
    """
    Converts final finger count to an RPS gesture:
    0 -> ROCK
    2 -> SCISSORS
    5 -> PAPER
    """
    if finger_count == 0:
        return "ROCK"
    if finger_count == 2:
        return "SCISSORS"
    if finger_count == 5:
        return "PAPER"
    return "UNKNOWN"


def map_model_output_to_rps(raw_prediction):
    """
    Supports using the existing hand_predictor.pkl for a first version.
    If the model predicts 0, 2, or 5 directly, we map them to RPS.
    If it predicts 1/3/4, we map to the nearest RPS gesture.
    Best future upgrade: train a dedicated rps_predictor.pkl with labels ROCK/PAPER/SCISSORS.
    """
    try:
        pred = int(raw_prediction)
    except Exception:
        pred_str = str(raw_prediction).upper()
        if "ROCK" in pred_str:
            return "ROCK"
        if "PAPER" in pred_str:
            return "PAPER"
        if "SCISSORS" in pred_str or "SCISSOR" in pred_str:
            return "SCISSORS"
        return "UNKNOWN"

    if pred == 0:
        return "ROCK"
    if pred == 2:
        return "SCISSORS"
    if pred == 5:
        return "PAPER"

    # Fallback mapping for numeric predictions that are not valid RPS gestures
    if pred <= 1:
        return "ROCK"
    if pred in [2, 3]:
        return "SCISSORS"
    return "PAPER"


def is_ok_gesture(hand_landmarks):
    """
    OK sign gesture for rematch:
    thumb tip close to index tip, while middle/ring/pinky are relatively open.
    This avoids using 0, 2, or 5, which are meaningful in Rock/Paper/Scissors.
    """
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])

    if hand_scale <= 0:
        return False

    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    thumb_index_dist = get_dist(thumb_tip, index_tip)

    def is_finger_open(tip_idx, mip_idx):
        mip_dist = get_dist(wrist, hand_landmarks.landmark[mip_idx])
        if mip_dist <= 0:
            return False
        tip_dist = get_dist(wrist, hand_landmarks.landmark[tip_idx])
        return tip_dist / mip_dist > 1.10

    middle_open = is_finger_open(12, 10)
    ring_open = is_finger_open(16, 14)
    pinky_open = is_finger_open(20, 18)

    return thumb_index_dist < hand_scale * 0.35 and middle_open and ring_open and pinky_open


def is_thumb_up_gesture(hand_landmarks):
    """
    Start gesture:
    thumb open, other fingers closed.
    This is intentionally different from ROCK/PAPER/SCISSORS and from OK.
    """
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


def is_back_gesture(hand_landmarks):
    """
    Back gesture: Shaka / Call Me sign.
    Thumb and pinky are open, while index, middle, and ring are closed.
    Used only in safe states: READY and RESULT.
    """
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

    return thumb_open and pinky_open and not index_open and not middle_open and not ring_open


def extract_prediction_features(hand_landmarks, previous_distances):
    """
    Extracts 14 features for the prediction model:
    - 5 normalized distances from wrist to fingertips
    - 5 movement speeds compared to previous frame
    - 4 distances between neighboring fingertips
    """
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])

    if hand_scale <= 0:
        return None, previous_distances

    tips = [4, 8, 12, 16, 20]
    current_distances = [get_dist(wrist, hand_landmarks.landmark[t]) / hand_scale for t in tips]

    if previous_distances is None:
        speeds = [0.0] * 5
    else:
        speeds = [cur - prv for cur, prv in zip(current_distances, previous_distances)]

    inter_finger_dist = []
    for i in range(len(tips) - 1):
        d = get_dist(
            hand_landmarks.landmark[tips[i]],
            hand_landmarks.landmark[tips[i + 1]]
        ) / hand_scale
        inter_finger_dist.append(d)

    features = current_distances + speeds + inter_finger_dist
    return features, current_distances


def reset_round():
    global countdown_start, locked_prediction_features, locked_features_time
    global has_predicted_this_round, predicted_user_move, actual_user_move
    global robot_move, confidence_percent, result_text, result_color
    global prev_distances

    countdown_start = time.time()
    locked_prediction_features = None
    locked_features_time = None
    has_predicted_this_round = False

    predicted_user_move = "UNKNOWN"
    actual_user_move = "UNKNOWN"
    robot_move = "UNKNOWN"
    confidence_percent = 0.0
    result_text = ""
    result_color = (255, 255, 255)
    prev_distances = None


def start_game():
    global game_state, last_state_change_time

    reset_round()
    game_state = "COUNTDOWN"
    last_state_change_time = time.time()
    print("RPS round started")


def calculate_result(actual_move, robot_choice):
    if actual_move == "UNKNOWN":
        return "No clear hand detected", (0, 165, 255)

    if robot_choice == "UNKNOWN":
        return "Robot did not choose", (0, 165, 255)

    if actual_move == robot_choice:
        return "TIE", (0, 255, 255)

    if WINNING_MOVE[actual_move] == robot_choice:
        return "ROBOT WINS!", (0, 255, 0)

    return "USER WINS!", (0, 0, 255)


# -------------------------------------------------
# Voice Recognition Callback
# -------------------------------------------------


def voice_callback(recognizer, audio):
    global game_state, last_state_change_time

    try:
        command = recognizer.recognize_google(audio, language="en-US").upper()
        print(f"Heard voice command: {command}")

        if game_state == "READY":
            if any(word in command for word in ["START", "BEGIN", "GO", "PLAY"]):
                start_game()

        elif game_state == "RESULT":
            if any(word in command for word in ["AGAIN", "GAIN", "GAME", "REMATCH", "BEGIN", "START"]):
                send_to_arduino_fingers(0)
                game_state = "READY"
                last_state_change_time = time.time()
                print("State changed to READY via Voice")

    except sr.UnknownValueError:
        pass
    except sr.RequestError as e:
        print(f"Voice Service Error: {e}")


# -------------------------------------------------
# Init Voice Recognition
# -------------------------------------------------

r = sr.Recognizer()
m = sr.Microphone()

r.energy_threshold = 1000
r.dynamic_energy_threshold = False
r.non_speaking_duration = 0.3
r.pause_threshold = 0.3

print("Calibrating microphone context...")
with m as source:
    r.adjust_for_ambient_noise(source, duration=1)

stop_listening = r.listen_in_background(m, voice_callback, phrase_time_limit=1.2)
print("Voice control active: say START / AGAIN")

# -------------------------------------------------
# Init MediaPipe and ML Model
# -------------------------------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

model = None
model_path = BASE_DIR / "rps_predictor.pkl"
fallback_model_path = BASE_DIR / "hand_predictor.pkl"

try:
    if model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("RPS Predictive ML Model Loaded: rps_predictor.pkl")
    else:
        with open(fallback_model_path, "rb") as f:
            model = pickle.load(f)
        print("Fallback ML Model Loaded: hand_predictor.pkl")
        print("ℹ For best results, train a dedicated rps_predictor.pkl later.")
except Exception as e:
    print(f"Could not load ML model: {e}")
    stop_listening(wait_for_stop=False)
    sys.exit(1)

ser = init_serial_connection()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera Error: could not open camera")
    stop_listening(wait_for_stop=False)
    if ser is not None and ser.is_open:
        ser.close()
    sys.exit(1)

print("Rock Paper Scissors Predictive Mode Started")
print("ℹ Press 'q' to quit")

# -------------------------------------------------
# Main Loop
# -------------------------------------------------

try:
    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8
    ) as hands:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            current_features = None
            current_finger_count = -1
            current_rps_gesture = "UNKNOWN"
            current_ok_gesture = False
            current_thumb_up_gesture = False
            current_back_gesture = False

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    current_finger_count = count_fingers_from_landmarks(hand_landmarks)
                    current_rps_gesture = classify_rps_from_fingers(current_finger_count)
                    current_ok_gesture = is_ok_gesture(hand_landmarks)
                    current_thumb_up_gesture = is_thumb_up_gesture(hand_landmarks)
                    current_back_gesture = is_back_gesture(hand_landmarks)
                    current_features, prev_distances = extract_prediction_features(hand_landmarks, prev_distances)
            else:
                prev_distances = None

            current_time = time.time()

            # -----------------------------
            # READY
            # -----------------------------
            if game_state == "READY":
                cv2.putText(frame, "Rock Paper Scissors", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                cv2.putText(frame, "Say 'START' or show THUMBS UP", (30, h // 2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 2)
                cv2.putText(frame, "Rock=0 | Paper=5 | Scissors=2", (30, h // 2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
                cv2.putText(frame, "Back gesture: return to Game Menu", (30, h // 2 + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)

                if current_finger_count != -1:
                    cv2.putText(frame, f"Detected now: {current_rps_gesture} ({current_finger_count})", (30, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

                if current_time - last_state_change_time > 1.0:
                    if current_back_gesture:
                        send_to_arduino_fingers(0)
                        print("Back gesture detected - returning to Game Menu")
                        sys.exit(10)
                    elif current_thumb_up_gesture:
                        start_game()

            # -----------------------------
            # COUNTDOWN + FEATURE LOCK 0.3s BEFORE END
            # -----------------------------
            elif game_state == "COUNTDOWN":
                elapsed = current_time - countdown_start
                remaining = max(0.0, COUNTDOWN_DURATION - elapsed)
                count = int(math.ceil(remaining))

                cv2.putText(frame, "Get ready: ROCK / PAPER / SCISSORS", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)

                if count > 0:
                    cv2.putText(frame, str(count), (w // 2 - 50, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 10)

                # Debug display only; prediction does NOT use this final text after lock.
                if current_finger_count != -1:
                    cv2.putText(frame, f"Detected now: {current_rps_gesture}", (30, h - 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

                if locked_prediction_features is None and elapsed >= FEATURE_LOCK_TIME:
                    if current_features is not None:
                        locked_prediction_features = list(current_features)
                        locked_features_time = elapsed
                        print(f"Features locked at {elapsed:.2f}s, exactly {FEATURE_LOCK_BEFORE_END:.1f}s before decision")
                    else:
                        cv2.putText(frame, "Waiting for hand features to lock...", (30, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)

                if locked_prediction_features is not None:
                    cv2.putText(frame, f"Features locked at {locked_features_time:.2f}s", (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if elapsed >= ROBOT_DECISION_TIME and not has_predicted_this_round:
                    if locked_prediction_features is not None:
                        probabilities = model.predict_proba([locked_prediction_features])[0]
                        raw_pred = int(np.argmax(probabilities))
                        confidence_percent = float(probabilities[raw_pred] * 100)

                        predicted_user_move = map_model_output_to_rps(raw_pred)

                        if predicted_user_move == "UNKNOWN":
                            robot_move = random.choice(["ROCK", "PAPER", "SCISSORS"])
                        else:
                            robot_move = WINNING_MOVE[predicted_user_move]

                        send_robot_move(robot_move)
                        has_predicted_this_round = True

                        print(
                            f" Prediction from locked features: "
                            f"User -> {predicted_user_move} ({confidence_percent:.1f}%) "
                            f"-> Robot plays {robot_move}"
                        )
                    else:
                        predicted_user_move = "UNKNOWN"
                        confidence_percent = 0.0
                        robot_move = random.choice(["ROCK", "PAPER", "SCISSORS"])
                        send_robot_move(robot_move)
                        has_predicted_this_round = True
                        print(f"No locked features. Robot fallback move: {robot_move}")

                    actual_user_move = current_rps_gesture
                    result_text, result_color = calculate_result(actual_user_move, robot_move)
                    game_state = "RESULT"
                    last_state_change_time = current_time

            # -----------------------------
            # RESULT
            # -----------------------------
            elif game_state == "RESULT":
                cv2.putText(frame, "Result", (50, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
                cv2.putText(frame, f"Predicted User: {predicted_user_move} ({confidence_percent:.1f}%)", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 0), 2)
                cv2.putText(frame, f"Actual User: {actual_user_move}", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
                cv2.putText(frame, f"Robot Played: {robot_move}", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                cv2.putText(frame, result_text, (50, 310), cv2.FONT_HERSHEY_SIMPLEX, 1.5, result_color, 4)
                cv2.putText(frame, "OK / AGAIN = rematch | Back gesture = Game Menu", (30, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

                if current_time - last_state_change_time > 1.0:
                    if current_back_gesture:
                        send_to_arduino_fingers(0)
                        print("Back gesture detected - returning to Game Menu")
                        sys.exit(10)
                    elif current_ok_gesture:
                        send_to_arduino_fingers(0)
                        game_state = "READY"
                        last_state_change_time = current_time
                        print("State changed to READY via OK Gesture")

            cv2.imshow("Predictive Rock Paper Scissors - Voice + Gesture", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    stop_listening(wait_for_stop=False)
    cap.release()
    cv2.destroyAllWindows()

    if ser is not None and ser.is_open:
        send_to_arduino_fingers(0)
        ser.close()
        print("Arduino connection closed")

    print("RPS Game Mode Closed")
