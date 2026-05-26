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
        "scikit-learn is required to load the ML model. Install it with:\n"
        "  python -m pip install scikit-learn"
    )
    sys.exit(1)

BASE_DIR = Path(__file__).resolve().parent

# -------------------------------------------------
# Game Configuration
# -------------------------------------------------

ARDUINO_PORT = "COM4"
ARDUINO_BAUD_RATE = 9600

COUNTDOWN_DURATION = 3.0

# Fair prediction timing:
# The hand features are locked only 0.3 seconds before the countdown ends.
# The robot decision is made at the end of the countdown using ONLY those locked features.
# This gives the robot only 0.3 seconds of information gap, not 2.5 seconds.
FEATURE_LOCK_BEFORE_END = 0.3
FEATURE_LOCK_TIME = COUNTDOWN_DURATION - FEATURE_LOCK_BEFORE_END
ROBOT_DECISION_TIME = COUNTDOWN_DURATION

MIN_FINGERS = 0
MAX_FINGERS = 5
BACK_EXIT_CODE = 10

# -------------------------------------------------
# Global Game Variables
# -------------------------------------------------

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

# -------------------------------------------------
# Arduino / Simulation Mode
# -------------------------------------------------


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
        print(f"   Serial details: {e}")
        return None


def send_to_arduino(command):
    """
    Sends a command to Arduino if connected.
    Otherwise prints what would have been sent.
    """
    global ser

    safe_command = limit_fingers(command)

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


# -------------------------------------------------
# Helper Functions
# -------------------------------------------------


def get_dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def limit_fingers(number):
    return max(MIN_FINGERS, min(MAX_FINGERS, int(number)))


def count_fingers_from_landmarks(hand_landmarks):
    """
    Stable rule-based finger counter.
    Used for menus/debug/result verification, not for the robot's prediction decision.
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


def is_ok_gesture(hand_landmarks):
    """
    OK sign gesture for rematch:
    thumb tip close to index tip, while middle/ring/pinky are relatively open.
    """
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])

    if hand_scale <= 0:
        return False

    thumb_index_dist = get_dist(hand_landmarks.landmark[4], hand_landmarks.landmark[8])

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


def is_back_gesture(hand_landmarks):
    """
    Back gesture: Shaka / Call Me sign.
    Thumb and pinky are open, index/middle/ring are closed.

    This gesture is used only in safe states:
    - CHOOSING
    - RESULT
    It is not active during COUNTDOWN.
    """
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])

    if hand_scale <= 0:
        return False

    def is_finger_open(tip_idx, mip_idx):
        mip_dist = get_dist(wrist, hand_landmarks.landmark[mip_idx])
        if mip_dist <= 0:
            return False
        tip_dist = get_dist(wrist, hand_landmarks.landmark[tip_idx])
        return tip_dist / mip_dist > 1.15

    thumb_open = get_dist(hand_landmarks.landmark[4], hand_landmarks.landmark[5]) > hand_scale * 0.65
    index_open = is_finger_open(8, 6)
    middle_open = is_finger_open(12, 10)
    ring_open = is_finger_open(16, 14)
    pinky_open = is_finger_open(20, 18)

    return thumb_open and pinky_open and not index_open and not middle_open and not ring_open


def return_to_previous_menu():
    """
    Cleans up the robot hand and exits with a dedicated code.
    The main menu can detect this code and return to the Game Mode menu.
    """
    send_to_arduino(0)
    print("Back gesture detected - returning to previous menu")
    sys.exit(BACK_EXIT_CODE)


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
        d = get_dist(hand_landmarks.landmark[tips[i]], hand_landmarks.landmark[tips[i + 1]]) / hand_scale
        inter_finger_dist.append(d)

    features = current_distances + speeds + inter_finger_dist
    return features, current_distances


def choose_robot_move_to_win(predicted_move, selected_side):
    predicted_move = limit_fingers(predicted_move)

    if selected_side == "EVEN":
        # User wins if total is even, so robot tries to make total odd.
        return 1 if predicted_move % 2 == 0 else 2

    if selected_side == "ODD":
        # User wins if total is odd, so robot tries to make total even.
        return 2 if predicted_move % 2 == 0 else 1

    return random.randint(1, 2)


def calculate_result(actual_move, robot_choice, selected_side):
    if actual_move < 0:
        return "No clear hand detected", (0, 165, 255)

    total = actual_move + robot_choice
    total_is_even = total % 2 == 0
    user_wins = (selected_side == "EVEN" and total_is_even) or (selected_side == "ODD" and not total_is_even)

    if user_wins:
        return "USER WINS!", (0, 255, 255)

    return "ROBOT WINS!", (0, 255, 0)


def reset_round(selected_side, start_time):
    global user_side, game_state, countdown_start, has_predicted_this_round
    global robot_move, predicted_user_move, actual_user_move, confidence_percent
    global result_text, result_color, locked_prediction_features, locked_features_time

    user_side = selected_side
    game_state = "COUNTDOWN"
    countdown_start = start_time
    has_predicted_this_round = False
    robot_move = 0
    predicted_user_move = -1
    actual_user_move = -1
    confidence_percent = 0.0
    result_text = ""
    result_color = (255, 255, 255)
    locked_prediction_features = None
    locked_features_time = None


# -------------------------------------------------
# Voice Recognition Callback
# -------------------------------------------------


def voice_callback(recognizer, audio):
    global game_state

    try:
        command = recognizer.recognize_google(audio, language="en-US").upper()
        print(f"Heard voice command: {command}")

        if game_state == "CHOOSING":
            if any(word in command for word in ["EVEN", "EVENT", "EVAN"]):
                reset_round("EVEN", time.time())
                print("State changed to COUNTDOWN (EVEN) via Voice")

            elif any(word in command for word in ["ODD", "ADD", "OLD"]):
                reset_round("ODD", time.time())
                print("State changed to COUNTDOWN (ODD) via Voice")

        elif game_state == "RESULT":
            if any(word in command for word in ["AGAIN", "GAIN", "GAME", "REMATCH", "BEGIN"]):
                send_to_arduino(0)
                game_state = "CHOOSING"
                print("State changed to CHOOSING via Voice")

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
print("Multi-modal control active (Voice + Gesture)!")

# -------------------------------------------------
# Init MediaPipe and ML Model
# -------------------------------------------------

try:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    print("MediaPipe does not expose mp.solutions directly. Check your mediapipe installation.")
    stop_listening(wait_for_stop=False)
    sys.exit(1)

try:
    with open(BASE_DIR / "hand_predictor.pkl", "rb") as f:
        model = pickle.load(f)
    print("Advanced Predictive ML Model Loaded")
except Exception as e:
    print(f"Could not load ML model: {e}. Run train_predictor.py first.")
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

print("Predictive Game Mode Started")
print(f"Feature lock time: {FEATURE_LOCK_TIME:.2f}s | Robot decision/end time: {ROBOT_DECISION_TIME:.2f}s")
print("Press 'q' to quit")

# -------------------------------------------------
# Main Loop
# -------------------------------------------------

try:
    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
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
            current_ok_gesture = False
            current_back_gesture = False

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    current_finger_count = count_fingers_from_landmarks(hand_landmarks)
                    current_ok_gesture = is_ok_gesture(hand_landmarks)
                    current_back_gesture = is_back_gesture(hand_landmarks)
                    current_features, prev_distances = extract_prediction_features(hand_landmarks, prev_distances)
            else:
                prev_distances = None

            current_time = time.time()

            # -----------------------------
            # CHOOSING
            # -----------------------------
            if game_state == "CHOOSING":
                cv2.putText(frame, "Choose your side", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                cv2.putText(frame, "Say 'EVEN' or Show 2 Fingers", (30, h // 2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(frame, "Say 'ODD' or Show 1 Finger", (30, h // 2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(frame, "Back gesture: return to Game Menu", (30, h // 2 + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)

                if current_finger_count != -1:
                    cv2.putText(frame, f"Detected fingers: {current_finger_count}", (30, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

                if current_time - last_state_change_time > 1.0:
                    if current_back_gesture:
                        return_to_previous_menu()
                    elif current_finger_count == 2:
                        reset_round("EVEN", current_time)
                        last_state_change_time = current_time
                        print("State changed to COUNTDOWN (EVEN) via Gesture")
                    elif current_finger_count == 1:
                        reset_round("ODD", current_time)
                        last_state_change_time = current_time
                        print("State changed to COUNTDOWN (ODD) via Gesture")

            # -----------------------------
            # COUNTDOWN + FEATURE LOCK + DECISION
            # -----------------------------
            elif game_state == "COUNTDOWN":
                elapsed = current_time - countdown_start
                remaining = max(0.0, COUNTDOWN_DURATION - elapsed)
                count = int(math.ceil(remaining))

                cv2.putText(frame, f"You chose: {user_side}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                if count > 0:
                    cv2.putText(frame, str(count), (w // 2 - 50, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 10)

                # Hidden during countdown on purpose: do not show/debug the final answer while predicting.
                # if current_finger_count != -1:
                #     cv2.putText(frame, f"Detected now: {current_finger_count}", (30, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

                # Step 1: lock features only 0.3 seconds before the countdown ends.
                if elapsed >= FEATURE_LOCK_TIME and locked_prediction_features is None:
                    if current_features is not None:
                        locked_prediction_features = list(current_features)
                        locked_features_time = elapsed
                        print(f"Features locked at {elapsed:.2f}s")
                    else:
                        cv2.putText(frame, "Waiting for hand features to lock...", (30, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)

                if locked_prediction_features is not None and not has_predicted_this_round:
                    cv2.putText(frame, f"Features locked at {locked_features_time:.2f}s", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # Step 2: decide at the end, using ONLY the locked features from 0.3 seconds earlier.
                if elapsed >= ROBOT_DECISION_TIME and not has_predicted_this_round:
                    if locked_prediction_features is not None:
                        probabilities = model.predict_proba([locked_prediction_features])[0]
                        predicted_user_move = int(np.argmax(probabilities))
                        confidence_percent = probabilities[predicted_user_move] * 100

                        robot_move = choose_robot_move_to_win(predicted_user_move, user_side)
                        send_to_arduino(robot_move)
                        has_predicted_this_round = True

                        print(
                            f"Prediction decided at {elapsed:.2f}s using features from {locked_features_time:.2f}s: "
                            f"User -> {predicted_user_move} ({confidence_percent:.1f}%) -> Robot plays {robot_move}"
                        )
                    else:
                        # If no lock was possible, use current early features as a fallback.
                        if current_features is not None:
                            locked_prediction_features = list(current_features)
                            locked_features_time = elapsed
                            print(f"Fallback lock at decision time: {elapsed:.2f}s")
                        else:
                            robot_move = random.randint(1, 2)
                            predicted_user_move = -1
                            confidence_percent = 0.0
                            send_to_arduino(robot_move)
                            has_predicted_this_round = True
                            print(f"No hand features. Robot fallback move: {robot_move}")

                if has_predicted_this_round:
                    cv2.putText(frame, "Prediction locked", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
                    cv2.putText(frame, f"Robot already chose: {robot_move}", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)

                if elapsed >= COUNTDOWN_DURATION:
                    if not has_predicted_this_round:
                        robot_move = random.randint(1, 2)
                        predicted_user_move = -1
                        confidence_percent = 0.0
                        send_to_arduino(robot_move)
                        has_predicted_this_round = True
                        print(f"No prediction was possible. Robot fallback move: {robot_move}")

                    actual_user_move = current_finger_count
                    result_text, result_color = calculate_result(actual_user_move, robot_move, user_side)
                    game_state = "RESULT"
                    last_state_change_time = current_time

            # -----------------------------
            # RESULT
            # -----------------------------
            elif game_state == "RESULT":
                cv2.putText(frame, f"User side: {user_side}", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(frame, f"Predicted Move: {predicted_user_move} ({confidence_percent:.1f}%)", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                cv2.putText(frame, f"Actual Move: {actual_user_move}", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(frame, f"Robot Played: {robot_move}", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

                if actual_user_move >= 0:
                    total = actual_user_move + robot_move
                    cv2.putText(frame, f"Total: {total}", (50, 275), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                cv2.putText(frame, result_text, (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, result_color, 4)
                cv2.putText(frame, "Say 'AGAIN' or Show OK Sign for Rematch", (30, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 2)
                cv2.putText(frame, "Back gesture: return to Game Menu", (30, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 2)

                if current_time - last_state_change_time > 1.0:
                    if current_back_gesture:
                        return_to_previous_menu()
                    elif current_ok_gesture:
                        send_to_arduino(0)
                        game_state = "CHOOSING"
                        last_state_change_time = current_time
                        print("State changed to CHOOSING via OK Gesture")

            cv2.imshow("Predictive Even-Odd Game - Voice + Gesture", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    stop_listening(wait_for_stop=False)
    cap.release()
    cv2.destroyAllWindows()

    if ser is not None and ser.is_open:
        send_to_arduino(0)
        ser.close()
        print("Arduino connection closed")

    print("Game Mode Closed")
