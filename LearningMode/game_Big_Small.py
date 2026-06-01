import cv2
import sys
import math
import time
import random
import serial
import mediapipe as mp
import subprocess

# -------------------------------------------------
# Configuration
# -------------------------------------------------
ARDUINO_PORT = "COM4"
ARDUINO_BAUD_RATE = 9600
ROBOT_SHOW_SECONDS = 2.0
ROBOT_GAP_SECONDS = 0.6
FEEDBACK_SECONDS = 2.2
ANSWER_STABLE_SECONDS = 0.8
MIN_ROBOT_FINGERS = 0
MAX_ROBOT_FINGERS = 5
MIN_ANSWER = 0
MAX_ANSWER = 10

game_state = "CHOOSE_MODE"
selected_mode = None            
first_number, second_number, correct_answer = 0, 0, 0
current_finger_count = -1
candidate_answer = None
candidate_start_time = 0
state_start_time = time.time()
last_state_change_time = 0
feedback_text = ""
feedback_color = (255, 255, 255)
last_answer_correct = False
SIMULATION_MODE = False
ser = None

def play_feedback_audio(text):
    try:
        safe_text = str(text).replace("'", "")
        command = (
            "Add-Type -AssemblyName System.Speech; "
            "$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "try { $speak.SelectVoice('Microsoft Zira Desktop'); } catch { } "
            f"$speak.Speak('{safe_text}');"
        )
        subprocess.Popen(["powershell", "-NoProfile", "-Command", command], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception: pass

def get_dist(p1, p2): return math.hypot(p1.x - p2.x, p1.y - p2.y)

def init_serial_connection():
    global SIMULATION_MODE
    try:
        arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD_RATE, timeout=1)
        time.sleep(2)
        return arduino
    except Exception: return None

def send_robot_fingers(number):
    global ser
    safe = max(MIN_ROBOT_FINGERS, min(MAX_ROBOT_FINGERS, int(number)))
    if ser is not None and ser.is_open:
        try: ser.write(str(safe).encode())
        except Exception: ser = None

def count_fingers_single_hand(hand_landmarks):
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
    return sum(fingers)

def count_fingers_two_hands(all_hand_landmarks):
    if not all_hand_landmarks: return -1
    total, detected = 0, 0
    for hand in all_hand_landmarks:
        c = count_fingers_single_hand(hand)
        if c >= 0: total += c; detected += 1
    return max(MIN_ANSWER, min(MAX_ANSWER, total)) if detected > 0 else -1

def is_ok_gesture(hand_landmarks):
    """ מחוות חזרה קבועה (👌) """
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])
    if hand_scale <= 0: return False
    thumb_index_dist = get_dist(hand_landmarks.landmark[4], hand_landmarks.landmark[8])
    def is_finger_open(tip_idx, mip_idx):
        mip_dist = get_dist(wrist, hand_landmarks.landmark[mip_idx])
        return mip_dist > 0 and (get_dist(wrist, hand_landmarks.landmark[tip_idx]) / mip_dist) > 1.10
    return thumb_index_dist < hand_scale * 0.35 and is_finger_open(12, 10) and is_finger_open(16, 14) and is_finger_open(20, 18)

def is_thumbs_up(hand_landmarks):
    """ מחוות המשחק החוזר והאישור (👍) """
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])
    if hand_scale <= 0: return False
    thumb_open = get_dist(hand_landmarks.landmark[4], hand_landmarks.landmark[5]) > hand_scale * 0.65
    def is_finger_closed(tip_idx, mip_idx):
        mip_dist = get_dist(wrist, hand_landmarks.landmark[mip_idx])
        return mip_dist > 0 and (get_dist(wrist, hand_landmarks.landmark[tip_idx]) / mip_dist) < 1.10
    return thumb_open and is_finger_closed(8, 6) and is_finger_closed(12, 10) and is_finger_closed(16, 14) and is_finger_closed(20, 18)

def start_new_exercise():
    global first_number, second_number, correct_answer, game_state, state_start_time, candidate_answer, candidate_start_time
    first_number = random.randint(0, 5)
    second_number = random.randint(0, 5)
    while second_number == first_number: second_number = random.randint(0, 5)
    correct_answer = max(first_number, second_number) if selected_mode == "GREATER" else min(first_number, second_number)
    candidate_answer, candidate_start_time = None, 0
    game_state, state_start_time = "SHOW_FIRST", time.time()
    send_robot_fingers(first_number)

ser = init_serial_connection()
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

try:
    with mp_hands.Hands(model_complexity=1, max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: continue
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            all_hand_landmarks = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_hand_landmarks or []
            
            current_finger_count = count_fingers_two_hands(all_hand_landmarks)
            current_ok_gesture = any(is_ok_gesture(hand) for hand in all_hand_landmarks)
            current_thumbs_up = any(is_thumbs_up(hand) for hand in all_hand_landmarks)

            for hand in all_hand_landmarks: mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            current_time = time.time()
            elapsed = current_time - state_start_time

            if game_state == "CHOOSE_MODE":
                cv2.putText(frame, "Greater / Smaller Mode", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                cv2.putText(frame, "Show 1 Finger = Greater", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.putText(frame, "Show 2 Fingers = Smaller", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.putText(frame, "👌 (OK Sign): return to Education Menu", (30, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                
                if current_time - last_state_change_time > 1.0:
                    if current_ok_gesture:
                        send_robot_fingers(0)
                        sys.exit(10)
                    elif current_finger_count == 1:
                        selected_mode = "GREATER"
                        last_state_change_time = current_time
                        start_new_exercise()
                    elif current_finger_count == 2:
                        selected_mode = "SMALLER"
                        last_state_change_time = current_time
                        start_new_exercise()

            elif game_state == "SHOW_FIRST":
                if elapsed >= ROBOT_SHOW_SECONDS:
                    send_robot_fingers(0)
                    game_state, state_start_time = "GAP", current_time

            elif game_state == "GAP":
                if elapsed >= ROBOT_GAP_SECONDS:
                    send_robot_fingers(second_number)
                    game_state, state_start_time = "SHOW_SECOND", current_time

            elif game_state == "SHOW_SECOND":
                if elapsed >= ROBOT_SHOW_SECONDS:
                    send_robot_fingers(0)
                    candidate_answer, candidate_start_time, game_state, state_start_time = None, 0, "WAIT_FOR_ANSWER", current_time

            elif game_state == "WAIT_FOR_ANSWER":
                if current_finger_count != -1:
                    if candidate_answer != current_finger_count:
                        candidate_answer = current_finger_count
                        candidate_start_time = current_time
                    elif current_time - candidate_start_time >= ANSWER_STABLE_SECONDS:
                        if current_finger_count == correct_answer:
                            feedback_text, feedback_color, last_answer_correct = "GOOD JOB", (0, 255, 0), True
                            play_feedback_audio("Good job")
                        else:
                            feedback_text, feedback_color, last_answer_correct = "TRY AGAIN", (0, 0, 255), False
                            play_feedback_audio("Try again")
                        game_state, state_start_time, last_state_change_time = "FEEDBACK", current_time, current_time
                else: candidate_answer, candidate_start_time = None, 0

            elif game_state == "FEEDBACK":
                cv2.putText(frame, feedback_text, (30, h // 2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.8, feedback_color, 4)
                if elapsed >= FEEDBACK_SECONDS:
                    game_state, state_start_time, last_state_change_time = "ROUND_END_MENU", current_time, current_time

            elif game_state == "ROUND_END_MENU":
                cv2.putText(frame, "👍 (Thumbs Up) to continue/retry", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
                cv2.putText(frame, "👌 (OK Sign): return to Education Menu", (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 2)
                
                if current_time - last_state_change_time > 1.0:
                    if current_ok_gesture:
                        send_robot_fingers(0)
                        sys.exit(10)
                    elif current_thumbs_up:
                        if last_answer_correct: start_new_exercise()
                        else: candidate_answer, candidate_start_time, game_state, state_start_time, last_state_change_time = None, 0, "WAIT_FOR_ANSWER", current_time, current_time

            cv2.imshow("Greater / Smaller Mode - Finger Answer Only", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
finally:
    send_robot_fingers(0)
    cap.release()
    cv2.destroyAllWindows()