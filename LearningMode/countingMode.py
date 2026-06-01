import cv2
import sys
import math
import time
import random
import serial
import mediapipe as mp
import speech_recognition as sr

# -------------------------------------------------
# Configuration
# -------------------------------------------------
ARDUINO_PORT = "COM4"
ARDUINO_BAUD_RATE = 9600
SIMULATION_MODE = False

learning_state = "START"
target_number = 0              
spoken_number = None           
current_finger_count = -1      
round_start_time = 0
feedback_start_time = 0
last_state_change_time = 0
feedback_text = ""
feedback_color = (255, 255, 255)
MIN_FINGERS = 0
MAX_FINGERS = 5

NUMBER_WORDS = {
    "ZERO": 0, "OH": 0, "ONE": 1, "WON": 1, "TWO": 2, "TO": 2, "TOO": 2,
    "THREE": 3, "TREE": 3, "FOUR": 4, "FOR": 4, "FIVE": 5, "FIFE": 5
}

def get_dist(p1, p2): return math.hypot(p1.x - p2.x, p1.y - p2.y)

def init_serial_connection():
    global SIMULATION_MODE
    try:
        ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD_RATE, timeout=1)
        time.sleep(2)
        SIMULATION_MODE = False
        return ser
    except Exception:
        SIMULATION_MODE = True
        return None

def send_robot_fingers(ser, number):
    safe_number = max(MIN_FINGERS, min(MAX_FINGERS, int(number)))
    if ser is not None and ser.is_open:
        try: ser.write(str(safe_number).encode())
        except Exception: pass

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
    """ מחוות חזרה (👌) """
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])
    if hand_scale <= 0: return False
    thumb_index_dist = get_dist(hand_landmarks.landmark[4], hand_landmarks.landmark[8])
    def is_finger_open(tip_idx, mip_idx):
        mip_dist = get_dist(wrist, hand_landmarks.landmark[mip_idx])
        return mip_dist > 0 and (get_dist(wrist, hand_landmarks.landmark[tip_idx]) / mip_dist) > 1.10
    return thumb_index_dist < hand_scale * 0.35 and is_finger_open(12, 10) and is_finger_open(16, 14) and is_finger_open(20, 18)

def is_thumbs_up_gesture(hand_landmarks):
    """ מחוות אישור / המשך (👍) """
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])
    if hand_scale <= 0: return False
    thumb_open = get_dist(hand_landmarks.landmark[4], hand_landmarks.landmark[5]) > hand_scale * 0.65
    def is_finger_closed(tip_idx, mip_idx):
        mip_dist = get_dist(wrist, hand_landmarks.landmark[mip_idx])
        return mip_dist > 0 and (get_dist(wrist, hand_landmarks.landmark[tip_idx]) / mip_dist) < 1.10
    return thumb_open and is_finger_closed(8, 6) and is_finger_closed(12, 10) and is_finger_closed(16, 14) and is_finger_closed(20, 18)

def voice_callback(recognizer, audio):
    global spoken_number, learning_state
    try:
        command = recognizer.recognize_google(audio, language="en-US").upper()
        for digit in range(6):
            if str(digit) in command: spoken_number = digit; return
        for word, number in NUMBER_WORDS.items():
            if word in command: spoken_number = number; return
    except Exception: pass

r, m = sr.Recognizer(), sr.Microphone()
r.energy_threshold, r.dynamic_energy_threshold, r.non_speaking_duration, r.pause_threshold = 1000, False, 0.3, 0.3
with m as source: r.adjust_for_ambient_noise(source, duration=1)
stop_listening = r.listen_in_background(m, voice_callback, phrase_time_limit=1.2)

ser = init_serial_connection()
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

try:
    with mp_hands.Hands(model_complexity=1, max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: continue
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            current_finger_count, current_back_gesture, current_thumbs_up = -1, False, False
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    current_finger_count = count_fingers_from_landmarks(hand_landmarks)
                    current_back_gesture = is_ok_gesture(hand_landmarks)
                    current_thumbs_up = is_thumbs_up_gesture(hand_landmarks)

            current_time = time.time()

            if learning_state == "START":
                cv2.putText(frame, "Show 👍 (Thumbs Up) to start", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, "👌 (OK Sign) = return to Education Menu", (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)
                if current_time - last_state_change_time > 1.0:
                    if current_back_gesture:
                        send_robot_fingers(ser, 0)
                        sys.exit(10)
                    elif current_thumbs_up:
                        learning_state = "SHOW_ROBOT"
                        last_state_change_time = current_time

            elif learning_state == "SHOW_ROBOT":
                target_number = random.randint(1, 5)
                spoken_number = None
                send_robot_fingers(ser, target_number)
                round_start_time = current_time
                learning_state = "WAIT_FOR_USER"

            elif learning_state == "WAIT_FOR_USER":
                cv2.putText(frame, f"Robot shows: {target_number}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                if current_finger_count != -1 and spoken_number is not None:
                    if current_finger_count == target_number and spoken_number == target_number:
                        feedback_text, feedback_color = "Correct! Great job!", (0, 255, 0)
                    else:
                        feedback_text, feedback_color = f"Fingers: {current_finger_count}, Said: {spoken_number}", (0, 0, 255)
                    feedback_start_time = current_time
                    learning_state = "FEEDBACK"

            elif learning_state == "FEEDBACK":
                cv2.putText(frame, feedback_text, (30, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, feedback_color, 2)
                if current_time - feedback_start_time > 2:
                    learning_state = "ROUND_END_MENU"
                    last_state_change_time = current_time

            elif learning_state == "ROUND_END_MENU":
                cv2.putText(frame, "Show 👍 (Thumbs Up) to continue", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.putText(frame, "👌 (OK Sign) = return to Education Menu", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                if current_time - last_state_change_time > 1.0:
                    if current_back_gesture:
                        send_robot_fingers(ser, 0)
                        sys.exit(10)
                    elif current_thumbs_up:
                        learning_state = "SHOW_ROBOT"
                        last_state_change_time = current_time

            cv2.imshow("Learning Mode - Count with Robot", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    send_robot_fingers(ser, 0)
    stop_listening(wait_for_stop=False)
    cap.release()
    cv2.destroyAllWindows()