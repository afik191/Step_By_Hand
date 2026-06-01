import cv2
import sys
import math
import time
import random
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

# -------------------------------------------------
# Configuration
# -------------------------------------------------

ARDUINO_PORT = "COM4"
ARDUINO_BAUD_RATE = 9600

BACK_EXIT_CODE = 10
GESTURE_HOLD_SECONDS = 1.5
ANSWER_STABLE_SECONDS = 0.8
FEEDBACK_SECONDS = 2.2

MIN_FINGERS = 0
MAX_FINGERS = 5

# Colors
TITLE_COLOR = (0, 0, 0)
OPTION_COLOR = (0, 0, 220)
NOTE_COLOR = (60, 60, 60)
HOLD_COLOR = (100, 40, 0)
GOOD_COLOR = (0, 140, 0)
BAD_COLOR = (0, 0, 180)
INFO_COLOR = (45, 45, 45)

# -------------------------------------------------
# Split screen display
# -------------------------------------------------

PANEL_BG_COLOR = (255, 255, 255)
DIVIDER_COLOR = (170, 170, 170)

def create_split_screen(frame):
    camera_view = frame.copy()
    panel = frame.copy()
    panel[:] = PANEL_BG_COLOR
    panel_h = panel.shape[0]
    cv2.line(panel, (0, 0), (0, panel_h), DIVIDER_COLOR, 3)
    return camera_view, panel

NUMBER_WORDS = {
    "ZERO": 0, "OH": 0, "ONE": 1, "WON": 1, "TWO": 2, "TO": 2, "TOO": 2,
    "THREE": 3, "TREE": 3, "FOUR": 4, "FOR": 4, "FIVE": 5, "FIFE": 5,
}

# -------------------------------------------------
# Global state
# -------------------------------------------------

state = "READY"
target_number = 0
spoken_number = None
heard_text = ""

candidate_fingers = None
candidate_start_time = 0.0
stable_finger_answer = None

feedback_text = ""
feedback_color = INFO_COLOR
feedback_start_time = 0.0

hold_action = None
hold_start_time = 0.0

voice_queue = deque()
stop_listening = None
voice_enabled = False

SIMULATION_MODE = False
ser = None

# -------------------------------------------------
# Helpers
# -------------------------------------------------

def get_dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def limit_fingers(number):
    return max(MIN_FINGERS, min(MAX_FINGERS, int(number)))

def draw_lines(frame, lines, start_y, color, scale=0.85, thickness=2, step=35):
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (30, start_y + i * step), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def parse_number_from_voice(command):
    command = command.upper()
    for digit in range(0, 6):
        if str(digit) in command:
            return digit
    for word, number in NUMBER_WORDS.items():
        if word in command:
            return number
    return None

def count_fingers_from_landmarks(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])
    if hand_scale <= 0: return -1
    fingers = []
    if get_dist(hand_landmarks.landmark[4], hand_landmarks.landmark[5]) > hand_scale * 0.6:
        fingers.append(1)
    else:
        fingers.append(0)
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
    wrist = hand_landmarks.landmark[0]
    hand_scale = get_dist(wrist, hand_landmarks.landmark[9])
    if hand_scale <= 0: return False
    thumb_index_dist = get_dist(hand_landmarks.landmark[4], hand_landmarks.landmark[8])
    def is_finger_open(tip_idx, mip_idx):
        mip_dist = get_dist(wrist, hand_landmarks.landmark[mip_idx])
        if mip_dist <= 0: return False
        tip_dist = get_dist(wrist, hand_landmarks.landmark[tip_idx])
        return tip_dist / mip_dist > 1.10
    middle_open = is_finger_open(12, 10)
    ring_open = is_finger_open(16, 14)
    pinky_open = is_finger_open(20, 18)
    return thumb_index_dist < hand_scale * 0.35 and middle_open and ring_open and pinky_open

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
    index_open = is_finger_open(8, 6)
    middle_open = is_finger_open(12, 10)
    ring_open = is_finger_open(16, 14)
    pinky_open = is_finger_open(20, 18)
    return thumb_open and not index_open and not middle_open and not ring_open and not pinky_open

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
        completed_action = hold_action
        hold_action, hold_start_time = None, 0.0
        return completed_action, elapsed
    return None, elapsed

def draw_hold_status(frame, current_time):
    h, _, _ = frame.shape
    if hold_action:
        progress = min(GESTURE_HOLD_SECONDS, current_time - hold_start_time)
        cv2.putText(frame, f"Hold action: {hold_action}", (30, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HOLD_COLOR, 2)
        cv2.putText(frame, f"Hold: {progress:.1f}s / {GESTURE_HOLD_SECONDS:.1f}s", (30, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HOLD_COLOR, 2)

# -------------------------------------------------
# Arduino
# -------------------------------------------------

def init_serial_connection(port=ARDUINO_PORT, baud_rate=ARDUINO_BAUD_RATE):
    global SIMULATION_MODE
    if serial is None:
        SIMULATION_MODE = True
        return None
    try:
        arduino = serial.Serial(port, baud_rate, timeout=1)
        time.sleep(2)
        SIMULATION_MODE = False
        return arduino
    except Exception:
        SIMULATION_MODE = True
        return None

def send_robot_fingers(number):
    safe_number = limit_fingers(number)
    if ser is not None and ser.is_open:
        try: ser.write(str(safe_number).encode())
        except Exception: pass

# -------------------------------------------------
# Voice
# -------------------------------------------------

def voice_callback(recognizer, audio):
    try:
        command = recognizer.recognize_google(audio, language="en-US").upper()
        voice_queue.append(command)
    except Exception: pass

def init_voice():
    global stop_listening, voice_enabled
    voice_enabled, stop_listening = False, None
    if sr is None: return
    try:
        recognizer, microphone = sr.Recognizer(), sr.Microphone()
        
        # --- Voice Recognition Improvements ---
        recognizer.energy_threshold = 300             # Lower baseline for quieter voices
        recognizer.dynamic_energy_threshold = True    # Automatically adjust to background noise
        recognizer.pause_threshold = 0.8              # Wait longer before assuming the user stopped talking
        recognizer.non_speaking_duration = 0.5        # Increased to prevent cutting off early
        
        with microphone as source: 
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
        # Increased phrase_time_limit from 1.2 to 3.0 to give users time to articulate
        stop_listening = recognizer.listen_in_background(microphone, voice_callback, phrase_time_limit=3.0)
        voice_enabled = True
    except Exception:
        voice_enabled, stop_listening = False, None

def stop_voice():
    global stop_listening
    if stop_listening is not None:
        try: stop_listening(wait_for_stop=False)
        except Exception: pass
    stop_listening = None

# -------------------------------------------------
# Game logic
# -------------------------------------------------

def start_round():
    global state, target_number, spoken_number, heard_text
    global candidate_fingers, candidate_start_time, stable_finger_answer
    global feedback_text, feedback_color
    target_number = random.randint(1, 5)
    spoken_number, heard_text = None, ""
    candidate_fingers, candidate_start_time, stable_finger_answer = None, 0.0, None
    feedback_text, feedback_color = "", INFO_COLOR
    send_robot_fingers(target_number)
    state = "WAIT_FOR_USER"

def go_to_feedback(finger_answer, voice_answer):
    global state, feedback_text, feedback_color, feedback_start_time
    finger_correct = finger_answer == target_number
    voice_correct = voice_answer == target_number if voice_enabled and voice_answer is not None else True

    if finger_correct and voice_correct:
        feedback_text, feedback_color = "GOOD JOB", GOOD_COLOR
    else:
        mistakes = []
        if not finger_correct: mistakes.append(f"Fingers: {finger_answer}")
        if not voice_correct: mistakes.append(f"Voice: {voice_answer}")
        feedback_text, feedback_color = "TRY AGAIN - " + " | ".join(mistakes), BAD_COLOR

    send_robot_fingers(0)
    feedback_start_time = time.time()
    state = "FEEDBACK"

def process_voice_command(command):
    global spoken_number, heard_text
    heard_text = command
    if "BACK" in command:
        send_robot_fingers(0)
        sys.exit(BACK_EXIT_CODE)
    if state in ["READY", "ROUND_END_MENU"]:
        if any(word in command for word in ["START", "BEGIN", "PLAY", "AGAIN"]):
            start_round()
            return
    if state == "WAIT_FOR_USER":
        number = parse_number_from_voice(command)
        if number is not None:
            spoken_number = number

# -------------------------------------------------
# Init & Main Loop
# -------------------------------------------------

ser = init_serial_connection()
init_voice()

try:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    stop_voice()
    raise SystemExit(1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    stop_voice()
    if ser is not None and ser.is_open: ser.close()
    raise SystemExit(1)

try:
    with mp_hands.Hands(model_complexity=1, max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: continue
            
            frame = cv2.flip(frame, 1)
            camera_view, panel = create_split_screen(frame)
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(camera_view, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            current_fingers = -1
            current_ok = current_thumbs = False

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(camera_view, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    current_fingers = count_fingers_from_landmarks(hand_landmarks)
                    current_ok = is_ok_gesture(hand_landmarks)
                    current_thumbs = is_thumbs_up(hand_landmarks)

            current_time = time.time()
            while voice_queue: process_voice_command(voice_queue.popleft())

            # Top Simulation Indicator
            if SIMULATION_MODE:
                cv2.putText(panel, "SIMULATION MODE", (30, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, NOTE_COLOR, 2)

            if state == "READY":
                draw_lines(panel, ["COUNTING / IMITATION"], 70, TITLE_COLOR, scale=1.0, thickness=2)
                draw_lines(panel, ["Thumbs up = Start learning", "Voice: say START"], 130, OPTION_COLOR, scale=0.85, thickness=2, step=35)
                draw_lines(panel, ["OK sign = Back to Menu", "Voice: say BACK"], 220, NOTE_COLOR, scale=0.7, thickness=2, step=35)

                action, _ = update_hold("START" if current_thumbs else ("BACK" if current_ok else None), current_time)
                if action == "START": start_round()
                elif action == "BACK": send_robot_fingers(0); sys.exit(BACK_EXIT_CODE)

            elif state == "WAIT_FOR_USER":
                draw_lines(panel, ["COPY THE ROBOT"], 70, TITLE_COLOR, scale=1.0, thickness=2)
                draw_lines(panel, [f"Robot shows: {target_number}"], 120, OPTION_COLOR, scale=0.9, thickness=2)
                draw_lines(panel, ["Show same number with fingers", "Say number out loud", "OK sign = Back"], 170, NOTE_COLOR, scale=0.65, thickness=2, step=30)

                # Dynamic layout area starting from Y=280 to avoid overlap
                current_y = 280
                
                if current_fingers != -1:
                    cv2.putText(panel, f"Detected fingers: {current_fingers}", (30, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, INFO_COLOR, 2)
                    current_y += 35
                    
                    if candidate_fingers != current_fingers:
                        candidate_fingers, candidate_start_time = current_fingers, current_time
                    else:
                        finger_hold = current_time - candidate_start_time
                        cv2.putText(panel, f"Hold: {finger_hold:.1f}s / {ANSWER_STABLE_SECONDS:.1f}s", (30, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HOLD_COLOR, 2)
                        current_y += 35
                        if finger_hold >= ANSWER_STABLE_SECONDS: stable_finger_answer = current_fingers

                if spoken_number is not None:
                    cv2.putText(panel, f"Heard number: {spoken_number}", (30, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, INFO_COLOR, 2)
                    current_y += 35
                elif heard_text:
                    cv2.putText(panel, f"Heard: {heard_text}", (30, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, INFO_COLOR, 2)
                    current_y += 35

                if stable_finger_answer is not None:
                    if voice_enabled:
                        if spoken_number is not None: go_to_feedback(stable_finger_answer, spoken_number)
                        else: cv2.putText(panel, "Waiting for voice...", (30, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HOLD_COLOR, 2)
                    else: go_to_feedback(stable_finger_answer, None)

                action, _ = update_hold("BACK" if current_ok else None, current_time)
                if action == "BACK": send_robot_fingers(0); sys.exit(BACK_EXIT_CODE)

            elif state == "FEEDBACK":
                draw_lines(panel, [feedback_text], h // 2 - 20, feedback_color, scale=0.8, thickness=2)
                if current_time - feedback_start_time >= FEEDBACK_SECONDS: state = "ROUND_END_MENU"

            elif state == "ROUND_END_MENU":
                draw_lines(panel, ["ROUND FINISHED"], 70, TITLE_COLOR, scale=1.0, thickness=2)
                draw_lines(panel, [feedback_text], 130, feedback_color, scale=0.8, thickness=2)
                draw_lines(panel, ["Thumbs up = Start again", "Voice: say START", "OK sign = Back", "Voice: say BACK"], 190, OPTION_COLOR, scale=0.7, thickness=2, step=35)

                action, _ = update_hold("START" if current_thumbs else ("BACK" if current_ok else None), current_time)
                if action == "START": start_round()
                elif action == "BACK": send_robot_fingers(0); sys.exit(BACK_EXIT_CODE)

            draw_hold_status(panel, current_time)

            combined_screen = cv2.hconcat([camera_view, panel])
            combined_screen = cv2.resize(combined_screen, (1280, 520))
            cv2.imshow("Learning Mode - Count with Robot", combined_screen)

            if cv2.waitKey(1) & 0xFF == ord("q"): break

finally:
    send_robot_fingers(0)
    stop_voice()
    cap.release()
    cv2.destroyAllWindows()
    if ser is not None and ser.is_open: ser.close()