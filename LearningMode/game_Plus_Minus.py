import cv2
import sys
import math
import time
import random
import subprocess
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

# Colors
TITLE_COLOR = (0, 0, 0)
OPTION_COLOR = (0, 0, 220)
NOTE_COLOR = (60, 60, 60)
HOLD_COLOR = (100, 40, 0)
GOOD_COLOR = (0, 140, 0)
BAD_COLOR = (0, 0, 180)
INFO_COLOR = (50, 50, 50)

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

def draw_lines(frame, lines, start_y, color, scale=0.9, thickness=2, step=35):
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (30, start_y + i * step), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def draw_hold_status(frame, current_time):
    h, _, _ = frame.shape
    if hold_action:
        progress = min(GESTURE_HOLD_SECONDS, current_time - hold_start_time)
        # Positioned strictly at the bottom
        cv2.putText(frame, f"Hold action: {hold_action}", (30, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HOLD_COLOR, 2)
        cv2.putText(frame, f"Hold selection: {progress:.1f}s / {GESTURE_HOLD_SECONDS:.1f}s", (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HOLD_COLOR, 2)

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

ARDUINO_PORT = "COM4"
ARDUINO_BAUD_RATE = 9600
ROBOT_SHOW_SECONDS = 2.0
ROBOT_GAP_SECONDS = 0.6
FEEDBACK_SECONDS = 2.0
BACK_EXIT_CODE = 10

state = "CHOOSE_OPERATION"
selected_operation = None
first_number = 0
second_number = 0
correct_answer = 0
candidate_answer = None
candidate_start_time = 0.0
state_start_time = 0.0
feedback_text = ""
feedback_color = INFO_COLOR
last_answer_correct = False

def play_feedback_audio(text):
    try:
        safe_text = str(text).replace("'", "")
        command = (
            "Add-Type -AssemblyName System.Speech; "
            "$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "try { $speak.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::Female,[System.Speech.Synthesis.VoiceAge]::Adult); } catch {}; "
            "$speak.Rate = -1; $speak.Volume = 90; "
            f"$speak.Speak('{safe_text}');"
        )
        subprocess.Popen(["powershell", "-NoProfile", "-Command", command], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

def generate_exercise(operation):
    if operation == "ADD":
        a = random.randint(0, 5)
        b = random.randint(0, 5)
        return a, b, a + b
    a = random.randint(0, 5)
    b = random.randint(0, a)
    return a, b, a - b

def get_operation_symbol():
    return "+" if selected_operation == "ADD" else "-"

def start_new_exercise():
    global first_number, second_number, correct_answer, state, state_start_time, candidate_answer, candidate_start_time
    first_number, second_number, correct_answer = generate_exercise(selected_operation)
    candidate_answer = None
    candidate_start_time = 0.0
    state = "SHOW_FIRST"
    state_start_time = time.time()
    send_robot_fingers(ser, first_number)

def go_to_operation_menu():
    global state, selected_operation, candidate_answer, candidate_start_time
    send_robot_fingers(ser, 0)
    state = "CHOOSE_OPERATION"
    selected_operation = None
    candidate_answer = None
    candidate_start_time = 0.0

def check_answer(answer_value):
    global feedback_text, feedback_color, last_answer_correct, state, state_start_time
    if answer_value == correct_answer:
        feedback_text = "GOOD JOB"
        feedback_color = GOOD_COLOR
        last_answer_correct = True
        play_feedback_audio("Good job")
    else:
        feedback_text = "TRY AGAIN"
        feedback_color = BAD_COLOR
        last_answer_correct = False
        play_feedback_audio("Try again")
    state = "FEEDBACK"
    state_start_time = time.time()

def process_voice_command(command):
    global state, selected_operation
    if "BACK" in command:
        if state == "CHOOSE_OPERATION":
            send_robot_fingers(ser, 0)
            sys.exit(BACK_EXIT_CODE)
        else:
            go_to_operation_menu()
        return True
    if state == "CHOOSE_OPERATION":
        if "ADD" in command or "PLUS" in command:
            selected_operation = "ADD"
            state = "READY"
            return True
        if "SUB" in command or "MINUS" in command:
            selected_operation = "SUB"
            state = "READY"
            return True
    if ("START" in command or "BEGIN" in command or "AGAIN" in command) and state in ["READY", "ROUND_END_MENU"]:
        if state == "READY":
            start_new_exercise()
        elif state == "ROUND_END_MENU":
            if last_answer_correct:
                start_new_exercise()
            else:
                global candidate_answer, candidate_start_time, state_start_time
                candidate_answer = None
                candidate_start_time = 0.0
                state = "WAIT_FOR_ANSWER"
                state_start_time = time.time()
        return True
    return False

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
ser = init_serial_connection(ARDUINO_PORT, ARDUINO_BAUD_RATE)
init_voice()

cap = cv2.VideoCapture(0)
attempts = 0
while not cap.isOpened() and attempts < 5:
    print("Camera busy... waiting 0.5s before retrying...")
    time.sleep(0.5)
    cap = cv2.VideoCapture(0)
    attempts += 1

if not cap.isOpened():
    stop_voice()
    raise SystemExit("Camera Error")

try:
    with mp_hands.Hands(model_complexity=1, max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue
            frame = cv2.flip(frame, 1)
            camera_view, panel = create_split_screen(frame)
            h, w, _ = frame.shape
            
            results = hands.process(cv2.cvtColor(camera_view, cv2.COLOR_BGR2RGB))
            all_hand_landmarks = results.multi_hand_landmarks if results.multi_hand_landmarks else []
            current_fingers = count_fingers_two_hands(all_hand_landmarks)
            current_ok = any(is_ok_gesture(hand) for hand in all_hand_landmarks) if all_hand_landmarks else False
            current_thumbs = any(is_thumbs_up(hand) for hand in all_hand_landmarks) if all_hand_landmarks else False
            
            for hand_landmarks in all_hand_landmarks:
                mp_drawing.draw_landmarks(camera_view, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
            current_time = time.time()
            while voice_queue:
                process_voice_command(voice_queue.popleft())
                
            elapsed = current_time - state_start_time
            
            # UI State Layouts - Compacted to avoid overlap
            if state == "CHOOSE_OPERATION":
                draw_lines(panel, ["ADDITION / SUBTRACTION"], 50, TITLE_COLOR, scale=1.0, thickness=2)
                draw_lines(panel, ["Show 1 finger = Addition", "Show 2 fingers = Subtraction"], 100, OPTION_COLOR, scale=0.8, thickness=2, step=30)
                draw_lines(panel, ["Voice: say PLUS / MINUS", "OK sign = Back to Menu", "Voice: BACK"], 180, NOTE_COLOR, scale=0.7, thickness=2, step=30)
                
                desired_action = None
                if current_ok:
                    desired_action = "BACK"
                elif current_fingers == 1:
                    desired_action = "ADD"
                elif current_fingers == 2:
                    desired_action = "SUB"
                action, _ = update_hold(desired_action, current_time)
                if action == "BACK":
                    send_robot_fingers(ser, 0)
                    sys.exit(BACK_EXIT_CODE)
                elif action == "ADD":
                    selected_operation = "ADD"
                    state = "READY"
                elif action == "SUB":
                    selected_operation = "SUB"
                    state = "READY"
                    
            elif state == "READY":
                draw_lines(panel, ["ADDITION / SUBTRACTION"], 50, TITLE_COLOR, scale=1.0, thickness=2)
                draw_lines(panel, [f"Operation: {get_operation_symbol()}", "Thumbs up = Start game", "Voice: START"], 100, OPTION_COLOR, scale=0.8, thickness=2, step=30)
                draw_lines(panel, ["OK sign = Back to operation choice", "Voice: BACK"], 200, NOTE_COLOR, scale=0.7, thickness=2, step=30)
                
                desired_action = "START" if current_thumbs else ("BACK" if current_ok else None)
                action, _ = update_hold(desired_action, current_time)
                if action == "START":
                    start_new_exercise()
                elif action == "BACK":
                    go_to_operation_menu()
                    
            elif state == "SHOW_FIRST":
                draw_lines(panel, ["WATCH THE ROBOT"], 50, TITLE_COLOR, scale=1.0, thickness=2)
                draw_lines(panel, [f"First number: {first_number}"], 110, OPTION_COLOR, scale=0.9, thickness=2)
                draw_lines(panel, [f"Operation: {get_operation_symbol()}", "OK sign = Back", "Voice: BACK"], 170, NOTE_COLOR, scale=0.7, thickness=2, step=30)
                
                desired_action = "BACK" if current_ok else None
                action, _ = update_hold(desired_action, current_time)
                if action == "BACK":
                    go_to_operation_menu()
                if elapsed >= ROBOT_SHOW_SECONDS and state == "SHOW_FIRST":
                    send_robot_fingers(ser, 0)
                    state = "GAP"
                    state_start_time = current_time
                    
            elif state == "GAP":
                draw_lines(panel, ["GET READY FOR NEXT NUMBER"], h // 2, NOTE_COLOR, scale=0.8, thickness=2)
                if elapsed >= ROBOT_GAP_SECONDS:
                    send_robot_fingers(ser, second_number)
                    state = "SHOW_SECOND"
                    state_start_time = current_time
                    
            elif state == "SHOW_SECOND":
                draw_lines(panel, ["WATCH THE ROBOT"], 50, TITLE_COLOR, scale=1.0, thickness=2)
                draw_lines(panel, [f"Second number: {second_number}"], 110, OPTION_COLOR, scale=0.9, thickness=2)
                draw_lines(panel, [f"Question: {first_number} {get_operation_symbol()} {second_number} = ?", "OK sign = Back", "Voice: BACK"], 170, NOTE_COLOR, scale=0.7, thickness=2, step=30)
                
                desired_action = "BACK" if current_ok else None
                action, _ = update_hold(desired_action, current_time)
                if action == "BACK":
                    go_to_operation_menu()
                if elapsed >= ROBOT_SHOW_SECONDS and state == "SHOW_SECOND":
                    send_robot_fingers(ser, 0)
                    candidate_answer = None
                    candidate_start_time = 0.0
                    state = "WAIT_FOR_ANSWER"
                    state_start_time = current_time
                    
            elif state == "WAIT_FOR_ANSWER":
                draw_lines(panel, ["SOLVE THE EXERCISE"], 50, TITLE_COLOR, scale=1.0, thickness=2)
                draw_lines(panel, [f"{first_number} {get_operation_symbol()} {second_number} = ?"], 100, OPTION_COLOR, scale=1.0, thickness=2)
                draw_lines(panel, ["Show answer with hands", "Hold your answer for 0.8s", "OK sign = Back to menu", "Voice: BACK"], 160, NOTE_COLOR, scale=0.7, thickness=2, step=30)
                
                if current_fingers != -1:
                    # Dynamically anchored from the bottom
                    cv2.putText(panel, f"Detected fingers: {current_fingers}", (30, h - 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, INFO_COLOR, 2)
                    if candidate_answer != current_fingers:
                        candidate_answer = current_fingers
                        candidate_start_time = current_time
                    else:
                        hold = current_time - candidate_start_time
                        cv2.putText(panel, f"Answer hold: {hold:.1f}s / {ANSWER_STABLE_SECONDS:.1f}s", (30, h - 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HOLD_COLOR, 2)
                        if hold >= ANSWER_STABLE_SECONDS:
                            check_answer(current_fingers)
                            
                desired_action = "BACK" if current_ok else None
                action, _ = update_hold(desired_action, current_time)
                if action == "BACK":
                    go_to_operation_menu()
                    
            elif state == "FEEDBACK":
                draw_lines(panel, [feedback_text], h // 2 - 20, feedback_color, scale=1.2, thickness=3)
                if elapsed >= FEEDBACK_SECONDS:
                    state = "ROUND_END_MENU"
                    state_start_time = current_time
                    
            elif state == "ROUND_END_MENU":
                draw_lines(panel, ["ROUND FINISHED"], 50, TITLE_COLOR, scale=1.0, thickness=2)
                draw_lines(panel, [feedback_text], 100, feedback_color, scale=0.9, thickness=2)
                msg = "Thumbs up = Next exercise" if last_answer_correct else "Thumbs up = Try again"
                draw_lines(panel, [msg, "Voice: START", "OK sign = Back to menu", "Voice: BACK"], 160, OPTION_COLOR, scale=0.7, thickness=2, step=30)
                
                desired_action = "START" if current_thumbs else ("BACK" if current_ok else None)
                action, _ = update_hold(desired_action, current_time)
                if action == "START":
                    if last_answer_correct:
                        start_new_exercise()
                    else:
                        candidate_answer = None
                        candidate_start_time = 0.0
                        state = "WAIT_FOR_ANSWER"
                        state_start_time = current_time
                elif action == "BACK":
                    go_to_operation_menu()
                    
            draw_hold_status(panel, current_time)
            combined_screen = cv2.hconcat([camera_view, panel])
            combined_screen = cv2.resize(combined_screen, (1280, 520))
            cv2.imshow("Math Mode - Addition and Subtraction", combined_screen)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    stop_voice()
    send_robot_fingers(ser, 0)
    cap.release()
    cv2.destroyAllWindows()
    if ser is not None and ser.is_open:
        ser.close()